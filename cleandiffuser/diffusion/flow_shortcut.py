"""Continuous-time Shortcut Flow backbone for the few-step planner.

Implements ``ContinuousShortcutFlow`` (Frans et al., 2024, "One Step Diffusion via
Shortcut Models", arXiv:2410.12557): one network predicts the *average* velocity of a
finite jump of size ``d`` (``d=0`` ⇒ flow matching), trained with a flow-matching +
EMA self-consistency objective so a single forward pass can span a whole ``t=1→0`` step.
See the class docstring for the loss and ``planning/IMPLEMENTATION_PLAN.md`` for context.
"""

from typing import Optional, Union

import einops
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.diffusion.basic import DiffusionModel
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    TensorDict,
    at_least_ndim,
    concat_zeros,
    dict_apply,
    get_sampling_scheduler,
)


class ContinuousShortcutFlow(DiffusionModel):
    """Continuous-time Shortcut Flow Model (Frans et al., 2024).

    Reference: "One Step Diffusion via Shortcut Models", arXiv:2410.12557.

    Note on naming: although this class inherits from ``DiffusionModel`` (the
    CleanDiffuser framework base that holds EMA / Lightning plumbing), the
    *generative model* itself is a **flow** model. It is built on the same
    linear interpolation forward process as ``ContinuousRectifiedFlow`` and
    uses a pure ODE during sampling — there is no stochastic diffusion term.

    A single network ``s_θ(xt, t, d, c)`` is trained to predict the *average*
    velocity from time ``t`` to ``t - d`` along the path
    ``xt = (1 - t)·x0 + t·ε``. When ``d = 0`` this reduces to standard flow
    matching (instantaneous velocity ``v* = x0 − ε``). When ``d > 0`` it
    predicts the chord velocity of a finite jump of size ``d``.

    Training uses a hybrid objective on each batch (kvfrans default 87.5/12.5,
    i.e. ``bootstrap_every=8``):

      * **Flow matching** (``d = 0``, ``fm_consistency_ratio`` of the batch):
        ``loss_fm = ||s_θ(xt, t, 0, c) − (x0 − ε)||²``

      * **Self-consistency** (``d > 0``, remaining batch fraction):
        Sample ``d ∈ {2^-K_max, …, 1}``, ``t ~ U[d, 1]``. Using the EMA
        network for the (stop-grad) target,

            s1     = s_ema(xt, t, d/2, c)
            x_mid  = xt + (d/2) · s1
            s2     = s_ema(x_mid, t - d/2, d/2, c)
            target = stopgrad((s1 + s2) / 2)

        ``loss_sc = ||s_θ(xt, t, d, c) − target||²``

    Sampling integrates the ODE from ``t = 1`` (noise) to ``t = 0`` (data)
    using Euler-like updates at the chosen ``d``:
    ``x_{t-d} = xt + d · s_θ(xt, t, d, c)``. Setting ``sample_steps = 1``
    yields true one-step generation (``d ≈ 1``).

    **Intrinsic Guidance** (``guided=True``; iSM, arXiv:2510.21250): the CFG scale
    ``w`` becomes an explicit network input (``DiT1dShortcut``'s ``w``), trained
    across ``w ∈ [w_min, w_max]``, so guidance is applied once per forward pass and
    can be varied at inference. This removes the original Shortcut restriction that
    fixes ``w`` before training, and avoids the exponential *compounding* of a
    post-hoc CFG blend over big jumps (iSM Prop. 1: a single large step compounds
    the scale to ≈ ``w^log2(N)``). The FM branch (``d=0``) regresses
    ``s_θ(xt,t,c,0,w)`` to ``(1+w)·v_cond − w·v_uncond`` (``v_uncond`` from the EMA
    net at ``w=0``, stop-grad), and the self-consistency bootstrap is evaluated at
    the *same* ``w``. ``sample(..., w=...)`` then does one guided forward per step
    (no post-hoc blend). Mirrors the iMF w-conditioning in ``ContinuousMeanFlow``.

    Args:
        nn_diffusion (BaseNNDiffusion): Network that supports the extra ``d``
            kwarg in its ``forward`` (e.g. ``DiT1dShortcut``).
        nn_condition (Optional[BaseNNCondition]): Optional CFG condition.
        fix_mask (Optional[torch.Tensor]): Boolean/float mask in ``x_shape``.
            Marked positions are pinned to ``prior`` at every Euler step.
        loss_weight (Optional[torch.Tensor]): Per-element loss weight.
        classifier: Must be ``None`` — classifier guidance is not supported
            (consistent with ``ContinuousRectifiedFlow``).
        ema_rate (float): EMA decay for the target network.
        optimizer_params (Optional[dict]): Same convention as other
            CleanDiffuser diffusion classes.
        x_max, x_min (Optional[torch.Tensor]): Output clipping bounds.
        K_max (int): Largest negative power of two for the shortcut step
            schedule (``d ∈ {2^-K_max, …, 2^0}``). Default 7 ⇒ 1/128 … 1.
        fm_consistency_ratio (float): Fraction of each batch trained with
            ``d = 0`` (pure flow matching). The rest is trained with the
            self-consistency loss. Default 0.875 (kvfrans ``bootstrap_every=8``).
        guided (bool): Enable iSM Intrinsic Guidance (``w`` as a conditioning
            input). Default False (vanilla Shortcut; ``w`` ignored everywhere).
        w_min, w_max (float): Range the guidance scale ``w`` is sampled from
            during guided training. Default ``[0, 4]``.
    """

    def __init__(
        self,
        nn_diffusion: BaseNNDiffusion,
        nn_condition: Optional[BaseNNCondition] = None,
        fix_mask: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
        classifier: Optional[BaseClassifier] = None,
        ema_rate: float = 0.999,
        optimizer_params: Optional[dict] = None,
        x_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
        K_max: int = 7,
        fm_consistency_ratio: float = 0.875,
        discrete_t: bool = False,
        bootstrap_target: str = "ema",
        guided: bool = False,
        w_min: float = 0.0,
        w_max: float = 4.0,
    ):
        super().__init__(
            nn_diffusion,
            nn_condition,
            fix_mask,
            loss_weight,
            classifier,
            ema_rate,
            optimizer_params,
        )
        assert classifier is None, "Shortcut Flow Models do not support classifier-guidance."
        assert 0.0 < fm_consistency_ratio < 1.0, "fm_consistency_ratio must be in (0, 1)."
        assert K_max >= 1, "K_max must be >= 1 (need at least d ∈ {1/2, 1})."
        assert bootstrap_target in ("ema", "current")

        self.K_max = K_max
        self.fm_consistency_ratio = fm_consistency_ratio
        # discrete_t: snap t to the dyadic grid of multiples of d (kvfrans original) so the
        # self-consistency recursion telescopes exactly onto the N-step inference grid;
        # continuous (default) samples t ~ U[d, 1]. bootstrap_target: "ema" (default, twin-EMA
        # target) vs "current" (the live model under stop-grad, as in kvfrans).
        self.discrete_t = discrete_t
        self.bootstrap_target = bootstrap_target
        # iSM Intrinsic Guidance: condition on the guidance scale w (inference-time CFG).
        self.guided = guided
        self.w_min = w_min
        self.w_max = w_max
        # Optional external guidance-gradient hook ``(xt, t) -> ∇E`` (same shape as ``xt``). When set
        # on a guided model it swaps the FM-branch CFG target for the *energy*-tilted target
        # ``v + w·[t/(1−t)]·∇E`` (iSM energy guidance; the SC bootstrap enforces its self-consistency
        # for free). ``None`` (default) ⇒ the CFG target is used and this path is byte-identical to the
        # original iSM. Set by the driver (e.g. E1's analytic or learned-CEP ∇E) before training.
        self.energy_grad_fn = None

        self.x_max = nn.Parameter(x_max, requires_grad=False) if x_max is not None else None
        self.x_min = nn.Parameter(x_min, requires_grad=False) if x_min is not None else None

    @property
    def supported_solvers(self):
        return ["euler_shortcut"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training ======================

    def add_noise(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
    ):
        """Linear-interpolation forward process: ``xt = (1-t)·x0 + t·ε``."""
        t = torch.rand((x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps
        xt = x0 + at_least_ndim(t, x0.dim()) * (eps - x0)
        xt = xt * (1.0 - self.fix_mask) + x0 * self.fix_mask
        return xt, t, eps

    def loss(
        self,
        x0: torch.Tensor,
        condition: Optional[Union[torch.Tensor, TensorDict]] = None,
        x1: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ):
        """Hybrid flow-matching + self-consistency loss.

        If ``return_components`` is True, returns ``(total_loss, components)`` where
        ``components`` is ``{"loss_fm": <tensor>, "loss_sc": <tensor>}`` holding the
        (detached) per-branch MSE means — useful for tracking the two objectives
        separately during training/validation. The default (False) returns only the
        combined scalar, so existing callers are unaffected.
        """
        B = x0.shape[0]
        if x1 is None:
            x1 = torch.randn_like(x0)
        else:
            assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

        # Encode the condition ONCE (with whatever CFG dropout the condition
        # module applies in train mode). Re-using the same encoded vector for
        # both the LHS and the EMA-target RHS keeps the self-consistency
        # objective well-defined under random label dropout.
        cond_emb = self.model["condition"](condition) if condition is not None else None

        B_fm = int(round(B * self.fm_consistency_ratio))
        B_fm = max(0, min(B, B_fm))
        B_sc = B - B_fm

        total_loss = x0.new_zeros(())
        loss_fm = x0.new_zeros(())  # raw FM-branch MSE (for component logging)
        loss_sc = x0.new_zeros(())  # raw SC-branch MSE (for component logging)

        # ---------- Flow-matching branch (d = 0) ----------
        if B_fm > 0:
            x0_fm = x0[:B_fm]
            x1_fm = x1[:B_fm]
            t_fm = torch.rand((B_fm,), device=self.device)
            xt_fm = x0_fm + at_least_ndim(t_fm, x0_fm.dim()) * (x1_fm - x0_fm)
            xt_fm = xt_fm * (1.0 - self.fix_mask) + x0_fm * self.fix_mask
            d_fm = torch.zeros((B_fm,), device=self.device)
            cond_fm = cond_emb[:B_fm] if cond_emb is not None else None

            v_fm = x0_fm - x1_fm  # data-ward velocity (constant along the straight path)
            w_fm = None
            if self.guided and self.energy_grad_fn is not None and cond_fm is not None:
                # iSM energy guidance: regress s_θ(xt,t,c,0,w) to the ENERGY-tilted velocity
                # v + w·[t/(1−t)]·∇E (∇E = score_pt − score_qt analytic, or a learned CEP gradient).
                # w=0 ⇒ unguided q0, w=1 ⇒ exact p0 (analytic ∇E). The SC branch threads the same w
                # through its own half-/full-steps, so it enforces this field's self-consistency for
                # free (target-agnostic). Gate the 1/(1−t) blow-up at the noise boundary (true tilt ~0
                # there) and cap the per-sample correction norm (protects a never-exactly-zero ∇E).
                w_fm = torch.rand((B_fm,), device=self.device) * (self.w_max - self.w_min) + self.w_min
                gE = self.energy_grad_fn(xt_fm, t_fm)
                coef = torch.where((1.0 - t_fm) < 1e-3, torch.zeros_like(t_fm),
                                   w_fm * t_fm / (1.0 - t_fm))
                corr = at_least_ndim(coef, gE.dim()) * gE
                cn = corr.flatten(1).norm(dim=1)
                corr = corr * at_least_ndim(torch.clamp(50.0 / (cn + 1e-9), max=1.0), corr.dim())
                target_fm = (v_fm + corr).detach()
            elif self.guided and cond_fm is not None:
                # iSM Intrinsic Guidance: regress s_θ(xt,t,c,0,w) to the CFG-tilted velocity
                # (1+w)·v_cond − w·v_uncond (v_uncond from the EMA net at w=0, stop-grad).
                w_fm = torch.rand((B_fm,), device=self.device) * (self.w_max - self.w_min) + self.w_min
                with torch.no_grad():
                    v_uncond = self.model_ema["diffusion"](
                        xt_fm, t_fm, None, d=d_fm, w=torch.zeros_like(t_fm)
                    )
                ww = at_least_ndim(w_fm, v_fm.dim())
                target_fm = ((1.0 + ww) * v_fm - ww * v_uncond).detach()
            else:
                target_fm = v_fm
            pred_fm = self.model["diffusion"](xt_fm, t_fm, cond_fm, d=d_fm, w=w_fm)
            loss_fm = ((pred_fm - target_fm) ** 2 * self.loss_weight * (1 - self.fix_mask)).mean()
            total_loss = total_loss + loss_fm * (B_fm / B)

        # ---------- Self-consistency branch (d > 0) ----------
        if B_sc > 0:
            x0_sc = x0[B_fm:]
            x1_sc = x1[B_fm:]

            # Sample log2(d) uniformly from {-K_max, …, 0}; d = 2^log2d.
            log2d = (
                torch.randint(0, self.K_max + 1, (B_sc,), device=self.device).float()
                - self.K_max
            )
            d_sc = 2.0 ** log2d  # ∈ {2^-K_max, …, 1}

            if self.discrete_t:
                # Snap t to the dyadic grid {d, 2d, …, 1} (kvfrans original): the SC
                # recursion then telescopes exactly onto the N-step inference grid.
                n_sec = (1.0 / d_sc).round()
                t_idx = (torch.rand((B_sc,), device=self.device) * n_sec).floor() + 1.0
                t_sc = (t_idx * d_sc).clamp(max=1.0)
            else:
                # Continuous: t ~ U[d, 1] so the full d-step lands inside [0, 1].
                t_sc = torch.rand((B_sc,), device=self.device) * (1 - d_sc) + d_sc

            xt_sc = x0_sc + at_least_ndim(t_sc, x0_sc.dim()) * (x1_sc - x0_sc)
            xt_sc = xt_sc * (1.0 - self.fix_mask) + x0_sc * self.fix_mask
            cond_sc = cond_emb[B_fm:] if cond_emb is not None else None

            d_half = d_sc / 2

            # iSM: the self-consistency bootstrap is evaluated at a single sampled w,
            # threaded through both half-steps AND the full d-step, so consistency holds
            # for the guided velocity field at every w (w=None ⇒ vanilla, w ignored).
            w_sc = None
            if self.guided and cond_sc is not None:
                w_sc = torch.rand((B_sc,), device=self.device) * (self.w_max - self.w_min) + self.w_min

            target_model = self.model_ema if self.bootstrap_target == "ema" else self.model
            with torch.no_grad():
                # First half-step (target backbone)
                s1 = target_model["diffusion"](xt_sc, t_sc, cond_sc, d=d_half, w=w_sc)
                x_mid = xt_sc + at_least_ndim(d_half, xt_sc.dim()) * s1
                x_mid = x_mid * (1.0 - self.fix_mask) + x0_sc * self.fix_mask
                # Second half-step (target backbone) at time t - d/2
                t_mid = t_sc - d_half
                s2 = target_model["diffusion"](x_mid, t_mid, cond_sc, d=d_half, w=w_sc)
                target_sc = (s1 + s2) / 2  # stop-grad target

            pred_sc = self.model["diffusion"](xt_sc, t_sc, cond_sc, d=d_sc, w=w_sc)
            loss_sc = ((pred_sc - target_sc) ** 2 * self.loss_weight * (1 - self.fix_mask)).mean()
            total_loss = total_loss + loss_sc * (B_sc / B)

        if return_components:
            return total_loss, {"loss_fm": loss_fm.detach(), "loss_sc": loss_sc.detach()}
        return total_loss

    def update_diffusion(
        self,
        x0: torch.Tensor,
        condition_cfg: Optional[torch.Tensor] = None,
        update_ema: bool = True,
        x1: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run one optimizer step on the diffusion network (thin override of the base
        ``DiffusionModel.update_diffusion`` that forwards the optional fixed noise ``x1``).

        Args:
            x0: Clean data batch ``(B, *x_shape)`` — the ``t=0`` endpoint.
            condition_cfg: CFG conditioning input, or ``None`` for the unconditional model.
            update_ema: If ``True``, update the EMA weights after the gradient step.
            x1: Optional fixed source/noise sample ``(B, *x_shape)``; drawn from a
                standard normal when ``None``.
            **kwargs: Accepted for base-class compatibility; ignored here.

        Returns:
            The base class's update result (the training loss / log dict).
        """
        return super().update_diffusion(x0, condition_cfg, update_ema, x1=x1)

    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step.

        Batch keys:
            ``x0`` (required): clean data, shape ``(B, *x_shape)``.
            ``condition_cfg`` (optional): CFG condition.
            ``x1`` (optional): source-distribution sample. If ``None``,
                standard Gaussian noise is used.
        """
        assert isinstance(batch, dict) and "x0" in batch.keys(), (
            "The batch should contain the key `x0` for the input data."
        )
        x0 = batch["x0"]
        condition_cfg = batch.get("condition_cfg", None)
        x1 = batch.get("x1", None)

        loss, components = self.loss(x0, condition_cfg, x1=x1, return_components=True)
        self.log("diffusion_loss", loss, prog_bar=True)
        self.log("fm_loss", components["loss_fm"], prog_bar=False)
        self.log("sc_loss", components["loss_sc"], prog_bar=False)

        if self.ema_update_schedule(batch_idx):
            self.ema_update()

        return loss

    # ==================== Sampling ======================

    def sample(
        self,
        prior: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        solver: str = "euler_shortcut",
        sample_steps: int = 1,
        sampling_schedule: str = "linear",
        sampling_schedule_params: Optional[dict] = None,
        use_ema: bool = True,
        temperature: float = 1.0,
        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        mask_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cfg: float = 0.0,
        condition_cg: None = None,
        w_cg: float = 0.0,
        w: float = 0.0,
        warm_start_reference: Optional[torch.Tensor] = None,
        warm_start_forward_level: float = 0.3,
        requires_grad: bool = False,
        preserve_history: bool = False,
        **kwargs,
    ):
        """Euler-like sampling with arbitrary step count.

        For ``sample_steps = 1`` and a converged model this produces a true
        one-step sample (``d ≈ 1``). For ``sample_steps = N`` the schedule
        partitions ``[t_min, 1]`` into ``N`` intervals; each Euler step uses
        ``d = t_curr − t_next``.

        ``w`` is the **iSM Intrinsic Guidance** scale (only used when the model was
        trained with ``guided=True``): each step is a single guided forward
        ``s_θ(xt,t,c,d,w)`` (no post-hoc blend). Pass ``condition_cfg`` with ``w=0``
        for the conditional field, ``w>0`` to guide, or ``condition_cfg=None`` (``w=0``)
        for the unconditional field. ``w_cfg`` is the legacy post-hoc CFG blend used
        only for the **unguided** model (and as a documented compounding baseline).

        ``warm_start_reference`` enables MPC-style re-planning: the previous
        plan is mixed with noise at level ``warm_start_forward_level`` and
        used as the starting ``x1`` instead of pure noise.
        """
        assert solver in self.supported_solvers, f"Solver {solver} is not supported."
        assert w_cg == 0.0 and condition_cg is None, (
            "Shortcut Flow Models do not support classifier-guidance."
        )

        n_samples = prior.shape[0]
        log = {"sample_history": []}
        model = self.model if not use_ema else self.model_ema

        sampling_schedule_params = sampling_schedule_params or {}

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and 0.0 < warm_start_forward_level < 1.0:
            warm_start_reference = warm_start_reference.to(self.device)
            t_c = torch.ones_like(prior) * warm_start_forward_level
            x1 = torch.randn_like(prior) * t_c + warm_start_reference * (1 - t_c)
            start_t = float(warm_start_forward_level)
        else:
            if x1 is None:
                x1 = torch.randn_like(prior) * temperature
            else:
                assert prior.shape == x1.shape, "prior and x1 must have the same shape"
            start_t = 1.0

        xt = x1
        xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"].append(xt.cpu().numpy())

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = (
                model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            )

        sampling_scheduler = get_sampling_scheduler(sampling_schedule, **sampling_schedule_params)
        t_schedule = sampling_scheduler(
            sample_steps, device=self.device, **sampling_schedule_params
        )
        if start_t != 1.0:
            t_schedule = t_schedule * start_t

        for i in reversed(range(1, sample_steps + 1)):
            t_curr = float(t_schedule[i].item())
            t_next = float(t_schedule[i - 1].item())
            d_val = t_curr - t_next  # positive

            t = torch.full((n_samples,), t_curr, dtype=torch.float32, device=self.device)
            d_tensor = torch.full((n_samples,), d_val, dtype=torch.float32, device=self.device)

            with torch.set_grad_enabled(requires_grad):
                if self.guided:
                    # iSM Intrinsic Guidance: one guided forward with w as an input
                    # (condition_vec_cfg=None & w=0 ⇒ unconditional; w>0 ⇒ guided).
                    vel = model["diffusion"](xt, t, condition_vec_cfg, d=d_tensor, w=w)
                elif w_cfg == 1.0:
                    vel = model["diffusion"](xt, t, condition_vec_cfg, d=d_tensor)
                elif w_cfg == 0.0 or condition_vec_cfg is None:
                    vel = model["diffusion"](xt, t, None, d=d_tensor)
                else:
                    condition = dict_apply(condition_vec_cfg, concat_zeros, dim=0)
                    vel_all = model["diffusion"](
                        einops.repeat(xt, "b ... -> (2 b) ..."),
                        t.repeat(2),
                        condition,
                        d=d_tensor.repeat(2),
                    )
                    vel, vel_uncond = torch.chunk(vel_all, 2, dim=0)
                    vel = w_cfg * vel + (1 - w_cfg) * vel_uncond

            # Euler-like step toward data (decreasing t)
            xt = xt + at_least_ndim(d_tensor, xt.dim()) * vel
            xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask

            if preserve_history:
                log["sample_history"].append(xt.cpu().numpy())

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        log["t_schedule"] = t_schedule
        return xt, log


if __name__ == "__main__":
    from cleandiffuser.nn_diffusion import DiT1dShortcut

    nn_diffusion = DiT1dShortcut(
        x_dim=11, x_seq_len=32, emb_dim=64, d_model=128, n_heads=4, depth=2,
        timestep_emb_type="untrainable_fourier", timestep_emb_params={"scale": 0.02},
    )
    flow = ContinuousShortcutFlow(nn_diffusion)
    prior = torch.zeros((2, 32, 11))
    x, _ = flow.sample(prior, sample_steps=1)
    print("1-step sample:", x.shape)
    x, _ = flow.sample(prior, sample_steps=4)
    print("4-step sample:", x.shape)
