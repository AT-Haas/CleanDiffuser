"""Continuous-time Shortcut Flow backbone for the few-step planner.

Implements ``ContinuousShortcutFlow`` (Frans et al., 2024, "One Step Diffusion via
Shortcut Models", arXiv:2410.12557): one network predicts the *average* velocity of a
finite jump of size ``d`` (``d=0`` ⇒ flow matching), trained with a flow-matching +
EMA self-consistency objective so a single forward pass can span a whole ``t=1→0`` step.
The forward process, guided-target tilts and Euler sampling scaffold live in
``ContinuousFlowMap`` (``flow_map.py``); this class contributes the hybrid FM+SC loss and
the ``d``-spanned per-step velocity. See ``planning/2026-06-19_IMPLEMENTATION_PLAN.md`` for context.
"""

from typing import Optional, Union

import einops
import torch

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.diffusion.flow_map import ContinuousFlowMap
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import TensorDict, at_least_ndim, concat_zeros, dict_apply


class ContinuousShortcutFlow(ContinuousFlowMap):
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

    Guidance-tilt placement (adjudicated in ``planning/2026-07-07_code_review.md``): the
    CFG/energy tilt enters **only the d=0 FM target** here — the SC bootstrap is
    *target-agnostic* (it threads the sampled ``w`` through the target net's own half-
    and full-steps), so whatever w-field the FM branch pins down at ``d=0`` is exactly
    the field whose self-consistency the SC branch enforces at ``d>0``. This differs
    deliberately from ``ContinuousMeanFlow`` (tilt on the whole identity incl. the JVP
    tangent): each is the unique correct port of its own objective, not an inconsistency.

    Faithfulness notes vs the reference (github.com/kvfrans/shortcut-models,
    ``targets_shortcut.py``; details in ``planning/2026-07-07_code_review.md`` N1): our
    SC branch trains LHS steps down to ``d=2^-K_max`` (the reference's smallest LHS
    bootstrap ``d`` is one octave larger, with ``2^-7`` only as a target half-step, and
    its FM branch uses a smallest-``d`` token where we use exact ``d=0`` — the paper's
    §3 form); we do not clip bootstrap intermediates/targets to ``[-4, 4]`` (image-
    domain stabilizer); with ``discrete_t=True`` our ``t ∈ {d, …, 1}`` grid is exactly
    the reference's ``{0, d, …, 1−d}`` after the noise-end convention flip, and the
    reference also ships an EMA-bootstrap option (``bootstrap_ema``), so the
    continuous-t + EMA default here deviates only in the *default*, not in kind.

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
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier,
            ema_rate, optimizer_params, x_max=x_max, x_min=x_min,
            guided=guided, w_min=w_min, w_max=w_max,
        )
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

    @property
    def supported_solvers(self):
        return ["euler_shortcut"]

    # ==================== Training ======================

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
        # objective well-defined under random label dropout. Condition-free nets
        # (``nn_condition=None``, e.g. the intrinsic-energy family, F7) never see a
        # cond input — ``condition`` then only feeds the ∇E callback's tilt below.
        cond_emb = (self.model["condition"](condition)
                    if (condition is not None and self._has_condition) else None)

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
            if self.guided and self.energy_grad_fn is not None:
                # iSM energy guidance: regress s_θ(xt,t,0,w) to the ENERGY-tilted velocity
                # v + w·[t/(1−t)]·∇E (∇E = score_pt − score_qt analytic, or a learned CEP
                # gradient). w=0 ⇒ unguided q0, w=1 ⇒ exact p0 (analytic ∇E). The SC branch
                # threads the same w through its own half-/full-steps, so it enforces this
                # field's self-consistency for free (target-agnostic). Gate/cap: _energy_tilt.
                # No net condition required: the intrinsic-energy family trains condition-free
                # (F7) — a cond input would let the net fit the reward-conditional base field
                # instead of q0 (the w=0 contamination seen in the 2026-07 converged runs).
                w_fm = self._sample_w(B_fm)
                # Hand the callback the raw condition slice (per-sample goals on Maze2D;
                # condition-free callbacks — e.g. the E1 toy's analytic ∇E — just ignore it).
                cond_raw_fm = condition[:B_fm] if isinstance(condition, torch.Tensor) else None
                target_fm = self._energy_tilt(xt_fm, t_fm, v_fm, w_fm, cond_raw_fm)
            elif self.guided and cond_fm is not None:
                # iSM Intrinsic Guidance: regress s_θ(xt,t,c,0,w) to the CFG-tilted velocity
                # (1+w)·v_cond − w·v_uncond (v_uncond from the EMA net at the label-dropout
                # null token + w=0, stop-grad; 2026-07-14_run_review F1).
                w_fm = self._sample_w(B_fm)
                target_fm = self._cfg_tilt(xt_fm, t_fm, v_fm, w_fm, cond_fm, d=d_fm)
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
            # Gated on ``guided`` alone: condition-free guided nets (intrinsic-energy, F7)
            # have cond_sc=None but still need their w-field's self-consistency enforced.
            w_sc = self._sample_w(B_sc) if self.guided else None

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

        # ---------- Reward loss (ractd; RACTD arXiv:2506.07822 Eq. 8–9) ----------
        # Default placement is "sc": the gradient reaches the net only through the sampled
        # rollout, whose every query is at d>0 — the jump map, RACTD's *student*. The d=0 FM
        # target above is untouched, which is the invariant `validate_guidance` gates on.
        # NOTE this deliberately breaks the class docstring's "SC bootstrap is target-agnostic"
        # property: the jump map is no longer the faithful self-consistent distillation of the
        # d=0 field. That divergence IS the mechanism (a displacement, not a reweighting) and
        # is readable from the loss_fm/loss_sc gap — it is not a bug.
        loss_reward = x0.new_zeros(())
        if self.reward_active() and self.reward_placement in ("sc", "anyt", "fm_loss", "both"):
            loss_reward = self.reward_loss_total(
                x0, condition,
                condition if isinstance(condition, torch.Tensor) else None)
            total_loss = total_loss + self.reward_sigma * loss_reward

        if return_components:
            return total_loss, {"loss_fm": loss_fm.detach(), "loss_sc": loss_sc.detach(),
                                "loss_reward": loss_reward.detach()}
        return total_loss

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

    def _default_w_cfg(self) -> float:
        return 0.0

    def _jump_to_data(self, model, xt, t, condition_vec, instantaneous: bool = False):
        """Shortcut jump to data: one forward at span ``d = t`` (the ``anyt`` placement)."""
        return model["diffusion"](xt, t, condition_vec, d=torch.zeros_like(t) if instantaneous else t,
                                  w=torch.zeros_like(t) if self.guided else None)

    def _step_velocity(self, model, xt, t, t_curr, t_next, condition_vec_cfg, w, w_cfg):
        """Shortcut per-step velocity: one forward at step size ``d = t_curr − t_next``.

        Guided models do a single guided forward with ``w`` as an input
        (``condition_vec_cfg=None`` & ``w=0`` ⇒ the unconditional field — queried at the
        label-dropout null token for conditional nets, F1); unguided models use the plain
        conditional/unconditional forward or, for ``w_cfg ∉ {0, 1}``, the legacy
        double-batch post-hoc CFG blend ``w_cfg·v_cond + (1−w_cfg)·v_uncond`` (the
        documented compounding baseline).
        """
        d_tensor = torch.full_like(t, t_curr - t_next)
        if self.guided:
            # iSM Intrinsic Guidance: one guided forward with w as an input
            # (condition_vec_cfg=None & w=0 ⇒ unconditional; w>0 ⇒ guided).
            if condition_vec_cfg is None:
                condition_vec_cfg = self._null_cond_vec(model, xt.shape[0])
            return model["diffusion"](xt, t, condition_vec_cfg, d=d_tensor, w=w)
        if w_cfg == 1.0 and condition_vec_cfg is not None:
            return model["diffusion"](xt, t, condition_vec_cfg, d=d_tensor)
        if w_cfg == 0.0 or condition_vec_cfg is None:
            return model["diffusion"](xt, t, self._null_cond_vec(model, xt.shape[0]),
                                      d=d_tensor)
        condition = dict_apply(condition_vec_cfg, concat_zeros, dim=0)
        vel_all = model["diffusion"](
            einops.repeat(xt, "b ... -> (2 b) ..."),
            t.repeat(2),
            condition,
            d=d_tensor.repeat(2),
        )
        vel, vel_uncond = torch.chunk(vel_all, 2, dim=0)
        return w_cfg * vel + (1 - w_cfg) * vel_uncond


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
