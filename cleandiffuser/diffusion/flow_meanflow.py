"""Continuous-time MeanFlow / improved-MeanFlow (iMF) backbone for the few-step planner.

Implements ``ContinuousMeanFlow`` (Geng et al., 2025, "Mean Flows for One-step
Generative Modeling", arXiv:2505.13447) plus the iMF guidance-as-conditioning and
``imf_vloss`` reparameterisation (arXiv:2512.02012). A single network predicts the
*average* velocity over ``[r, t]``, so one forward pass integrates a whole sampling
step (``sample_steps=1`` ⇒ one-step generation). See the class docstring for the
MeanFlow identity/loss and ``planning/IMPLEMENTATION_PLAN.md`` for project context.
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


class ContinuousMeanFlow(DiffusionModel):
    """Continuous-time MeanFlow model (Geng et al., 2025, arXiv:2505.13447).

    A single network ``u_θ(xt, r, t, c)`` predicts the **average velocity** over
    the interval ``[r, t]`` along the linear path ``xt = (1-t)·x0 + t·ε`` (our
    convention: ``t=1`` noise, ``t=0`` data; instantaneous velocity
    ``v* = x0 − ε``). It is trained with the **MeanFlow identity**

        u(z, r, t) = v(z, t) − (t − r) · d/dt u(z, r, t)

    where the total time-derivative ``du/dt`` (along the forward path, whose
    geometric velocity is ``dz/dt = ε − x0 = −v`` in our data-ward convention) is
    obtained by a forward-mode JVP through the network w.r.t. ``(z, r, t)`` with
    tangent ``(−v, 0, 1)`` (``torch.func.jvp``). The regression target
    ``(v − (t−r)·du/dt)`` is stop-gradient'd; loss is an adaptively-weighted MSE
    ``1/(mse+ε)^p`` (Geng et al. §4). When ``r = t`` the identity collapses to
    plain flow matching (``u = v``), so a fraction of each batch is trained with
    ``r = t``.

    Sampling integrates ``t: 1 → 0`` with ``x_{t-d} = x_t + d · u(x_t, t-d, t)``;
    ``sample_steps = 1`` gives one-step generation.

    Guidance:
      * **Training-time baked CFG** (``cfg_omega``/``cfg_kappa`` ≠ unguided):
        the velocity in the identity is replaced by a CFG-tilted velocity
        ``ω·v + κ·u_cond + (1−ω−κ)·u_uncond`` (stop-grad), so a *fixed* guidance
        scale is baked into the weights (category-(i)). This is mainly for the
        CPU smoke / a category-(i) comparison point.
      * For inference-time control over the guidance scale use the ``iMF``
        variant (``guided=True`` + a ``w`` input on ``DiT1dMeanFlow``); the real
        GPU CFG runs use that, not the baked scale here.

    Args mirror ``ContinuousShortcutFlow`` plus:
        time_mu, time_sigma (float): logit-normal params for sampling ``(r, t)``.
        r_not_equal_t_ratio (float): fraction of the batch with ``r < t`` (the
            rest use ``r = t`` ⇒ flow matching). Default 0.75.
        adaptive_p (float): exponent of the adaptive-MSE weight (0 ⇒ plain MSE).
        cfg_omega, cfg_kappa (float): baked-CFG mix (``ω=1, κ=0`` ⇒ unguided).
        use_jvp (bool): JVP via ``torch.func.jvp`` (default) or a finite-
            difference ``du/dt`` fallback if the JVP is unavailable.
        imf_vloss (bool): iMF *v-loss* reparameterisation (arXiv:2512.02012).
            When ``False`` (default) the JVP z-tangent is the conditional/CFG
            velocity ``−v_eff`` (= original MeanFlow). When ``True`` it is the
            model's own instantaneous velocity ``−v_θ = −u_θ(z,t,t)`` (boundary
            condition, stop-grad), which makes the regression target network-
            independent (lower-variance, more stable) at the cost of one extra
            forward per ``r<t`` step. With the JVP stop-grad'd the two share the
            same ``target``/gradient *except* for this tangent. Inference is
            identical either way. Default ``False`` (compute-cheap baseline).
    """

    def __init__(
        self,
        nn_diffusion: BaseNNDiffusion,
        nn_condition: Optional[BaseNNCondition] = None,
        fix_mask: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
        classifier: Optional[BaseClassifier] = None,
        ema_rate: float = 0.9999,
        optimizer_params: Optional[dict] = None,
        x_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
        time_mu: float = -0.4,
        time_sigma: float = 1.0,
        r_not_equal_t_ratio: float = 0.75,
        adaptive_p: float = 1.0,
        cfg_omega: float = 1.0,
        cfg_kappa: float = 0.0,
        use_jvp: bool = True,
        imf_vloss: bool = False,
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
        assert classifier is None, "MeanFlow does not support classifier-guidance."
        assert 0.0 < r_not_equal_t_ratio <= 1.0

        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.r_not_equal_t_ratio = r_not_equal_t_ratio
        self.adaptive_p = adaptive_p
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.use_jvp = use_jvp
        self.imf_vloss = imf_vloss  # iMF v-loss: JVP tangent = model's own velocity v_θ
        self.guided = guided  # iMF: condition on the guidance scale w (inference-time CFG)
        self.w_min = w_min
        self.w_max = w_max
        # Optional external guidance-gradient hook ``(xt, t) -> ∇E`` (same shape as ``xt``). When set
        # on a guided model it swaps the CFG-tilted identity velocity for the *energy*-tilted
        # ``v + w·[t/(1−t)]·∇E`` (iMF energy guidance; the MeanFlow bootstrap enforces its
        # average-velocity self-consistency). ``None`` (default) ⇒ the CFG target is used and this path
        # is byte-identical to the original iMF. Set by the driver (e.g. E1's ∇E) before training.
        self.energy_grad_fn = None

        self.x_max = nn.Parameter(x_max, requires_grad=False) if x_max is not None else None
        self.x_min = nn.Parameter(x_min, requires_grad=False) if x_min is not None else None

    @property
    def supported_solvers(self):
        return ["euler_meanflow"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    @property
    def baked_cfg(self):
        return (self.cfg_omega != 1.0) or (self.cfg_kappa != 0.0)

    # ==================== Training ======================

    def _sample_r_t(self, B: int):
        """Logit-normal ``(r, t)`` sorted so ``r ≤ t``; force ``r = t`` on a
        ``(1 - r_not_equal_t_ratio)`` fraction (pure flow matching)."""
        rt = torch.sigmoid(
            torch.randn((B, 2), device=self.device) * self.time_sigma + self.time_mu
        )
        rt, _ = torch.sort(rt, dim=1)
        r, t = rt[:, 0], rt[:, 1]
        fm_mask = torch.rand((B,), device=self.device) >= self.r_not_equal_t_ratio  # r==t
        r = torch.where(fm_mask, t, r)
        return r, t, fm_mask

    def loss(
        self,
        x0: torch.Tensor,
        condition: Optional[Union[torch.Tensor, TensorDict]] = None,
        x1: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ):
        """MeanFlow training loss on one batch (the MeanFlow identity with a JVP
        ``du/dt`` term; see the class docstring and the ``imf_vloss`` note in ``loss``).

        Args:
            x0: Clean data batch ``(B, H, D)`` — the ``t=0`` endpoint of the path.
            condition: Network conditioning (e.g. goal / return embedding input), or
                ``None`` for the unconditional model.
            x1: Optional fixed noise ``(B, H, D)`` — the ``t=1`` endpoint; a fresh
                standard normal is drawn when ``None``.
            return_components: If ``True``, also return a dict with the flow-matching
                (``r=t``) and mean-flow (``r<t``) loss components for logging.

        Returns:
            The scalar loss, or ``(loss, components)`` when ``return_components`` is set.
        """
        B = x0.shape[0]
        eps = torch.randn_like(x0) if x1 is None else x1
        cond_emb = self.model["condition"](condition) if condition is not None else None

        r, t, fm_mask = self._sample_r_t(B)
        xt = x0 + at_least_ndim(t, x0.dim()) * (eps - x0)  # (1-t)x0 + t·ε
        xt = xt * (1.0 - self.fix_mask) + x0 * self.fix_mask
        v = x0 - eps  # instantaneous data-ward velocity

        # Effective (optionally CFG-tilted) velocity used in the identity, plus the
        # iMF guidance-scale input ``w_in`` (None unless training the guided variant).
        w_in = None
        if self.guided and self.energy_grad_fn is not None and cond_emb is not None:
            # iMF energy guidance: the identity's instantaneous velocity becomes the ENERGY-tilted
            # v + w·[t/(1−t)]·∇E (∇E = score_pt − score_qt analytic, or a learned CEP gradient), so the
            # MeanFlow bootstrap enforces the average-velocity consistency of this field at the sampled
            # w. w=0 ⇒ unguided q0, w=1 ⇒ exact p0 (analytic ∇E). Gate the noise boundary + cap the
            # per-sample correction norm (protects a never-exactly-zero learned ∇E).
            w_in = torch.rand((B,), device=self.device) * (self.w_max - self.w_min) + self.w_min
            gE = self.energy_grad_fn(xt, t)
            coef = torch.where((1.0 - t) < 1e-3, torch.zeros_like(t), w_in * t / (1.0 - t))
            corr = at_least_ndim(coef, gE.dim()) * gE
            cn = corr.flatten(1).norm(dim=1)
            corr = corr * at_least_ndim(torch.clamp(50.0 / (cn + 1e-9), max=1.0), corr.dim())
            v_eff = (v + corr).detach()
        elif self.guided and cond_emb is not None:
            # iMF: condition on a sampled guidance scale w; regress u_w to the CFG-tilted
            # velocity (1+w)·v_cond − w·v_uncond (uncond from the EMA net, stop-grad).
            w_in = torch.rand((B,), device=self.device) * (self.w_max - self.w_min) + self.w_min
            with torch.no_grad():
                v_uncond = self.model_ema["diffusion"](xt, t, None, r=t, w=torch.zeros_like(t))
            ww = at_least_ndim(w_in, v.dim())
            v_eff = ((1.0 + ww) * v - ww * v_uncond).detach()
        elif self.baked_cfg and cond_emb is not None:
            with torch.no_grad():
                u_cond = self.model["diffusion"](xt, t, cond_emb, r=t)
                u_uncond = self.model["diffusion"](xt, t, None, r=t)
            v_eff = (
                self.cfg_omega * v
                + self.cfg_kappa * u_cond
                + (1.0 - self.cfg_omega - self.cfg_kappa) * u_uncond
            ).detach()
        else:
            v_eff = v

        net = self.model["diffusion"]

        def fn(z, rr, tt):
            return net(z, tt, cond_emb, r=rr, w=w_in)

        # The MeanFlow identity needs du/dt = ∂_t u + ∂_z u · (dz/dt), the TOTAL
        # derivative along the forward path z_t = (1-t)x0 + t·ε. The geometric flow
        # velocity is dz/dt = ε − x0 = −(instantaneous velocity) (our data-ward
        # convention defines v_eff = x0 − ε), so the JVP z-tangent is the *negated*
        # instantaneous velocity. Two choices of that velocity:
        #   • original MeanFlow (default): the conditional/CFG velocity v_eff (= ε−x
        #     per sample) ⇒ tangent −v_eff.
        #   • iMF v-loss (``imf_vloss``): the model's OWN instantaneous velocity
        #     v_θ = u_θ(z,t,t) (boundary condition, stop-grad) ⇒ tangent −v_θ. This
        #     makes the regression target network-independent (lower-variance, more
        #     stable; arXiv:2512.02012) at the cost of one extra forward. With the JVP
        #     stop-grad'd the two share the same `target`/gradient *except* for this
        #     tangent (the `target` below still uses the data-ward v_eff either way).
        # (See validate_meanflow.py's JVP-vs-finite-difference + tangent-variance checks.)
        if self.imf_vloss:
            v_theta = fn(xt, t, t).detach()  # u_θ(z,t,t): model's instantaneous velocity
            dz_dt = -v_theta
        else:
            dz_dt = -v_eff
        if self.use_jvp:
            u, dudt = torch.func.jvp(
                fn, (xt, r, t), (dz_dt, torch.zeros_like(r), torch.ones_like(t))
            )
        else:
            # finite-difference du/dt along (dz=dz_dt, dr=0, dt=1)
            h = 1e-3
            u = fn(xt, r, t)
            u_h = fn(xt + h * dz_dt, r, t + h)
            dudt = (u_h - u) / h

        target = (v_eff - at_least_ndim(t - r, v_eff.dim()) * dudt).detach()
        err = (u - target) ** 2 * self.loss_weight * (1.0 - self.fix_mask)
        mse = err.flatten(1).mean(1)  # per-sample

        if self.adaptive_p > 0.0:
            w = 1.0 / (mse.detach() + 1e-3) ** self.adaptive_p
            total_loss = (w * mse).mean()
        else:
            total_loss = mse.mean()

        if return_components:
            loss_fm = mse[fm_mask].mean().detach() if bool(fm_mask.any()) else x0.new_zeros(())
            loss_mf = mse[~fm_mask].mean().detach() if bool((~fm_mask).any()) else x0.new_zeros(())
            return total_loss, {"loss_fm": loss_fm, "loss_mf": loss_mf}
        return total_loss

    def training_step(self, batch, batch_idx):
        """One Lightning training step: compute the loss, log its FM/MF components,
        and update the EMA weights on schedule.

        Args:
            batch: Dict with ``"x0"`` (required clean data) and optional
                ``"condition_cfg"`` and ``"x1"`` (fixed noise).
            batch_idx: Lightning batch index, used to gate the EMA update schedule.

        Returns:
            The scalar training loss (for the optimizer).
        """
        assert isinstance(batch, dict) and "x0" in batch.keys()
        x0 = batch["x0"]
        condition_cfg = batch.get("condition_cfg", None)
        x1 = batch.get("x1", None)

        loss, components = self.loss(x0, condition_cfg, x1=x1, return_components=True)
        self.log("diffusion_loss", loss, prog_bar=True)
        self.log("fm_loss", components["loss_fm"], prog_bar=False)
        self.log("mf_loss", components["loss_mf"], prog_bar=False)

        if self.ema_update_schedule(batch_idx):
            self.ema_update()
        return loss

    # ==================== Sampling ======================

    def sample(
        self,
        prior: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        solver: str = "euler_meanflow",
        sample_steps: int = 1,
        sampling_schedule: str = "linear",
        sampling_schedule_params: Optional[dict] = None,
        use_ema: bool = True,
        temperature: float = 1.0,
        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        mask_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cfg: float = 1.0,
        condition_cg: None = None,
        w_cg: float = 0.0,
        warm_start_reference: Optional[torch.Tensor] = None,
        warm_start_forward_level: float = 0.3,
        requires_grad: bool = False,
        preserve_history: bool = False,
        **kwargs,
    ):
        """Euler sampling of the average-velocity field from ``t=1`` to ``t=0``.

        Each step integrates the whole interval ``[t_next, t_curr]`` in one call:
        ``x ← x + (t_curr − t_next) · u(x, r=t_next, t=t_curr, c)``. For vanilla
        MeanFlow guidance is *baked* (use ``w_cfg=1``); a post-hoc CFG blend is
        still available for ``w_cfg ∉ {0, 1}`` for experimentation.
        """
        assert solver in self.supported_solvers, f"Solver {solver} is not supported."
        assert w_cg == 0.0 and condition_cg is None, "MeanFlow does not support classifier-guidance."

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
                assert prior.shape == x1.shape
            start_t = 1.0

        xt = x1 * (1.0 - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"].append(xt.cpu().numpy())

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = (
                model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            )

        sampling_scheduler = get_sampling_scheduler(sampling_schedule, **sampling_schedule_params)
        t_schedule = sampling_scheduler(sample_steps, device=self.device, **sampling_schedule_params)
        if start_t != 1.0:
            t_schedule = t_schedule * start_t

        for i in reversed(range(1, sample_steps + 1)):
            t_curr = float(t_schedule[i].item())
            t_next = float(t_schedule[i - 1].item())
            d_val = t_curr - t_next  # positive interval length

            t = torch.full((n_samples,), t_curr, dtype=torch.float32, device=self.device)
            r = torch.full((n_samples,), t_next, dtype=torch.float32, device=self.device)

            with torch.set_grad_enabled(requires_grad):
                if self.guided and condition_vec_cfg is not None:
                    # iMF: one forward; w_cfg is the textbook guidance strength (0 = unguided)
                    w_in = torch.full((n_samples,), float(w_cfg), dtype=torch.float32, device=self.device)
                    vel = model["diffusion"](xt, t, condition_vec_cfg, r=r, w=w_in)
                elif w_cfg == 1.0 or condition_vec_cfg is None:
                    vel = model["diffusion"](xt, t, condition_vec_cfg, r=r)
                elif w_cfg == 0.0:
                    vel = model["diffusion"](xt, t, None, r=r)
                else:
                    condition = dict_apply(condition_vec_cfg, concat_zeros, dim=0)
                    vel_all = model["diffusion"](
                        einops.repeat(xt, "b ... -> (2 b) ..."),
                        t.repeat(2),
                        condition,
                        r=r.repeat(2),
                    )
                    vel, vel_uncond = torch.chunk(vel_all, 2, dim=0)
                    vel = w_cfg * vel + (1 - w_cfg) * vel_uncond

            xt = xt + d_val * vel
            xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"].append(xt.cpu().numpy())

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        log["t_schedule"] = t_schedule
        return xt, log


if __name__ == "__main__":
    from cleandiffuser.nn_diffusion import DiT1dMeanFlow

    nn_diffusion = DiT1dMeanFlow(
        x_dim=4, x_seq_len=8, emb_dim=64, d_model=128, n_heads=4, depth=2,
        timestep_emb_type="untrainable_fourier", timestep_emb_params={"scale": 0.02},
    )
    flow = ContinuousMeanFlow(nn_diffusion)
    x0 = torch.randn((16, 8, 4))
    loss, comp = flow.loss(x0, return_components=True)
    loss.backward()
    print("loss", float(loss), "components", {k: float(v) for k, v in comp.items()})
    prior = torch.zeros((2, 8, 4))
    for N in (1, 4, 32):
        x, _ = flow.sample(prior, sample_steps=N)
        print(f"N={N} sample", x.shape, "finite", bool(torch.isfinite(x).all()))
