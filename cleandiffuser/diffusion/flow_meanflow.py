"""Continuous-time MeanFlow / improved-MeanFlow (iMF) backbone for the few-step planner.

Implements ``ContinuousMeanFlow`` (Geng et al., 2025, "Mean Flows for One-step
Generative Modeling", arXiv:2505.13447) plus the iMF guidance-as-conditioning and
``imf_vloss`` reparameterisation (arXiv:2512.02012). A single network predicts the
*average* velocity over ``[r, t]``, so one forward pass integrates a whole sampling
step (``sample_steps=1`` ⇒ one-step generation). The forward process, guided-target
tilts and Euler sampling scaffold live in ``ContinuousFlowMap`` (``flow_map.py``); this
class contributes the JVP MeanFlow-identity loss and the ``r``-spanned per-step velocity.
See the class docstring for the identity and ``planning/IMPLEMENTATION_PLAN.md`` for context.
"""

import logging
from typing import Optional, Union

import einops
import torch

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.diffusion.flow_map import ContinuousFlowMap

_flow_map_log = logging.getLogger("cleandiffuser.diffusion.flow_meanflow")
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import TensorDict, at_least_ndim, concat_zeros, dict_apply


class ContinuousMeanFlow(ContinuousFlowMap):
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

    Relation to the reference iMF (arXiv:2512.02012, github.com/Lyy-iiis/imeanflow) —
    our ``guided=True`` is a documented **simplification**: the reference regresses to
    ``v_t + (1−1/ω)(v_cond − v_uncond)`` with *live-net* cond/uncond legs, samples ω from a
    power law on ``[1, 1+s_max]``, and additionally conditions on a guidance *interval*
    ``[t_min, t_max]``; we use the standard CFG extrapolation ``(1+w)·v_t − w·v_uncond``
    with the **EMA** net's ``(cond=None, w=0)`` leg, ``w ~ U[w_min, w_max]``, and no
    interval input. ``imf_vloss`` keeps the reference's tangent (the w-conditioned
    ``v_θ = u_θ(z,t,t,w)``) as a tangent swap — gradient-identical to the reference's
    compound-predictor form since the swapped term is stop-grad.

    Guidance-tilt placement (adjudicated in ``planning/code_review_2026-07.md``): the
    CFG/energy tilt enters ``v_eff``, which feeds **both** the identity target and the
    JVP z-tangent — the identity's ``du/dt`` is the total derivative *along the flow of
    the field being learned*, so the tangent must be the tilted field's velocity. This
    differs deliberately from ``ContinuousShortcutFlow`` (tilt on the d=0 FM target only,
    with the target-agnostic SC bootstrap propagating it): each is the unique correct
    port of its own objective, not an inconsistency.

    Args mirror ``ContinuousShortcutFlow`` plus:
        time_mu, time_sigma (float): logit-normal params for sampling ``(r, t)``.
        r_not_equal_t_ratio (float): fraction of the batch with ``r < t`` (the
            rest use ``r = t`` ⇒ flow matching). Default 0.25 — the reference
            optimum (arXiv:2505.13447 Table 1a: 25% r≠t best, FID 61.06 vs
            67.32 @100%; official repo ``data_proportion=0.75`` ⇒ 25% r≠t;
            iMF uses 50%). Until 2026-07 this defaulted to 0.75 (inverted vs
            the reference — review finding B2); runs record their value in
            ``config.json``.
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
        r_not_equal_t_ratio: float = 0.25,
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
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier,
            ema_rate, optimizer_params, x_max=x_max, x_min=x_min,
            guided=guided, w_min=w_min, w_max=w_max,
        )
        assert 0.0 < r_not_equal_t_ratio <= 1.0

        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.r_not_equal_t_ratio = r_not_equal_t_ratio
        self.adaptive_p = adaptive_p
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.use_jvp = use_jvp
        self.imf_vloss = imf_vloss  # iMF v-loss: JVP tangent = model's own velocity v_θ

    @property
    def supported_solvers(self):
        return ["euler_meanflow"]

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
        # Condition-free nets (``nn_condition=None``, e.g. the intrinsic-energy family, F7)
        # never see a cond input — ``condition`` then only feeds the ∇E callback below.
        cond_emb = (self.model["condition"](condition)
                    if (condition is not None and self._has_condition) else None)

        r, t, fm_mask = self._sample_r_t(B)
        xt = x0 + at_least_ndim(t, x0.dim()) * (eps - x0)  # (1-t)x0 + t·ε
        xt = xt * (1.0 - self.fix_mask) + x0 * self.fix_mask
        v = x0 - eps  # instantaneous data-ward velocity

        # Effective (optionally CFG-tilted) velocity used in the identity, plus the
        # iMF guidance-scale input ``w_in`` (None unless training the guided variant).
        w_in = None
        if self.guided and self.energy_grad_fn is not None:
            # iMF energy guidance: the identity's instantaneous velocity becomes the
            # ENERGY-tilted v + w·[t/(1−t)]·∇E (∇E = score_pt − score_qt analytic, or a
            # learned CEP gradient), so the MeanFlow bootstrap enforces the average-velocity
            # consistency of this field at the sampled w. w=0 ⇒ unguided q0, w=1 ⇒ exact p0
            # (analytic ∇E). Gate/cap: _energy_tilt. No net condition required — the
            # intrinsic-energy family trains condition-free (F7): a cond input would let the
            # net fit the reward-conditional base instead of q0 (w=0 contamination).
            w_in = self._sample_w(B)
            # Hand the callback the raw condition too (per-sample goals on Maze2D;
            # condition-free callbacks — e.g. the E1 toy's analytic ∇E — just ignore it).
            cond_raw = condition if isinstance(condition, torch.Tensor) else None
            v_eff = self._energy_tilt(xt, t, v, w_in, cond_raw)
        elif self.guided and cond_emb is not None:
            # iMF: condition on a sampled guidance scale w; regress u_w to the CFG-tilted
            # velocity (1+w)·v_cond − w·v_uncond (uncond from the EMA net at the
            # label-dropout null token, stop-grad; run_review_2026-07-14 F1).
            w_in = self._sample_w(B)
            v_eff = self._cfg_tilt(xt, t, v, w_in, cond_emb, r=t)
        elif self.baked_cfg and cond_emb is not None:
            with torch.no_grad():
                u_cond = self.model["diffusion"](xt, t, cond_emb, r=t)
                u_uncond = self.model["diffusion"](xt, t, torch.zeros_like(cond_emb), r=t)
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
            batch: Dict with ``x0`` (required), ``condition_cfg`` / ``x1`` (optional).
            batch_idx: Lightning batch index (drives the EMA update schedule).

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

    def _default_w_cfg(self) -> float:
        return 1.0

    def _step_velocity(self, model, xt, t, t_curr, t_next, condition_vec_cfg, w, w_cfg):
        """MeanFlow per-step velocity: one forward integrating ``[t_next, t_curr]`` via
        ``r = t_next``.

        Guided (iMF) models do a single guided forward with the dedicated ``w`` kwarg as
        the textbook strength (0 = conditional; ``condition_vec_cfg=None`` & ``w=0`` ⇒ the
        unconditional field, queried at the label-dropout null token for conditional nets
        (F1) — mirrors ``ContinuousShortcutFlow``). Legacy spelling: guided
        callers that passed the scale through ``w_cfg`` (pre-2026-07) still work via a
        shim + WARNING, except the ambiguous ``w_cfg=1.0`` (indistinguishable from the
        class default), which now means ``w=0``. Unguided models use the plain
        conditional/unconditional forward or, for ``w_cfg ∉ {0, 1}``, the legacy
        double-batch post-hoc CFG blend (the documented compounding baseline).
        """
        r = torch.full_like(t, t_next)
        if self.guided:
            w_eff = float(w)
            if w_eff == 0.0 and w_cfg is not None and w_cfg != 1.0:
                _flow_map_log.warning(
                    "ContinuousMeanFlow: guided sampling received the scale via the legacy "
                    "w_cfg=%s — pass w=<scale> instead (w_cfg shim kept for back-compat).",
                    w_cfg,
                )
                w_eff = float(w_cfg)
            w_in = torch.full_like(t, w_eff)
            if condition_vec_cfg is None:
                condition_vec_cfg = self._null_cond_vec(model, xt.shape[0])
            return model["diffusion"](xt, t, condition_vec_cfg, r=r, w=w_in)
        if condition_vec_cfg is None:
            return model["diffusion"](xt, t, self._null_cond_vec(model, xt.shape[0]), r=r)
        if w_cfg == 1.0:
            return model["diffusion"](xt, t, condition_vec_cfg, r=r)
        if w_cfg == 0.0:
            return model["diffusion"](xt, t, self._null_cond_vec(model, xt.shape[0]), r=r)
        condition = dict_apply(condition_vec_cfg, concat_zeros, dim=0)
        vel_all = model["diffusion"](
            einops.repeat(xt, "b ... -> (2 b) ..."),
            t.repeat(2),
            condition,
            r=r.repeat(2),
        )
        vel, vel_uncond = torch.chunk(vel_all, 2, dim=0)
        return w_cfg * vel + (1 - w_cfg) * vel_uncond
