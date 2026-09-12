"""Shared base class for continuous-time flow-map generative models.

``ContinuousFlowMap`` hosts everything ``ContinuousShortcutFlow`` (Frans et al., 2024,
arXiv:2410.12557) and ``ContinuousMeanFlow`` (Geng et al., 2025, arXiv:2505.13447) share:
the linear forward path ``xt = (1-t)·x0 + t·ε`` in our data-ward convention (``t=1`` noise,
``v* = x0 − ε``), clean-inject inpainting via ``fix_mask`` (Diffuser §3.3, arXiv:2205.09991),
the intrinsic-guidance machinery (``w`` sampling plus the CFG/energy target tilts of iSM
arXiv:2510.21250 / iMF arXiv:2512.02012 / CEP arXiv:2304.12824), and the whole Euler
``sample()`` scaffold. Subclasses keep only their objective (``loss``) and the one
backbone-specific line of sampling (``_step_velocity``: shortcut spans via ``d = t−t_next``,
meanflow via ``r = t_next``) — a new flow-map backbone is those two methods plus a solver name.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.diffusion.basic import DiffusionModel
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (TensorDict, at_least_ndim, get_mask, get_sampling_scheduler,
                                 null_cond_emb)

log = logging.getLogger("cleandiffuser.diffusion.flow_map")


class ContinuousFlowMap(DiffusionModel):
    """Base class for continuous-time flow-map models (Shortcut / MeanFlow).

    See the module docstring for scope. Subclasses must provide ``supported_solvers``,
    ``loss``, ``_step_velocity`` and ``_default_w_cfg``; they reuse ``_sample_w`` /
    ``_energy_tilt`` / ``_cfg_tilt`` to build their guided training targets.

    Args:
        nn_diffusion: Backbone accepting the subclass's extra span kwarg (``d=`` / ``r=``)
            and, for guided models, the ``w`` conditioning input.
        nn_condition: Optional condition encoder (CFG label-dropout inside).
        fix_mask: Inpainting mask in ``x_shape``; pinned entries are clean-injected at
            train time and after every Euler step (Diffuser §3.3 clean-inject).
        loss_weight: Optional per-entry loss weight in ``x_shape``.
        classifier: Must be ``None`` — flow maps do not support classifier guidance.
        ema_rate: EMA decay of the target/sampling network.
        optimizer_params: Same convention as the other CleanDiffuser diffusion classes.
        x_max, x_min: Optional output clipping bounds (non-trainable ``nn.Parameter``s,
            same attribute names as before the base-class extraction — state-dict stable).
        guided: Train the intrinsic-guidance variant (``w`` as a network input, iSM/iMF).
            ``False`` ⇒ vanilla model, ``w`` ignored everywhere.
        w_min, w_max: Training range the guidance scale ``w`` is sampled from.
    """

    # The guidance scale at which the CFG blend `(1+w)·v_cond − w·v_uncond` is exactly the
    # unconditional field: (1+w) = 0. Not a family convention — it falls out of the
    # parameterization, which is why the FM anchor can short-circuit it arithmetically.
    CFG_UNCOND_W: float = -1.0

    # Per-sample cap on the energy-guidance correction norm, and the noise-boundary gate
    # on 1−t below which the t/(1−t) tilt is forced to 0: the true tilt vanishes at pure
    # noise, and the gate stops the blow-up of a never-exactly-zero *learned* ∇E.
    ENERGY_CORR_CAP: float = 50.0
    NOISE_GATE: float = 1e-3

    # Per-sample cap on the reward-gradient norm used by the ``ractd`` family — the direct
    # analogue of ENERGY_CORR_CAP. RACTD's own σ-ablation is non-monotone and collapses past
    # σ≈1 (arXiv:2506.07822 App. E.4 Table 15: 0.8→108.3, 1.5→50.5, 2.0→17.0), an
    # uncapped-correction failure; set to 0 to disable the cap and reproduce that cliff.
    REWARD_GRAD_CAP: float = 50.0

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
        guided: bool = False,
        w_min: float = 0.0,
        w_max: float = 4.0,
        cfg_uncond_anchor: bool = False,
    ):
        super().__init__(nn_diffusion, nn_condition, fix_mask, loss_weight, classifier,
                         ema_rate, optimizer_params)
        assert classifier is None, f"{type(self).__name__} does not support classifier-guidance."
        # iSM/iMF Intrinsic Guidance: condition on the guidance scale w (inference-time CFG).
        self.guided = guided
        self.w_min = w_min
        self.w_max = w_max
        # Repair for the measured `cfg` amortization tax (docs/plans/2026-08-11_guided_net_
        # capacity_and_budget.md). Two coupled changes, both inert at the False default so
        # every existing net and checkpoint is byte-identical:
        #   train  — label-dropped rows are given the plain data target `v`, independent of w.
        #            Their condition IS null, so v_cond == v_uncond and the correct target does
        #            not depend on w at all; the historical `(1+w)v − w·v_uncond` is unbiased
        #            for them but carries (1+w)² times the noise (3× on average over w~U[-1,2],
        #            ~10× on maze2d's [0,4] hull) around a value that never moves — and at
        #            w=−1 it carries no data whatsoever.
        #   sample — at w = CFG_UNCOND_W the CFG blend is *exactly* the unconditional field,
        #            so query the null token instead of asking the net to reproduce an identity
        #            it can only learn. This is what the FM anchor already does
        #            (backbones._anchor_span_velocity short-circuits both wc==0 and wc==1),
        #            and it is why the anchor pays no tax while the flow maps do.
        # Together they collapse the two slots that currently both represent q0 (the null token,
        # which is data-trained, and w=−1, which was defined by pointing at it through the EMA)
        # into the one that data can actually reach.
        self.cfg_uncond_anchor = cfg_uncond_anchor
        # Optional external guidance-gradient hook ``(xt, t, cond) -> ∇E`` (same shape as
        # ``xt``; ``cond`` is the raw condition slice or None — condition-free callbacks
        # ignore it). When set on a guided model it swaps the CFG target for the
        # energy-tilted target (``_energy_tilt``); ``None`` (default) keeps the CFG target,
        # byte-identical to plain iSM/iMF. Set by the driver before training.
        self.energy_grad_fn = None
        # Reward-loss (``ractd``) wiring — RACTD's L_Reward = −R(x̂₀) (arXiv:2506.07822 Eq. 8–9)
        # ported onto the flow map. Set by the driver alongside ``energy_grad_fn``; **inert at
        # these defaults** (``reward_sigma = 0``), so every existing net and checkpoint is
        # byte-identical and the σ=0 reproduction gate holds by construction.
        #   reward_grad_fn:    ``(x0_hat, cond) -> ∇r`` at the CLEAN sample — no ``t`` argument,
        #                      because the whole point of the one-step head is that the reward
        #                      is evaluated noise-free (RACTD §3.4).
        #   reward_sigma:      the σ of ``L = … + σ·L_Reward``; a *training-time* knob, so unlike
        #                      cfg's ``w`` it needs one net per value.
        #   reward_placement:  which x̂₀ the reward loss is evaluated on — always RACTD's
        #                      L_Reward, only the attachment point differs:
        #                        "sc"      rollout from pure noise at a reward_nfes budget
        #                                  (RACTD as published; the jump map / *student*).
        #                        "fm_loss" the instantaneous field's exact x₀-prediction
        #                                  x_t + t·u(x_t,t,d=0) — the *teacher* position, and
        #                                  the base of the shortcut d-ladder.
        #                        "anyt"    G_θ(x_t,t,0) at sampled t (span = t).
        #                        "both"    fm_loss + sc.
        #                      See planning/2026-07-27_ractd_reward_loss_arm.md.
        #   reward_nfes:       step budgets the reward term is applied at, drawn uniformly per
        #                      optimizer step. ``(1,)`` reproduces RACTD exactly; a wider tuple
        #                      spreads the pressure so the model stays usable across the N axis.
        self.reward_grad_fn = None
        self.reward_sigma = 0.0
        self.reward_placement = "sc"
        self.reward_nfes = (1,)
        self.x_max = nn.Parameter(x_max, requires_grad=False) if x_max is not None else None
        self.x_min = nn.Parameter(x_min, requires_grad=False) if x_min is not None else None

    # ==================== shared properties / forward process ====================

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    def add_noise(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
    ):
        """Linear-interpolation forward process ``xt = (1-t)·x0 + t·ε`` with clean-inject
        of the ``fix_mask``-pinned entries. Draws ``t ~ U[0,1]`` / ``ε ~ N(0,I)`` if not given."""
        t = torch.rand((x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps
        xt = x0 + at_least_ndim(t, x0.dim()) * (eps - x0)
        xt = xt * (1.0 - self.fix_mask) + x0 * self.fix_mask
        return xt, t, eps

    def update_diffusion(
        self,
        x0: torch.Tensor,
        condition_cfg: Optional[torch.Tensor] = None,
        update_ema: bool = True,
        x1: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """One optimizer step on the diffusion network — a thin override of the framework
        ``DiffusionModel.update_diffusion`` that forwards the optional fixed noise ``x1``
        to ``loss`` (previously present on the shortcut class only; review 2026-07).

        Args:
            x0: Clean data batch ``(B, *x_shape)``.
            condition_cfg: CFG condition input, or ``None``.
            update_ema: Update the EMA weights after the gradient step.
            x1: Optional fixed source/noise sample ``(B, *x_shape)``.
            **kwargs: Accepted for base-class compatibility; ignored.
        """
        return super().update_diffusion(x0, condition_cfg, update_ema, x1=x1)

    # ==================== shared guided-target helpers (loss-side) ====================

    def _sample_w(self, n: int) -> torch.Tensor:
        """Per-sample guidance scales ``w ~ U[w_min, w_max]`` for guided training."""
        return torch.rand((n,), device=self.device) * (self.w_max - self.w_min) + self.w_min

    def _energy_tilt(self, xt, t, v, w, cond_raw):
        """Energy-tilted regression target ``(v + cap(w·[t/(1−t)]·∇E)).detach()``.

        ``∇E = self.energy_grad_fn(xt, t, cond_raw)`` — analytic ``score_pt − score_qt`` or
        a learned CEP gradient (arXiv:2304.12824); ``w=0`` ⇒ unguided, ``w=1`` ⇒ exact
        ``p0`` for the analytic ∇E. The coefficient is gated to 0 at ``1−t < NOISE_GATE``
        and the per-sample correction norm capped at ``ENERGY_CORR_CAP`` (class constants).
        ``cond_raw`` is the caller's raw condition slice (tensor) or ``None``.
        """
        gE = self.energy_grad_fn(xt, t, cond_raw)
        coef = torch.where((1.0 - t) < self.NOISE_GATE, torch.zeros_like(t),
                           w * t / (1.0 - t))
        corr = at_least_ndim(coef, gE.dim()) * gE
        cn = corr.flatten(1).norm(dim=1)
        corr = corr * at_least_ndim(torch.clamp(self.ENERGY_CORR_CAP / (cn + 1e-9), max=1.0),
                                    corr.dim())
        return (v + corr).detach()

    # ---------------- reward loss (the ``ractd`` family) ----------------

    def reward_active(self) -> bool:
        """Whether the ``ractd`` reward term is live (hook set **and** ``σ > 0``). Every
        reward code path is gated on this, so a net built without the hook — or with
        ``σ = 0`` — trains exactly as it did before this feature existed."""
        return self.reward_grad_fn is not None and self.reward_sigma > 0.0

    def _reward_grad(self, x, cond_raw):
        """Per-sample ``∇r`` at ``x``, norm-capped at ``REWARD_GRAD_CAP`` and detached.

        Detaching is not an approximation: ``∇r`` is the gradient *evaluated at* ``x``, and
        the surrogate below re-attaches it to the graph through ``x0_hat``. Caps are applied
        per sample (the ``_energy_tilt`` idiom) so one runaway sample cannot dominate a batch.

        Args:
            x: Points at which to evaluate the reward gradient (clean samples for the loss,
                noisy iterates for the ``fm`` tilt).
            cond_raw: Raw condition slice handed to the callback, or ``None``.
        """
        g = self.reward_grad_fn(x, cond_raw)
        if self.REWARD_GRAD_CAP > 0:
            gn = g.flatten(1).norm(dim=1)
            g = g * at_least_ndim(
                torch.clamp(self.REWARD_GRAD_CAP / (gn + 1e-9), max=1.0), g.dim())
        return g.detach()

    def _reward_loss(self, x0_hat, cond_raw=None):
        """RACTD's ``L_Reward = −R(x̂₀)`` (Eq. 8) as a linear surrogate ``−⟨x̂₀, ∇r⟩``.

        The surrogate has the *same gradient* as ``−R(x̂₀)`` w.r.t. the network parameters —
        ``∂/∂θ [−⟨x̂₀, sg(∇r)⟩] = −∇r·∂x̂₀/∂θ`` — while requiring only a ``∇r`` **callback**
        rather than a torch-differentiable reward. That is what lets the analytic NumPy
        rewards in ``toys/`` drive this loss at all; it is the same trick ``_energy_tilt``
        uses for ``∇E``. Its *value* is not the reward and is not comparable across σ — read
        ``reward_mean`` from the measure loop, not this scalar.

        Args:
            x0_hat: Differentiable one-/few-step sample ``x̂₀`` (from ``_reward_x0_hat``).
            cond_raw: Raw condition slice for the callback, or ``None``.
        """
        g = self._reward_grad(x0_hat.detach(), cond_raw)
        return -(x0_hat * g).flatten(1).sum(1).mean()

    def _jump_to_data(self, model, xt, t, condition_vec, instantaneous: bool = False):
        """One jump straight to data from a **per-sample** noise level ``t`` (span ``= t``):
        the model's ``x̂₀`` estimate at ``t``, i.e. ``G_θ(x_t, t, 0)``. Backbone-specific only
        in the span kwarg (shortcut ``d=t`` / meanflow ``r=0``); returns the velocity, the
        caller integrates. ``instantaneous=True`` queries the ``d=0``/``r=t`` field instead
        (span 0) — the base of the ladder, used by the ``fm_loss`` placement."""
        raise NotImplementedError

    def _reward_x0_hat_anyt(self, x0, condition_cfg=None, instantaneous: bool = False):
        """``x̂₀`` from a **sampled** noise level rather than from pure noise — the ``anyt``
        placement.

        ``_reward_x0_hat`` rewards ``G_θ(x_T, T, 0)``, one point on the jump ladder (span 1),
        which is why its effect is confined to the budgets it is trained at. This instead
        draws ``t ~ U[0,1]`` and rewards ``G_θ(x_t, t, 0)`` — span ``t``, so over training it
        touches **every** span at **one** forward pass, where covering the ladder by rollout
        costs ``E[N]``. It is the reward-side analogue of how the DSM branch covers all ``t``,
        and it is still a legitimate RACTD port: CTM's ``G_θ(x_t, t, 0)`` is defined for any
        ``t``; RACTD simply evaluates it at ``t = T``.

        Args:
            x0: Clean batch — supplies the shape, the diffusion endpoint and the fix_mask prior.
            condition_cfg: Condition input, or ``None``.
        """
        t = torch.rand((x0.shape[0],), device=self.device)
        eps = torch.randn_like(x0)
        xt = x0 + at_least_ndim(t, x0.dim()) * (eps - x0)
        xt = xt * (1.0 - self.fix_mask) + x0 * self.fix_mask
        cvec = (self.model["condition"](condition_cfg)
                if (condition_cfg is not None and self._has_condition) else None)
        vel = self._jump_to_data(self.model, xt, t, cvec, instantaneous=instantaneous)
        x0_hat = xt + at_least_ndim(t, xt.dim()) * vel
        return x0_hat * (1.0 - self.fix_mask) + x0 * self.fix_mask

    def reward_loss_total(self, x0, condition_cfg=None, cond_raw=None):
        """The placement-resolved reward loss — the single entry point both backbones call.

        ``"both"`` genuinely means *two* terms (instantaneous x₀-prediction **plus** the
        rollout), which is why this exists rather than a lone ``_reward_x0_hat``: that
        returns one tensor, so ``"both"`` silently degenerated to ``"sc"`` and its ablation
        row came out bit-identical to the ``sc`` row.

        Args:
            x0: Clean batch. condition_cfg: condition input or ``None``.
            cond_raw: Raw condition slice for the ∇r callback, or ``None``.
        """
        if self.reward_placement == "both":
            return (self._reward_loss(self._reward_x0_hat_anyt(x0, condition_cfg,
                                                               instantaneous=True), cond_raw)
                    + self._reward_loss(self._reward_x0_hat(x0, condition_cfg), cond_raw))
        return self._reward_loss(self._reward_x0_hat(x0, condition_cfg), cond_raw)

    def _reward_x0_hat(self, x0, condition_cfg=None):
        """Differentiable ``x̂₀`` for the reward loss — RACTD's ``G_θ(x_T, T, 0)``.

        Routed through ``self.sample`` (not a hand-rolled rollout) so the reward gradient
        lands on **exactly** the map used at inference, including the clean-inject and the
        solver's step schedule; ``use_ema=False`` because the term must train the live
        weights. The step budget is drawn from ``reward_nfes`` per call, which is what keeps
        the net usable across the N axis instead of being tuned to one budget.

        Args:
            x0: Clean batch — supplies the shape and the ``fix_mask`` prior (pinned entries
                keep their data values, as at inference).
            condition_cfg: Condition input forwarded to ``sample``, or ``None``.
        """
        if self.reward_placement in ("anyt", "fm_loss"):  # "both" handled above
            return self._reward_x0_hat_anyt(
                x0, condition_cfg, instantaneous=(self.reward_placement == "fm_loss"))
        i = int(torch.randint(len(self.reward_nfes), (1,)).item())
        x0_hat, _ = self.sample(
            prior=x0, x1=torch.randn_like(x0), sample_steps=int(self.reward_nfes[i]),
            use_ema=False, requires_grad=True, condition_cfg=condition_cfg,
            w_cfg=0.0, preserve_history=False,
        )
        return x0_hat

    def _null_cond_vec(self, model, n: int):
        """The label-dropout null token (zeroed condition embedding, ``(n, emb)``) for a
        *conditional* model, or ``None`` for an unconditional one. This — not
        ``condition=None``, which skips the DiT's ``cond_proj`` and is an input a
        conditional net never sees in training — is the correct unconditional query
        (2026-07-14_run_review F1; matches label dropout and the ``concat_zeros`` blend).

        Args:
            model: The ``ModuleDict`` being sampled (``self.model`` or ``self.model_ema``).
            n: Batch size of the returned embedding.
        """
        if not getattr(self, "_has_condition", False):
            return None
        return null_cond_emb(model["diffusion"], n, self.device)

    def _encode_condition(self, condition):
        """Encode the condition, and (under ``cfg_uncond_anchor``) report which rows survived
        label dropout.

        The condition module already accepts an explicit ``mask=``; the historical call passes
        none, so the module rolls its own Bernoulli mask internally and returns only the masked
        embedding — the loss can never tell which rows were blanked, and gives them the same
        w-dependent CFG target as conditional rows. Rolling the mask here and handing it in is
        what makes that distinction available to :meth:`_cfg_tilt`.

        Args:
            condition: The raw condition batch, or ``None``.

        Returns:
            ``(cond_emb, keep)`` — the encoded condition and a ``(B,)`` float vector that is 1.0
            on rows whose label survived and 0.0 on blanked rows, or ``None`` when the
            distinction is unavailable or unused (no condition, eval mode, zero dropout, a
            ``TensorDict`` condition, or the anchor disabled). With the anchor off this is the
            historical single-argument call, so the RNG stream and the weights are untouched.
        """
        if condition is None or not self._has_condition:
            return None, None
        module = self.model["condition"]
        if not self.cfg_uncond_anchor:
            return module(condition), None
        prob = float(getattr(module, "dropout", 0.0) or 0.0) if module.training else 0.0
        if prob <= 0.0 or not isinstance(condition, torch.Tensor):
            return module(condition), None
        mask = get_mask(condition, prob, dims=0)
        return module(condition, mask), mask.reshape(mask.shape[0], -1)[:, 0]

    def _cfg_tilt(self, xt, t, v, w, cond_emb, keep=None, **span_kwargs):
        """CFG-tilted regression target ``((1+w)·v − w·v_uncond).detach()`` (iSM/iMF).

        ``v_uncond`` is the EMA net at the **label-dropout null token** (the zeroed
        condition embedding — what dropout actually trains as the unconditional branch;
        F1 of 2026-07-14_run_review, superseding the old ``cond=None`` query, which the
        net never sees in training) and ``w=0``, with the subclass's span kwarg
        (shortcut ``d=0`` / meanflow ``r=t``) — the training-time definition of the
        unguided reference field (also the accessor convention).

        Args:
            xt, t, v: Noised batch, times, and per-sample data velocity (regression base).
            w: Per-sample guidance scales ``(B,)``.
            cond_emb: The encoded condition batch (non-``None`` in every guided branch);
                only its shape/device seed the zeroed null token.
            keep: Optional ``(B,)`` label-dropout survival mask from :meth:`_encode_condition`.
                On blanked rows the condition *is* null, so ``v_cond == v_uncond`` and the
                correct target is ``v_uncond`` for **every** ``w`` — the historical expression is
                unbiased for them but scales the data noise by ``(1+w)²`` around a value that
                does not move, and at ``w=−1`` contains no data at all. Given the mask, those
                rows get the plain data velocity instead: exactly ``q0``'s target, which is the
                only route by which data can reach the unconditional field (no sample carrying a
                condition ``c`` is an unbiased draw of it, since ``E[v|x_t,c] = v_cond``).
            **span_kwargs: The subclass's span input (``d=`` / ``r=``).
        """
        with torch.no_grad():
            v_uncond = self.model_ema["diffusion"](xt, t, torch.zeros_like(cond_emb),
                                                   w=torch.zeros_like(t), **span_kwargs)
        ww = at_least_ndim(w, v.dim())
        target = (1.0 + ww) * v - ww * v_uncond
        if keep is not None:
            k = at_least_ndim(keep, v.dim())
            target = k * target + (1.0 - k) * v
        return target.detach()

    def _anchor_uncond_cond(self, model, cond_vec, w, batch_size):
        """Under ``cfg_uncond_anchor``, replace the condition with the null token at
        ``w = CFG_UNCOND_W``, where the CFG blend is *exactly* the unconditional field.

        The flow maps take ``w`` as a network input, so without this they must *learn* an
        identity that is arithmetically known — and they learn it from a target built out of
        their own EMA, which is the drift measured as the amortization tax. The FM anchor
        computes it instead (``backbones._anchor_span_velocity``: ``wc == 0.0 ⇒ v_uncond``) and
        pays no tax. Returns ``cond_vec`` unchanged whenever the anchor is off or ``w`` is any
        other scale, so guided sampling is untouched.

        Args:
            model: The ``model``/``model_ema`` dict being sampled from. cond_vec: The encoded
                condition (already non-``None`` at the call sites). w: This step's scale.
                batch_size: Row count for the null token.
        """
        if self.cfg_uncond_anchor and float(w) == self.CFG_UNCOND_W:
            return self._null_cond_vec(model, batch_size)
        return cond_vec

    # ==================== shared sampling scaffold ====================

    def _default_w_cfg(self) -> float:
        """Class default for the legacy post-hoc ``w_cfg`` blend (shortcut ``0.0``,
        meanflow ``1.0`` — the one deliberate API divergence, kept for back-compat)."""
        raise NotImplementedError

    def _step_velocity(self, model, xt, t, t_curr, t_next, condition_vec_cfg, w, w_cfg):
        """Per-step (average) velocity for the jump ``t_curr → t_next`` — the only
        backbone-specific part of sampling (span kwarg, guided single-forward vs the
        legacy ``w_cfg`` double-batch blend). Called inside the grad-mode context."""
        raise NotImplementedError

    def sample(
        self,
        prior: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        solver: Optional[str] = None,
        sample_steps: int = 1,
        sampling_schedule: str = "linear",
        sampling_schedule_params: Optional[dict] = None,
        use_ema: bool = True,
        temperature: float = 1.0,
        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        mask_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cfg: Optional[float] = None,
        condition_cg: None = None,
        w_cg: float = 0.0,
        w: float = 0.0,
        diffusion_x_sampling_steps: int = 0,
        warm_start_reference: Optional[torch.Tensor] = None,
        warm_start_forward_level: float = 0.3,
        requires_grad: bool = False,
        preserve_history: bool = False,
        **kwargs,
    ):
        """Euler sampling of the flow map from ``t = 1`` (noise) to ``t ≈ 0`` (data).

        The schedule partitions ``[t_min, 1]`` into ``sample_steps`` intervals; each step
        calls ``_step_velocity`` once and advances ``x ← x + (t_curr − t_next)·vel`` with
        per-step clean-inject of the ``fix_mask``-pinned entries.

        Args (beyond the shared CleanDiffuser conventions):
            solver: Must be in the subclass's ``supported_solvers``; ``None`` ⇒ its default.
            w: Intrinsic-guidance scale (guided models: one guided forward per step).
            w_cfg: Legacy post-hoc CFG blend for unguided-conditional models (``None`` ⇒
                per-class default) — the documented compounding baseline, not the
                recommended guidance path.
            diffusion_x_sampling_steps: Accepted for API parity with the SDE samplers but
                a no-op for flow maps (no "diffusion-x" warmup); logs a WARNING if set.
            warm_start_reference / warm_start_forward_level: MPC-style re-planning start —
                mix the previous plan with noise at the given level, integrate from there.
                ``level`` is the **noise** weight in this module's ``t = 1`` noise convention:
                ``x = level * eps + (1 - level) * reference``, integrated from ``t = level``.
                STP (arXiv:2607.09336 §A.3) writes the same mixture as ``t = 0.3`` in the
                opposite convention, so its 0.7-noise setting is ``level = 0.7`` here.  ``eps``
                is the caller's ``x1`` when one is given, so an on/off contrast is paired.
                ``sample_steps`` is **not** reduced: the same number of steps is compressed into
                ``[0, level]``, which is STP's protocol.  Pass a lower ``sample_steps`` as well
                for Diffuser's NFE-reducing variant (arXiv:2205.09991 §5.4).
        """
        solver = solver or self.supported_solvers[0]
        assert solver in self.supported_solvers, f"Solver {solver} is not supported."
        assert w_cg == 0.0 and condition_cg is None, (
            f"{type(self).__name__} does not support classifier-guidance."
        )
        if w_cfg is None:
            w_cfg = self._default_w_cfg()
        if diffusion_x_sampling_steps:
            log.warning("%s ignores diffusion_x_sampling_steps=%d (flow maps have no "
                        "diffusion-x warmup)", type(self).__name__, diffusion_x_sampling_steps)

        n_samples = prior.shape[0]
        sample_log = {"sample_history": []}
        model = self.model if not use_ema else self.model_ema
        sampling_schedule_params = sampling_schedule_params or {}

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and 0.0 < warm_start_forward_level < 1.0:
            warm_start_reference = warm_start_reference.to(self.device)
            assert prior.shape == warm_start_reference.shape, (
                "prior and warm_start_reference must have the same shape")
            # Consume the CALLER's noise when it supplies some. Drawing from the global torch
            # stream here would make a warm-start contrast partly a noise contrast, and would
            # silently ignore an explicitly passed x1; callers that key their rollout noise by
            # (episode, step) rely on the mixture using exactly the draw they handed in.
            eps = torch.randn_like(prior) if x1 is None else x1.to(self.device)
            assert prior.shape == eps.shape, "prior and x1 must have the same shape"
            start_t = float(warm_start_forward_level)
            x1 = eps * start_t + warm_start_reference * (1.0 - start_t)
        else:
            if x1 is None:
                x1 = torch.randn_like(prior) * temperature
            else:
                assert prior.shape == x1.shape, "prior and x1 must have the same shape"
            start_t = 1.0

        xt = x1 * (1.0 - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            sample_log["sample_history"].append(xt.cpu().numpy())

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = (
                model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            )

        sampling_scheduler = get_sampling_scheduler(sampling_schedule, **sampling_schedule_params)
        t_schedule = sampling_scheduler(sample_steps, device=self.device,
                                        **sampling_schedule_params)
        if start_t != 1.0:
            t_schedule = t_schedule * start_t

        for i in reversed(range(1, sample_steps + 1)):
            t_curr = float(t_schedule[i].item())
            t_next = float(t_schedule[i - 1].item())
            t = torch.full((n_samples,), t_curr, dtype=torch.float32, device=self.device)

            with torch.set_grad_enabled(requires_grad):
                vel = self._step_velocity(model, xt, t, t_curr, t_next,
                                          condition_vec_cfg, w, w_cfg)

            # Euler-like step toward data (decreasing t) + clean-inject
            xt = xt + (t_curr - t_next) * vel
            xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                sample_log["sample_history"].append(xt.cpu().numpy())

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        sample_log["t_schedule"] = t_schedule
        sample_log["start_t"] = start_t
        return xt, sample_log
