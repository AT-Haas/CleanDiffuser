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
from cleandiffuser.utils import TensorDict, at_least_ndim, get_sampling_scheduler, null_cond_emb

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
    # Per-sample cap on the *correction* the `fm` placement adds to the FM target — the
    # reward-side twin of ENERGY_CORR_CAP, and distinct from the ∇r cap above: the tilt
    # coefficient σ·t/(1−t) is what blows up near the noise end, not the gradient.
    REWARD_CORR_CAP: float = 50.0

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
    ):
        super().__init__(nn_diffusion, nn_condition, fix_mask, loss_weight, classifier,
                         ema_rate, optimizer_params)
        assert classifier is None, f"{type(self).__name__} does not support classifier-guidance."
        # iSM/iMF Intrinsic Guidance: condition on the guidance scale w (inference-time CFG).
        self.guided = guided
        self.w_min = w_min
        self.w_max = w_max
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
        #   reward_placement:  where the reward pressure enters — ``"sc"`` (default; the jump /
        #                      self-consistency branch, = RACTD's reward-aware *student*),
        #                      ``"fm"`` (the instantaneous branch, = a reward-aware *teacher*,
        #                      which is what every other in-weights family in this repo does),
        #                      or ``"both"``. See planning/2026-07-27_ractd_reward_loss_arm.md.
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

    def _jump_to_data(self, model, xt, t, condition_vec):
        """One jump straight to data from a **per-sample** noise level ``t`` (span ``= t``):
        the model's ``x̂₀`` estimate at ``t``, i.e. ``G_θ(x_t, t, 0)``. Backbone-specific only
        in the span kwarg (shortcut ``d=t`` / meanflow ``r=0``); returns the velocity, the
        caller integrates. Used by the ``anyt`` reward placement."""
        raise NotImplementedError

    def _reward_x0_hat_anyt(self, x0, condition_cfg=None):
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
        vel = self._jump_to_data(self.model, xt, t, cvec)
        x0_hat = xt + at_least_ndim(t, xt.dim()) * vel
        return x0_hat * (1.0 - self.fix_mask) + x0 * self.fix_mask

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
        if self.reward_placement == "anyt":
            return self._reward_x0_hat_anyt(x0, condition_cfg)
        i = int(torch.randint(len(self.reward_nfes), (1,)).item())
        x0_hat, _ = self.sample(
            prior=x0, x1=torch.randn_like(x0), sample_steps=int(self.reward_nfes[i]),
            use_ema=False, requires_grad=True, condition_cfg=condition_cfg,
            w_cfg=0.0, preserve_history=False,
        )
        return x0_hat

    def _reward_tilt(self, xt, t, v, cond_raw):
        """Reward-tilted regression target ``(v + cap(σ·[t/(1−t)]·∇r(x_t))).detach()`` — the
        ``fm`` placement arm, structurally identical to ``_energy_tilt`` with ``σ·∇r`` in
        place of ``∇E``.

        **The two are not the same object, and the gap is the point of this arm.** ``∇E`` is
        the exact time-dependent ``∇log p_t − ∇log q_t``; ``∇r`` here is the *clean-sample*
        reward gradient evaluated at the noisy iterate — precisely the noise-unaware reward
        model RACTD §3.4 argues against ("predicting the correct reward from highly corrupted
        input could be very challenging"). On ManyWell the two coincide at ``t=0`` (its tilt
        is pointwise, gated in ``validate_e3``); on the GMM targets they differ even there,
        because those tilt by component reweighting.

        Args:
            xt, t, v: Noised batch, times, and per-sample data velocity (regression base).
            cond_raw: Raw condition slice for the callback, or ``None``.
        """
        gr = self._reward_grad(xt, cond_raw)
        coef = torch.where((1.0 - t) < self.NOISE_GATE, torch.zeros_like(t),
                           self.reward_sigma * t / (1.0 - t))
        corr = at_least_ndim(coef, gr.dim()) * gr
        # Cap the CORRECTION, not just ∇r — the same guard _energy_tilt applies, and it is
        # load-bearing here: t ~ U[0,1] on the FM branch and the gate only fires at
        # 1−t < 1e-3, so coef = σ·t/(1−t) reaches ~500 (p99 ≈ 46 at σ=0.5) and its mean
        # diverges logarithmically. Without this the FM target is dominated by a handful of
        # near-noise samples — measured: loss_fm 1.48 → 27.4, i.e. the arm was fitting a
        # wrecked field, not a tilted one.
        cn = corr.flatten(1).norm(dim=1)
        corr = corr * at_least_ndim(torch.clamp(self.REWARD_CORR_CAP / (cn + 1e-9), max=1.0),
                                    corr.dim())
        return (v + corr).detach()

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

    def _cfg_tilt(self, xt, t, v, w, cond_emb, **span_kwargs):
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
            **span_kwargs: The subclass's span input (``d=`` / ``r=``).
        """
        with torch.no_grad():
            v_uncond = self.model_ema["diffusion"](xt, t, torch.zeros_like(cond_emb),
                                                   w=torch.zeros_like(t), **span_kwargs)
        ww = at_least_ndim(w, v.dim())
        return ((1.0 + ww) * v - ww * v_uncond).detach()

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
            t_c = torch.ones_like(prior) * warm_start_forward_level
            x1 = torch.randn_like(prior) * t_c + warm_start_reference * (1 - t_c)
            start_t = float(warm_start_forward_level)
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
        return xt, sample_log
