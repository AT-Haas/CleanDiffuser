"""Prior Guidance (Ki et al. 2025, *Prior-Guided Diffusion Planning for Offline RL*, arXiv:2505.10881).

Instead of guiding the reverse/denoising process, Prior Guidance (PG) **replaces the standard
Gaussian prior over the initial noise** ``x_T ~ N(0, I)`` of a frozen behavior-cloned planner with a
*learnable* distribution ``p_ψ(x_T | s)`` optimized so that decoding from it yields high-value plans,
under a behavior-regularization toward ``N(0, I)``. The denoiser ``g`` stays frozen; PG only learns
*where in noise space to start*, so inference is a single draw + one decode (no per-step guidance, no
candidate resampling).

Three pieces live here, all framework-agnostic (they take a frozen decoder and a value callable, never
touching the flow internals):

* :class:`LearnableNoisePrior` — a diagonal Gaussian (``n_components=1``) or diagonal Gaussian mixture
  (``n_components>1``, natural for a multimodal target) over the initial noise, with reparameterized
  sampling and a KL-to-``N(0,I)`` term. **Unconditional**: the right family wherever there is no state
  to condition on (the analytic toys), and the ablation baseline elsewhere.
* :class:`ConditionalNoisePrior` — the paper's actual prior ``p_ψ(x_T | s)``: a GRU over the plan
  horizon emitting a per-step ``(mean, log_std)``, conditioned on the current state. Use this wherever
  a state exists; :class:`LearnableNoisePrior` cannot express "for *this* state, shift noise *that*
  way", which is the entire mechanism of the paper.
* :class:`LatentValue` — the latent value ``V̄_φ(x_T) ≈ V(g(x_T))`` (paper Eq. 4), MSE-regressed so the
  prior update (paper Eq. 5) never back-propagates through the frozen denoiser ``g``. Conditional fits
  feed it ``concat([x_T, s])``, matching the reference, whose latent critic is state-conditioned too.

:func:`fit_prior_guidance` runs the alternating fit for either prior; :func:`sample_with_prior` is the
generic inference hook reused by downstream experiments.
"""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG2PI = math.log(2.0 * math.pi)


def std_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    """Log-density of the standard normal ``N(0, I)`` evaluated row-wise.

    Args:
        z: Points ``(n, dim)``.

    Returns:
        Per-row log-density ``(n,)``.
    """
    dim = z.shape[-1]
    return -0.5 * ((z ** 2).sum(-1) + dim * _LOG2PI)


class LearnableNoisePrior(nn.Module):
    """Global learnable distribution over the initial noise (PG, arXiv:2505.10881, §3).

    Parameterizes either a single diagonal Gaussian (``n_components == 1``) or a ``K``-component
    diagonal-Gaussian mixture over the (flattened) initial noise. Sampling is reparameterized
    (pathwise-differentiable in the component means/log-stds); for a mixture the component index is a
    hard categorical draw, so the mixture ``logits`` receive gradient only through :meth:`log_prob`
    (i.e. through the KL term), which is the intended behavior.

    **This is the unconditional arm.** On an unconditional target (the analytic toys) it *is* the
    paper's family — ``p_ψ(x_T|s) ≡ p_ψ(x_T)`` when there is no ``s`` — and being a free parameter
    table it is strictly more expressive there than a net emitting the same diagonal Gaussian. Where a
    state does exist it is a deliberate ablation of :class:`ConditionalNoisePrior`, isolating what the
    reference's observation conditioning buys; it cannot express a per-state shift, so conditioning
    reaches it only through whatever the decoder inpaints (see ``planning/2026-07-07_code_review.md``
    N7).

    Args:
        dim: Dimensionality of the (flattened) initial-noise vector (E1 ring toy: ``2``).
        n_components: Number of mixture components ``K`` (``1`` ⇒ a single diagonal Gaussian).
        init_std: Initial per-dim standard deviation (log-std initialized to ``log(init_std)``).
        tanh_squash: If set, the component **means** are squashed as ``mean_scale·tanh(μ)``, bounding
            the shift while leaving the sample density Gaussian given the mean, so no
            change-of-variables Jacobian is needed. Note this is *ours*, not the reference's: PG
            squashes the drawn **sample** (``pg.py:148``, ``tanh(z)·prior_squash_mean``) and takes the
            KL on the pre-squash distribution — :class:`ConditionalNoisePrior` mirrors that instead.
        mean_scale: Squash amplitude used only when ``tanh_squash`` is set.
        logstd_min: Lower clamp on the log-std (numerical floor on the component spread).
        logstd_max: Upper clamp on the log-std (numerical ceiling on the component spread).
    """

    def __init__(self, dim: int, n_components: int = 1, init_std: float = 1.0,
                 tanh_squash: bool = False, mean_scale: float = 3.0,
                 logstd_min: float = -5.0, logstd_max: float = 2.0):
        super().__init__()
        self.dim = int(dim)
        self.n_components = int(n_components)
        self.tanh_squash = bool(tanh_squash)
        self.mean_scale = float(mean_scale)
        self.logstd_min, self.logstd_max = float(logstd_min), float(logstd_max)
        # Spread the initial component means so a mixture does not collapse to one mode at init.
        spread = 0.0 if self.n_components == 1 else 1.0
        self.means = nn.Parameter(torch.randn(self.n_components, self.dim) * spread)
        self.log_stds = nn.Parameter(torch.full((self.n_components, self.dim), math.log(init_std)))
        self.logits = nn.Parameter(torch.zeros(self.n_components))

    def _params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the effective ``(means (K,dim), stds (K,dim), logits (K,))`` after squash/clamp."""
        means = self.mean_scale * torch.tanh(self.means) if self.tanh_squash else self.means
        stds = torch.exp(self.log_stds.clamp(self.logstd_min, self.logstd_max))
        return means, stds, self.logits

    def rsample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Draw ``n`` reparameterized samples ``(n, dim)`` (pathwise-differentiable in means/stds).

        Args:
            n: Number of samples.
            generator: Optional RNG (must live on this module's device) for reproducibility.

        Returns:
            Samples ``(n, dim)``.
        """
        means, stds, logits = self._params()
        dev = means.device
        eps = torch.randn(n, self.dim, generator=generator, device=dev)
        if self.n_components == 1:
            return means[0] + stds[0] * eps
        probs = torch.softmax(logits, dim=0)
        comp = torch.multinomial(probs, n, replacement=True, generator=generator)  # (n,)
        return means[comp] + stds[comp] * eps

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Mixture log-density at ``x`` ``(n, dim)`` → ``(n,)`` (differentiable in all parameters)."""
        means, stds, logits = self._params()
        if self.n_components == 1:
            var = stds[0] ** 2
            return -0.5 * (((x - means[0]) ** 2) / var + torch.log(var) + _LOG2PI).sum(-1)
        diff = x.unsqueeze(1) - means.unsqueeze(0)            # (n, K, dim)
        var = (stds ** 2).unsqueeze(0)                        # (1, K, dim)
        comp_lp = -0.5 * ((diff ** 2) / var + torch.log(var) + _LOG2PI).sum(-1)  # (n, K)
        logw = torch.log_softmax(logits, dim=0).unsqueeze(0)  # (1, K)
        return torch.logsumexp(comp_lp + logw, dim=1)         # (n,)

    def rsample_with_logprob(self, n: int, generator: Optional[torch.Generator] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: reparameterized ``(z, log p_ψ(z))`` sharing one draw (both differentiable)."""
        z = self.rsample(n, generator=generator)
        return z, self.log_prob(z)

    def kl_to_standard_normal(self, n_mc: int = 256, generator: Optional[torch.Generator] = None,
                              estimator: str = "auto") -> torch.Tensor:
        """Behavior-regularization ``KL(p_ψ ‖ N(0, I))`` (scalar; differentiable).

        For a single Gaussian (``K == 1``) the closed form is used; a mixture has no closed-form KL to
        a Gaussian, so a reparameterized Monte-Carlo estimate ``E_q[log p_ψ(z) − log N(0,I)(z)]`` is
        used (works for both forms).

        Args:
            n_mc: MC sample count for the mixture estimator.
            generator: Optional RNG for the MC draw.
            estimator: ``"closed"`` (Gaussian only), ``"mc"``, or ``"auto"`` (closed iff ``K == 1``).

        Returns:
            Scalar KL.
        """
        use_closed = estimator == "closed" or (estimator == "auto" and self.n_components == 1)
        if use_closed:
            assert self.n_components == 1, "closed-form KL only defined for a single Gaussian"
            means, stds, _ = self._params()
            mu, var = means[0], stds[0] ** 2
            return 0.5 * (mu ** 2 + var - 1.0 - torch.log(var)).sum()
        z, logq = self.rsample_with_logprob(n_mc, generator=generator)
        return (logq - std_normal_logprob(z)).mean()


class ConditionalNoisePrior(nn.Module):
    """State-conditioned learnable prior ``p_ψ(x_T | s)`` — the paper's actual prior net (PG §4).

    A port of the reference ``TanhStochasticGRU`` (``ku-dmlab/PG``, ``network.py:504-559``): the
    conditioning vector is projected once, then a GRU is unrolled over the plan horizon emitting a
    per-step ``(mean, log_std)`` pair, giving a diagonal Gaussian over the ``(horizon, obs_dim)``
    initial noise. Three details are load-bearing and are mirrored exactly:

    * **The GRU is fed the projected condition at step 0 and zeros thereafter** — all temporal
      structure flows through the hidden state, not through a repeated input. Feeding the condition
      at every step instead is a different (and easier) model.
    * **The squash is on the drawn sample**, ``tanh(z)·squash_scale``, not on the mean.
    * **The KL is taken on the pre-squash Gaussian.** The reference does the same; it means the
      regularizer measures the shift the net asked for, not the shift that survived the squash.

    Why this class exists at all: the unconditional :class:`LearnableNoisePrior` learns one global
    shift shared across every state, which is the average of the per-state optima rather than any of
    them. Conditioning is the mechanism the paper is about, so the two are kept as separate arms and
    measured against each other rather than one silently standing in for the other.

    Single-Gaussian only (the paper's default; its Table 2 finds the mixture variant marginal and
    domain-dependent), which is also why the KL is always available in closed form.

    Args:
        cond_dim: Conditioning-vector width (maze2d: ``obs_dim`` for the start state, ``+2`` when the
            goal xy is also supplied).
        horizon: Plan length in rows — the number of GRU unroll steps, so the emitted noise is
            ``(horizon, obs_dim)``. All rows are emitted; the decoder's ``fix_mask`` overwrites the
            pinned ones, keeping the flattened shape contract identical to the unconditional arm.
        obs_dim: Per-row noise width.
        hidden: GRU / projection width (reference default 256).
        tanh_squash: Squash the drawn sample (reference default on for every D4RL domain).
        squash_scale: Squash amplitude (reference ``prior_squash_mean``; 2.0 on maze2d).
        logstd_min: Lower clamp on the log-std (reference ``LOG_STD_MIN = -5``).
        logstd_max: Upper clamp on the log-std (reference ``LOG_STD_MAX = 2``).
    """

    def __init__(self, cond_dim: int, horizon: int, obs_dim: int, hidden: int = 256,
                 tanh_squash: bool = True, squash_scale: float = 2.0,
                 logstd_min: float = -5.0, logstd_max: float = 2.0):
        super().__init__()
        self.cond_dim, self.horizon, self.obs_dim = int(cond_dim), int(horizon), int(obs_dim)
        self.dim = self.horizon * self.obs_dim
        self.tanh_squash, self.squash_scale = bool(tanh_squash), float(squash_scale)
        self.logstd_min, self.logstd_max = float(logstd_min), float(logstd_max)
        self.inp = nn.Linear(self.cond_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.mean_head = nn.Linear(hidden, self.obs_dim)
        self.logstd_head = nn.Linear(hidden, self.obs_dim)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Condition ``(B, cond_dim)`` → per-step ``(means, stds)``, each ``(B, horizon, obs_dim)``."""
        x = F.relu(self.ln1(self.inp(cond)))
        h = torch.zeros_like(x)
        zeros = torch.zeros_like(x)
        means, logstds = [], []
        for step in range(self.horizon):
            h = self.cell(x if step == 0 else zeros, h)
            y = F.relu(self.ln2(h))
            means.append(self.mean_head(y))
            logstds.append(self.logstd_head(y))
        mean = torch.stack(means, dim=1)
        logstd = torch.stack(logstds, dim=1).clamp(self.logstd_min, self.logstd_max)
        return mean, torch.exp(logstd)

    def rsample(self, cond: torch.Tensor, generator: Optional[torch.Generator] = None
                ) -> torch.Tensor:
        """Reparameterized draw ``(B, horizon*obs_dim)``, flattened to match the unconditional arm.

        Args:
            cond: Conditioning batch ``(B, cond_dim)`` — one draw per row.
            generator: Optional RNG (must live on this module's device) for reproducibility.
        """
        mean, std = self(cond)
        eps = torch.randn(mean.shape, generator=generator, device=mean.device)
        z = mean + std * eps
        if self.tanh_squash:
            z = torch.tanh(z) * self.squash_scale
        return z.reshape(z.shape[0], self.dim)

    def kl_to_standard_normal(self, cond: torch.Tensor) -> torch.Tensor:
        """Behavior regularization ``E_s[KL(p_ψ(·|s) ‖ N(0, I))]`` (scalar; differentiable).

        Closed form per condition (a single diagonal Gaussian), then averaged over the batch —
        the reference's ``tfd.kl_divergence(dist, std_normal).mean()``. Taken on the **pre-squash**
        distribution, as the reference does.

        Args:
            cond: Conditioning batch ``(B, cond_dim)``.
        """
        mean, std = self(cond)
        var = std ** 2
        return 0.5 * (mean ** 2 + var - 1.0 - torch.log(var)).sum(dim=(1, 2)).mean()


class LatentValue(nn.Module):
    """Latent value ``V̄_φ(x_T) ≈ V(g(x_T))`` (PG Eq. 4): a small MLP regressing the *decoded* value.

    Regressing in noise space lets the prior update (Eq. 5) maximize value via this surrogate without
    differentiating through the frozen decoder ``g``.

    A conditional fit feeds it ``concat([x_T, s])`` and sizes ``dim`` accordingly, matching the
    reference, whose latent critic reads ``concat([obs, x_T])`` too: with a state-conditioned prior the
    decoded value depends on the state, so a surrogate blind to it would be regressing an average.

    Args:
        dim: Input dimensionality — the flattened initial noise, plus ``cond_dim`` when conditional.
        hidden: Hidden width.
        depth: Number of hidden layers.
    """

    def __init__(self, dim: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers, d = [], int(dim)
        for _ in range(int(depth)):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map noise ``(n, dim)`` → scalar value ``(n,)``."""
        return self.net(x).squeeze(-1)


def fit_prior_guidance(
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    value_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    *,
    alpha: float,
    n_components: int = 4,
    rounds: int = 60,
    value_steps: int = 10,
    prior_steps: int = 10,
    batch: int = 512,
    lr: float = 3e-3,
    kl_estimator: str = "auto",
    coverage_mix: float = 0.5,
    tanh_squash: bool = False,
    mean_scale: float = 3.0,
    value_hidden: int = 128,
    value_depth: int = 2,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
    prior_init: Optional[LearnableNoisePrior] = None,
    value_init: Optional[LatentValue] = None,
    cond_sampler: Optional[Callable[[int, Optional[torch.Generator]], torch.Tensor]] = None,
    cond_dim: int = 0,
    cond_prior_kwargs: Optional[dict] = None,
) -> Tuple[nn.Module, LatentValue, dict]:
    """Fit one Prior-Guidance prior for a single ``alpha`` (PG Algorithm 1, alternating Eq. 4 / Eq. 5).

    Alternates (a) regressing the latent value ``V̄_φ`` toward ``value_fn(decode_fn(z))`` on a mix of
    base-``N(0,I)`` and current-prior draws (so the surrogate stays accurate where the prior moves),
    and (b) updating the prior to maximize ``E[V̄_φ(z)] − alpha·KL(p_ψ ‖ N(0,I))`` via reparameterized
    samples (``decode_fn`` is never called in this phase). The decoder is treated as frozen — call it
    under ``no_grad``.

    Passing ``cond_sampler`` switches to the **conditional** fit: the prior becomes a
    :class:`ConditionalNoisePrior`, the latent value reads ``concat([z, cond])``, and both the value
    target and the KL are taken per condition. Leaving it ``None`` takes the unconditional path
    unchanged, byte for byte — the toys and the existing ``prior`` family depend on that.

    Args:
        decode_fn: Frozen decoder (the behavior-cloned flow map at a fixed step count), called under
            ``no_grad``: ``z (B, dim) → x0 (B, *x_shape)`` unconditionally, or ``(z, cond) → x0`` when
            ``cond_sampler`` is given. The conditional form must derive whatever the decoder inpaints
            from ``cond`` itself, so the same closure serves the fit and the closed-loop rollout.
        value_fn: Plan value ``x0 → (B,)``. Oracle on the ``_gt``-style arms (``target.reward`` on the
            toys, the task verifier on maze2d); a learned offline critic on the ``_v`` arms, which is
            what the reference uses and the only arm whose value error PG has to survive.
        dim: Initial-noise dimensionality (flattened; excludes ``cond_dim``).
        alpha: Behavior-regularization coefficient (lower ⇒ stronger value-seeking; paper sweeps
            ``{50, 10, 1, 0.1, 0.01, 0.001}``).
        n_components: Mixture components for the prior (``1`` ⇒ single Gaussian).
        rounds: Number of outer alternations.
        value_steps: Inner value-regression steps per round.
        prior_steps: Inner prior-update steps per round.
        batch: Samples per inner step.
        lr: Adam learning rate (shared by both optimizers).
        kl_estimator: KL estimator passed to :meth:`LearnableNoisePrior.kl_to_standard_normal`.
        coverage_mix: Fraction of the value-regression batch drawn from base ``N(0,I)`` (the rest from
            the current prior), so ``V̄_φ`` covers both the reference and the shifted support. This mix
            is our robustness addition — the reference (``ku-dmlab/PG``) regresses on current-prior
            draws only (2026-07-07_code_review.md N7).
        tanh_squash: Squash the prior means (see :class:`LearnableNoisePrior`).
        mean_scale: Squash amplitude (used iff ``tanh_squash``).
        value_hidden: Latent-value MLP width.
        value_depth: Latent-value MLP depth.
        device: Torch device.
        generator: Optional RNG (kept on ``device``) for reproducibility.
        prior_init: Optional warm-start prior (e.g. reuse across ``alpha`` / NFE); fresh if ``None``.
        value_init: Optional warm-start latent value; fresh if ``None``.
        cond_sampler: ``fn(n, generator) -> cond (n, cond_dim)`` drawing conditioning states for the
            fit. Supplying it selects the conditional fit; ``None`` keeps the unconditional one. Draw
            from *training* states, not the evaluation batch — a conditional prior is the one arm with
            something to generalize, so fitting it on the eval tasks would measure memorization.
        cond_dim: Conditioning width; required (``> 0``) when ``cond_sampler`` is given.
        cond_prior_kwargs: Extra :class:`ConditionalNoisePrior` args (``horizon``, ``obs_dim``,
            ``hidden``, ``tanh_squash``, ``squash_scale``). ``horizon``/``obs_dim`` are required there,
            and their product must equal ``dim``.

    Returns:
        ``(prior, value, history)`` where ``history`` logs per-round mean value / KL / losses.
    """
    conditional = cond_sampler is not None
    if conditional:
        ck = dict(cond_prior_kwargs or {})
        if cond_dim <= 0:
            raise ValueError("a conditional fit needs cond_dim > 0")
        if ck.get("horizon", 0) * ck.get("obs_dim", 0) != dim:
            raise ValueError(f"cond_prior_kwargs horizon*obs_dim must equal dim={dim}, got {ck}")
        prior = prior_init if prior_init is not None else ConditionalNoisePrior(
            cond_dim, **ck).to(device)
    else:
        prior = prior_init if prior_init is not None else LearnableNoisePrior(
            dim, n_components=n_components, tanh_squash=tanh_squash, mean_scale=mean_scale).to(device)
    value = value_init if value_init is not None else LatentValue(
        dim + (cond_dim if conditional else 0), hidden=value_hidden, depth=value_depth).to(device)
    opt_v = torch.optim.Adam(value.parameters(), lr=lr)
    opt_p = torch.optim.Adam(prior.parameters(), lr=lr)
    hist = {"loss_v": [], "loss_p": [], "kl": [], "val": []}
    n0 = int(round(coverage_mix * batch))

    def _v_in(z, cond):
        """Latent-value input: the noise alone, or ``concat([z, cond])`` for a conditional fit."""
        return z if cond is None else torch.cat([z, cond], dim=-1)

    for _ in range(int(rounds)):
        # (a) latent-value regression — no gradient through the frozen decoder g
        for p in value.parameters():
            p.requires_grad_(True)
        lv = 0.0
        for _ in range(int(value_steps)):
            with torch.no_grad():
                cond = cond_sampler(batch, generator) if conditional else None
                z_base = torch.randn(n0, dim, generator=generator, device=device)
                z_pri = (prior.rsample(cond[n0:], generator=generator) if conditional
                         else prior.rsample(batch - n0, generator=generator))
                z = torch.cat([z_base, z_pri], dim=0)
                x0 = decode_fn(z, cond) if conditional else decode_fn(z)
                v_tgt = value_fn(x0).reshape(-1).to(device).float()
            loss_v = F.mse_loss(value(_v_in(z, cond)), v_tgt)
            opt_v.zero_grad(); loss_v.backward(); opt_v.step()
            lv = float(loss_v)

        # (b) prior update — reparameterized; freeze the value net so only ψ moves
        for p in value.parameters():
            p.requires_grad_(False)
        lp = kl_v = val_v = 0.0
        for _ in range(int(prior_steps)):
            if conditional:
                cond = cond_sampler(batch, generator)
                z = prior.rsample(cond, generator=generator)
                kl = prior.kl_to_standard_normal(cond)
            else:
                cond = None
                z, logq = prior.rsample_with_logprob(batch, generator=generator)
                if kl_estimator == "closed" or (kl_estimator == "auto" and prior.n_components == 1):
                    kl = prior.kl_to_standard_normal(estimator="closed")
                else:
                    kl = (logq - std_normal_logprob(z)).mean()
            val = value(_v_in(z, cond))
            loss_p = -(val.mean() - float(alpha) * kl)
            opt_p.zero_grad(); loss_p.backward(); opt_p.step()
            lp, kl_v, val_v = float(loss_p), float(kl), float(val.mean())
        hist["loss_v"].append(lv); hist["loss_p"].append(lp)
        hist["kl"].append(kl_v); hist["val"].append(val_v)

    for p in value.parameters():
        p.requires_grad_(True)
    prior.eval()
    return prior, value, hist


@torch.no_grad()
def sample_with_prior(flow, prior_model: LearnableNoisePrior, sample_kwargs: dict, n: int, *,
                      x_shape: Tuple[int, ...] = (1, 2), prior_inpaint: Optional[torch.Tensor] = None,
                      generator: Optional[torch.Generator] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Generic PG inference: draw ``x1 ~ p_ψ`` and decode once with the frozen ``flow``.

    Reused by downstream experiments (the E1 driver inlines the equivalent call so it can route the
    per-step metric suite). Any goal/state inpainting is handled by the flow's own ``fix_mask`` plus
    ``prior_inpaint`` — the learnable prior then effectively governs the free noise dimensions.

    **Unconditional arm only.** A :class:`ConditionalNoisePrior` needs its condition at draw time and
    its inpainting derived from that same condition, so the drivers call it directly rather than
    through here.

    Args:
        flow: A frozen CleanDiffuser flow exposing ``sample(prior=, x1=, **sample_kwargs)``.
        prior_model: The fitted :class:`LearnableNoisePrior`.
        sample_kwargs: Extra kwargs forwarded to ``flow.sample`` (``solver``, ``sample_steps``, ...).
        n: Number of plans to sample.
        x_shape: Per-sample tensor shape the noise is reshaped to (E1: ``(1, 2)``).
        prior_inpaint: Inpainting tensor ``(n, *x_shape)`` for the fixed dims; zeros if ``None``.
        generator: Optional RNG for the prior draw.

    Returns:
        ``(x0, x1, log)``: decoded plans, the drawn initial noise, and the sampler log.
    """
    dev = next(prior_model.parameters()).device
    x1 = prior_model.rsample(n, generator=generator).reshape(n, *x_shape)
    prior_t = prior_inpaint if prior_inpaint is not None else torch.zeros((n, *x_shape), device=dev)
    x0, log = flow.sample(prior=prior_t, x1=x1, **sample_kwargs)
    return x0, x1, log
