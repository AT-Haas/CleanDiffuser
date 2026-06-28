"""Prior Guidance (Ki et al. 2025, *Prior-Guided Diffusion Planning for Offline RL*, arXiv:2505.10881).

Instead of guiding the reverse/denoising process, Prior Guidance (PG) **replaces the standard
Gaussian prior over the initial noise** ``x_T ~ N(0, I)`` of a frozen behavior-cloned planner with a
*learnable* distribution ``p_ψ(x_T | s)`` optimized so that decoding from it yields high-value plans,
under a behavior-regularization toward ``N(0, I)``. The denoiser ``g`` stays frozen; PG only learns
*where in noise space to start*, so inference is a single draw + one decode (no per-step guidance, no
candidate resampling).

Two pieces live here, both framework-agnostic (they take a frozen decoder and a value callable, never
touching the flow internals):

* :class:`LearnableNoisePrior` — a diagonal Gaussian (``n_components=1``) or diagonal Gaussian mixture
  (``n_components>1``, natural for a multimodal target) over the initial noise, with reparameterized
  sampling and a KL-to-``N(0,I)`` term.
* :class:`LatentValue` — the latent value ``V̄_φ(x_T) ≈ V(g(x_T))`` (paper Eq. 4), MSE-regressed so the
  prior update (paper Eq. 5) never back-propagates through the frozen denoiser ``g``.

:func:`fit_prior_guidance` runs the alternating fit; :func:`sample_with_prior` is the generic
inference hook reused by downstream experiments.
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
    (i.e. through the KL term), which is the intended behavior. ``E1`` is unconditional, so this is a
    *global* prior (no state conditioning); the D4RL extension conditions an analogous net on the
    observation.

    Args:
        dim: Dimensionality of the (flattened) initial-noise vector (E1 ring toy: ``2``).
        n_components: Number of mixture components ``K`` (``1`` ⇒ a single diagonal Gaussian).
        init_std: Initial per-dim standard deviation (log-std initialized to ``log(init_std)``).
        tanh_squash: If set, the component **means** are squashed as ``mean_scale·tanh(μ)`` (bounds the
            shift, matching the repo's "tanh-squash on the mean"); the sample density given the mean is
            still Gaussian, so no change-of-variables Jacobian is needed.
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


class LatentValue(nn.Module):
    """Latent value ``V̄_φ(x_T) ≈ V(g(x_T))`` (PG Eq. 4): a small MLP regressing the *decoded* value.

    Regressing in noise space lets the prior update (Eq. 5) maximize value via this surrogate without
    differentiating through the frozen decoder ``g``.

    Args:
        dim: Input (initial-noise) dimensionality.
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
) -> Tuple[LearnableNoisePrior, LatentValue, dict]:
    """Fit one Prior-Guidance prior for a single ``alpha`` (PG Algorithm 1, alternating Eq. 4 / Eq. 5).

    Alternates (a) regressing the latent value ``V̄_φ`` toward ``value_fn(decode_fn(z))`` on a mix of
    base-``N(0,I)`` and current-prior draws (so the surrogate stays accurate where the prior moves),
    and (b) updating the prior to maximize ``E[V̄_φ(z)] − alpha·KL(p_ψ ‖ N(0,I))`` via reparameterized
    samples (``decode_fn`` is never called in this phase). The decoder is treated as frozen — call it
    under ``no_grad``.

    Args:
        decode_fn: Frozen decoder ``z (B, dim) → x0 (B, *x_shape)`` (the behavior-cloned flow map at a
            fixed step count). Should not require grad.
        value_fn: Plan value ``x0 → (B,)`` (E1: ``target.reward``; D4RL: a learned critic).
        dim: Initial-noise dimensionality.
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
            the current prior), so ``V̄_φ`` covers both the reference and the shifted support.
        tanh_squash: Squash the prior means (see :class:`LearnableNoisePrior`).
        mean_scale: Squash amplitude (used iff ``tanh_squash``).
        value_hidden: Latent-value MLP width.
        value_depth: Latent-value MLP depth.
        device: Torch device.
        generator: Optional RNG (kept on ``device``) for reproducibility.
        prior_init: Optional warm-start prior (e.g. reuse across ``alpha`` / NFE); fresh if ``None``.
        value_init: Optional warm-start latent value; fresh if ``None``.

    Returns:
        ``(prior, value, history)`` where ``history`` logs per-round mean value / KL / losses.
    """
    prior = prior_init if prior_init is not None else LearnableNoisePrior(
        dim, n_components=n_components, tanh_squash=tanh_squash, mean_scale=mean_scale).to(device)
    value = value_init if value_init is not None else LatentValue(
        dim, hidden=value_hidden, depth=value_depth).to(device)
    opt_v = torch.optim.Adam(value.parameters(), lr=lr)
    opt_p = torch.optim.Adam(prior.parameters(), lr=lr)
    hist = {"loss_v": [], "loss_p": [], "kl": [], "val": []}
    n0 = int(round(coverage_mix * batch))

    for _ in range(int(rounds)):
        # (a) latent-value regression — no gradient through the frozen decoder g
        for p in value.parameters():
            p.requires_grad_(True)
        lv = 0.0
        for _ in range(int(value_steps)):
            with torch.no_grad():
                z_base = torch.randn(n0, dim, generator=generator, device=device)
                z_pri = prior.rsample(batch - n0, generator=generator)
                z = torch.cat([z_base, z_pri], dim=0)
                v_tgt = value_fn(decode_fn(z)).reshape(-1).to(device).float()
            loss_v = F.mse_loss(value(z), v_tgt)
            opt_v.zero_grad(); loss_v.backward(); opt_v.step()
            lv = float(loss_v)

        # (b) prior update — reparameterized; freeze the value net so only ψ moves
        for p in value.parameters():
            p.requires_grad_(False)
        lp = kl_v = val_v = 0.0
        for _ in range(int(prior_steps)):
            z, logq = prior.rsample_with_logprob(batch, generator=generator)
            val = value(z)
            if kl_estimator == "closed" or (kl_estimator == "auto" and prior.n_components == 1):
                kl = prior.kl_to_standard_normal(estimator="closed")
            else:
                kl = (logq - std_normal_logprob(z)).mean()
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
