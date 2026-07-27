"""Best-of-N (MCSS) candidate selection for diffusion/flow planners.

The classic inference-time improvement: draw ``N`` candidate plans from a frozen planner, score each,
and keep the best. This is the Monte-Carlo Sample Selection (MCSS) baseline that Prior Guidance
(Ki et al. 2025, arXiv:2505.10881) is designed to beat — useful to plot side-by-side. Its honest cost
is ``N × base_steps`` function evaluations per decision.

Both helpers are framework-agnostic: :func:`best_of_n_sample` takes a frozen ``decode_fn`` and a
pluggable ``score_fn`` (an oracle reward, or :func:`fit_value_net`'s learned critic), so the same code
serves the E1 toy and the D4RL path.
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from cleandiffuser.nn_prior import LatentValue


def best_of_n_sample(
    decode_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    n_candidates: int,
    prior: torch.Tensor,
    *,
    noise_std: float = 1.0,
    generator: Optional[torch.Generator] = None,
    return_x1: bool = False,
    return_candidates: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """Draw ``n_candidates`` fresh-noise plans per point and keep the highest-scoring one (MCSS).

    All candidates for all points are decoded in **one** batched call (the eval set is expanded by
    ``n_candidates`` along the batch), so the extra cost is purely the ``N×`` larger forward — charged
    honestly by the caller as ``N × base_steps`` NFE.

    Args:
        decode_fn: Frozen decoder ``(prior_exp (M,*xs), x1_exp (M,*xs)) → x0 (M,*xs)`` (one
            ``flow.sample`` under the hood); ``M = n_points × n_candidates``.
        score_fn: Plan scorer ``x0 (M,*xs) → (M,)`` (E1 oracle: ``target.reward``; or a learned
            critic from :func:`fit_value_net`). Higher is better.
        n_candidates: Number of candidates ``N`` drawn per evaluation point.
        prior: Inpainting tensor ``(n_points, *x_shape)`` for the fixed dims (zeros ⇒ no inpainting).
        noise_std: Std of the fresh per-candidate initial noise (temperature).
        generator: Optional RNG (on ``prior``'s device) for reproducible candidate noise.
        return_x1: Also return the winning initial noise ``(n_points, *x_shape)`` (lets the caller
            re-decode the winner *with* history for per-step metrics).
        return_candidates: Also return all decoded candidates ``(n_points, N, *x_shape)`` (for a
            candidate-cloud visualization).

    Returns:
        ``(winners (n_points, *x_shape), info)`` where ``info`` always has ``idx`` (winning candidate
        index per point) and ``scores`` ``(n_points, N)``, plus ``x1`` / ``candidates`` if requested.
    """
    n_points = prior.shape[0]
    shape = tuple(prior.shape[1:])
    dev = prior.device
    K = int(n_candidates)

    prior_exp = prior.repeat_interleave(K, dim=0)                                   # (n_points*K, *s)
    x1 = torch.randn((n_points * K, *shape), generator=generator, device=dev) * noise_std
    x0 = decode_fn(prior_exp, x1)                                                   # (n_points*K, *s)

    scores = score_fn(x0).reshape(n_points, K)                                      # (n_points, K)
    idx = scores.argmax(dim=1)                                                      # (n_points,)
    ar = torch.arange(n_points, device=dev)
    x0v = x0.reshape(n_points, K, *shape)
    winners = x0v[ar, idx]                                                          # (n_points, *s)

    info = {"idx": idx, "scores": scores}
    if return_x1:
        info["x1"] = x1.reshape(n_points, K, *shape)[ar, idx]
    if return_candidates:
        info["candidates"] = x0v
    return winners, info


def fit_value_net(
    x0: torch.Tensor,
    value: torch.Tensor,
    *,
    dim: int = 2,
    hidden: int = 128,
    depth: int = 2,
    steps: int = 1500,
    batch: int = 256,
    lr: float = 3e-3,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
    net: Optional[LatentValue] = None,
) -> LatentValue:
    """Fit a small data-space critic ``V̂(x0) ≈ value`` on offline ``(x0, value)`` pairs.

    The realistic best-of-N score: a critic trained only on behavior samples and their values (never
    the true reward function), so selecting by it folds in the critic's generalization error — the gap
    to oracle selection is exactly that error.

    Args:
        x0: Behavior samples ``(M, *x_shape)`` (flattened to ``(M, dim)`` internally).
        value: Their scalar values ``(M,)`` or ``(M, 1)``.
        dim: Flattened sample dimensionality.
        hidden: Critic MLP width.
        depth: Critic MLP depth.
        steps: SGD steps.
        batch: Minibatch size.
        lr: Adam learning rate.
        device: Torch device.
        generator: Optional RNG for minibatch indexing.
        net: Optional pre-built ``LatentValue`` to train in place, instead of constructing one
            here. Mirrors ``fit_cep_energy``'s ``net`` argument and exists for the same reason:
            it lets the fit be checkpointed like any other net (``ractd_v`` needs the critic as
            a DAG node, so the model store must be able to rebuild it for loading). Default
            ``None`` keeps the historical construct-here behaviour, byte-identical for ``boN_v``.

    Returns:
        A trained (eval-mode) :class:`~cleandiffuser.nn_prior.LatentValue` callable ``(B, dim) → (B,)``.
    """
    x = torch.as_tensor(x0, dtype=torch.float32, device=device).reshape(-1, dim)
    y = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
    vhat = (LatentValue(dim, hidden=hidden, depth=depth) if net is None else net).to(device)
    opt = torch.optim.Adam(vhat.parameters(), lr=lr)
    n = x.shape[0]
    vhat.train()
    for _ in range(int(steps)):
        idx = torch.randint(0, n, (batch,), generator=generator, device=device)
        loss = F.mse_loss(vhat(x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    vhat.eval()
    return vhat
