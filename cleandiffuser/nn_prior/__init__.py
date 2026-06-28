"""Learnable noise priors — Prior Guidance (Ki et al. 2025, arXiv:2505.10881).

A frozen behavior-cloned planner keeps the standard ``N(0, I)`` prior over its initial noise; Prior
Guidance instead learns *where in noise space to start* so a single decode yields high-value plans.
This package holds the framework-agnostic pieces (a learnable prior, a latent value surrogate, the
alternating fit, and a generic sampling hook); experiment drivers supply the frozen decoder and the
value callable.
"""

from .learnable_prior import (
    LearnableNoisePrior,
    LatentValue,
    fit_prior_guidance,
    sample_with_prior,
    std_normal_logprob,
)

__all__ = [
    "LearnableNoisePrior",
    "LatentValue",
    "fit_prior_guidance",
    "sample_with_prior",
    "std_normal_logprob",
]
