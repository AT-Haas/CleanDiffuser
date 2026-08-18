"""Process-global execution switches for the 2026-08 GPU throughput study, all OFF by default.

Every flag here changes *how* a training step executes, never *what* it computes. Only two are
bit-exact (``sync_free_telemetry``, ``foreach_ema``); the rest reassociate reductions or drop
mantissa bits, so they must clear the drift gate before anything adopts them. Defaults are
therefore all off and an unflagged run is byte-identical to the pre-2026-08-18 code, which is
what keeps every regression fixture passing while the levers are being measured.

Why an environment variable and not a config key: the drivers dispatch units into **spawned**
worker processes, so a flag flipped in the parent does not survive into them. That is the same
reason ``maze2d.apply_tf32`` exists and is called inside the worker rather than in ``main``.
The environment *is* inherited by spawn, so ``SP_PERF`` reaches every worker for free, with no
new CLI surface on four drivers for switches that are meant to be temporary::

    SP_PERF=sync_free_telemetry,foreach_ema,fused_sdpa python experiments/maze2d.py ...
    SP_PERF=amp=bf16,fused_adam python experiments/maze2d.py ...

Unknown names raise rather than being ignored: a silently-misspelled perf flag would produce a
"no speedup" measurement that is really a "the flag never turned on" measurement, and that is a
wrong answer rather than a missing one.

This is the measurement harness's channel, not a production API. When a lever is adopted it
should become a real flag with a real default and its entry here should go away.
"""

import contextlib
import os
from typing import Optional

import torch

__all__ = ["PERF", "configure", "reset", "from_env", "describe", "override"]

#: ``amp=<name>`` values accepted by :func:`configure`. ``None`` means autocast is off.
_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None, "none": None}


class _Perf:
    """The switch block itself. One instance, ``PERF``, is the module's public surface.

    Attributes:
        sync_free_telemetry: Accumulate the per-step loss/grad-norm into a device buffer and
            copy it to the host once per progress-log boundary, instead of one ``float()`` per
            value per step. **Bit-exact**: the recorded numbers are the same, only the moment
            they cross the PCIe bus changes. Removes 3 to 5 pipeline drains per step, which is
            what lets the CPU run ahead of the GPU at all.
        foreach_ema: Run the EMA update as two ``torch._foreach_*`` calls over the whole
            parameter list rather than a Python loop issuing two kernels per tensor.
            **Bit-exact** (same ops, same order, per element); collapses ~128 micro-launches
            per step on the maze2d net to 2.
        fused_sdpa: Pass ``need_weights=False`` to ``nn.MultiheadAttention`` so it takes
            ``scaled_dot_product_attention``'s fused path instead of materialising a
            ``(B*H, L, L)`` score matrix, softmaxing it, and averaging it over heads only for
            the caller to discard the average. Mathematically equivalent, not bit-exact.
        fused_branch_forward: In the shortcut loss, evaluate the flow-matching and
            self-consistency student branches in **one** network call over the concatenated
            batch rather than two calls at ``B_fm`` and ``B_sc``. The second is small enough
            (12.5% of the batch by default) that it pays full launch latency for a fraction of
            the work. Equivalent per sample, not bit-exact.
        amp_dtype: Autocast dtype for the loss and backward, or ``None`` for full fp32. Master
            weights stay fp32, and bf16 needs no ``GradScaler`` (it keeps fp32's exponent
            range). Reduced precision.
        fused_adam: Build ``torch.optim.Adam`` with ``fused=True``. Different kernel, so not
            bit-exact even though the arithmetic is nominally the same.
        compile_net: ``torch.compile`` the diffusion network. **Shortcut only.** Measured
            2026-08-18 on torch 2.2.2: MeanFlow raises ``InternalTorchDynamoError: Cannot
            access data pointer of Tensor that doesn't have storage`` because Dynamo traces
            through ``torch.func.jvp``'s dual tensors, which have no storage. That is the same
            forward-AD wall ``fused_sdpa`` hits, but it cannot be narrowed the same way: the
            SDPA flag can be scoped to the JVP region, while Dynamo owns the whole forward.
            MeanFlow is the campaign's critical path, so this lever helps the cheap net and not
            the expensive one.
    """

    __slots__ = ("sync_free_telemetry", "foreach_ema", "fused_sdpa", "fused_branch_forward",
                 "amp_dtype", "fused_adam", "compile_net")

    def __init__(self):
        reset(self)

    def __repr__(self):
        return f"<Perf {describe(self)}>"

    @property
    def autocast_enabled(self) -> bool:
        """Whether a ``torch.autocast`` context should wrap the forward at all."""
        return self.amp_dtype is not None


def reset(perf: Optional["_Perf"] = None) -> "_Perf":
    """Restore every switch to its shipped default (all off). Returns the block it reset."""
    perf = PERF if perf is None else perf
    perf.sync_free_telemetry = False
    perf.foreach_ema = False
    perf.fused_sdpa = False
    perf.fused_branch_forward = False
    perf.amp_dtype = None
    perf.fused_adam = False
    perf.compile_net = False
    return perf


def configure(perf: Optional["_Perf"] = None, **flags) -> "_Perf":
    """Set switches by keyword, validating names and the ``amp`` value.

    Args:
        perf: Block to mutate; defaults to the module-global ``PERF``.
        **flags: Any attribute named in :class:`_Perf`, plus the alias ``amp`` taking one of
            ``bf16`` / ``fp16`` / ``off`` (or a ``torch.dtype``) and writing ``amp_dtype``.

    Returns:
        The mutated block, so callers can chain off it.

    Raises:
        KeyError: On an unknown flag name. Deliberate — see the module docstring on why a
            typo'd perf flag must not read as a null result.
        ValueError: On an unrecognised ``amp`` value.
    """
    perf = PERF if perf is None else perf
    for name, value in flags.items():
        if name == "amp":
            if isinstance(value, torch.dtype) or value is None:
                perf.amp_dtype = value
            elif str(value).lower() in _AMP_DTYPES:
                perf.amp_dtype = _AMP_DTYPES[str(value).lower()]
            else:
                raise ValueError(
                    f"perf: amp={value!r} unknown; expected one of {sorted(_AMP_DTYPES)}")
        elif name in _Perf.__slots__:
            setattr(perf, name, value)
        else:
            raise KeyError(
                f"perf: unknown switch {name!r}; known: {sorted(_Perf.__slots__) + ['amp']}")
    return perf


def from_env(value: Optional[str] = None, perf: Optional["_Perf"] = None) -> "_Perf":
    """Apply the ``SP_PERF`` spec, e.g. ``"sync_free_telemetry,foreach_ema,amp=bf16"``.

    Args:
        value: The spec to parse; defaults to ``os.environ["SP_PERF"]`` (absent or empty means
            "leave every default alone").
        perf: Block to mutate; defaults to the module-global ``PERF``.

    Returns:
        The mutated block. Bare names set their switch to ``True``; ``name=value`` is passed
        through :func:`configure`, so the same validation and the same loud failure apply.
    """
    spec = os.environ.get("SP_PERF", "") if value is None else value
    flags = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, val = item.partition("=")
        flags[key.strip()] = val.strip() if sep else True
    return configure(perf, **flags)


@contextlib.contextmanager
def override(perf: Optional["_Perf"] = None, **flags):
    """Temporarily set switches inside a region, restoring them on exit.

    Exists for the places where a lever is *provably* unusable rather than merely untested, so
    the code can narrow it locally instead of the operator having to remember. The live case is
    ``fused_sdpa`` around ``torch.func.jvp``: the flash and mem-efficient SDPA kernels have no
    forward-mode AD rule in torch 2.2, so MeanFlow's JVP raises ``NotImplementedError`` under
    them. Narrowing there keeps the flag usable on every other forward in the same net.

    Args:
        perf: Block to mutate; defaults to the module-global ``PERF``.
        **flags: Same names and validation as :func:`configure`.
    """
    perf = PERF if perf is None else perf
    saved = {n: getattr(perf, n) for n in _Perf.__slots__}
    try:
        yield configure(perf, **flags)
    finally:
        for name, value in saved.items():
            setattr(perf, name, value)


def describe(perf: Optional["_Perf"] = None) -> str:
    """One-line summary of what is on, for the ``[perf]`` line every benchmark run logs.

    Returns ``"default (all off)"`` when nothing is set, so a log line can never be mistaken
    for a configured run that happened to look slow.
    """
    perf = PERF if perf is None else perf
    on = [n for n in _Perf.__slots__ if n != "amp_dtype" and getattr(perf, n)]
    if perf.amp_dtype is not None:
        on.append(f"amp={str(perf.amp_dtype).removeprefix('torch.')}")
    return " ".join(on) if on else "default (all off)"


#: The process's switch block. Populated from ``SP_PERF`` at import, which is also what makes
#: it correct in spawned workers: they re-import this module and re-read the inherited env.
PERF = _Perf()
from_env()
