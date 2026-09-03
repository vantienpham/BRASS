"""Shared post-processing of sweep results.

Both the figure generator and the bits-saved analysis need the same notion of
"what a practitioner could achieve without allocating", so it lives here rather
than being written twice and drifting.
"""

from __future__ import annotations

import math
import re

__all__ = [
    "BASELINE_FAMILIES",
    "envelope",
    "rate_for_quality",
    "envelope_saving",
    "is_allocated",
]

# Every configuration reachable by choosing one (b, r) for the whole network --
# including the cheaper-factor and refined variants. Leaving those out would
# flatter the allocator by comparing it against a baseline a practitioner would
# not actually settle for.
BASELINE_FAMILIES = (
    r"gptq-b\d+",          # no low-rank component
    r"gilora-b\d+r\d+",    # GPTQ-intrinsic LoRA, FP16 factors
    r"olrc-b\d+r\d+",      # GPTQ + optimal post-hoc compensation
    r"giloraQ-b\d+r\d+",   # ... with the factors quantized to 4/8 bits
    r"giloraL-b\d+r\d+",   # ... with one OLrC + Bid-Up refinement loop
)


def is_allocated(name: str) -> bool:
    """True for an allocated operating point, false for the ablations.

    ``alloc`` and ``allocT`` are the plain allocations; ``allocQ`` and
    ``allocL`` are the same allocation with quantized factors and with a
    refinement loop, and must be counted here -- the corresponding *uniform*
    variants are in `BASELINE_FAMILIES`, so leaving them out would compare an
    allocation carrying FP16 factors against a baseline allowed cheap ones.
    ``allocNW``/``allocTNW`` (no sensitivity weights) and ``allocB``/``allocR``
    (one knob pinned) are ablations and are deliberately excluded.
    """
    return re.fullmatch(r"alloc(T|Q|L)?-.*", name) is not None


def envelope(rows, metric="wikitext2_ppl", families=BASELINE_FAMILIES):
    """Lower envelope of the named families as ``[(bpw, log metric), ...]``.

    The envelope is ``E(x) = min{y_i : x_i <= x}``, a non-increasing step
    function of rate. Sweeping in ascending rate and keeping the running record
    returns the points where it steps down -- the configurations that are
    actually worth choosing.
    """
    pts = sorted(
        (r["bits_per_weight"], math.log(r[metric]))
        for r in rows
        if metric in r
        and r["name"] != "fp16"
        and any(re.fullmatch(f, r["name"]) for f in families)
        and isinstance(r[metric], (int, float))
        and math.isfinite(r[metric])
        and r[metric] > 0
    )
    keep, best = [], math.inf
    for x, y in pts:
        if y < best:
            best = y
            keep.append((x, y))
    return keep


def rate_for_quality(env, log_metric: float) -> float | None:
    """Rate at which the envelope first reaches ``log_metric``.

    Linear interpolation in ``(rate, log metric)`` -- the scale on which the
    high-resolution distortion model is linear, so a measured saving and a
    predicted one are read off the same axes. Returns ``None`` outside the
    envelope's range rather than extrapolating.
    """
    if len(env) < 2:
        return None
    xs = [x for x, _ in env]
    ys = [y for _, y in env]
    if log_metric > ys[0] or log_metric < ys[-1]:
        return None
    for i in range(len(env) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 >= log_metric >= y1:
            if y0 == y1:
                return xs[i]
            t = (y0 - log_metric) / (y0 - y1)
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


def envelope_saving(env, bpw: float, log_metric: float):
    """Rate saved against the envelope, handling points beyond its reach.

    Returns ``(saving, status)``. ``status`` is

      ``"interp"``    the envelope reaches this quality inside its measured
                      range and the saving is exact;
      ``"censored"``  the allocated point is *better* than every uniform
                      configuration tested, so the envelope never reaches it.
                      The saving is then only bounded below, by the distance to
                      the envelope's highest measured rate;
      ``None``        the point is worse than the envelope's cheapest measured
                      point, so there is nothing to compare against.

    Dropping censored points, which is what simply discarding a ``None`` from
    `rate_for_quality` does, discards exactly the cases where allocation did
    best and biases the reported saving downward.
    """
    if len(env) < 2:
        return None, None
    xs = [x for x, _ in env]
    ys = [y for _, y in env]
    if log_metric < ys[-1]:                     # better than the envelope's best
        return xs[-1] - bpw, "censored"
    if log_metric > ys[0]:                      # worse than its cheapest point
        return None, None
    return rate_for_quality(env, log_metric) - bpw, "interp"
