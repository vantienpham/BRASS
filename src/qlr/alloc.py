"""Rate-distortion allocation of bit-width and rank across layers.

Zhang & Saab compress every layer at one uniform ``(b, r)``. This module treats
the pair as a per-layer decision under a single network-wide storage budget.

Rate.  A layer stores ``b`` bits per weight plus two low-rank factors:

    C(b, r) = b N N' + b_LR r (N + N') + (scale/zero overhead),

and dividing by ``N N'`` gives the *effective bit-width* -- the common currency
in which bits and rank are traded.

Distortion.  Corollary 3.9 of Zhang & Saab bounds the layer-wise error of
GPTQ-intrinsic LoRA by

    D(b, r) <= (1/4) sum_j delta_j(b)^2  ( tail_r(H) + (N + r) lambda ),

which factorises as ``g(b) h(r)`` with ``g(b) = S / (4 (2^b - 1)^2)``,
``S = beta^2 sum_j range(W_.j)^2``, and ``h(r) = sum_{i>r} sigma_i^2 + (N+r)
lambda``. Both factors are measurable from statistics a GPTQ pass already
computes -- the Hessian eigenvalues and the per-column weight ranges -- so the
whole allocation is solved without touching a GPU.

Two facts make the problem tractable, both proved in the paper accompanying
this code:

  * ``h`` is convex on the integers for *every* Hessian, because its second
    difference is ``sigma_{r+1}^2 - sigma_{r+2}^2 >= 0``. Spectral ordering
    alone buys convexity of the rank-distortion curve; no assumption needed.
  * The Lagrangian sweep therefore traces the lower convex hull of each
    layer's achievable (rate, distortion) set, and the sweep's suboptimality
    at a given budget is bounded by ``nu`` times one rate quantum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "LayerStats",
    "attach_measured_distortion",
    "fit_rank_calibration",
    "layer_lower_bound",
    "network_lower_bound",
    "Allocation",
    "surrogate_distortion",
    "layer_rate",
    "effective_bits",
    "allocate_lagrangian",
    "allocate_uniform",
    "continuous_waterfill",
    "coding_gain",
]

FP_BITS = 16.0  # storage precision of the low-rank factors, unless overridden


@dataclass
class LayerStats:
    """Everything the allocator needs about one linear layer.

    ``eigs`` are the Hessian eigenvalues in *descending* order. ``S`` is the
    weight-range statistic ``sum_j beta_j^2 (max_i W_ij - min_i W_ij)^2``, so
    that ``sum_j delta_j(b)^2 = S / (2^b - 1)^2`` exactly for the paper's
    asymmetric per-channel grid.

    When the clipping factors are chosen per bit-width by `quant.search_beta`,
    ``S`` becomes bit-dependent and ``S_bits`` carries the measured value for
    each admissible ``b``; ``S`` then holds the unclipped (beta = 1) value,
    which is what the high-resolution theory of the accompanying paper uses.
    """

    name: str
    N: int  # input features (rows of W)
    Nout: int  # output features (columns of W)
    S: float
    eigs: np.ndarray
    S_bits: dict[int, float] | None = None
    damp: float = 0.0
    omega: float = 1.0  # downstream sensitivity weight
    dtable: dict[tuple[int, int], float] | None = None
    rank_cal: dict[int, float] | None = None
    tail: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        e = np.asarray(self.eigs, dtype=np.float64)
        if e.ndim != 1:
            raise ValueError("eigs must be one-dimensional")
        # tail[r] = sum_{i>r} sigma_i^2, for r = 0 .. len(e). Computed once;
        # every distortion query is then an O(1) lookup.
        self.tail = np.concatenate([np.cumsum(e[::-1])[::-1], [0.0]])

    @property
    def n_params(self) -> int:
        return self.N * self.Nout

    def h(self, r: np.ndarray | int) -> np.ndarray:
        """Rank factor ``tail_r + (N + r) lambda``.

        With ``rank_cal`` attached, the bound's rank dependence is shrunk toward
        no-improvement by a measured factor ``kappa(r) in [0, 1]``:

            h_cal(r) = h(0) - kappa(r) * (h(0) - h(r)),

        so ``kappa = 1`` recovers the bound and ``kappa = 0`` says rank buys
        nothing. The bound claims a rank-8 correction removes 56% of the error
        where the measurement says 8% (`fit_rank_calibration`), and this is the
        one-parameter-per-rank repair of that.
        """
        r = np.asarray(r)
        raw = self.tail[np.clip(r, 0, len(self.tail) - 1)] + (self.N + r) * self.damp
        if self.rank_cal is None:
            return raw
        h0 = float(self.tail[0] + self.N * self.damp)
        ks = np.array(sorted(self.rank_cal), dtype=np.float64)
        vs = np.array([self.rank_cal[int(k)] for k in ks], dtype=np.float64)
        kappa = np.interp(np.asarray(r, dtype=np.float64), ks, vs)
        return h0 - kappa * (h0 - raw)

    def g(self, b: np.ndarray | int) -> np.ndarray:
        """Bit factor ``S(b) / (4 (2^b - 1)^2)``."""
        b_arr = np.asarray(b)
        denom = 4.0 * (np.power(2.0, b_arr.astype(np.float64)) - 1.0) ** 2
        if self.S_bits is None:
            return self.S / denom
        flat = b_arr.reshape(-1)
        num = np.array([self.S_bits.get(int(bi), self.S) for bi in flat], dtype=np.float64)
        return (num.reshape(b_arr.shape) / denom) if b_arr.ndim else float(num[0] / denom)


def surrogate_distortion(st: LayerStats, b, r) -> np.ndarray:
    """Distortion the allocator optimises: ``omega * g(b) * h(r)``.

    When the layer carries a ``dtable`` -- a distortion surface measured by
    `scripts/profile_layers.py` -- that is used instead of the bound. The
    measured variant is the upper baseline: it says how well allocation can do
    when the distortion model is exactly right, which is what makes the gap to
    the free surrogate interpretable.
    """
    if st.dtable is None:
        return st.omega * st.g(b) * st.h(r)
    b_arr = np.asarray(b, dtype=int)
    r_arr = np.asarray(r, dtype=int)
    flat_b, flat_r = b_arr.reshape(-1), r_arr.reshape(-1)
    out = np.empty(flat_b.size, dtype=np.float64)
    for i, (bi, ri) in enumerate(zip(flat_b, flat_r)):
        val = st.dtable.get((int(bi), int(ri)))
        if val is None:
            # Outside the measured grid: fall back to the bound rather than
            # silently interpolating something the profile never observed.
            val = float(st.g(bi) * st.h(ri))
        out[i] = st.omega * val
    return out.reshape(b_arr.shape) if b_arr.ndim else float(out[0])


def layer_rate(st: LayerStats, b, r, lr_bits: float = FP_BITS) -> np.ndarray:
    """Stored bits for the layer at ``(b, r)``, including grid parameters."""
    b = np.asarray(b, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    # Two fp16 grid parameters (scale, zero) per output channel. Small, but it
    # is the term that makes very low b stop paying off, so it is not dropped.
    overhead = 2.0 * FP_BITS * st.Nout
    return b * st.n_params + lr_bits * r * (st.N + st.Nout) + overhead


def effective_bits(st: LayerStats, b, r, lr_bits: float = FP_BITS) -> np.ndarray:
    """Rate expressed per original weight."""
    return layer_rate(st, b, r, lr_bits) / st.n_params


@dataclass
class Allocation:
    """A per-layer choice of ``(b, r)`` plus its aggregate rate and distortion."""

    bits: np.ndarray
    ranks: np.ndarray
    total_rate: float
    total_distortion: float
    total_params: int
    nu: float | None = None
    gap_bound: float | None = None

    @property
    def bits_per_weight(self) -> float:
        return self.total_rate / self.total_params

    def as_dict(self, layers: list[LayerStats]) -> dict:
        return {
            "bits_per_weight": self.bits_per_weight,
            "total_rate_bits": self.total_rate,
            "total_distortion": self.total_distortion,
            "nu": self.nu,
            "gap_bound": self.gap_bound,
            "per_layer": {
                st.name: {"bits": int(b), "rank": int(r)}
                for st, b, r in zip(layers, self.bits, self.ranks)
            },
        }


def _best_per_layer(
    st: LayerStats,
    nu: float,
    bit_choices: np.ndarray,
    rank_choices: np.ndarray,
    lr_bits: float,
) -> tuple[int, int, float, float]:
    """argmin over the (b, r) grid of ``omega D(b,r) + nu C(b,r)`` for one layer."""
    B, R = np.meshgrid(bit_choices, rank_choices, indexing="ij")
    D = surrogate_distortion(st, B, R)
    C = layer_rate(st, B, R, lr_bits)
    J = D + nu * C
    k = int(np.argmin(J))
    i, j = np.unravel_index(k, J.shape)
    return int(B[i, j]), int(R[i, j]), float(D[i, j]), float(C[i, j])


def _sweep(layers, nu, bit_choices, rank_choices, lr_bits):
    bits = np.empty(len(layers), dtype=int)
    ranks = np.empty(len(layers), dtype=int)
    tot_d = tot_c = 0.0
    for k, st in enumerate(layers):
        b, r, d, c = _best_per_layer(st, nu, bit_choices, rank_choices, lr_bits)
        bits[k], ranks[k] = b, r
        tot_d += d
        tot_c += c
    return bits, ranks, tot_d, tot_c


def allocate_lagrangian(
    layers: list[LayerStats],
    target_bits_per_weight: float,
    bit_choices=(2, 3, 4, 5, 6, 8),
    rank_choices=None,
    lr_bits: float = FP_BITS,
    tol: float = 1e-4,
    max_iter: int = 200,
) -> Allocation:
    """Solve the budgeted allocation by bisection on the Lagrange multiplier.

    For each ``nu > 0`` every layer independently minimises
    ``omega D(b,r) + nu C(b,r)``; the resulting total rate is non-increasing in
    ``nu``, so bisection finds the largest feasible allocation. The returned
    ``gap_bound`` is ``nu * (budget - achieved rate)``, an upper bound on how
    far the allocation's distortion can be from the true constrained optimum.
    """
    if not layers:
        raise ValueError("no layers to allocate over")
    bit_choices = np.asarray(sorted(bit_choices), dtype=int)
    if rank_choices is None:
        rmax = max(1, min(min(st.N, st.Nout) // 2 for st in layers))
        rank_choices = _log_rank_grid(rmax)
    rank_choices = np.asarray(sorted(set(int(r) for r in rank_choices)), dtype=int)

    total_params = sum(st.n_params for st in layers)
    budget = target_bits_per_weight * total_params

    # Bracket nu. Large nu -> cheapest configuration; small nu -> most expensive.
    lo, hi = 1e-30, 1.0
    _, _, _, c_hi = _sweep(layers, hi, bit_choices, rank_choices, lr_bits)
    grow = 0
    while c_hi > budget and grow < 300:
        hi *= 10.0
        _, _, _, c_hi = _sweep(layers, hi, bit_choices, rank_choices, lr_bits)
        grow += 1
    _, _, _, c_lo = _sweep(layers, lo, bit_choices, rank_choices, lr_bits)
    if c_lo <= budget:
        # Budget covers the richest configuration on the grid; take it.
        bits, ranks, d, c = _sweep(layers, lo, bit_choices, rank_choices, lr_bits)
        return Allocation(bits, ranks, c, d, total_params, nu=lo, gap_bound=0.0)

    best = None
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)  # geometric bisection: nu spans many decades
        bits, ranks, d, c = _sweep(layers, mid, bit_choices, rank_choices, lr_bits)
        if c <= budget:
            best = (bits, ranks, d, c, mid)
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.0 + tol:
            break

    if best is None:  # pragma: no cover - only if even nu -> inf overshoots
        bits, ranks, d, c = _sweep(layers, hi, bit_choices, rank_choices, lr_bits)
        best = (bits, ranks, d, c, hi)

    bits, ranks, d, c, nu = best
    return Allocation(
        bits=bits,
        ranks=ranks,
        total_rate=c,
        total_distortion=d,
        total_params=total_params,
        nu=nu,
        gap_bound=nu * max(0.0, budget - c),
    )


def _log_rank_grid(rmax: int) -> np.ndarray:
    """Ranks on a log grid plus 0. Keeps the sweep cheap without coarsening
    the interesting small-rank region."""
    vals = {0}
    r = 1
    while r <= rmax:
        vals.add(r)
        r = max(r + 1, int(round(r * 1.5)))
    vals.add(rmax)
    return np.array(sorted(vals), dtype=int)


def allocate_uniform(
    layers: list[LayerStats],
    bits: int,
    rank: int,
    lr_bits: float = FP_BITS,
) -> Allocation:
    """The baseline: the same ``(b, r)`` everywhere, as in Zhang & Saab."""
    n = len(layers)
    b = np.full(n, bits, dtype=int)
    r = np.array([min(rank, st.N, st.Nout) for st in layers], dtype=int)
    d = sum(float(surrogate_distortion(st, bi, ri)) for st, bi, ri in zip(layers, b, r))
    c = sum(float(layer_rate(st, bi, ri, lr_bits)) for st, bi, ri in zip(layers, b, r))
    return Allocation(b, r, c, d, sum(st.n_params for st in layers))


def continuous_waterfill(
    layers: list[LayerStats], nu: float, lr_bits: float = FP_BITS
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary point of the continuous relaxation, for the theory section.

    Equating marginal distortion per marginal bit across both knobs and all
    layers gives two conditions. In ``b``, since ``D ~ 4^-b``,

        2 ln2 * D_l / (N N')  =  nu      i.e.  D_l  proportional to  N N',

    so distortion should be equalised *per parameter*, not per layer. In ``r``,

        sigma_{l,r+1}^2  =  lambda + (2 ln2 b_LR (N+N') / (N N')) h_l(r),

    a water-filling rule: buy rank while the next eigenvalue exceeds a
    multiple of the energy still left in the tail.

    Returns the integer ranks satisfying the second condition and the
    real-valued bit-widths satisfying the first.
    """
    ranks = np.empty(len(layers), dtype=int)
    bits = np.empty(len(layers), dtype=np.float64)
    for k, st in enumerate(layers):
        kappa = 2.0 * math.log(2.0) * lr_bits * (st.N + st.Nout) / st.n_params
        e = np.asarray(st.eigs, dtype=np.float64)
        r = 0
        while r < len(e) and e[r] > st.damp + kappa * float(st.h(r)):
            r += 1
        ranks[k] = r
        # 2 ln2 * omega g(b) h(r) / (N N') = nu  ->  solve for b.
        coef = st.omega * st.S * float(st.h(r)) * 2.0 * math.log(2.0) / (4.0 * st.n_params)
        x = math.sqrt(max(coef / nu, 1e-300))  # x = 2^b - 1
        bits[k] = math.log2(1.0 + x)
    return bits, ranks


def coding_gain(layers: list[LayerStats], ranks: np.ndarray) -> dict:
    """Predicted benefit of optimal bit allocation over a uniform bit-width.

    At a fixed total rate and fixed ranks, writing ``D_l = N_l N'_l a_l 4^-b_l``
    with ``a_l = omega_l S_l h_l(r_l) / (4 N_l N'_l)``, the optimum equalises
    ``a_l 4^-b_l`` while uniform ``b`` does not. The ratio of the two total
    distortions is the parameter-weighted arithmetic mean of ``a_l`` over its
    parameter-weighted geometric mean -- at least 1 by AM-GM, with equality
    only if every layer is equally hard.

    The gain converts to a bit-width saving of ``0.5 log2(ratio)`` at equal
    distortion, and is computable before any quantisation is run.
    """
    P = np.array([st.n_params for st in layers], dtype=np.float64)
    a = np.array(
        [
            st.omega * st.S * float(st.h(int(r))) / (4.0 * st.n_params)
            for st, r in zip(layers, ranks)
        ],
        dtype=np.float64,
    )
    if np.any(a <= 0):
        raise ValueError("non-positive distortion coefficient; check S and eigs")
    w = P / P.sum()
    am = float(np.sum(w * a))
    gm = float(np.exp(np.sum(w * np.log(a))))
    return {
        "arithmetic_mean": am,
        "geometric_mean": gm,
        "gain_ratio": am / gm,
        "bits_saved": 0.5 * math.log2(am / gm),
        "log_spread_std": float(np.sqrt(np.sum(w * (np.log2(a) - np.sum(w * np.log2(a))) ** 2))),
    }


def attach_measured_distortion(
    layers: list[LayerStats], profile: dict, key: str = "gilora"
) -> tuple[list[LayerStats], list[int], list[int]]:
    """Attach measured ``D_l(b, r)`` surfaces from `profile_layers.py`.

    Returns the layers that were matched together with the measured bit and
    rank grids, so the caller can restrict the allocation search to
    configurations that were actually observed. Layers absent from the profile
    are dropped rather than silently left on the bound -- mixing measured and
    modelled distortions inside one objective would make the comparison between
    them meaningless.
    """
    bits: set[int] = set()
    ranks: set[int] = set()
    matched = []
    for st in layers:
        entry = profile.get(st.name)
        if entry is None:
            continue
        table = {}
        for br, rec in entry["D"].items():
            b_s, r_s = br.split(",")
            b_i, r_i = int(b_s), int(r_s)
            if key not in rec:
                continue
            table[(b_i, r_i)] = float(rec[key])
            bits.add(b_i)
            ranks.add(r_i)
        if table:
            st.dtable = table
            matched.append(st)
    return matched, sorted(bits), sorted(ranks)


# --- information-theoretic lower bounds --------------------------------------
#
# Zhang & Saab prove, for a finite alphabet of half-width B = 2^(b-1), a rank
# budget r and a magnitude budget rho on the low-rank part, that some weight
# matrix in the non-spiky unit ball forces
#
#     ||X (W - Q - LR)||_F^2  >~  sigma_min^2 /
#         [ (2B+1)^(2J/(J-1)) * (41 sqrt(pi e / 6) rho)^(2/(J-1)) ],
#     J = P / (r (N + N' + 1) + 2).
#
# These are per-layer impossibility statements. Because both the layer-wise
# objective and the storage cost are sums over layers, they compose into a
# network-level envelope by solving exactly the same allocation problem -- which
# is what `network_lower_bound` does.

_LB_CONST = 41.0 * math.sqrt(math.pi * math.e / 6.0)


def layer_lower_bound(
    st: LayerStats, b, r, rho: float = 1.0, wfro2: float | None = None
) -> np.ndarray:
    """Per-layer information-theoretic lower bound, scaled to this layer's norm.

    ``rho`` is the Frobenius budget on the low-rank component. Larger ``rho``
    admits more compressors and so gives a *smaller* lower bound; ``rho = 1``
    is the permissive regime of Zhang & Saab and is the conservative choice
    when the bound is to be quoted as a limit on all algorithms.

    Returns ``0`` wherever the rank budget is large enough that ``J <= 1``, i.e.
    where the low-rank part alone has as many parameters as the matrix and the
    bound degenerates.
    """
    b = np.asarray(b, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    B = np.power(2.0, b - 1.0)
    J = st.n_params / (r * (st.N + st.Nout + 1.0) + 2.0)
    scale = float(wfro2 if wfro2 is not None else 1.0)
    sigma_min2 = float(np.min(st.eigs))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        expo = J / (J - 1.0)
        val = sigma_min2 / (
            np.power(2.0 * B + 1.0, 2.0 * expo)
            * np.power(_LB_CONST * rho, 2.0 / (J - 1.0))
        )
    val = np.where(J > 1.0, val, 0.0)
    return st.omega * scale * np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)


def network_lower_bound(
    layers: list[LayerStats],
    target_bits_per_weight: float,
    wfro2: dict[str, float] | None = None,
    rho: float = 1.0,
    bit_choices=(2, 3, 4, 5, 6, 8),
    rank_choices=None,
    lr_bits: float = FP_BITS,
    tol: float = 1e-4,
) -> Allocation:
    """Allocate the per-layer lower bounds under one budget.

    The result is the largest total distortion that is provably unavoidable at
    this rate: no low-precision plus low-rank compressor, however clever, can
    beat it on every network. It is obtained by the same Lagrangian sweep as
    `allocate_lagrangian`, with the lower bound in place of the surrogate --
    which is legitimate because the sweep only needs separability, not any
    particular distortion model.
    """
    wfro2 = wfro2 or {}
    if rank_choices is None:
        rank_choices = _log_rank_grid(max(1, min(min(s.N, s.Nout) // 2 for s in layers)))
    bit_choices = np.asarray(sorted(bit_choices), dtype=int)
    rank_choices = np.asarray(sorted(set(int(x) for x in rank_choices)), dtype=int)
    Bg, Rg = np.meshgrid(bit_choices, rank_choices, indexing="ij")

    total_params = sum(s.n_params for s in layers)
    budget = target_bits_per_weight * total_params
    # Both surfaces are budget-independent, so build them once and let the
    # bisection reduce to argmin over a precomputed pair of tables.
    grids = [
        (
            layer_lower_bound(st, Bg, Rg, rho, wfro2.get(st.name)),
            layer_rate(st, Bg, Rg, lr_bits),
        )
        for st in layers
    ]

    def sweep(nu):
        bits = np.empty(len(layers), dtype=int)
        ranks = np.empty(len(layers), dtype=int)
        td = tc = 0.0
        for k, (D, C) in enumerate(grids):
            i, j = np.unravel_index(int(np.argmin(D + nu * C)), D.shape)
            bits[k], ranks[k] = int(Bg[i, j]), int(Rg[i, j])
            td += float(D[i, j])
            tc += float(C[i, j])
        return bits, ranks, td, tc

    lo, hi = 1e-40, 1.0
    for _ in range(400):
        if sweep(hi)[3] <= budget:
            break
        hi *= 10.0
    best = None
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        bits, ranks, d, c = sweep(mid)
        if c <= budget:
            best = (bits, ranks, d, c, mid)
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.0 + tol:
            break
    if best is None:  # pragma: no cover
        bits, ranks, d, c = sweep(hi)
        best = (bits, ranks, d, c, hi)
    bits, ranks, d, c, nu = best
    return Allocation(bits, ranks, c, d, total_params, nu=nu, gap_bound=None)


def fit_rank_calibration(profile: dict, layers: list[LayerStats], key: str = "gilora"):
    """One ``kappa`` per profiled rank, from measured against predicted decay.

    For each rank in the profile, compare the median measured
    ``D_l(r)/D_l(0)`` with the median predicted ``h_l(r)/h_l(0)`` and set

        kappa(r) = (1 - measured) / (1 - predicted),

    the fraction of the bound's promised improvement that actually materialises.
    `LayerStats.h` then interpolates between these knots.

    The measured ratio is close to layer-independent -- its 90th/10th percentile
    spread is only 1.10 at r = 8 and 1.22 at r = 128 -- which is what makes a
    single global curve, fitted once on one model, a defensible repair rather
    than a per-layer fit that would need profiling every network.
    """
    by_name = {l.name: l for l in layers}
    bits, ranks = set(), set()
    for entry in profile.values():
        for br in entry.get("D", {}):
            b_s, r_s = br.split(",")
            bits.add(int(b_s))
            ranks.add(int(r_s))
    cal: dict[int, float] = {0: 0.0}
    for r in sorted(ranks):
        if r == 0:
            continue
        meas, pred = [], []
        for name, entry in profile.items():
            st = by_name.get(name)
            if st is None:
                continue
            d0 = [entry["D"].get(f"{b},0", {}).get(key) for b in sorted(bits)]
            dr = [entry["D"].get(f"{b},{r}", {}).get(key) for b in sorted(bits)]
            if any(v is None for v in d0 + dr):
                continue
            meas.append(float(np.median(np.asarray(dr) / np.asarray(d0))))
            pred.append(float(st.h(r) / st.h(0)))
        if meas:
            mm, pp = float(np.median(meas)), float(np.median(pred))
            cal[r] = float(np.clip((1.0 - mm) / max(1.0 - pp, 1e-9), 0.0, 1.0))
    return cal
