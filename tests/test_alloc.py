"""The allocation theory, checked against brute force and closed forms."""

import math

import numpy as np
import pytest

from qlr.alloc import (
    LayerStats,
    _sweep,
    allocate_lagrangian,
    allocate_uniform,
    coding_gain,
    continuous_waterfill,
    layer_rate,
    surrogate_distortion,
)


def make_layers(k=24, seed=0):
    g = np.random.default_rng(seed)
    out = []
    for i in range(k):
        N = int(g.choice([256, 512, 1024]))
        M = int(g.choice([256, 512, 1024]))
        e = np.sort(g.exponential(1.0, N) ** g.uniform(1, 4))[::-1] * 10 ** g.uniform(-1, 2)
        out.append(
            LayerStats(f"l{i}", N, M, float(10 ** g.uniform(-2, 1)), e,
                       damp=float(e.sum() / N * 0.01))
        )
    return out


@pytest.mark.parametrize("seed", range(8))
def test_rank_distortion_curve_is_convex(seed):
    """Second difference is sigma_{r+1}^2 - sigma_{r+2}^2 >= 0 by spectral
    ordering, so convexity needs no assumption on the Hessian."""
    g = np.random.default_rng(seed)
    n = 128
    e = np.sort(g.exponential(1.0, n) ** g.uniform(0.5, 5))[::-1]
    st = LayerStats("l", n, n, 1.0, e, damp=float(g.uniform(0, 0.1)))
    r = np.arange(0, n - 1)
    assert np.all(st.h(r + 2) - 2 * st.h(r + 1) + st.h(r) >= -1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_rank_has_a_finite_useful_maximum(seed):
    """h is *not* monotone: its first difference is lambda - sigma_{r+1}^2, so
    once the spectrum falls below the damping level, extra rank costs more
    regularisation than the tail energy it removes. The minimiser is exactly
    r* = #{i : sigma_i^2 > lambda}, and convexity makes it unique.

    This is why the allocator never has to be told an upper rank limit.
    """
    g = np.random.default_rng(seed)
    n = 128
    e = np.sort(g.exponential(1.0, n) ** g.uniform(0.5, 5))[::-1]
    lam = float(np.median(e))
    st = LayerStats("l", n, n, 1.0, e, damp=lam)
    r_star = int((e > lam).sum())
    vals = st.h(np.arange(0, n + 1))
    assert int(np.argmin(vals)) == r_star
    assert np.all(np.diff(vals[: r_star + 1]) <= 1e-12)   # decreasing up to r*
    assert np.all(np.diff(vals[r_star:]) >= -1e-12)       # increasing after


def test_allocator_never_buys_rank_beyond_the_useful_maximum():
    layers = make_layers(k=12, seed=5)
    a = allocate_lagrangian(layers, 6.0, rank_choices=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    for st, r in zip(layers, a.ranks):
        r_star = int((np.asarray(st.eigs) > st.damp).sum())
        assert r <= r_star


def test_total_rate_is_monotone_in_nu():
    layers = make_layers()
    bc, rc = np.array([2, 3, 4, 6, 8]), np.array([0, 1, 2, 4, 8, 16, 32, 64])
    rates = [_sweep(layers, nu, bc, rc, 16.0)[3] for nu in np.logspace(-14, -1, 40)]
    assert np.all(np.diff(rates) <= 1e-6)


def test_coding_gain_matches_the_measured_optimum():
    """D_uniform / D_optimal at fixed rate equals the parameter-weighted
    AM/GM ratio of the per-layer difficulty coefficients."""
    layers = make_layers(k=40, seed=3)
    ranks = np.zeros(len(layers), dtype=int)
    cg = coding_gain(layers, ranks)

    P = np.array([s.n_params for s in layers], float)
    a = np.array([s.omega * s.S * float(s.h(0)) / (4 * s.n_params) for s in layers])
    bbar = 4.0
    d_uniform = float(np.sum(P * a * 4.0**-bbar))
    log_c = (P * np.log2(a)).sum() / P.sum() - 2 * bbar
    d_optimal = float(P.sum() * 2.0**log_c)
    assert cg["gain_ratio"] == pytest.approx(d_uniform / d_optimal, rel=1e-9)
    assert cg["bits_saved"] == pytest.approx(0.5 * math.log2(cg["gain_ratio"]), rel=1e-12)
    assert cg["gain_ratio"] >= 1.0  # AM-GM


def test_allocation_is_feasible_and_beats_uniform():
    layers = make_layers()
    for b, r in [(2, 16), (3, 32), (4, 8)]:
        u = allocate_uniform(layers, b, r)
        a = allocate_lagrangian(layers, u.bits_per_weight,
                                rank_choices=[0, 1, 2, 4, 8, 16, 32, 64, 128])
        assert a.total_rate <= u.total_rate * (1 + 1e-9)
        assert a.total_distortion <= u.total_distortion


def test_lagrangian_point_is_optimal_for_its_own_multiplier():
    """Every layer independently minimises D + nu C, so no single-layer change
    can lower the Lagrangian -- the defining property of the sweep."""
    layers = make_layers(k=6, seed=7)
    bc, rc = np.array([2, 3, 4, 6, 8]), np.array([0, 2, 8, 32, 128])
    nu = 1e-8
    bits, ranks, _, _ = _sweep(layers, nu, bc, rc, 16.0)
    for k, st in enumerate(layers):
        base = surrogate_distortion(st, bits[k], ranks[k]) + nu * layer_rate(st, bits[k], ranks[k])
        for b in bc:
            for r in rc:
                alt = surrogate_distortion(st, b, r) + nu * layer_rate(st, b, r)
                assert alt >= base - 1e-9 * abs(base)


def test_gap_bound_is_reported_and_sound():
    layers = make_layers()
    a = allocate_lagrangian(layers, 3.5)
    assert a.gap_bound is not None and a.gap_bound >= 0
    # The bound is nu times the unspent budget, so it vanishes when the sweep
    # exactly exhausts the budget.
    total_params = sum(s.n_params for s in layers)
    unspent = 3.5 * total_params - a.total_rate
    assert a.gap_bound == pytest.approx(a.nu * max(0.0, unspent), rel=1e-9)


def test_waterfill_rank_rule_stops_at_the_threshold():
    layers = make_layers(k=5, seed=11)
    _, ranks = continuous_waterfill(layers, nu=1e-10)
    for st, r in zip(layers, ranks):
        kappa = 2 * math.log(2) * 16.0 * (st.N + st.Nout) / st.n_params
        if r < len(st.eigs):
            assert st.eigs[r] <= st.damp + kappa * float(st.h(int(r))) + 1e-12
        if r > 0:
            assert st.eigs[r - 1] > st.damp + kappa * float(st.h(int(r) - 1)) - 1e-12
