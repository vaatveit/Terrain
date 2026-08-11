"""Landscape evolution. The steady-state relation gives exact answers to check against.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import flow, metrics, stage1, stage2  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


def line_graph(n_cells, spacing_km=10.0, cell_area_km2=100.0):
    """A single chain of cells: cell 0 is the outlet, flow runs 0 <- 1 <- 2 <- ..."""
    nbr = np.full((n_cells, 2), -1, dtype=np.int32)
    nbr[:-1, 0] = np.arange(1, n_cells)
    nbr[1:, 1] = np.arange(n_cells - 1)
    arc = np.full((n_cells, 2), spacing_km, dtype=np.float32)
    edge = np.full((n_cells, 2), spacing_km, dtype=np.float32)
    area = np.full(n_cells, cell_area_km2, dtype=np.float32)
    return nbr, arc, edge, area


# --------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------


def test_implicit_solver_reaches_the_analytic_steady_state():
    """At steady state U = K A^m S^n, so slope is exactly (U/K)^(1/n) A^(-m/n).

    Checked on a single channel where drainage area is known in closed form.
    """
    n_cells = 200
    spacing_km, cell_area = 10.0, 100.0
    nbr, arc, edge, area = line_graph(n_cells, spacing_km, cell_area)
    is_ocean = np.zeros(n_cells, bool)
    is_ocean[0] = True

    K, m, n = 1e-5, 0.5, 1.0
    u_m_per_myr = np.where(is_ocean, 0.0, 200.0).astype(np.float32)
    z0 = np.zeros(n_cells, dtype=np.float32)

    class G:  # minimal stand-in for a Grid
        cell_count = n_cells
        neighbours, arc_dist, edge_len = nbr, arc, edge
    G.area = area

    p = stage2.LemParams(K=K, m=m, n=n, D=0.0, dt_myr=1.0)
    res = stage2.evolve(G, z0, u_m_per_myr, is_ocean, p, n_steps=4000)

    fr = flow.route(res.elevation, nbr, arc, area, is_ocean)
    interior = slice(20, 180)
    predicted = (200.0 / 1e6 / K) * (fr.acc[interior] * 1e6) ** (-m / n)
    measured = fr.slope[interior]
    assert np.allclose(measured, predicted, rtol=0.02), (
        f"steady-state slope off: {measured[:3]} vs {predicted[:3]}")


@pytest.mark.parametrize("m,n", [(0.3, 1.0), (0.45, 1.0), (0.6, 1.0)])
def test_recovered_concavity_equals_m_over_n(m, n):
    """theta = m/n is the model's defining prediction. Recover it from the output.

    This is the end-to-end check: it only passes if the implicit solver, the flow routing,
    the spherical cell areas and the metrics all agree. Uplift must be UNIFORM -- see
    stage2.uplift_field for why a spatially varying U inflates the measured theta.
    """
    g = build_goldberg(96)
    crust = stage1.generate_crust(g, seed=1)
    is_ocean = ~crust.land_mask
    uplift = np.where(crust.land_mask, 300.0, 0.0).astype(np.float32)

    # Scale K with m so every case gets a comparable time to steady state. Incision
    # celerity is K*A^m, so at a typical A of 1e10 m^2, plain K=1e-6 makes m=0.3 converge
    # about 22x slower than m=0.45 -- it reads as a wrong exponent when it is really an
    # under-converged one.
    K = 1e-6 * (1e10) ** (0.45 - m)
    p = stage2.LemParams(K=K, m=m, n=n, D=0.05, dt_myr=0.5)
    res = stage2.evolve(g, crust.elevation, uplift, is_ocean, p, n_steps=350)

    fr = flow.route(res.elevation, g.neighbours, g.arc_dist, g.area, is_ocean)
    ch = metrics.channel_mask(fr.acc, float(np.median(g.area) * 6))
    reg, _, _ = metrics.slope_area(fr, ch)
    assert -reg.exponent == pytest.approx(m / n, abs=0.06)
    assert reg.r2 > 0.9


def test_erosion_moves_concavity_from_noise_toward_real_terrain():
    """The whole point of Stage 2: noise-only terrain has almost no concavity."""
    g = build_goldberg(96)
    crust = stage1.generate_crust(g, seed=1)
    is_ocean = ~crust.land_mask
    uplift = np.where(crust.land_mask, 300.0, 0.0).astype(np.float32)
    thr = float(np.median(g.area) * 6)

    before = metrics.report(crust.elevation, g.neighbours, g.arc_dist, g.area,
                            is_sink=is_ocean, channel_threshold_km2=thr)
    res = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                        stage2.LemParams(dt_myr=0.5), n_steps=300)
    after = metrics.report(res.elevation, g.neighbours, g.arc_dist, g.area,
                           is_sink=is_ocean, channel_threshold_km2=thr)

    assert before["concavity_theta"] < 0.25, "noise baseline should be nearly flat"
    assert 0.35 < after["concavity_theta"] < 0.65, "erosion should produce real concavity"
    assert after["concavity_theta"] > before["concavity_theta"] + 0.2


def test_solver_is_stable_at_large_timesteps():
    """Braun-Willett is unconditionally stable; a 10x timestep must not blow up.

    An explicit scheme would diverge here, which is the entire reason for the implicit one.
    """
    g = build_goldberg(64)
    crust = stage1.generate_crust(g, seed=4)
    is_ocean = ~crust.land_mask
    uplift = np.where(crust.land_mask, 300.0, 0.0).astype(np.float32)

    coarse = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                           stage2.LemParams(dt_myr=5.0), n_steps=40)
    fine = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                         stage2.LemParams(dt_myr=0.5), n_steps=400)
    assert np.isfinite(coarse.elevation).all()
    land = crust.land_mask
    # Same total time, so the steady states should broadly agree.
    assert coarse.elevation[land].max() == pytest.approx(
        fine.elevation[land].max(), rel=0.25)


def test_uplift_and_erosion_balance_at_steady_state():
    g = build_goldberg(64)
    crust = stage1.generate_crust(g, seed=6)
    is_ocean = ~crust.land_mask
    uplift = np.where(crust.land_mask, 300.0, 0.0).astype(np.float32)

    res = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                        stage2.LemParams(dt_myr=0.5), n_steps=500)
    final_rate = res.history[-1][1]
    assert final_rate < 0.05 * 300.0, f"not converged: {final_rate:.1f} m/Myr vs U=300"


def test_ocean_bathymetry_is_preserved():
    g = build_goldberg(48)
    crust = stage1.generate_crust(g, seed=7)
    is_ocean = ~crust.land_mask
    uplift = np.where(crust.land_mask, 300.0, 0.0).astype(np.float32)
    z0 = crust.elevation.copy()

    res = stage2.evolve(g, z0, uplift, is_ocean, stage2.LemParams(), n_steps=20)
    assert np.array_equal(res.elevation[is_ocean], z0[is_ocean])


def test_land_grades_to_sea_level_not_to_the_sea_floor():
    """Rivers must stop at sea level. Grading to the abyssal plain drowns the continents.

    The ocean floor sits ~4 km below sea level. If ocean cells present that depth to the
    land draining into them, coastal rivers keep incising past zero and cratons erode away
    entirely -- measured land minimum -1371 m, with only orogen belts left above water.
    """
    g = build_goldberg(64)
    crust = stage1.generate_crust(g, seed=11)
    is_ocean = ~crust.land_mask
    land = crust.land_mask
    # Low uplift, long run: the case where over-incision shows up most clearly.
    uplift = np.where(land, 30.0, 0.0).astype(np.float32)

    res = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                        stage2.LemParams(dt_myr=1.0), n_steps=200)

    drowned = (res.elevation[land] < 0).mean()
    assert drowned < 0.10, f"{drowned:.1%} of land eroded below sea level"
    assert res.elevation[land].min() > -100.0


def test_no_cell_incises_below_its_receiver():
    """Cutting below the receiver would invert the flow direction and corrupt the network."""
    g = build_goldberg(64)
    crust = stage1.generate_crust(g, seed=2)
    is_ocean = ~crust.land_mask
    uplift = stage2.uplift_field(g, crust.land_mask, seed=2)

    res = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                        stage2.LemParams(dt_myr=2.0), n_steps=60)
    fr = flow.route(res.elevation, g.neighbours, g.arc_dist, g.area, is_ocean)
    moving = fr.rec != np.arange(g.cell_count)
    drop = res.elevation[moving] - res.elevation[fr.rec[moving]]
    assert np.all(drop >= -1e-3), f"minimum drop to receiver was {drop.min():.4f} m"


# --------------------------------------------------------------------------------------
# Diffusion
# --------------------------------------------------------------------------------------


def test_diffusion_conserves_mass_and_smooths():
    g = build_goldberg(48)
    rng = np.random.default_rng(0)
    z = rng.standard_normal(g.cell_count) * 100.0
    not_fixed = np.zeros(g.cell_count, dtype=np.bool_)
    dz = np.empty(g.cell_count)

    # Diffusion is explicit and therefore CONDITIONALLY stable, unlike the implicit stream
    # power solve: it needs D*dt*sum(e/L) < A/2. Pick D*dt an order of magnitude inside that.
    d_dt = 0.05 * float(g.area.min()) * 1e6 / 6.0

    before_var = z.var()
    total_before = float((z * g.area).sum())
    for _ in range(20):
        stage2.hillslope_diffusion(z, g.neighbours, g.edge_len, g.arc_dist, g.area,
                                   d_dt, 1.0, not_fixed, dz)
        z = z + dz

    assert np.isfinite(z).all()
    assert z.var() < before_var, "diffusion must reduce roughness"
    assert float((z * g.area).sum()) == pytest.approx(total_before, rel=1e-6), \
        "the finite-volume Laplacian must conserve mass on a closed sphere"


def test_diffusion_blows_up_past_the_stability_limit():
    """Guard the limit explicitly, so the constraint is documented by a failing case."""
    g = build_goldberg(48)
    rng = np.random.default_rng(0)
    z = rng.standard_normal(g.cell_count) * 100.0
    not_fixed = np.zeros(g.cell_count, dtype=np.bool_)
    dz = np.empty(g.cell_count)

    d_dt = 20.0 * float(g.area.min()) * 1e6 / 6.0  # far past the limit
    for _ in range(20):
        stage2.hillslope_diffusion(z, g.neighbours, g.edge_len, g.arc_dist, g.area,
                                   d_dt, 1.0, not_fixed, dz)
        z = z + dz
    assert z.var() > 1e6 * 100.0**2, "expected divergence past the explicit stability limit"


def test_diffusion_does_almost_no_direct_work_at_continental_resolution():
    """Documents a real limitation rather than pretending the term does something.

    Hillslope diffusion sets the convex-hilltop-to-concave-valley transition, which occurs
    over tens to hundreds of metres. At ~10 km cells that is far below the grid scale, so at
    the default D the term moves almost nothing and fluvial incision does all the work.

    Measured DIRECTLY, as one step applied to evolved terrain. Comparing two full runs
    instead would conflate this with drainage reorganisation -- see the chaos test below.
    """
    g = build_goldberg(64)
    crust = stage1.generate_crust(g, seed=3)
    is_ocean = ~crust.land_mask
    uplift = np.where(crust.land_mask, 300.0, 0.0).astype(np.float32)
    res = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                        stage2.LemParams(D=0.0), n_steps=100)

    z = res.elevation.astype(np.float64)
    dz = np.empty(g.cell_count)
    stage2.hillslope_diffusion(z, g.neighbours, g.edge_len, g.arc_dist, g.area,
                               0.05, 0.5e6, np.ascontiguousarray(is_ocean), dz)

    land = crust.land_mask
    relief = float(z[land].max() - z[land].min())
    assert np.abs(dz[land]).max() < 1e-3 * relief, (
        f"diffusion moved up to {np.abs(dz[land]).max():.4f} m per step against "
        f"{relief:.0f} m of relief; if this grows the docstring needs revisiting")


def test_transient_perturbations_decay_away():
    """A one-off 1 cm nudge leaves essentially no trace, and less of one the longer you run.

    Worth pinning down, because the intuition that landscape evolution is chaotic is only
    half right. The implicit solve contracts deviations by 1/(1+F) every step, so a transient
    shrinks geometrically and disappears into float32 rounding. Measured here: ~0.018 m of
    residual after 25 steps against 4 km of relief, and bit-identical output by 100 steps for
    most perturbations. The steady state is a strong attractor.

    What the model IS sensitive to is *persistent* forcing, which accumulates until it flips
    a receiver at a divide and reassigns a whole sub-basin -- see the diffusion test. So the
    determinism concern for parallel reductions still stands: those bias every step, they
    are not a one-off.
    """
    g = build_goldberg(64)
    crust = stage1.generate_crust(g, seed=3)
    is_ocean = ~crust.land_mask
    uplift = np.where(crust.land_mask, 300.0, 0.0).astype(np.float32)
    p = stage2.LemParams(D=0.0)
    land = crust.land_mask

    rng = np.random.default_rng(0)
    nudged_z = crust.elevation.copy()
    nudged_z[land] += (rng.standard_normal(int(land.sum())) * 0.01).astype(np.float32)

    residual = {}
    for steps in (25, 150):
        base = stage2.evolve(g, crust.elevation, uplift, is_ocean, p, n_steps=steps)
        nudged = stage2.evolve(g, nudged_z, uplift, is_ocean, p, n_steps=steps)
        delta = np.abs(base.elevation[land] - nudged.elevation[land])
        relief = float(base.elevation[land].max() - base.elevation[land].min())
        residual[steps] = (float(delta.max()), relief)

    for steps, (worst, relief) in residual.items():
        assert worst < 1e-4 * relief, (
            f"after {steps} steps a 1 cm nudge still shows {worst:.3f} m "
            f"against {relief:.0f} m of relief")
    assert residual[150][0] <= residual[25][0], "the transient must keep decaying"


def test_persistent_forcing_reorganises_divides():
    """Diffusion moves <1.5 m directly over 100 steps, yet shifts some cells by hundreds.

    The gap is the point. A term too small to matter pointwise still biases every step in
    the same direction, and once it tips a receiver choice at a divide the affected
    sub-basin re-routes wholesale. Measured: median difference 0.014 m, maximum 720 m.
    """
    g = build_goldberg(64)
    crust = stage1.generate_crust(g, seed=3)
    is_ocean = ~crust.land_mask
    uplift = np.where(crust.land_mask, 300.0, 0.0).astype(np.float32)
    land = crust.land_mask

    without = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                            stage2.LemParams(D=0.0), n_steps=100)
    with_d = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                           stage2.LemParams(D=0.05), n_steps=100)

    delta = np.abs(with_d.elevation[land] - without.elevation[land])
    relief = float(without.elevation[land].max() - without.elevation[land].min())
    assert np.median(delta) < 0.001 * relief, "the typical cell should barely move"
    assert delta.max() > 0.05 * relief, "expected divide reorganisation somewhere"


def test_same_seed_reproduces_bit_for_bit():
    """The flip side of the chaos above: identical inputs must give identical output."""
    g = build_goldberg(48)
    crust = stage1.generate_crust(g, seed=9)
    is_ocean = ~crust.land_mask
    uplift = stage2.uplift_field(g, crust.land_mask, seed=9)
    p = stage2.LemParams(dt_myr=1.0)

    a = stage2.evolve(g, crust.elevation, uplift, is_ocean, p, n_steps=60)
    b = stage2.evolve(g, crust.elevation, uplift, is_ocean, p, n_steps=60)
    assert np.array_equal(a.elevation, b.elevation)
