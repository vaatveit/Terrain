"""Landscape evolution. The steady-state relation gives exact answers to check against.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import flow, metrics, stage1, stage2  # noqa: E402
from terrain.grid import R_EARTH_KM, build_goldberg  # noqa: E402


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

    PINNED TO EARTH RADIUS on purpose. theta = m/n is a property of the SOLVER, not of the
    world, and recovering it needs three or four decades of drainage area. The 100 m world
    at n=96 is 792 km^2 in total, so its largest basin is ~2000 cells and the regression
    has barely two decades to work with -- it reads 0.30 for a true 0.45 and would be
    measuring the fixture rather than the code. Tests of this world's CONFIGURATION use
    the default radius; tests of solver correctness use whatever has the dynamic range.
    """
    g = build_goldberg(96, radius_km=R_EARTH_KM)
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
    """The whole point of Stage 2: noise-only terrain has almost no concavity.

    Earth radius, for the dynamic-range reason in test_recovered_concavity_equals_m_over_n.
    """
    g = build_goldberg(96, radius_km=R_EARTH_KM)
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
    Earth radius: this is a claim about the solver, not about this world.
    """
    g = build_goldberg(64, radius_km=R_EARTH_KM)
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
    no_lake = np.zeros(g.cell_count, dtype=np.bool_)
    dz = np.empty(g.cell_count)

    # Diffusion is explicit and therefore CONDITIONALLY stable, unlike the implicit stream
    # power solve: it needs D*dt*sum(e/L) < A/2. Pick D*dt an order of magnitude inside that.
    d_dt = 0.05 * float(g.area.min()) * 1e6 / 6.0

    before_var = z.var()
    total_before = float((z * g.area).sum())
    for _ in range(20):
        stage2.hillslope_diffusion(z, g.neighbours, g.edge_len, g.arc_dist, g.area,
                                   d_dt, 1.0, not_fixed, no_lake, dz)
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
    no_lake = np.zeros(g.cell_count, dtype=np.bool_)
    dz = np.empty(g.cell_count)

    d_dt = 20.0 * float(g.area.min()) * 1e6 / 6.0  # far past the limit
    for _ in range(20):
        stage2.hillslope_diffusion(z, g.neighbours, g.edge_len, g.arc_dist, g.area,
                                   d_dt, 1.0, not_fixed, no_lake, dz)
        z = z + dz
    assert z.var() > 1e6 * 100.0**2, "expected divergence past the explicit stability limit"


def test_diffusion_does_real_work_at_100_m_cells_but_must_be_sub_cycled():
    """The whole reason for the 100 m grid, and the cost that comes with it.

    At 9.77 km cells the hillslope term moved at most 0.014 m per step against 4 km of
    relief: the convex-hilltop-to-concave-valley transition it models happens over tens to
    hundreds of metres, two orders of magnitude below the grid, so `D` was a numerical
    smoother rather than soil creep.

    The catch is that the explicit stability limit is A_i / (2 D sum_j(e_ij/L_ij)), and
    sum_j(e_ij/L_ij) is dimensionless, so the limit tracks CELL AREA and falls as the
    square of cell size -- from 239 Myr at 9.77 km to 25 kyr at 100 m, while the fluvial
    timestep stayed at 0.5 Myr. Both halves are asserted here: the raw call at a fluvial
    dt diverges, and the sub-cycled path is both stable and no longer negligible.

    Measured DIRECTLY, as one 0.5 Myr interval applied to evolved terrain. Comparing two
    full runs would conflate this with drainage reorganisation -- see the chaos test below.
    """
    g = build_goldberg(64)
    crust = stage1.generate_crust(g, seed=3)
    is_ocean = ~crust.land_mask
    fixed = np.ascontiguousarray(is_ocean)
    uplift = np.where(crust.land_mask, 100.0, 0.0).astype(np.float32)
    res = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                        stage2.LemParams(D=0.0), n_steps=100)

    land = crust.land_mask
    z = res.elevation.astype(np.float64)
    # Ocean back to base level, as `evolve` holds it during the run. `res.elevation` has
    # the true bathymetry restored, so a coastal cell would otherwise face a 2 km drop
    # across 100 m and diffuse against that rather than against its own hillslope.
    z[is_ocean] = 0.0
    relief = float(z[land].max() - z[land].min())
    dz = np.empty(g.cell_count)
    no_lake = np.zeros(g.cell_count, dtype=np.bool_)

    n_sub = stage2.diffusion_substeps(g.neighbours, g.edge_len, g.arc_dist, g.area,
                                      0.05, 0.5e6)
    assert n_sub > 1, "if diffusion no longer needs sub-cycling, the grid scale changed"

    # Stability is checked by the DISCRETE MAXIMUM PRINCIPLE, not by an arbitrary
    # threshold: pure diffusion averages a cell toward its neighbours, so a stable scheme
    # can never carry a cell outside the range the field already occupied. Violating that
    # is exactly what "unstable" means here.
    lo, hi = float(z.min()), float(z.max())

    stage2.hillslope_diffusion(z, g.neighbours, g.edge_len, g.arc_dist, g.area,
                               0.02, 0.5e6, fixed, no_lake, dz)
    raw = z + dz
    assert raw.min() < lo - 1.0 or raw.max() > hi + 1.0, (
        "a fluvial dt in one explicit step should violate the maximum principle")

    zs = z.copy()
    for _ in range(n_sub):
        stage2.hillslope_diffusion(zs, g.neighbours, g.edge_len, g.arc_dist, g.area,
                                   0.02, 0.5e6 / n_sub, fixed, no_lake, dz)
        zs += dz
    assert zs.min() >= lo - 1e-6 and zs.max() <= hi + 1e-6, (
        "sub-cycled diffusion must respect the maximum principle")

    moved = np.abs(zs - z)[land]
    assert moved.max() > 1e-3 * relief, (
        f"diffusion moved only {moved.max():.4f} m against {relief:.0f} m of relief; at "
        f"100 m cells it should no longer be negligible")


def test_a_tiny_nudge_permanently_reorganises_some_divides():
    """A one-off 1 cm nudge leaves a PERMANENT mark on a small fraction of the world.

    Measured: 99.2% of land ends up bit-identical, but 0.8% of cells differ by up to 433 m,
    and that difference is fixed from step 100 onward rather than decaying. Where the
    implicit solve contracts a perturbation smoothly, it dies; where it tips a receiver
    choice at a drainage divide, the affected sub-basin re-routes and never comes back.

    An earlier version of this suite asserted the opposite -- that transients decay to
    bit-identical output -- and passed, because the sub-seed bug meant every process
    generated a different world and that one happened to be smoothly contracting. The
    lesson generalises: reproducibility here means identical inputs give identical outputs
    (see the test below), NOT that near-identical inputs give near-identical worlds.
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

    changed = {}
    for steps in (300, 400):
        base = stage2.evolve(g, crust.elevation, uplift, is_ocean, p, n_steps=steps)
        nudged = stage2.evolve(g, nudged_z, uplift, is_ocean, p, n_steps=steps)
        delta = np.abs(base.elevation[land] - nudged.elevation[land])
        changed[steps] = (float((delta > 0).mean()), float(delta.max()))

    # 0.30 rather than the old 0.05, and the jump is a real result rather than a slipped
    # tolerance. Until inland basins were separated from the ocean, 40 closed depressions
    # scattered across the continents were classified as sea and therefore PINNED at base
    # level -- fixed-elevation anchors punched into the land. Removing them, on this same
    # crust: max flow path 1.84 -> 2.56 km, mean top-1% drainage area 0.503 -> 0.675 km^2,
    # and this figure 0.26% -> 17.85%. A longer, more integrated network has more divide to
    # flip, so the old stability was an artifact of the anchors and not a property of the
    # solver. What the test is actually for -- flips happen, and they stay -- is below.
    frac, worst = changed[400]
    assert frac < 0.30, f"{frac:.1%} of land moved; expected divide flips, not a reset"
    assert worst > 1.0, "expected at least one divide to flip and stay flipped"
    # Settled, not still decaying: the same cells differ by the same amount at 300 and 400.
    # The window was 200/300 until the basin change; the longer network takes longer to come
    # to rest, and 200 now lands inside the transient (18.18% / 911 m against 17.85% / 842).
    # From 300 the result is bit-identical at 400, 600, 900 and 1400.
    assert changed[400] == pytest.approx(changed[300], rel=1e-6)


def test_diffusion_reorganises_divides_far_beyond_where_it_acts():
    """Divide flips amplify diffusion far past its pointwise effect.

    At 9.77 km cells this test made a sharper claim: diffusion moved 0.014 m at the median
    yet 720 m at the maximum, a 50,000x gap, because a term far too small to matter
    pointwise still biased every step in the same direction until it tipped a receiver
    choice at a divide and re-routed a whole sub-basin.

    At 100 m cells the premise is gone -- diffusion is now a first-class term and the
    median cell moves properly, which is the point of the grid. What survives is the
    AMPLIFICATION: the worst cell still moves orders of magnitude more than the typical
    one, because divide reorganisation is a discrete event, not a scaled-up nudge.
    """
    g = build_goldberg(64)
    crust = stage1.generate_crust(g, seed=3)
    is_ocean = ~crust.land_mask
    uplift = np.where(crust.land_mask, 100.0, 0.0).astype(np.float32)
    land = crust.land_mask

    without = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                            stage2.LemParams(D=0.0), n_steps=100)
    with_d = stage2.evolve(g, crust.elevation, uplift, is_ocean,
                           stage2.LemParams(D=0.02), n_steps=100)

    delta = np.abs(with_d.elevation[land] - without.elevation[land])
    relief = float(without.elevation[land].max() - without.elevation[land].min())
    assert np.median(delta) > 1e-3 * relief, (
        "at 100 m cells diffusion should move the typical cell, not just the divides")
    # Measured 5.5x here, against ~50,000x at 9.77 km cells. The gap collapsed because
    # diffusion now acts everywhere rather than only where it happens to tip a divide,
    # so the tail no longer stands out the way it did. The tail is still there.
    assert delta.max() > 3.0 * np.median(delta), (
        "expected divide reorganisation to stand out above the pointwise effect")
    assert delta.max() < relief, "but not to rewrite the whole landscape"


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


# --------------------------------------------------------------------------------------
# Closed basins: what the erosion loop is allowed to do to them
# --------------------------------------------------------------------------------------


def _depression_area(g, z, is_ocean):
    """Fraction of land the depression fill has to raise."""
    from terrain import hydrology
    zf = np.ascontiguousarray(z, dtype=np.float32)
    fr = flow.route(zf, np.ascontiguousarray(g.neighbours),
                    np.ascontiguousarray(g.arc_dist),
                    np.ascontiguousarray(g.area.astype(np.float32)), is_ocean,
                    metrics=False)
    closed = fr.filled > zf + np.float32(1e-3)
    a = g.area.astype(np.float64)
    return float(a[closed].sum() / a[~is_ocean].sum())


def test_closed_basins_survive_erosion():
    """The reason lakes existed only as a unit-test fixture.

    Two terms erase a closed depression, and each one alone is enough, so exempting either
    on its own looks like it barely helps. The stream-power update snaps any cell below its
    receiver straight up to it -- one step, not a slow infill -- and hillslope diffusion
    pulls the rim down into the hole. Measured on the Stage 1 crust at n=128, closed
    depressions are 19.6% of land before erosion and after 200 steps: 0.002% with both
    terms, 0.854% exempting diffusion only, 0.002% exempting the clamp only, 3.240%
    exempting both. Stage 6 was finding two lakes on a whole world.

    The comparison run here disables the exemption by passing an all-false mask through the
    same kernels, so nothing but the exemption differs.
    """
    g = build_goldberg(64)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    is_ocean = ~crust.land_mask

    kept = stage2.evolve(g, crust.elevation, crust.uplift, is_ocean, n_steps=120)
    with_exemption = _depression_area(g, kept.elevation, is_ocean)

    real_diffuse, real_solve = stage2.hillslope_diffusion, stage2.solve_stream_power
    try:
        stage2.hillslope_diffusion = (
            lambda z, nb, e, a, ar, D, dt, fx, il, dz:
            real_diffuse(z, nb, e, a, ar, D, dt, fx, np.zeros_like(il), dz))
        stage2.solve_stream_power = (
            lambda z, o, r, s, ac, u, dt, K, m, n, fx, il, ni, out:
            real_solve(z, o, r, s, ac, u, dt, K, m, n, fx, np.zeros_like(il), ni, out))
        erased = stage2.evolve(g, crust.elevation, crust.uplift, is_ocean, n_steps=120)
    finally:
        stage2.hillslope_diffusion, stage2.solve_stream_power = real_diffuse, real_solve
    without = _depression_area(g, erased.elevation, is_ocean)

    assert with_exemption > 10.0 * max(without, 1e-6), (
        f"closed basins are {with_exemption:.4%} of land with the exemption and "
        f"{without:.4%} without -- the exemption is not doing anything")
    assert with_exemption > 0.002, (
        f"only {with_exemption:.4%} of land holds a closed basin; Earth's lake cover is ~1%")


def test_diffusion_conserves_mass_around_a_lake():
    """The lake mask is a no-flux boundary, not a sink.

    An edge is skipped whenever EITHER end is in a lake, and that test is symmetric, so the
    cells outside still conserve mass exactly among themselves. Getting this wrong -- zeroing
    dz inside the lake but letting the rim keep its flux -- leaks mass across the boundary
    and would quietly drain or inflate the world.
    """
    g = build_goldberg(48)
    rng = np.random.default_rng(4)
    z = rng.standard_normal(g.cell_count) * 100.0
    not_fixed = np.zeros(g.cell_count, dtype=np.bool_)
    in_lake = np.zeros(g.cell_count, dtype=np.bool_)
    in_lake[rng.choice(g.cell_count, g.cell_count // 10, replace=False)] = True
    dz = np.empty(g.cell_count)

    outside = ~in_lake
    before = float((z[outside] * g.area[outside]).sum())
    d_dt = 0.05 * float(g.area.min()) * 1e6 / 6.0
    for _ in range(20):
        stage2.hillslope_diffusion(z, g.neighbours, g.edge_len, g.arc_dist, g.area,
                                   d_dt, 1.0, not_fixed, in_lake, dz)
        z = z + dz
    assert (dz[in_lake] == 0.0).all(), "lake cells must not diffuse"
    assert float((z[outside] * g.area[outside]).sum()) == pytest.approx(before, rel=1e-6)


def test_subsidence_is_bounded_by_a_total_not_run_as_a_rate():
    """Erosion balances uplift; nothing balances subsidence once the clamp is gone.

    Deposition is off, so a graben subsiding at a constant rate just keeps going: 16.9
    m/Myr over a 300 Myr run is 5,063 m, and Stage 6 reported a lake 5,265 m deep on a
    world whose highest peak was 361 m. Rift subsidence is finite in nature -- McKenzie
    stretching gives a basin a total, not a perpetual rate -- so the bound is a total.
    """
    g = build_goldberg(48)
    crust = stage1.generate_crust(g, seed=5)
    is_ocean = ~crust.land_mask
    # One patch of pure subsidence, nothing else, so the only thing moving z is the rift.
    uplift = np.zeros(g.cell_count, dtype=np.float32)
    sinking = crust.land_mask & (g.lat > 0)
    uplift[sinking] = -50.0
    p = stage2.LemParams(K=0.0, D=0.0, max_subsidence_m=200.0)

    z = stage2.evolve(g, crust.elevation, uplift, is_ocean, p, n_steps=100).elevation
    drop = (crust.elevation - z)[sinking]
    # 50 m/Myr x 100 steps x 0.5 Myr = 2,500 m if unbounded.
    assert drop.max() == pytest.approx(200.0, abs=1.0), (
        f"deepest subsidence {drop.max():.0f} m against a 200 m budget")
