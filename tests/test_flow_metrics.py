"""Flow routing and metrics, checked against constructed cases with known answers.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import flow, metrics  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


# --------------------------------------------------------------------------------------
# A small planar lattice with an explicit graph, so expected answers are hand-computable.
# --------------------------------------------------------------------------------------


def lattice(h, w, spacing_km=1.0, cell_area_km2=1.0):
    n = h * w
    rows, cols = np.divmod(np.arange(n), w)
    offs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    nbr = np.full((n, 4), -1, dtype=np.int32)
    for s, (dr, dc) in enumerate(offs):
        r2, c2 = rows + dr, cols + dc
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        nbr[ok, s] = (r2[ok] * w + c2[ok]).astype(np.int32)
    arc = np.full((n, 4), spacing_km, dtype=np.float32)
    area = np.full(n, cell_area_km2, dtype=np.float32)
    return nbr, arc, area


def test_accumulation_conserves_area():
    """Every cell's area must arrive at exactly one outlet, and none may be invented."""
    h = w = 40
    nbr, arc, area = lattice(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    z = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2).astype(np.float32)  # bowl -> drains inward
    is_sink = np.zeros(h * w, bool)
    is_sink[(h // 2) * w + w // 2] = True

    fr = flow.route(z.ravel(), nbr, arc, area, is_sink)
    outlets = fr.rec == np.arange(h * w)
    assert fr.acc[outlets].sum() == pytest.approx(area.sum(), rel=1e-4)
    assert fr.acc[(h // 2) * w + w // 2] == pytest.approx(area.sum(), rel=1e-4)


def test_receivers_follow_the_gradient():
    """On a plane tilted along +x, every cell must drain to its -x neighbour."""
    h = w = 20
    nbr, arc, area = lattice(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    z = (100.0 - xx * 5.0).astype(np.float32)
    is_sink = np.zeros(h * w, bool)
    is_sink[np.arange(h) * w + (w - 1)] = True

    fr = flow.route(z.ravel(), nbr, arc, area, is_sink)
    interior = np.ones(h * w, bool)
    interior[np.arange(h) * w + (w - 1)] = False
    assert np.all(fr.rec[interior] == np.arange(h * w)[interior] + 1)


def test_priority_flood_fills_a_pit_and_never_lowers():
    h = w = 21
    nbr, arc, area = lattice(h, w)
    z = np.full((h, w), 100.0, dtype=np.float32)
    z[10, 10] = 50.0  # isolated pit
    is_sink = np.zeros(h * w, bool)
    is_sink[0] = True

    fr = flow.route(z.ravel(), nbr, arc, area, is_sink)
    assert np.all(fr.filled >= z.ravel() - 1e-4), "fill lowered terrain"
    assert fr.filled.reshape(h, w)[10, 10] > 99.0, "pit was not filled"
    assert fr.raised[10 * w + 10]
    assert not fr.raised[0]


def test_topological_order_puts_receivers_first():
    h = w = 30
    nbr, arc, area = lattice(h, w)
    rng = np.random.default_rng(3)
    z = rng.random((h, w)).astype(np.float32) * 100
    is_sink = np.zeros(h * w, bool)
    is_sink[0] = True

    fr = flow.route(z.ravel(), nbr, arc, area, is_sink)
    position = np.empty(h * w, np.int64)
    position[fr.order] = np.arange(h * w)
    moving = fr.rec != np.arange(h * w)
    assert np.all(position[fr.rec[moving]] < position[np.arange(h * w)[moving]])


# --------------------------------------------------------------------------------------


def test_flow_path_slope_survives_elevation_quantisation():
    """Regression guard for the bug that put slope-area concavity at 1.87 instead of 0.5.

    SRTM stores int16 metres. On a gentle channel the drop across one 30 m cell rounds to
    zero, so the cell-to-cell gradient is 0 (or a meaningless +eps residual after filling)
    exactly where drainage area is largest. Measuring over a flow-path baseline recovers
    the true gradient.
    """
    n = 400
    spacing_km = 0.03  # 30 m cells
    true_slope = 0.004  # 4 m/km -- a real river gradient, under 1 m per cell
    z_exact = (1000.0 - np.arange(n) * true_slope * spacing_km * 1000.0).astype(np.float32)
    z = np.round(z_exact).astype(np.float32)  # the quantisation that breaks things

    nbr = np.full((n, 2), -1, dtype=np.int32)
    nbr[:-1, 0] = np.arange(1, n)
    nbr[1:, 1] = np.arange(n - 1)
    arc = np.full((n, 2), spacing_km, dtype=np.float32)
    area = np.ones(n, dtype=np.float32)
    is_sink = np.zeros(n, bool)
    is_sink[-1] = True

    fr = flow.route(z, nbr, arc, area, is_sink, slope_baseline_km=0.5)

    mid = slice(50, 350)
    cell_to_cell = fr.slope[mid]
    baseline = np.median(fr.ch_slope[mid])

    # Cell-to-cell is bimodal garbage: the true drop is 0.12 m per cell, so most cells
    # round to no drop at all (leaving only the fill's +eps residual) and roughly one in
    # eight carries a full 1 m step. Neither value is anywhere near the truth.
    near_truth = np.abs(cell_to_cell - true_slope) < 0.5 * true_slope
    assert near_truth.mean() < 0.05, "cell-to-cell slope should almost never be right"
    assert np.median(cell_to_cell) < 0.1 * true_slope, "median collapses to the eps residual"
    assert cell_to_cell.max() > 5 * true_slope, "and the rest overshoot by a whole metre"

    # The baseline gradient recovers the truth.
    assert baseline == pytest.approx(true_slope, rel=0.05)


def test_interior_mask_keeps_upstream_subbasins():
    """Direction check: testing the OUTLET instead of the catchment excludes everything.

    Border cells are the sinks, so every basin's outlet is on the border -- an
    outlet-based test returns an empty interior and silently kills Hack's law.
    """
    h = w = 60
    nbr, arc, area = lattice(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    # A cone draining outward. (A bowl would drain inward, become one giant filled
    # depression, and tell us nothing about catchment bookkeeping.)
    z = (200.0 - np.abs(xx - w / 2) * 3.0 - np.abs(yy - h / 2) * 3.0).astype(np.float32)

    is_border = np.zeros((h, w), bool)
    is_border[0, :] = is_border[-1, :] = True
    is_border[:, 0] = is_border[:, -1] = True
    is_border = is_border.ravel()

    fr = flow.route(z.ravel(), nbr, arc, area, is_border)
    valid = metrics.interior_mask(fr, is_border, nbr)
    assert valid.mean() > 0.5, "interior sub-basins were all excluded"
    assert not valid[is_border].any(), "border cells must never be interior"


def test_strahler_orders_a_known_confluence():
    """Two order-1 streams meeting must make order 2; one joining order 2 must not."""
    #  0 -> 2 <- 1  ;  2 -> 3 ; 4 -> 3  (4 is a lone tributary onto the order-2 trunk)
    rec = np.array([2, 2, 3, 3, 3], dtype=np.int32)
    order = np.array([3, 2, 4, 0, 1], dtype=np.int32)
    don_ptr = np.zeros(6, dtype=np.int64)
    donors = {0: [], 1: [], 2: [0, 1], 3: [2, 4], 4: []}
    flat = []
    for i in range(5):
        don_ptr[i] = len(flat)
        flat.extend(donors[i])
    don_ptr[5] = len(flat)
    don = np.array(flat, dtype=np.int32)

    is_channel = np.ones(5, dtype=bool)
    out = np.empty(5, dtype=np.int32)
    flow.strahler_order(order, don_ptr, don, is_channel, out)

    assert out[0] == 1 and out[1] == 1 and out[4] == 1
    assert out[2] == 2, "two order-1 tributaries must promote to order 2"
    assert out[3] == 2, "order 1 joining order 2 must not promote"


def test_hypsometric_integral_of_a_cone():
    """A cone has a known hypsometric integral of 1/3."""
    n = 500
    r = np.linspace(0, 1, n)
    z = (1.0 - r).astype(np.float32)
    area = (2 * np.pi * r).astype(np.float32)  # annulus area grows with radius
    _, _, hi = metrics.hypsometry(z, area, bins=200)
    assert hi == pytest.approx(1 / 3, abs=0.02)


# --------------------------------------------------------------------------------------


def test_metrics_run_on_the_spherical_grid():
    """The same code must run unmodified on the Goldberg tiling, pentagons and all."""
    g = build_goldberg(72)
    rng = np.random.default_rng(11)
    # Smooth field: a few random plane waves over the sphere.
    z = np.zeros(g.cell_count)
    for _ in range(24):
        k = rng.standard_normal(3)
        k /= np.linalg.norm(k)
        z += np.cos(4.0 * (g.xyz @ k) * np.pi + rng.uniform(0, 2 * np.pi))
    z = np.ascontiguousarray(z * 300.0, dtype=np.float32)
    is_ocean = z <= np.quantile(z, 0.3)

    r = metrics.report(
        z, g.neighbours, g.arc_dist, g.area, is_ocean,
        channel_threshold_km2=float(np.median(g.area) * 5), label="sphere",
    )
    assert r["cells"] == g.cell_count
    assert r["channel_cells"] > 0
    assert np.isfinite(r["concavity_theta"])
    assert np.isfinite(r["hack_h"])
    # Drainage area must never exceed the sphere.
    assert r["_flow"].acc.max() <= 4 * np.pi * g.radius_km**2 * 1.001
