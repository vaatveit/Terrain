"""Validation for the Goldberg grid. Run before anything is built on top of it.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import grid as grid_mod  # noqa: E402
from terrain.grid import R_EARTH_KM, build_goldberg  # noqa: E402


@pytest.fixture(scope="module")
def g():
    return build_goldberg(48)


def test_cell_count(g):
    assert g.cell_count == 10 * g.n**2 + 2


def test_degree_distribution(g):
    deg = (g.neighbours >= 0).sum(axis=1)
    assert int((deg == 5).sum()) == 12, "exactly 12 pentagons"
    assert int((deg == 6).sum()) == g.cell_count - 12
    # The unused pentagon slot must be last, so operators can stop at the first -1.
    pent = np.where(deg == 5)[0]
    assert np.all(g.neighbours[pent, 5] == -1)


def test_neighbours_symmetric_and_distinct(g):
    for i in range(0, g.cell_count, 37):
        nb = [x for x in g.neighbours[i] if x >= 0]
        assert len(set(nb)) == len(nb), f"duplicate neighbour at cell {i}"
        assert i not in nb, f"cell {i} is its own neighbour"
        for j in nb:
            assert i in g.neighbours[j], f"{i}->{j} not mirrored"


def test_areas_sum_to_sphere(g):
    sphere = 4 * np.pi * g.radius_km**2
    assert abs(float(g.area.sum()) - sphere) / sphere < 1e-6


def test_area_uniformity(g):
    a = g.area
    # The requirement is "approximately equal area"; slerp subdivision gets well inside it.
    assert a.std() / a.mean() < 0.05, "cell area sigma/mean must be under 5%"
    assert a.max() / a.min() < 1.25


def _hex_spacing_km(n, radius_km=R_EARTH_KM):
    """Centre-to-centre spacing implied by packing 10n^2+2 hexagons onto the sphere."""
    cell_area = 4 * np.pi * radius_km**2 / (10 * n**2 + 2)
    return np.sqrt(2 * cell_area / np.sqrt(3))  # area of a hexagon = (sqrt(3)/2) d^2


def test_spacing_matches_hex_packing(g):
    expect = _hex_spacing_km(g.n, g.radius_km)
    d = g.arc_dist[g.neighbours >= 0]
    assert abs(d.mean() - expect) / expect < 0.01


def test_every_grid_has_100_m_cells_whatever_its_frequency(g):
    """Cell size is the invariant of this world, not the radius.

    Every tuned length scale in the pipeline -- lake and island minima, the channel-head
    discharge, the diffusion sub-step count -- is denominated against 100 m cells, so a
    fixture at low n has to shrink the sphere rather than coarsen the grid. If this ever
    fails, every one of those parameters is wrong for the grid under test.
    """
    d = g.arc_dist[g.neighbours >= 0]
    assert d.mean() == pytest.approx(grid_mod.TARGET_CELL_KM, rel=1e-3)


def test_the_production_world_meets_its_spec():
    """n=785 at 100 m cells. Checked analytically, not by building 6.2M cells.

    Three specs were asked for -- 100 m cells, 4096 cells around the equator, and
    R = 65.19 km -- and the 12 pentagons mean they cannot all hold at once. Cell size is
    the one held exact; the other two come out 0.4% off. This pins that trade so it stays
    a decision rather than drift.
    """
    n = grid_mod.N_PRODUCTION
    assert n == 785
    assert 10 * n**2 + 2 == 6_162_252
    r = grid_mod.radius_for_cell_km(n)
    assert r == pytest.approx(64.927, abs=0.01)
    around_equator = 2 * np.pi * r / grid_mod.TARGET_CELL_KM
    assert around_equator == pytest.approx(grid_mod.EQUATOR_CELLS, rel=0.005)


def test_positions_are_unit_vectors(g):
    assert np.allclose(np.linalg.norm(g.xyz, axis=1), 1.0, atol=1e-12)


def test_latlon_roundtrip(g):
    lat = np.radians(g.lat.astype(np.float64))
    lon = np.radians(g.lon.astype(np.float64))
    rebuilt = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1
    )
    assert np.max(np.abs(rebuilt - g.xyz)) < 1e-6


def test_no_duplicate_cell_positions(g):
    """Catches a broken edge/corner id mapping, which would fuse two cells into one."""
    key = np.round(g.xyz, 9)
    uniq = np.unique(key, axis=0)
    assert uniq.shape[0] == g.cell_count


def test_connectivity_matches_scipy_delaunay():
    """The analytic stitching must agree with a real triangulation.

    For points on a sphere the convex hull IS the Delaunay triangulation, so scipy gives
    an independent ground truth. Done at low n because Qhull on millions of points is
    slow and memory-hungry -- which is exactly why the production path is analytic.
    """
    g = build_goldberg(12)
    hull = ConvexHull(g.xyz)

    truth = set()
    for tri in hull.simplices:
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            truth.add((min(u, v), max(u, v)))

    ours = set()
    for i in range(g.cell_count):
        for j in g.neighbours[i]:
            if j >= 0:
                ours.add((min(i, int(j)), max(i, int(j))))

    assert ours == truth, (
        f"{len(ours - truth)} spurious and {len(truth - ours)} missing edges"
    )


def test_edge_count_is_eulerian(g):
    """A sphere triangulation has exactly 3V-6 edges; anything else means bad stitching."""
    n_edges = int((g.neighbours >= 0).sum()) // 2
    assert n_edges == 3 * g.cell_count - 6


def test_voronoi_edges_are_reciprocal(g):
    """The shared edge between i and j must have one length, whichever side you ask."""
    for i in range(0, g.cell_count, 53):
        for s, j in enumerate(g.neighbours[i]):
            if j < 0:
                continue
            back = int(np.where(g.neighbours[j] == i)[0][0])
            assert g.edge_len[i, s] == pytest.approx(g.edge_len[j, back], rel=1e-4)
            assert g.arc_dist[i, s] == pytest.approx(g.arc_dist[j, back], rel=1e-5)


def test_spherical_not_planar_distance(g):
    """Regression guard: anything treating lon/lat as planar coordinates fails this.

    Two cells one degree apart in longitude are half as far apart at 60 degrees north as
    they are on the equator. A sqrt(dlon^2 + dlat^2) implementation gets the same answer
    for both.
    """
    def gc(a, b):
        return g.radius_km * np.arccos(np.clip(np.dot(a, b), -1, 1))

    def unit(lat_deg, lon_deg):
        la, lo = np.radians(lat_deg), np.radians(lon_deg)
        return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])

    at_eq = gc(unit(0, 0), unit(0, 1))
    at_60 = gc(unit(60, 0), unit(60, 1))
    assert at_60 / at_eq == pytest.approx(0.5, rel=1e-3)


def test_great_circle_walk_crosses_pentagon_smoothly(g):
    """Step lengths must not blow up when a path passes one of the 12 pentagons."""
    deg = (g.neighbours >= 0).sum(axis=1)
    start = int(np.where(deg == 5)[0][0])
    target = -g.xyz[start]  # walk to the antipode, straight over the pentagon

    cur, steps = start, []
    for _ in range(4 * g.n):
        nb = [x for x in g.neighbours[cur] if x >= 0]
        best = max(nb, key=lambda j: float(np.dot(g.xyz[j], target)))
        if np.dot(g.xyz[best], target) <= np.dot(g.xyz[cur], target):
            break
        s = int(np.where(g.neighbours[cur] == best)[0][0])
        steps.append(float(g.arc_dist[cur, s]))
        cur = best

    steps = np.array(steps)
    assert len(steps) > g.n // 2, "walk terminated early"
    assert steps.max() / steps.min() < 1.6, "step length discontinuity near a pentagon"
