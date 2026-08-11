"""Plate tectonics: Euler poles, boundary classification, derived uplift.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import plates, stage1  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


@pytest.fixture(scope="module")
def g():
    return build_goldberg(72)


@pytest.fixture(scope="module")
def pl(g):
    return plates.build_plates(g, plates.PlateParams(), seed=1)


def test_partition_covers_every_cell_exactly_once(g, pl):
    assert pl.plate_id.shape[0] == g.cell_count
    assert pl.plate_id.min() >= 0
    assert pl.plate_id.max() < pl.sites.shape[0]
    assert len(np.unique(pl.plate_id)) == pl.sites.shape[0], "a plate owns no cells"


def test_seeds_respect_the_minimum_separation():
    """Without Poisson-disk rejection, plate sizes come out lottery-distributed."""
    g = build_goldberg(48)
    p = plates.PlateParams(plate_count=14, min_separation_deg=18.0)
    sites = plates.seed_plates(g, p, seed=3)
    dots = sites @ sites.T
    np.fill_diagonal(dots, -1.0)
    min_angle = np.degrees(np.arccos(np.clip(dots.max(), -1, 1)))
    assert min_angle > 12.0, f"closest pair only {min_angle:.1f} deg apart"


def test_plate_sizes_are_not_wildly_uneven(g, pl):
    counts = np.bincount(pl.plate_id, minlength=pl.sites.shape[0])
    assert counts.max() / counts.min() < 12.0


def test_euler_velocity_is_tangent_to_the_sphere(g, pl):
    """Rotation about an Euler pole can have no radial component. A planar velocity field
    cannot satisfy this, which is why the sphere makes the model exact rather than a hack."""
    v = plates.plate_velocity(g.xyz, pl.euler_axis, pl.euler_rate, pl.plate_id)
    radial = np.einsum("ij,ij->i", v, g.xyz)
    assert np.abs(radial).max() < 1e-9 * max(np.abs(v).max(), 1.0)


def test_plate_speeds_are_earthlike(g, pl):
    v = plates.plate_velocity(g.xyz, pl.euler_axis, pl.euler_rate, pl.plate_id)
    speed = np.linalg.norm(v, axis=1)  # km/Myr == mm/yr
    assert 1.0 < speed.mean() < 120.0
    assert speed.max() < 200.0


def test_boundaries_lie_exactly_where_plates_meet(g, pl):
    is_b = pl.boundary_type != plates.NONE
    differs = np.zeros(g.cell_count, dtype=bool)
    for s in range(g.neighbours.shape[1]):
        j = g.neighbours[:, s]
        ok = j >= 0
        differs[ok] |= pl.plate_id[ok] != pl.plate_id[j[ok]]
    assert np.array_equal(is_b, differs)


def test_convergence_sign_is_right_way_round():
    """Two plates rotating oppositely about the same axis must converge somewhere.

    Sign errors here are silent and catastrophic: mountains would appear at rifts.
    """
    g = build_goldberg(48)
    pid = (g.xyz[:, 0] > 0).astype(np.int16)  # split the sphere at x = 0
    axis = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    rate = np.array([0.02, -0.02])  # counter-rotating
    vel = np.ascontiguousarray(plates.plate_velocity(g.xyz, axis, rate, pid))

    b_type = np.zeros(g.cell_count, np.uint8)
    b_stress = np.zeros(g.cell_count, np.float32)
    plates._classify_boundaries(np.ascontiguousarray(g.xyz),
                                np.ascontiguousarray(g.neighbours),
                                np.ascontiguousarray(pid), vel, 2.5, b_type, b_stress)

    boundary = b_type != plates.NONE
    assert boundary.any()
    assert (b_type[boundary] == plates.CONVERGENT).any(), "no convergence anywhere"
    assert (b_type[boundary] == plates.DIVERGENT).any(), "no divergence anywhere"
    # Convergent cells must carry positive normal stress, divergent negative.
    assert np.all(b_stress[b_type == plates.CONVERGENT] > 0)
    assert np.all(b_stress[b_type == plates.DIVERGENT] < 0)


def test_boundary_distance_is_zero_on_boundaries_and_grows_inward(g, pl):
    is_b = pl.boundary_type != plates.NONE
    d = pl.boundary_dist_km
    assert np.all(d[is_b] == 0.0)
    assert d[~is_b].min() > 0.0
    # Nothing should be closer than roughly one cell spacing off a boundary.
    spacing = float(np.median(g.arc_dist[g.neighbours >= 0]))
    assert d[~is_b].min() > 0.5 * spacing


def test_boundary_distance_respects_the_triangle_inequality(g, pl):
    """A cheap correctness check on the relaxation sweeps."""
    d = pl.boundary_dist_km
    for s in range(g.neighbours.shape[1]):
        j = g.neighbours[:, s]
        ok = (j >= 0) & (d < 1e3) & (d[np.maximum(j, 0)] < 1e3)
        assert np.all(d[ok] <= d[j[ok]] + g.arc_dist[ok, s] + 1e-3)


def test_boundary_mix_is_plausible(g, pl):
    """Earth is roughly 25% convergent, 55% divergent, 20% transform by length.

    The naive threshold (transform whenever tangential exceeds normal) gave 44% transform,
    because random Euler poles produce oblique motion far more often than a real plate
    system does.
    """
    b = pl.boundary_type[pl.boundary_type != plates.NONE]
    frac_transform = (b == plates.TRANSFORM).mean()
    assert 0.05 < frac_transform < 0.40, f"transform fraction {frac_transform:.0%}"
    assert (b == plates.CONVERGENT).any() and (b == plates.DIVERGENT).any()


def test_boundary_warp_bends_boundaries_without_breaking_the_partition(g):
    straight = plates.build_plates(g, plates.PlateParams(boundary_warp=0.0), seed=5)
    warped = plates.build_plates(g, plates.PlateParams(boundary_warp=0.10), seed=5)
    assert not np.array_equal(straight.plate_id, warped.plate_id)
    # Every cell still belongs to exactly one plate, and none has vanished.
    assert len(np.unique(warped.plate_id)) == warped.sites.shape[0]
    # Warping should lengthen the boundary network, not shorten it.
    assert (warped.boundary_type != plates.NONE).sum() > \
           (straight.boundary_type != plates.NONE).sum()


# --------------------------------------------------------------------------------------
# Uplift
# --------------------------------------------------------------------------------------


def test_uplift_is_high_at_convergent_margins_and_low_in_cratons(g, pl):
    is_cont = np.ones(g.cell_count, dtype=bool)
    land = np.ones(g.cell_count, dtype=bool)
    p = plates.PlateParams()
    u = plates.derive_uplift(pl, is_cont, land, p)

    near_conv = (pl.boundary_dist_km < 100) & (
        pl.boundary_type[np.maximum(pl.nearest_boundary, 0)] == plates.CONVERGENT)
    interior = pl.boundary_dist_km > 900
    assert near_conv.any() and interior.any()
    assert u[near_conv].mean() > 5 * u[interior].mean()
    assert u.max() <= p.u_max_m_per_myr + 1e-6, "shape function must be clamped before scaling"
    assert u[land].min() >= p.craton_floor_m_per_myr - 1e-6


def test_uplift_is_zero_over_ocean(g, pl):
    is_cont = np.ones(g.cell_count, dtype=bool)
    land = g.lat > 0  # arbitrary split
    u = plates.derive_uplift(pl, is_cont, land, plates.PlateParams())
    assert np.all(u[~land] == 0.0)


def test_contrast_cap_bounds_the_uplift_ratio(g, pl):
    is_cont = np.ones(g.cell_count, dtype=bool)
    land = np.ones(g.cell_count, dtype=bool)
    uncapped = plates.derive_uplift(pl, is_cont, land, plates.PlateParams())
    capped = plates.derive_uplift(pl, is_cont, land, plates.PlateParams(), contrast_cap=4.0)
    assert uncapped.max() / uncapped.min() > 10
    assert capped.max() / capped.min() == pytest.approx(4.0, rel=0.01)


# --------------------------------------------------------------------------------------
# Full Stage 1
# --------------------------------------------------------------------------------------


def test_tectonic_stage1_produces_the_declared_outputs(g):
    c = stage1.generate_crust_tectonic(g, seed=2)
    n = g.cell_count
    for name in ("crust_thickness", "crust_density", "elevation", "land_mask",
                 "plate_id", "boundary_type", "boundary_stress", "margin_class", "uplift"):
        assert getattr(c, name).shape[0] == n, f"{name} has the wrong length"
    assert c.land_fraction == pytest.approx(0.29, abs=0.01)
    assert np.all(c.crust_thickness > 0)
    assert np.all(c.elevation[c.land_mask] > 0)


def test_tectonic_thickening_builds_crustal_roots(g):
    """Orogens need a root; that is what lifts them isostatically."""
    c = stage1.generate_crust_tectonic(g, seed=2)
    conv = (c.boundary_type == plates.CONVERGENT)
    assert c.crust_thickness.max() > 45.0, "no crustal root formed"
    if conv.any():
        assert c.crust_thickness[conv].mean() > c.crust_thickness.mean()


def test_tectonic_crust_has_more_relief_than_the_throwaway(g):
    """The whole point of adding tectonics: mountains come from crustal thickening."""
    plain = stage1.generate_crust(g, seed=2)
    tect = stage1.generate_crust_tectonic(g, seed=2)
    assert tect.elevation.max() > 2 * plain.elevation.max()


def test_margins_are_only_assigned_on_the_coast(g):
    c = stage1.generate_crust_tectonic(g, seed=2)
    tagged = c.margin_class != plates.INLAND
    assert tagged.any()
    assert np.all(c.land_mask[tagged]), "a margin was tagged on an ocean cell"
    # Every tagged cell must actually touch water.
    touches = np.zeros(g.cell_count, dtype=bool)
    for s in range(g.neighbours.shape[1]):
        j = g.neighbours[:, s]
        ok = j >= 0
        touches[ok] |= ~c.land_mask[j[ok]]
    assert np.all(touches[tagged])


def test_tectonic_generation_is_reproducible(g):
    a = stage1.generate_crust_tectonic(g, seed=13)
    b = stage1.generate_crust_tectonic(g, seed=13)
    c = stage1.generate_crust_tectonic(g, seed=14)
    assert np.array_equal(a.elevation, b.elevation)
    assert np.array_equal(a.uplift, b.uplift)
    assert not np.array_equal(a.elevation, c.elevation)
