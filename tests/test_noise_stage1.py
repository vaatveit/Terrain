"""Noise and throwaway Stage 1.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import noise, stage1  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


@pytest.fixture(scope="module")
def g():
    return build_goldberg(48)


# --------------------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------------------


def _sample(n=4000, seed=5, scale=4.0):
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 3))
    return np.ascontiguousarray(p / np.linalg.norm(p, axis=1, keepdims=True) * scale)


def test_simplex_is_bounded_and_centred():
    pts = _sample()
    perm = noise.permutation(1)
    v = np.array([noise.simplex3(*p, perm, noise.GRAD3) for p in pts])
    assert np.all(np.abs(v) <= 1.2), "simplex noise escaped its nominal range"
    assert abs(v.mean()) < 0.1, "noise should be roughly zero-mean"
    assert v.std() > 0.15, "noise is suspiciously flat"


def test_noise_is_deterministic_and_seed_dependent():
    pts = _sample()
    out = np.empty(pts.shape[0])
    a = noise.fbm(pts, noise.permutation(7), noise.GRAD3, 4, 2.0, 2.0, 0.5, out).copy()
    b = noise.fbm(pts, noise.permutation(7), noise.GRAD3, 4, 2.0, 2.0, 0.5, out).copy()
    c = noise.fbm(pts, noise.permutation(8), noise.GRAD3, 4, 2.0, 2.0, 0.5, out).copy()
    assert np.array_equal(a, b), "same seed must give identical output"
    assert not np.allclose(a, c), "different seeds must give different output"


def test_noise_is_continuous_on_the_sphere():
    """Nearby points must give nearby values -- no seams, no pole pinching.

    This is the payoff for sampling 3D at unit vectors rather than 2D over (lon, lat):
    continuity is automatic, including across the antimeridian and at both poles.
    """
    perm = noise.permutation(3)
    rng = np.random.default_rng(0)
    base = rng.standard_normal((2000, 3))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    direction = rng.standard_normal((2000, 3))

    out = np.empty(2000)

    def max_change(eps):
        near = base + direction * eps
        near /= np.linalg.norm(near, axis=1, keepdims=True)
        a = noise.fbm(np.ascontiguousarray(base * 3), perm, noise.GRAD3,
                      4, 2.0, 2.0, 0.5, out).copy()
        b = noise.fbm(np.ascontiguousarray(near * 3), perm, noise.GRAD3,
                      4, 2.0, 2.0, 0.5, out).copy()
        return float(np.max(np.abs(b - a)))

    # An absolute tolerance would be arbitrary -- the field is Lipschitz, so the honest
    # test is that the response SCALES with the displacement. A seam or a pole singularity
    # would keep the maximum change large however small the step gets.
    coarse = max_change(4e-3)
    fine = max_change(1e-3)
    assert fine < coarse, "shrinking the step must shrink the change"
    assert fine < 0.4 * coarse, f"response is not scaling with displacement ({fine:.4f} vs {coarse:.4f})"


def test_antimeridian_and_poles_have_no_seam():
    """lon=+180 and lon=-180 are the same point in 3D, so they cannot disagree."""
    perm = noise.permutation(4)
    lat = np.radians(np.linspace(-89.9, 89.9, 200))
    for lon_a, lon_b in ((np.pi, -np.pi),):
        a = np.stack([np.cos(lat) * np.cos(lon_a), np.cos(lat) * np.sin(lon_a), np.sin(lat)], 1)
        b = np.stack([np.cos(lat) * np.cos(lon_b), np.cos(lat) * np.sin(lon_b), np.sin(lat)], 1)
        out = np.empty(200)
        va = noise.fbm(np.ascontiguousarray(a * 2), perm, noise.GRAD3, 5, 2.0, 2.0, 0.5, out).copy()
        vb = noise.fbm(np.ascontiguousarray(b * 2), perm, noise.GRAD3, 5, 2.0, 2.0, 0.5, out).copy()
        assert np.max(np.abs(va - vb)) < 1e-9


def test_domain_warp_changes_the_field():
    pts = _sample()
    out = np.empty(pts.shape[0])
    plain = noise.fbm(pts, noise.permutation(9), noise.GRAD3, 4, 2.0, 2.0, 0.5, out).copy()
    warped = noise.warped_fbm(pts, noise.permutation(9), noise.GRAD3, 4, 2.0, 2.0, 0.5,
                              0.4, 2.0, out).copy()
    assert not np.allclose(plain, warped)
    assert np.isfinite(warped).all()


# --------------------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------------------


def test_area_quantile_is_area_weighted_not_count_weighted():
    """One huge cell must outweigh many small ones. A plain np.quantile gets this wrong."""
    values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    area = np.array([1.0, 1.0, 1.0, 97.0], dtype=np.float32)
    assert stage1.area_quantile(values, area, 0.5) == pytest.approx(3.0)
    assert np.quantile(values, 0.5) == pytest.approx(1.5)  # what we must NOT get


def test_label_components_on_a_split_graph():
    #  0-1-2   3-4   (two components)
    nbr = np.array([[1, -1], [0, 2], [1, -1], [4, -1], [3, -1]], dtype=np.int32)
    mask = np.ones(5, dtype=np.bool_)
    labels = np.empty(5, dtype=np.int32)
    stack = np.empty(5, dtype=np.int32)
    n = stage1.label_components(mask, nbr, labels, stack)
    assert n == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_label_components_ignores_masked_out_cells():
    nbr = np.array([[1, -1], [0, 2], [1, -1]], dtype=np.int32)
    mask = np.array([True, False, True])  # cell 1 removed -> 0 and 2 disconnected
    labels = np.empty(3, dtype=np.int32)
    stack = np.empty(3, dtype=np.int32)
    assert stage1.label_components(mask, nbr, labels, stack) == 2
    assert labels[1] == -1


def test_sea_level_hits_the_target_land_fraction(g):
    for target in (0.15, 0.29, 0.50):
        p = stage1.CrustParams(target_land_fraction=target, min_island_area_km2=0.0)
        c = stage1.generate_crust(g, p, seed=2)
        assert c.land_fraction == pytest.approx(target, abs=0.01)


def test_elevation_is_referenced_to_sea_level(g):
    c = stage1.generate_crust(g, seed=2)
    assert c.sea_level_m == 0.0
    assert np.all(c.elevation[c.land_mask] > 0)
    assert np.all(c.elevation[~c.land_mask] <= 0)


def test_continental_crust_sits_above_oceanic(g):
    """Isostasy sanity: thicker, lighter crust must float higher."""
    p = stage1.CrustParams()
    c = stage1.generate_crust(g, seed=2)

    # Derived from the params rather than hardcoded, because the isostasy is radius-blind
    # and its thicknesses were cut ~3x when the sphere shrank: absolute 25/10/3000 km/m
    # bars silently selected nothing at all once t_continental fell from 35 km to 12.
    def floats_at(t_km, rho):
        return t_km * (1.0 - rho / p.rho_mantle) * 1000.0

    expect = (floats_at(p.t_continental_km, p.rho_continental)
              - floats_at(p.t_oceanic_km, p.rho_oceanic))
    thick = c.crust_thickness > 0.75 * p.t_continental_km
    thin = c.crust_thickness < 1.5 * p.t_oceanic_km
    assert thick.any() and thin.any(), "thickness bands must actually select cells"
    assert c.elevation[thick].mean() > c.elevation[thin].mean() + 0.5 * expect


def test_island_pruning_removes_small_landmasses(g):
    loose = stage1.generate_crust(g, stage1.CrustParams(min_island_area_km2=0.0), seed=3)
    # As a fraction of the sphere. The old absolute 5e6 km^2 was ~1% of Earth; on a
    # 53,000 km^2 world it exceeds the whole planet, so every landmass was pruned and
    # the test died on an empty array rather than on a wrong answer.
    bar = 0.01 * float(g.area.sum())
    strict_p = stage1.CrustParams(min_island_area_km2=bar)
    strict = stage1.generate_crust(g, strict_p, seed=3)
    assert strict.landmass_areas.size < loose.landmass_areas.size
    assert strict.landmass_areas.min() >= bar
    assert strict.land_fraction < loose.land_fraction


def test_generation_is_reproducible(g):
    a = stage1.generate_crust(g, seed=42)
    b = stage1.generate_crust(g, seed=42)
    c = stage1.generate_crust(g, seed=43)
    assert np.array_equal(a.elevation, b.elevation)
    assert not np.array_equal(a.elevation, c.elevation)


def test_sub_seeds_are_stable_across_processes():
    """The same seed must give the same world after restarting Python.

    Python randomises hash() for str and for any tuple containing one, so deriving
    sub-seeds as `hash(("crust", seed))` produced a DIFFERENT PLANET on every interpreter
    restart. A same-process test cannot see it -- within one run the hash is fixed -- which
    is exactly why this one shells out. It surfaced as a Stage 2 test that passed alone and
    failed in a suite, with a different number each time.
    """
    import subprocess

    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from terrain import noise\n"
        "print(noise.sub_seed('crust', 7), noise.sub_seed('relief', 7), "
        "noise.sub_seed('uplift', 7))"
    ) % str(pathlib.Path(__file__).resolve().parents[1])

    runs = set()
    for hashseed in ("0", "1", "12345"):
        env = dict(**__import__("os").environ, PYTHONHASHSEED=hashseed)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, check=True)
        runs.add(out.stdout.strip())
    assert len(runs) == 1, f"sub-seeds vary with PYTHONHASHSEED: {runs}"


def test_sub_seeds_differ_by_name_and_seed():
    """Distinct names must not collide, or two stages would share a noise field."""
    from terrain import noise as n
    assert n.sub_seed("crust", 1) != n.sub_seed("relief", 1)
    assert n.sub_seed("crust", 1) != n.sub_seed("crust", 2)
    assert n.sub_seed("crust", 1) == n.sub_seed("crust", 1)


def test_changing_relief_does_not_reshuffle_the_continents(g):
    """Sub-seeds are derived by name, so one knob must not move an unrelated field.

    If every sub-step drew sequentially from one RNG stream, changing the relief octaves
    would reshuffle the continent layout and no A/B comparison would mean anything.
    """
    base = stage1.generate_crust(g, stage1.CrustParams(relief_octaves=7), seed=8)
    tweaked = stage1.generate_crust(g, stage1.CrustParams(relief_octaves=3), seed=8)
    agreement = (base.land_mask == tweaked.land_mask).mean()
    assert agreement > 0.95, f"continents moved when only relief changed ({agreement:.1%})"


def test_landmass_areas_are_sorted_and_sum_to_land(g):
    c = stage1.generate_crust(g, seed=5)
    a = c.landmass_areas
    assert np.all(np.diff(a) <= 0), "landmass areas must be descending"
    assert a.sum() == pytest.approx(float(g.area[c.land_mask].sum()), rel=1e-4)
