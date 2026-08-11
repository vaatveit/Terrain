"""Climate: temperature, prevailing wind, orographic precipitation.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import climate, stage1, stage2  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


@pytest.fixture(scope="module")
def world():
    g = build_goldberg(96)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    res = stage2.evolve(g, crust.elevation, crust.uplift, ~crust.land_mask,
                        stage2.LemParams(), n_steps=150)
    cl = climate.compute_climate(g, res.elevation, crust.land_mask)
    return g, res.elevation, crust.land_mask, cl


# --------------------------------------------------------------------------------------
# Temperature
# --------------------------------------------------------------------------------------


def test_temperature_falls_from_equator_to_pole(world):
    g, _, _, cl = world
    t = cl.temperature_c
    tropics = t[np.abs(g.lat) < 10].mean()
    polar = t[np.abs(g.lat) > 75].mean()
    assert tropics > 20.0
    assert polar < -10.0
    assert tropics - polar > 30.0


def test_lapse_rate_is_applied_exactly(world):
    g, z, _, cl = world
    p = climate.ClimateParams()
    sea = np.zeros(g.cell_count)
    t_sea, _ = climate.temperature(g, sea, cl.coast_distance_km, p)
    t_high, _ = climate.temperature(g, np.full(g.cell_count, 2000.0),
                                    cl.coast_distance_km, p)
    assert np.allclose(t_sea - t_high, 2.0 * p.lapse_rate_c_per_km, atol=1e-4)


def test_seasonality_grows_inland(world):
    g, _, land, cl = world
    d = cl.coast_distance_km
    near = land & (d < 200)
    far = land & (d > 800)
    if far.sum() > 10:
        assert cl.seasonality_c[far].mean() > cl.seasonality_c[near].mean()


# --------------------------------------------------------------------------------------
# Wind
# --------------------------------------------------------------------------------------


def test_wind_is_tangent_to_the_sphere(world):
    g, _, _, cl = world
    radial = np.einsum("ij,ij->i", cl.wind.astype(np.float64), g.xyz)
    assert np.abs(radial).max() < 1e-6


def test_wind_bands_have_the_right_sense(world):
    """Trades easterly, mid-latitude westerly, polar easterly.

    Getting this backwards would put every rain shadow on the wrong side of every range.
    """
    g, _, _, cl = world
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(np.broadcast_to(pole, g.xyz.shape), g.xyz)
    norm = np.linalg.norm(east, axis=1, keepdims=True)
    east = np.divide(east, norm, out=np.zeros_like(east), where=norm > 1e-12)
    zonal = np.einsum("ij,ij->i", cl.wind.astype(np.float64), east)

    lat = np.abs(g.lat)
    assert zonal[(lat > 5) & (lat < 25)].mean() < 0, "trades must be easterly"
    assert zonal[(lat > 35) & (lat < 55)].mean() > 0, "mid-latitudes must be westerly"
    assert zonal[(lat > 65) & (lat < 85)].mean() < 0, "polar cell must be easterly"


def test_wind_goes_slack_between_bands(world):
    """Doldrums, horse latitudes and the polar front are where the circulation stalls."""
    g, _, _, cl = world
    speed = np.linalg.norm(cl.wind.astype(np.float64), axis=1)
    lat = np.abs(g.lat)
    between = speed[(lat < 3) | ((lat > 27) & (lat < 33)) | ((lat > 57) & (lat < 63))]
    within = speed[((lat > 12) & (lat < 18)) | ((lat > 42) & (lat < 48))]
    assert between.mean() < 0.4 * within.mean()


# --------------------------------------------------------------------------------------
# Precipitation
# --------------------------------------------------------------------------------------


def test_precipitation_is_finite_and_bounded(world):
    g, _, land, cl = world
    p = cl.precipitation_mm
    assert np.isfinite(p).all()
    assert (p >= 0).all()
    assert p[land].mean() == pytest.approx(1000.0, rel=0.01)


def test_moisture_never_exceeds_saturation(world):
    """The saturation cap is what makes the fixed-point iteration converge at all.

    Without it, incoming advection weights sum to more than one where the wind converges,
    the operator stops being a contraction, and moisture grows without bound: measured
    1.3e7 mm/yr before the cap was added.
    """
    g, z, land, cl = world
    p = climate.ClimateParams()
    capacity = np.exp(p.saturation_gain * (cl.temperature_c.astype(np.float64) - 15.0))
    capacity = np.clip(capacity, 0.05, 4.0)
    assert np.all(cl.moisture <= capacity + 1e-6)


def test_precipitation_falls_from_tropics_to_poles(world):
    """Wet tropics, dry poles -- the trend the capacity model does capture.

    Deliberately NOT asserting a subtropical desert belt. Earth dips near 25-30 degrees
    because air there descends from the top of the Hadley cell already wrung dry over the
    ITCZ, and a surface-only model has no aloft to bring dry air down from. See the module
    docstring in terrain/climate.py for the three surrogates tried and measured. Asserting
    a belt here would only lock in a fiction.
    """
    g, _, land, cl = world
    p = cl.precipitation_mm
    lat = np.abs(g.lat)
    equator = p[lat < 10].mean()
    mid = p[(lat > 37) & (lat < 52)].mean()
    polar = p[lat > 70].mean()
    assert equator > 2 * mid > 4 * polar


def test_interiors_are_drier_than_coasts_but_not_dead(world):
    g, _, land, cl = world
    p, d = cl.precipitation_mm, cl.coast_distance_km
    coast = p[land & (d < 200)].mean()
    interior = p[land & (d > 800)]
    if interior.size > 10:
        assert interior.mean() < coast
        # Evapotranspiration recycling: without it every drop falling on a continent had to
        # arrive from the sea and interiors came out bone dry. The bound is loose because
        # this fights the orographic term (see the table in ClimateParams) and because it
        # moves with how large the continents happen to be.
        assert coast / interior.mean() < 20.0


def _walk(start, steps, wind, xyz, nbr, downwind):
    c = start
    for _ in range(steps):
        best, best_align = -1, 0.0
        for s in range(nbr.shape[1]):
            j = nbr[c, s]
            if j < 0:
                continue
            align = float((xyz[j] - xyz[c]) @ wind[c])
            if not downwind:
                align = -align
            if align > best_align:
                best_align, best = align, j
        if best < 0:
            return c
        c = best
    return c


def test_windward_slopes_are_wetter_than_leeward(world):
    """Measured by walking along the wind, not by looking at the slope ahead of a cell.

    The model rains where air has ALREADY climbed, so a cell just past a crest still carries
    the ascent rain and reads as a wet 'lee'. The shadow only shows a few cells downwind.

    This is the RAW ratio and it is not purely orographic -- see the next test.
    """
    g, z, land, cl = world
    ridges = np.flatnonzero(land & (z > np.percentile(z[land], 92)))
    ridges = np.random.default_rng(0).choice(ridges, min(400, ridges.size), replace=False)
    ratio = _shadow_ratio(g, land, cl, ridges)
    assert ratio > 1.5, f"windward/leeward only {ratio:.2f}x"


def _shadow_ratio(g, land, cl, ridges, steps=4):
    p, wind, xyz, nbr = cl.precipitation_mm, cl.wind, g.xyz, g.neighbours
    up, down = [], []
    for r in ridges:
        a = _walk(r, steps, wind, xyz, nbr, downwind=False)
        b = _walk(r, steps, wind, xyz, nbr, downwind=True)
        if land[a] and land[b]:
            up.append(p[a])
            down.append(p[b])
    return float(np.mean(up) / max(np.mean(down), 1.0)) if up else np.nan


def test_the_shadow_is_actually_orographic(world):
    """Flatten the terrain and re-measure at the SAME points: the excess is the mountain.

    The control is the whole test. Flat ground scores about 1.9x on its own, because
    walking upwind from a ridge heads toward the sea and downwind heads inland, so the raw
    windward/leeward ratio is mostly the coast-to-interior gradient. An earlier version of
    this suite reported 1.9-2.9x "rain shadows" that were very largely continentality.

    Two other framings that do NOT work: comparing the p90/p10 spread (flat terrain gives a
    *wider* spread, 154 against 44.9, because interiors go bone dry without orographic
    rain), and expecting the flat ratio to approach 1.0 (it does not, for the reason above).
    """
    g, z, land, cl = world
    flat = np.where(land, 100.0, -3000.0).astype(np.float32)
    flat_cl = climate.compute_climate(g, flat, land)

    # Same sample points for both -- flat ground has no ridges of its own to pick.
    ridges = np.flatnonzero(land & (z > np.percentile(z[land], 92)))
    ridges = np.random.default_rng(0).choice(ridges, min(400, ridges.size), replace=False)

    relief_ratio = _shadow_ratio(g, land, cl, ridges)
    flat_ratio = _shadow_ratio(g, land, flat_cl, ridges)
    isolated = relief_ratio / flat_ratio
    # The bound is deliberately modest, and that is itself the finding: the orographic
    # signal in this model is WEAK. Measured across worlds and settings it ranges 1.10-1.40,
    # and it fights two other knobs -- land_recycling refills the lee (see the trade-off
    # table in ClimateParams) and wind meandering makes the upwind/downwind walk noisier.
    # 1.10 is what the model reliably clears; anything stricter would be tuning the model
    # to a number rather than measuring it.
    assert isolated > 1.10, (
        f"relief adds only {isolated:.2f}x over flat ground "
        f"({relief_ratio:.2f} vs {flat_ratio:.2f}) -- the shadow is not orographic")


def test_climate_is_reproducible(world):
    g, z, land, cl = world
    again = climate.compute_climate(g, z, land)
    assert np.array_equal(cl.precipitation_mm, again.precipitation_mm)
    assert np.array_equal(cl.temperature_c, again.temperature_c)
