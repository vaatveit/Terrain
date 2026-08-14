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
    """Terrain from the COUPLED pipeline, which is what the pipeline actually produces.

    Not an incidental choice. On uncoupled terrain the isolated orographic ratio measures
    0.92 -- no signal at all -- against 1.27 after coupling. The feedback amplifies itself:
    wet windward slopes get incised, which deepens the relief that drives the orographic
    rainfall in the first place. Testing the climate against uncoupled terrain would be
    testing a state the pipeline never emits.
    """
    from terrain import coupled
    g = build_goldberg(96)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=2, steps_per_pass=150)
    return g, res.elevation, crust.land_mask, res.climate


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
# Seasons
# --------------------------------------------------------------------------------------


def test_insolation_has_polar_night_and_midnight_sun():
    """The day-length term, which is the whole reason this is not cos(lat - declination).

    That shortcut gives the winter pole an ordinary winter and makes the summer pole merely
    warm. Polar night is not a small correction to it -- it is zero insolation for months,
    and Koppen's D and E classes turn on exactly that.
    """
    assert climate.daily_insolation(90.0, -23.44) == 0.0, "no polar night"
    assert climate.daily_insolation(70.0, -23.44) == 0.0, "no polar night at the circle"
    assert climate.daily_insolation(90.0, 0.0) == pytest.approx(0.0, abs=1e-9)

    # Under the midnight sun the pole receives more in a day than anywhere else on Earth.
    pole = climate.daily_insolation(90.0, 23.44)
    assert pole > climate.daily_insolation(0.0, 23.44)
    assert pole > climate.daily_insolation(45.0, 23.44)


def test_the_equinox_reproduces_the_unseasonal_profile_exactly(world):
    """Seasons are additive on top of the tuned annual profile, so a caller that does not
    ask for them gets bit-identical output to before they existed."""
    g, _, _, _ = world
    p = climate.ClimateParams()
    lat = np.radians(g.lat.astype(np.float64))
    expected = p.t_pole_c + (p.t_equator_c - p.t_pole_c) * np.cos(lat) ** 1.5
    assert np.array_equal(climate.equilibrium_temperature(g, p, 0.0), expected)


def test_the_seasons_average_back_to_the_annual_profile(world):
    """`t_equator_c` and `t_pole_c` are calibrated as ANNUAL MEANS, so the seasons have to
    average back to them or every downstream threshold shifts.

    This is why the seasonal term is not clipped to the profile's range. Clipping truncated
    the winter pole and left the polar annual mean about 5 C above `t_pole_c`; unclipped,
    the insolation anomaly averages to zero by construction and the mean is exact. The price
    is a winter pole below `t_pole_c`, which is the correct shape.
    """
    g, _, _, _ = world
    p = climate.ClimateParams()
    base = climate.equilibrium_temperature(g, p, 0.0)
    decls = [climate.declination_deg(f * p.days_per_year, p) for f in climate.SEASON_PHASE]
    mean = np.mean([climate.equilibrium_temperature(g, p, d) for d in decls], axis=0)
    # 0.21 C measured, on a 52 C profile. Not zero because four points cannot integrate the
    # higher harmonics of the insolation curve exactly, only the first two.
    assert np.abs(mean - base).max() < 0.3


def test_four_samples_are_four_distinct_seasons(world):
    """A steady-state model carries the date only through the declination, so two dates
    with the same sun give the same world. Sampling the solstices and equinoxes would give
    three seasons and a duplicate; SEASON_PHASE is offset from them for that reason."""
    g, _, _, _ = world
    p = climate.ClimateParams()
    decls = [climate.declination_deg(f * p.days_per_year, p) for f in climate.SEASON_PHASE]
    assert len(decls) == 4
    for i in range(4):
        for j in range(i + 1, 4):
            assert abs(decls[i] - decls[j]) > 1.0, f"samples {i} and {j} share a sun"


def test_land_takes_a_season_harder_than_sea(world):
    """Differential heating: rock follows the sun, water stores it. The contrast is the
    monsoon's engine, and without it the June land-sea contrast measured 0.58 C with land
    running COLDER than the sea -- nothing for a thermal low to form on."""
    g, z, land, cl = world
    p = climate.ClimateParams()
    warm, cold = [], []
    for decl in (23.44, -23.44):
        t, _ = climate.temperature(g, z, cl.coast_distance_km, p, decl_deg=decl,
                                   land_mask=land)
        (warm if decl > 0 else cold).append(t)
    swing = np.abs(warm[0] - cold[0])
    north = g.lat > 20
    assert swing[land & north].mean() > 1.5 * swing[~land & north].mean()


def test_seasons_keep_the_annual_fields_the_pipeline_runs_on(world):
    """Erosion has to see a year's water, and every downstream stage reads these two names.
    Seasonality is an output; it must not leak into the fields Stage 4 couples to."""
    g, z, land, _ = world
    cs = climate.compute_seasons(g, z, land)
    assert cs.temperature_season.shape == (4, g.cell_count)
    assert cs.precipitation_season.shape == (4, g.cell_count)
    assert np.allclose(cs.temperature_c, cs.temperature_season.mean(axis=0), atol=1e-4)
    assert np.allclose(cs.precipitation_mm, cs.precipitation_season.mean(axis=0), atol=1e-2)
    # And the swing is measured from the solved seasons rather than modelled from distance.
    assert cs.seasonality_c[land].mean() > 0.0


def test_the_monsoon_wind_actually_reverses(world):
    """Migrating the circulation bands gives a wet season; only the thermal low gives a
    MONSOON, and the difference is a wind that turns round.

    Measured on the term itself rather than on the fraction of coast that flips, because
    that fraction depends on how much tropical coast a world happens to have -- 32.5% to
    38.7% at n=256, but only 31.7% to 32.8% on this small fixture. What the term does is
    world-independent: it must blow ONSHORE in the local summer and OFFSHORE in the local
    winter, and reverse with the hemisphere.
    """
    g, z, land, cl = world
    p = climate.ClimateParams()

    def onshore(wind):
        out = np.zeros(g.cell_count)
        for s in range(g.neighbours.shape[1]):
            j = g.neighbours[:, s]
            jj = np.maximum(j, 0)
            d = g.xyz[jj] - np.einsum('ij,ij->i', g.xyz[jj], g.xyz)[:, None] * g.xyz
            dn = np.linalg.norm(d, axis=1)
            out += np.where((j >= 0) & ~land[jj] & (dn > 0),
                            -np.einsum('ij,ij->i', wind, d) / np.where(dn > 0, dn, 1.0), 0.0)
        return out

    added = {}
    for decl in (23.44, -23.44):
        t, _ = climate.temperature(g, z, cl.coast_distance_km, p, decl_deg=decl,
                                   land_mask=land)
        added[decl] = (onshore(climate.monsoon_wind(g, land, t, p, decl_deg=decl))
                       - onshore(climate.wind_field(g, p, decl_deg=decl)))

    north = land & (g.lat > 5) & (g.lat < 35) & (added[23.44] != 0)
    south = land & (g.lat < -5) & (g.lat > -35) & (added[23.44] != 0)
    assert north.sum() > 50 and south.sum() > 50

    # Northern summer is the June solstice, southern summer the December one.
    assert added[23.44][north].mean() > 0, "no onshore pull over a northern summer"
    assert added[-23.44][north].mean() < 0, "northern winter must blow offshore"
    assert added[-23.44][south].mean() > 0, "no onshore pull over a southern summer"
    assert added[23.44][south].mean() < 0, "southern winter must blow offshore"


# --------------------------------------------------------------------------------------
# Ocean heat transport
# --------------------------------------------------------------------------------------


def test_currents_stay_in_the_sea_and_follow_the_shore(world):
    """A current that runs into a continent stops advecting, so it must turn instead.

    Turning is not a tidiness measure -- it is the mechanism. A gyre only acquires a
    poleward western boundary because the flow cannot cross the coast.
    """
    g, _, land, _ = world
    cur = climate.current_field(g, land, climate.ClimateParams())
    speed = np.linalg.norm(cur, axis=1)

    assert np.allclose(speed[land], 0.0), "current on land"
    assert (speed[~land] > 0.99).mean() > 0.99, "sea cells should carry a unit current"
    # Tangent to the sphere, like the wind.
    assert np.abs(np.einsum('ij,ij->i', cur, g.xyz)).max() < 1e-9

    # No cell flows into land. Checked against the same tangent direction the advection
    # weights use, so this is exactly the quantity that would leak.
    for s in range(g.neighbours.shape[1]):
        j = g.neighbours[:, s]
        jj = np.maximum(j, 0)
        d = g.xyz[jj] - np.einsum('ij,ij->i', g.xyz[jj], g.xyz)[:, None] * g.xyz
        dn = np.linalg.norm(d, axis=1)
        into = np.einsum('ij,ij->i', cur, d) / np.where(dn > 0, dn, 1.0)
        shore = ~land & (j >= 0) & land[jj] & (dn > 0)
        if shore.any():
            assert into[shore].max() < 1e-6, "current flows into the coast"


def test_the_anomaly_cannot_manufacture_heat(world):
    """The sweep gathers a weighted MEAN of its upwind neighbours, so the result is bounded
    by the extremes of the equilibrium profile it started from. `_advect_step` needed a
    saturation cap to get this property for moisture; a mean has it for free.

    Tested with `strip_zonal_mean` off, because the bound belongs to the advection and
    removing the zonal mean afterwards can legitimately push a cell outside the original
    range -- measured -26.9 C against an equilibrium floor of -25.0.
    """
    import dataclasses
    g, _, land, _ = world
    p = dataclasses.replace(climate.ClimateParams(), strip_zonal_mean=False)
    eq = climate.equilibrium_temperature(g, p)
    anom = climate.sst_anomaly(g, land, eq, p)
    sst = eq + anom
    assert sst[~land].min() >= eq.min() - 1e-6
    assert sst[~land].max() <= eq.max() + 1e-6
    assert np.allclose(anom[land], 0.0), "land is not sea surface"


def test_ocean_heat_transport_switches_off_cleanly(world):
    """`ocean_heat_c = 0` must return the pre-transport climate exactly.

    Note it is the GAIN that switches this off, not `wind_speed = 0`. With no wind there is
    still the isotropic `eddy` share, so the sweep degenerates to Laplacian smoothing of a
    curved equilibrium profile and leaves a small real anomaly near the equator and poles.
    """
    import dataclasses
    g, z, land, _ = world
    off = dataclasses.replace(climate.ClimateParams(), ocean_heat_c=0.0)
    c_off = climate.compute_climate(g, z, land, off)
    assert np.allclose(c_off.sst_anomaly_c, 0.0)

    # Fully disconnected, not merely scaled to nothing: changing how far a current carries
    # its heat must make no difference at all once the gain is zero.
    other = dataclasses.replace(off, current_scale_km=off.current_scale_km * 8,
                                maritime_scale_km=off.maritime_scale_km * 8)
    assert np.array_equal(c_off.temperature_c,
                          climate.compute_climate(g, z, land, other).temperature_c)


def test_onshore_coasts_feel_the_sea_and_offshore_coasts_do_not(world):
    """Norway against Labrador, which is the whole point of the exercise.

    The contrast is not that one coast sits beside warmer water -- it is that the
    westerlies carry maritime air INTO Europe, while at the same latitude the wind leaves
    Labrador for the sea and the warm water offshore never reaches the land. So the test is
    directional, and it is the one an isotropic model cannot pass.

    Measured: an earlier version decayed the anomaly as exp(-distance) from the coast and
    scored 0.64x here -- offshore coasts came out MORE maritime than onshore ones, because
    a geometric decay carries no direction and was really just re-deriving continentality.
    """
    g, z, land, _ = world
    p = climate.ClimateParams()
    wind = climate.wind_field(g, p)
    anom = climate.sst_anomaly(g, land, climate.equilibrium_temperature(g, p), p)
    maritime = climate.maritime_anomaly(g, land, anom, wind, p)

    # How strongly the wind blows from a sea neighbour onto each coastal land cell.
    onshore = np.zeros(g.cell_count)
    for s in range(g.neighbours.shape[1]):
        j = g.neighbours[:, s]
        jj = np.maximum(j, 0)
        d = g.xyz[jj] - np.einsum('ij,ij->i', g.xyz[jj], g.xyz)[:, None] * g.xyz
        dn = np.linalg.norm(d, axis=1)
        towards = -np.einsum('ij,ij->i', wind, d) / np.where(dn > 0, dn, 1.0)
        onshore += np.where((j >= 0) & ~land[jj] & (dn > 0), towards, 0.0)

    windward = land & (onshore > 0.2)
    leeward = land & (onshore < -0.2)
    assert windward.sum() > 100 and leeward.sum() > 100
    ratio = np.abs(maritime[windward]).mean() / np.abs(maritime[leeward]).mean()
    assert ratio > 1.1, f"onshore/offshore maritime influence {ratio:.2f}x; expected > 1.1"


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
    g, z, land, _ = world
    p = climate.ClimateParams()
    # A SINGLE solve, because that is where the invariant lives. The seasonal aggregate
    # averages four moisture fields against four different capacities, and the mean of
    # capacities is not the capacity of the mean temperature -- the cap would look violated
    # where it never was.
    one = climate.compute_climate(g, z, land, p)
    capacity = np.exp(p.saturation_gain * (one.temperature_c.astype(np.float64) - 15.0))
    capacity = np.clip(capacity, 0.05, 4.0)
    assert np.all(one.moisture <= capacity + 1e-6)


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

    # EVERY ridge cell, not a sample. Drawing 400 of the ~2,100 gave this ratio a spread of
    # 0.86-1.51 across draw seeds on a single fixed world -- wider than the effect being
    # measured, so the old 400-point version was asserting partly on the luck of the draw.
    # The full pool is deterministic and costs nothing.
    ridges = np.flatnonzero(land & (z > np.percentile(z[land], 92)))

    relief_ratio = _shadow_ratio(g, land, cl, ridges)
    flat_ratio = _shadow_ratio(g, land, flat_cl, ridges)
    isolated = relief_ratio / flat_ratio
    # The bound is weak, and that IS the finding: the orographic signal in this model is
    # weak, and it varies more between worlds than it does between settings. Measured on
    # the full ridge pool over three seeds and two uplift configurations, the isolated
    # ratio spans 1.05-2.53, with no systematic difference between the configurations
    # (means 1.77 and 1.67). One world therefore cannot support a bar above about 1.0;
    # a tighter one passes or fails on which seed the fixture happens to use.
    #
    # It also fights two other knobs: land_recycling refills the lee (see the trade-off
    # table in ClimateParams) and wind meandering makes the upwind/downwind walk noisier.
    assert isolated > 1.0, (
        f"relief adds only {isolated:.2f}x over flat ground "
        f"({relief_ratio:.2f} vs {flat_ratio:.2f}) -- the shadow is not orographic")


def test_climate_is_reproducible(world):
    """Both entry points, since the pipeline now runs the seasonal one."""
    g, z, land, _ = world
    for solve in (climate.compute_climate, climate.compute_seasons):
        a, b = solve(g, z, land), solve(g, z, land)
        assert np.array_equal(a.precipitation_mm, b.precipitation_mm)
        assert np.array_equal(a.temperature_c, b.temperature_c)
