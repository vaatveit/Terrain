"""Stage 6: lakes, endorheic basins, rivers.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import coupled, hydrology, stage1  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


# --------------------------------------------------------------------------------------
# Water balance
# --------------------------------------------------------------------------------------


def test_pet_rises_with_temperature():
    p = hydrology.HydrologyParams()
    pet = hydrology.potential_evapotranspiration(np.array([-10.0, 0.0, 10.0, 25.0]), p)
    assert np.all(np.diff(pet) > 0)
    assert pet[1] == pytest.approx(300.0, rel=0.01)      # ~300 mm at freezing
    assert 500 < pet[2] < 700                            # ~600 mm at 10 C
    assert pet[3] < p.pet_max_mm


def test_runoff_is_precipitation_minus_evapotranspiration():
    p = hydrology.HydrologyParams()
    t = np.array([10.0, 10.0, 10.0])
    r = hydrology.runoff_mm(np.array([2000.0, 600.0, 100.0]), t, p)
    pet = hydrology.potential_evapotranspiration(t, p)[0]
    assert r[0] == pytest.approx(2000.0 - pet)
    assert r[2] == 0.0, "arid ground yields no runoff, never negative"
    assert (r >= 0).all()


# --------------------------------------------------------------------------------------
# The endorheic/exorheic decision, on constructed basins with known answers
# --------------------------------------------------------------------------------------


def _bowl_world(n_side=24, rain_mm=2000.0, temp_c=10.0, depth_m=500.0):
    """A single square depression in a flat plain, with a controllable climate.

    Constructed rather than generated because the emergent worlds contain very few closed
    basins, so the endorheic branch would otherwise go untested.
    """
    from terrain.dem import D8
    h = w = n_side
    n = h * w
    rows, cols = np.divmod(np.arange(n), w)
    nbr = np.full((n, 8), -1, dtype=np.int32)
    for s, (dr, dc) in enumerate(D8):
        r2, c2 = rows + dr, cols + dc
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        nbr[ok, s] = (r2[ok] * w + c2[ok]).astype(np.int32)

    class G:
        cell_count = n
        neighbours = nbr
        arc_dist = np.full((n, 8), 10.0, dtype=np.float32)
        area = np.full(n, 100.0, dtype=np.float32)
        lat = np.zeros(n, dtype=np.float32)

    # A paraboloid, not a flat-floored box. With a flat floor the lake area jumps
    # discontinuously from zero to the whole floor, so the equilibrium level always lands
    # exactly on the bed and no lake of positive depth can ever be found -- a degenerate
    # case, not a realistic one. Real basins have a hypsometric curve.
    cy = cx = (n_side - 1) / 2.0
    rad = np.sqrt(((rows - cy) / cy) ** 2 + ((cols - cx) / cx) ** 2)
    z = np.where(rad < 1.0, 1000.0 - depth_m * (1.0 - rad**2), 1000.0).astype(np.float32)
    z[0] = 0.0  # one outlet cell so the domain drains somewhere
    is_ocean = np.zeros(n, dtype=bool)
    is_ocean[0] = True

    return (G, z, is_ocean, np.full(n, rain_mm, dtype=np.float32),
            np.full(n, temp_c, dtype=np.float32))


def test_a_wet_basin_fills_to_its_spill_point():
    g, z, is_ocean, rain, temp = _bowl_world(rain_mm=3000.0, temp_c=5.0)
    h = hydrology.analyse(g, z, is_ocean, rain, temp,
                          hydrology.HydrologyParams(min_lake_area_km2=100.0))
    assert h.lakes, "no lake found in a constructed depression"
    lake = max(h.lakes, key=lambda L: L["area_km2"])
    assert not lake["endorheic"]
    assert lake["level_m"] == pytest.approx(lake["spill_m"], abs=1.0)


def test_an_arid_basin_settles_below_its_spill_point():
    """The Caspian case: evaporation balances inflow before the water reaches the rim.

    Note the climate has to be chosen with care. Push the temperature too high and land
    runoff drops to exactly zero -- PET exceeds rainfall, nothing reaches the basin, and
    the answer is 'no lake' rather than 'endorheic lake'. The endorheic window needs
    positive runoff that lake-surface evaporation can still outpace, roughly P < 2*PET.
    """
    g, z, is_ocean, rain, temp = _bowl_world(rain_mm=900.0, temp_c=10.0)
    h = hydrology.analyse(g, z, is_ocean, rain, temp,
                          hydrology.HydrologyParams(min_lake_area_km2=100.0))
    assert h.lakes, "no lake found in a constructed depression"
    lake = max(h.lakes, key=lambda L: L["area_km2"])
    assert lake["endorheic"], (
        f"lake filled to {lake['level_m']:.0f} m against a spill of {lake['spill_m']:.0f} m; "
        f"expected it to stop short")
    assert lake["level_m"] < lake["spill_m"]


def test_a_desert_basin_holds_no_lake_at_all():
    g, z, is_ocean, rain, temp = _bowl_world(rain_mm=200.0, temp_c=28.0)
    h = hydrology.analyse(g, z, is_ocean, rain, temp,
                          hydrology.HydrologyParams(min_lake_area_km2=100.0))
    assert not h.lake_mask.any(), "a basin with no runoff should stay dry"


def test_warming_the_same_basin_switches_it_from_exorheic_to_endorheic():
    """Same geometry, same rainfall, only the temperature differs."""
    params = hydrology.HydrologyParams(min_lake_area_km2=100.0)
    g, z, is_ocean, rain, _ = _bowl_world(rain_mm=1200.0)

    cool = hydrology.analyse(g, z, is_ocean, rain,
                             np.full(g.cell_count, 2.0, dtype=np.float32), params)
    warm = hydrology.analyse(g, z, is_ocean, rain,
                             np.full(g.cell_count, 12.0, dtype=np.float32), params)
    cool_lake = max(cool.lakes, key=lambda L: L["area_km2"])
    warm_lake = max(warm.lakes, key=lambda L: L["area_km2"])
    assert not cool_lake["endorheic"]
    assert warm_lake["endorheic"]
    assert warm_lake["area_km2"] < cool_lake["area_km2"]


def test_lake_depth_is_never_negative():
    g, z, is_ocean, rain, temp = _bowl_world(rain_mm=900.0, temp_c=15.0)
    h = hydrology.analyse(g, z, is_ocean, rain, temp,
                          hydrology.HydrologyParams(min_lake_area_km2=100.0))
    assert (h.lake_depth_m >= 0).all()
    assert (h.lake_depth_m[~h.lake_mask] == 0).all()


# --------------------------------------------------------------------------------------
# Rivers
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    g = build_goldberg(128)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=2, steps_per_pass=200)
    h = hydrology.analyse(g, res.elevation, ~crust.land_mask,
                          res.climate.precipitation_mm, res.climate.temperature_c)
    return g, crust, res, h


def test_discharge_grows_downstream_and_is_bounded(world):
    g, crust, res, h = world
    assert (h.discharge >= 0).all()
    # Total discharge reaching outlets cannot exceed total runoff generated.
    total_runoff = float((h.runoff_mm * 1e-6 * g.area).sum())
    assert h.discharge.max() <= total_runoff * 1.001


def test_rivers_follow_discharge_not_area(world):
    """Width from discharge, w = c*Q^0.5, is the standard hydraulic-geometry relation."""
    g, crust, res, h = world
    assert h.river_mask.any()
    w, q = h.river_width_m[h.river_mask], h.discharge[h.river_mask]
    fit = np.polyfit(np.log10(q), np.log10(w), 1)[0]
    assert fit == pytest.approx(0.5, abs=0.02)
    assert (h.river_width_m[~h.river_mask] == 0).all()


def test_rivers_do_not_run_through_lakes_or_ocean(world):
    g, crust, res, h = world
    assert not (h.river_mask & h.lake_mask).any()
    assert not (h.river_mask & ~crust.land_mask).any()


def test_arid_ground_yields_less_runoff_than_it_receives(world):
    g, crust, res, h = world
    land = crust.land_mask
    p = res.climate.precipitation_mm
    assert (h.runoff_mm[land] <= p[land] + 1e-3).all()
    dry = land & (p < np.percentile(p[land], 25))
    assert h.runoff_mm[dry].mean() < 0.5 * h.runoff_mm[land].mean()


def test_hydrology_is_reproducible(world):
    g, crust, res, h = world
    again = hydrology.analyse(g, res.elevation, ~crust.land_mask,
                              res.climate.precipitation_mm, res.climate.temperature_c)
    assert np.array_equal(h.discharge, again.discharge)
    assert np.array_equal(h.lake_mask, again.lake_mask)
