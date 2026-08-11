"""Stage 7: Whittaker biome classification.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import biomes, coupled, stage1  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


def _one(t, p, elev=0.0, lat=0.0):
    return int(biomes.classify(np.array([t]), np.array([p]), np.array([True]),
                               np.array([elev]), np.array([lat]))[0])


@pytest.mark.parametrize("t,p,expected", [
    (26, 3000, biomes.TROPICAL_RAINFOREST),
    (26, 2000, biomes.TROPICAL_SEASONAL_FOREST),
    (26, 800, biomes.SAVANNA),
    (26, 100, biomes.SUBTROPICAL_DESERT),
    (12, 2200, biomes.TEMPERATE_RAINFOREST),
    (12, 1200, biomes.TEMPERATE_SEASONAL_FOREST),
    (12, 600, biomes.WOODLAND_SHRUBLAND),
    (12, 200, biomes.TEMPERATE_GRASSLAND),
    (0, 500, biomes.BOREAL_FOREST),
    (0, 50, biomes.TUNDRA),
    (-8, 300, biomes.TUNDRA),
    (-20, 300, biomes.ICE),
])
def test_whittaker_corners(t, p, expected):
    assert _one(t, p) == expected, (
        f"{t} C / {p} mm gave {biomes.NAMES[_one(t, p)]}, expected {biomes.NAMES[expected]}")


def test_ocean_overrides_everything():
    b = biomes.classify(np.array([26.0]), np.array([3000.0]), np.array([False]))
    assert b[0] == biomes.OCEAN


def test_snowline_falls_from_equator_to_pole():
    assert biomes.snowline_m(0.0) > biomes.snowline_m(45.0) > biomes.snowline_m(80.0)
    assert biomes.snowline_m(90.0) == pytest.approx(0.0, abs=1.0)


def test_snowline_is_finite_at_the_poles():
    """cos(radians(90)) comes back very slightly negative, and a negative base to a
    fractional power is NaN -- which compares false against every elevation and would
    silently leave the polar caps unmarked rather than failing loudly."""
    vals = biomes.snowline_m(np.array([-90.0, -89.999, 0.0, 89.999, 90.0, 90.0001]))
    assert np.isfinite(vals).all()
    assert (vals >= 0).all()


def test_high_ground_becomes_ice_even_when_warm_at_sea_level():
    """Alpine ice is what makes mountains read as mountains."""
    warm_low = _one(24.0, 2000.0, elev=200.0, lat=5.0)
    warm_high = _one(24.0, 2000.0, elev=6000.0, lat=5.0)
    assert warm_low != biomes.ICE
    assert warm_high == biomes.ICE


def test_snowline_is_latitude_dependent():
    """3000 m is above the snowline at 60 degrees but well below it at the equator."""
    assert _one(20.0, 1000.0, elev=3000.0, lat=0.0) != biomes.ICE
    assert _one(-2.0, 1000.0, elev=3000.0, lat=60.0) == biomes.ICE


# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    g = build_goldberg(96)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=2, steps_per_pass=150)
    b = biomes.classify(res.climate.temperature_c, res.climate.precipitation_mm,
                        crust.land_mask, res.elevation, g.lat)
    return g, crust, res, b


def test_biomes_are_ordered_sensibly_by_latitude(world):
    g, crust, res, b = world
    land = crust.land_mask
    lat = np.abs(g.lat)

    tropical = {biomes.TROPICAL_RAINFOREST, biomes.TROPICAL_SEASONAL_FOREST,
                biomes.SAVANNA, biomes.SUBTROPICAL_DESERT}
    polar = {biomes.TUNDRA, biomes.ICE}

    equatorial = b[land & (lat < 20)]
    high_lat = b[land & (lat > 70)]
    assert np.isin(equatorial, list(tropical)).mean() > 0.7
    assert np.isin(high_lat, list(polar)).mean() > 0.7
    assert not np.isin(equatorial, list(polar)).any() or \
        np.isin(equatorial, list(polar)).mean() < 0.05


def test_every_land_cell_gets_a_biome(world):
    g, crust, res, b = world
    assert (b[crust.land_mask] != biomes.OCEAN).all()
    assert (b[~crust.land_mask] == biomes.OCEAN).all()
    assert set(np.unique(b)).issubset(set(biomes.NAMES))


def test_summary_shares_sum_to_one(world):
    g, crust, res, b = world
    rows = biomes.summary(b, g.area, crust.land_mask)
    assert sum(share for _, share in rows) == pytest.approx(1.0, abs=1e-6)
    assert rows == sorted(rows, key=lambda r: -r[1])


def test_a_plausible_variety_of_biomes_appears(world):
    """A world that collapses to two or three biomes means climate has gone flat."""
    g, crust, res, b = world
    rows = biomes.summary(b, g.area, crust.land_mask)
    assert len(rows) >= 7, f"only {len(rows)} biomes present: {[r[0] for r in rows]}"
    assert rows[0][1] < 0.45, f"{rows[0][0]} dominates at {rows[0][1]:.0%}"


def test_every_biome_has_a_colour():
    assert set(biomes.COLOURS) == set(biomes.NAMES)
