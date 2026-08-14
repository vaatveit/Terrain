"""Stage 7b: Koppen-Geiger classification.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import climate, coupled, koppen, stage1  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


@pytest.fixture(scope="module")
def world():
    g = build_goldberg(96)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=2, steps_per_pass=150)
    return g, res.elevation, crust.land_mask, res.climate


def test_four_samples_reconstruct_themselves_exactly():
    """The whole argument for solving four seasons rather than two.

    Four evenly spaced points determine a two-harmonic series exactly, so evaluating the
    reconstruction back at the sample points must return the samples. If this drifts, the
    twelve monthly values Koppen reads are an assumption rather than an interpolation.
    """
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((4, 50))
    back = koppen.monthly_from_seasons(samples, months=4)
    assert np.abs(back - samples).max() < 1e-9


def test_the_reconstruction_is_a_smooth_year():
    """Twelve months from four samples must not ring between them. A pure annual cycle in,
    the same annual cycle out -- amplitude preserved, no harmonic invented."""
    phase = 2 * np.pi * np.arange(4) / 4
    samples = (10.0 + 5.0 * np.cos(phase))[:, None] * np.ones((1, 3))
    months = koppen.monthly_from_seasons(samples, months=12)
    t = 2 * np.pi * np.arange(12) / 12
    assert np.allclose(months[:, 0], 10.0 + 5.0 * np.cos(t), atol=1e-9)


def test_every_land_cell_is_classified_and_no_sea_cell_is(world):
    g, z, land, cl = world
    k = koppen.classify(cl.temperature_season, cl.precipitation_season, land, g.lat,
                        annual_precip_mm=cl.precipitation_mm)
    assert (k[land] >= 0).all(), "unclassified land"
    assert (k[~land] == -1).all(), "the sea is not a climate zone"
    assert set(np.unique(k[land]).tolist()).issubset(set(koppen.NAMES))


def test_aridity_outranks_temperature(world):
    """B is tested before everything else, which is the rule that makes Koppen Koppen: a
    hot desert is a desert first and tropical second. Whittaker cannot express that, because
    it thresholds temperature and rainfall independently."""
    seasons = np.full((4, 3), 25.0)                       # hot all year
    dry = np.full((4, 3), 5.0)                            # ~20 mm/yr
    land = np.ones(3, dtype=bool)
    k = koppen.classify(seasons, dry, land, np.zeros(3), annual_precip_mm=np.full(3, 20.0))
    assert (k == koppen.BWH).all(), f"hot and bone dry should be BWh, got {k}"


def test_polar_classes_need_a_cold_summer(world):
    """ET against EF is the warmest month either side of freezing, and that distinction only
    exists because the insolation model has polar day and night."""
    land = np.ones(2, dtype=bool)
    warm = np.array([[-20.0, -40.0], [2.0, -25.0], [-20.0, -40.0], [-30.0, -50.0]])
    wet = np.full((4, 2), 30.0)
    k = koppen.classify(warm, wet, land, np.array([80.0, 85.0]),
                        annual_precip_mm=np.full(2, 120.0))
    assert k[0] == koppen.ET, "a summer above freezing is tundra"
    assert k[1] == koppen.EF, "never above freezing is ice cap"


def test_the_classes_are_recognisably_earthlike(world):
    """Not a tight comparison -- this world is drier and flatter than Earth, and the shares
    show it. What is asserted is that every main group appears and none runs away."""
    g, z, land, cl = world
    k = koppen.classify(cl.temperature_season, cl.precipitation_season, land, g.lat,
                        annual_precip_mm=cl.precipitation_mm)
    share = {name: s for name, s in koppen.summary(k, g.area, land)}
    groups = {}
    for name, s in share.items():
        groups[name.split()[0][0]] = groups.get(name.split()[0][0], 0.0) + s
    for letter in "ABCDE":
        assert groups.get(letter, 0.0) > 0.005, f"no {letter} climates at all"
    assert groups["B"] < 0.60, f"arid classes ran away at {groups['B']:.0%} of land"


def test_koppen_separates_climates_whittaker_cannot(world):
    """The reason for having both. A summer-dry Mediterranean and a summer-wet subtropical
    climate can share an annual mean temperature and an annual rainfall total, so Whittaker
    must give them the same class; Koppen reads WHEN the rain falls and splits them."""
    land = np.ones(2, dtype=bool)
    t = np.array([[10.0, 10.0], [24.0, 24.0], [16.0, 16.0], [16.0, 16.0]])
    # Same annual total, opposite phase: dry in the warm season against wet in it.
    p = np.array([[80.0, 10.0], [5.0, 80.0], [40.0, 40.0], [40.0, 40.0]])
    k = koppen.classify(t, p, land, np.array([35.0, 35.0]),
                        annual_precip_mm=np.full(2, 660.0))
    assert k[0] != k[1], "identical annual means must still split on seasonality"
    assert k[0] == koppen.CSA, f"summer-dry should be Csa, got {koppen.NAMES[int(k[0])]}"
