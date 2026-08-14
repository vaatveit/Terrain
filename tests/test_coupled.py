"""Stage 4: the coupled climate-erosion loop.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import coupled, stage1, stage2  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    g = build_goldberg(96)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    return g, crust


# --------------------------------------------------------------------------------------
# discharge_weight
# --------------------------------------------------------------------------------------


def test_uniform_precipitation_leaves_the_weight_equal_to_area(setup):
    g, crust = setup
    land = crust.land_mask
    w = coupled.discharge_weight(g.area, np.full(g.cell_count, 900.0), land)
    assert np.allclose(w, g.area, rtol=1e-5)


def test_normalisation_holds_mean_relief_response_at_one(setup):
    """The weight is normalised on w^(-m/n), the quantity relief actually responds to.

    Normalising the weight itself does not normalise the relief. Precipitation is heavily
    right-skewed, so the area-weighted mean of w^(-m/n) sits well above 1 even when the
    median weight is exactly 1, and the coupling then inflates the whole world instead of
    redistributing erosion within it.
    """
    g, crust = setup
    land = crust.land_mask
    rng = np.random.default_rng(0)
    # Deliberately skewed, like real precipitation.
    p = np.exp(rng.normal(0.0, 1.5, g.cell_count)) * 300.0

    for exponent in (0.3, 0.45, 0.6):
        w = coupled.discharge_weight(g.area, p, land, exponent=exponent) / g.area
        response = float((g.area[land] * w[land] ** -exponent).sum() / g.area[land].sum())
        assert response == pytest.approx(1.0, rel=1e-3), (
            f"mean w^-{exponent} is {response:.3f}, so coupling shifts global relief")


def test_normalisation_does_not_drift_with_resolution(setup):
    """The skew of the precipitation field sharpens with resolution.

    A median-based normalisation therefore inflated relief more at fine resolution than at
    coarse -- the coupled/uncoupled peak ratio ran 1.16x at n=256 and 1.63x at n=785. The
    relief-neutral scale is defined by an invariant, so it does not drift.
    """
    from terrain.grid import build_goldberg
    rng = np.random.default_rng(1)
    responses = []
    for n in (48, 72):
        gg = build_goldberg(n)
        land = np.ones(gg.cell_count, dtype=bool)
        p = np.exp(rng.normal(0.0, 1.8, gg.cell_count)) * 300.0
        w = coupled.discharge_weight(gg.area, p, land) / gg.area
        responses.append(float((gg.area * w**-0.45).sum() / gg.area.sum()))
    assert all(r == pytest.approx(1.0, rel=1e-3) for r in responses)


def test_weight_tracks_precipitation_and_respects_the_floor(setup):
    g, crust = setup
    land = crust.land_mask
    p = np.full(g.cell_count, 1000.0)
    p[:100] = 4000.0     # very wet
    p[100:200] = 0.0     # bone dry
    w = coupled.discharge_weight(g.area, p, land, floor=0.3)
    ratio = w / g.area
    assert ratio[:100].mean() > 3.0
    assert np.allclose(ratio[100:200], 0.3, rtol=1e-5), "arid cells must hit the floor, not zero"
    assert (w > 0).all()


# --------------------------------------------------------------------------------------
# Coupling
# --------------------------------------------------------------------------------------


def test_uniform_precipitation_reproduces_the_uncoupled_run_exactly(setup):
    """The coupling must degenerate to Stage 2 when climate is flat.

    Discharge-weighted accumulation is only a reparameterisation of drainage area, so with
    P constant it has to give bit-identical output -- not merely close. Anything else means
    the weighting has smuggled in a scale factor and K no longer means what it was
    calibrated to mean.
    """
    g, crust = setup
    is_ocean = ~crust.land_mask
    p = stage2.LemParams()

    base = stage2.evolve(g, crust.elevation, crust.uplift, is_ocean, p, n_steps=80)
    w = coupled.discharge_weight(g.area, np.ones(g.cell_count), crust.land_mask)
    weighted = stage2.evolve(g, crust.elevation, crust.uplift, is_ocean, p, n_steps=80,
                             discharge_weight=w)
    assert np.array_equal(base.elevation, weighted.elevation)


def test_wet_ground_incises_more_than_arid_ground(setup):
    """The whole point of Stage 4: wet windward slopes cut deeper, arid lee slopes hold up.

    Measured against an uncoupled run of the same total duration, so the comparison is
    about WHERE the erosion went, not how much there was.
    """
    g, crust = setup
    is_ocean = ~crust.land_mask
    land = crust.land_mask

    uncoupled = stage2.evolve(g, crust.elevation, crust.uplift, is_ocean,
                              stage2.LemParams(), n_steps=450).elevation
    res = coupled.run(g, crust.elevation, crust.uplift, is_ocean,
                      passes=3, steps_per_pass=150)

    p = res.climate.precipitation_mm
    wet = land & (p > np.percentile(p[land], 75))
    dry = land & (p < np.percentile(p[land], 25))

    wet_change = res.elevation[wet].mean() - uncoupled[wet].mean()
    dry_change = res.elevation[dry].mean() - uncoupled[dry].mean()
    assert wet_change < dry_change, (
        f"coupling moved wet ground by {wet_change:+.0f} m and arid ground by "
        f"{dry_change:+.0f} m; wet should end up relatively lower")


def test_coupling_makes_elevation_anticorrelate_with_rainfall(setup):
    g, crust = setup
    is_ocean = ~crust.land_mask
    land = crust.land_mask

    uncoupled = stage2.evolve(g, crust.elevation, crust.uplift, is_ocean,
                              stage2.LemParams(), n_steps=450).elevation
    res = coupled.run(g, crust.elevation, crust.uplift, is_ocean,
                      passes=3, steps_per_pass=150)

    lp = np.log10(res.climate.precipitation_mm[land] + 1.0)
    before = np.corrcoef(lp, uncoupled[land])[0, 1]
    after = np.corrcoef(lp, res.elevation[land])[0, 1]
    assert after < before, f"correlation went {before:+.3f} -> {after:+.3f}"


def test_successive_passes_move_the_world_less(setup):
    """Two or three passes should be enough; check the loop is actually settling."""
    g, crust = setup
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=4, steps_per_pass=150)
    moved = [m for _, m, _, _ in res.convergence]
    assert moved[-1] < moved[0], f"passes are not settling: {moved}"
    assert moved[-1] < 0.35 * moved[0]


def test_coupled_run_is_reproducible(setup):
    g, crust = setup
    a = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                    passes=2, steps_per_pass=80)
    b = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                    passes=2, steps_per_pass=80)
    assert np.array_equal(a.elevation, b.elevation)
    assert np.array_equal(a.climate.precipitation_mm, b.climate.precipitation_mm)


def test_returned_climate_describes_the_returned_topography(setup):
    """Easy to get wrong: returning the climate computed on the PREVIOUS pass's terrain."""
    g, crust = setup
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=2, steps_per_pass=80)
    from terrain import climate as climate_mod
    # Seasonal, matching what `run` solves with. It used to close with a plain
    # `compute_climate` while the loop ran on `compute_seasons`, so the world was eroded
    # against a year and exported carrying an equinox -- two fields that differ by 22% at
    # the median cell.
    fresh = climate_mod.compute_seasons(g, res.elevation, crust.land_mask)
    assert np.array_equal(res.climate.precipitation_mm, fresh.precipitation_mm)
    assert np.array_equal(res.climate.temperature_season, fresh.temperature_season)


def test_arid_high_ground_keeps_more_relief_than_wet_high_ground(setup):
    """Rain shadows should show up as topography, not just as a precipitation map."""
    g, crust = setup
    land = crust.land_mask
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=3, steps_per_pass=150)
    z, p = res.elevation, res.climate.precipitation_mm

    # Compare like with like: cells under comparable uplift, split by rainfall.
    u = crust.uplift
    orogen = land & (u > np.percentile(u[land], 80))
    wet = orogen & (p > np.percentile(p[orogen], 60))
    dry = orogen & (p < np.percentile(p[orogen], 40))
    assert wet.sum() > 20 and dry.sum() > 20
    assert z[dry].mean() > z[wet].mean(), (
        f"arid orogen {z[dry].mean():.0f} m vs wet orogen {z[wet].mean():.0f} m")
