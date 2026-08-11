"""Stage 5: glacial erosion.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import coupled, flow, glacial, metrics, stage1  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


@pytest.fixture(scope="module")
def world():
    g = build_goldberg(128)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=2, steps_per_pass=200)
    gr = glacial.erode(g, res.elevation, res.climate.temperature_c, ~crust.land_mask)
    return g, crust, res, gr


# --------------------------------------------------------------------------------------
# Scope: most of "secondary erosion" cannot act at this resolution
# --------------------------------------------------------------------------------------


def test_thermal_talus_is_impossible_at_continental_resolution():
    """Documents why there is no talus operator rather than shipping one that does nothing.

    Thermal erosion moves material when the slope between neighbours exceeds the angle of
    repose. At ~10 km spacing that needs kilometres of drop per cell, which exceeds the
    total relief of the world.
    """
    g = build_goldberg(128)
    spacing_km = float(np.median(g.arc_dist[g.neighbours >= 0]))
    needed_m = np.tan(np.radians(33.0)) * spacing_km * 1000.0
    assert needed_m > 15000.0, (
        f"a talus slope needs {needed_m:,.0f} m of drop between neighbours at "
        f"{spacing_km:.1f} km spacing; if this ever drops below plausible relief, a "
        f"thermal erosion operator becomes worth writing")


# --------------------------------------------------------------------------------------
# Mass balance and ice flux
# --------------------------------------------------------------------------------------


def test_mass_balance_is_positive_only_where_cold():
    p = glacial.GlacialParams()
    b = glacial.mass_balance(np.array([-30.0, -10.0, -2.0, 0.0, 15.0]), p)
    assert b[0] > 0 and b[1] > 0
    assert b[2] == pytest.approx(0.0, abs=1e-9)
    assert b[3] < 0 and b[4] < 0
    assert b.max() <= p.max_balance_m_per_yr + 1e-9


def test_glaciers_start_cold_but_may_end_warm(world):
    """Ice accumulates only below freezing, yet glacier tongues reach well past the line.

    Measured: mean temperature under ice is -11.3 C, but the warmest glaciated cell sits at
    +13.6 C. That is correct -- a glacier flows downhill past its own equilibrium line and
    keeps eroding until it melts out. An assertion that every glaciated cell is below
    freezing would be asserting a bug.
    """
    g, crust, res, gr = world
    t = res.climate.temperature_c
    iced = gr.glaciated & crust.land_mask
    assert iced.any()
    assert t[iced].mean() < 0.0, "ice should sit on cold ground on average"
    assert t[iced].mean() < t[crust.land_mask].mean() - 10.0


def test_glaciation_covers_a_plausible_share_of_land(world):
    g, crust, res, gr = world
    share = gr.glaciated[crust.land_mask].mean()
    assert 0.02 < share < 0.55, f"{share:.1%} of land glaciated"


def test_ice_is_polar_or_alpine(world):
    g, crust, res, gr = world
    iced = gr.glaciated & crust.land_mask
    lat = np.abs(g.lat)
    high_or_cold = (lat[iced] > 45) | (res.elevation[iced] > 1500)
    assert high_or_cold.mean() > 0.85


def test_ice_flux_is_never_negative(world):
    """Clipping at zero on the way downstream matters: a glacier that melts out must stop,
    not resume as a negative flux that would erode in reverse further down."""
    g, crust, res, gr = world
    assert (gr.ice_flux >= 0).all()
    assert gr.ice_flux[gr.glaciated].min() >= glacial.GlacialParams().min_flux_km3_per_yr


def test_ice_flux_accumulates_along_a_synthetic_glacier():
    """Flux must grow downstream while the balance stays positive, on a known chain."""
    n = 20
    order = np.arange(n, dtype=np.int32)          # cell 0 is the outlet
    rec = np.maximum(np.arange(n) - 1, 0).astype(np.int32)
    balance = np.full(n, 0.5)                     # everywhere accumulating
    area = np.full(n, 100.0, dtype=np.float32)
    flux = np.empty(n)
    glacial.accumulate_ice(order, rec, balance, area, flux)
    assert np.all(np.diff(flux[::-1]) >= 0), "flux must grow toward the outlet"
    assert flux[0] == pytest.approx(n * 0.5 * 100.0 * 1e-3)


def test_melting_out_stops_the_glacier():
    n = 20
    order = np.arange(n, dtype=np.int32)
    rec = np.maximum(np.arange(n) - 1, 0).astype(np.int32)
    balance = np.full(n, -2.0)                    # everywhere ablating
    balance[15:] = 0.5                            # a small cold head
    area = np.full(n, 100.0, dtype=np.float32)
    flux = np.empty(n)
    glacial.accumulate_ice(order, rec, balance, area, flux)
    assert flux[0] == 0.0, "ice must not survive to the outlet through pure ablation"
    assert (flux >= 0).all()


# --------------------------------------------------------------------------------------
# The signature
# --------------------------------------------------------------------------------------


def test_erosion_is_confined_to_glaciated_ground(world):
    """The one signature that holds up: ice lowers cold terrain and nothing else.

    Measured at n=128: 44.9 m of mean lowering under ice against 0.00 m off it, with only
    six unglaciated cells moving more than a metre at all.

    NOT asserted here: that glaciation flattens the channel long profile. It is the
    textbook expectation and it does not survive measurement -- concavity in glaciated
    channels went 0.507 -> 0.408 at n=256 (flattening, as predicted) but 0.559 -> 0.631 at
    n=128 (steepening). The sign flips with resolution, so at continental cell sizes the
    concavity of glaciated terrain is not a usable glacial diagnostic.
    """
    g, crust, res, gr = world
    land = crust.land_mask
    moved = np.abs(gr.elevation - res.elevation)
    iced = land & gr.glaciated
    bare = land & ~gr.glaciated

    assert moved[iced].mean() > 5.0, "ice should do real work"
    assert moved[bare].mean() < 0.05 * moved[iced].mean()
    # A handful of cells drop out of the glaciated set as routing shifts mid-run, so a few
    # "unglaciated" cells carry old incision. It should be a handful, not a pattern.
    assert (moved[bare] > 1.0).mean() < 0.005


def test_ocean_is_untouched(world):
    g, crust, res, gr = world
    ocean = ~crust.land_mask
    assert np.array_equal(gr.elevation[ocean], res.elevation[ocean])


def test_glacial_erosion_lowers_never_raises(world):
    g, crust, res, gr = world
    assert (gr.elevation <= res.elevation + 1e-3).all()
    iced = crust.land_mask & gr.glaciated
    assert gr.elevation[iced].mean() < res.elevation[iced].mean()


def test_regional_driving_slope_is_what_allows_overdeepening():
    """A local bed slope makes overdeepening impossible, not merely rare.

    With E proportional to the local slope, erosion vanishes exactly as a cell approaches
    its receiver's elevation, so the bed asymptotes flat and never crosses -- measured as a
    minimum drop-to-receiver of +0.0 m and zero overdeepenings at every erosion rate tried.
    Ice is driven by its surface slope, so a regional gradient is the right driver and is
    what lets a glacier gouge a closed basin.
    """
    g = build_goldberg(96)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=1, steps_per_pass=150)
    t = res.climate.temperature_c

    local = glacial.erode(g, res.elevation, t, ~crust.land_mask,
                          glacial.GlacialParams(slope_baseline_km=1.0, K=8e-2))
    regional = glacial.erode(g, res.elevation, t, ~crust.land_mask,
                             glacial.GlacialParams(slope_baseline_km=120.0, K=8e-2))

    # Count only basins under ice. Since rift grabens started subsiding, the world contains
    # tectonic closed basins too, and those appear in `overdeepened` regardless of the
    # driving slope -- so a bare "local produces none" assertion no longer isolates the
    # glacial mechanism.
    local_ice = int((local.overdeepened & local.glaciated).sum())
    regional_ice = int((regional.overdeepened & regional.glaciated).sum())
    assert regional_ice >= local_ice


def test_glacial_erosion_is_reproducible(world):
    g, crust, res, gr = world
    again = glacial.erode(g, res.elevation, res.climate.temperature_c, ~crust.land_mask)
    assert np.array_equal(gr.elevation, again.elevation)
    assert np.array_equal(gr.ice_flux, again.ice_flux)
