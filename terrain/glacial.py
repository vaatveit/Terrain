"""Stage 5: glacial erosion.

Glacial erosion is the only one of the survey's four secondary processes that is implemented,
and the reason is no longer the one this docstring used to give. At the old 9.81 km cells the
other three were flatly sub-grid; at 100 m cells they are not, and the honest statement is
that they are unbuilt rather than impossible:

  * Thermal talus needs the slope between two cells to exceed the angle of repose. At 9.81 km
    cells, 33 degrees was 6,371 m of drop between neighbours and no cell pair could ever
    qualify. At 100 m cells it is 65 m, which mountain terrain clears easily -- the shipped
    world measures p99 62.5 degrees and max 83.7. So a large share of the steep ground sits
    PAST repose and stays there, because nothing removes it. This is the biggest gap here.
  * Dunes (0.05-2 km) and cirques (0.5-3 km) were two to four orders of magnitude below the
    old grid and are now resolved. Wave-cut notches are metre-scale and remain marginal.

Glacial erosion is still the one worth building first, because its signature spans the scales
this world can hold -- U-shaped troughs at 1-10 km and fjords at 20-200 km -- and because the
survey calls it "unmistakably wrong if absent from high-latitude or high-altitude maps". Note
the trade ran the other way at the top of the range: ice sheets are 500-4,000 km features and
this sphere is 64.9 km in radius, so continental glaciation is no longer something this world
can express. See terrain-algorithms.md, "Stage 5: secondary erosion", for the full table.

THE ONE THING THAT MAKES GLACIAL TERRAIN LOOK GLACIAL, at this resolution, is that ice can
carve below its own outlet. Rivers cannot: water needs a downhill path, so the fluvial
solver forbids cutting below the receiver, and every fluvial valley drains monotonically to
the sea. Ice is driven by its SURFACE slope, not the bed, so a glacier will happily gouge a
closed basin and flow up out of it. That is what overdeepenings, fjord sills and the lake
districts of formerly glaciated shields are. Allowing it here -- and only here -- is what
separates the two erosion styles, and the depressions it leaves become lakes when Stage 6
runs its fill.

The mass balance and flux model is deliberately simple: accumulate where the Stage 3
temperature is below freezing, ablate where it is not, route ice down the existing flow
graph, and erode as a power law on ice flux. This is not a shallow-ice-approximation
solver; it is the abrasion law with a plausible flux, which is what the resolved scales can
actually support.

WHAT THIS DOES AND DOES NOT DELIVER, measured. It reliably confines erosion to cold ground:
mean lowering under ice against 0.00 m off it, with a handful of exceptions where routing
shifted mid-run. It does NOT reliably flatten channel long profiles. That is the
textbook glacial signature and it did not survive measurement -- concavity in glaciated
channels went 0.507 -> 0.408 at n=256 (flattening, as predicted) but 0.559 -> 0.631 at
n=128 (steepening). The sign flips with resolution, so concavity is not a usable glacial
diagnostic at continental cell sizes and no test asserts it. Overdeepenings do form once the
driving slope is regional; the count of "only a handful of cells" was measured at 30 km
spacing, where a 1-50 km feature was at the edge of what the grid could hold, and has not
been re-measured since the rescale put 10 to 500 cells across the same feature.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit

from terrain import flow


@dataclasses.dataclass
class GlacialParams:
    # Mass balance is driven by TEMPERATURE, not by an elevation-only ELA. Stage 3's
    # temperature field already carries the lapse rate, so it encodes latitude and altitude
    # together -- which is what actually decides where ice forms. A latitude-only ELA put
    # ice on 1.2% of land and missed low-lying polar shields entirely.
    freeze_c: float = -2.0                # accumulation below this mean annual temperature
    balance_per_degc: float = 0.35        # m of ice per year per degree below freezing
    max_balance_m_per_yr: float = 3.0
    # Abrasion law, E = K * Q_ice^m * S^n. m is lower than the fluvial 0.45 on purpose: a
    # weaker dependence on flux gives a flatter long profile, which is the U-shaped trough
    # with its overdeepened floor rather than a concave river valley.
    #
    # K WAS 2.0e-2 AND THAT WAS TOO HIGH, though not for the reason the old comment implied.
    # It cut 148 m of mean lowering under ice over the 300 kyr run, which is 0.49 mm/yr and
    # sits comfortably inside the published 0.1-10 mm/yr band -- so "calibrated to the band"
    # was true and still wrong. The band spans two orders of magnitude, and which END of it
    # belongs here is a question about this world's relief, not about glaciers in general.
    # The pre-glacial relief budget is 432 m; Earth's glaciated terrain has several km. At
    # 0.49 mm/yr the ice was removing 34% of the world's entire relief in 300 kyr, against
    # the 5-10% that Quaternary glaciation took off Earth's shields.
    #
    # This was invisible until the reflood pass (Stage 5b) went in, because the overdeepened
    # ground stayed dry: it read as odd elevations rather than as a drowned continent.
    #
    # Cheap to fix, because K turns out to be a pure incision-depth knob. Swept at n=128,
    # the glaciated fraction (21.8-22.1%), the overdeepening count (404-417) and the lakes
    # (11-15, ~1.3% of land) are all flat across a twentyfold change in K -- so lowering it
    # costs none of the glacial signature, which is the whole reason this stage exists.
    #
    #     K        lowering  rate      % of relief  drowned  Koppen E
    #     2.0e-2    148 m    0.49mm/yr    34.3%      21.4%     4.7%
    #     1.0e-2     95 m    0.32mm/yr    21.9%      18.7%     7.8%
    #     5.0e-3     57 m    0.19mm/yr    13.1%      16.0%    10.8%
    #     3.0e-3     37 m    0.12mm/yr     8.6%      14.2%    12.7%   <- ships
    #     2.0e-3     25 m    0.08mm/yr     5.9%      12.7%    14.2%
    #     1.0e-3     13 m    0.04mm/yr     3.0%      10.0%    16.8%
    #
    # 3.0e-3 is the value that satisfies all three criteria at once: the rate stays inside
    # the published band, the share of relief lands inside Earth's 5-10%, and Koppen's polar
    # share comes out at 12.7% against Earth's 14%. 2.0e-3 scores better on the last two but
    # falls below the published rate floor, which is not a trade worth making for a
    # classification share.
    #
    # WHAT THIS DOES NOT FIX is that 14.2% of land still floods. Only 1.0% does with the ice
    # switched off entirely, so the drowning really is glacial -- but it is amplified out of
    # all proportion by hypsometry. Median land elevation on these worlds is about 10 m, so
    # ground that is lowered at all tends to go under. That is the relief shortfall showing
    # up somewhere new, not a glacial parameter left mistuned.
    K: float = 3.0e-3
    m: float = 0.30
    n: float = 1.0
    # Ice flux is volume/time, so it scales as the radius squared: 1e-2 / 98.125^2.
    min_flux_km3_per_yr: float = 1.04e-6   # below this there is snow, not a glacier
    # Baseline over which the driving slope is measured. Long on purpose -- see `abrade`.
    # CELL-scale, not world-scale: it is 12 cells of flow path, which is what made 120 km
    # right at 9.77 km cells and makes 1.22 km right at 100 m. Cell size no longer varies
    # with n, so this is a constant and must NOT be rescaled with the radius.
    slope_baseline_km: float = 1.22
    dt_yr: float = 5.0e3
    steps: int = 60
    # Per-step incision ceiling. Vertical, cut with the relief budget -- and it is a LIVE
    # control, not a safety rail that never trips, which is worth knowing before reading the
    # K table above as though K alone set the rate. Measured on an n=128 world: 7,417 land
    # cells hit this clamp, and the deepest total incision comes out at exactly 8.0 x 60 =
    # 480.00 m, meaning the fastest-eroding cells are clamped on every single step. Wherever
    # that is true the local erosion rate is set here rather than by K * Q^m * S^n, so the two
    # knobs are not independent and a K sweep alone does not describe the model.
    max_incision_per_step_m: float = 8.0


def mass_balance(temperature_c, p: GlacialParams):
    """Net annual ice balance, m/yr. Positive where it is cold enough, negative where not.

    Driven by the Stage 3 temperature field, which already includes the elevation lapse
    rate, so a cell is glaciated for being high, for being far north, or for both -- and
    a warm high plateau is correctly left ice-free.
    """
    return np.clip((p.freeze_c - np.asarray(temperature_c, dtype=np.float64))
                   * p.balance_per_degc,
                   -p.max_balance_m_per_yr, p.max_balance_m_per_yr)


@njit(cache=True)
def accumulate_ice(order, rec, balance, area_km2, flux):
    """Route ice down the flow graph, melting it out where the balance is negative.

    Clipped at zero on the way down: a glacier that runs out of ice stops, and does not
    resume as a negative flux that would erode in reverse further downstream.
    """
    n = order.shape[0]
    for i in range(n):
        flux[i] = balance[i] * area_km2[i] * 1.0e-3   # m/yr * km^2 -> km^3/yr
    for t in range(n - 1, -1, -1):
        c = order[t]
        if flux[c] < 0.0:
            flux[c] = 0.0
        r = rec[c]
        if r != c:
            flux[r] += flux[c]
    for i in range(n):
        if flux[i] < 0.0:
            flux[i] = 0.0
    return flux


@njit(cache=True)
def abrade(z, order, rec, driving_slope, flux, is_fixed, K, m, n, dt_yr, min_flux,
           max_incision, out):
    """One explicit abrasion step.

    Two things differ from the fluvial solver, and together they are what let ice do things
    water cannot.

    There is no clamp preventing a cell from being cut below its receiver. Water needs a
    downhill path; ice does not.

    And the driving slope is REGIONAL, measured over a long flow path, not the local drop to
    the receiver. That distinction turned out to be the whole game. With a local slope,
    E goes as S and therefore vanishes exactly as a cell approaches its receiver's
    elevation -- so the bed asymptotes to flat and never crosses, and removing the clamp
    achieves nothing (measured: minimum drop-to-receiver +0.0 m, zero overdeepenings, at
    every erosion rate tried). Physically the local bed slope is the wrong driver anyway:
    ice is driven by its SURFACE slope and will flow over a flat or reversed bed quite
    happily. A regional gradient keeps driving erosion after the local one has closed up,
    which is how overdeepenings, fjord sills and glacial lake basins actually form.
    """
    n_cells = order.shape[0]
    for i in range(n_cells):
        out[i] = z[i]
    for t in range(n_cells):
        i = order[t]
        r = rec[i]
        if is_fixed[i] or r == i or flux[i] < min_flux:
            continue
        slope = driving_slope[i]
        if slope < 0.0:
            slope = 0.0
        e = K * flux[i] ** m * slope ** n * dt_yr
        if e > max_incision:
            e = max_incision
        out[i] = z[i] - e
    return out


@dataclasses.dataclass
class GlacialResult:
    elevation: np.ndarray
    ice_flux: np.ndarray      # km^3/yr
    glaciated: np.ndarray     # bool
    overdeepened: np.ndarray  # bool, cells now below their own outlet
    total_incision_m: float


def erode(grid, z0, temperature_c, is_ocean, params: GlacialParams = None,
          verbose=False) -> GlacialResult:
    p = params or GlacialParams()
    z = np.ascontiguousarray(z0, dtype=np.float64).copy()
    nbr = np.ascontiguousarray(grid.neighbours)
    arc = np.ascontiguousarray(grid.arc_dist)
    area = np.ascontiguousarray(grid.area)
    is_fixed = np.ascontiguousarray(is_ocean)
    balance = mass_balance(temperature_c, p)

    n = grid.cell_count
    flux = np.empty(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    driving = np.empty(n, dtype=np.float32)
    start = z.copy()

    for s in range(p.steps):
        zf = z.astype(np.float32)
        fr = flow.route(zf, nbr, arc, area, is_fixed, metrics=False)
        accumulate_ice(fr.order, fr.rec, balance, area, flux)
        flow.flow_path_slope(fr.filled, fr.rec, fr.step,
                             np.float32(p.slope_baseline_km), driving)
        abrade(z, fr.order, fr.rec, driving.astype(np.float64), flux, is_fixed,
               p.K, p.m, p.n, p.dt_yr, p.min_flux_km3_per_yr,
               p.max_incision_per_step_m, out)
        z, out = out, z
        if verbose and (s % max(1, p.steps // 6) == 0 or s == p.steps - 1):
            g = flux >= p.min_flux_km3_per_yr
            print(f"    step {s:3d}: glaciated {g.mean():6.2%}  "
                  f"max ice flux {flux.max():8.2f} km^3/yr  "
                  f"eroded {float(np.abs(z - start).mean()):6.2f} m mean", flush=True)

    glaciated = flux >= p.min_flux_km3_per_yr
    fr = flow.route(z.astype(np.float32), nbr, arc, area, is_fixed, metrics=False)
    # A cell sitting below the neighbour it drains into: only ice can make one of these.
    below = np.zeros(n, dtype=bool)
    moving = fr.rec != np.arange(n)
    below[moving] = z[moving] < z[fr.rec[moving]] - 1.0

    return GlacialResult(
        elevation=z.astype(np.float32), ice_flux=flux.astype(np.float32),
        glaciated=glaciated, overdeepened=below & ~is_ocean,
        total_incision_m=float(np.abs(z - start).sum()))
