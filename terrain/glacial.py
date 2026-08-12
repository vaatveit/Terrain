"""Stage 5: glacial erosion.

Of the four processes the survey lists under secondary erosion, three cannot do anything at
continental resolution and it is worth being explicit about why rather than shipping three
no-ops. At n=785 the cells are 9.81 km across:

  * Thermal talus needs the slope between two cells to exceed the angle of repose. At 33
    degrees that is 6371 m of drop between neighbours, against 8 km of total relief in the
    whole world. No cell pair ever qualifies.
  * Dunes have wavelengths of 50 m to 2 km, and wave-cut notches are metre-scale. Both are
    two to four orders of magnitude below the grid.
  * Cirques run 0.5-3 km. Also sub-grid.

Glacial erosion is the exception, because its signature spans scales that survive: U-shaped
troughs are marginal at 1-10 km, but fjords (20-200 km) and ice sheets (500-4000 km) are
firmly resolved. It is also the process the survey calls "unmistakably wrong if absent from
high-latitude or high-altitude maps".

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
44.9 m of mean lowering under ice against 0.00 m off it, with a handful of exceptions where
routing shifted mid-run. It does NOT reliably flatten channel long profiles. That is the
textbook glacial signature and it did not survive measurement -- concavity in glaciated
channels went 0.507 -> 0.408 at n=256 (flattening, as predicted) but 0.559 -> 0.631 at
n=128 (steepening). The sign flips with resolution, so concavity is not a usable glacial
diagnostic at continental cell sizes and no test asserts it. Overdeepenings do form once
the driving slope is regional, but only a handful of cells at 30 km spacing -- they are
1-50 km features and sit right at the edge of what the grid can hold.
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
    # Abrasion law, E = K * Q_ice^m * S^n, calibrated to a glacial erosion rate of roughly
    # 0.1-10 mm/yr. m is lower than the fluvial 0.45 on purpose: a weaker dependence on
    # flux gives a flatter long profile, which is the U-shaped trough with its overdeepened
    # floor rather than a concave river valley.
    K: float = 2.0e-2
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
    max_incision_per_step_m: float = 8.0   # vertical, cut with the relief budget


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
