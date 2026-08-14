"""Stage 4: the coupled climate-erosion loop.

Erode, recompute climate on the new topography, re-erode with the new discharge, repeat.
This is the stage most implementations skip and the one that most improves plausibility at
continental scale, because without it every mountain range erodes at the same rate
regardless of which side of it the rain falls on.

The coupling runs through DISCHARGE, not through a fudged erodibility. Stream power is
properly E = K*Q^m*S^n, and drainage area is only a stand-in for discharge under uniform
rainfall; replacing the accumulated quantity with `area * P/mean(P)` restores the real
variable. Algebraically that is the same as scaling K by P^m, but it needs no change to the
solver, and it degenerates exactly to the uncoupled run when precipitation is uniform --
which is worth having as a regression test.

Why it is a loop rather than a single pass: precipitation depends on topography (air has to
be lifted over something), and topography depends on precipitation (wet slopes incise). Two
or three passes is enough; see `convergence` in the result for the measured evidence.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from terrain import climate as climate_mod
from terrain import stage2


@dataclasses.dataclass
class CoupledResult:
    elevation: np.ndarray
    climate: climate_mod.Climate
    discharge_weight: np.ndarray
    passes: int
    seconds: float
    # Per pass: (pass index, mean |dz| between passes, max elevation, mean precip on land)
    convergence: list


def _relief_neutral_scale(p, area, land_mask, floor, exponent, lo=1e-6, hi=1e6):
    """Find c such that weighting by max(c*P, floor) leaves mean relief unchanged.

    Steady-state slope goes as Q^(-m/n), so a cell's relief responds to w^(-m/n), not to w.
    Normalising the WEIGHT (by its mean or median) therefore does not normalise the relief:
    precipitation is heavily right-skewed, so the area-weighted mean of w^(-m/n) comes out
    well above 1 even when the median weight is exactly 1, and the coupling inflates the
    whole world rather than redistributing erosion within it.

    Solving for c directly is self-calibrating and needs no per-resolution tuning, which
    matters because the skew grows with resolution -- the same settings gave 7.7 km peaks
    at n=256 and 13.7 km at n=785.
    """
    a = np.asarray(area, dtype=np.float64)[land_mask]
    pl = np.asarray(p, dtype=np.float64)[land_mask]
    total = a.sum()
    if total <= 0 or pl.size == 0:
        return 1.0

    def mean_response(c):
        w = np.maximum(c * pl, floor)
        return float((a * w ** (-exponent)).sum() / total)

    # Monotonically decreasing in c, so bisect.
    if mean_response(lo) < 1.0:
        return lo
    if mean_response(hi) > 1.0:
        return hi
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if mean_response(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


DISCHARGE_FLOOR = 0.30
"""Minimum discharge weight, as a fraction of the typical cell.

One constant, referenced by both `discharge_weight` and `run`. They previously carried
separate defaults of 0.50 and 0.30, so every production run used 0.30 while the docstring
recommended 0.50 -- and the K calibration was done against 0.30 without that being written
down anywhere.
"""


def discharge_weight(area, precipitation_mm, land_mask, floor=DISCHARGE_FLOOR,
                     exponent=0.45):
    """Cell area scaled by precipitation, normalised so the mean weight is one cell area.

    Normalising matters: it keeps Q in area units, so K retains its calibrated meaning and
    the concavity target theta = m/n is untouched.

    The floor trades peak height against the wet/dry contrast this stage exists to create.
    Relief goes as Q^(-m/n), so an arid cell at half the typical discharge erodes to
    0.5^-0.45 = 1.4x the relief of a normal one. Measured at n=256 over three passes,
    against an uncoupled run peaking at 6688 m:

        floor        0.15    0.30    0.50
        peak (m)    11950    9531    7735
        wet-dry (m)  -293    -246    -220

    0.30 is the shipped value and the one K is calibrated against. It is also the right
    shape physically: arid basins erode slowly, not never -- episodic runoff, groundwater
    and hillslope processes all keep working when the rain does not.

    Note the floor sets the coupled/uncoupled peak ratio, at floor^(-m/n) = 1.37x, and that
    is a feature. Arid rain-shadow orogens are where the highest ground is, which is why
    the Altiplano and Tibet stand where they do. The absolute level is set by K, which is
    recalibrated for the coupled parameterisation -- see stage2.LemParams.
    """
    p = np.asarray(precipitation_mm, dtype=np.float64)
    mask = np.asarray(land_mask, dtype=bool)
    if not mask.any() or p[mask].max() <= 0:
        return np.ascontiguousarray(area, dtype=np.float32)

    # Normalise so that MEAN RELIEF is unchanged, not so that the mean or median weight is
    # one. Two earlier normalisations both failed, in opposite directions, and both looked
    # fine on the drainage metrics while getting the vertical scale badly wrong:
    #
    #   by the mean    typical cell far below 1 and against the floor, every cell eroding
    #                  ~1.7x too slowly -> 15.4 km peaks against 8.4 km uncoupled
    #   by the median  typical weight correct, but relief responds to w^(-m/n) and the
    #                  skew makes its mean exceed 1 -> 13.7 km peaks, worsening with
    #                  resolution as the precipitation distribution sharpens
    #
    # Solving for the scale that holds mean(w^(-m/n)) at 1 fixes the level by construction
    # and needs no per-resolution retuning.
    c = _relief_neutral_scale(p, area, mask, floor, exponent)
    scale = np.maximum(c * p, floor)
    return np.ascontiguousarray(area * scale, dtype=np.float32)


def run(grid, z0, uplift_m_per_myr, is_ocean, lem: stage2.LemParams = None,
        clim: climate_mod.ClimateParams = None, passes=3, steps_per_pass=200,
        discharge_floor=DISCHARGE_FLOOR, verbose=False, seasonal=True) -> CoupledResult:
    """Alternate erosion and climate until the pair stops moving.

    `seasonal` solves four points through the year and couples erosion to their annual mean,
    at about 4.5x the climate cost. It is on by default because the two annual fields are
    not interchangeable: measured against a single equinox solve, the seasonal annual total
    differs by 22% at the median cell and correlates only 0.82 in the log. The gap is the
    migrating rain band, which an equinox-only solve never moves -- so a monsoon belt comes
    out permanently dry, and erosion would then be coupled to a year that never happens.
    """
    lem = lem or stage2.LemParams()
    land_mask = ~np.asarray(is_ocean)
    z = np.ascontiguousarray(z0, dtype=np.float32).copy()
    solve = (climate_mod.compute_seasons if seasonal else climate_mod.compute_climate)

    t0 = time.perf_counter()
    convergence = []
    cl = None
    weight = None

    for p_i in range(passes):
        # Climate on the CURRENT topography -- this is the half of the loop that makes it
        # a loop rather than a one-off post-process.
        cl = solve(grid, z, land_mask, clim)
        weight = discharge_weight(grid.area, cl.precipitation_mm, land_mask,
                                  floor=discharge_floor, exponent=lem.m / lem.n)

        before = z.copy()
        res = stage2.evolve(grid, z, uplift_m_per_myr, is_ocean, lem,
                            n_steps=steps_per_pass, discharge_weight=weight)
        z = res.elevation

        moved = float(np.abs(z[land_mask] - before[land_mask]).mean())
        convergence.append((p_i, moved, float(z[land_mask].max()),
                            float(cl.precipitation_mm[land_mask].mean())))
        if verbose:
            print(f"    pass {p_i}: mean |dz| between passes {moved:8.1f} m   "
                  f"max z {z[land_mask].max():7.0f} m", flush=True)

    # One last climate pass so the returned fields describe the returned topography. Through
    # `solve`, not `compute_climate` -- calling the plain one here left the loop eroding on
    # a seasonal year while the world was exported carrying an equinox one, and the two
    # differ by 22% at the median cell.
    cl = solve(grid, z, land_mask, clim)
    return CoupledResult(elevation=z, climate=cl, discharge_weight=weight,
                         passes=passes, seconds=time.perf_counter() - t0,
                         convergence=convergence)
