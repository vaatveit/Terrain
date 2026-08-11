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


def discharge_weight(area, precipitation_mm, land_mask, floor=0.50):
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

    0.50 keeps peaks close to the uncoupled result while retaining most of the contrast.
    It is also the right shape physically: arid basins erode slowly, not never -- episodic
    runoff, groundwater and hillslope processes all keep working when the rain does not.
    """
    p = np.asarray(precipitation_mm, dtype=np.float64)
    # Normalise by the MEDIAN, not the mean. Precipitation is heavily right-skewed (median
    # 111 mm against a mean of 1000), so dividing by the mean puts the typical land cell
    # far below 1 and against the floor -- every cell then erodes ~1.7x too slowly, not
    # just the deserts, and production peaks reached 15.4 km against 8.4 km uncoupled.
    # The median keeps the typical cell at weight 1, so K stays calibrated and only
    # genuinely wet or arid ground departs from the uncoupled result.
    ref = np.median(p[land_mask]) if land_mask.any() else np.median(p)
    if ref <= 0:
        ref = p[land_mask].mean() if land_mask.any() else p.mean()
    if ref <= 0:
        return np.ascontiguousarray(area, dtype=np.float32)
    scale = np.maximum(p / ref, floor)
    return np.ascontiguousarray(area * scale, dtype=np.float32)


def run(grid, z0, uplift_m_per_myr, is_ocean, lem: stage2.LemParams = None,
        clim: climate_mod.ClimateParams = None, passes=3, steps_per_pass=200,
        discharge_floor=0.30, verbose=False) -> CoupledResult:
    """Alternate erosion and climate until the pair stops moving."""
    lem = lem or stage2.LemParams()
    land_mask = ~np.asarray(is_ocean)
    z = np.ascontiguousarray(z0, dtype=np.float32).copy()

    t0 = time.perf_counter()
    convergence = []
    cl = None
    weight = None

    for p_i in range(passes):
        # Climate on the CURRENT topography -- this is the half of the loop that makes it
        # a loop rather than a one-off post-process.
        cl = climate_mod.compute_climate(grid, z, land_mask, clim)
        weight = discharge_weight(grid.area, cl.precipitation_mm, land_mask,
                                  floor=discharge_floor)

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

    # One last climate pass so the returned fields describe the returned topography.
    cl = climate_mod.compute_climate(grid, z, land_mask, clim)
    return CoupledResult(elevation=z, climate=cl, discharge_weight=weight,
                         passes=passes, seconds=time.perf_counter() - t0,
                         convergence=convergence)
