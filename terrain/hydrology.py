"""Stage 6: lakes, endorheic basins, and the river network.

Depression filling is a numerical device used inside the erosion loop -- it exists so water
has somewhere to go. This stage is where depressions become geography instead: some fill to
their spill point and overflow, and some never fill at all.

THE DISTINCTION THAT MATTERS is exorheic versus endorheic. A basin whose catchment delivers
more water than its lake surface can evaporate fills to the spill elevation and drains
onward; the Great Lakes work this way. A basin in an arid enough setting reaches equilibrium
far below its spill point, where evaporation from the lake surface exactly balances inflow,
and never overflows at all -- the Caspian, the Great Basin, Lake Chad. Filling every
depression to its brim, which is what the erosion loop does internally, turns every one of
those into an overflowing freshwater lake and quietly deletes a whole class of landscape.

Solving it needs a water balance rather than geometry, which is why this stage comes after
climate: runoff is precipitation minus evapotranspiration, and lake evaporation scales with
temperature. Both come from Stage 3.

Deltas and alluvial fans are deliberately absent: they are depositional landforms that
belong to the sediment-flux term in Stage 2, not something painted on at the end.

MEANDERS are absent for a weaker reason, and it is worth being honest about which. They used
to be dismissed as sub-grid at a 0.1-10 km wavelength against ~10 km cells. That figure was
an Earth-river figure, and it does not carry over: meander wavelength tracks channel width
(Leopold & Wolman, lambda ~ 11w), and this world's channels are metres wide, so its meanders
are hundreds of metres -- a few cells across on the largest rivers, sub-cell on the rest. The
biggest rivers here are therefore wide enough to meander at a resolved scale and do not,
which makes this a gap rather than a resolution limit.

River WIDTH is the encoding: `river_width_m` is metres, and zero means no river. There is no
separate boolean, because `width > 0` is exactly the same test -- the channel threshold puts
a 0.65 m floor under every cell that qualifies, so a mask would have been a second copy of
the same fact. Width stays sub-cell even after the rescale, but for the opposite reason it
used to: cells went from ~10 km to 100 m, and the basins on a 65 km sphere shrank with them,
so `90*sqrt(Q)` tops out in the tens of metres. That is precisely why the number is carried
rather than a flag -- nothing downstream can recover a river's size from cell geometry.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit

from terrain import flow


@dataclasses.dataclass
class HydrologyParams:
    # Thornthwaite-style annual potential evapotranspiration, mm/yr, from mean annual
    # temperature: 300 + 25T + 0.05T^3. About 300 mm at freezing, 600 at 10 C, 1700 at 25 C.
    pet_base_mm: float = 300.0
    pet_linear: float = 25.0
    pet_cubic: float = 0.05
    pet_max_mm: float = 2500.0
    # A lake counts as endorheic if it settles more than this far below its spill point.
    # Vertical, but cut with the relief budget: basins now hold 50-200 m of depth range
    # where they held 500-2,000 m, so 5 m would be a much stricter test than it was.
    endorheic_margin_m: float = 2.0
    # Channel definition and width. w = coeff * Q^exponent with Q in km^3/yr; the 0.5
    # exponent is the standard hydraulic-geometry value.
    #
    # Discharge is area x depth / time, so it scales as the radius SQUARED: 0.5 km^3/yr
    # / 98.125^2. That keeps the threshold at the same ~6 cells of drainage it marked at
    # Earth radius, and 6 cells is now 0.05 km^2 -- inside the real 0.01-1 km^2 channel-
    # head band, so the fluvial/hillslope split lands where it physically should.
    #
    # `width_coeff_m` is NOT rescaled. w = 90*sqrt(Q) is a physical hydraulic-geometry
    # relation, and Q already carries the smaller basins: a 100 km^2 catchment delivers
    # what a 100 km^2 catchment delivers, whatever planet it is on.
    channel_threshold_km3_per_yr: float = 5.2e-5
    width_coeff_m: float = 90.0
    width_exponent: float = 0.5
    min_lake_area_km2: float = 0.0415


@dataclasses.dataclass
class Hydrology:
    lake_mask: np.ndarray      # (N,) bool
    lake_level_m: np.ndarray   # (N,) f32, water surface where lake_mask, else nan
    lake_depth_m: np.ndarray   # (N,) f32
    lake_id: np.ndarray        # (N,) i32, -1 off-lake
    endorheic: np.ndarray      # (N,) bool, cells belonging to a terminal lake
    runoff_mm: np.ndarray      # (N,) f32, precipitation minus evapotranspiration
    discharge: np.ndarray      # (N,) f32, km^3/yr accumulated downstream
    river_width_m: np.ndarray  # (N,) f32, 0 where there is no river
    lakes: list                # per lake: dict of id, area, level, spill, endorheic


def potential_evapotranspiration(temperature_c, p: HydrologyParams):
    t = np.clip(np.asarray(temperature_c, dtype=np.float64), -20.0, 40.0)
    pet = p.pet_base_mm + p.pet_linear * t + p.pet_cubic * t**3
    return np.clip(pet, 0.0, p.pet_max_mm)


def runoff_mm(precipitation_mm, temperature_c, p: HydrologyParams):
    """Precipitation minus what the land gives back to the air, on the Budyko curve.

        AET = P / sqrt(1 + (P/PET)^2)        (Turc-Pike)

    The obvious form, AET = min(P, PET), is wrong in a way that matters here: it puts a
    KINK at P = PET and hands back exactly zero runoff everywhere drier than that. Measured
    on an n=128 world, that was 59% of all land -- 92% of the 15-30 degree belt, 84% of
    30-45, and 65% of the tropics -- so the subtropics carried no rivers at all, not thin
    ones. Real catchments do not work that way: rain arrives in events, and some of it runs
    off before the atmosphere can take it back, however arid the annual means look.

    Turc-Pike is the standard smooth alternative and has no free parameter. It approaches
    both physical limits correctly -- AET -> PET when water is plentiful, AET -> P when it
    is not -- but reaches them asymptotically, so runoff is small in a desert instead of
    identically zero. Same world: zero-runoff land 59% -> 0%, mean runoff 530 -> 600 mm.

    That the mean runoff RISES while the deserts get their trickle back is worth reading
    correctly. It is not extra water; this pipeline already produces more land runoff than
    Earth (600 mm against ~315). What it has is the wrong distribution -- see the note on
    `convective_efficiency` in ClimateParams -- and this function is downstream of that.
    """
    pet = potential_evapotranspiration(temperature_c, p)
    pr = np.maximum(np.asarray(precipitation_mm, dtype=np.float64), 0.0)
    ratio = pr / np.maximum(pet, 1e-9)
    return np.maximum(pr - pr / np.sqrt(1.0 + ratio * ratio), 0.0)


@njit(cache=True)
def _label_depressions(filled, z, nbr, tol, labels, stack):
    """Connected components of cells the fill had to raise."""
    n = filled.shape[0]
    k = nbr.shape[1]
    for i in range(n):
        labels[i] = -1
    n_lab = 0
    for seed in range(n):
        if labels[seed] >= 0 or filled[seed] <= z[seed] + tol:
            continue
        labels[seed] = n_lab
        sp = 0
        stack[sp] = seed
        sp += 1
        while sp > 0:
            sp -= 1
            c = stack[sp]
            for s in range(k):
                j = nbr[c, s]
                if j < 0 or labels[j] >= 0 or filled[j] <= z[j] + tol:
                    continue
                labels[j] = n_lab
                stack[sp] = j
                sp += 1
        n_lab += 1
    return n_lab


def analyse(grid, z, is_ocean, precipitation_mm, temperature_c,
            params: HydrologyParams = None) -> Hydrology:
    p = params or HydrologyParams()
    n = grid.cell_count
    z = np.ascontiguousarray(z, dtype=np.float32)
    nbr = np.ascontiguousarray(grid.neighbours)
    arc = np.ascontiguousarray(grid.arc_dist)
    area = np.ascontiguousarray(grid.area)
    is_ocean = np.ascontiguousarray(is_ocean)

    r_mm = runoff_mm(precipitation_mm, temperature_c, p)
    r_mm[is_ocean] = 0.0
    # mm/yr over km^2 -> km^3/yr
    local_volume = (r_mm * 1e-6) * area

    fr = flow.route(z, nbr, arc, area, is_ocean, metrics=False)
    discharge = np.empty(n, dtype=np.float32)
    flow.flow_accumulate(fr.order, fr.rec, local_volume.astype(np.float32), discharge)

    labels = np.empty(n, dtype=np.int32)
    stack = np.empty(n, dtype=np.int32)
    n_lab = _label_depressions(fr.filled, z, nbr, np.float32(1e-3), labels, stack)

    pet = potential_evapotranspiration(temperature_c, p)
    lake_level = np.full(n, np.nan, dtype=np.float32)
    lake_mask = np.zeros(n, dtype=bool)
    endorheic = np.zeros(n, dtype=bool)
    lake_id = np.full(n, -1, dtype=np.int32)
    lakes = []

    for lab in range(n_lab):
        cells = np.flatnonzero(labels == lab)
        if cells.size == 0:
            continue
        spill = float(fr.filled[cells].max())
        basin_area = float(area[cells].sum())
        if basin_area < p.min_lake_area_km2:
            continue

        # Inflow is everything the catchment delivers. The depression's outlet carries the
        # accumulated discharge of everything upstream of it, which is exactly that.
        inflow = float(discharge[cells].max())

        # Equilibrium level: raise the water until the lake surface evaporates as much as
        # arrives. Cells are sorted by bed elevation, so the surface at level L covers the
        # cells below L and its area is the running total.
        order = np.argsort(z[cells])
        beds = z[cells][order]
        cum_area = np.cumsum(area[cells][order])
        pet_lake = float(np.mean(pet[cells]))
        evap = cum_area * pet_lake * 1e-6           # km^3/yr at each candidate level

        idx = int(np.searchsorted(evap, inflow))
        if idx >= beds.size:
            level, is_endo = spill, False           # evaporation never catches up: overflows
        else:
            level = float(beds[idx])
            if level >= spill - p.endorheic_margin_m:
                level, is_endo = spill, False
            else:
                is_endo = True

        wet = cells[z[cells] < level]
        if wet.size == 0 or float(area[wet].sum()) < p.min_lake_area_km2:
            continue
        lake_mask[wet] = True
        lake_level[wet] = level
        lake_id[wet] = len(lakes)
        endorheic[wet] = is_endo
        lakes.append({"id": len(lakes), "cells": int(wet.size),
                      "area_km2": float(area[wet].sum()), "level_m": level,
                      "spill_m": spill, "endorheic": is_endo,
                      "inflow_km3_per_yr": inflow})

    depth = np.zeros(n, dtype=np.float32)
    depth[lake_mask] = lake_level[lake_mask] - z[lake_mask]

    river = (discharge >= p.channel_threshold_km3_per_yr) & ~is_ocean & ~lake_mask
    width = np.zeros(n, dtype=np.float32)
    width[river] = p.width_coeff_m * discharge[river] ** p.width_exponent

    return Hydrology(lake_mask=lake_mask, lake_level_m=lake_level, lake_depth_m=depth,
                     lake_id=lake_id, endorheic=endorheic,
                     runoff_mm=r_mm.astype(np.float32), discharge=discharge,
                     river_width_m=width, lakes=lakes)
