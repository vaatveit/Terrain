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

Sub-grid and deliberately absent: meander migration (0.1-10 km wavelength), and deltas and
alluvial fans, which are depositional landforms that belong to the sediment-flux term in
Stage 2 rather than being painted on at the end. River WIDTH is carried as a per-cell
attribute since it is wanted downstream for rendering, but at ~10 km cells no river is
geometrically resolved -- the Amazon at its mouth would be one cell across.
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
    endorheic_margin_m: float = 5.0
    # Channel definition and width. w = coeff * Q^exponent with Q in km^3/yr; the 0.5
    # exponent is the standard hydraulic-geometry value.
    channel_threshold_km3_per_yr: float = 0.5
    width_coeff_m: float = 90.0
    width_exponent: float = 0.5
    min_lake_area_km2: float = 400.0


@dataclasses.dataclass
class Hydrology:
    lake_mask: np.ndarray      # (N,) bool
    lake_level_m: np.ndarray   # (N,) f32, water surface where lake_mask, else nan
    lake_depth_m: np.ndarray   # (N,) f32
    lake_id: np.ndarray        # (N,) i32, -1 off-lake
    endorheic: np.ndarray      # (N,) bool, cells belonging to a terminal lake
    runoff_mm: np.ndarray      # (N,) f32, precipitation minus evapotranspiration
    discharge: np.ndarray      # (N,) f32, km^3/yr accumulated downstream
    river_mask: np.ndarray     # (N,) bool
    river_width_m: np.ndarray  # (N,) f32
    lakes: list                # per lake: dict of id, area, level, spill, endorheic


def potential_evapotranspiration(temperature_c, p: HydrologyParams):
    t = np.clip(np.asarray(temperature_c, dtype=np.float64), -20.0, 40.0)
    pet = p.pet_base_mm + p.pet_linear * t + p.pet_cubic * t**3
    return np.clip(pet, 0.0, p.pet_max_mm)


def runoff_mm(precipitation_mm, temperature_c, p: HydrologyParams):
    """Precipitation minus what the land gives back to the air.

    Actual evapotranspiration cannot exceed either the water available or the atmospheric
    demand, so AET = min(P, PET) and runoff is whatever survives that.
    """
    pet = potential_evapotranspiration(temperature_c, p)
    return np.maximum(np.asarray(precipitation_mm, dtype=np.float64) - pet, 0.0)


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
                     river_mask=river, river_width_m=width, lakes=lakes)
