"""Plate tectonics on the sphere: seeds, Euler-pole motion, boundary classification, uplift.

On a sphere this is the real model rather than an approximation. Plate motion IS rotation
about an Euler pole -- an axis unit vector and an angular rate -- and the velocity at any
point is the cross product of the two. A planar velocity field is a hack for something that
is genuinely rotational, and it cannot be made consistent over a whole world.

Convenient unit coincidence used throughout: 1 km/Myr = 1 mm/yr exactly, so plate speeds
read the same whichever unit you have in mind.

What plates are NOT used for here: deciding where continents are. Crust type comes from
noise, independent of the plate layout, because six of Earth's seven major plates carry
both continental and oceanic crust. Plates drive boundary type, stress, crustal thickening
and uplift -- see stage1.generate_crust_tectonic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit

# boundary_type encoding
NONE, CONVERGENT, DIVERGENT, TRANSFORM = 0, 1, 2, 3
# margin_class encoding
INLAND, PASSIVE, ACTIVE = 0, 1, 2


@dataclasses.dataclass
class PlateParams:
    plate_count: int = 18            # 12-30; drives orogen layout, not continent shape
    lloyd_iterations: int = 3        # more = more uniform plates, less natural variety
    min_separation_deg: float = 18.0
    max_speed_mm_per_yr: float = 80.0
    # A boundary counts as transform only when tangential motion exceeds this multiple of
    # the normal component. The naive threshold of 1.0 classified 44% of boundaries as
    # transform against Earth's ~20%: random Euler poles make relative motion oblique far
    # more often than a real plate system, where ridges and trenches organise the motion
    # that creates the plates in the first place. Raising the bar restores a plausible mix
    # without pretending to simulate the organisation.
    transform_ratio: float = 2.5
    # Domain warp applied to the Voronoi lookup, so plate boundaries -- and therefore the
    # orogens that track them -- are not dead-straight great-circle arcs.
    boundary_warp: float = 0.10
    boundary_warp_frequency: float = 2.5
    # Uplift geometry, from the boundary-case table in terrain-algorithms.md.
    orogen_decay_km: float = 200.0   # continental collision, peaked on the boundary
    arc_offset_km: float = 150.0     # volcanic arc, inland of an oceanic-continental trench
    arc_width_km: float = 100.0
    rift_shoulder_km: float = 80.0   # uplifted flanks either side of a continental graben
    # Crustal thickness response.
    convergent_thickening_km: float = 15.0
    divergent_thinning_km: float = 8.0
    # Magnitudes. Published rock uplift for active orogens spans 500-5000 m/Myr; this sits
    # at the low end deliberately. Steady-state relief goes as U/K, and at the default
    # K=2.5e-6 an orogen uplift of 3000 m/Myr drove peaks to 19.3 km. Raising K by the same
    # factor would work equally well physically, but keeping K identical across both Stage 1
    # variants means an A/B comparison of crust models is not confounded by a different
    # erodibility, so the uplift is what gives.
    u_max_m_per_myr: float = 1200.0
    craton_floor_m_per_myr: float = 20.0


@dataclasses.dataclass
class Plates:
    plate_id: np.ndarray         # (N,) i16
    sites: np.ndarray            # (P, 3) unit vectors
    euler_axis: np.ndarray       # (P, 3) unit vectors
    euler_rate: np.ndarray       # (P,) rad/Myr
    boundary_type: np.ndarray    # (N,) u8
    boundary_stress: np.ndarray  # (N,) f32, signed normal velocity, mm/yr
    boundary_dist_km: np.ndarray  # (N,) f32, great-circle distance to nearest boundary
    nearest_boundary: np.ndarray  # (N,) i32, index of that boundary cell


# --------------------------------------------------------------------------------------
# 1-2. Seeds and partition
# --------------------------------------------------------------------------------------


def seed_plates(grid, params: PlateParams, seed=0):
    """Poisson-disk sample on the sphere, then Lloyd-relax against real cell areas.

    Rejection sampling with a minimum great-circle separation. Without it, uniform random
    seeds give a lottery distribution of plate sizes -- a few enormous plates and a scatter
    of slivers.
    """
    rng = np.random.default_rng(seed)
    min_dot = np.cos(np.radians(params.min_separation_deg))
    sites = []
    separation = min_dot
    for attempt in range(20000):
        p = rng.standard_normal(3)
        p /= np.linalg.norm(p)
        if not sites or np.max(np.asarray(sites) @ p) < separation:
            sites.append(p)
            if len(sites) == params.plate_count:
                break
        if attempt % 2000 == 1999:
            separation = 1.0 - (1.0 - separation) * 1.3  # relax if we are stuck
    sites = np.asarray(sites)

    # Lloyd: move each seed to the area-weighted centroid of the cells it owns.
    for _ in range(params.lloyd_iterations):
        owner = np.argmax(grid.xyz @ sites.T, axis=1)
        for p in range(sites.shape[0]):
            m = owner == p
            if not m.any():
                continue
            c = (grid.xyz[m] * grid.area[m, None]).sum(axis=0)
            norm = np.linalg.norm(c)
            if norm > 0:
                sites[p] = c / norm
    return sites


def partition(grid, sites, params: PlateParams = None, seed=0):
    """Spherical Voronoi: nearest seed by maximum dot product. No geometry required.

    The lookup is done at DOMAIN-WARPED positions rather than the true ones. Raw Voronoi
    boundaries are exact great-circle arcs, and since orogens track plate boundaries, that
    renders as mountain ranges running in dead-straight lines -- unmistakably artificial.
    Warping the query position bends the boundaries without changing the partition's
    topology: every cell still belongs to exactly one plate and plates stay contiguous.

    Multi-octave warp on purpose. A single frequency gives smooth sine-like wobbles that
    read as fake just as quickly as the straight lines did.
    """
    p = params or PlateParams()
    if p.boundary_warp <= 0.0:
        return np.argmax(grid.xyz @ sites.T, axis=1).astype(np.int16)

    from terrain import noise

    xyz = np.ascontiguousarray(grid.xyz)
    n = xyz.shape[0]
    warped = xyz.copy()
    out = np.empty(n, dtype=np.float64)
    for axis_i, tag in enumerate(("wx", "wy", "wz")):
        perm = noise.permutation(noise.sub_seed(tag, seed))
        noise.fbm(xyz, perm, noise.GRAD3, 4, p.boundary_warp_frequency, 2.0, 0.5, out)
        warped[:, axis_i] += p.boundary_warp * out
    warped /= np.linalg.norm(warped, axis=1, keepdims=True)
    return np.argmax(warped @ sites.T, axis=1).astype(np.int16)


def assign_motion(n_plates, params: PlateParams, seed=0):
    """A random Euler pole and angular rate per plate.

    Rate is chosen so the fastest surface speed lands near max_speed_mm_per_yr. Earth's
    plates run roughly 10-100 mm/yr.
    """
    rng = np.random.default_rng(seed + 991)
    axis = rng.standard_normal((n_plates, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    # v = omega * R * sin(theta); take sin(theta) ~ 1 for the scaling.
    max_rate = params.max_speed_mm_per_yr / 6371.0  # rad/Myr
    rate = rng.uniform(0.15, 1.0, n_plates) * max_rate
    return axis, rate


def plate_velocity(xyz, axis, rate, plate_id):
    """Surface velocity in km/Myr (== mm/yr), as the Euler-pole cross product."""
    return np.cross(axis[plate_id] * rate[plate_id, None], xyz) * 6371.0


# --------------------------------------------------------------------------------------
# 3. Boundaries
# --------------------------------------------------------------------------------------


@njit(cache=True)
def _classify_boundaries(xyz, nbr, plate_id, vel, transform_ratio, b_type, b_stress):
    """Per boundary cell, decompose relative plate motion into normal and tangential."""
    n = xyz.shape[0]
    k = nbr.shape[1]
    for i in range(n):
        b_type[i] = NONE
        b_stress[i] = 0.0
        best = 0.0
        for s in range(k):
            j = nbr[i, s]
            if j < 0 or plate_id[j] == plate_id[i]:
                continue

            # Tangent direction from i toward j, projected onto the local tangent plane.
            dot = xyz[j, 0] * xyz[i, 0] + xyz[j, 1] * xyz[i, 1] + xyz[j, 2] * xyz[i, 2]
            nx = xyz[j, 0] - dot * xyz[i, 0]
            ny = xyz[j, 1] - dot * xyz[i, 1]
            nz = xyz[j, 2] - dot * xyz[i, 2]
            norm = np.sqrt(nx * nx + ny * ny + nz * nz)
            if norm <= 0.0:
                continue
            nx, ny, nz = nx / norm, ny / norm, nz / norm

            dvx = vel[i, 0] - vel[j, 0]
            dvy = vel[i, 1] - vel[j, 1]
            dvz = vel[i, 2] - vel[j, 2]
            normal = dvx * nx + dvy * ny + dvz * nz  # >0: i moves toward j
            tx = dvx - normal * nx
            ty = dvy - normal * ny
            tz = dvz - normal * nz
            tangential = np.sqrt(tx * tx + ty * ty + tz * tz)

            magnitude = np.sqrt(normal * normal + tangential * tangential)
            if magnitude <= best:
                continue
            best = magnitude
            b_stress[i] = normal
            if tangential > transform_ratio * abs(normal):
                b_type[i] = TRANSFORM
            else:
                b_type[i] = CONVERGENT if normal > 0.0 else DIVERGENT
    return b_type, b_stress


@njit(cache=True)
def boundary_distance(is_boundary, nbr, arc_dist, dist, source, max_km):
    """Great-circle distance to the nearest boundary cell, by relaxation sweeps.

    Gauss-Seidel relaxation rather than Dijkstra: a heap over 6M cells would need a few
    hundred MB of scratch, and uplift only cares out to a few hundred km. Sweeps converge in
    roughly (max_km / cell spacing) rounds and stop early once nothing changes.
    """
    n = is_boundary.shape[0]
    k = nbr.shape[1]
    INF = np.float32(1e30)
    for i in range(n):
        if is_boundary[i]:
            dist[i] = 0.0
            source[i] = i
        else:
            dist[i] = INF
            source[i] = -1

    for _ in range(400):
        changed = False
        for i in range(n):
            di = dist[i]
            if di >= max_km:
                continue
            for s in range(k):
                j = nbr[i, s]
                if j < 0:
                    continue
                nd = di + arc_dist[i, s]
                if nd < dist[j] and nd <= max_km:
                    dist[j] = nd
                    source[j] = source[i]
                    changed = True
        for i in range(n - 1, -1, -1):  # reverse sweep: roughly halves the round count
            di = dist[i]
            if di >= max_km:
                continue
            for s in range(k):
                j = nbr[i, s]
                if j < 0:
                    continue
                nd = di + arc_dist[i, s]
                if nd < dist[j] and nd <= max_km:
                    dist[j] = nd
                    source[j] = source[i]
                    changed = True
        if not changed:
            break
    return dist, source


def build_plates(grid, params: PlateParams = None, seed=0) -> Plates:
    p = params or PlateParams()
    sites = seed_plates(grid, p, seed)
    plate_id = partition(grid, sites, p, seed)
    axis, rate = assign_motion(sites.shape[0], p, seed)
    vel = np.ascontiguousarray(plate_velocity(grid.xyz, axis, rate, plate_id))

    n = grid.cell_count
    b_type = np.zeros(n, dtype=np.uint8)
    b_stress = np.zeros(n, dtype=np.float32)
    _classify_boundaries(np.ascontiguousarray(grid.xyz), np.ascontiguousarray(grid.neighbours),
                         np.ascontiguousarray(plate_id), vel, p.transform_ratio,
                         b_type, b_stress)

    dist = np.empty(n, dtype=np.float32)
    source = np.empty(n, dtype=np.int32)
    boundary_distance(b_type != NONE, np.ascontiguousarray(grid.neighbours),
                      np.ascontiguousarray(grid.arc_dist), dist, source,
                      np.float32(1200.0))
    dist[dist > 1e29] = 1e4  # unreachable within the cap: treat as deep interior

    return Plates(plate_id=plate_id, sites=sites, euler_axis=axis, euler_rate=rate,
                  boundary_type=b_type, boundary_stress=b_stress,
                  boundary_dist_km=dist, nearest_boundary=source)


# --------------------------------------------------------------------------------------
# 4. Uplift
# --------------------------------------------------------------------------------------


def derive_uplift(plates: Plates, is_continental, land_mask, params: PlateParams = None,
                  contrast_cap=None):
    """Uplift from boundary type, stress and distance. A separate operator on purpose.

    Splitting it from crust generation means uplift models can be swapped without
    regenerating plates -- and uplift is the thing worth iterating on most, because Stage 2
    equilibrates against whatever it is given.

    Cases follow the boundary table in terrain-algorithms.md: continental collision gives a
    broad orogen peaked on the boundary; an oceanic-continental trench puts the arc ~150 km
    inland on the continental side; continental rifting gives uplifted shoulders flanking a
    subsiding graben; transform boundaries do little vertically.

    `contrast_cap` raises the cratonic floor to bound the craton-to-orogen ratio. Earth's
    real contrast is 50x or more, but Stage 2 measurements showed uplift contrast dominates
    the slope-area regression above ~4x, so bounding it is what keeps the concavity metric
    comparable to published values. Leave it None for realistic tectonics.
    """
    p = params or PlateParams()
    d = plates.boundary_dist_km
    src = plates.nearest_boundary
    valid = src >= 0

    near_type = np.zeros(d.shape[0], dtype=np.uint8)
    near_stress = np.zeros(d.shape[0], dtype=np.float32)
    near_type[valid] = plates.boundary_type[src[valid]]
    near_stress[valid] = plates.boundary_stress[src[valid]]

    # Normalise stress to 0..1 so magnitudes stay in the 500-5000 m/Myr band.
    scale = np.percentile(np.abs(plates.boundary_stress[plates.boundary_type != NONE]), 90)
    strength = np.clip(np.abs(near_stress) / max(scale, 1e-6), 0.0, 1.5)

    u = np.zeros(d.shape[0], dtype=np.float64)

    conv = near_type == CONVERGENT
    # Continental collision: broad orogen, peaked on the boundary, decaying ~200 km.
    collision = conv & is_continental
    u[collision] += np.exp(-d[collision] / p.orogen_decay_km) * strength[collision]
    # Volcanic arc: offset inland from the trench, on the continental side only.
    arc = conv & is_continental
    u[arc] += 0.6 * np.exp(-((d[arc] - p.arc_offset_km) / p.arc_width_km) ** 2) * strength[arc]

    # Continental rifting: shoulders up, graben down.
    rift = (near_type == DIVERGENT) & is_continental
    shoulder = np.exp(-((d[rift] - p.rift_shoulder_km) / p.rift_shoulder_km) ** 2)
    graben = np.exp(-(d[rift] / (0.5 * p.rift_shoulder_km)) ** 2)
    u[rift] += (0.45 * shoulder - 0.35 * graben) * strength[rift]

    # Transform: little vertical motion, a little at restraining bends.
    trans = near_type == TRANSFORM
    u[trans] += 0.12 * np.exp(-d[trans] / 100.0) * strength[trans]

    # Clamp the SHAPE before scaling. The collision and arc terms overlap, so letting them
    # sum unbounded pushed peak uplift to 4943 m/Myr against a 3000 cap.
    #
    # The lower bound is -1, not 0. Rift grabens SUBSIDE -- that is what the boundary table
    # specifies and what the rift term above computes -- and clipping at zero silently threw
    # it away. The consequence showed up two stages later: with nothing subsiding, the world
    # had essentially no closed basins, so Stage 6 found four tiny coastal lakes and not a
    # single endorheic one. Tectonic subsidence is where the Dead Sea, Baikal and the East
    # African rift lakes come from.
    u = np.clip(u, -1.0, 1.0) * p.u_max_m_per_myr
    # The cratonic floor applies to ground that is rising, not to ground that is dropping.
    u = np.where(u >= 0.0, np.maximum(u, p.craton_floor_m_per_myr), u)

    if contrast_cap is not None:
        # Bound the ratio across RISING ground only. Applying the floor to everything would
        # lift subsiding grabens back out of the ground.
        rising = u > 0.0
        if rising.any():
            u[rising] = np.maximum(u[rising], u[rising].max() / contrast_cap)

    u[~land_mask] = 0.0  # ocean is base level
    return u.astype(np.float32)


def classify_margins(grid, land_mask, plates: Plates, active_within_km=400.0):
    """Coastal cells tagged passive or active by the nearest boundary type.

    Worth more for downstream realism than refining the coastline shape itself: it drives
    shelf width, whether rivers end in deltas or plunge straight in, and where fjords versus
    barrier islands belong. Passive margins get a wide shelf and low coastal relief; active
    margins get mountains within tens of km of the water.
    """
    nbr = grid.neighbours
    coastal = np.zeros(land_mask.shape[0], dtype=bool)
    for s in range(nbr.shape[1]):
        j = nbr[:, s]
        ok = j >= 0
        coastal[ok] |= land_mask[ok] & ~land_mask[j[ok]]

    out = np.full(land_mask.shape[0], INLAND, dtype=np.uint8)
    src = plates.nearest_boundary
    near_type = np.zeros(land_mask.shape[0], dtype=np.uint8)
    near_type[src >= 0] = plates.boundary_type[src[src >= 0]]
    near_active = (near_type == CONVERGENT) | (near_type == TRANSFORM)
    close = plates.boundary_dist_km <= active_within_km

    out[coastal] = PASSIVE
    out[coastal & near_active & close] = ACTIVE
    return out
