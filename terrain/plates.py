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

# How far inland boundary stress is felt, as a FRACTION OF THE SPHERE'S RADIUS. Was a
# hardcoded 1200 km, which is 0.188 R on Earth and larger than the whole world at
# R = 64.9 km. See `build_plates`.
UPLIFT_REACH_R = 0.188

# How close to a convergent or transform boundary a coast has to be to count as an active
# margin, again as a fraction of the radius: 400 km on Earth. Same reason as above -- held in
# km it was 400 on a world 64.9 km in radius, more than thirty times the uplift reach, so the
# test could never fail and every coast near a boundary of the right TYPE read as active
# regardless of distance. See `classify_margins`.
ACTIVE_MARGIN_R = 0.0628


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
    # Passes of along-strike averaging applied to boundary stress before it is propagated
    # inland. Each boundary cell hands its stress to a wedge of interior cells, so raw
    # cell-to-cell scatter becomes hard stripes fanning out from the belt. 8 passes spreads
    # the stress over roughly 8 cells of strike, which removes the stripes while leaving
    # the along-belt variation that makes one end of a range higher than the other.
    stress_smoothing_passes: int = 8
    # Uplift geometry, from the boundary-case table in terrain-algorithms.md. These are
    # HORIZONTAL distances on the sphere, so they scale with the radius: the Earth-radius
    # values were 200 / 150 / 100 / 80 km, divided by 98.125 for R = 64.927 km.
    orogen_decay_km: float = 2.04    # continental collision, peaked on the boundary
    arc_offset_km: float = 1.53      # volcanic arc, inland of an oceanic-continental trench
    arc_width_km: float = 1.02
    rift_shoulder_km: float = 0.82   # uplifted flanks either side of a continental graben
    # Transform boundaries, the restraining-bend term. This one was a bare 100.0 inside the
    # expression in `derive_uplift` rather than a field here, so `for_radius` never saw it and
    # it stayed at its Earth value through the rescale. Over the 12.2 km uplift reach on this
    # sphere, exp(-d/100) runs 1.000 to 0.885 -- i.e. flat, painting every transform domain
    # with a uniform slab instead of a ridge that hugs the boundary. Same failure as the
    # hardcoded 1200 km reach above, and it survived for the same reason the pruning bug did:
    # a literal in an expression carries no comment and answers to no rescale.
    #
    # It is a real terrain change, not a cosmetic one: transform domains are 17.6% of land,
    # and they go from a broad slab averaging 3.5 m/Myr to the cratonic floor except within
    # ~2 km of the boundary. Measured at n=256, 2 passes x 200 steps, seed 1:
    #
    #                          before    after
    #     U on transform land   3.51     2.13   m/Myr
    #     land at craton floor  70.9%    76.3%
    #     theta (40 cells)      0.500    0.438   (r2 0.934 -> 0.919)
    #     Hack h                0.489    0.515   (SRTM east 0.543)
    #     peak                  692.4    691.5  m
    #
    # Both thetas sit inside the published 0.4-0.6 band and everything else is unmoved. The
    # n=128 fixture reports this as theta 0.474 -> 0.026, which is NOT a real degradation:
    # at that size the 40-cell threshold leaves 1,612 channel cells and no basin large enough
    # to fit, so r2 runs 0.55 -> 0.00 and both numbers are noise. Do not tune against n=128.
    transform_decay_km: float = 1.02
    # Crustal thickness response. VERTICAL, so it does not follow the radius -- it follows
    # the relief budget instead, cut by 3x with the rest of the isostasy. See CrustParams.
    convergent_thickening_km: float = 5.0
    divergent_thinning_km: float = 2.7
    # Magnitudes. VERTICAL rates, so again not a radius rescale -- but they had to come
    # down 12x anyway, and the reason is the whole point of the 100 m grid.
    #
    # Hillslope diffusion sets a steady-state hilltop curvature of exactly U/D. At the old
    # U=1200 m/Myr with D=0.05 m^2/yr that is 0.024 1/m, which over a 100 m cell means
    # 120 m of convexity PER CELL -- the hilltops would be spikes. That inconsistency was
    # always in the parameters; it was invisible only because hillslopes were sub-grid and
    # D acted as a smoother. Making hillslopes resolved makes U and D answerable to each
    # other for the first time.
    #
    # At U=100 m/Myr, C = U/D = 5e-3 1/m gives 100 m of convex relief over a 200 m hillslope
    # and a basal slope of 1.0 -- 45 degrees, right at the angle of repose. The craton floor
    # gives C = 1e-4, so 2 m over the same length: gentle rolling ground. Raising D instead
    # would need 0.6 m^2/yr, well outside the published 0.001-0.1 range for soil creep, so
    # the uplift is what gives.
    #
    # THIS BUDGET IS SPENT, and that is the operative fact about `u_max_m_per_myr`: it is a
    # ceiling, not a setting with room above it. Any proposal to raise U for more relief
    # multiplies every hillslope gradient by the same factor and is answered here rather than
    # by trying it.
    #
    # The ANALYTIC argument above is what carries that, and it is worth separating from the
    # measurement that used to be quoted beside it. This comment read "measured slope
    # distribution on the shipped n=256 world -- p99 62.5 deg, p99.9 76.1, max 83.7 -- so the
    # steep ground is already past repose". Those percentiles are real and that reading of
    # them is wrong: decomposed while building `stage2.mass_wasting`, 98.1% of the edges past
    # repose have an OCEAN cell at one end and 1.8% have a lake cell, leaving 0.05% that are
    # genuinely hillslope. It is a statistic about the shoreline -- a land cell 100 m from
    # water held at base level -- not about hillslopes, which run p50 0.9 deg and p90 9.4.
    #
    # So the budget is spent by C = U/D and the basal gradient it implies, which is
    # arithmetic, and NOT by a measured slope distribution sitting at failure. The conclusion
    # is unchanged; one of its two supports was never load-bearing.
    #
    # These figures read 2e-3 / 40 m / 0.4 and a craton C of 4e-5 until they were recomputed:
    # they were derived against D=0.05 and never updated when stage2.py took D to 0.02 as a
    # deliberate trade. The correction goes the unhelpful way -- the budget is 2.5x more
    # spent than the old comment claimed, not less.
    u_max_m_per_myr: float = 100.0
    craton_floor_m_per_myr: float = 2.0

    def for_radius(self, radius_km):
        """Copy with the WORLD-SCALE lengths rescaled from the production sphere.

        The five uplift-geometry distances are fractions of the world wearing kilometres:
        2.04 km of orogen decay means 3.1% of the radius, which is what 200 km meant on
        Earth. Authoring them in km keeps them readable against the boundary-case table
        in terrain-algorithms.md; rescaling here keeps them meaningful on a fixture at a
        different n, where cell size is still 100 m but the sphere is smaller.

        Without this, a test at n=72 gets a 5.96 km world carrying a 2.04 km decay
        length, so every cell is effectively on a boundary and the uplift field is flat.
        Note this is the OPPOSITE of the cell-scale parameters -- lake and island minima,
        the channel-head discharge -- which are absolute and must NOT be rescaled,
        because cell size does not vary with n.
        """
        from terrain.grid import R_WORLD_KM

        f = float(radius_km) / R_WORLD_KM
        return dataclasses.replace(
            self,
            orogen_decay_km=self.orogen_decay_km * f,
            arc_offset_km=self.arc_offset_km * f,
            arc_width_km=self.arc_width_km * f,
            rift_shoulder_km=self.rift_shoulder_km * f,
            transform_decay_km=self.transform_decay_km * f,
        )


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
    radius_km: float = 0.0       # the sphere these distances were measured on


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


def assign_motion(n_plates, params: PlateParams, seed=0, radius_km=None):
    """A random Euler pole and angular rate per plate.

    Rate is chosen so the fastest surface speed lands near max_speed_mm_per_yr. Earth's
    plates run roughly 10-100 mm/yr, and that is a SURFACE speed, so it is the right
    thing to hold fixed on a smaller sphere -- rock does not know the planet's radius.

    The radius cancels between here and `plate_velocity` (rate is v/R, velocity is
    rate*R), so threading it changes no result. It is threaded anyway because the pair
    of hardcoded 6371s only agreed by luck, and `euler_rate` is exported as an angular
    rate that a reader is entitled to interpret on the actual sphere.
    """
    from terrain.grid import R_EARTH_KM

    rng = np.random.default_rng(seed + 991)
    axis = rng.standard_normal((n_plates, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    # v = omega * R * sin(theta); take sin(theta) ~ 1 for the scaling.
    max_rate = params.max_speed_mm_per_yr / float(radius_km or R_EARTH_KM)  # rad/Myr
    rate = rng.uniform(0.15, 1.0, n_plates) * max_rate
    return axis, rate


def plate_velocity(xyz, axis, rate, plate_id, radius_km=None):
    """Surface velocity in km/Myr (== mm/yr), as the Euler-pole cross product."""
    from terrain.grid import R_EARTH_KM

    return np.cross(axis[plate_id] * rate[plate_id, None], xyz) * float(
        radius_km or R_EARTH_KM)


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


def boundary_distance(is_boundary, xyz, radius_km, dist, source, max_km):
    """True great-circle distance to the nearest boundary cell, and which cell that is.

    This used to accumulate `arc_dist` along the cell graph by Gauss-Seidel relaxation,
    which measures the wrong thing. Graph distance on a hex lattice is ANISOTROPIC: a path
    at 30 degrees to a lattice row has to zigzag, costing 2/sqrt(3) = 1.155x the direct
    route, so the level sets come out hexagonal rather than circular. Measured against the
    true distance at n=256 it ran 8% long on median and up to 21.8% in the worst direction.

    Uplift is an exponential in this distance, so that error became lattice-aligned ridges
    fanning out from every mountain belt -- the artifact was baked into the uplift field
    before a single erosion step ran.

    A KD-tree over just the boundary cells fixes it exactly and is also far cheaper than
    400 relaxation sweeps: there are ~15k boundary cells at n=256, not 6M.
    """
    from scipy.spatial import cKDTree

    src_idx = np.flatnonzero(np.asarray(is_boundary))
    if src_idx.size == 0:
        dist[:] = np.float32(1e30)
        source[:] = -1
        return dist, source

    xyz = np.asarray(xyz, dtype=np.float64)
    chord, hit = cKDTree(xyz[src_idx]).query(xyz, k=1, workers=-1)
    # Chord length on the unit sphere -> arc length.
    d = 2.0 * float(radius_km) * np.arcsin(np.clip(chord * 0.5, 0.0, 1.0))
    beyond = d > float(max_km)
    dist[:] = np.where(beyond, 1e30, d).astype(np.float32)
    source[:] = np.where(beyond, -1, src_idx[hit]).astype(np.int32)
    return dist, source


@njit(cache=True)
def smooth_along_boundary(is_boundary, stress, nbr, passes, out):
    """Average boundary stress with its neighbours ALONG the belt.

    `derive_uplift` gives every interior cell the stress of the single nearest boundary
    cell, so each boundary cell owns a wedge of territory -- a median of 7 cells at n=256.
    Adjacent boundary cells carry unrelated stress values, so those wedges render as hard
    stripes fanning out from the belt, which is the second half of the same artifact.

    Smoothing along strike is also the physically right move. Stress on a real plate
    boundary varies smoothly along its length; the cell-to-cell scatter here is
    discretisation noise from classifying each cell's velocity jump independently.
    """
    n = stress.shape[0]
    k = nbr.shape[1]
    cur = stress.copy()
    for _ in range(passes):
        for i in range(n):
            if not is_boundary[i]:
                out[i] = cur[i]
                continue
            acc = cur[i]
            cnt = 1.0
            for s in range(k):
                j = nbr[i, s]
                if j < 0 or not is_boundary[j]:
                    continue
                acc += cur[j]
                cnt += 1.0
            out[i] = acc / cnt
        for i in range(n):
            cur[i] = out[i]
    return out


def build_plates(grid, params: PlateParams = None, seed=0) -> Plates:
    p = params or PlateParams()
    sites = seed_plates(grid, p, seed)
    plate_id = partition(grid, sites, p, seed)
    axis, rate = assign_motion(sites.shape[0], p, seed, grid.radius_km)
    vel = np.ascontiguousarray(
        plate_velocity(grid.xyz, axis, rate, plate_id, grid.radius_km))

    n = grid.cell_count
    b_type = np.zeros(n, dtype=np.uint8)
    b_stress = np.zeros(n, dtype=np.float32)
    _classify_boundaries(np.ascontiguousarray(grid.xyz), np.ascontiguousarray(grid.neighbours),
                         np.ascontiguousarray(plate_id), vel, p.transform_ratio,
                         b_type, b_stress)

    is_b = b_type != NONE
    b_stress = smooth_along_boundary(np.ascontiguousarray(is_b), b_stress,
                                     np.ascontiguousarray(grid.neighbours),
                                     p.stress_smoothing_passes, np.empty_like(b_stress))

    dist = np.empty(n, dtype=np.float32)
    source = np.empty(n, dtype=np.int32)
    # Reach as a FRACTION OF THE RADIUS, not an absolute distance. 1200 km on Earth was
    # 0.188 R; hardcoding the km made it larger than the whole world once the sphere
    # shrank, which would have marked every cell boundary-adjacent and flattened the
    # uplift field into a constant.
    reach = np.float32(UPLIFT_REACH_R * grid.radius_km)
    boundary_distance(is_b, grid.xyz, grid.radius_km, dist, source, reach)
    dist[dist > 1e29] = np.float32(10.0 * reach)  # beyond the cap: deep interior

    return Plates(plate_id=plate_id, sites=sites, euler_axis=axis, euler_rate=rate,
                  boundary_type=b_type, boundary_stress=b_stress,
                  boundary_dist_km=dist, nearest_boundary=source,
                  radius_km=float(grid.radius_km))


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
    if plates.radius_km:  # 0.0 means a Plates built without one; leave params alone
        p = p.for_radius(plates.radius_km)
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
    u[trans] += 0.12 * np.exp(-d[trans] / p.transform_decay_km) * strength[trans]

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


def classify_margins(grid, land_mask, plates: Plates, active_within_km=None):
    """Coastal cells tagged passive or active by the nearest boundary type.

    Exported alongside the fields so anything drawing coasts can use it: an active margin
    puts mountains within tens of km of the water, a passive one gets a wide shelf and low
    coastal relief. Nothing in this pipeline consumes it yet -- shelf width, deltas versus
    plunging river mouths, and fjords versus barrier islands are all downstream of a renderer
    that does not exist. It ships because it is nearly free to compute here and impossible to
    recover later.

    `active_within_km` defaults to a fraction of the sphere, not the old fixed 400 km, which
    on this world was larger than the whole uplift reach and so never excluded anything.
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
    if active_within_km is None:
        active_within_km = ACTIVE_MARGIN_R * (plates.radius_km or grid.radius_km)
    close = plates.boundary_dist_km <= active_within_km

    out[coastal] = PASSIVE
    out[coastal & near_active & close] = ACTIVE
    return out
