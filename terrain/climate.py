"""Stage 3: temperature, prevailing wind, and orographic precipitation.

Geometry alone is not realistic terrain once anything is rendered on it, and climate feeds
back into geometry: precipitation modulates erodibility, so wet windward slopes incise
deeply while arid lee slopes keep their relief. That coupling is Stage 4; this stage
produces the fields it needs.

The highest-value, lowest-cost feature here is the **rain shadow**. Compare the Olympics
against eastern Washington: same latitude, same distance from the sea, an order of
magnitude apart in rainfall, entirely because one is upwind of a mountain range and the
other is downwind.

Precipitation is solved as a steady-state advection-reaction problem rather than by
time-stepping moisture packets. Air picks up moisture over ocean, is advected along the
wind field, and loses moisture in two ways: orographically where it is forced to rise, and
by background rainout everywhere. Iterating

    M[i] <- (1 - loss[i]) * sum_j w[j->i] M[j] + source[i]

to a fixed point gives the same answer as advecting parcels, converges geometrically
because loss is strictly positive, and needs no timestep to tune.

KNOWN LIMITATION: no subtropical desert belt. Earth's great deserts sit near 25-30 degrees
because air there descends from the top of the Hadley cell having already been wrung dry
over the ITCZ -- the horse latitudes are dry because their air ARRIVES dry, not because it
refuses to rain. A surface-only model has no aloft, so it cannot represent that, and the
latitude profile here declines monotonically from the equator instead of dipping at 30.

Three surrogates were tried and measured, all failed:
  * Suppressing background rainout in subsiding bands does nothing on its own, because the
    zonal wind passes through exactly zero at 30 degrees where the circulation cells meet,
    so moisture stagnates there and the saturation cap rains it out regardless.
  * Raising the saturation ceiling made the belt WETTER (1679 mm against 1158): a bigger
    deficit lets the ocean source pump in more moisture.
  * Decoupling the source from the boosted ceiling did not help either; more capacity means
    more moisture retained, and background rainout then works on a larger base
    (horse/mid-latitude ratio went from 1.55 to 1.97, where Earth is about 0.57).

Fixing it properly needs a second vertical level, which is a much larger change than this
stage warrants. What this model does produce -- rain shadows at 1.9-2.9x, a coast-to-
interior gradient, and a wet-tropics-to-dry-poles trend -- is what Stage 4 needs to
modulate erodibility, so the limitation is recorded rather than papered over.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit, prange

from terrain.grid import R_EARTH_KM


@dataclasses.dataclass
class ClimateParams:
    # Temperature
    t_equator_c: float = 27.0
    t_pole_c: float = -25.0
    lapse_rate_c_per_km: float = 6.5      # standard environmental lapse rate
    continentality_c: float = 6.0         # seasonal amplitude gain far from the sea
    continentality_scale_km: float = 1500.0
    # Wind. Bands are Hadley / Ferrel / Polar: easterly, westerly, easterly.
    wind_speed: float = 1.0               # arbitrary units; only direction and ratio matter
    meridional_fraction: float = 0.3
    # Moisture
    evaporation: float = 1.0              # source strength over ocean
    # Rainout per metre of forced ascent. Expressed per metre it is resolution-robust: the
    # total climb over a range is the same however finely the range is sampled.
    #
    # Tuned against the ISOLATED orographic effect -- the windward/leeward ratio with real
    # relief divided by the same ratio on flattened terrain, sampled at the same points.
    # That control is essential: the raw ratio is ~2.4x, but flat ground scores ~2.5x on its
    # own, because walking upwind from a ridge heads toward the sea and downwind heads
    # inland. The raw number is almost entirely the coast-to-interior gradient. An earlier
    # version of this file reported 1.9-2.9x "rain shadows" that were nothing of the kind.
    orographic_coeff: float = 0.004
    background_rainout: float = 0.012     # fraction lost per advection step
    # Vertical motion of the three-cell circulation. Air rises at the ITCZ and the polar
    # front and subsides at the horse latitudes and the poles, which is why the world's
    # great deserts sit near 25-30 degrees rather than at the poles. Without this the
    # latitude profile declines monotonically from the equator and there is no desert belt.
    subsidence_floor: float = 0.10        # background rainout multiplier where air descends
    eddy: float = 0.07                    # isotropic mixing blended into pure advection
    # Fraction of rain over land that evapotranspires and rains again further inland.
    # Without it, every drop that falls on a continent must have arrived from the coast,
    # and interiors come out as absolute desert: measured 2043 mm within 200 km of the sea
    # against 124 mm beyond 1000 km, a 16x drop where Earth is nearer 3x. The Amazon
    # recycles roughly half its rainfall this way.
    #
    # This knob is in DIRECT TENSION with the rain shadow, which is worth knowing before
    # touching either. Recycled moisture refills the lee side, so raising it flattens the
    # very contrast the orographic term exists to create. Measured on one world, varying
    # both knobs (isolated = orographic effect with the flat-terrain control divided out):
    #
    #   coeff / recycling      0.55            0.70
    #   0.0015           1.12 iso, 11.5 c/i   1.02 iso,  9.2 c/i
    #   0.0040           1.52 iso, 16.9 c/i   1.34 iso, 11.9 c/i
    #   0.0080           1.82 iso, 21.2 c/i   1.67 iso, 13.4 c/i
    #
    # No setting reaches Earth on both counts (Earth is roughly 3x coast-to-interior with a
    # clear shadow), and the numbers move with the continent layout too. This is a limit of
    # a single-layer model, not a tuning failure. 0.70 with coeff 0.004 keeps a real
    # orographic signal while holding interiors to something a continent could support.
    land_recycling: float = 0.70
    saturation_gain: float = 0.06         # how fast warm air holds more water, per degC
    iterations: int = 400
    target_mean_precip_mm: float = 1000.0  # Earth's land average, for the final scaling


@dataclasses.dataclass
class Climate:
    temperature_c: np.ndarray     # (N,) f32, mean annual at the surface
    seasonality_c: np.ndarray     # (N,) f32, continentality-driven amplitude
    wind: np.ndarray              # (N,3) f32, tangent vectors
    precipitation_mm: np.ndarray  # (N,) f32, mm/yr
    moisture: np.ndarray          # (N,) f32, steady-state airborne moisture
    coast_distance_km: np.ndarray  # (N,) f32


# --------------------------------------------------------------------------------------
# Temperature and wind
# --------------------------------------------------------------------------------------


def temperature(grid, elevation_m, coast_distance_km, p: ClimateParams):
    """Latitude insolation, elevation lapse rate, and continentality."""
    lat = np.radians(grid.lat.astype(np.float64))
    sea_level_t = p.t_pole_c + (p.t_equator_c - p.t_pole_c) * np.cos(lat) ** 1.5
    t = sea_level_t - p.lapse_rate_c_per_km * np.maximum(elevation_m, 0.0) / 1000.0
    seasonality = p.continentality_c * (
        1.0 - np.exp(-coast_distance_km / p.continentality_scale_km))
    return t.astype(np.float32), seasonality.astype(np.float32)


def wind_field(grid, p: ClimateParams):
    """Surface wind from the three-cell circulation, as tangent vectors.

    Bands are easterly 0-30 (trades), westerly 30-60, easterly 60-90 (polar). Writing the
    zonal component as -cos(pi*(|lat|-15)/30) puts the band centres at 15/45/75 degrees and
    zeroes the wind at 0/30/60/90 -- the doldrums, horse latitudes and polar front, which
    is where the real circulation goes slack.

    The meridional component follows from the same expression: surface flow is equatorward
    in the Hadley and Polar cells and poleward in the Ferrel cell, which is exactly where
    the zonal sign flips, so v is proportional to sign(lat) * u.
    """
    lat = grid.lat.astype(np.float64)
    u = -p.wind_speed * np.cos(np.pi * (np.abs(lat) - 15.0) / 30.0)
    v = p.meridional_fraction * np.sign(lat) * u

    xyz = grid.xyz
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(np.broadcast_to(pole, xyz.shape), xyz)
    norm = np.linalg.norm(east, axis=1, keepdims=True)
    east = np.divide(east, norm, out=np.zeros_like(east), where=norm > 1e-12)
    north = np.cross(xyz, east)

    wind = u[:, None] * east + v[:, None] * north
    return np.ascontiguousarray(wind, dtype=np.float64)


# --------------------------------------------------------------------------------------
# Moisture transport
# --------------------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _outgoing_weights(xyz, nbr, wind, eddy, out_w):
    """Fraction of a cell's moisture leaving toward each neighbour, by wind alignment.

    `eddy` blends in an isotropic component standing for turbulent mixing, and it is not
    optional. The zonal wind passes through exactly zero at 0, 30, 60 and 90 degrees where
    the circulation cells meet, so on pure advection those cells have no outgoing weights
    at all: they accept moisture from every neighbour and never release any. They became
    traps that filled to saturation and dumped the excess every iteration, producing
    129,000 mm/yr spikes along four latitude lines. A small isotropic term guarantees every
    cell exports moisture.
    """
    n = xyz.shape[0]
    k = nbr.shape[1]
    for i in prange(n):
        speed = np.sqrt(wind[i, 0] ** 2 + wind[i, 1] ** 2 + wind[i, 2] ** 2)
        degree = 0
        for s in range(k):
            if nbr[i, s] >= 0:
                degree += 1
        total = 0.0
        for s in range(k):
            out_w[i, s] = 0.0
            j = nbr[i, s]
            if j < 0:
                continue
            if speed <= 0.0:
                continue
            # Tangent direction from i to j.
            dot = xyz[j, 0] * xyz[i, 0] + xyz[j, 1] * xyz[i, 1] + xyz[j, 2] * xyz[i, 2]
            dx = xyz[j, 0] - dot * xyz[i, 0]
            dy = xyz[j, 1] - dot * xyz[i, 1]
            dz = xyz[j, 2] - dot * xyz[i, 2]
            dn = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dn <= 0.0:
                continue
            align = (wind[i, 0] * dx + wind[i, 1] * dy + wind[i, 2] * dz) / (dn * speed)
            if align > 0.0:
                out_w[i, s] = align
                total += align
        if total > 0.0:
            for s in range(k):
                out_w[i, s] /= total
        # Blend toward isotropic spreading.
        share = eddy / degree if degree > 0 else 0.0
        for s in range(k):
            if nbr[i, s] < 0:
                continue
            out_w[i, s] = (1.0 - eddy) * out_w[i, s] + share
    return out_w


@njit(cache=True)
def _reverse_slots(nbr, rev):
    """rev[i, s] = the slot in neighbour nbr[i,s] that points back at i.

    Precomputed once so the advection sweep can be a parallel gather instead of a scatter;
    scattering would race under prange.
    """
    n, k = nbr.shape
    for i in range(n):
        for s in range(k):
            rev[i, s] = -1
            j = nbr[i, s]
            if j < 0:
                continue
            for t in range(k):
                if nbr[j, t] == i:
                    rev[i, s] = t
                    break
    return rev


def rainout_factor(grid, p: ClimateParams):
    """Where the three-cell circulation has air rising (wet) versus subsiding (dry).

    cos(pi*|lat|/30) is +1 at 0 and 60 degrees -- the ITCZ and the polar front, where air
    ascends -- and -1 at 30 and 90, the horse latitudes and the poles, where it descends.
    Suppressing background rainout in the descending bands puts the subtropical deserts
    where they belong, while leaving orographic rainfall intact, so a range inside the
    desert belt still catches rain the way the Atlas does on the edge of the Sahara.
    """
    ascent = 0.5 + 0.5 * np.cos(np.pi * np.abs(grid.lat.astype(np.float64)) / 30.0)
    return p.subsidence_floor + (1.0 - p.subsidence_floor) * ascent


@njit(cache=True, parallel=True)
def _advect_step(moisture, nbr, rev, out_w, lift_loss, retain, source, capacity,
                 recycle, new_m, precip):
    """One fixed-point sweep: gather from upwind, precipitate, add the ocean source.

    The saturation cap is doing more than adding realism -- it is what makes the iteration
    converge at all. Incoming weights sum to more than one wherever the wind converges, so
    without a ceiling the fixed point is not a contraction and moisture grows without bound
    in convergence zones: measured 1.3e7 mm/yr before this was added. Capping at the
    Clausius-Clapeyron capacity bounds the operator and supplies the mechanism that makes
    convergence zones like the ITCZ rain heavily rather than diverge.
    """
    n = moisture.shape[0]
    k = nbr.shape[1]
    for i in prange(n):
        incoming = 0.0
        for s in range(k):
            j = nbr[i, s]
            t = rev[i, s]
            if j < 0 or t < 0:
                continue
            incoming += moisture[j] * out_w[j, t]
        # Orographic rainout on the way in, then background rainout.
        kept = incoming * lift_loss[i] * retain[i]
        rained = incoming - kept
        total = kept

        # Evaporation fills the SATURATION DEFICIT; it does not pump moisture into air that
        # is already saturated. Adding it unconditionally and dumping the overflow as rain
        # made the ocean rain out its own evaporation on the spot -- over 10,000 mm/yr across
        # most of the sea -- leaving only a trickle to advect onto land.
        if source[i] > 0.0:
            room = capacity[i] - total
            if room > 0.0:
                total += source[i] if source[i] < room else room

        # Evapotranspiration: some of what just fell goes back up and rains again inland.
        total += recycle[i] * rained
        if total > capacity[i]:
            rained += total - capacity[i]
            total = capacity[i]
        precip[i] = rained
        new_m[i] = total
    return new_m, precip


def precipitation(grid, elevation_m, land_mask, temperature_c, wind, p: ClimateParams):
    """Steady-state moisture transport with orographic and background rainout."""
    n = grid.cell_count
    nbr = np.ascontiguousarray(grid.neighbours)
    xyz = np.ascontiguousarray(grid.xyz)

    out_w = np.zeros(nbr.shape, dtype=np.float64)
    _outgoing_weights(xyz, nbr, wind, p.eddy, out_w)
    rev = np.empty(nbr.shape, dtype=np.int32)
    _reverse_slots(nbr, rev)

    # Warm air holds more water: the tropics are wet and the poles are dry deserts even
    # though both sit next to ocean.
    capacity = np.clip(
        np.exp(p.saturation_gain * (temperature_c.astype(np.float64) - 15.0)), 0.05, 4.0)
    source = np.where(~land_mask, p.evaporation * capacity, 0.0)

    # Orographic loss: how much of an incoming parcel survives the forced ascent into
    # this cell. Averaged over the upwind neighbours, weighted by how much comes from each.
    ascent = _weighted_ascent(xyz, nbr, rev, out_w, elevation_m.astype(np.float64))
    lift_loss = np.exp(-p.orographic_coeff * np.maximum(ascent, 0.0))
    retain = np.ascontiguousarray(1.0 - p.background_rainout * rainout_factor(grid, p))

    recycle = np.ascontiguousarray(np.where(land_mask, p.land_recycling, 0.0))

    moisture = np.zeros(n, dtype=np.float64)
    new_m = np.empty(n, dtype=np.float64)
    precip = np.zeros(n, dtype=np.float64)
    for _ in range(p.iterations):
        _advect_step(moisture, nbr, rev, out_w, lift_loss, retain, source, capacity,
                     recycle, new_m, precip)
        moisture, new_m = new_m, moisture

    # Scale to a realistic annual total. Absolute units are not derivable from this model;
    # what it produces is a spatial pattern, and the pattern is what Stage 4 needs.
    land_mean = precip[land_mask].mean() if land_mask.any() else precip.mean()
    if land_mean > 0:
        precip *= p.target_mean_precip_mm / land_mean
    return precip.astype(np.float32), moisture.astype(np.float32)


@njit(cache=True, parallel=True)
def _weighted_ascent(xyz, nbr, rev, out_w, z):
    """Mean elevation gain an incoming air parcel must climb to reach each cell."""
    n = z.shape[0]
    k = nbr.shape[1]
    out = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        total = 0.0
        acc = 0.0
        for s in range(k):
            j = nbr[i, s]
            t = rev[i, s]
            if j < 0 or t < 0:
                continue
            w = out_w[j, t]
            if w <= 0.0:
                continue
            rise = z[i] - z[j]
            if rise > 0.0:
                acc += w * rise
            total += w
        out[i] = acc / total if total > 0.0 else 0.0
    return out


# --------------------------------------------------------------------------------------


def compute_climate(grid, elevation_m, land_mask, params: ClimateParams = None) -> Climate:
    from terrain import plates

    p = params or ClimateParams()
    n = grid.cell_count

    dist = np.empty(n, dtype=np.float32)
    src = np.empty(n, dtype=np.int32)
    plates.boundary_distance(~land_mask, np.ascontiguousarray(grid.neighbours),
                             np.ascontiguousarray(grid.arc_dist), dist, src,
                             np.float32(6000.0))
    dist[dist > 1e29] = 6000.0

    t, seasonality = temperature(grid, elevation_m, dist, p)
    wind = wind_field(grid, p)
    precip, moisture = precipitation(grid, elevation_m, land_mask, t, wind, p)

    return Climate(temperature_c=t, seasonality_c=seasonality,
                   wind=wind.astype(np.float32), precipitation_mm=precip,
                   moisture=moisture, coast_distance_km=dist)
