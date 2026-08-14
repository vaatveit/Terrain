"""Stage 3: temperature, prevailing wind, ocean heat transport, and orographic rainfall.

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

OCEAN HEAT TRANSPORT is the one non-zonal mechanism here. Everything else in the
temperature field is a function of where a cell IS -- its latitude, its height, its
distance from the sea -- and a field built only from those is a latitude profile with
decorations. The maritime anomaly is a function of where a cell's AIR CAME FROM, which is
the only thing on the list that can distinguish two coasts at the same latitude.

It is worth being exact about what it buys, because the honest answer is "less than the
zonality statistic wants". Temperature zonality moves 0.9926 -> 0.9898 on an n=256 world,
which looks like nothing; the same measurement says the maritime term supplies 0.66 C of
the world's 1.46 C of non-zonal temperature signal, second only to continental cooling.
Both are true. The latitude term has a standard deviation of 13.7 C and everything else put
together has 1.5 C, so no horizontal process can move the aggregate. The measurement that
actually shows the mechanism working is directional: onshore-wind coasts carry 1.9x the
maritime anomaly of offshore-wind coasts, which is Norway against Labrador, and which an
isotropic distance decay scores at 0.64x -- backwards.

The limit is vertical, not horizontal. Land elevation on these worlds has a standard
deviation of 61 m against Earth's several hundred, so the lapse rate contributes 0.28 C
where on Earth it would contribute around 5. Relief, not distance, is what the temperature
field is short of.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit, prange

from terrain.grid import R_EARTH_KM, R_WORLD_KM


@dataclasses.dataclass
class ClimateParams:
    # Temperature
    t_equator_c: float = 27.0
    t_pole_c: float = -25.0
    lapse_rate_c_per_km: float = 6.5      # standard environmental lapse rate
    continentality_c: float = 6.0         # seasonal amplitude gain far from the sea
    # Horizontal, so it scales with the radius: 1500 km / 98.125. The temperature and
    # rainfall constants around it are NOT rescaled -- degrees, mm/yr and the lapse rate
    # per km of ELEVATION are all physical, and the advection constants are per grid step
    # on a cell graph whose topology has not changed.
    continentality_scale_km: float = 15.3
    # Annual-MEAN cooling deep inside a continent, at high latitude. Distinct from the
    # seasonal amplitude above, and the more important of the two here: the sea moderates
    # the annual mean, not just its swing. Siberia at 60N averages about -10 C where the
    # Norwegian coast at the same latitude averages +5 C.
    #
    # Without this, temperature is a pure function of latitude and elevation and comes out
    # 96% zonal, which is what drove the ruler-straight biome bands. Coast distance depends
    # on continent shape, so it is the cheapest genuinely non-zonal signal available.
    continental_cooling_c: float = 15.0
    # Wind. Bands are Hadley / Ferrel / Polar: easterly, westerly, easterly.
    wind_speed: float = 1.0               # arbitrary units; only direction and ratio matter
    meridional_fraction: float = 0.3
    # Ocean currents. Surface Ekman transport runs about 45 degrees to the right of the wind
    # in the north and to the left in the south; that deflection is what closes the trades
    # and the westerlies into a gyre instead of two opposed zonal streams.
    ekman_deg: float = 45.0
    # Distance a current carries its temperature anomaly before relaxing back to the local
    # equilibrium, in world km at the reference radius: 10 km here is 981 km on Earth.
    # Short against a real western boundary current, which reaches several thousand km --
    # this value is what the per-step form worked out to at n=785 and is kept so the change
    # of units alters nothing at production resolution. Worth revisiting on its merits.
    current_scale_km: float = 10.0
    current_iterations: int = 400
    # Gain on the sea surface anomaly where it comes ashore. 0.0 disables ocean heat
    # transport entirely and is the regression switch -- see `sst_anomaly` for why zeroing
    # the current speed instead does NOT give a null result.
    ocean_heat_c: float = 1.0
    # How far inland maritime air keeps its anomaly, world km at the reference radius.
    # 5 km here is 491 km on Earth, which is the right order for how far Atlantic air
    # reaches into Europe.
    maritime_scale_km: float = 5.0
    maritime_iterations: int = 300
    # Keep only the departure from the zonal mean; see `sst_anomaly` for why the mean has
    # to go. Off is available for diagnosing the transport itself, not for production.
    strip_zonal_mean: bool = True
    zonal_band_deg: float = 2.0
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
    # Distance over which airborne moisture decays to 1/e, in the same units as
    # `continentality_scale_km` -- world km at the reference radius, so 8.33 here is 818 km
    # on Earth. It used to be a fraction lost per advection STEP, which is a length in CELLS
    # and therefore a different physical distance on every sphere: the same 0.012 reached
    # 0.13 R at n=785 and 0.39 R at n=256, so anything tuned at one resolution was wrong at
    # the other. 8.33 reproduces the old 0.012 exactly at n=785, where it was tuned.
    background_rainout_km: float = 8.33
    # Vertical motion of the three-cell circulation. Air rises at the ITCZ and the polar
    # front and subsides at the horse latitudes and the poles, which is why the world's
    # great deserts sit near 25-30 degrees rather than at the poles. Without this the
    # latitude profile declines monotonically from the equator and there is no desert belt.
    subsidence_floor: float = 0.10        # background rainout multiplier where air descends
    eddy: float = 0.07                    # isotropic mixing blended into pure advection
    # Low-frequency wobble applied to the band latitudes, standing in for Rossby waves.
    # Without it the circulation is perfectly zonal, so the four latitudes where the zonal
    # wind passes through zero are dead-straight lines all the way round the world -- and
    # they show up in the output as ruler-straight stripes of anomalous rainfall, most
    # obviously along the equator. Real cells meander by tens of degrees of longitude.
    meander_deg: float = 7.0
    meander_frequency: float = 1.8
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
    sst_anomaly_c: np.ndarray = None  # (N,) f32, sea surface departure from its latitude


# --------------------------------------------------------------------------------------
# Temperature and wind
# --------------------------------------------------------------------------------------


def mean_spacing_km(grid):
    """Mean centre-to-centre distance between neighbouring cells."""
    return float(np.mean(grid.arc_dist[grid.neighbours >= 0]))


def per_step_relax(grid, scale_km, spacing_km=None):
    """Convert an e-folding DISTANCE into the per-step rate the advection sweeps take.

    Every transport constant in this module was originally a fraction lost per advection
    step, which is an e-folding length denominated in CELLS. That is not a physical
    distance: the same 0.012 background rainout reached 0.13 of the radius at n=785 and
    0.39 at n=256, so a value tuned at one resolution was simply wrong at another -- the
    same failure mode as the hardcoded lengths the rescale to 100 m cells had to find.

    Expressing the constant as a distance and dividing by the cell spacing fixes the
    physical length and lets the step count follow the grid. The distance itself is in
    world km at the reference radius, exactly as `continentality_scale_km` is, so it also
    tracks the size of the sphere.
    """
    if spacing_km is None:
        spacing_km = mean_spacing_km(grid)
    scale = float(scale_km) * float(grid.radius_km) / R_WORLD_KM
    if scale <= 0.0:
        return 1.0
    # Capped below 1: a step longer than the e-folding distance would otherwise ask for
    # more than complete decay in a single sweep.
    return min(spacing_km / scale, 0.9)


def equilibrium_temperature(grid, p: ClimateParams):
    """Sea-level temperature from latitude alone: the profile everything departs from."""
    lat = np.radians(grid.lat.astype(np.float64))
    return p.t_pole_c + (p.t_equator_c - p.t_pole_c) * np.cos(lat) ** 1.5


def temperature(grid, elevation_m, coast_distance_km, p: ClimateParams,
                maritime_anom=None):
    """Latitude insolation, elevation lapse rate, continentality, and ocean heat transport.

    Continentality affects the annual MEAN, not only the seasonal swing. The latitude
    weighting matters: the sea moderates a cold winter far more than it moderates a warm
    one, so the effect is strong at high latitude and vanishes at the equator -- the
    interior of the Amazon is not colder than its coast.

    The maritime term is the only one here that is not a function of latitude, elevation or
    distance, and it is the reason this field can stop being zonal at all. Everything else
    varies with WHERE a cell is; the anomaly varies with where its air came FROM.
    """
    lat = np.radians(grid.lat.astype(np.float64))
    inland = 1.0 - np.exp(-np.asarray(coast_distance_km, dtype=np.float64)
                          / p.continentality_scale_km)
    t = (equilibrium_temperature(grid, p)
         - p.lapse_rate_c_per_km * np.maximum(elevation_m, 0.0) / 1000.0
         - p.continental_cooling_c * inland * np.abs(np.sin(lat)) ** 1.5)

    if maritime_anom is not None and p.ocean_heat_c != 0.0:
        t += p.ocean_heat_c * np.asarray(maritime_anom, dtype=np.float64)

    seasonality = p.continentality_c * inland
    return t.astype(np.float32), seasonality.astype(np.float32)


def _effective_lat(grid, p: ClimateParams, seed=0):
    """Latitude with the circulation bands wobbled, so they are not perfect circles."""
    lat = grid.lat.astype(np.float64)
    if p.meander_deg <= 0.0:
        return lat
    from terrain import noise
    out = np.empty(grid.cell_count, dtype=np.float64)
    noise.fbm(np.ascontiguousarray(grid.xyz), noise.permutation(noise.sub_seed("wind", seed)),
              noise.GRAD3, 3, p.meander_frequency, 2.0, 0.5, out)
    return np.clip(lat + p.meander_deg * out, -90.0, 90.0)


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
    lat = _effective_lat(grid, p)
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
# Ocean currents and heat transport
# --------------------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _deflect_and_bound(xyz, nbr, lat, wind, is_ocean, ekman_rad, cur):
    """Ekman-deflect the wind over water, then make the result follow the coast.

    Two steps, and the second is the one that matters for heat transport. Deflection alone
    gives a gyre-like swirl but lets flow run straight into a continent, where the advection
    would simply stop. Removing the shoreward component instead turns the flow ALONG the
    coast -- which is how a subtropical gyre acquires a poleward western boundary, and the
    poleward western boundary is the entire Norway-against-Labrador effect.
    """
    n = xyz.shape[0]
    k = nbr.shape[1]
    for i in prange(n):
        cur[i, 0] = 0.0
        cur[i, 1] = 0.0
        cur[i, 2] = 0.0
        if not is_ocean[i]:
            continue
        # Rotate the tangent vector about the outward normal. n x v is v turned 90 degrees
        # anticlockwise seen from outside, so a positive angle turns left; the north needs
        # a right turn, hence the sign of the latitude.
        a = -ekman_rad if lat[i] >= 0.0 else ekman_rad
        ca, sa = np.cos(a), np.sin(a)
        nx, ny, nz = xyz[i, 0], xyz[i, 1], xyz[i, 2]
        wx, wy, wz = wind[i, 0], wind[i, 1], wind[i, 2]
        vx = wx * ca + (ny * wz - nz * wy) * sa
        vy = wy * ca + (nz * wx - nx * wz) * sa
        vz = wz * ca + (nx * wy - ny * wx) * sa

        # Collect the unit tangent direction toward each land neighbour, then project the
        # flow off every one of them that it still points into.
        #
        # Projecting off their AVERAGE is not enough, and the failure is not exotic: a cell
        # in a strait or at the head of a bay has land on two sides, the mean normal points
        # between them, and removing that one component leaves flow running straight into
        # one of the two. Measured before this loop: 1,580 coastal cells leaking, the worst
        # of them at full speed into the shore.
        ld = np.empty((6, 3), dtype=np.float64)
        n_land = 0
        for s in range(k):
            j = nbr[i, s]
            if j < 0 or is_ocean[j]:
                continue
            dot = xyz[j, 0] * nx + xyz[j, 1] * ny + xyz[j, 2] * nz
            dx = xyz[j, 0] - dot * nx
            dy = xyz[j, 1] - dot * ny
            dz = xyz[j, 2] - dot * nz
            dn = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dn > 0.0:
                ld[n_land, 0] = dx / dn
                ld[n_land, 1] = dy / dn
                ld[n_land, 2] = dz / dn
                n_land += 1
        # Sweeping repeatedly because removing one component can reintroduce another; a
        # cell walled in on every side converges to no current at all, which is correct.
        for _ in range(8):
            worst = 0.0
            for m in range(n_land):
                into = vx * ld[m, 0] + vy * ld[m, 1] + vz * ld[m, 2]
                if into > 0.0:
                    vx -= into * ld[m, 0]
                    vy -= into * ld[m, 1]
                    vz -= into * ld[m, 2]
                    if into > worst:
                        worst = into
            if worst <= 1e-12:
                break

        # Normalise, then CHECK, and drop the cell entirely if the result still runs ashore.
        # A cell whose land neighbours span more than half the compass has no admissible
        # direction at all: the sweeps shrink the vector toward zero, and normalising that
        # turns leftover rounding into a confident unit vector pointing wherever the noise
        # lay -- often straight inland. Enforcing the invariant on the final vector is both
        # exact and self-describing, where thresholds on the magnitude only ever moved the
        # worst leak around (0.89 into the shore, then 0.028).
        speed = np.sqrt(vx * vx + vy * vy + vz * vz)
        if speed <= 1e-12:
            continue
        vx, vy, vz = vx / speed, vy / speed, vz / speed
        ok = True
        for m in range(n_land):
            if vx * ld[m, 0] + vy * ld[m, 1] + vz * ld[m, 2] > 1e-9:
                ok = False
                break
        if ok:
            cur[i, 0] = vx
            cur[i, 1] = vy
            cur[i, 2] = vz
    return cur


def current_field(grid, land_mask, p: ClimateParams):
    """Surface ocean current, as tangent vectors. Zero on land.

    Wind-driven rather than prescribed. The three-cell circulation already alternates
    easterly (0-30) and westerly (30-60), and a constant Ekman deflection turns that
    alternation into a closed anticyclonic gyre in each basin without anything having to
    know where the basins are.
    """
    wind = wind_field(grid, p)
    cur = np.zeros((grid.cell_count, 3), dtype=np.float64)
    _deflect_and_bound(np.ascontiguousarray(grid.xyz), np.ascontiguousarray(grid.neighbours),
                       np.ascontiguousarray(grid.lat, dtype=np.float64), wind,
                       np.ascontiguousarray(~np.asarray(land_mask)),
                       float(np.radians(p.ekman_deg)), cur)
    return cur


@njit(cache=True, parallel=True)
def _advect_relax(field, nbr, rev, out_w, target, relax, pinned, new):
    """One relaxation sweep: what arrives from upwind, pulled back toward a local target.

    Used twice with different flow fields. Over the sea it advects temperature along the
    CURRENT, pinned on land, and gives the sea surface anomaly. Over land it advects that
    anomaly along the WIND, pinned at sea, and is how maritime air reaches inland.

    A weighted MEAN of the upwind cells, not a weighted sum. `_advect_step` below documents
    what happens with a sum: incoming weights exceed one wherever flow converges, the
    operator is not a contraction, and moisture had to be capped at saturation to stop it
    diverging to 1.3e7 mm/yr. A mean is bounded by the extremes of its inputs by
    construction, so temperature needs no cap and nothing to tune -- and it is the right
    physics anyway, since mixing two bodies of water averages their heat rather than
    accumulating it.
    """
    n = field.shape[0]
    k = nbr.shape[1]
    for i in prange(n):
        if pinned[i]:
            new[i] = target[i]
            continue
        acc = 0.0
        total = 0.0
        for s in range(k):
            j = nbr[i, s]
            t = rev[i, s]
            if j < 0 or t < 0:
                continue
            w = out_w[j, t]
            if w <= 0.0:
                continue
            acc += w * field[j]
            total += w
        upstream = acc / total if total > 0.0 else target[i]
        new[i] = (1.0 - relax) * upstream + relax * target[i]
    return new


def sst_anomaly(grid, land_mask, equilibrium_c, p: ClimateParams, spacing_km=None):
    """Departure of sea surface temperature from its own latitude's equilibrium.

    Positive where a current has brought warmer water than belongs here, negative where it
    has brought colder. Land is zero.

    NOTE that `wind_speed = 0` is not the way to switch this off. With no wind there is no
    advection, but `_outgoing_weights` still returns the isotropic `eddy` share, so the
    sweep degenerates to Laplacian smoothing of a curved equilibrium profile and leaves a
    small anomaly near the equator and the poles. That is diffusion, and it is real; the
    switch is `ocean_heat_c = 0`.
    """
    is_ocean = np.ascontiguousarray(~np.asarray(land_mask))
    if not is_ocean.any() or p.ocean_heat_c == 0.0:
        return np.zeros(grid.cell_count, dtype=np.float32)

    nbr = np.ascontiguousarray(grid.neighbours)
    out_w = np.zeros(nbr.shape, dtype=np.float64)
    _outgoing_weights(np.ascontiguousarray(grid.xyz), nbr,
                      current_field(grid, land_mask, p), p.eddy, out_w)
    rev = np.empty(nbr.shape, dtype=np.int32)
    _reverse_slots(nbr, rev)

    eq = np.ascontiguousarray(equilibrium_c, dtype=np.float64)
    sst = eq.copy()
    new = np.empty_like(sst)
    pinned = np.ascontiguousarray(~is_ocean)
    relax = per_step_relax(grid, p.current_scale_km, spacing_km)
    for _ in range(p.current_iterations):
        _advect_relax(sst, nbr, rev, out_w, eq, relax, pinned, new)
        sst, new = new, sst
    anom = np.where(is_ocean, sst - eq, 0.0)

    # Strip the zonal mean, and not as a cosmetic step. What is left after advection is a
    # net POLEWARD heat transport -- measured -5.4 C in the zonal mean at 30-45 degrees and
    # +2.8 C at 60-75 -- which is correct physics and must still not be added here, because
    # it is already in `t_equator_c` and `t_pole_c`. Those are calibrated against Earth, and
    # Earth's equator-to-pole gradient is as shallow as it is BECAUSE its oceans move heat
    # poleward. Adding a modelled transport on top of a profile that already contains one
    # counts it twice, and the result was a redistributed latitude profile rather than a
    # broken one: zonality 0.9926 -> 0.9830 with the mean left in.
    #
    # The departure from the zonal mean is the part that is genuinely new information -- it
    # is where a basin's western boundary differs from its eastern -- and it is what
    # `t_equator_c` cannot contain, because a single latitude profile has nowhere to put it.
    if p.strip_zonal_mean:
        anom -= _band_mean(anom, grid.lat, is_ocean, p.zonal_band_deg)[1]
        anom = np.where(is_ocean, anom, 0.0)
    return anom.astype(np.float32)


def maritime_anomaly(grid, land_mask, sst_anom, wind, p: ClimateParams, spacing_km=None):
    """Carry the sea surface anomaly inland ON THE WIND.

    The first version of this decayed the anomaly isotropically with distance from the
    coast, and it could not produce the effect it exists for. Measured: a mean land
    temperature change of 1.1 C and zonality 0.9926 -> 0.9805, against a maximum of 10.4 C
    at the coast. The extremes were right and the field was still a latitude profile.

    Isotropic decay was the wrong model, because the Norway-against-Labrador contrast is
    not that one coast sits next to warmer water. It is that the westerlies carry Atlantic
    air hundreds of kilometres INTO Europe, while at the same latitude in Labrador the wind
    blows from the continent out to sea and the warm water offshore never reaches the land.
    Direction is the whole mechanism, and distance alone throws it away.

    So the anomaly advects on the wind field, using the same sweep the currents use and
    relaxing toward zero over land -- an air mass losing its maritime character as the
    continent works on it. `maritime_scale_km` sets how far inland it gets.
    """
    anom = np.asarray(sst_anom, dtype=np.float64)
    if p.ocean_heat_c == 0.0 or not np.any(anom):
        return np.zeros(grid.cell_count, dtype=np.float32)

    nbr = np.ascontiguousarray(grid.neighbours)
    out_w = np.zeros(nbr.shape, dtype=np.float64)
    _outgoing_weights(np.ascontiguousarray(grid.xyz), nbr, wind, p.eddy, out_w)
    rev = np.empty(nbr.shape, dtype=np.int32)
    _reverse_slots(nbr, rev)

    # Ocean cells hold the sea surface anomaly; land relaxes toward zero.
    is_ocean = np.ascontiguousarray(~np.asarray(land_mask))
    target = np.ascontiguousarray(np.where(is_ocean, anom, 0.0))
    air = target.copy()
    new = np.empty_like(air)
    relax = per_step_relax(grid, p.maritime_scale_km, spacing_km)
    for _ in range(p.maritime_iterations):
        _advect_relax(air, nbr, rev, out_w, target, relax, is_ocean, new)
        air, new = new, air
    return air.astype(np.float32)


def _band_mean(values, lat_deg, valid, band_deg):
    """Latitude-band means of `values` over `valid`, and those means scattered per cell."""
    n_bands = max(int(round(180.0 / band_deg)), 1)
    idx = np.clip(((np.asarray(lat_deg, dtype=np.float64) + 90.0) / 180.0
                   * n_bands).astype(np.int64), 0, n_bands - 1)
    w = np.asarray(valid, dtype=np.float64)
    count = np.bincount(idx, weights=w, minlength=n_bands)
    total = np.bincount(idx, weights=w * np.asarray(values, dtype=np.float64),
                        minlength=n_bands)
    np.divide(total, count, out=total, where=count > 0)
    return total, total[idx]


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
    ascent = 0.5 + 0.5 * np.cos(np.pi * np.abs(_effective_lat(grid, p)) / 30.0)
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


def precipitation(grid, elevation_m, land_mask, temperature_c, wind, p: ClimateParams,
                  spacing_km=None):
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
    rainout = per_step_relax(grid, p.background_rainout_km, spacing_km)
    retain = np.ascontiguousarray(1.0 - rainout * rainout_factor(grid, p))

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
    # Cap as a FRACTION OF THE RADIUS. 6000 km on Earth is 0.942 R; as a hardcoded km it
    # would exceed the antipodal distance on the small sphere, so nothing would ever be
    # "deep interior" and continentality would lose its ceiling.
    # Continentality is a world-scale length -- how far inland the sea stops moderating --
    # so it is a fraction of the radius, not an absolute distance. See
    # PlateParams.for_radius for the same argument at length.
    p = dataclasses.replace(p, continentality_scale_km=(
        p.continentality_scale_km * float(grid.radius_km) / R_WORLD_KM))

    cap = np.float32(0.942 * grid.radius_km)
    plates.boundary_distance(~land_mask, grid.xyz, grid.radius_km, dist, src, cap)
    dist[dist > 1e29] = cap

    # Wind comes first now: temperature depends on it, because the sea surface anomaly has
    # to be blown ashore before it can be added. It used to be the other way round, when
    # temperature was a function of latitude, elevation and coast distance only.
    wind = wind_field(grid, p)
    spacing = mean_spacing_km(grid)
    anom = sst_anomaly(grid, land_mask, equilibrium_temperature(grid, p), p, spacing)
    maritime = maritime_anomaly(grid, land_mask, anom, wind, p, spacing)
    t, seasonality = temperature(grid, elevation_m, dist, p, maritime_anom=maritime)
    precip, moisture = precipitation(grid, elevation_m, land_mask, t, wind, p, spacing)

    return Climate(temperature_c=t, seasonality_c=seasonality,
                   wind=wind.astype(np.float32), precipitation_mm=precip,
                   moisture=moisture, coast_distance_km=dist, sst_anomaly_c=anom)
