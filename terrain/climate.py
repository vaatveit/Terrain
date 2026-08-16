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

The limit is vertical, not horizontal. Measured at production resolution, land elevation
above the datum has a standard deviation of 104 m against Earth's ~800, so the lapse rate
contributes 0.68 C where on Earth it would contribute around 5 -- against a latitude term
whose standard deviation is 13.7 C. Relief, not distance, is what the temperature field is
short of.

There is no knob for it, and that is settled rather than open. Raising uplift is blocked by a
hillslope budget already past the angle of repose; reaching Earth's relief by growing the
world would take 2.35e9 cells, 382x the production budget. Relative to its own size this
world is already 12.7x steeper than Earth -- what it lacks is absolute metres, which is
exactly what the 100 m cell decision traded away. The lapse rate is the only physics here
denominated in absolute metres, which is why it alone reads the shrink as a defect. (These
figures read 61 m and 0.28 C until they were measured at n=785: the originals came off a
prototype 3x smaller in radius AND were taken over land including cells below the datum,
which inverted the sign of the scaling law. See terrain-algorithms.md.)
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
    # Seasons. Tilt drives declination; the lag is real physics but it is NOT what keeps
    # the four samples distinct -- the sample dates are.
    #
    # This model has no heat storage: each solve is a steady state, and declination is the
    # only thing that carries the date into it. So two dates with the SAME declination give
    # bit-identical climates, and sampling at the solstices and equinoxes yields three
    # distinct seasons, not four, because both equinoxes sit at declination zero. Sampling
    # four evenly spaced dates offset from them gives four. See SEASON_PHASE.
    #
    # The lag is kept because it is true -- seasons trail the sun by four to eight weeks
    # because the ocean stores heat, so what a place feels today is the sun it had a month
    # ago -- and it shifts which calendar dates those declinations fall on.
    axial_tilt_deg: float = 23.44
    seasonal_lag_days: float = 30.0
    days_per_year: float = 365.24
    # Degrees of profile shift per unit of insolation departure. 1.0 puts the seasonal
    # swing at 1.4 C on the equator, 14.6 at 45 degrees and 20.7 at the pole. Setting it to
    # 0 removes seasons entirely and every solve returns the annual mean.
    seasonal_gain: float = 1.0
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
    # How far the circulation follows the subsolar point. 0.4 of a 23.44 degree declination
    # gives the ITCZ a 9.4 degree excursion each way, matching Earth's zonal mean.
    itcz_follow: float = 0.4
    # Latitude at which the seasonal migration has died away entirely. Keeps the swing in
    # the tropics where monsoons live and off the westerlies where rain shadows live.
    itcz_taper_deg: float = 45.0
    # Monsoon: how hard the wind is pulled toward summer-heated land, relative to
    # `wind_speed`, for a land-sea temperature contrast of `monsoon_ref_c`. At 0 the bands
    # still migrate with the sun but the wind never reverses, which is a wet season without
    # a monsoon.
    # IN DIRECT TENSION WITH THE RAIN SHADOW, the same way land_recycling is with
    # orographic_coeff, and worth knowing before touching it. The orographic signal needs a
    # consistent prevailing direction; a thermal wind strong enough to reverse the coast is
    # also strong enough to blur which side of a ridge is upwind. Measured on one world,
    # windward/leeward ratio against the fraction of tropical coast whose wind reverses
    # between solstices (shadow at n=96, reversal at n=256):
    #
    #   monsoon_c    0.00    0.10    0.20    0.30    0.60
    #   shadow      2.15x   2.30x   2.32x   1.91x   1.41x
    #   reversal    32.5%   34.3%   38.7%   43.2%   58.0%
    #
    # 0.20 is the knee and is shipped: it is the best shadow of the five -- better than
    # switching the monsoon off, because a little seasonal steering makes each season's wind
    # more consistent, not less -- while still moving the reversal. 0.60 buys a much
    # stronger monsoon for a third of the rain shadow, which is not a trade worth making.
    monsoon_c: float = 0.2
    # The contrast that counts as a strong thermal low, and the distance it acts over. The
    # reference length must scale with the sphere: the first version used the CELL SPACING,
    # which is 100 m at every radius, so the reference gradient came out 80 C/km and the
    # monsoon term contributed 0.006 against a wind speed of 1 -- nothing at all.
    monsoon_ref_c: float = 8.0
    monsoon_ref_km: float = 5.0
    # How much more than the sea a continent swings through the year. Land has almost no
    # heat capacity and follows the sun; the ocean stores the season and barely moves.
    # 1.8x deep inland against 0.35x at sea is what opens a thermal low over a summer
    # continent, and without it there is no monsoon to deflect anything.
    seasonal_continentality: float = 0.8
    sea_seasonal_damping: float = 0.35
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
    # Anchor for the final scaling. Earth's land MEDIAN, and the median is the point.
    #
    # This model produces a pattern, not depths, so the anchor is a free choice -- and the
    # mean is the wrong free choice, because this pattern has a very heavy wet tail. The
    # wettest 5% of land takes 47.5% of all the rain that falls on land, against roughly
    # 15% on Earth, so pinning the mean at Earth's 1000 mm divided everyone else down to a
    # median of 321 mm where Earth's is ~600. Half the world was arid by construction.
    #
    # Judged on the classification, which is what actually reads absolute millimetres.
    # Koppen arid share against Earth's 30%, on one n=128 world:
    #
    #   anchor       mean 1000   med 450   med 600   med 750
    #   Koppen B        41.3%     35.4%     30.5%     27.0%
    #   tropical rf     ~7%        11%       15%       18%     (earth 11%)
    #   runoff R/P       0.59      0.66      0.72      0.76    (earth ~0.40)
    #
    # 600 is Earth's figure rather than the one that scored best, and B lands on 30.5%.
    # The costs are real and are the SHAPE showing through: the wet tail overshoots the
    # rainforest share, and runoff ratio drifts further from Earth because a heavy-tailed
    # rainfall runs off more efficiently than an even one. No anchor fixes that; see the
    # note on why the shape does not respond to tuning in `_advect_step`.
    target_median_precip_mm: float = 600.0


@dataclasses.dataclass
class Climate:
    temperature_c: np.ndarray     # (N,) f32, mean annual at the surface
    seasonality_c: np.ndarray     # (N,) f32, continentality-driven amplitude
    wind: np.ndarray              # (N,3) f32, tangent vectors
    precipitation_mm: np.ndarray  # (N,) f32, mm/yr
    moisture: np.ndarray          # (N,) f32, steady-state airborne moisture
    coast_distance_km: np.ndarray  # (N,) f32
    sst_anomaly_c: np.ndarray = None  # (N,) f32, sea surface departure from its latitude
    # Present only from `compute_seasons`. (4, N) stacks through the year, in the order of
    # SEASON_PHASE; `temperature_c` and `precipitation_mm` remain their annual summaries.
    temperature_season: np.ndarray = None
    precipitation_season: np.ndarray = None
    declination_deg: np.ndarray = None    # (4,) f32, the sun of each sample


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


# Where in the year the four samples sit, as fractions of it. Evenly spaced, because the
# harmonic fit that turns them into twelve months needs even spacing; offset off the
# solstices, because a sample set containing both equinoxes contains the same climate twice.
# The offset here puts the declinations near -22, -8, +22, +8 degrees.
SEASON_PHASE = (0.0, 0.25, 0.5, 0.75)


def declination_deg(day_of_year, p: ClimateParams):
    """Subsolar latitude on a given day, with the seasonal lag already applied.

    The lag is subtracted from the date rather than added to the answer: what a place feels
    today is the sun it had some weeks ago, because the ocean has been storing the
    difference.
    """
    phase = 2.0 * np.pi * ((day_of_year - p.seasonal_lag_days) / p.days_per_year - 0.22)
    return p.axial_tilt_deg * np.sin(phase)


def daily_insolation(lat_deg, decl_deg):
    """Daily-mean insolation, normalised to 1 at the subsolar latitude of an equinox.

    The day-length term is the reason this is not just cos(latitude - declination). That
    shortcut has no polar night: it makes the summer pole the warmest place on the planet
    and gives the winter pole an ordinary winter, when in fact one has twenty-four hours of
    low sun and the other has none at all. Koppen's D and E classes turn on exactly that
    difference, so the closed form is worth its ten lines.

        Q ~ H0 sin(lat) sin(decl) + cos(lat) cos(decl) sin(H0)

    with H0 the half-day length as an hour angle: pi under the midnight sun, 0 in polar
    night, arccos(-tan(lat) tan(decl)) in between.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    decl = np.radians(float(decl_deg))
    cos_h0 = np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0)
    h0 = np.arccos(cos_h0)
    q = h0 * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(h0)
    return np.maximum(q, 0.0) / np.pi


def equilibrium_temperature(grid, p: ClimateParams, decl_deg=0.0, mean_insol=None):
    """Sea-level temperature from latitude and the season: the profile all else departs from.

    At `decl_deg = 0` this reproduces the original cos(lat)**1.5 profile exactly, so every
    caller that does not care about seasons is unaffected. `t_equator_c` and `t_pole_c` go
    on meaning the annual means they were calibrated as; see the note on clipping below for
    why that survives.
    """
    lat = np.radians(grid.lat.astype(np.float64))
    annual = np.cos(lat) ** 1.5
    if decl_deg == 0.0 or p.seasonal_gain == 0.0:
        return p.t_pole_c + (p.t_equator_c - p.t_pole_c) * annual

    # ADDITIVE, not multiplicative. Scaling the annual profile by an insolation ratio was
    # the obvious first try and it leaves the poles with no seasons at all: cos(lat)**1.5
    # is zero there, so no factor can move it, and the pole sat at t_pole_c in every season
    # -- deleting the one distinction the day-length term was added to capture.
    #
    # Departing from the ANNUAL MEAN insolation at each latitude works everywhere, because
    # the mean is nonzero at the pole (it has a summer) and the departure from it is the
    # largest on the planet. Measured swings at gain 1: 1.4 C at the equator, 14.6 C at 45
    # degrees, 20.7 C at the pole, against Earth's roughly 2, 10-25 and 35.
    # Deliberately NOT clipped to [0, 1]. The insolation anomaly averages to zero over the
    # year by construction, so leaving it unclipped makes the annual mean of the seasons
    # reproduce the tuned profile EXACTLY -- `t_equator_c` and `t_pole_c` go on meaning the
    # annual means they were calibrated as. Clipping at zero cost that: it truncated the
    # winter pole and pulled the polar annual mean about 5 C above `t_pole_c`.
    #
    # The price is that the winter pole runs colder than `t_pole_c` and the summer pole
    # warmer, which is the correct shape -- Earth's north pole averages about -18 C between
    # a -34 C winter and a 0 C summer.
    # The datum does not depend on the season, so a caller solving several seasons on one
    # grid should compute it once and hand it in -- it is 24 insolation evaluations over
    # every cell, and recomputing it per season made four seasons cost 6.7x one rather
    # than 4x.
    if mean_insol is None:
        mean_insol = _mean_annual_insolation(grid.lat, p)
    anomaly = daily_insolation(grid.lat, decl_deg) - mean_insol
    return (p.t_pole_c + (p.t_equator_c - p.t_pole_c)
            * (annual + p.seasonal_gain * anomaly))


def _mean_annual_insolation(lat_deg, p: ClimateParams, samples=24):
    """Insolation averaged over the year, per latitude. The datum seasons depart from."""
    days = np.arange(samples) * p.days_per_year / samples
    total = np.zeros(np.shape(lat_deg), dtype=np.float64)
    for d in days:
        total += daily_insolation(lat_deg, declination_deg(d, p))
    return total / samples


def temperature(grid, elevation_m, coast_distance_km, p: ClimateParams,
                maritime_anom=None, decl_deg=0.0, mean_insol=None, land_mask=None):
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
    t = (equilibrium_temperature(grid, p, decl_deg, mean_insol)
         - p.lapse_rate_c_per_km * np.maximum(elevation_m, 0.0) / 1000.0
         - p.continental_cooling_c * inland * np.abs(np.sin(lat)) ** 1.5)

    # DIFFERENTIAL HEATING. Rock and water do not take a season the same way: land has
    # little heat capacity and swings with the sun, the sea stores the heat and barely
    # moves. That difference IS the monsoon -- a summer continent becomes hot relative to
    # the ocean beside it, the pressure over it drops, and maritime air is pulled in.
    #
    # Without this term the model had no land-sea contrast to speak of. Measured at the June
    # solstice: 0.58 C of standard deviation, with land averaging 0.42 C COLDER than the sea
    # rather than warmer, because continental cooling was the only thing distinguishing
    # them. The thermal low had nothing to form on and the monsoon wind did nothing at all.
    if decl_deg != 0.0 and p.seasonal_gain != 0.0 and land_mask is not None:
        departure = (equilibrium_temperature(grid, p, decl_deg, mean_insol)
                     - equilibrium_temperature(grid, p, 0.0))
        amplitude = np.where(np.asarray(land_mask),
                             1.0 + p.seasonal_continentality * inland,
                             p.sea_seasonal_damping)
        t += (amplitude - 1.0) * departure

    if maritime_anom is not None and p.ocean_heat_c != 0.0:
        t += p.ocean_heat_c * np.asarray(maritime_anom, dtype=np.float64)

    seasonality = p.continentality_c * inland
    return t.astype(np.float32), seasonality.astype(np.float32)


def band_shift_deg(lat_deg, decl_deg, p: ClimateParams):
    """How far the circulation slides with the sun, per latitude.

    Damped because the ITCZ does not track the subsolar point one for one: the ocean's heat
    capacity holds it back, so Earth's migrates roughly 0 to 10 degrees over a year against
    a subsolar point covering 47. Land does move further -- the monsoon trough reaches 25 N
    over India -- but that is the thermal low in `monsoon_wind`, not the zonal mean.

    TAPERED WITH LATITUDE. Shifting every band by the same 9.4 degrees moves the
    mid-latitude westerlies too, and since the bands are only 30 degrees wide that reverses
    the prevailing wind for any cell within 9 degrees of a boundary. Earth does not work
    that way: the ITCZ swings, the storm track shifts a little, and the westerlies at 45
    degrees blow from the west in every month of the year.

    Kept on those grounds alone. It was added while chasing a collapse in the rain shadow
    (1.56x to 1.08x) and did not fix it -- the cause was `compute_seasons` returning one
    season's wind beside a year's rainfall, and correcting that took the ratio to 1.48x with
    the taper contributing nothing measurable either way.
    """
    lat = np.abs(np.asarray(lat_deg, dtype=np.float64))
    taper = np.maximum(0.0, 1.0 - (lat / p.itcz_taper_deg) ** 2)
    return p.itcz_follow * float(decl_deg) * taper


def _effective_lat(grid, p: ClimateParams, seed=0, decl_deg=0.0):
    """Latitude measured from the circulation's centre, with the bands wobbled.

    Two separate displacements of the same coordinate. The wobble is spatial and standing
    in for Rossby waves; the shift is seasonal and is what carries the wet band over a
    tropical continent in its summer and away again in its winter.
    """
    lat = grid.lat.astype(np.float64)
    lat = lat - band_shift_deg(lat, decl_deg, p)
    if p.meander_deg <= 0.0:
        return np.clip(lat, -90.0, 90.0)
    from terrain import noise
    out = np.empty(grid.cell_count, dtype=np.float64)
    noise.fbm(np.ascontiguousarray(grid.xyz), noise.permutation(noise.sub_seed("wind", seed)),
              noise.GRAD3, 3, p.meander_frequency, 2.0, 0.5, out)
    return np.clip(lat + p.meander_deg * out, -90.0, 90.0)


def wind_field(grid, p: ClimateParams, decl_deg=0.0):
    """Surface wind from the three-cell circulation, as tangent vectors.

    Bands are easterly 0-30 (trades), westerly 30-60, easterly 60-90 (polar). Writing the
    zonal component as -cos(pi*(|lat|-15)/30) puts the band centres at 15/45/75 degrees and
    zeroes the wind at 0/30/60/90 -- the doldrums, horse latitudes and polar front, which
    is where the real circulation goes slack.

    The meridional component follows from the same expression: surface flow is equatorward
    in the Hadley and Polar cells and poleward in the Ferrel cell, which is exactly where
    the zonal sign flips, so v is proportional to sign(lat) * u.
    """
    lat = _effective_lat(grid, p, decl_deg=decl_deg)
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


@njit(cache=True, parallel=True)
def _thermal_gradient(xyz, nbr, arc_dist, field, grad):
    """Surface gradient of a scalar on the mesh, as a tangent vector per cell.

    Same tangent construction the advection weights use, so "uphill in temperature" and
    "downwind" are measured in one geometry rather than two.
    """
    n = xyz.shape[0]
    k = nbr.shape[1]
    for i in prange(n):
        gx = gy = gz = 0.0
        total = 0.0
        for s in range(k):
            j = nbr[i, s]
            if j < 0:
                continue
            d = arc_dist[i, s]
            if d <= 0.0:
                continue
            dot = xyz[j, 0] * xyz[i, 0] + xyz[j, 1] * xyz[i, 1] + xyz[j, 2] * xyz[i, 2]
            tx = xyz[j, 0] - dot * xyz[i, 0]
            ty = xyz[j, 1] - dot * xyz[i, 1]
            tz = xyz[j, 2] - dot * xyz[i, 2]
            tn = np.sqrt(tx * tx + ty * ty + tz * tz)
            if tn <= 0.0:
                continue
            slope = (field[j] - field[i]) / d
            gx += slope * tx / tn
            gy += slope * ty / tn
            gz += slope * tz / tn
            total += 1.0
        if total > 0.0:
            grad[i, 0] = gx / total
            grad[i, 1] = gy / total
            grad[i, 2] = gz / total
        else:
            grad[i, 0] = grad[i, 1] = grad[i, 2] = 0.0
    return grad


def monsoon_wind(grid, land_mask, temperature_c, p: ClimateParams, decl_deg=0.0):
    """The three-cell wind, plus a component blowing toward summer-heated land.

    This is the term that makes a monsoon a monsoon. Migrating the circulation bands with
    the sun (`itcz_lat`) moves the wet band over a tropical continent and gives it a wet
    season, but the wind itself never turns round: at a fixed latitude it is the same
    easterly in January as in July. A real monsoon REVERSES, because summer heating over a
    continent drops the surface pressure and pulls maritime air in, and winter cooling does
    the opposite.

    Modelled as flow up the land-sea temperature contrast -- toward the warm anomaly, which
    is inland in summer and offshore in winter. The contrast is measured against the zonal
    mean so it is a land-sea difference rather than the equator-to-pole gradient, which is
    far larger and would otherwise swamp it and simply blow everything equatorward.
    """
    wind = wind_field(grid, p, decl_deg=decl_deg)
    if p.monsoon_c <= 0.0:
        return wind

    t = np.asarray(temperature_c, dtype=np.float64)
    contrast = np.ascontiguousarray(t - _band_mean(t, grid.lat, np.ones_like(t),
                                                   p.zonal_band_deg)[1])
    grad = np.empty((grid.cell_count, 3), dtype=np.float64)
    _thermal_gradient(np.ascontiguousarray(grid.xyz),
                      np.ascontiguousarray(grid.neighbours),
                      np.ascontiguousarray(grid.arc_dist, dtype=np.float64),
                      contrast, grad)

    # Scale so the term is comparable to `wind_speed` for a contrast of monsoon_ref_c
    # degrees across one cell; without a reference the units are metres and the result is
    # either nothing or everything.
    # No land mask on the result. The contrast is near zero over open water anyway -- that
    # is what makes it a land-sea signal -- so masking would only be belt and braces.
    ref = p.monsoon_ref_c / (p.monsoon_ref_km * float(grid.radius_km) / R_WORLD_KM)
    return np.ascontiguousarray(wind + (p.monsoon_c / ref) * grad)


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
    """Latitude-band means of `values` over `valid`, evaluated smoothly at each cell.

    INTERPOLATED between band centres, not held constant across each band, and the
    difference is visible from orbit. Taking a cell's own band's mean makes the subtracted
    field a staircase, so every band edge is a step -- measured at 0.11 C median and 1.55 C
    at worst on a 2 degree grid. That is small, and it still ruled the map: a large desert
    interior sits near the arid threshold over its whole area rather than at scattered
    cells, so one step flips a whole band at once. The classification came out ruled with
    horizontal stripes, and a Fourier transform of the class fraction against latitude put
    them at 2.02 degrees -- `zonal_band_deg`, exactly.

    Linear interpolation between band centres removes the steps outright and costs one
    np.interp. Bands with nothing in them are skipped rather than treated as zero, so a
    latitude with no ocean does not drag its neighbours down.
    """
    n_bands = max(int(round(180.0 / band_deg)), 1)
    lat = np.asarray(lat_deg, dtype=np.float64)
    idx = np.clip(((lat + 90.0) / 180.0 * n_bands).astype(np.int64), 0, n_bands - 1)
    w = np.asarray(valid, dtype=np.float64)
    count = np.bincount(idx, weights=w, minlength=n_bands)
    total = np.bincount(idx, weights=w * np.asarray(values, dtype=np.float64),
                        minlength=n_bands)
    np.divide(total, count, out=total, where=count > 0)

    filled = count > 0
    if not filled.any():
        return total, np.zeros_like(lat)
    centres = -90.0 + (np.arange(n_bands) + 0.5) * (180.0 / n_bands)
    return total, np.interp(lat, centres[filled], total[filled])


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


def rainout_factor(grid, p: ClimateParams, decl_deg=0.0):
    """Where the three-cell circulation has air rising (wet) versus subsiding (dry).

    cos(pi*|lat|/30) is +1 at 0 and 60 degrees -- the ITCZ and the polar front, where air
    ascends -- and -1 at 30 and 90, the horse latitudes and the poles, where it descends.
    Suppressing background rainout in the descending bands puts the subtropical deserts
    where they belong, while leaving orographic rainfall intact, so a range inside the
    desert belt still catches rain the way the Atlas does on the edge of the Sahara.
    """
    ascent = 0.5 + 0.5 * np.cos(
        np.pi * np.abs(_effective_lat(grid, p, decl_deg=decl_deg)) / 30.0)
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

    THE OUTPUT DISTRIBUTION IS TWICE AS WIDE AS EARTH'S AND NOTHING HERE FIXES IT. Land
    rainfall comes out with a log10 standard deviation of 0.83 against Earth's ~0.45, and
    the wettest 5% of land takes 47.5% of all land rain against roughly 15%. Recorded here
    because the obvious diagnosis is wrong and the obvious knobs do not work.

    It is NOT convergence. The wettest 5% of land has a median incoming advection weight of
    1.001 against 0.999 for the other 95% -- identical -- and only 11.9% of those cells are
    even saturated. A `convective_efficiency` factor scaling the overflow above was written
    and measured before being reverted: a tenfold change moved log10 sd 0.837 -> 0.823.
    Nothing above depends on it, so the surplus is not where the tail rain comes from.

    What the tail is instead is the two intended mechanisms multiplying. The wettest cells
    hold 5.3x the moisture of the rest AND sit where `rainout_factor` is ~1 rather than at
    `subsidence_floor`; wet-and-ascending against dry-and-subsiding is a product, not a sum.
    So the spread is the three-cell circulation being the only thing that shapes rainfall,
    and every knob on it trades one Earth comparison for another. Measured on one world
    (log10 sd, land median mm, share of rain in the wettest 5%):

        shipped                  0.833   321   47.5%
        background_rainout 2x    1.048   264   55.0%      drier median, fatter tail
        background_rainout 0.5x  1.072   321   42.1%
        eddy 0.07 -> 0.70        0.916   231   51.9%      narrower, but median falls
        land_recycling 0.85      1.047   357   42.6%      mild, and fights the shadow
        iterations 400 -> 1200   1.055   317   47.0%      converged; not a solver artifact

    (The single-solve figures run wider than the shipped 0.833 because averaging the four
    seasons compresses the distribution.) This is the same limit already recorded at the
    top of this module: a surface-only model has no aloft. The anchor is chosen to make the
    best of it -- see `target_median_precip_mm` -- and the shape is left alone.
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
                  spacing_km=None, decl_deg=0.0, anchor=True):
    """Steady-state moisture transport with orographic and background rainout.

    `anchor=False` returns the pattern in the solver's own arbitrary units instead of
    millimetres. `compute_seasons` needs that -- see `_annual_anchor_scale` for why a season
    must not normalise itself.
    """
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
    retain = np.ascontiguousarray(
        1.0 - rainout * rainout_factor(grid, p, decl_deg=decl_deg))

    recycle = np.ascontiguousarray(np.where(land_mask, p.land_recycling, 0.0))

    moisture = np.zeros(n, dtype=np.float64)
    new_m = np.empty(n, dtype=np.float64)
    precip = np.zeros(n, dtype=np.float64)
    for _ in range(p.iterations):
        _advect_step(moisture, nbr, rev, out_w, lift_loss, retain, source, capacity,
                     recycle, new_m, precip)
        moisture, new_m = new_m, moisture

    # Scale to a realistic annual total. Absolute units are not derivable from this model;
    # what it produces is a spatial pattern, and the pattern is what Stage 4 needs. The
    # anchor is the area-weighted land MEDIAN, not the mean, because the mean here is a
    # statistic of the wet tail rather than of the world -- see `target_median_precip_mm`.
    if anchor:
        ref = _area_median(precip, grid.area, land_mask)
        if ref > 0:
            precip *= p.target_median_precip_mm / ref
    return precip.astype(np.float32), moisture.astype(np.float32)


# The anchor is a division, and this is the test for whether the divisor still means
# anything: the land median has to be at least this fraction of the land mean. See
# `_annual_anchor_scale`, and read the whole of it before touching the number.
MIN_ANCHOR_RATIO = 0.03


def _annual_anchor_scale(precip, area, land_mask, target_mm):
    """One factor for the whole year, checked before it is used.

    TWO CHANGES LIVE HERE, and the second is the reason the first is safe.

    **Each season used to normalise its own median to `target_median_precip_mm`, and no
    longer does.** That was wrong on its face -- a dry season SHOULD read a lower median than
    a wet one, and pinning them equal is what the seasonal amplitude means. It was also
    dangerous, because the normalisation is a division by a median that can underflow.
    Measured on the SHIPPED world at n=785 (seed 1, Stage 1 crust), median/mean per season:

        season 0   0.00101      scale 6.7e7      peak 2.5e8 mm/yr
        season 1   0.0380       scale 1.7e6      peak 7.0e6
        season 2   0.00126      scale 4.5e7      peak 1.7e8
        season 3   0.0275       scale 2.2e6      peak 9.8e6

    Two of the four solstice/equinox solves anchor on numerical floor. Because each was then
    scaled to the same median, their MEANS came out at 5.9e5, 1.6e4, 4.7e5 and 2.2e4 -- so
    the annual mean of the four was ~96% seasons 0 and 2, the two degenerate ones, amplified
    thirtyfold against the two sound ones. The year the pipeline runs on was mostly floor
    noise wearing a healthy annual median of exactly 600.0 mm.

    Anchoring once, on the annual field, removes the amplification: the seasons keep their
    relative amplitude and a dry season stays dry. The same n=785 world's four season
    medians afterwards, in mm:

        before (each self-normalised)    600.0    600.0    600.0    600.0
        after  (one annual factor)         6.0    228.1      8.8    178.1

    which is the model saying plainly that on a world this size the continents get almost no
    rain in two seasons of four. That was always true; normalising it away is what hid it.

    **And the surviving division is checked.** `ref / mean` under `MIN_ANCHOR_RATIO` means
    the median land cell holds the numerical floor of the moisture advection rather than a
    place on the rainfall distribution, and the reciprocal is reading noise. Measured on the
    annual field, which is the only anchor left:

        shipped 0.29/0.40   0.2666 (n=64)  0.2365 (n=128)  0.2050 (n=256)  0.0946 (n=785)
        0.40/0.50           --             0.2095          --              --
        0.45/0.55           --             0.00899         --              --
        0.50/0.65           --             0.00515         0.00044         --

    0.03 sits 3.2x under the worst healthy world and 3.3x over the best broken one. That
    margin is NOT generous and it is the honest one -- an earlier attempt claimed a hundred-
    fold separation by leaving n=785 out of the table.

    THE THRESHOLD IS A TRIPWIRE AND IT CERTIFIES NOTHING. A previous attempt at this guard
    was calibrated on n=64/128/256 and fired on the shipped world at n=785, because the
    healthy ratio falls with world size -- the continents widen in km while the moisture
    model's reach does not -- and a straight-line fit through three sizes missed the real
    value by fifty times. Passing this check does NOT mean the interiors are physical. At
    n=785 the shipped world has 22.8% of land under 1 mm/yr, and rainfall past ~8 km from
    the coast is effectively zero:

        distance from coast   0-1km   2-4km   4-8km    8-16km    16-24.7km
        share of land         19.5%   19.3%   22.8%    19.6%     4.4%
        median precip (mm)    3,480     941     442      1.51     1.4e-6

    That is a physics gap -- a surface-only model has no aloft and no soil-moisture
    recycling, so there is nothing to rain in a continental interior -- and the fix for it is
    a land moisture source, not a lower limit here. If this ever fires on a world that looks
    reasonable, that is the finding, and the number below is not what should move.
    """
    if not np.any(land_mask):
        return 1.0
    ref = _area_median(precip, area, land_mask)
    mean = float(np.average(precip[land_mask], weights=area[land_mask]))
    if mean <= 0.0 or ref <= 0.0:
        return 1.0
    ratio = ref / mean
    if ratio < MIN_ANCHOR_RATIO:
        raise ValueError(
            f"precipitation anchor is degenerate: the land median is {ratio:.2g} of the "
            f"land mean, under the {MIN_ANCHOR_RATIO} limit, so the median cell holds the "
            "numerical floor of the moisture advection rather than a place on the rainfall "
            "distribution. The continents are too wide for a surface-only moisture model to "
            "carry water across. Lower continental_fraction or target_land_fraction, or "
            "break the continents up with a higher crust_frequency.")
    return target_mm / ref


def _area_median(values, area, mask):
    """Area-weighted median over `mask`. Cell areas vary, so a plain median is wrong."""
    v = np.asarray(values, dtype=np.float64)
    a = np.asarray(area, dtype=np.float64)
    if mask is not None and np.any(mask):
        v, a = v[mask], a[mask]
    if v.size == 0:
        return 0.0
    order = np.argsort(v)
    cum = np.cumsum(a[order])
    if cum[-1] <= 0:
        return 0.0
    return float(np.interp(0.5 * cum[-1], cum, v[order]))


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


def compute_climate(grid, elevation_m, land_mask, params: ClimateParams = None,
                    decl_deg=0.0, mean_insol=None, anchor=True) -> Climate:
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

    # Order matters and has changed twice. Wind before temperature, because the sea surface
    # anomaly has to be blown ashore before it can be added. Then temperature before the
    # MONSOON wind, because that one blows toward warm land -- so the plain three-cell wind
    # carries the ocean heat, and the thermally corrected wind carries the rain.
    spacing = mean_spacing_km(grid)
    wind = wind_field(grid, p, decl_deg=decl_deg)
    if mean_insol is None and decl_deg != 0.0:
        mean_insol = _mean_annual_insolation(grid.lat, p)
    eq = equilibrium_temperature(grid, p, decl_deg, mean_insol)
    anom = sst_anomaly(grid, land_mask, eq, p, spacing)
    maritime = maritime_anomaly(grid, land_mask, anom, wind, p, spacing)
    t, seasonality = temperature(grid, elevation_m, dist, p, maritime_anom=maritime,
                                 decl_deg=decl_deg, mean_insol=mean_insol,
                                 land_mask=land_mask)
    wind = monsoon_wind(grid, land_mask, t, p, decl_deg=decl_deg)
    precip, moisture = precipitation(grid, elevation_m, land_mask, t, wind, p, spacing,
                                     decl_deg=decl_deg, anchor=anchor)

    return Climate(temperature_c=t, seasonality_c=seasonality,
                   wind=wind.astype(np.float32), precipitation_mm=precip,
                   moisture=moisture, coast_distance_km=dist, sst_anomaly_c=anom)


def compute_seasons(grid, elevation_m, land_mask, params: ClimateParams = None) -> Climate:
    """Four solves through the year, returned as one Climate with seasonal stacks.

    `temperature_c` and `precipitation_mm` stay the ANNUAL mean and total, so `coupled.run`,
    `discharge_weight`, `biomes.classify`, hydrology and the export all see exactly what
    they saw before. Erosion has to be driven by a year's water; seasonality is an output.

    Four samples rather than two because four evenly spaced points determine a two-harmonic
    year exactly -- mean, annual and semiannual -- which is what makes the twelve monthly
    values Koppen needs an interpolation rather than an assumption. Measured: the mean of
    the four reproduces the tuned annual temperature profile to 0.03 C.
    """
    p = params or ClimateParams()
    decls = [declination_deg(f * p.days_per_year, p) for f in SEASON_PHASE]
    mean_insol = _mean_annual_insolation(grid.lat, p)
    # anchor=False: the seasons come back in the solver's own units and keep their relative
    # amplitude. One factor is fitted below, on the year. See `_annual_anchor_scale`.
    seasons = [compute_climate(grid, elevation_m, land_mask, p, decl_deg=d,
                               mean_insol=mean_insol, anchor=False) for d in decls]

    t = np.mean([s.temperature_c for s in seasons], axis=0).astype(np.float32)
    precip = np.mean([s.precipitation_mm for s in seasons], axis=0)
    stack_p = np.stack([s.precipitation_mm for s in seasons]).astype(np.float64)

    # THE anchor, and now the only one: the four solves above are unnormalised, so this
    # single factor is what puts the year in millimetres. Carrying it into the season stack
    # unchanged is what preserves the seasonal SHAPE, which is what Koppen reads -- and now
    # the seasonal AMPLITUDE too, which self-normalising seasons destroyed.
    scale = _annual_anchor_scale(precip, grid.area, land_mask, p.target_median_precip_mm)
    precip = precip * scale
    stack_p = stack_p * scale
    precip = precip.astype(np.float32)
    stack_p = stack_p.astype(np.float32)
    stack_t = np.stack([s.temperature_c for s in seasons])
    # The seasonal swing is now measured rather than modelled from coast distance alone.
    seasonality = ((stack_t.max(axis=0) - stack_t.min(axis=0)) * 0.5).astype(np.float32)

    # EVERY aggregate field is a mean over the seasons, wind and moisture included. Taking
    # them from the first season instead was wrong in a way that is easy to miss and hard to
    # see: the returned object then carried one season's wind beside a year's rainfall, and
    # anything asking which side of a ridge is upwind -- the rain shadow metric, for one --
    # was answering with a wind that only blew for a quarter of the year.
    wind = np.mean([s.wind for s in seasons], axis=0).astype(np.float32)
    moisture = np.mean([s.moisture for s in seasons], axis=0).astype(np.float32)
    anom = np.mean([s.sst_anomaly_c for s in seasons], axis=0).astype(np.float32)

    return dataclasses.replace(
        seasons[0], temperature_c=t, precipitation_mm=precip, seasonality_c=seasonality,
        wind=wind, moisture=moisture, sst_anomaly_c=anom,
        temperature_season=stack_t, precipitation_season=stack_p,
        declination_deg=np.array(decls, dtype=np.float32))
