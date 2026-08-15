"""Stage 7: biome classification from temperature and precipitation.

The Whittaker diagram: mean annual temperature against annual precipitation, partitioned
into biomes. It is the simple approach -- Koppen-Geiger is the rigorous one and needs
seasonality and a monthly cycle, neither of which this pipeline produces yet.

This stage is nearly free, because Stage 3 already computed both axes. That is the whole
reason it is worth doing before the remaining erosion stages: it turns the climate fields
from numbers into something you can look at and immediately judge.

Two things the classification needs beyond raw Whittaker:

  * Permanent ice, which the diagram has no cell for. High-altitude ice is what makes
    mountains read as mountains, and its elevation varies enormously with latitude -- the
    snowline sits near 4900 m at the equator and reaches sea level at the poles.
  * A guard for the fact that Whittaker's axes are not independent in practice. Very cold
    air cannot hold much moisture, so the cold-and-wet corner of the diagram is nearly
    empty in reality and any terrain that lands there is telling you the climate model has
    gone wrong somewhere.
"""

from __future__ import annotations

import numpy as np

OCEAN = 0
ICE = 1
TUNDRA = 2
BOREAL_FOREST = 3
TEMPERATE_GRASSLAND = 4
WOODLAND_SHRUBLAND = 5
TEMPERATE_SEASONAL_FOREST = 6
TEMPERATE_RAINFOREST = 7
SUBTROPICAL_DESERT = 8
SAVANNA = 9
TROPICAL_SEASONAL_FOREST = 10
TROPICAL_RAINFOREST = 11
# Arid ground outside the tropics. Without these, aridity can only be expressed where it is
# also hot, so the desert class becomes a latitude band by construction and every dry
# temperate region collapses into grassland -- which came out at 23% of land against
# Earth's 9%. The Gobi sits at 43N and Patagonia at 45S; neither is a warm desert.
TEMPERATE_DESERT = 12
COLD_DESERT = 13
# Fresh water. Not a Whittaker class -- it is a surface type, and it wins over whatever
# climate would otherwise have put here, the same way OCEAN does. Salt and fresh water need
# to be told apart by anything reading the map, and elevation cannot do it: an inland basin
# floor sits below sea level just as the abyssal plain does.
LAKE = 14

NAMES = {
    OCEAN: "ocean", ICE: "ice", TUNDRA: "tundra", BOREAL_FOREST: "boreal forest",
    TEMPERATE_GRASSLAND: "temperate grassland", WOODLAND_SHRUBLAND: "woodland/shrubland",
    TEMPERATE_SEASONAL_FOREST: "temperate seasonal forest",
    TEMPERATE_RAINFOREST: "temperate rainforest", SUBTROPICAL_DESERT: "subtropical desert",
    SAVANNA: "savanna", TROPICAL_SEASONAL_FOREST: "tropical seasonal forest",
    TROPICAL_RAINFOREST: "tropical rainforest",
    TEMPERATE_DESERT: "temperate desert", COLD_DESERT: "cold desert",
    LAKE: "lake",
}

# Map colours, roughly the conventional ones so the render reads without a legend.
COLOURS = {
    OCEAN: (0.09, 0.20, 0.42), ICE: (0.94, 0.96, 0.98),
    TUNDRA: (0.60, 0.62, 0.55), BOREAL_FOREST: (0.20, 0.38, 0.29),
    TEMPERATE_GRASSLAND: (0.79, 0.76, 0.45),
    WOODLAND_SHRUBLAND: (0.65, 0.62, 0.32),
    TEMPERATE_SEASONAL_FOREST: (0.35, 0.55, 0.26),
    TEMPERATE_RAINFOREST: (0.16, 0.45, 0.33),
    SUBTROPICAL_DESERT: (0.85, 0.72, 0.48),
    SAVANNA: (0.72, 0.70, 0.33),
    TROPICAL_SEASONAL_FOREST: (0.38, 0.62, 0.24),
    TROPICAL_RAINFOREST: (0.13, 0.42, 0.16),
    TEMPERATE_DESERT: (0.76, 0.64, 0.44),
    COLD_DESERT: (0.68, 0.65, 0.58),
    # Deliberately lighter and cooler than OCEAN's navy: the two are adjacent on the map and
    # the whole point of the class is that you can tell them apart at a glance.
    LAKE: (0.29, 0.53, 0.75),
}


def snowline_m(lat_deg, equatorial_m=4900.0, polar_m=0.0):
    """Elevation of permanent snow, by latitude.

    Roughly 4900 m at the equator falling to sea level near the poles. Not linear in
    latitude -- the maximum actually sits near 25 degrees over the subtropical deserts,
    because snowline tracks dryness as much as temperature, but cos(lat) is close enough
    for a first pass and monotonic, which keeps the ice caps where they belong.

    THESE ARE EARTH'S METRES ON A WORLD THAT TRADED ABSOLUTE METRES AWAY, so the elevation
    branch of the ice test is currently dead. Production peaks are 2,428 m, which only clears
    this snowline poleward of 49.8 degrees; on an n=128 fixture peaking at 347 m it takes 79
    degrees, and the measured count of land cells above their own snowline is exactly ZERO.
    Every ICE cell the classifier assigns comes from the -12 C floor in `classify` instead,
    which is 11.3% of land on that fixture. High-altitude ice -- the thing this function
    exists to add, and what makes mountains read as mountains -- does not occur.

    Left as Earth's figures deliberately. Scaling the snowline down to this world's relief
    would paint ice on ground that is 300 m above the sea and 10 C above freezing, which is
    worse than having none: the temperature field is the honest authority on where ice
    belongs, and it is already consulted. This is the same accounting gap recorded for the
    lapse rate in climate.py -- elevation is an EMERGENT field, and the rescale's three-class
    parameter table has no row for those.
    """
    # Clip before the fractional power: cos comes back very slightly negative at the poles
    # from rounding, and a negative base to a fractional exponent is NaN, which would then
    # compare false against every elevation and silently leave the ice caps unmarked.
    c = np.clip(np.cos(np.radians(lat_deg)), 0.0, 1.0)
    return polar_m + (equatorial_m - polar_m) * c ** 1.6


def classify(temperature_c, precipitation_mm, land_mask, elevation_m=None, lat_deg=None,
             lake_mask=None):
    """Whittaker classification, plus permanent ice above the snowline and open water."""
    t = np.asarray(temperature_c, dtype=np.float64)
    p = np.asarray(precipitation_mm, dtype=np.float64)
    out = np.full(t.shape, OCEAN, dtype=np.uint8)

    warm = t >= 20.0
    temperate = (t >= 5.0) & (t < 20.0)
    cold = (t >= -5.0) & (t < 5.0)
    frigid = t < -5.0

    # Tropical
    out[warm & (p < 250)] = SUBTROPICAL_DESERT
    out[warm & (p >= 250) & (p < 1500)] = SAVANNA
    out[warm & (p >= 1500) & (p < 2500)] = TROPICAL_SEASONAL_FOREST
    out[warm & (p >= 2500)] = TROPICAL_RAINFOREST

    # Temperate. The arid cut-offs sit below textbook Whittaker (which puts temperate
    # desert nearer 250 mm) because this pipeline's precipitation carries a heavy dry tail
    # -- land p25 is about 15 mm against Earth's ~250 -- so absolute thresholds
    # over-assign aridity. Note that lowering them barely helps: temperate desert only
    # falls from 20% of land to 16% between cut-offs of 200 mm and 80 mm. The real cause
    # is the Stage 3 precipitation skew, not the classification.
    out[temperate & (p < 120)] = TEMPERATE_DESERT
    out[temperate & (p >= 120) & (p < 400)] = TEMPERATE_GRASSLAND
    out[temperate & (p >= 400) & (p < 900)] = WOODLAND_SHRUBLAND
    out[temperate & (p >= 900) & (p < 1800)] = TEMPERATE_SEASONAL_FOREST
    out[temperate & (p >= 1800)] = TEMPERATE_RAINFOREST

    # Cold. Below this there is not enough water for closed-canopy forest.
    out[cold & (p < 90)] = COLD_DESERT
    out[cold & (p >= 90)] = BOREAL_FOREST

    out[frigid] = TUNDRA

    # Permanent ice: hard temperature floor, or anything above the local snowline.
    ice = t < -12.0
    if elevation_m is not None and lat_deg is not None:
        ice |= np.asarray(elevation_m) > snowline_m(np.asarray(lat_deg))
    out[ice] = ICE

    out[~np.asarray(land_mask)] = OCEAN
    # Lakes last. They sit on the land side of the mask, so they must be painted after the
    # ocean sweep or that line would have to know about them. Ice wins nothing here: a
    # frozen lake is still a lake, and the surface type is what this class exists to record.
    if lake_mask is not None:
        out[np.asarray(lake_mask)] = LAKE
    return out


def summary(biome, area, land_mask):
    """Share of land area by biome, largest first."""
    land = np.asarray(land_mask)
    total = float(area[land].sum())
    rows = []
    for code, name in NAMES.items():
        if code == OCEAN:
            continue
        m = land & (biome == code)
        if not m.any():
            continue
        rows.append((name, float(area[m].sum()) / total))
    return sorted(rows, key=lambda r: -r[1])
