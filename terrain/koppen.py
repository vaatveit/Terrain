"""Stage 7b: Koppen-Geiger climate classification, alongside Whittaker.

Whittaker classifies on the annual MEAN temperature and rainfall. That is the whole of its
input, and it is why a Mediterranean summer-dry climate and a monsoon summer-wet one at the
same annual totals are the same biome to it -- they are not the same place at all. Koppen
asks when the rain falls and how cold the winter gets, which is why the survey named it the
rigorous alternative, and which is why it had to wait for Stage 3 to grow a seasonal cycle.

IT DOES NOT REPLACE WHITTAKER, deliberately. `field_biome` stays the renderer's colour
source and the classification the tuning history is written against -- every biome share in
terrain-algorithms.md is a Whittaker share, and switching would silently invalidate the lot.
The two are exported side by side and disagree in informative places.

TWELVE MONTHS FROM FOUR SAMPLES. Koppen's rules read the coldest month, the warmest month,
the driest month and the summer/winter rainfall split, and Stage 3 solves four points in the
year rather than twelve. Four evenly spaced samples determine a two-harmonic Fourier series
EXACTLY -- the mean, the annual cycle, and the semiannual one -- so reconstructing the
months is an interpolation with no free parameters rather than an assumed sine. That is the
real argument for solving four seasons instead of two, and it is why the sample dates must
stay evenly spaced (see climate.SEASON_PHASE).

What the reconstruction cannot recover is the second semiannual coefficient, which needs a
fifth sample. In practice that term is small against the annual cycle everywhere except the
double-peaked equatorial rainfall regime, so Af and Am are the classes to distrust first.
"""

from __future__ import annotations

import numpy as np

# Main groups, then the sub-classes actually distinguished here.
AF = 0    # tropical rainforest
AM = 1    # tropical monsoon
AW = 2    # tropical savanna, winter dry
BWH = 3   # hot desert
BWK = 4   # cold desert
BSH = 5   # hot steppe
BSK = 6   # cold steppe
CSA = 7   # Mediterranean, hot summer
CFA = 8   # humid subtropical
CFB = 9   # oceanic
DFA = 10  # humid continental
DFB = 11  # continental, cool summer
DFC = 12  # subarctic
ET = 13   # tundra
EF = 14   # ice cap

NAMES = {
    AF: "Af tropical rainforest", AM: "Am tropical monsoon", AW: "Aw tropical savanna",
    BWH: "BWh hot desert", BWK: "BWk cold desert",
    BSH: "BSh hot steppe", BSK: "BSk cold steppe",
    CSA: "Csa mediterranean", CFA: "Cfa humid subtropical", CFB: "Cfb oceanic",
    DFA: "Dfa humid continental", DFB: "Dfb continental", DFC: "Dfc subarctic",
    ET: "ET tundra", EF: "EF ice cap",
}

# Roughly conventional Koppen map colours, so the two classifications can be read side by
# side without a legend lookup.
COLOURS = {
    AF: (0.00, 0.00, 0.78), AM: (0.00, 0.47, 1.00), AW: (0.28, 0.68, 1.00),
    BWH: (1.00, 0.00, 0.00), BWK: (1.00, 0.60, 0.60),
    BSH: (0.96, 0.65, 0.00), BSK: (1.00, 0.86, 0.40),
    CSA: (1.00, 1.00, 0.00), CFA: (0.78, 1.00, 0.39), CFB: (0.39, 1.00, 0.39),
    DFA: (0.34, 1.00, 1.00), DFB: (0.24, 0.78, 0.78), DFC: (0.24, 0.55, 0.78),
    ET: (0.70, 0.70, 0.70), EF: (0.40, 0.40, 0.40),
}


def monthly_from_seasons(season_stack, months=12):
    """Reconstruct a monthly cycle from four evenly spaced samples through the year.

    Exact for the mean, the annual harmonic and the cosine of the semiannual one, which is
    everything four points can carry. Returns (months, N).
    """
    s = np.asarray(season_stack, dtype=np.float64)
    n_s = s.shape[0]
    k = np.arange(n_s)
    a0 = s.mean(axis=0)
    a1 = 2.0 / n_s * np.tensordot(np.cos(2 * np.pi * k / n_s), s, axes=1)
    b1 = 2.0 / n_s * np.tensordot(np.sin(2 * np.pi * k / n_s), s, axes=1)
    a2 = 1.0 / n_s * np.tensordot(np.cos(np.pi * k), s, axes=1)

    t = 2 * np.pi * np.arange(months) / months
    return (a0[None, :]
            + np.cos(t)[:, None] * a1[None, :]
            + np.sin(t)[:, None] * b1[None, :]
            + np.cos(2 * t)[:, None] * a2[None, :])


def monthly_precipitation(precipitation_season, annual_precip_mm=None):
    """Twelve monthly totals in mm, summing to the annual total. Returns (12, N).

    `precipitation_season` carries a per-season RATE in the same units the annual field
    uses, so a season's share of the year is its value divided by the sum over seasons.
    Shared by `classify` and the export, so the months that ship are the months Koppen read.
    """
    p_m = np.maximum(monthly_from_seasons(precipitation_season), 0.0)

    # Scale the reconstructed months so they sum to the annual total the pipeline reports.
    annual = (np.asarray(annual_precip_mm, dtype=np.float64) if annual_precip_mm is not None
              else np.asarray(precipitation_season, dtype=np.float64).mean(axis=0))
    total = p_m.sum(axis=0)
    return p_m * np.divide(annual, total, out=np.zeros_like(total), where=total > 0)[None, :]


def classify(temperature_season, precipitation_season, land_mask, lat_deg,
             annual_precip_mm=None):
    """Koppen-Geiger from Stage 3's seasonal stacks. Ocean is left as -1."""
    t_m = monthly_from_seasons(temperature_season)
    p_m = monthly_precipitation(precipitation_season, annual_precip_mm)
    annual = (np.asarray(annual_precip_mm, dtype=np.float64) if annual_precip_mm is not None
              else np.asarray(precipitation_season, dtype=np.float64).mean(axis=0))

    t_cold, t_hot = t_m.min(axis=0), t_m.max(axis=0)
    t_mean = t_m.mean(axis=0)
    months_over_10 = (t_m >= 10.0).sum(axis=0)
    p_dry = p_m.min(axis=0)

    # Summer is the warmer half of the year at that cell, found from the data rather than
    # from the hemisphere -- which is the same thing here but does not assume the seasonal
    # cycle is in phase with the calendar, and costs nothing.
    order = np.argsort(t_m, axis=0)
    warm_half = order[t_m.shape[0] // 2:, :]
    idx = np.arange(t_m.shape[1])
    p_summer = p_m[warm_half, idx].sum(axis=0)
    p_winter = annual - p_summer
    p_dry_summer = np.where(p_m[warm_half, idx].size, p_m[warm_half, idx].min(axis=0), 0.0)
    cold_half = order[:t_m.shape[0] // 2, :]
    p_dry_winter = p_m[cold_half, idx].min(axis=0)

    out = np.full(t_mean.shape, -1, dtype=np.int8)
    land = np.asarray(land_mask, dtype=bool)

    # B first: aridity overrides everything, which is why a hot desert is not a savanna.
    # The threshold is 20*T plus a term for WHEN the rain falls -- the same total goes much
    # further in a wet winter than in a wet summer, because less of it evaporates.
    share_summer = np.divide(p_summer, annual, out=np.full_like(annual, 0.5), where=annual > 0)
    offset = np.where(share_summer >= 0.7, 280.0, np.where(share_summer <= 0.3, 0.0, 140.0))
    arid = 20.0 * t_mean + offset

    is_b = annual < arid
    is_w = annual < 0.5 * arid
    hot = t_mean >= 18.0

    out[land & is_b & is_w & hot] = BWH
    out[land & is_b & is_w & ~hot] = BWK
    out[land & is_b & ~is_w & hot] = BSH
    out[land & is_b & ~is_w & ~hot] = BSK

    rest = land & ~is_b

    # E: no month warm enough for trees.
    e = rest & (t_hot < 10.0)
    out[e & (t_hot < 0.0)] = EF
    out[e & (t_hot >= 0.0)] = ET

    # A: no winter at all.
    a = rest & ~e & (t_cold >= 18.0)
    af = a & (p_dry >= 60.0)
    # Am against Aw: a monsoon has a dry month but so much rain in the wet season that the
    # forest survives it. Aw simply runs out.
    am = a & ~af & (p_dry >= 100.0 - annual / 25.0)
    out[af] = AF
    out[am] = AM
    out[a & ~af & ~am] = AW

    # C and D split on how cold the winter gets; the third letter is the summer.
    cd = rest & ~e & ~a
    c = cd & (t_cold >= -3.0)
    d = cd & (t_cold < -3.0)

    dry_summer = (p_dry_summer < 40.0) & (p_dry_summer < p_dry_winter / 3.0)
    out[c & dry_summer] = CSA
    out[c & ~dry_summer & (t_hot >= 22.0)] = CFA
    out[c & ~dry_summer & (t_hot < 22.0)] = CFB

    out[d & (t_hot >= 22.0)] = DFA
    out[d & (t_hot < 22.0) & (months_over_10 >= 4)] = DFB
    out[d & (t_hot < 22.0) & (months_over_10 < 4)] = DFC

    return out


def summary(code, area, land_mask):
    """Share of land area by Koppen class, largest first."""
    land = np.asarray(land_mask)
    total = float(area[land].sum())
    rows = []
    for c, name in NAMES.items():
        m = land & (code == c)
        if m.any():
            rows.append((name, float(area[m].sum() / total)))
    return sorted(rows, key=lambda r: -r[1])
