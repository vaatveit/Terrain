"""Terrain realism metrics, computed over a cell graph.

The point of the testbed is comparing generation strategies, so these need to give an
objective answer to "is this more realistic" rather than a screenshot. Each returns a
scalar or curve computable on both generated terrain and real DEM tiles.

Everything here is scale-free by design or explicitly area-weighted, so a 30 m SRTM window
and a 9.8 km global mesh are comparable on exponents even though their absolute drainage
densities differ by orders of magnitude. That is exactly why the scaling laws are the
right diagnostics.

Not implemented yet: coastline fractal dimension (needs Stage 1 coastlines to exist; must
use a great-circle ruler walk, not box-counting, which has no meaning on a sphere) and the
spherical-harmonic power spectrum (needs a HEALPix resample; the weakest metric on the
list, and slope-area concavity already catches the failure mode it targets).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit

from terrain import flow


@njit(cache=True)
def _propagate_upstream_flag(order, rec, flagged, out):
    """Mark every cell whose upstream catchment contains a flagged cell.

    Runs upstream-to-downstream (reverse topological order), the same direction as flow
    accumulation, pushing each cell's flag into its receiver.
    """
    n = order.shape[0]
    for i in range(n):
        out[i] = flagged[i]
    for t in range(n - 1, -1, -1):
        c = order[t]
        r = rec[c]
        if r != c and out[c]:
            out[r] = True
    return out


@dataclasses.dataclass
class Regression:
    exponent: float
    intercept: float
    r2: float
    n_points: int

    def __str__(self):
        return f"{self.exponent:+.3f} (R^2 {self.r2:.3f}, n={self.n_points:,})"


def _loglog_fit(x, y) -> Regression:
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log10(x[ok]), np.log10(y[ok])
    if lx.size < 3:
        return Regression(np.nan, np.nan, np.nan, int(lx.size))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return Regression(float(slope), float(intercept), r2, int(lx.size))


def _log_bin_median(x, y, bins=40):
    """Median y in log-spaced x bins. Standard for slope-area, which is very noisy."""
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if x.size < 10:
        return np.empty(0), np.empty(0)
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), bins + 1)
    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    cx, cy = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            cx.append(np.sqrt(edges[b] * edges[b + 1]))
            cy.append(np.median(y[m]))
    return np.asarray(cx), np.asarray(cy)


# --------------------------------------------------------------------------------------


def channel_mask(acc, threshold_km2):
    """Cells carrying a channel, by critical drainage area."""
    return acc >= np.float32(threshold_km2)


def slope_area(fr: flow.FlowResult, is_channel, bins=40) -> tuple[Regression, np.ndarray, np.ndarray]:
    """Slope-area relation: S ~ A^-theta in channels. Concavity theta should be 0.4-0.6.

    The cleanest single diagnostic that an erosion model is behaving correctly, and the one
    that catches noise-only terrain: it looks fine in a screenshot and fails here.

    Uses the flow-path gradient, not the cell-to-cell one -- see flow.flow_path_slope for
    why single-cell slope is unusable on a quantised real DEM.

    Cells lifted by the depression fill are also excluded: they sit on an artificial +eps
    ramp and collect in valley bottoms, so they land at the high-A end of the regression.
    """
    m = is_channel & (fr.ch_slope > 0) & (fr.step > 0) & ~fr.raised
    cx, cy = _log_bin_median(fr.acc[m], fr.ch_slope[m], bins)
    reg = _loglog_fit(cx, cy)
    return reg, cx, cy


@njit(cache=True)
def _propagate_root(order, rec, root):
    """Label every cell with the outlet it ultimately drains to."""
    n = order.shape[0]
    for t in range(n):
        c = order[t]
        r = rec[c]
        root[c] = c if r == c else root[r]
    return root


def basin_labels(fr: flow.FlowResult) -> np.ndarray:
    root = np.empty(fr.rec.shape[0], dtype=np.int32)
    return _propagate_root(fr.order, fr.rec, root)


def slope_area_by_basin(fr: flow.FlowResult, is_channel, area, min_channel_cells=400,
                        max_basins=40) -> tuple[np.ndarray, np.ndarray]:
    """Concavity measured separately within each large basin.

    Measuring theta over a whole planet at once is not the same statistic as the published
    0.4-0.6, and not comparable to a DEM window either. Steady-state stream power gives
    S = (U/K)^(1/n) * A^(-m/n) with U the LOCAL uplift, so theta = m/n only holds where
    uplift is roughly uniform. Globally it is not: after evolution, high-uplift regions
    have become headwater divides (small A, steep) and low-uplift regions lowlands (large
    A, gentle), so U ends up anti-correlated with A and that anti-correlation lands
    directly in the regression. Measured, this pushed theta from 0.45 to 0.97 on an
    otherwise identical run -- the model was right and the measurement was wrong.

    A single basin spans a much narrower range of uplift, which is why published values are
    per-basin, and why the SRTM references here are single 63 km windows.

    Returns (theta per basin, basin drainage area km^2), largest basins first.
    """
    roots = basin_labels(fr)
    ch_roots = roots[is_channel]
    if ch_roots.size == 0:
        return np.empty(0), np.empty(0)

    uniq, counts = np.unique(ch_roots, return_counts=True)
    keep = uniq[counts >= min_channel_cells]
    if keep.size == 0:
        return np.empty(0), np.empty(0)
    keep = keep[np.argsort(fr.acc[keep])[::-1]][:max_basins]

    thetas, areas = [], []
    for r in keep:
        m = is_channel & (roots == r) & (fr.ch_slope > 0) & (fr.step > 0) & ~fr.raised
        if m.sum() < min_channel_cells:
            continue
        cx, cy = _log_bin_median(fr.acc[m], fr.ch_slope[m], bins=25)
        reg = _loglog_fit(cx, cy)
        if np.isfinite(reg.exponent) and reg.r2 > 0.5:
            thetas.append(-reg.exponent)
            areas.append(float(fr.acc[r]))
    return np.asarray(thetas), np.asarray(areas)


def hacks_law(fr: flow.FlowResult, is_channel, valid, min_area_km2, max_points=200_000,
              seed=0) -> Regression:
    """Hack's law: main-channel length L ~ A^h, h ~ 0.57. Pure noise fails this badly.

    Uses nested sub-basins -- every channel cell is treated as a basin outlet, which is the
    usual approach from a DEM and gives good statistics. `valid` excludes basins clipped by
    the window border, whose areas and lengths would both be wrong.
    """
    m = is_channel & valid & (fr.acc >= np.float32(min_area_km2)) & (fr.up_len > 0)
    idx = np.flatnonzero(m)
    if idx.size > max_points:
        idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
    return _loglog_fit(fr.acc[idx], fr.up_len[idx])


def horton_ratios(fr: flow.FlowResult, is_channel, valid) -> dict:
    """Horton's laws of stream numbers, lengths and areas, from Strahler ordering.

    Real networks cluster tightly: Rb ~ 3-5, Rl ~ 1.5-3.5, Ra ~ 3-6.
    """
    n = fr.rec.shape[0]
    so = np.empty(n, dtype=np.int32)
    flow.strahler_order(fr.order, fr.don_ptr, fr.don, is_channel, so)

    use = is_channel & valid & (so > 0)
    # A stream ends where its order changes going downstream (or it leaves the network).
    downstream_order = so[fr.rec]
    is_stream_end = use & ((downstream_order != so) | (fr.rec == np.arange(n)))

    max_order = int(so[use].max()) if use.any() else 0
    counts, lengths, areas = {}, {}, {}
    for w in range(1, max_order + 1):
        ends = is_stream_end & (so == w)
        n_streams = int(ends.sum())
        if n_streams == 0:
            continue
        counts[w] = n_streams
        # Total length of order-w cells / number of order-w streams = mean stream length.
        lengths[w] = float(fr.step[use & (so == w)].sum() / n_streams)
        areas[w] = float(fr.acc[ends].mean())

    def _ratio(d, invert=False):
        """Horton's laws are geometric in stream order, so fit log(value) vs order.

        A mean of successive ratios is what the textbook definition suggests, but it is
        badly unstable here: the highest order typically has only one or two streams, and
        that single noisy ratio then dominates the average (measured Rb of 9.2 where the
        regression gives a sane value). The log-linear fit weights all orders properly.
        """
        ws = sorted(w for w in d if d[w] > 0)
        if len(ws) < 2:
            return np.nan
        vals = np.array([d[w] for w in ws], dtype=float)
        if len(ws) >= 3:
            slope = np.polyfit(np.array(ws, dtype=float), np.log10(vals), 1)[0]
            return float(10 ** (slope if invert else -slope))
        pair = [
            (d[b] / d[a]) if invert else (d[a] / d[b])
            for a, b in zip(ws, ws[1:]) if b == a + 1
        ]
        return float(np.mean(pair)) if pair else np.nan

    return {
        "max_order": max_order,
        "stream_counts": counts,
        "bifurcation_ratio": _ratio(counts),
        "length_ratio": _ratio(lengths, invert=True),
        "area_ratio": _ratio(areas, invert=True),
    }


def hypsometry(z, area, valid=None, bins=100) -> tuple[np.ndarray, np.ndarray, float]:
    """Area-weighted elevation distribution. The integral separates young from mature.

    Computed over the whole domain, not per basin. Published hypsometric integrals are
    per-basin and normalised to that basin's own relief, so these values are a relative
    comparator between terrains rather than something to check against a literature range
    -- a window holding one deep valley and a few peaks reads low regardless of maturity.
    """
    if valid is not None:
        z, area = z[valid], area[valid]
    zmin, zmax = float(z.min()), float(z.max())
    if zmax <= zmin:
        return np.empty(0), np.empty(0), np.nan
    rel_z = (z - zmin) / (zmax - zmin)

    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(rel_z, edges) - 1, 0, bins - 1)
    per_bin = np.bincount(idx, weights=area.astype(np.float64), minlength=bins)
    total = per_bin.sum()
    # Fraction of area at or above each elevation.
    above = (per_bin[::-1].cumsum())[::-1] / total
    centres = 0.5 * (edges[:-1] + edges[1:])
    integral = float(np.trapezoid(above, centres))
    return centres, above, integral


def slope_stats(fr: flow.FlowResult, valid=None) -> dict:
    """Slope distribution. Pure fBm is visibly too uniform here."""
    s = fr.slope if valid is None else fr.slope[valid]
    s = s[np.isfinite(s) & (s >= 0)]
    deg = np.degrees(np.arctan(s))
    return {
        "mean_deg": float(deg.mean()),
        "median_deg": float(np.median(deg)),
        "p90_deg": float(np.percentile(deg, 90)),
        "p99_deg": float(np.percentile(deg, 99)),
        "frac_over_30deg": float((deg > 30).mean()),
    }


def drainage_density(fr: flow.FlowResult, is_channel, area, valid=None) -> float:
    """Total channel length per unit area, km/km^2. Should vary with climate."""
    m = is_channel if valid is None else (is_channel & valid)
    total_area = float(area.sum() if valid is None else area[valid].sum())
    if total_area <= 0:
        return np.nan
    return float(fr.step[m].sum() / total_area)


# --------------------------------------------------------------------------------------


def interior_mask(fr: flow.FlowResult, is_border, nbr=None) -> np.ndarray:
    """Cells whose entire upstream catchment lies inside the window.

    A basin whose catchment crosses the window edge has a truncated area and a truncated
    main-channel length, and including it corrupts both Hack's law and the Horton area ratio.

    Note the direction. Testing whether the *outlet* is on the border excludes everything,
    because border cells are the sinks -- in a 31 km window every drop of water leaves
    through an edge. What matters is whether anything *upstream* of a cell is
    edge-contaminated, which leaves interior sub-basins usable. Those are complete by
    definition, since a catchment lies entirely upstream of its outlet.
    """
    flagged = np.ascontiguousarray(is_border).copy()
    if nbr is not None:
        # Cells adjacent to the border may be fed from outside the window.
        for s in range(nbr.shape[1]):
            j = nbr[:, s]
            ok = j >= 0
            flagged[ok] |= is_border[j[ok]]

    tainted = np.zeros(fr.rec.shape[0], dtype=np.bool_)
    _propagate_upstream_flag(fr.order, fr.rec, flagged, tainted)
    return ~tainted


def report(z, nbr, arc_dist, area, is_sink, is_border=None, channel_threshold_km2=1.0,
           hack_min_area_km2=None, label="") -> dict:
    """Run the full metric suite. Returns a dict of scalars plus the curves."""
    fr = flow.route(z, nbr, arc_dist, area, is_sink)
    is_channel = channel_mask(fr.acc, channel_threshold_km2)

    if is_border is None:
        valid = np.ones(z.shape[0], dtype=bool)
    else:
        valid = interior_mask(fr, is_border, nbr)

    if hack_min_area_km2 is None:
        hack_min_area_km2 = channel_threshold_km2 * 10

    sa_reg, sa_x, sa_y = slope_area(fr, is_channel)
    hack = hacks_law(fr, is_channel, valid, hack_min_area_km2)
    horton = horton_ratios(fr, is_channel, valid)
    hyp_x, hyp_y, hyp_int = hypsometry(fr.filled, area)
    basin_theta, basin_area = slope_area_by_basin(fr, is_channel, area)

    return {
        "label": label,
        "cells": int(z.shape[0]),
        "basin_theta_median": float(np.median(basin_theta)) if basin_theta.size else np.nan,
        "basin_theta_p25": float(np.percentile(basin_theta, 25)) if basin_theta.size else np.nan,
        "basin_theta_p75": float(np.percentile(basin_theta, 75)) if basin_theta.size else np.nan,
        "basin_count": int(basin_theta.size),
        "_basin_theta": basin_theta,
        "_basin_area": basin_area,
        "interior_fraction": float(valid.mean()),
        "channel_cells": int(is_channel.sum()),
        "relief_m": float(fr.filled.max() - fr.filled.min()),
        "concavity_theta": -sa_reg.exponent,
        "concavity_r2": sa_reg.r2,
        "hack_h": hack.exponent,
        "hack_r2": hack.r2,
        "hack_n": hack.n_points,
        "hypsometric_integral": hyp_int,
        "drainage_density": drainage_density(fr, is_channel, area),
        **{f"slope_{k}": v for k, v in slope_stats(fr).items()},
        **{f"horton_{k}": v for k, v in horton.items() if k != "stream_counts"},
        "horton_stream_counts": horton["stream_counts"],
        "_curves": {"slope_area": (sa_x, sa_y), "hypsometry": (hyp_x, hyp_y)},
        "_flow": fr,
    }
