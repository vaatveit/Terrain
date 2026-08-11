"""Flow routing over a cell graph. Grid-agnostic: works on the Goldberg tiling and on a
lat/lon DEM raster alike, so the same metrics run against generated and real terrain.

The graph is three flat arrays:
    neighbours  (N, k) int32   -1 padded (pentagons on the hex grid, borders on a raster)
    arc_dist    (N, k) float32 great-circle centre-to-centre distance, km
    area        (N,)   float32 real cell area, km^2

`k` is 6 on the Goldberg grid and 8 for D8 on a raster; kernels read it from the array
shape, so nothing here is grid-specific.

Everything is @njit(cache=True) over preallocated arrays. Do not pass a Grid object in --
that would mean @jitclass, which is fragile and slow to compile.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit, prange

EPS = np.float32(1e-4)  # priority-flood fill increment, metres


# --------------------------------------------------------------------------------------
# Binary min-heap over preallocated arrays.
# Numba does support heapq, but over typed lists, with per-operation overhead that matters
# when priority-flood pushes several million times.
# --------------------------------------------------------------------------------------


@njit(cache=True, inline="always")
def heap_push(keys, idxs, size, key, val):
    c = size
    keys[c] = key
    idxs[c] = val
    while c > 0:
        p = (c - 1) >> 1
        if keys[p] <= keys[c]:
            break
        keys[p], keys[c] = keys[c], keys[p]
        idxs[p], idxs[c] = idxs[c], idxs[p]
        c = p
    return size + 1


@njit(cache=True, inline="always")
def heap_pop(keys, idxs, size):
    top_key = keys[0]
    top_val = idxs[0]
    size -= 1
    keys[0] = keys[size]
    idxs[0] = idxs[size]
    p = 0
    while True:
        l = 2 * p + 1
        if l >= size:
            break
        r = l + 1
        m = l if (r >= size or keys[l] <= keys[r]) else r
        if keys[p] <= keys[m]:
            break
        keys[p], keys[m] = keys[m], keys[p]
        idxs[p], idxs[m] = idxs[m], idxs[p]
        p = m
    return top_key, top_val, size


# --------------------------------------------------------------------------------------


@njit(cache=True)
def priority_flood(z, nbr, is_sink, filled, closed, keys, idxs):
    """Fill depressions inward from the sinks (Barnes, Lehman & Mulla 2014, +eps variant).

    `is_sink` is where water leaves the domain: ocean cells on a globe, border cells on a
    DEM window. A globe has no border, which is the whole point -- the ocean is the outlet.
    """
    n = z.shape[0]
    k = nbr.shape[1]
    for i in range(n):
        filled[i] = z[i]
        closed[i] = False

    size = 0
    for i in range(n):
        if is_sink[i]:
            closed[i] = True
            size = heap_push(keys, idxs, size, z[i], i)

    while size > 0:
        key, c, size = heap_pop(keys, idxs, size)
        for s in range(k):
            j = nbr[c, s]
            if j < 0 or closed[j]:
                continue
            closed[j] = True
            if filled[j] <= key:
                filled[j] = key + EPS
            size = heap_push(keys, idxs, size, filled[j], j)
    return filled


@njit(cache=True, parallel=True)
def compute_receivers(z, nbr, arc_dist, is_sink, rec):
    """Steepest descent. Sinks point at themselves and terminate the tree."""
    n = z.shape[0]
    k = nbr.shape[1]
    for i in prange(n):
        if is_sink[i]:
            rec[i] = i
            continue
        best = i
        best_slope = np.float32(0.0)
        for s in range(k):
            j = nbr[i, s]
            if j < 0:
                continue
            slope = (z[i] - z[j]) / arc_dist[i, s]
            if slope > best_slope:
                best_slope = slope
                best = j
        rec[i] = best
    return rec


@njit(cache=True)
def build_order(rec, ndon, don_ptr, don, order):
    """Topological order (receivers strictly before donors) plus the donor CSR lists."""
    n = rec.shape[0]
    for i in range(n):
        ndon[i] = 0
    for i in range(n):
        r = rec[i]
        if r != i:
            ndon[r] += 1

    don_ptr[0] = 0
    for i in range(n):
        don_ptr[i + 1] = don_ptr[i] + ndon[i]

    fill = don_ptr[:n].copy()
    for i in range(n):
        r = rec[i]
        if r != i:
            don[fill[r]] = i
            fill[r] += 1

    ns = 0
    for i in range(n):
        if rec[i] == i:
            order[ns] = i
            ns += 1
    k = 0
    while k < ns:
        c = order[k]
        k += 1
        for p in range(don_ptr[c], don_ptr[c + 1]):
            order[ns] = don[p]
            ns += 1
    return ns


@njit(cache=True)
def flow_accumulate(order, rec, area, acc):
    """Upstream drainage area. Accumulates real cell AREAS -- never cell counts.

    Cell area varies ~0.9% across the Goldberg grid and by cos(lat) on a lat/lon raster;
    counting cells instead biases Hack's law and the slope-area relation, which are the
    exact diagnostics this exists to produce.
    """
    n = order.shape[0]
    for i in range(n):
        acc[i] = area[i]
    for k in range(n - 1, -1, -1):
        c = order[k]
        r = rec[c]
        if r != c:
            acc[r] += acc[c]
    return acc


@njit(cache=True)
def longest_upstream_path(order, rec, don_ptr, don, arc_dist, nbr, up_len):
    """Length of the longest flow path upstream of each cell, km. The `L` in Hack's law."""
    n = order.shape[0]
    k = nbr.shape[1]
    for i in range(n):
        up_len[i] = np.float32(0.0)
    for t in range(n - 1, -1, -1):
        c = order[t]
        best = np.float32(0.0)
        for p in range(don_ptr[c], don_ptr[c + 1]):
            d = don[p]
            step = np.float32(0.0)
            for s in range(k):
                if nbr[d, s] == c:
                    step = arc_dist[d, s]
                    break
            cand = up_len[d] + step
            if cand > best:
                best = cand
        up_len[c] = best
    return up_len


@njit(cache=True)
def strahler_order(order, don_ptr, don, is_channel, out):
    """Strahler stream order. Two equal-order tributaries promote; otherwise inherit."""
    n = order.shape[0]
    for i in range(n):
        out[i] = 0
    for t in range(n - 1, -1, -1):
        c = order[t]
        if not is_channel[c]:
            continue
        max_o = 0
        count = 0
        for p in range(don_ptr[c], don_ptr[c + 1]):
            d = don[p]
            if not is_channel[d]:
                continue
            o = out[d]
            if o > max_o:
                max_o = o
                count = 1
            elif o == max_o and o > 0:
                count += 1
        if max_o == 0:
            out[c] = 1
        elif count >= 2:
            out[c] = max_o + 1
        else:
            out[c] = max_o
    return out


@njit(cache=True, parallel=True)
def receiver_step(rec, nbr, arc_dist, step):
    """Distance from each cell to its receiver, km. 0 at outlets."""
    n = rec.shape[0]
    k = nbr.shape[1]
    for i in prange(n):
        step[i] = np.float32(0.0)
        r = rec[i]
        if r == i:
            continue
        for s in range(k):
            if nbr[i, s] == r:
                step[i] = arc_dist[i, s]
                break
    return step


@njit(cache=True)  # NOT inline="always": inlining puts the walk back inside the parfor
def _walk_downstream(z, rec, step, baseline_km, start):
    c = np.int64(start)  # pin the type; parfor lowering otherwise infers float64 here
    dist = np.float32(0.0)
    while dist < baseline_km:
        r = rec[c]
        if r == c:
            break
        dist += step[c]
        c = r
    if dist > 0.0:
        return (z[start] - z[c]) / (dist * np.float32(1000.0))
    return np.float32(0.0)


@njit(cache=True, parallel=True)
def flow_path_slope(z, rec, step, baseline_km, out):
    """Channel gradient measured over a minimum flow-path distance, not cell to cell.

    Single-cell D8 slope is unusable in low-gradient channels on a real DEM: SRTM stores
    int16 metres, so where a river drops less than 1 m across a 30 m cell the measured
    gradient is exactly zero (or, after depression filling, a meaningless +eps residual).
    Those cells are precisely the high-drainage-area ones, so they wreck the slope-area
    regression -- measured theta of 1.87 against an expected 0.4-0.6.

    Walking downstream until at least `baseline_km` has accumulated averages the
    quantisation out. The walk always takes at least one step, so on a coarse mesh where
    one cell already exceeds the baseline this degrades gracefully to cell-to-cell.
    """
    n = z.shape[0]
    for i in prange(n):
        # The walk lives in a helper because numba's parfor pass rejects reassigning a
        # variable seeded from the prange index.
        out[i] = _walk_downstream(z, rec, step, baseline_km, i)
    return out


@njit(cache=True, parallel=True)
def receiver_slope(z, rec, step, slope):
    """Gradient along the flow direction, dimensionless (m/m)."""
    n = z.shape[0]
    for i in prange(n):
        r = rec[i]
        if r == i or step[i] <= 0.0:
            slope[i] = np.float32(0.0)
        else:
            # step is km, z is m.
            slope[i] = (z[i] - z[r]) / (step[i] * np.float32(1000.0))
    return slope


# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class FlowResult:
    filled: np.ndarray   # (N,) f32 m, depression-filled elevation
    rec: np.ndarray      # (N,) i32 receiver index; outlets point at themselves
    order: np.ndarray    # (N,) i32 topological order, receivers first
    don_ptr: np.ndarray  # (N+1,) i64 CSR donor offsets
    don: np.ndarray      # (N,) i32 CSR donor list
    acc: np.ndarray      # (N,) f32 km^2, upstream drainage area
    step: np.ndarray     # (N,) f32 km, distance to receiver
    slope: np.ndarray    # (N,) f32 m/m, local gradient to receiver
    ch_slope: np.ndarray  # (N,) f32 m/m, gradient over the flow-path baseline
    up_len: np.ndarray   # (N,) f32 km, longest upstream flow path
    raised: np.ndarray   # (N,) bool, cell sits inside a filled depression


def route(z, nbr, arc_dist, area, is_sink, slope_baseline_km=0.5,
          metrics=True) -> FlowResult:
    """Full flow-routing stack. ~1.1 s at 6.2M cells.

    `metrics=False` skips the longest-upstream-path and flow-path-slope fields, which only
    Hack's law and the slope-area regression need. The erosion loop calls this every
    timestep and uses neither, so skipping them is worth it there.
    """
    z = np.ascontiguousarray(z, dtype=np.float32)
    n = z.shape[0]
    if nbr.shape[0] != n or area.shape[0] != n:
        raise ValueError("graph arrays disagree on cell count")

    filled = np.empty(n, np.float32)
    closed = np.empty(n, np.bool_)
    keys = np.empty(n, np.float32)
    idxs = np.empty(n, np.int32)
    rec = np.empty(n, np.int32)
    ndon = np.empty(n, np.int32)
    don_ptr = np.empty(n + 1, np.int64)
    don = np.empty(n, np.int32)
    order = np.empty(n, np.int32)
    acc = np.empty(n, np.float32)
    step = np.empty(n, np.float32)
    slope = np.empty(n, np.float32)
    ch_slope = np.empty(n, np.float32)
    up_len = np.empty(n, np.float32)

    priority_flood(z, nbr, is_sink, filled, closed, keys, idxs)
    compute_receivers(filled, nbr, arc_dist, is_sink, rec)
    ns = build_order(rec, ndon, don_ptr, don, order)
    if ns != n:
        raise AssertionError(f"topological order covered {ns:,} of {n:,} cells")
    flow_accumulate(order, rec, area, acc)
    receiver_step(rec, nbr, arc_dist, step)
    receiver_slope(filled, rec, step, slope)
    if metrics:
        flow_path_slope(filled, rec, step, np.float32(slope_baseline_km), ch_slope)
        longest_upstream_path(order, rec, don_ptr, don, arc_dist, nbr, up_len)
    else:
        ch_slope[:] = 0.0
        up_len[:] = 0.0

    # Cells the fill lifted sit on an artificial +eps ramp, so their gradient is a fiction.
    # They cluster in valley bottoms -- i.e. at high drainage area -- so leaving them in
    # the slope-area regression drags the concavity wildly steep. Flag them for exclusion.
    raised = filled > z + np.float32(1e-3)

    return FlowResult(filled, rec, order, don_ptr, don, acc, step, slope, ch_slope,
                      up_len, raised)
