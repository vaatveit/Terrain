"""Feasibility spike: how fast are the erosion-loop graph kernels in Numba at production scale?

Everything downstream is sized by this number. The production grid is a class-I Goldberg
(geodesic hex) tiling at frequency n=785 -> 10*785^2+2 = 6,162,252 cells, each with 6
neighbours (12 pentagons excepted). Here we stand in a periodic triangular lattice of the
same cell count and the same 6-neighbour degree, which reproduces the degree and the memory
locality without needing the real grid generator to exist yet.

Kernels timed, in the order the erosion loop calls them:
  1. priority-flood depression fill (O(N log N), the suspected bottleneck)
  2. receiver computation, steepest descent over 6 neighbours (O(N))
  3. topological ordering of the flow tree (O(N))
  4. flow accumulation of real cell areas (O(N))

Run:  .venv\\Scripts\\python.exe spikes\\bench_flow_kernels.py
"""

import time

import numpy as np
from numba import njit, prange

N_TARGET = 10 * 785**2 + 2  # 6,162,252 production cells
SEA_FRACTION = 0.30  # ocean cells act as the outlets priority-flood floods inward from
EPS = np.float32(1e-4)  # priority-flood increment, metres


# ---------------------------------------------------------------------------
# Stand-in grid: periodic triangular lattice, degree 6
# ---------------------------------------------------------------------------


def build_lattice(n_target):
    """Neighbour table for a W x H triangular lattice on a torus, ~n_target cells."""
    w = int(round(np.sqrt(n_target)))
    h = int(np.ceil(n_target / w))
    n = w * h

    idx = np.arange(n, dtype=np.int64)
    row, col = idx // w, idx % w
    # Triangular lattice: 4 axis-aligned + 2 diagonal, giving degree 6.
    offsets = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1))

    nbr = np.empty((n, 6), dtype=np.int32)
    for s, (dr, dc) in enumerate(offsets):
        nbr[:, s] = ((row + dr) % h) * w + (col + dc) % w
    return n, w, h, nbr


# ---------------------------------------------------------------------------
# Hand-rolled binary min-heap over preallocated arrays
# (Numba does support heapq, but over typed lists, with per-op overhead that
#  matters when you push several million times.)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 1. Priority-flood depression fill (Barnes, Lehman & Mulla 2014, +epsilon variant)
# ---------------------------------------------------------------------------


@njit(cache=True)
def priority_flood(z, nbr, is_ocean, filled, closed, keys, idxs):
    n = z.shape[0]
    for i in range(n):
        filled[i] = z[i]
        closed[i] = False

    size = 0
    for i in range(n):
        if is_ocean[i]:
            closed[i] = True
            size = heap_push(keys, idxs, size, z[i], i)

    max_size = size
    while size > 0:
        key, c, size = heap_pop(keys, idxs, size)
        for s in range(6):
            j = nbr[c, s]
            if j < 0 or closed[j]:
                continue
            closed[j] = True
            if filled[j] <= key:
                filled[j] = key + EPS
            size = heap_push(keys, idxs, size, filled[j], j)
        if size > max_size:
            max_size = size
    return max_size


# ---------------------------------------------------------------------------
# 2. Receivers: steepest descent over 6 neighbours. Outlets point at themselves.
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def compute_receivers(z, nbr, arc_dist, is_ocean, rec):
    n = z.shape[0]
    for i in prange(n):
        if is_ocean[i]:
            rec[i] = i
            continue
        best = i
        best_slope = np.float32(0.0)
        for s in range(6):
            j = nbr[i, s]
            if j < 0:
                continue
            slope = (z[i] - z[j]) / arc_dist[i, s]
            if slope > best_slope:
                best_slope = slope
                best = j
        rec[i] = best
    return rec


# ---------------------------------------------------------------------------
# 3. Topological order of the flow tree (parents strictly before children)
# ---------------------------------------------------------------------------


@njit(cache=True)
def build_order(rec, ndon, don_ptr, don, order):
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


# ---------------------------------------------------------------------------
# 4. Flow accumulation: sum real cell AREAS upstream, never cell counts.
# ---------------------------------------------------------------------------


@njit(cache=True)
def flow_accumulate(order, rec, area, acc):
    n = order.shape[0]
    for i in range(n):
        acc[i] = area[i]
    for k in range(n - 1, -1, -1):
        c = order[k]
        r = rec[c]
        if r != c:
            acc[r] += acc[c]
    return acc


# ---------------------------------------------------------------------------


def main():
    print("Building stand-in grid...")
    n, w, h, nbr = build_lattice(N_TARGET)
    print(f"  {n:,} cells ({w} x {h} torus, degree 6); target was {N_TARGET:,}")

    rng = np.random.default_rng(12345)
    # A COHERENT field matters more than it looks. On white noise, ocean is scattered
    # everywhere, so every land cell is adjacent to an outlet, the priority-flood heap
    # stays tiny and the benchmark flatters itself. Real terrain has continent-scale
    # structure and deep drainage trees, which is what actually stresses the fill.
    # Build 1/f^2 fBm by filtering white noise in Fourier space.
    print("Building coherent 1/f^2 elevation field...")
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    freq = np.sqrt(fy**2 + fx**2)
    freq[0, 0] = 1.0
    spec = np.fft.fft2(white) / (freq**2)
    spec[0, 0] = 0.0
    z = np.fft.ifft2(spec).real
    z = (z - z.mean()) / z.std()
    z = np.ascontiguousarray(z.ravel(), dtype=np.float32)
    z *= np.float32(1000.0)

    sea_level = np.float32(np.quantile(z, SEA_FRACTION))
    is_ocean = z <= sea_level
    print(f"  ocean cells: {is_ocean.sum():,} ({is_ocean.mean():.1%})")

    arc_dist = np.full((n, 6), 9.773, dtype=np.float32)
    area = np.full(n, 82.71, dtype=np.float32)

    filled = np.empty(n, dtype=np.float32)
    closed = np.empty(n, dtype=np.bool_)
    keys = np.empty(n, dtype=np.float32)
    idxs = np.empty(n, dtype=np.int32)
    rec = np.empty(n, dtype=np.int32)
    ndon = np.empty(n, dtype=np.int32)
    don_ptr = np.empty(n + 1, dtype=np.int64)
    don = np.empty(n, dtype=np.int32)
    order = np.empty(n, dtype=np.int32)
    acc = np.empty(n, dtype=np.float32)

    for dt, arr in (("float32", z), ("float32", filled), ("float32", acc)):
        assert arr.dtype == np.dtype(dt), f"float32 discipline broken: {arr.dtype}"

    print("\nCompiling (small warmup input)...")
    t0 = time.perf_counter()
    sn, sw, sh, snbr = build_lattice(10_000)
    sz = rng.random(sn, dtype=np.float32)
    socean = sz <= 0.3
    priority_flood(
        sz, snbr, socean,
        np.empty(sn, np.float32), np.empty(sn, np.bool_),
        np.empty(sn, np.float32), np.empty(sn, np.int32),
    )
    compute_receivers(sz, snbr, np.ones((sn, 6), np.float32), socean, np.empty(sn, np.int32))
    build_order(
        np.arange(sn, dtype=np.int32), np.empty(sn, np.int32),
        np.empty(sn + 1, np.int64), np.empty(sn, np.int32), np.empty(sn, np.int32),
    )
    flow_accumulate(
        np.arange(sn, dtype=np.int32), np.arange(sn, dtype=np.int32),
        np.ones(sn, np.float32), np.empty(sn, np.float32),
    )
    print(f"  compile: {time.perf_counter() - t0:.1f} s (cache=True amortises this)")

    print(f"\nProduction scale, {n:,} cells:")
    results = {}

    t0 = time.perf_counter()
    max_heap = priority_flood(z, nbr, is_ocean, filled, closed, keys, idxs)
    results["priority_flood"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    compute_receivers(filled, nbr, arc_dist, is_ocean, rec)
    results["compute_receivers"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ns = build_order(rec, ndon, don_ptr, don, order)
    results["build_order"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    flow_accumulate(order, rec, area, acc)
    results["flow_accumulate"] = time.perf_counter() - t0

    assert ns == n, f"topological order covered {ns:,} of {n:,} cells"
    assert np.all(filled >= z - 1e-3), "priority-flood lowered terrain"
    land = ~is_ocean
    print(f"  sanity: order covers all {ns:,} cells")
    print(f"          max drainage basin {acc.max():,.0f} km^2 "
          f"({acc.max() / 82.71:,.0f} cells, {100 * acc.max() / (land.sum() * 82.71):.1f}% of land)")
    print(f"          cells raised by fill: {(filled > z + 1e-3).sum():,}")
    print(f"          peak priority-flood heap: {max_heap:,} entries "
          f"({100 * max_heap / n:.1f}% of grid)")

    print()
    per_iter = sum(results.values())
    for name, dt in results.items():
        print(f"  {name:20s} {dt:7.2f} s   {100 * dt / per_iter:5.1f}%")
    print(f"  {'TOTAL per iteration':20s} {per_iter:7.2f} s")

    print("\nProjected erosion-loop wall time (fill every iteration):")
    for iters in (200, 500, 1000, 2000):
        print(f"  {iters:5d} timesteps -> {iters * per_iter / 3600:6.2f} h")

    without_fill = per_iter - results["priority_flood"]
    print("\nIf depression fill runs only every 10th iteration:")
    amortised = without_fill + results["priority_flood"] / 10
    for iters in (200, 500, 1000, 2000):
        print(f"  {iters:5d} timesteps -> {iters * amortised / 3600:6.2f} h")


if __name__ == "__main__":
    main()
