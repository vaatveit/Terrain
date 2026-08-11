"""Feasibility spike: how fast are the erosion-loop graph kernels at production scale?

Everything downstream is sized by this number. The production grid is a class-I Goldberg
(geodesic hex) tiling at frequency n=785 -> 10*785^2+2 = 6,162,252 cells, each with 6
neighbours (12 pentagons excepted). Here we stand in a periodic triangular lattice of the
same cell count and degree, which reproduces the memory locality without needing the real
grid to exist. See bench_real_grid.py for the same measurement on the actual grid.

Kernels come from terrain.flow -- this file times them, it does not reimplement them.

Run:  .venv\\Scripts\\python.exe spikes\\bench_flow_kernels.py
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain.flow import (  # noqa: E402
    build_order, compute_receivers, flow_accumulate, priority_flood,
)

N_TARGET = 10 * 785**2 + 2
SEA_FRACTION = 0.30


def build_lattice(n_target):
    """Neighbour table for a W x H triangular lattice on a torus, ~n_target cells."""
    w = int(round(np.sqrt(n_target)))
    h = int(np.ceil(n_target / w))
    n = w * h

    idx = np.arange(n, dtype=np.int64)
    row, col = idx // w, idx % w
    offsets = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1))

    nbr = np.empty((n, 6), dtype=np.int32)
    for s, (dr, dc) in enumerate(offsets):
        nbr[:, s] = ((row + dr) % h) * w + (col + dc) % w
    return n, w, h, nbr


def coherent_field(h, w, seed=12345):
    """1/f^2 fBm via Fourier filtering.

    A COHERENT field matters more than it looks. On white noise, ocean is scattered
    everywhere, every land cell is adjacent to an outlet, the priority-flood heap stays
    tiny and the benchmark flatters itself -- measured 3x faster with a realistic field
    than with white noise, because organised terrain has far fewer pits.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    freq = np.sqrt(fy**2 + fx**2)
    freq[0, 0] = 1.0
    spec = np.fft.fft2(white) / (freq**2)
    spec[0, 0] = 0.0
    z = np.fft.ifft2(spec).real
    z = (z - z.mean()) / z.std()
    return np.ascontiguousarray(z.ravel(), dtype=np.float32) * np.float32(1000.0)


def main():
    print("Building stand-in grid...")
    n, w, h, nbr = build_lattice(N_TARGET)
    print(f"  {n:,} cells ({w} x {h} torus, degree 6); target was {N_TARGET:,}")

    z = coherent_field(h, w)
    sea_level = np.float32(np.quantile(z, SEA_FRACTION))
    is_ocean = np.ascontiguousarray(z <= sea_level)
    print(f"  ocean cells: {is_ocean.sum():,} ({is_ocean.mean():.1%})")

    arc_dist = np.full((n, 6), 9.773, dtype=np.float32)
    area = np.full(n, 82.71, dtype=np.float32)

    buf = {name: np.empty(n, dt) for name, dt in (
        ("filled", np.float32), ("closed", np.bool_), ("keys", np.float32),
        ("idxs", np.int32), ("rec", np.int32), ("ndon", np.int32),
        ("don", np.int32), ("order", np.int32), ("acc", np.float32),
    )}
    don_ptr = np.empty(n + 1, np.int64)

    print("\nCompiling (small warmup input)...")
    t0 = time.perf_counter()
    sn, sw, sh, snbr = build_lattice(10_000)
    sz = coherent_field(sh, sw, seed=1)[:sn]
    socean = np.ascontiguousarray(sz <= np.quantile(sz, 0.3))
    sbuf = {k: np.empty(sn, v.dtype) for k, v in buf.items()}
    priority_flood(sz, snbr, socean, sbuf["filled"], sbuf["closed"], sbuf["keys"], sbuf["idxs"])
    compute_receivers(sz, snbr, np.ones((sn, 6), np.float32), socean, sbuf["rec"])
    build_order(sbuf["rec"], sbuf["ndon"], np.empty(sn + 1, np.int64), sbuf["don"], sbuf["order"])
    flow_accumulate(sbuf["order"], sbuf["rec"], np.ones(sn, np.float32), sbuf["acc"])
    print(f"  compile: {time.perf_counter() - t0:.1f} s (cache=True amortises this)")

    print(f"\nProduction scale, {n:,} cells:")
    results = {}

    t0 = time.perf_counter()
    priority_flood(z, nbr, is_ocean, buf["filled"], buf["closed"], buf["keys"], buf["idxs"])
    results["priority_flood"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    compute_receivers(buf["filled"], nbr, arc_dist, is_ocean, buf["rec"])
    results["compute_receivers"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ns = build_order(buf["rec"], buf["ndon"], don_ptr, buf["don"], buf["order"])
    results["build_order"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    flow_accumulate(buf["order"], buf["rec"], area, buf["acc"])
    results["flow_accumulate"] = time.perf_counter() - t0

    assert ns == n, f"topological order covered {ns:,} of {n:,} cells"
    assert np.all(buf["filled"] >= z - 1e-3), "priority-flood lowered terrain"
    print(f"  (sanity: order covers all {ns:,} cells; "
          f"cells raised by fill {(buf['filled'] > z + 1e-3).sum():,})")

    print()
    per_iter = sum(results.values())
    for name, dt in results.items():
        print(f"  {name:20s} {dt:7.2f} s   {100 * dt / per_iter:5.1f}%")
    print(f"  {'TOTAL per iteration':20s} {per_iter:7.2f} s")

    print("\nProjected erosion-loop wall time:")
    for iters in (500, 1000, 2000):
        print(f"  {iters:5d} timesteps -> {iters * per_iter / 3600:5.2f} h")


if __name__ == "__main__":
    main()
