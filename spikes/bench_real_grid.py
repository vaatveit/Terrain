"""Re-run the flow kernels on the real Goldberg grid, not the lattice stand-in.

bench_flow_kernels.py used a periodic triangular lattice as a proxy for cell count and
degree. This confirms the numbers hold on the actual grid, which differs in ways that
could matter: 12 pentagons, non-uniform cell areas and arc distances, and an index
ordering (by icosahedron face) with different cache behaviour from a raster.

Run:  .venv\\Scripts\\python.exe spikes\\bench_real_grid.py
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from spikes.bench_flow_kernels import (  # noqa: E402
    build_order, compute_receivers, flow_accumulate, priority_flood,
)
from terrain.grid import Grid  # noqa: E402

GRID_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "grid_n785.npz"
SEA_FRACTION = 0.30


def coherent_field(xyz, n_waves=96, seed=7):
    """Smooth 1/k^2 field on the sphere: sum of random plane waves.

    A stand-in for the real 3D-noise operator, which doesn't exist yet. What matters here
    is that the field is CONTINENT-SCALE coherent -- on white noise every land cell touches
    ocean, the priority-flood heap stays tiny and the benchmark flatters itself.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(xyz.shape[0], dtype=np.float64)
    for _ in range(n_waves):
        k = rng.standard_normal(3)
        k /= np.linalg.norm(k)
        wavenumber = rng.uniform(1.0, 12.0)
        phase = rng.uniform(0, 2 * np.pi)
        out += np.cos(wavenumber * (xyz @ k) * np.pi + phase) / wavenumber**2
    out -= out.mean()
    out /= out.std()
    return np.ascontiguousarray(out * 1000.0, dtype=np.float32)


def main():
    if not GRID_PATH.exists():
        sys.exit(f"missing {GRID_PATH}; run: python terrain/grid.py -n 785 --save {GRID_PATH}")

    print(f"Loading {GRID_PATH.name}...")
    t0 = time.perf_counter()
    g = Grid.load(GRID_PATH)
    print(f"  {g.cell_count:,} cells in {time.perf_counter() - t0:.1f} s")

    n = g.cell_count
    nbr = np.ascontiguousarray(g.neighbours)
    arc = np.ascontiguousarray(g.arc_dist)
    area = np.ascontiguousarray(g.area)

    print("Building coherent elevation field...")
    z = coherent_field(g.xyz)
    sea_level = np.float32(np.quantile(z, SEA_FRACTION))
    is_ocean = np.ascontiguousarray(z <= sea_level)
    print(f"  ocean {is_ocean.mean():.1%}, land {1 - is_ocean.mean():.1%}")

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

    results = {}
    t0 = time.perf_counter()
    max_heap = priority_flood(z, nbr, is_ocean, filled, closed, keys, idxs)
    results["priority_flood"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    compute_receivers(filled, nbr, arc, is_ocean, rec)
    results["compute_receivers"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ns = build_order(rec, ndon, don_ptr, don, order)
    results["build_order"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    flow_accumulate(order, rec, area, acc)
    results["flow_accumulate"] = time.perf_counter() - t0

    assert ns == n, f"topological order covered {ns:,} of {n:,}"
    assert np.all(filled >= z - 1e-3)

    land_area = float(area[~is_ocean].sum())
    print(f"\n  order covers all {ns:,} cells")
    print(f"  largest basin {acc.max():,.0f} km^2 ({100 * acc.max() / land_area:.1f}% of land)")
    print(f"  cells raised by fill {(filled > z + 1e-3).sum():,}")
    print(f"  peak heap {max_heap:,} ({100 * max_heap / n:.1f}% of grid)")

    print()
    per_iter = sum(results.values())
    for name, dt in results.items():
        print(f"  {name:20s} {dt:7.2f} s   {100 * dt / per_iter:5.1f}%")
    print(f"  {'TOTAL per iteration':20s} {per_iter:7.2f} s")
    print("\n  projected erosion loop:")
    for iters in (500, 1000, 2000):
        print(f"    {iters:5d} timesteps -> {iters * per_iter / 3600:5.2f} h")


if __name__ == "__main__":
    main()
