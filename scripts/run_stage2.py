"""Run Stage 1 then Stage 2 and measure whether erosion actually produced realism.

The success criterion is specific: slope-area concavity theta must move from the noise-only
baseline of ~0.11 into the 0.4-0.6 band that real terrain occupies. Theory says a stream
power model at steady state gives theta = m/n exactly, so this is an end-to-end check on
the solver, the flow routing, the spherical geometry and the metrics all at once.

Run:  .venv\\Scripts\\python.exe scripts\\run_stage2.py --n 256 --steps 300
"""

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import metrics, stage1, stage2  # noqa: E402
from terrain.grid import Grid, build_goldberg  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRTM_EAST = {"concavity_theta": 0.516, "hack_h": 0.543}
STAGE1_BASELINE = {"concavity_theta": 0.106, "hack_h": 0.469}


def load_or_build(n):
    cached = ROOT / "data" / f"grid_n{n}.npz"
    if cached.exists():
        return Grid.load(cached)
    g = build_goldberg(n, verbose=True)
    cached.parent.mkdir(parents=True, exist_ok=True)
    g.save(cached)
    return g


def measure(grid, z, is_ocean, label):
    thr = float(np.median(grid.area) * 6)
    return metrics.report(
        np.ascontiguousarray(z, dtype=np.float32), grid.neighbours, grid.arc_dist,
        grid.area, is_sink=is_ocean, channel_threshold_km2=thr, label=label,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--K", type=float, default=2.5e-6)
    ap.add_argument("--m", type=float, default=0.45)
    ap.add_argument("--n-exp", type=float, dest="n_exp", default=1.0)
    ap.add_argument("--D", type=float, default=0.05)
    ap.add_argument("--G", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=0.5, help="timestep, Myr")
    ap.add_argument("--uplift", type=float, default=800.0, help="max uplift, m/Myr")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    print(f"Grid n={args.n}...")
    grid = load_or_build(args.n)
    print(f"  {grid.cell_count:,} cells, spacing "
          f"{grid.arc_dist[grid.neighbours >= 0].mean():.2f} km")

    print("\nStage 1...")
    crust = stage1.generate_crust(grid, seed=args.seed)
    is_ocean = ~crust.land_mask
    print(f"  land {crust.land_fraction:.1%}, relief "
          f"{crust.elevation.min():,.0f}..{crust.elevation.max():,.0f} m")

    before = measure(grid, crust.elevation, is_ocean, "stage1")

    uplift = stage2.uplift_field(grid, crust.land_mask, seed=args.seed,
                                 u_max_m_per_myr=args.uplift)
    print(f"  uplift {uplift[crust.land_mask].min():.0f}.."
          f"{uplift[crust.land_mask].max():.0f} m/Myr")

    p = stage2.LemParams(K=args.K, m=args.m, n=args.n_exp, D=args.D, G=args.G,
                         dt_myr=args.dt)
    print(f"\nStage 2: {args.steps} steps x {args.dt} Myr = "
          f"{args.steps * args.dt:,.0f} Myr")
    print(f"  K={p.K:g}  m={p.m}  n={p.n}  D={p.D}  G={p.G}")
    print(f"  predicted steady-state theta = m/n = {p.m / p.n:.3f}")

    res = stage2.evolve(grid, crust.elevation, uplift, is_ocean, p,
                        n_steps=args.steps, verbose=True,
                        report_every=max(1, args.steps // 8))
    print(f"  evolved in {res.seconds:.1f} s "
          f"({res.seconds / args.steps * 1000:.0f} ms/step)")

    land = crust.land_mask
    z = res.elevation
    print(f"  relief now {z[land].min():,.0f}..{z[land].max():,.0f} m "
          f"(was {crust.elevation[land].min():,.0f}..{crust.elevation[land].max():,.0f})")

    first, last = res.history[0][1], res.history[-1][1]
    print(f"  mean |dz| {first:,.1f} -> {last:,.2f} m/Myr "
          f"({'converging' if last < first else 'NOT converging'})")

    print("\nMeasuring...")
    t0 = time.perf_counter()
    after = measure(grid, z, is_ocean, "stage2")
    print(f"  measured in {time.perf_counter() - t0:.1f} s")

    print(f"\n  {'metric':22s} {'stage1':>9s} {'stage2':>9s} {'SRTM east':>10s}   verdict")
    for key, name, lo, hi in (
        ("basin_theta_median", "theta, per-basin", 0.40, 0.60),
        ("concavity_theta", "theta, whole planet*", 0.40, 0.60),
        ("hack_h", "Hack exponent h", 0.50, 0.62),
    ):
        v = after[key]
        ok = np.isfinite(v) and lo <= v <= hi
        ref = SRTM_EAST.get(key, SRTM_EAST["concavity_theta"])
        print(f"  {name:22s} {before[key]:9.3f} {v:9.3f} {ref:10.3f}   "
              f"{'INSIDE' if ok else 'outside'} {lo}-{hi}")
    print("  *whole-planet theta mixes cratons and orogens, so it is NOT comparable to")
    print("   published values or to a DEM window. Per-basin is the like-for-like number.")
    if after["basin_count"]:
        print(f"  {'  basins measured':22s} {after['basin_count']:9d}   "
              f"IQR {after['basin_theta_p25']:.3f}-{after['basin_theta_p75']:.3f}")
    for key, name in (("hack_r2", "  hack fit R^2"),
                      ("hypsometric_integral", "hypsometric integral"),
                      ("horton_bifurcation_ratio", "Horton Rb"),
                      ("channel_cells", "channel cells")):
        print(f"  {name:22s} {before[key]:9.3f} {after[key]:9.3f}")

    if args.render:
        from run_stage1 import render_equirect
        crust.elevation = z
        crust.land_mask = z > 0
        out = str(ROOT / "data" / f"stage2_n{args.n}_seed{args.seed}.png")
        print(f"\nRendering {out}...")
        theta = after["basin_theta_median"]
        render_equirect(grid, crust, out, title=(
            f"Stage 2 — {args.steps * args.dt:,.0f} Myr of erosion, "
            f"peaks {z[land].max():,.0f} m, per-basin θ {theta:.3f}"))


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()
