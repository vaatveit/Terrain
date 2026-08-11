"""Run the metric suite against real SRTM tiles to establish reference values.

These numbers are what generated terrain gets compared against. Two deliberately
contrasting tiles:
  east  southern Appalachians  -- mature, humid, purely fluvial
  west  Olympics / N Cascades  -- young, active margin, glaciated, reaches sea level

Run:  .venv\\Scripts\\python.exe scripts\\validate_metrics.py
"""

import argparse
import glob
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import dem, metrics  # noqa: E402

# Published ranges for fluvial terrain. Read an [OUT] on the west tile as a finding, not a
# failure: glacial erosion genuinely violates stream-power scaling -- U-shaped valleys have
# flat floors and steep walls, which flattens slope-area and shortens main channels
# relative to basin area. The east tile is unglaciated and is the stricter reference.
EXPECTED = {
    "concavity_theta": (0.40, 0.60, "slope-area concavity"),
    "hack_h": (0.50, 0.62, "Hack exponent"),
    "horton_bifurcation_ratio": (3.0, 5.0, "Horton bifurcation Rb"),
    "horton_length_ratio": (1.5, 3.5, "Horton length Rl"),
    "horton_area_ratio": (3.0, 6.0, "Horton area Ra"),
}


def summarise(r):
    print(f"\n=== {r['label']} ===")
    print(f"  cells {r['cells']:,}   channel cells {r['channel_cells']:,}   "
          f"interior {r['interior_fraction']:.1%}   relief {r['relief_m']:,.0f} m")
    print(f"  slope-area concavity theta   {r['concavity_theta']:+.3f}  (R^2 {r['concavity_r2']:.3f})")
    print(f"  Hack exponent h              {r['hack_h']:+.3f}  (R^2 {r['hack_r2']:.3f}, n={r['hack_n']:,})")
    print(f"  Horton  Rb {r['horton_bifurcation_ratio']:.2f}   "
          f"Rl {r['horton_length_ratio']:.2f}   Ra {r['horton_area_ratio']:.2f}   "
          f"max order {r['horton_max_order']}")
    print(f"  stream counts by order       {r['horton_stream_counts']}")
    print(f"  hypsometric integral         {r['hypsometric_integral']:.3f}")
    print(f"  drainage density             {r['drainage_density']:.3f} km/km^2")
    print(f"  slope  mean {r['slope_mean_deg']:.1f} deg   median {r['slope_median_deg']:.1f}   "
          f"p90 {r['slope_p90_deg']:.1f}   p99 {r['slope_p99_deg']:.1f}   "
          f">30deg {r['slope_frac_over_30deg']:.1%}")

    print("  vs. published ranges:")
    for key, (lo, hi, name) in EXPECTED.items():
        v = r.get(key, np.nan)
        mark = "ok " if (np.isfinite(v) and lo <= v <= hi) else "OUT"
        print(f"    [{mark}] {name:28s} {v:6.3f}   expected {lo}-{hi}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Projects\Data")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--threshold-km2", type=float, default=1.0,
                    help="critical drainage area defining a channel")
    ap.add_argument("--connectivity", type=int, default=8, choices=(4, 8))
    args = ap.parse_args()

    tiles = sorted(glob.glob(str(pathlib.Path(args.data) / "*.tif")))
    if not tiles:
        sys.exit(f"no .tif under {args.data}")

    results = []
    for path in tiles:
        label = pathlib.Path(path).stem
        print(f"\nLoading {label} ({args.size}x{args.size} window, D{args.connectivity})...")
        t0 = time.perf_counter()
        win = dem.load_window(path, size=args.size, connectivity=args.connectivity, name=label)
        print(f"  window at lat {win.lat.min():.3f}..{win.lat.max():.3f}, "
              f"lon {win.lon.min():.3f}..{win.lon.max():.3f}")
        print(f"  cell area {win.area.min():.6f}..{win.area.max():.6f} km^2, "
              f"spacing ~{np.median(win.arc_dist[win.arc_dist > 0]) * 1000:.1f} m")
        print(f"  loaded in {time.perf_counter() - t0:.1f} s")

        t0 = time.perf_counter()
        r = metrics.report(
            win.z, win.neighbours, win.arc_dist, win.area, win.is_sink,
            is_border=win.is_border, channel_threshold_km2=args.threshold_km2, label=label,
        )
        print(f"  metrics in {time.perf_counter() - t0:.1f} s")
        summarise(r)
        results.append(r)

    if len(results) == 2:
        a, b = results
        print(f"\n=== {a['label']}  vs  {b['label']} ===")
        for key in ("concavity_theta", "hack_h", "hypsometric_integral",
                    "drainage_density", "slope_mean_deg", "relief_m"):
            print(f"  {key:24s} {a[key]:10.3f}   {b[key]:10.3f}")


if __name__ == "__main__":
    main()
