"""Run the metric suite against real SRTM tiles to establish reference values.

These numbers are what generated terrain gets compared against. Nine tiles, chosen to
span climate and tectonic setting rather than to be good fluvial examples:

  east  southern Appalachians   mature, humid, purely fluvial
  west  Olympics / N Cascades   young, active margin, glaciated, reaches sea level
  a     Pyrenees                glaciated collisional
  b     Ahaggar, Sahara         hyper-arid, relict drainage
  c     Peruvian Andes          arid active orogen
  d     Tien Shan               semi-arid active orogen
  e     Guiana / lower Amazon   humid cratonic lowland
  f     Mindanao                wet tropical volcanic arc
  g     Uinta Mts, Utah         glaciated high plateau

It ran on two tiles for a long time, and that was the single largest measurement error
in the project: east has the highest concavity of the nine, and its theta got quoted
project-wide as "the SRTM reference" against which generated terrain was judged to be
failing. It was not. See `terrain-algorithms.md`, "Measured reference values".

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

# Two bands per metric: the PUBLISHED range, and the range actually MEASURED across the
# nine tiles above. They disagree badly and the disagreement is the point.
#
# Published values were established on actively incising humid fluvial terrain, because
# that is where stream-power scaling is cleanest. Most of the Earth is not that. Measured
# here, only four of nine tiles land inside the published theta band and seven of nine sit
# ABOVE the published Rb band -- so "outside published" means the tile is not a humid
# fluvial orogen, not that anything is wrong with it or with the measurement.
#
# This block used to carry the published band alone and print a bare [OUT] against it. The
# docstring hedged that for glaciation on the west tile and nowhere else, which left the
# impression that east was normal and everything else was a special case. It is the
# other way round.
EXPECTED = {
    #                            published     measured over the nine tiles
    "concavity_theta":          ((0.40, 0.60), (0.130, 0.516), "slope-area concavity"),
    "hack_h":                   ((0.50, 0.62), (0.434, 0.602), "Hack exponent"),
    "horton_bifurcation_ratio": ((3.0, 5.0), (4.67, 6.95), "Horton bifurcation Rb"),
    "horton_length_ratio":      ((1.5, 3.5), (1.75, 3.28), "Horton length Rl"),
    "horton_area_ratio":        ((3.0, 6.0), (3.91, 5.77), "Horton area Ra"),
}


def summarise(r):
    print(f"\n=== {r['label']} ===")
    print(f"  cells {r['cells']:,}   channel cells {r['channel_cells']:,}   "
          f"interior {r['interior_fraction']:.1%}   relief {r['relief_m']:,.0f} m")
    print(f"  slope-area concavity theta   {r['concavity_theta']:+.3f}  (R^2 {r['concavity_r2']:.3f})")
    # The per-basin figure is the one generated worlds are quoted on, and it is not the
    # same number: on the east tile it is 0.398 against a whole-window 0.516. Comparing
    # across that difference is how a 0.23 phantom gap got into the docs.
    print(f"    per basin                  {r['basin_theta_median']:+.3f}  "
          f"(IQR {r['basin_theta_p25']:.3f}-{r['basin_theta_p75']:.3f}, "
          f"n={r['basin_count']})")
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

    print("  vs. published range, and vs. the nine-tile measured range:")
    for key, (pub, meas, name) in EXPECTED.items():
        v = r.get(key, np.nan)
        def mark(band):
            return "in " if (np.isfinite(v) and band[0] <= v <= band[1]) else "out"
        print(f"    {name:24s} {v:6.3f}   published {pub[0]}-{pub[1]} [{mark(pub)}]   "
              f"9-tile {meas[0]}-{meas[1]} [{mark(meas)}]")


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

    if len(results) >= 2:
        # The distribution across tiles, not any one tile, is the reference. Printing a
        # single tile's numbers and calling them "the SRTM values" is exactly the mistake
        # this script made for as long as it had two tiles in it.
        print(f"\n=== across {len(results)} tiles ===")
        print(f"  {'metric':24s} {'mean':>8s} {'sd':>8s} {'min':>8s} {'max':>8s}")
        for key in ("concavity_theta", "basin_theta_median", "hack_h",
                    "hypsometric_integral", "drainage_density", "slope_mean_deg",
                    "relief_m"):
            v = np.array([r[key] for r in results], dtype=float)
            v = v[np.isfinite(v)]
            print(f"  {key:24s} {v.mean():8.3f} {v.std(ddof=1):8.3f} "
                  f"{v.min():8.3f} {v.max():8.3f}")


if __name__ == "__main__":
    main()
