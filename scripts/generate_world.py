"""Run the whole pipeline and export a world.

Stages 0-9 end to end: grid, tectonic crust, landscape evolution coupled with climate,
glacial erosion, hydrology, biomes, amplification, export.

Run:  .venv\\Scripts\\python.exe scripts\\generate_world.py --n 256 --out data/world01
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import (  # noqa: E402
    biomes, climate, coupled, export, flow, glacial, hydrology, metrics, plates, stage1,
    stage2,
)
from terrain.grid import Grid, build_goldberg  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRTM_EAST = {"concavity_theta": 0.516, "hack_h": 0.543}


def load_or_build(n):
    cached = ROOT / "data" / f"grid_n{n}.npz"
    if cached.exists():
        return Grid.load(cached)
    g = build_goldberg(n, verbose=True)
    cached.parent.mkdir(parents=True, exist_ok=True)
    g.save(cached)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--width", type=int, default=0, help="raster width; 0 = 8x grid n")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-glacial", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="reload the exported GeoTIFF through the SRTM path and measure it")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out or (ROOT / "data" / f"world_n{args.n}_s{args.seed}"))
    width = args.width or max(1024, 8 * args.n)
    t_start = time.perf_counter()

    print(f"[0] grid n={args.n}")
    grid = load_or_build(args.n)
    print(f"    {grid.cell_count:,} cells, "
          f"{np.mean(grid.arc_dist[grid.neighbours >= 0]):.2f} km spacing")

    print("[1] tectonic crust")
    crust = stage1.generate_crust_tectonic(grid, seed=args.seed)
    land, is_ocean = crust.land_mask, ~crust.land_mask
    print(f"    land {crust.land_fraction:.1%}, {len(crust.landmass_areas)} landmasses, "
          f"relief {crust.elevation.min():,.0f}..{crust.elevation.max():,.0f} m")

    print(f"[2-4] landscape evolution, {args.passes} passes coupled with climate")
    per_pass = max(1, args.steps // args.passes)
    cres = coupled.run(grid, crust.elevation, crust.uplift, is_ocean,
                       passes=args.passes, steps_per_pass=per_pass, verbose=True)
    z, clim = cres.elevation, cres.climate
    print(f"    {cres.seconds:.0f} s, peaks {z[land].max():,.0f} m")

    if not args.no_glacial:
        print("[5] glacial erosion")
        gres = glacial.erode(grid, z, clim.temperature_c, is_ocean)
        moved = np.abs(gres.elevation - z)
        print(f"    glaciated {gres.glaciated[land].mean():.1%} of land, "
              f"{moved[land & gres.glaciated].mean():.0f} m mean lowering under ice")
        z = gres.elevation

    print("[6] hydrology")
    hyd = hydrology.analyse(grid, z, is_ocean, clim.precipitation_mm, clim.temperature_c)
    endo = sum(1 for L in hyd.lakes if L["endorheic"])
    print(f"    {len(hyd.lakes)} lakes ({endo} endorheic), "
          f"{hyd.river_mask.sum():,} river cells, "
          f"max discharge {hyd.discharge.max():,.0f} km^3/yr")

    print("[7] biomes")
    biome = biomes.classify(clim.temperature_c, clim.precipitation_mm, land, z, grid.lat)
    shares = biomes.summary(biome, grid.area, land)
    print("    " + ", ".join(f"{n} {s:.0%}" for n, s in shares[:4]))

    print("[9] metrics on the mesh")
    fr = flow.route(z, grid.neighbours, grid.arc_dist, grid.area, is_ocean)
    thr = float(np.median(grid.area) * 6)
    m = metrics.report(z, grid.neighbours, grid.arc_dist, grid.area, is_sink=is_ocean,
                       channel_threshold_km2=thr, label="mesh")
    summary = {k: (float(m[k]) if np.isfinite(m[k]) else None) for k in
               ("basin_theta_median", "concavity_theta", "hack_h",
                "hypsometric_integral", "horton_bifurcation_ratio")}
    print(f"    per-basin theta {m['basin_theta_median']:.3f} "
          f"(SRTM east {SRTM_EAST['concavity_theta']}), "
          f"Hack h {m['hack_h']:.3f} (SRTM east {SRTM_EAST['hack_h']})")

    print(f"[8-9] amplify and export -> {out_dir}  ({width}x{width // 2})")
    params = {
        "crust": stage1.CrustParams().__dict__, "plates": plates.PlateParams().__dict__,
        "lem": stage2.LemParams().__dict__, "climate": climate.ClimateParams().__dict__,
        "glacial": None if args.no_glacial else glacial.GlacialParams().__dict__,
        "hydrology": hydrology.HydrologyParams().__dict__,
        "steps": args.steps, "passes": args.passes,
    }
    res = export.export_world(
        out_dir, grid, z, land, args.seed, slope=fr.slope, drainage_km2=fr.acc,
        temperature_c=clim.temperature_c, precipitation_mm=clim.precipitation_mm,
        biome=biome, width=width, params=params, metrics_summary=summary)
    for name, path in res.files.items():
        size = pathlib.Path(path).stat().st_size / 1e6
        print(f"    {name:10s} {size:8.1f} MB  {pathlib.Path(path).name}")

    if args.verify:
        print("\n[verify] fidelity: raster pixels against the mesh field")
        f = export.verify_fidelity(res.files["elevation"], grid, z)
        print(f"    mean |error| {f['mean_abs_error_m']:.1f} m, p99 {f['p99_abs_error_m']:.0f} m, "
              f"bias {f['bias_m']:+.2f} m, against {f['relief_m']:,.0f} m of relief")
        print("    (non-zero by design: Stage 8 adds sub-grid detail on export)")

        print("[verify] the metrics module reads the export unmodified")
        r = export.verify_roundtrip(res.files["elevation"], window=min(512, width // 4))
        print(f"    theta {r['concavity_theta']:.3f}, Hack h {r['hack_h']:.3f}, "
              f"{r['channel_cells']:,} channel cells")
        print("    (not SRTM-comparable: ~5 km/px against SRTM's 30 m -- see "
              "export.verify_roundtrip)")

    print(f"\ndone in {time.perf_counter() - t_start:.0f} s")
    print(json.dumps({k: v for k, v in res.manifest.items()
                      if k in ("grid", "raster", "land_fraction", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
