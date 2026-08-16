"""Run the whole pipeline and export a world.

Stages 0-9 end to end: grid, tectonic crust, landscape evolution coupled with climate,
glacial erosion, hydrology, biomes, export. The output is the native mesh archive;
scripts/render_world.py turns it into a picture.

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
    biomes, climate, coupled, export, flow, glacial, hydrology, koppen, metrics, plates,
    stage1, stage2,
)
from terrain.grid import Grid, build_goldberg, radius_for_cell_km  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRTM_EAST = {"concavity_theta": 0.516, "hack_h": 0.543}


def load_or_build(n):
    # The radius is in the filename because it is no longer Earth's: a cached grid built
    # before the rescale has the same n and a 98x larger sphere, and would be loaded
    # silently. Everything downstream would then be tuned for the wrong world.
    cached = ROOT / "data" / f"grid_n{n}_r{radius_for_cell_km(n) * 1000:.0f}m.npz"
    if cached.exists():
        return Grid.load(cached)
    g = build_goldberg(n, verbose=True)
    cached.parent.mkdir(parents=True, exist_ok=True)
    g.save(cached)
    return g


def main():
    crust_defaults = stage1.CrustParams()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--target-land-fraction", type=float,
                    default=crust_defaults.target_land_fraction,
                    help="share of the globe above sea level; sea level is solved to hit it "
                         "(default %(default)s)")
    ap.add_argument("--continental-fraction", type=float,
                    default=crust_defaults.continental_fraction,
                    help="share of SURFACE that is continental crust; the excess drowns as "
                         "shelf (default %(default)s)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-glacial", action="store_true")
    args = ap.parse_args()

    crust_params = stage1.CrustParams(
        target_land_fraction=args.target_land_fraction,
        continental_fraction=args.continental_fraction)

    out_dir = pathlib.Path(args.out or (ROOT / "data" / f"world_n{args.n}_s{args.seed}"))
    t_start = time.perf_counter()

    print(f"[0] grid n={args.n}")
    grid = load_or_build(args.n)
    print(f"    {grid.cell_count:,} cells, "
          f"{np.mean(grid.arc_dist[grid.neighbours >= 0]):.2f} km spacing")

    print("[1] tectonic crust")
    crust = stage1.generate_crust_tectonic(grid, crust_params, seed=args.seed)
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

    # The coastline is not a constant of the pipeline. Erosion -- glacial erosion above
    # all, which cuts below its own outlet -- leaves ground below the datum that is joined
    # to the sea, and it has to be let in or those are dry fjords.
    print("[5b] reflood")
    labels = np.empty(grid.cell_count, dtype=np.int32)
    stack = np.empty(grid.cell_count, dtype=np.int32)
    land = stage1.reflood(z, np.ascontiguousarray(grid.neighbours), grid.area,
                          labels, stack)
    is_ocean = ~land
    print(f"    land {crust.land_fraction:.1%} -> "
          f"{grid.area[land].sum() / grid.area.sum():.1%}")
    # Margins follow the coastline, so they have to be re-derived on the one that ships for
    # the same reason the climate below does. The plates themselves are untouched -- boundary
    # type and distance are geometry, not elevation -- so this is just the coastal test again.
    margin = plates.classify_margins(grid, land, crust.plates)

    # Re-solve the climate on the coastline that actually ships. Everything below reads
    # absolute millimetres and degrees against the new mask, and the old fields were solved
    # for a continent that is now part sea -- coast distance, the ocean moisture source and
    # the sea-surface anomaly all move when a fifth of the land floods. This does NOT feed
    # back into erosion: Stages 2 and 5 have run, so it is the reported climate that is
    # being made consistent, not the landscape. Stage 5 therefore still sees the pre-reflood
    # temperature, which is correct -- that is the coastline the ice grew on.
    # The rivers have to be re-routed before that solve, not carried over. River evaporation
    # reads the discharge field, and both of its inputs moved: glacial erosion recut the
    # valleys and the reflood moved the outlets. Solving here without one at all would be
    # worse still -- it would export an ocean-only climate onto terrain that was eroded with
    # rivers, which is the drier world the whole of item 2c was about.
    print("[5c] climate on the new coastline")
    q = coupled.routed_discharge(grid, z, cres.discharge_weight, is_ocean)
    clim = climate.compute_seasons(grid, z, land, discharge=q)
    print(f"    land mean {clim.temperature_c[land].mean():.1f} C, "
          f"precip median "
          f"{climate._area_median(clim.precipitation_mm, grid.area, land):,.0f} mm")

    print("[6] hydrology")
    hyd = hydrology.analyse(grid, z, is_ocean, clim.precipitation_mm, clim.temperature_c)
    endo = sum(1 for L in hyd.lakes if L["endorheic"])
    print(f"    {len(hyd.lakes)} lakes ({endo} endorheic), "
          f"{(hyd.river_width_m > 0).sum():,} river cells, "
          f"max discharge {hyd.discharge.max():,.3g} km^3/yr")

    print("[7] biomes")
    biome = biomes.classify(clim.temperature_c, clim.precipitation_mm, land, z, grid.lat,
                            lake_mask=hyd.lake_mask)
    shares = biomes.summary(biome, grid.area, land)
    print("    " + ", ".join(f"{n} {s:.0%}" for n, s in shares[:4]))

    kop = None
    if clim.temperature_season is not None:
        kop = koppen.classify(clim.temperature_season, clim.precipitation_season, land,
                              grid.lat, annual_precip_mm=clim.precipitation_mm)
        print("    koppen: " + ", ".join(
            f"{n.split()[0]} {s:.0%}" for n, s in koppen.summary(kop, grid.area, land)[:5]))

    print("[9] metrics on the mesh")
    fr = flow.route(z, grid.neighbours, grid.arc_dist, grid.area, is_ocean)
    # 40 cells of drainage, not 6. The threshold has to sit ABOVE the channel head or the
    # slope-area regression is fitted across the hillslope limb as well as the fluvial one,
    # and theta comes out meaningless. At 9.77 km cells this could not happen -- 6 cells was
    # 500 km^2, far above any channel head -- but at 100 m cells 6 cells IS the head.
    #
    # Re-measured on the shipped n=785 world, seed 42, exported mask:
    #
    #     cells      6      13      20      40      80     120     400
    #     theta  0.285   0.289   0.280   0.271   0.265   0.310   0.203
    #     r2     0.919   0.907   0.888   0.848   0.809   0.891   0.613
    #
    # 40 is kept, but the reason has changed and it is worth being honest about: theta is
    # FLAT at 0.27-0.31 across every sensible threshold here, so the choice barely matters
    # and 40 is no longer buying what it was chosen to buy. This comment used to record
    # 0.392 / 0.467 / 0.556 at 6 / 40 / 400 with r2 0.904, which said theta RISES with the
    # threshold. It falls. Those figures predate the climate work -- precipitation feeds
    # discharge weight, which feeds erosion -- and no longer describe anything this code
    # produces. Hack h does still respond, peaking at 0.564 near 80-120 cells against the
    # SRTM east reference of 0.543, and reading 0.557 at 40.
    thr = float(np.median(grid.area) * 40)
    m = metrics.report(z, grid.neighbours, grid.arc_dist, grid.area, is_sink=is_ocean,
                       channel_threshold_km2=thr, label="mesh")
    summary = {k: (float(m[k]) if np.isfinite(m[k]) else None) for k in
               ("basin_theta_median", "concavity_theta", "hack_h",
                "hypsometric_integral", "horton_bifurcation_ratio")}
    print(f"    per-basin theta {m['basin_theta_median']:.3f} "
          f"(SRTM east {SRTM_EAST['concavity_theta']}), "
          f"Hack h {m['hack_h']:.3f} (SRTM east {SRTM_EAST['hack_h']})")

    print(f"[9] export -> {out_dir}")
    params = {
        # The one that RAN, not a fresh default -- the two now differ whenever a crust flag
        # is passed, and the provenance file is the only record of which world this is.
        "crust": crust_params.__dict__, "plates": plates.PlateParams().__dict__,
        "lem": stage2.LemParams().__dict__, "climate": climate.ClimateParams().__dict__,
        "glacial": None if args.no_glacial else glacial.GlacialParams().__dict__,
        "hydrology": hydrology.HydrologyParams().__dict__,
        "steps": args.steps, "passes": args.passes,
    }
    res = export.export_world(
        out_dir, grid, z, land, args.seed, slope=fr.slope, drainage_km2=fr.acc,
        temperature_c=clim.temperature_c, precipitation_mm=clim.precipitation_mm,
        biome=biome, koppen=kop, flow_receiver=fr.rec, discharge=hyd.discharge,
        river_width_m=hyd.river_width_m, margin_class=margin,
        lake_mask=hyd.lake_mask,
        lake_level_m=hyd.lake_level_m, lake_depth_m=hyd.lake_depth_m,
        lake_id=hyd.lake_id, endorheic=hyd.endorheic, lakes=hyd.lakes,
        params=params, metrics_summary=summary)
    for name, path in res.files.items():
        size = pathlib.Path(path).stat().st_size / 1e6
        print(f"    {name:10s} {size:8.1f} MB  {pathlib.Path(path).name}")

    print(f"\ndone in {time.perf_counter() - t_start:.0f} s")
    print(json.dumps({k: v for k, v in res.manifest.items()
                      if k in ("grid", "land_fraction", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
