"""Run the throwaway Stage 1 and measure it.

The measurement is the point. Noise-only terrain is expected to FAIL the drainage metrics
-- that is the thesis of the whole survey ("noise gives you shape, erosion gives you
realism"), and this establishes the baseline number that Stage 2 has to beat. If the
generated slope-area concavity already looked like real terrain, the landscape evolution
model would have nothing to do.

Run:  .venv\\Scripts\\python.exe scripts\\run_stage1.py --n 256
"""

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import metrics, stage1  # noqa: E402
from terrain.grid import Grid, build_goldberg, radius_for_cell_km  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Reference values measured on real SRTM tiles; see terrain-algorithms.md.
SRTM_EAST = {"concavity_theta": 0.516, "hack_h": 0.543, "drainage_density": 0.744}


def load_or_build(n):
    cached = ROOT / "data" / f"grid_n{n}_r{radius_for_cell_km(n) * 1000:.0f}m.npz"
    if cached.exists():
        print(f"Loading {cached.name}...")
        return Grid.load(cached)
    print(f"Building grid n={n}...")
    g = build_goldberg(n, verbose=True)
    cached.parent.mkdir(parents=True, exist_ok=True)
    g.save(cached)
    return g


def render_biomes(grid, biome, path, width=2048, title=None):
    """Equirectangular biome map, coloured by the conventional scheme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.spatial import cKDTree

    from terrain import biomes as bio

    height = width // 2
    lon = np.radians(np.linspace(-180, 180, width, endpoint=False) + 180.0 / width)
    lat = np.radians(np.linspace(90, -90, height, endpoint=False) - 90.0 / height)
    lo, la = np.meshgrid(lon, lat)
    pts = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    _, idx = cKDTree(grid.xyz).query(pts.reshape(-1, 3), workers=-1)
    b = biome[idx].reshape(height, width)

    img = np.zeros((height, width, 3), dtype=np.float32)
    present = []
    for code, colour in bio.COLOURS.items():
        m = b == code
        if m.any():
            img[m] = colour
            if code != bio.OCEAN:
                present.append(code)

    fig, ax = plt.subplots(figsize=(width / 200, height / 200), dpi=200)
    ax.imshow(img, extent=(-180, 180, -90, 90), interpolation="nearest")
    ax.set_title(title or "Biomes (Whittaker)", fontsize=8)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(labelsize=6)
    ax.legend(handles=[Patch(facecolor=bio.COLOURS[c], label=bio.NAMES[c])
                       for c in present],
              loc="lower left", fontsize=5, ncol=4, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def render_equirect(grid, crust, path, width=2048, title=None):
    """Nearest-cell resample of the mesh onto an equirectangular image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    height = width // 2
    lon = np.radians(np.linspace(-180, 180, width, endpoint=False) + 180.0 / width)
    lat = np.radians(np.linspace(90, -90, height, endpoint=False) - 90.0 / height)
    lo, la = np.meshgrid(lon, lat)
    pts = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)

    tree = cKDTree(grid.xyz)
    _, idx = tree.query(pts.reshape(-1, 3), workers=-1)
    z = crust.elevation[idx].reshape(height, width)
    land = crust.land_mask[idx].reshape(height, width)

    img = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.clip((crust.sea_level_m - z) / 5000.0, 0, 1)
    img[..., 0] = 0.02 + 0.06 * (1 - depth)
    img[..., 1] = 0.10 + 0.30 * (1 - depth)
    img[..., 2] = 0.25 + 0.45 * (1 - depth)

    h = np.clip((z - crust.sea_level_m) / 3000.0, 0, 1)
    green = np.stack([0.20 + 0.55 * h, 0.42 + 0.35 * h, 0.18 + 0.45 * h], axis=-1)
    img[land] = green[land]

    fig, ax = plt.subplots(figsize=(width / 200, height / 200), dpi=200)
    ax.imshow(img, extent=(-180, 180, -90, 90), interpolation="nearest")
    ax.set_title(title or (f"Stage 1 (throwaway) — land fraction "
                           f"{crust.land_fraction:.1%}, sea level "
                           f"{crust.sea_level_m:,.0f} m"), fontsize=8)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grid = load_or_build(args.n)
    print(f"  {grid.cell_count:,} cells, mean spacing "
          f"{grid.arc_dist[grid.neighbours >= 0].mean():.2f} km")

    print("\nGenerating crust...")
    t0 = time.perf_counter()
    crust = stage1.generate_crust(grid, seed=args.seed, verbose=True)
    print(f"  generated in {time.perf_counter() - t0:.1f} s")

    z = crust.elevation
    print(f"\n  land fraction     {crust.land_fraction:.1%}  (target 29%)")
    print(f"  sea level         {crust.sea_level_m:,.0f} m")
    print(f"  elevation         {z.min():,.0f} .. {z.max():,.0f} m")
    print(f"  thickness         {crust.crust_thickness.min():.1f} .. "
          f"{crust.crust_thickness.max():.1f} km")

    iso = stage1.isostasy_check(crust, grid.area)
    print(f"  mean continental  {iso['mean_continental_m']:,.0f} m "
          f"(freeboard {iso['freeboard_m']:,.0f} m)")
    print(f"  mean abyssal      {iso['mean_abyssal_m']:,.0f} m")
    print(f"  gap               {iso['gap_km']:.2f} km   "
          f"(dry-column formula prints 4.45; not a physical target)")

    a = crust.landmass_areas
    total_land = a.sum()
    print(f"\n  landmasses kept   {a.size:,}")
    print(f"  largest           {a[0]:,.0f} km^2 ({a[0] / total_land:.1%} of land)")
    print(f"  top 5             {', '.join(f'{x:,.0f}' for x in a[:5])} km^2")
    print(f"  top 3 share       {a[:3].sum() / total_land:.1%}  "
          f"(want heavy-tailed: a flat distribution means too little low-frequency content)")

    # Metrics. Channel threshold scaled to the grid: ~6 cells of drainage.
    thr = float(np.median(grid.area) * 6)
    print(f"\nMetrics (channel threshold {thr:,.0f} km^2)...")
    t0 = time.perf_counter()
    r = metrics.report(
        crust.elevation, grid.neighbours, grid.arc_dist, grid.area,
        is_sink=~crust.land_mask, channel_threshold_km2=thr,
        label=f"stage1-n{args.n}",
    )
    print(f"  measured in {time.perf_counter() - t0:.1f} s")

    print(f"\n  {'metric':22s} {'generated':>12s} {'SRTM east':>12s}   verdict")
    for key, name, lo, hi in (
        ("concavity_theta", "slope-area theta", 0.40, 0.60),
        ("hack_h", "Hack exponent h", 0.50, 0.62),
    ):
        v = r[key]
        ok = np.isfinite(v) and lo <= v <= hi
        print(f"  {name:22s} {v:12.3f} {SRTM_EAST[key]:12.3f}   "
              f"{'inside' if ok else 'OUTSIDE'} {lo}-{hi}")
    # Drainage density is the one metric here that is NOT scale-free -- it is channel length
    # per unit area, so a 9.8 km mesh and a 30 m DEM are orders of magnitude apart by
    # construction. Shown for tracking across runs at fixed resolution, not for comparison.
    print(f"  {'drainage density*':22s} {r['drainage_density']:12.5f} "
          f"{SRTM_EAST['drainage_density']:12.3f}   *not scale-free; do not compare")
    print(f"  {'channel cells':22s} {r['channel_cells']:12,}")
    print(f"  {'hypsometric integral':22s} {r['hypsometric_integral']:12.3f}")
    print(f"  {'Horton Rb':22s} {r['horton_bifurcation_ratio']:12.3f}")

    if not args.no_render:
        out = args.out or str(ROOT / "data" / f"stage1_n{args.n}_seed{args.seed}.png")
        print(f"\nRendering {out}...")
        t0 = time.perf_counter()
        render_equirect(grid, crust, out)
        print(f"  rendered in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
