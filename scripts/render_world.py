"""Render an exported world from its native mesh archive.

Reads world.npz -- the per-cell fields -- and samples them onto an equirectangular raster
for display. The raster is a rendering detail that lives and dies inside this script; it is
not an export format. Sampling is Stage 8's job (terrain/amplify.py), which is what that
module is still for now that the export writes no rasters of its own.

Run:  .venv\\Scripts\\python.exe scripts\\render_world.py data/world_n785
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import amplify, biomes as bio, export  # noqa: E402
from terrain.grid import Grid, build_goldberg  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def hillshade(z, dx, azimuth=315.0, altitude=45.0, exaggeration=12.0):
    gy, gx = np.gradient(z * exaggeration, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    shade = (np.sin(alt) * np.cos(slope)
             + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(shade, 0, 1)


def load_grid(n, radius_km):
    """The cached grid if generate_world.py left one, otherwise rebuild it."""
    cached = ROOT / "data" / f"grid_n{n}_r{radius_km * 1000:.0f}m.npz"
    if cached.exists():
        return Grid.load(cached)
    return build_goldberg(n, radius_km=radius_km, verbose=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world", help="directory written by generate_world.py")
    ap.add_argument("--width", type=int, default=4096, help="output width")
    ap.add_argument("--resample", choices=amplify.RESAMPLE_MODES, default="nearest",
                    help="how the elevation is sampled onto the raster. 'nearest' "
                         "(default) blends nothing and carries mesh values unaltered; "
                         "'barycentric' is linear over the mesh triangles; 'c1' also "
                         "removes triangle facets from the hillshade and is worth it "
                         "above ~2 px per mesh cell, so pair it with a larger --width")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.spatial import cKDTree

    d = pathlib.Path(args.world)
    manifest = json.loads((d / "manifest.json").read_text())
    h = args.width // 2

    grid = load_grid(int(manifest["grid"]["n"]), float(manifest["grid"]["radius_km"]))
    _, fields, _ = export.load_native(d / "world.npz", grid=grid)

    # One KD-tree, two fields. Biome is categorical and is always nearest -- the average of
    # 'tundra' and 'savanna' is not a biome -- so it never takes the --resample path.
    pts, shape = amplify.equirect_points(args.width)
    tree = cKDTree(grid.xyz)
    nearest, _ = amplify.neighbour_weights(grid, pts, k=1, tree=tree)
    b = np.asarray(fields["biome"])[nearest].reshape(shape)
    z = (np.asarray(fields["elevation"])[nearest].reshape(shape)
         if args.resample == "nearest" else
         amplify.interpolate(grid, fields["elevation"], pts, mode=args.resample, tree=tree)
         .reshape(shape).astype(np.float32))

    land = b != bio.OCEAN
    # Metres per pixel at the equator, from THIS world's radius. It was hardcoded to
    # Earth's 40,030 km circumference, which after the rescale to 100 m cells overstated dx
    # by 98x -- every gradient came out 98x too shallow and the hillshade was flat.
    circumference_m = 2.0 * np.pi * float(manifest["grid"]["radius_km"]) * 1000.0
    shade = hillshade(z, dx=circumference_m / args.width)

    img = np.zeros((h, args.width, 3), dtype=np.float32)
    for code, colour in bio.COLOURS.items():
        m = b == code
        if m.any():
            img[m] = colour
    # Bathymetry keeps its own shading; land takes the hillshade. Normalised on the world's
    # own deepest point for the same reason as dx above -- a fixed 6000 m left every ocean
    # on a 1400 m-deep world the same flat blue.
    floor = min(float(manifest["elevation_m"]["min"]), -1.0)
    depth = np.clip(z / floor, 0, 1)
    img[~land] = np.stack([0.03 + 0.05 * (1 - depth), 0.10 + 0.26 * (1 - depth),
                           0.26 + 0.38 * (1 - depth)], axis=-1)[~land]
    img[land] *= (0.45 + 0.75 * shade)[land, None]
    img = np.clip(img, 0, 1)

    # The manifest stores null, not NaN, for a metric that came back non-finite -- so the
    # default in .get() never fires and only an explicit `or` catches it.
    theta = manifest["metrics"].get("basin_theta_median") or float("nan")

    present = [c for c in bio.COLOURS if c != bio.OCEAN and (b == c).any()]
    fig, ax = plt.subplots(figsize=(args.width / 200, h / 200), dpi=200)
    ax.imshow(img, extent=(-180, 180, -90, 90), interpolation="nearest")
    ax.set_title(
        f"n={manifest['grid']['n']} · {manifest['grid']['cells']:,} cells · "
        f"land {manifest['land_fraction']:.1%} · peaks {z.max():,.0f} m · "
        f"per-basin θ {theta:.3f}",
        fontsize=9)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(labelsize=6)
    ax.legend(handles=[Patch(facecolor=bio.COLOURS[c], label=bio.NAMES[c])
                       for c in present],
              loc="lower left", fontsize=5, ncol=4, framealpha=0.85)
    fig.tight_layout()
    out = d / "render.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
