"""Render an exported world from its GeoTIFFs.

Reads only the exported files, never the mesh -- which doubles as a check that the export
is self-sufficient.

Run:  .venv\\Scripts\\python.exe scripts\\render_world.py data/world_n785
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import biomes as bio  # noqa: E402


def hillshade(z, dx, azimuth=315.0, altitude=45.0, exaggeration=12.0):
    gy, gx = np.gradient(z * exaggeration, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    shade = (np.sin(alt) * np.cos(slope)
             + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(shade, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world", help="directory written by generate_world.py")
    ap.add_argument("--width", type=int, default=4096, help="output width")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.patches import Patch
    from rasterio.enums import Resampling

    d = pathlib.Path(args.world)
    manifest = json.loads((d / "manifest.json").read_text())
    h = args.width // 2

    with rasterio.open(d / "elevation.tif") as ds:
        z = ds.read(1, out_shape=(h, args.width), resampling=Resampling.average)
    with rasterio.open(d / "biomes.tif") as ds:
        b = ds.read(1, out_shape=(h, args.width), resampling=Resampling.nearest)

    land = b != bio.OCEAN
    shade = hillshade(z, dx=(40030.0 / args.width) * 1000.0)

    img = np.zeros((h, args.width, 3), dtype=np.float32)
    for code, colour in bio.COLOURS.items():
        m = b == code
        if m.any():
            img[m] = colour
    # Bathymetry keeps its own shading; land takes the hillshade.
    depth = np.clip(-z / 6000.0, 0, 1)
    img[~land] = np.stack([0.03 + 0.05 * (1 - depth), 0.10 + 0.26 * (1 - depth),
                           0.26 + 0.38 * (1 - depth)], axis=-1)[~land]
    img[land] *= (0.45 + 0.75 * shade)[land, None]
    img = np.clip(img, 0, 1)

    present = [c for c in bio.COLOURS if c != bio.OCEAN and (b == c).any()]
    fig, ax = plt.subplots(figsize=(args.width / 200, h / 200), dpi=200)
    ax.imshow(img, extent=(-180, 180, -90, 90), interpolation="nearest")
    ax.set_title(
        f"n={manifest['grid']['n']} · {manifest['grid']['cells']:,} cells · "
        f"land {manifest['land_fraction']:.1%} · peaks {z.max():,.0f} m · "
        f"per-basin θ {manifest['metrics'].get('basin_theta_median', float('nan')):.3f}",
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
