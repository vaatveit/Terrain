"""Compare generated worlds side by side on two 3D globes in the browser.

    .venv\\Scripts\\python.exe scripts\\view_worlds.py data/world_n256_s1_continents data/world_n256_s2_earth

Both worlds are optional: with none the page opens empty and every world under data/ is in
its pickers; with one it opens full width. The page itself is viewer/ -- plain ES modules and
three.js from a CDN, no build step.

Modelled on BotWorld2's globe (C:\\Projects\\BotWorld2): the real Goldberg cells drawn as
geometry, each a 7-vertex fan whose corners are the Voronoi circumcentres. What differs is that
it is two globes rather than one game board, so:

  - Geometry depends only on the grid, and is exported once per grid and shared on the GPU by
    both globes. A world contributes only its corner elevations and per-cell fields.
  - Colour ranges are the UNION of the two worlds', so a colour means the same value on both.
  - Hovering either globe reads the same cell off both, and rings that place on both.

An export is cached beside its world.npz, in <world>/viewer/, and rebuilt whenever world.npz
changes or this file's EXPORT_VERSION does. It takes a few seconds at n=256.
"""

from __future__ import annotations

import argparse
import http.server
import json
import pathlib
import sys
import threading
import urllib.parse
import webbrowser

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terrain import biomes, koppen, plates, stage1  # noqa: E402
from terrain.grid import Grid, build_goldberg  # noqa: E402

VIEWER_DIR = ROOT / "viewer"
DATA_DIR = ROOT / "data"
EXPORT_VERSION = 1
POS_SCALE = 32767.0

if sys.byteorder != "little":
    raise SystemExit("big-endian host: the binary format is read back as little-endian")


# --------------------------------------------------------------------------------------
# Geometry -- per grid, shared by every world on it
# --------------------------------------------------------------------------------------

def _next_slot(deg):
    """Index of the neighbour after slot k in azimuthal order; pentagons wrap at 5."""
    k = np.arange(6)
    return np.where(deg[:, None] == 5, (k + 1) % 5, (k + 1) % 6)


def voronoi_corners(xyz, neighbours):
    """(N, 6, 3) polygon corners in winding order: circumcentres of (centre, nb_k, nb_k+1).

    The neighbour table is already sorted by azimuth, so these come out in polygon order. A
    pentagon's slot 5 repeats slot 4, so every cell is a uniform 6-triangle fan and the
    duplicate degenerates to zero area.
    """
    valid = neighbours >= 0
    deg = valid.sum(axis=1)
    p = xyz[np.where(valid, neighbours, 0)]
    p_next = np.take_along_axis(p, _next_slot(deg)[..., None].repeat(3, axis=2), axis=1)
    c = xyz[:, None, :]
    cc = np.cross(p - c, p_next - c)
    norm = np.linalg.norm(cc, axis=2, keepdims=True)
    cc = np.divide(cc, norm, out=np.zeros_like(cc), where=norm > 0)
    flip = np.einsum("mkj,mj->mk", cc, xyz) < 0          # keep it on the cell's own side
    cc = np.where(flip[..., None], -cc, cc)
    pent = deg == 5
    cc[pent, 5] = cc[pent, 4]
    return cc


def corner_elevation(elevation, neighbours, how):
    """Elevation at each corner, reduced over the three cells that share it.

    Symmetric in the three cells, so all of them agree on a shared corner and the surface is
    watertight. 'mean' is the smooth globe; 'min' is flat tiles that bevel down to their
    lowest neighbour at the rim, so the wall between two tiles is geometry and not a hole.
    """
    valid = neighbours >= 0
    deg = valid.sum(axis=1)
    e_self = elevation
    e_nb = np.where(valid, elevation[np.where(valid, neighbours, 0)], e_self[:, None])
    e_next = np.take_along_axis(e_nb, _next_slot(deg), axis=1)
    if how == "min":
        out = np.minimum(np.minimum(e_self[:, None], e_nb), e_next)
    else:
        out = (e_self[:, None] + e_nb + e_next) / 3.0
    pent = deg == 5
    out[pent, 5] = out[pent, 4]
    return out


def load_grid(n, radius_km):
    cached = DATA_DIR / f"grid_n{n}_r{radius_km * 1000:.0f}m.npz"
    if cached.exists():
        return Grid.load(cached)
    print(f"  rebuilding the n={n} grid (no cache at {cached.name})", flush=True)
    return build_goldberg(n, radius_km=radius_km, verbose=False)


def grid_key(n, radius_km):
    return f"n{n}_r{radius_km * 1000:.0f}m"


def export_geometry(grid, path):
    """int16 SNORM unit directions, (N, 7, 3): the centre, then the six corners."""
    dirs = np.empty((grid.xyz.shape[0], 7, 3))
    dirs[:, 0] = grid.xyz
    dirs[:, 1:] = voronoi_corners(grid.xyz, grid.neighbours)
    q = np.clip(np.rint(dirs * POS_SCALE), -POS_SCALE, POS_SCALE).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(q.tobytes())
    tmp.replace(path)


# --------------------------------------------------------------------------------------
# World export
# --------------------------------------------------------------------------------------

def _palette(module):
    return [{"code": int(c), "name": module.NAMES[c],
             "color": [round(float(v), 4) for v in module.COLOURS[c]]}
            for c in sorted(module.NAMES)]


def _f32(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0,
                         neginf=0.0).astype("<f4")


def landmass_ranks(land, grid):
    """Per cell, 1 + the size rank of its landmass (1 = largest), 0 at sea; plus the areas."""
    nbr = np.ascontiguousarray(grid.neighbours)
    lab = np.empty(land.size, np.int32)
    stack = np.empty(land.size, np.int32)
    k = stage1.label_components(land, nbr, lab, stack)
    area = np.zeros(k)
    np.add.at(area, lab[land], grid.area[land])
    order = np.argsort(-area)
    rank = np.empty(k, np.int32)
    rank[order] = np.arange(1, k + 1)
    out = np.zeros(land.size, np.float32)
    out[land] = rank[lab[land]]
    return out, area[order]


def world_stats(f, man, grid, land, areas):
    z = f["field_elevation"]
    tot = float(areas.sum())
    land_area = grid.area[land]
    params = man.get("params", {})
    belt = None
    if "plates" in params:
        belt = params["plates"]["orogen_decay_km"] / plates.PlateParams().orogen_decay_km
    metrics = man.get("metrics", {})
    return {
        "seed": int(f["_seed"]),
        "n": int(f["_n"]),
        "radiusKm": float(f["_radius_km"]),
        "preset": params.get("preset", "earth"),
        "plateCount": params.get("plates", {}).get("plate_count"),
        "beltWidth": belt,
        "landFraction": float(land_area.sum() / grid.area.sum()),
        "landmasses": int(areas.size),
        "largeLandmasses": int((areas >= 0.05 * tot).sum()) if tot else 0,
        "largestShare": float(areas[0] / tot) if tot else 0.0,
        "landShares": [float(a / tot) for a in areas[:64]] if tot else [],
        "peakM": float(z[land].max()) if land.any() else 0.0,
        "above1000": float(land_area[z[land] > 1000].sum() / tot) if tot else 0.0,
        "above300": float(land_area[z[land] > 300].sum() / tot) if tot else 0.0,
        "hackH": metrics.get("hack_h"),
        "basinTheta": metrics.get("basin_theta_median"),
    }


def _range(a, mask=None, q=None):
    """[min, max], or [min, q-th percentile] for a field with a long tail.

    Precipitation is the case: a few orographic cells reach ~90 m/yr against a median of a few
    hundred mm, and a ramp stretched to them leaves every other cell at its dry end.
    """
    a = a if mask is None else a[mask]
    a = a[np.isfinite(a)]
    if not a.size:
        return [0.0, 1.0]
    return [float(a.min()), float(a.max() if q is None else np.percentile(a, q))]


def export_world(world_dir, out_dir):
    """Write <out_dir>/world.bin and meta.json for one world. Returns the meta dict."""
    f = dict(np.load(world_dir / "world.npz"))
    man_path = world_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    n, radius_km = int(f["_n"]), float(f["_radius_km"])
    grid = load_grid(n, radius_km)
    key = grid_key(n, radius_km)
    geom = DATA_DIR / "viewer_cache" / f"geometry_{key}.bin"
    if not geom.exists():
        print(f"  exporting geometry for grid {key}", flush=True)
        export_geometry(grid, geom)

    cells = grid.xyz.shape[0]
    z = f["field_elevation"].astype(np.float64)
    land = f["field_land_mask"].astype(bool)
    lake = f.get("field_lake_mask", np.zeros(cells, np.uint8)).astype(bool)
    river_w = f.get("field_river_width_m", np.zeros(cells, np.float32))
    zero = np.zeros(cells, np.float32)
    rank, areas = landmass_ranks(land, grid)

    smooth = np.empty((cells, 7))
    smooth[:, 0] = z
    smooth[:, 1:] = corner_elevation(z, grid.neighbours, "mean")
    flat = np.empty((cells, 7))
    flat[:, 0] = z
    flat[:, 1:] = corner_elevation(z, grid.neighbours, "min")

    scalars_a = np.stack([z, f.get("field_temperature_c", zero),
                          f.get("field_precipitation_mm", zero),
                          f.get("field_discharge_km3_per_yr", zero)], axis=1)
    scalars_b = np.stack([f.get("field_uplift", zero), river_w,
                          np.where(lake, f.get("field_lake_level_m", zero), 0.0),
                          rank], axis=1)
    flags = (land.astype(np.uint8) | (lake.astype(np.uint8) << 1)
             | ((np.nan_to_num(river_w) > 0).astype(np.uint8) << 2))
    k = f.get("field_koppen", np.full(cells, -1, np.int8)).astype(np.int16) + 1
    codes = np.stack([f.get("field_biome", np.zeros(cells, np.uint8)), flags,
                      k.astype(np.uint8),
                      (f.get("field_plate_id", np.zeros(cells, np.int16)) & 0xFF)
                      .astype(np.uint8)], axis=1).astype(np.uint8)

    sections = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "world.bin.tmp"
    with open(tmp, "wb") as fh:
        def put(name, arr):
            fh.write(b"\0" * ((-fh.tell()) % 8))
            sections[name] = {"offset": fh.tell(), "count": int(arr.size)}
            fh.write(np.ascontiguousarray(arr).tobytes())
        clip = lambda e: np.clip(np.rint(e), -32767, 32767).astype("<i2")  # noqa: E731
        put("elevSmooth", clip(smooth))
        put("elevFlat", clip(flat))
        put("scalarsA", _f32(scalars_a))
        put("scalarsB", _f32(scalars_b))
        put("codes", codes)
    tmp.replace(out_dir / "world.bin")

    uplift = scalars_b[:, 0]
    width = river_w[np.isfinite(river_w) & (river_w > 0)]
    stamp = (world_dir / "world.npz").stat()
    meta = {
        "version": EXPORT_VERSION,
        "source": {"mtime_ns": stamp.st_mtime_ns, "size": stamp.st_size},
        "name": world_dir.name,
        "cells": cells,
        "geometry": key,
        "sections": sections,
        "stats": world_stats(f, man, grid, land, areas),
        "ranges": {
            "elevation": _range(z),
            "landElevation": _range(z, land),
            "temperature": _range(scalars_a[:, 1], land),
            "precipitation": _range(scalars_a[:, 2], land, q=99.5),
            "discharge": _range(scalars_a[:, 3], land, q=99.9),
            "uplift": _range(uplift, land),
            "riverWidth": [float(width.min()), float(width.max())] if width.size else [1, 1],
        },
        "plates": int(f["field_plate_id"].max()) + 1 if "field_plate_id" in f else 0,
        "biomes": _palette(biomes),
        "koppen": _palette(koppen),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


# --------------------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------------------

class Catalogue:
    """World name -> directory, and the export cache in front of it."""

    def __init__(self, extra_dirs):
        self.extra = {d.name: d for d in extra_dirs}
        self.lock = threading.Lock()

    def worlds(self):
        found = {}
        if DATA_DIR.exists():
            for d in sorted(DATA_DIR.iterdir()):
                if (d / "world.npz").exists():
                    found[d.name] = d
        found.update(self.extra)
        return found

    def meta(self, name):
        world_dir = self.worlds().get(name)
        if world_dir is None:
            return None
        out = world_dir / "viewer"
        with self.lock:
            stamp = (world_dir / "world.npz").stat()
            try:
                meta = json.loads((out / "meta.json").read_text())
                fresh = (meta.get("version") == EXPORT_VERSION
                         and meta["source"] == {"mtime_ns": stamp.st_mtime_ns,
                                                "size": stamp.st_size}
                         and (out / "world.bin").exists())
            except (OSError, ValueError, KeyError):
                fresh = False
            if not fresh:
                print(f"exporting {name} ...", flush=True)
                meta = export_world(world_dir, out)
                print(f"  done: {name}", flush=True)
        return meta


def make_handler(cat, defaults):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(VIEWER_DIR), **kw)

        def log_message(self, fmt, *args):  # quiet: only errors
            if args and str(args[1])[:1] in "45":
                super().log_message(fmt, *args)

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path):
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parts = [urllib.parse.unquote(p)
                     for p in urllib.parse.urlparse(self.path).path.split("/") if p]
            if parts[:1] != ["api"]:
                return super().do_GET()
            try:
                if parts == ["api", "worlds"]:
                    return self._json({"worlds": list(cat.worlds()), "defaults": defaults})
                if len(parts) == 4 and parts[1] == "world":
                    meta = cat.meta(parts[2])
                    if meta is None:
                        return self._json({"error": f"no world {parts[2]}"}, 404)
                    if parts[3] == "meta.json":
                        return self._json(meta)
                    if parts[3] == "world.bin":
                        return self._file(cat.worlds()[parts[2]] / "viewer" / "world.bin")
                if len(parts) == 3 and parts[1] == "geometry":
                    path = DATA_DIR / "viewer_cache" / f"geometry_{parts[2]}.bin"
                    if path.exists() and "/" not in parts[2] and ".." not in parts[2]:
                        return self._file(path)
                return self._json({"error": "not found"}, 404)
            except Exception as exc:  # report to the page rather than dropping the socket
                import traceback
                traceback.print_exc()
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("worlds", nargs="*", help="up to two generate_world output directories")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ap.add_argument("--export-only", action="store_true",
                    help="build the export caches for the given worlds and exit")
    args = ap.parse_args()
    if len(args.worlds) > 2:
        ap.error("at most two worlds")

    dirs = [pathlib.Path(w).resolve() for w in args.worlds]
    for d in dirs:
        if not (d / "world.npz").exists():
            ap.error(f"{d} has no world.npz")
    cat = Catalogue(dirs)
    names = [d.name for d in dirs]
    for name in names:          # export up front, so the page opens onto ready data
        cat.meta(name)
    if args.export_only:
        return

    query = urllib.parse.urlencode(dict(zip("ab", names)))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port),
                                             make_handler(cat, names))
    url = f"http://127.0.0.1:{args.port}/" + (f"?{query}" if query else "")
    print(f"viewer at {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
