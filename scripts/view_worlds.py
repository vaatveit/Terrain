"""Compare generated worlds side by side on two 3D globes in the browser.

    .venv\\Scripts\\python.exe scripts\\view_worlds.py data/world_n256_s1_continents data/world_n256_s2_earth

Both worlds are optional: with none the page opens empty. The pickers list every world in this
repo's data/ and in the sibling Data folder (C:\\Projects\\Data); --data-dir adds more. With one
world the page opens full width. The page itself is viewer/ -- plain ES modules and three.js
from a CDN, no build step.

Modelled on BotWorld2's globe (C:\\Projects\\BotWorld2): the real Goldberg cells drawn as
geometry, each a 7-vertex fan whose corners are the Voronoi circumcentres. What differs is that
it is two globes rather than one game board, so:

  - Geometry depends only on the grid, and is exported once per grid and shared on the GPU by
    both globes. A world contributes only its corner elevations and per-cell fields.
  - Colour ranges are the UNION of the two worlds', so a colour means the same value on both.
  - Hovering either globe reads the same cell off both, and rings that place on both.

Exports are cached in data/viewer_cache/ -- never beside the world, so a shared Data folder is
only ever read -- and rebuilt whenever world.npz changes or EXPORT_VERSION does. One takes a
few seconds at n=256 and about a minute at n=785, where the cache is ~0.4 GB per world plus
~0.26 GB of geometry shared by every world on that grid.
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
CACHE_DIR = DATA_DIR / "viewer_cache"
EXPORT_VERSION = 2
POS_SCALE = 32767.0
# Cells per pass. A whole n=785 grid at once would hold several (6.2M, 6, 3) float64
# temporaries, ~0.9 GB each.
SLICE = 500_000

if sys.byteorder != "little":
    raise SystemExit("big-endian host: the binary format is read back as little-endian")


# --------------------------------------------------------------------------------------
# Geometry -- per grid, shared by every world on it
# --------------------------------------------------------------------------------------

def _next_slot(deg):
    """Index of the neighbour after slot k in azimuthal order; pentagons wrap at 5."""
    k = np.arange(6)
    return np.where(deg[:, None] == 5, (k + 1) % 5, (k + 1) % 6)


def _slices(cells):
    for lo in range(0, cells, SLICE):
        yield np.arange(lo, min(lo + SLICE, cells))


def voronoi_corners(xyz, neighbours, ids):
    """(m, 6, 3) polygon corners of `ids` in winding order: circumcentres of (centre, nb_k, nb_k+1).

    The neighbour table is already sorted by azimuth, so these come out in polygon order. A
    pentagon's slot 5 repeats slot 4, so every cell is a uniform 6-triangle fan and the
    duplicate degenerates to zero area.
    """
    nb = neighbours[ids]
    valid = nb >= 0
    deg = valid.sum(axis=1)
    p = xyz[np.where(valid, nb, 0)]
    p_next = np.take_along_axis(p, _next_slot(deg)[..., None].repeat(3, axis=2), axis=1)
    c = xyz[ids]
    cc = np.cross(p - c[:, None, :], p_next - c[:, None, :])
    norm = np.linalg.norm(cc, axis=2, keepdims=True)
    cc = np.divide(cc, norm, out=np.zeros_like(cc), where=norm > 0)
    flip = np.einsum("mkj,mj->mk", cc, c) < 0           # keep it on the cell's own side
    cc = np.where(flip[..., None], -cc, cc)
    pent = deg == 5
    cc[pent, 5] = cc[pent, 4]
    return cc


def corner_elevation(elevation, neighbours, ids, how):
    """Elevation at each corner of `ids`, reduced over the three cells that share it.

    Symmetric in the three cells, so all of them agree on a shared corner and the surface is
    watertight. 'mean' is the smooth globe; 'min' is flat tiles that bevel down to their
    lowest neighbour at the rim, so the wall between two tiles is geometry and not a hole.
    """
    nb = neighbours[ids]
    valid = nb >= 0
    deg = valid.sum(axis=1)
    e_self = elevation[ids]
    e_nb = np.where(valid, elevation[np.where(valid, nb, 0)], e_self[:, None])
    e_next = np.take_along_axis(e_nb, _next_slot(deg), axis=1)
    if how == "min":
        out = np.minimum(np.minimum(e_self[:, None], e_nb), e_next)
    else:
        out = (e_self[:, None] + e_nb + e_next) / 3.0
    pent = deg == 5
    out[pent, 5] = out[pent, 4]
    return out


def _main_checkout():
    """The main checkout's root: ROOT itself, or -- from a git worktree -- the one it hangs off."""
    dotgit = ROOT / ".git"
    if dotgit.is_file():
        gitdir = pathlib.Path(dotgit.read_text().split("gitdir:", 1)[1].strip())
        if gitdir.parent.name == "worktrees":
            return gitdir.parent.parent.parent
    return ROOT


def _grid_dirs():
    """This checkout's data/, then the main checkout's.

    generate_world.py caches each grid there, and the n=785 one is 0.67 GB and slow to
    rebuild, so a worktree borrows the main checkout's rather than building its own.
    """
    return [DATA_DIR] + ([_main_checkout() / "data"] if _main_checkout() != ROOT else [])


def load_grid(n, radius_km):
    name = f"grid_n{n}_r{radius_km * 1000:.0f}m.npz"
    for d in _grid_dirs():
        if (d / name).exists():
            return Grid.load(d / name)
    print(f"  rebuilding the n={n} grid (no cached {name})", flush=True)
    return build_goldberg(n, radius_km=radius_km, verbose=False)


def grid_key(n, radius_km):
    return f"n{n}_r{radius_km * 1000:.0f}m"


def export_geometry(grid, path):
    """int16 SNORM unit directions, (N, 7, 3): the centre, then the six corners."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        for ids in _slices(grid.xyz.shape[0]):
            dirs = np.empty((ids.size, 7, 3))
            dirs[:, 0] = grid.xyz[ids]
            dirs[:, 1:] = voronoi_corners(grid.xyz, grid.neighbours, ids)
            q = np.clip(np.rint(dirs * POS_SCALE), -POS_SCALE, POS_SCALE).astype("<i2")
            fh.write(q.tobytes())
    tmp.replace(path)


# --------------------------------------------------------------------------------------
# World export
# --------------------------------------------------------------------------------------

# Only the fields drawn here: an n=785 world.npz also carries ~0.6 GB of monthly climate.
FIELDS = ("_n", "_radius_km", "_seed", "field_elevation", "field_land_mask",
          "field_lake_mask", "field_river_width_m", "field_temperature_c",
          "field_precipitation_mm", "field_discharge_km3_per_yr", "field_uplift",
          "field_lake_level_m", "field_koppen", "field_biome", "field_plate_id")


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


def export_world(world_dir, out_dir, name):
    """Write <out_dir>/world.bin and meta.json for one world. Returns the meta dict."""
    with np.load(world_dir / "world.npz") as npz:
        f = {k: npz[k] for k in FIELDS if k in npz.files}
    man_path = world_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    n, radius_km = int(f["_n"]), float(f["_radius_km"])
    grid = load_grid(n, radius_km)
    key = grid_key(n, radius_km)
    geom = CACHE_DIR / f"geometry_{key}.bin"
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

    # The per-cell tables go out already padded to the square texture the page uploads them
    # as, so it can hand the bytes to WebGL without making a second, padded copy.
    tex_w = int(np.ceil(np.sqrt(cells)))

    def padded(a):
        out = np.zeros((tex_w * tex_w, a.shape[1]), a.dtype)
        out[:cells] = a
        return out

    sections = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "world.bin.tmp"
    with open(tmp, "wb") as fh:
        def start(section, count):
            fh.write(b"\0" * ((-fh.tell()) % 8))
            sections[section] = {"offset": fh.tell(), "count": int(count)}

        for how, section in (("mean", "elevSmooth"), ("min", "elevFlat")):
            start(section, cells * 7)
            for ids in _slices(cells):
                e = np.empty((ids.size, 7))
                e[:, 0] = z[ids]
                e[:, 1:] = corner_elevation(z, grid.neighbours, ids, how)
                fh.write(np.clip(np.rint(e), -32767, 32767).astype("<i2").tobytes())
        for section, arr in (("scalarsA", _f32(padded(scalars_a))),
                             ("scalarsB", _f32(padded(scalars_b))),
                             ("codes", padded(codes))):
            start(section, arr.size)
            fh.write(np.ascontiguousarray(arr).tobytes())
    tmp.replace(out_dir / "world.bin")

    width = river_w[np.isfinite(river_w) & (river_w > 0)]
    stamp = (world_dir / "world.npz").stat()
    meta = {
        "version": EXPORT_VERSION,
        "source": {"path": str(world_dir), "mtime_ns": stamp.st_mtime_ns,
                   "size": stamp.st_size},
        "name": name,
        "cells": cells,
        "texW": tex_w,
        "geometry": key,
        "sections": sections,
        "stats": world_stats(f, man, grid, land, areas),
        "ranges": {
            "elevation": _range(z),
            "landElevation": _range(z, land),
            "temperature": _range(scalars_a[:, 1], land),
            "precipitation": _range(scalars_a[:, 2], land, q=99.5),
            "discharge": _range(scalars_a[:, 3], land, q=99.9),
            "uplift": _range(scalars_b[:, 0], land),
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

def default_data_dirs():
    """This repo's data/, and the sibling Data folder where BotWorld's production worlds live."""
    return [d for d in (DATA_DIR, _main_checkout().parent / "Data") if d.is_dir()]


class Server(http.server.ThreadingHTTPServer):
    # http.server sets SO_REUSEADDR, which on Windows lets a second server bind a port that
    # is already serving -- silently, with the first still answering. Refuse instead.
    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True


def label_for(world_dir):
    """A world's name in the pickers: bare for this repo's data/, else `<folder>/<world>`.

    The folder prefix keeps two folders' same-named worlds apart, and says which one it is.
    """
    if world_dir.parent.resolve() == DATA_DIR.resolve():
        return world_dir.name
    return f"{world_dir.parent.name}/{world_dir.name}"


def cache_dir_for(label):
    return CACHE_DIR / "worlds" / "".join(c if c.isalnum() or c in "-_." else "_" for c in label)


class Catalogue:
    """World label -> directory, and the export cache in front of it."""

    def __init__(self, data_dirs, extra_dirs):
        self.data_dirs = data_dirs
        self.extra = extra_dirs
        self.lock = threading.Lock()

    def worlds(self):
        found = {}
        for root in self.data_dirs:
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "world.npz").exists():
                    found[label_for(d)] = d
        for d in self.extra:
            found.setdefault(label_for(d), d)
        return found

    def meta(self, name):
        world_dir = self.worlds().get(name)
        if world_dir is None:
            return None
        out = cache_dir_for(name)
        with self.lock:
            stamp = (world_dir / "world.npz").stat()
            try:
                meta = json.loads((out / "meta.json").read_text())
                fresh = (meta.get("version") == EXPORT_VERSION
                         and meta["source"] == {"path": str(world_dir),
                                                "mtime_ns": stamp.st_mtime_ns,
                                                "size": stamp.st_size}
                         and (out / "world.bin").exists())
            except (OSError, ValueError, KeyError):
                fresh = False
            if not fresh:
                print(f"exporting {name} ...", flush=True)
                meta = export_world(world_dir, out, name)
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
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(path, "rb") as fh:        # streamed: an n=785 world is ~0.4 GB
                while chunk := fh.read(1 << 22):
                    self.wfile.write(chunk)

        def do_GET(self):
            # Split before unquoting, so a `Data/world_42_r` label arrives as one segment.
            parts = [urllib.parse.unquote(p)
                     for p in urllib.parse.urlparse(self.path).path.split("/") if p]
            if parts[:1] != ["api"]:
                return super().do_GET()
            try:
                if parts == ["api", "worlds"]:
                    worlds = cat.worlds()
                    return self._json({
                        "worlds": [{"name": k, "group": str(v.parent)} for k, v in worlds.items()],
                        "defaults": defaults,
                    })
                if len(parts) == 4 and parts[1] == "world":
                    meta = cat.meta(parts[2])
                    if meta is None:
                        return self._json({"error": f"no world {parts[2]}"}, 404)
                    if parts[3] == "meta.json":
                        return self._json(meta)
                    if parts[3] == "world.bin":
                        return self._file(cache_dir_for(parts[2]) / "world.bin")
                if len(parts) == 3 and parts[1] == "geometry":
                    path = CACHE_DIR / f"geometry_{parts[2]}.bin"
                    if path.exists() and "/" not in parts[2] and ".." not in parts[2]:
                        return self._file(path)
                return self._json({"error": "not found"}, 404)
            except (ConnectionError, BrokenPipeError):
                return None                     # the page navigated away mid-download
            except Exception as exc:  # report to the page rather than dropping the socket
                import traceback
                traceback.print_exc()
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("worlds", nargs="*", help="up to two generate_world output directories")
    ap.add_argument("--data-dir", action="append", default=[], metavar="DIR",
                    help="another folder of worlds for the pickers (repeatable); this repo's "
                         "data/ and C:\\Projects\\Data are always listed when they exist")
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
    data_dirs = default_data_dirs()
    for extra in args.data_dir:
        p = pathlib.Path(extra).resolve()
        if not p.is_dir():
            ap.error(f"--data-dir {extra} is not a folder")
        data_dirs.append(p)
    cat = Catalogue(data_dirs, dirs)
    names = [label_for(d) for d in dirs]
    for name in names:          # export up front, so the page opens onto ready data
        cat.meta(name)
    if args.export_only:
        return

    print("worlds from: " + ", ".join(str(d) for d in data_dirs), flush=True)
    query = urllib.parse.urlencode(dict(zip("ab", names)))
    try:
        server = Server(("127.0.0.1", args.port), make_handler(cat, names))
    except OSError as exc:
        raise SystemExit(f"port {args.port} is taken ({exc.strerror}) -- is a viewer already "
                         f"running? Pass --port to use another.")
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
