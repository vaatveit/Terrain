"""Stage 9: export.

ONE artifact. The per-cell field arrays plus the grid frequency and the seed. The grid is
fully regenerable from n alone, so there is no reason to store 650 MB of geometry alongside
the fields. This is the format to reload, to render, and to continue simulating from.

There used to be a second: equirectangular GeoTIFFs, justified on the grounds that real
DEMs arrive as GeoTIFF, so one metrics implementation would serve generated and real
terrain alike. That justification did not survive contact. `verify_roundtrip` -- which
existed to exercise the claim -- had to document that its numbers were NOT comparable to
the SRTM references, because a window wide enough to span three decades of drainage area
also spans tens of degrees of latitude, where equirectangular cells are badly distorted.
The world is judged by the mesh metrics, `terrain/metrics.py` run on the mesh graph. Real
DEMs still arrive as GeoTIFF and `terrain/dem.py` still reads them; nothing about that
needed the export to be one too.

The manifest records the parameters and the measured metrics next to the data. A world
whose provenance is lost is not reproducible even when the generator is deterministic.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np


def save_native(path, grid, fields: dict, seed: int, params: dict = None):
    """Field arrays plus the little that is needed to rebuild the grid."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"field_{k}": np.asarray(v) for k, v in fields.items()}
    payload["_n"] = np.int64(grid.n)
    payload["_radius_km"] = np.float64(grid.radius_km)
    payload["_seed"] = np.int64(seed)
    payload["_params_json"] = np.array(json.dumps(params or {}, default=str))
    np.savez_compressed(path, **payload)
    return path


def load_native(path, grid=None):
    """Reload field arrays and rebuild the grid from n. Returns (grid, fields, meta).

    Pass `grid` to supply one already in hand -- rebuilding at n=785 is minutes of work,
    and a caller with the cached grid on disk should not pay it twice.
    """
    from terrain.grid import build_goldberg

    d = np.load(path, allow_pickle=False)
    if grid is None:
        grid = build_goldberg(int(d["_n"]), radius_km=float(d["_radius_km"]))
    fields = {k[len("field_"):]: d[k] for k in d.files if k.startswith("field_")}
    meta = {"n": int(d["_n"]), "seed": int(d["_seed"]),
            "params": json.loads(str(d["_params_json"]))}
    return grid, fields, meta


@dataclasses.dataclass
class ExportResult:
    directory: pathlib.Path
    files: dict
    manifest: dict


def export_world(out_dir, grid, elevation, land_mask, seed, *, slope=None,
                 drainage_km2=None, temperature_c=None, precipitation_mm=None,
                 biome=None, flow_receiver=None, discharge=None, river_width_m=None,
                 lake_mask=None, lake_level_m=None, lake_depth_m=None, lake_id=None,
                 endorheic=None, lakes=None,
                 params=None, metrics_summary=None) -> ExportResult:
    """Write the native archive and the manifest."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    def as_flag(arr):
        # Masks travel as uint8, matching land_mask, so the archive carries one boolean
        # convention rather than two.
        return None if arr is None else np.asarray(arr).astype(np.uint8)

    fields = {"elevation": elevation, "land_mask": land_mask.astype(np.uint8)}
    for name, arr in (("temperature_c", temperature_c),
                      ("precipitation_mm", precipitation_mm), ("biome", biome),
                      ("slope", slope), ("drainage_km2", drainage_km2),
                      # The flow tree, as cell ids rather than edge slots. A slot would need
                      # a sentinel for the terminating case; `rec[i] == i` gets it free, and
                      # cell ids chain -- follow them and you walk downstream to the sea.
                      # The exit EDGE is `nbr[i] == rec[i]`, derived on load, not stored.
                      ("flow_receiver", flow_receiver),
                      ("discharge_km3_per_yr", discharge),
                      # No river_mask: it would be `river_width_m > 0` verbatim.
                      ("river_width_m", river_width_m),
                      ("lake_mask", as_flag(lake_mask)),
                      ("lake_level_m", lake_level_m),
                      ("lake_depth_m", lake_depth_m),
                      ("lake_id", lake_id),
                      ("endorheic", as_flag(endorheic))):
        if arr is not None:
            fields[name] = arr
    files["native"] = str(save_native(out_dir / "world.npz", grid, fields, seed, params))

    manifest = {
        "grid": {"n": int(grid.n), "cells": int(grid.cell_count),
                 "radius_km": float(grid.radius_km),
                 "mean_spacing_km": float(np.mean(
                     grid.arc_dist[grid.neighbours >= 0]))},
        "seed": int(seed),
        "land_fraction": float(grid.area[land_mask].sum() / grid.area.sum()),
        "elevation_m": {"min": float(elevation.min()), "max": float(elevation.max())},
        "params": params or {},
        "metrics": metrics_summary or {},
        "files": files,
    }
    if lakes is not None:
        # field_lake_id indexes this table. Without it the ids are bare labels and the
        # per-lake quantities -- spill point, inflow, whether it is terminal -- are lost.
        manifest["lakes"] = list(lakes)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    files["manifest"] = str(out_dir / "manifest.json")
    return ExportResult(out_dir, files, manifest)
