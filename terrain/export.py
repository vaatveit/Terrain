"""Stage 9: export.

Two artifacts, for two different purposes.

NATIVE. The per-cell field arrays plus the grid frequency and the seed. The grid is fully
regenerable from n alone, so there is no reason to store 650 MB of geometry alongside the
fields. This is the format to reload for further simulation.

STATIC. An equirectangular GeoTIFF at 8192x4096, which is the natural match for 4096 cells
around the equator. GeoTIFF rather than .npy for one specific reason: real DEMs arrive as
GeoTIFF, so the metrics module runs unmodified against generated and real terrain alike.
That claim is worth more than the convenience -- it is what makes the SRTM references in
terrain-algorithms.md comparable at all -- so `verify_roundtrip` below actually exercises
it rather than taking it on trust.

The manifest records the parameters and the measured metrics next to the data. A world
whose provenance is lost is not reproducible even when the generator is deterministic.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np


def _transform(width, height):
    from rasterio.transform import from_origin
    return from_origin(-180.0, 90.0, 360.0 / width, 180.0 / height)


def write_geotiff(path, bands, dtype, nodata=None, descriptions=None):
    """Write an equirectangular GeoTIFF in EPSG:4326.

    `bands` is a single 2-D array or a list of them. All bands share a dtype, which is a
    GeoTIFF constraint -- hence separate files for elevation (float32) and biome (uint8)
    rather than one mixed stack.
    """
    import rasterio

    if isinstance(bands, np.ndarray) and bands.ndim == 2:
        bands = [bands]
    height, width = bands[0].shape
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": len(bands),
        "dtype": dtype, "crs": "EPSG:4326", "transform": _transform(width, height),
        "compress": "deflate", "predictor": 2 if dtype.startswith("float") else 1,
        "tiled": True, "blockxsize": 512, "blockysize": 512,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as ds:
        for i, b in enumerate(bands, start=1):
            ds.write(np.asarray(b, dtype=dtype), i)
            if descriptions:
                ds.set_band_description(i, descriptions[i - 1])
    return path


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


def load_native(path):
    """Reload field arrays and rebuild the grid from n. Returns (grid, fields, meta)."""
    from terrain.grid import build_goldberg

    d = np.load(path, allow_pickle=False)
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
                 biome=None, discharge=None, river_mask=None, river_width_m=None,
                 lake_mask=None, lake_level_m=None, lake_depth_m=None, lake_id=None,
                 endorheic=None, lakes=None,
                 width=8192, params=None, metrics_summary=None,
                 amplify_params=None, detail=True) -> ExportResult:
    """Write the native archive, the rasters, and the manifest."""
    from terrain import amplify as amp_mod

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    mode = getattr(amplify_params, "resample", None) or amp_mod.AmplifyParams().resample

    def as_flag(arr):
        # Masks travel as uint8, matching land_mask, so the archive carries one boolean
        # convention rather than two.
        return None if arr is None else np.asarray(arr).astype(np.uint8)

    fields = {"elevation": elevation, "land_mask": land_mask.astype(np.uint8)}
    for name, arr in (("temperature_c", temperature_c),
                      ("precipitation_mm", precipitation_mm), ("biome", biome),
                      ("slope", slope), ("drainage_km2", drainage_km2),
                      ("discharge_km3_per_yr", discharge),
                      ("river_mask", as_flag(river_mask)),
                      ("river_width_m", river_width_m),
                      ("lake_mask", as_flag(lake_mask)),
                      ("lake_level_m", lake_level_m),
                      ("lake_depth_m", lake_depth_m),
                      ("lake_id", lake_id),
                      ("endorheic", as_flag(endorheic))):
        if arr is not None:
            fields[name] = arr
    files["native"] = str(save_native(out_dir / "world.npz", grid, fields, seed, params))

    # Elevation raster, with sub-grid detail if slope and drainage are available.
    can_detail = detail and slope is not None and drainage_km2 is not None
    if can_detail:
        z_r, land_r = amp_mod.amplified_raster(grid, elevation, slope, drainage_km2,
                                               land_mask, width=width, p=amplify_params,
                                               seed=seed, detail=True)
    else:
        pts, shape = amp_mod.equirect_points(width)
        z_r = (amp_mod.interpolate(grid, elevation, pts, mode=mode)
               .reshape(shape).astype(np.float32))
        # Nearest for the mask: land/ocean is categorical and must not be averaged.
        near, _ = amp_mod.neighbour_weights(grid, pts, k=1)
        land_r = np.asarray(land_mask, dtype=bool)[near].reshape(shape)
    files["elevation"] = str(write_geotiff(out_dir / "elevation.tif", z_r, "float32",
                                           descriptions=["elevation_m"]))

    if (temperature_c is not None and precipitation_mm is not None) or biome is not None:
        pts, shape = amp_mod.equirect_points(width)
        if mode == "nearest":
            nearest, _ = amp_mod.neighbour_weights(grid, pts, k=1)
            idx = w = None
        else:
            idx, w = amp_mod.barycentric_weights(grid, pts)
            nearest = idx[:, 0]
        if temperature_c is not None and precipitation_mm is not None:
            def sample(f):
                return (np.asarray(f)[nearest] if idx is None
                        else amp_mod.apply_weights(f, idx, w))
            t_r = sample(temperature_c).reshape(shape)
            p_r = sample(precipitation_mm).reshape(shape)
            files["climate"] = str(write_geotiff(
                out_dir / "climate.tif", [t_r, p_r], "float32",
                descriptions=["temperature_c", "precipitation_mm"]))
        if biome is not None:
            # Nearest, not interpolated: averaging categorical codes is meaningless.
            b_r = np.asarray(biome)[nearest].reshape(shape)
            files["biomes"] = str(write_geotiff(out_dir / "biomes.tif", b_r, "uint8",
                                                descriptions=["biome_code"]))

    manifest = {
        "grid": {"n": int(grid.n), "cells": int(grid.cell_count),
                 "radius_km": float(grid.radius_km),
                 "mean_spacing_km": float(np.mean(
                     grid.arc_dist[grid.neighbours >= 0]))},
        "seed": int(seed),
        "raster": {"width": width, "height": width // 2, "crs": "EPSG:4326",
                   "degrees_per_pixel": 360.0 / width, "amplified": bool(can_detail)},
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


def best_land_window(tif_path, size, max_abs_lat=55.0, sea_level=0.0):
    """Pick a window that looks like an SRTM tile: mostly land, well away from the poles.

    Two things make a naive window useless. Taking the tallest-relief window can land on
    the ice caps, and any window tall enough to reach a pole is measuring a coordinate
    singularity rather than terrain -- equirectangular cells collapse to zero width there
    and the D8 graph across the pole is meaningless. Real DEM tiles are regional and
    mid-latitude, so the comparison should be too.
    """
    import rasterio

    with rasterio.open(tif_path) as ds:
        h, w = ds.height, ds.width
        size = int(min(size, h, w))
        probe = max(1, size // 32)
        coarse = ds.read(1, out_shape=(max(1, h // probe), max(1, w // probe)),
                         resampling=rasterio.enums.Resampling.average)
        top, dlat = ds.bounds.top, (ds.bounds.top - ds.bounds.bottom) / h

    ch, cw = coarse.shape
    step = max(1, size // probe)
    best, best_score = (0, 0), -1.0
    for r in range(0, max(1, ch - step), max(1, step // 4)):
        row0 = min(r * probe, h - size)
        lat_hi = top - row0 * dlat
        lat_lo = top - (row0 + size) * dlat
        if max(abs(lat_hi), abs(lat_lo)) > max_abs_lat:
            continue
        for c in range(0, max(1, cw - step), max(1, step // 4)):
            block = coarse[r:r + step, c:c + step]
            if block.size == 0:
                continue
            land = float((block > sea_level).mean())
            relief = float(block[block > sea_level].std()) if (block > sea_level).any() else 0.0
            score = land * (1.0 + relief / 1000.0)
            if score > best_score:
                best_score = score
                best = (row0, min(c * probe, w - size))
    return best


def verify_fidelity(tif_path, grid, elevation, samples=200_000, seed=0, mode="nearest"):
    """Does the exported raster carry the world faithfully? This is the export's own test.

    Compares raster pixels against the mesh field sampled at the same coordinates. Any
    disagreement beyond the sub-grid detail deliberately added in Stage 8 means the
    georeferencing, the resampling or the write is wrong.

    `mode` must match what the export used, or this measures the difference between two
    resampling schemes rather than the fidelity of the write.
    """
    import rasterio
    from terrain import amplify as amp_mod

    with rasterio.open(tif_path) as ds:
        z_r = ds.read(1)
        h, w = ds.height, ds.width
        b = ds.bounds

    rng = np.random.default_rng(seed)
    rows = rng.integers(0, h, samples)
    cols = rng.integers(0, w, samples)
    lat = b.top - (rows + 0.5) * (b.top - b.bottom) / h
    lon = b.left + (cols + 0.5) * (b.right - b.left) / w
    la, lo = np.radians(lat), np.radians(lon)
    pts = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)

    expect = amp_mod.interpolate(grid, elevation, pts, mode=mode)
    got = z_r[rows, cols].astype(np.float64)
    err = got - expect
    return {"samples": int(samples), "mean_abs_error_m": float(np.abs(err).mean()),
            "p99_abs_error_m": float(np.percentile(np.abs(err), 99)),
            "max_abs_error_m": float(np.abs(err).max()),
            "bias_m": float(err.mean()),
            "relief_m": float(expect.max() - expect.min())}


def verify_roundtrip(tif_path, window=1024, channel_threshold_km2=None, sea_level=0.0):
    """Run the metrics module on the exported GeoTIFF, by the same route SRTM tiles take.

    What this proves and what it does not. It proves the export is readable by the metrics
    code with no changes -- which is the reason the survey chose GeoTIFF over .npy in the
    first place. It does NOT produce numbers comparable to the SRTM references, and that is
    a resolution gap rather than a defect: the export runs about 5 km per pixel against
    SRTM's 30 m, so any window wide enough to span three decades of drainage area also
    spans tens of degrees of latitude, where equirectangular cells are badly distorted.
    Judge the world by the mesh metrics; use this to confirm the file is well-formed.

    Two details are still load-bearing, and getting either wrong gave a concavity of -0.081
    -- slope rising with drainage area, which is not a landscape. The window must avoid the
    poles, and the sea must be declared as base level, or rivers incise across the abyssal
    plain.
    """
    from terrain import dem, metrics

    row0, col0 = best_land_window(str(tif_path), window, sea_level=sea_level)
    win = dem.load_window(str(tif_path), size=window, row0=row0, col0=col0,
                          name=pathlib.Path(tif_path).stem, sink_below=sea_level)
    thr = channel_threshold_km2 or float(np.median(win.area) * 400)
    return metrics.report(win.z, win.neighbours, win.arc_dist, win.area, win.is_sink,
                          is_border=win.is_border, channel_threshold_km2=thr,
                          label="exported-geotiff")
