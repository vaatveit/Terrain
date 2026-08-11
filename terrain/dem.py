"""Adapter turning a lat/lon DEM raster into the same cell graph the Goldberg grid produces.

This is what lets one metrics implementation run against both generated worlds and real
SRTM tiles. The geometry is spherical throughout -- cell areas shrink as cos(lat) and all
distances are great-circle arcs, so a tile at 48N is not silently treated as a plane.

A full SRTM tile is too large to hold an explicit D8 graph for (the east tile is 127M
cells, which would need ~4 GB just for the neighbour table), so work on a window. That is
also standard practice in geomorphology, and the metrics that matter here -- Hack's
exponent, slope-area concavity -- are scaling exponents and therefore scale-free.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import rasterio

from terrain.grid import R_EARTH_KM

# D8 offsets as (drow, dcol). Row 0 is the northern edge (rasterio origin is top-left).
D8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclasses.dataclass
class DemWindow:
    name: str
    z: np.ndarray          # (N,) f32 m
    shape: tuple           # (H, W)
    lat: np.ndarray        # (H,) f64 degrees, row centres
    lon: np.ndarray        # (W,) f64 degrees, column centres
    neighbours: np.ndarray  # (N, k) i32
    arc_dist: np.ndarray   # (N, k) f32 km
    area: np.ndarray       # (N,) f32 km^2
    is_sink: np.ndarray    # (N,) bool -- window border, where flow leaves
    is_border: np.ndarray  # (N,) bool -- same cells; basins touching these are clipped

    @property
    def cell_count(self):
        return self.z.shape[0]


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def best_window(path, size, probe=16):
    """Pick the window with the most relief.

    A centred window is a poor default -- on the west tile it lands in Puget Sound. Relief
    is what the erosion metrics need, so search coarsely for the roughest region.
    """
    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        if h <= size or w <= size:
            return 0, 0
        coarse = ds.read(
            1, out_shape=(max(1, h // probe), max(1, w // probe)),
            resampling=rasterio.enums.Resampling.average,
        ).astype(np.float32)
        nodata = ds.nodata

    if nodata is not None:
        coarse[coarse == nodata] = np.nan

    step = max(1, size // probe)
    ch, cw = coarse.shape
    best, best_score = (0, 0), -np.inf
    for r in range(0, max(1, ch - step), max(1, step // 4)):
        for c in range(0, max(1, cw - step), max(1, step // 4)):
            block = coarse[r:r + step, c:c + step]
            if block.size == 0 or np.all(np.isnan(block)):
                continue
            score = np.nanstd(block)
            if score > best_score:
                best_score = score
                best = (min(r * probe, h - size), min(c * probe, w - size))
    return best


def load_window(path, size=2048, row0=None, col0=None, connectivity=8, name=None):
    """Read a square window and build its spherical cell graph."""
    if row0 is None or col0 is None:
        row0, col0 = best_window(path, size)

    with rasterio.open(path) as ds:
        size = min(size, ds.height, ds.width)
        row0 = int(np.clip(row0, 0, ds.height - size))
        col0 = int(np.clip(col0, 0, ds.width - size))
        win = rasterio.windows.Window(col0, row0, size, size)
        z = ds.read(1, window=win).astype(np.float32)
        nodata = ds.nodata
        transform = ds.window_transform(win)

    h, w = z.shape
    dlon = float(transform.a)
    dlat = float(-transform.e)  # positive; rows run north -> south
    lon_left = float(transform.c)
    lat_top = float(transform.f)

    lat = lat_top - (np.arange(h) + 0.5) * dlat
    lon = lon_left + (np.arange(w) + 0.5) * dlon

    # Spherical cell area: R^2 * dlon * (sin(lat_north) - sin(lat_south)). Row-dependent only.
    lat_n = np.radians(lat_top - np.arange(h) * dlat)
    lat_s = np.radians(lat_top - (np.arange(h) + 1) * dlat)
    row_area = (R_EARTH_KM**2 * np.radians(dlon) * (np.sin(lat_n) - np.sin(lat_s))).astype(np.float32)
    area = np.repeat(row_area, w)

    offsets = D8 if connectivity == 8 else D4
    k = len(offsets)
    n = h * w

    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    nbr = np.full((h, w, k), -1, dtype=np.int32)
    dist_rows = np.zeros((h, k), dtype=np.float32)

    for s, (dr, dc) in enumerate(offsets):
        r2, c2 = rows + dr, cols + dc
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        nbr[..., s] = np.where(ok, r2 * w + c2, -1)
        # Distance depends only on the row and the offset, not the column.
        rr = np.clip(np.arange(h) + dr, 0, h - 1)
        dist_rows[:, s] = _haversine_km(lat, 0.0, lat[rr], dc * dlon).astype(np.float32)

    arc = np.broadcast_to(dist_rows[:, None, :], (h, w, k)).reshape(n, k).copy()
    nbr = nbr.reshape(n, k)

    z = np.ascontiguousarray(z.reshape(n), dtype=np.float32)

    is_border = np.zeros((h, w), dtype=bool)
    is_border[0, :] = is_border[-1, :] = True
    is_border[:, 0] = is_border[:, -1] = True
    is_border = is_border.reshape(n)

    is_sink = is_border.copy()
    if nodata is not None:
        bad = z == np.float32(nodata)
        if bad.any():
            is_sink |= bad
            z[bad] = np.float32(0.0)

    return DemWindow(
        name=name or f"{path.split(chr(92))[-1]}[{row0},{col0}+{size}]",
        z=z, shape=(h, w), lat=lat, lon=lon, neighbours=nbr,
        arc_dist=arc, area=area.astype(np.float32),
        is_sink=is_sink, is_border=is_border,
    )
