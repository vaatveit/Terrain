"""Stage 8: sub-grid detail, added when sampling the mesh to a finer target.

The simulation is expensive, so it runs coarse and detail goes on afterwards. That only
means anything if there is a finer target to add detail to: the mesh is ~9.8 km at n=785
and the export raster is 8192x4096, which is ~4.9 km per pixel at the equator. Detail lives
in that gap.

Two separate problems, and it is worth keeping them apart.

INTERPOLATION. Nearest-cell sampling of a hex mesh renders the hexagons themselves --
flat-shaded cells with visible seams, which reads as a rendering bug rather than terrain.
Inverse-distance weighting over the three nearest cells removes that before any detail is
considered.

DETAIL. Fractal noise, but never uniform. The doc's rule is that detail should be modulated
by slope, curvature and drainage area -- rough on steep faces, smooth on valley floors --
and uniform noise over a landscape is the single most obvious tell of a procedural map. A
valley floor is where sediment collects; it is smooth because it is being filled, not
because nothing is happening there.

WHERE THIS MUST NOT BE APPLIED: back onto the simulation mesh. Detail added to the mesh
perturbs elevations, which changes steepest-descent receivers, which reorganises the
drainage network the previous five stages were built to get right. Measured, on the mesh at
a modest amplitude, this shifts slope-area concavity and Hack's exponent -- see
tests/test_amplify.py. Amplification is an output-side operation and the metrics belong on
the pre-amplification field.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit, prange

from terrain import noise


@dataclasses.dataclass
class AmplifyParams:
    # Detail amplitude in metres at the roughest ground. Kept well below the mesh's own
    # cell-to-cell relief so detail reads as texture rather than as new landforms.
    max_amplitude_m: float = 260.0
    # Slope at which detail reaches full strength (m/m). Gentle ground gets proportionally
    # less, which is what keeps plains from looking like crumpled paper.
    slope_reference: float = 0.06
    # Detail retained on perfectly flat ground. Without a floor the slope term drives
    # amplitude to exactly zero on plains, on valley floors AND across the whole sea bed
    # (base level is flat, so its slope is zero) -- measured 0 m of detail everywhere
    # except the steepest decile. Real plains have texture; they are not billiard tables.
    flat_floor: float = 0.18
    # Drainage area, km^2, over which detail decays toward zero. Alluvial floors are smooth
    # because sediment fills them. This must be a LARGE-RIVER area, not a few cells: at
    # 3000 km^2 it was roughly four cells of drainage at n=256 and smoothed the entire
    # channel network flat.
    valley_smoothing_km2: float = 60000.0
    # Sub-ocean detail is muted: the sea floor is blanketed in sediment.
    ocean_scale: float = 0.35
    octaves: int = 6
    frequency: float = 180.0
    lacunarity: float = 2.0
    gain: float = 0.5
    ridged: float = 0.35      # blend toward ridged noise, which reads as rock rather than blobs
    neighbours: int = 3       # cells used for inverse-distance interpolation


def detail_amplitude(slope, drainage_km2, land_mask, p: AmplifyParams = None):
    """Per-position detail amplitude, in metres.

    Rough where it is steep, smooth where water collects, muted under the sea.
    """
    p = p or AmplifyParams()
    s = np.clip(np.asarray(slope, dtype=np.float64) / p.slope_reference, 0.0, 1.0)
    s = p.flat_floor + (1.0 - p.flat_floor) * s
    valley = np.exp(-np.asarray(drainage_km2, dtype=np.float64) / p.valley_smoothing_km2)
    amp = p.max_amplitude_m * s * valley
    return np.where(np.asarray(land_mask), amp, amp * p.ocean_scale)


@njit(cache=True, parallel=True)
def _detail_noise(points, perm, grad, octaves, frequency, lacunarity, gain, ridged, out):
    """Multi-octave 3D noise, blended toward ridged, evaluated at unit-sphere positions."""
    for i in prange(points.shape[0]):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        total = 0.0
        amp = 1.0
        freq = frequency
        norm = 0.0
        for _ in range(octaves):
            v = noise.simplex3(x * freq, y * freq, z * freq, perm, grad)
            # Ridged: 1 - |v| gives sharp crests where plain fBm gives rounded blobs.
            r = 2.0 * (1.0 - abs(v)) - 1.0
            total += amp * ((1.0 - ridged) * v + ridged * r)
            norm += amp
            amp *= gain
            freq *= lacunarity
        out[i] = total / norm
    return out


def neighbour_weights(grid, points, k=3, tree=None):
    """One KD-tree query, reusable for every field sampled at the same points.

    Worth doing explicitly: an 8192x4096 export is 33.5M points, and querying separately
    for elevation, slope, drainage and the land mask would repeat the most expensive step
    four times over.
    """
    from scipy.spatial import cKDTree
    tree = tree or cKDTree(grid.xyz)
    dist, idx = tree.query(points, k=k, workers=-1)
    if k == 1:
        return idx, None
    w = 1.0 / np.maximum(dist, 1e-12) ** 2
    w /= w.sum(axis=1, keepdims=True)
    return idx, w


def apply_weights(field, idx, w):
    if w is None:
        return np.asarray(field)[idx]
    return (np.asarray(field, dtype=np.float64)[idx] * w).sum(axis=1)


def interpolate(grid, field, points, k=3, tree=None):
    """Inverse-distance interpolation of a mesh field onto arbitrary unit-sphere points.

    Nearest-cell sampling renders the hexagons; three-neighbour IDW does not.
    """
    idx, w = neighbour_weights(grid, points, k=k, tree=tree)
    return apply_weights(field, idx, w)


def amplify(points, base_elevation, amplitude, p: AmplifyParams = None, seed=0):
    """Add modulated fractal detail to an interpolated base field."""
    p = p or AmplifyParams()
    pts = np.ascontiguousarray(points, dtype=np.float64)
    out = np.empty(pts.shape[0], dtype=np.float64)
    _detail_noise(pts, noise.permutation(noise.sub_seed("detail", seed)), noise.GRAD3,
                  p.octaves, p.frequency, p.lacunarity, p.gain, p.ridged, out)
    return np.asarray(base_elevation, dtype=np.float64) + out * np.asarray(amplitude)


def equirect_points(width):
    """Unit vectors for the pixel centres of an equirectangular raster."""
    height = width // 2
    lon = np.radians(np.linspace(-180, 180, width, endpoint=False) + 180.0 / width)
    lat = np.radians(np.linspace(90, -90, height, endpoint=False) - 90.0 / height)
    lo, la = np.meshgrid(lon, lat)
    pts = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    return pts.reshape(-1, 3), (height, width)


def amplified_raster(grid, elevation, slope, drainage_km2, land_mask, width=8192,
                     p: AmplifyParams = None, seed=0, detail=True):
    """Sample the mesh onto an equirectangular raster, with sub-grid detail.

    Returns (elevation, land_mask) rasters of shape (width//2, width).
    """
    from scipy.spatial import cKDTree

    p = p or AmplifyParams()
    pts, shape = equirect_points(width)
    tree = cKDTree(grid.xyz)

    # One query, every field. The nearest cell is column 0 of the same result, so the
    # categorical land mask comes free rather than costing a second traversal.
    idx, w = neighbour_weights(grid, pts, k=p.neighbours, tree=tree)
    nearest = idx[:, 0] if w is not None else idx

    z = apply_weights(elevation, idx, w)
    land = np.asarray(land_mask)[nearest]
    if not detail:
        return z.reshape(shape).astype(np.float32), land.reshape(shape)

    s = apply_weights(slope, idx, w)
    a = apply_weights(drainage_km2, idx, w)
    amp = detail_amplitude(s, a, land, p)
    z = amplify(pts, z, amp, p, seed)
    return z.reshape(shape).astype(np.float32), land.reshape(shape)
