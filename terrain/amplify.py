"""Stage 8: sampling the mesh onto a raster, and the sub-grid detail that used to need.

Two separate problems, and it is worth keeping them apart -- because the rescale to 100 m
cells killed one of them and left the other more useful than ever.

WHAT THE DETAIL HALF WAS FOR, AND WHY IT IS NOW OFF. The simulation is expensive, so it ran
coarse and detail went on afterwards. That only means anything if there is a finer target
to add detail to. At Earth radius the mesh was ~9.8 km at n=785 against ~4.9 km pixels, and
detail lived in that gap. At 100 m cells the gap is inverted: the render raster is ~1.5 px
per cell, `octaves_for_width` caps the noise at ONE octave, and that octave has a ~2.3 km
wavelength -- 23 cells across. That is not sub-grid detail, it is a coarse ripple laid over
ground the landscape evolution already resolved properly. So nothing calls `amplify` with
`detail=True` any more; the machinery is kept because it is correct and because a future
export at 4x the mesh resolution would want it back.

WHAT THE SAMPLING HALF IS FOR. Unchanged and load-bearing: `scripts/render_world.py` reads
the native mesh archive and has to get it onto a raster to draw it. The `interpolate` modes
below are that path.

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
    # Detail has to stay under the mesh's OWN cell-to-cell relief or it reads as new
    # landforms. The invariant is the drop across one cell at `slope_reference`, not the
    # world's total relief: that drop was 9770 m * 0.06 = 586 m at Earth radius and is
    # 100 m * 0.06 = 6 m here, so 260 m (44% of the drop) becomes 3 m.
    max_amplitude_m: float = 3.0
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
    #
    # It is a FRACTION OF THE SPHERE, not an absolute area, because "large river" is
    # defined against the size of the continent rather than against the cell. The old
    # hardcoded 60000 km^2 hid this: at Earth radius it was 19 cells of drainage at n=128
    # and 725 at n=785, i.e. it was already tracking sphere area and only looked constant
    # because the sphere was. 60000 / (4 pi 6371^2) = 1.18e-4, which reproduces the tuned
    # behaviour at every radius and every n -- 6.2 km^2 on the production world.
    valley_smoothing_sphere_fraction: float = 1.18e-4
    # Sub-ocean detail is muted: the sea floor is blanketed in sediment.
    ocean_scale: float = 0.35
    # Octave ceiling. The number ACTUALLY used is capped against the raster resolution by
    # `octaves_for_width` -- detail finer than the pixel cannot be represented and comes
    # back as aliasing, not as texture.
    octaves: int = 6
    frequency: float = 180.0
    lacunarity: float = 2.0
    gain: float = 0.5
    ridged: float = 0.35      # blend toward ridged noise, which reads as rock rather than blobs
    # Minimum pixels per feature of the finest octave kept. 2 is the Nyquist limit and is
    # not enough in practice -- simplex features are not sinusoids and the raster is not
    # band-limited before sampling. 3 leaves a margin.
    min_px_per_feature: float = 3.0
    # How mesh fields are sampled onto the raster. See `interpolate`.
    #   "nearest"     each pixel takes the value of the cell it falls in. The default:
    #                 it is the mesh data, unaltered, and blends nothing.
    #   "barycentric" linear over the mesh triangles. Exact at cell centres and for a
    #                 linear field, but only C0 -- its gradient jumps at triangle edges.
    #   "c1"          C1 Shepard blend of per-cell linear polynomials, for elevation.
    resample: str = "nearest"


def octaves_for_width(width, p: AmplifyParams = None):
    """How many detail octaves the raster can actually carry.

    Detail below the pixel size is not detail. Sampling it produces aliasing: energy folded
    back to low frequencies as long, straight, evenly spaced moire fringes -- which read as
    ruled lines across the terrain, exactly the artifact detail is meant to avoid.

    At the shipped 6 octaves from a base frequency of 180, the finest octave has features
    of 1.1 km against a ~4.9 km pixel at width 8192. Measured against an 8x supersampled
    reference of the same field, 56% of the sampled signal was aliasing rather than signal.

    Capping is the whole fix, and it costs nothing real: the octaves being dropped were
    never representable. The honest consequence is that a coarse raster gets less detail,
    which is correct -- if more is wanted, raise --width.
    """
    p = p or AmplifyParams()
    px_rad = 2.0 * np.pi / max(int(width), 1)
    limit = 1.0 / (p.min_px_per_feature * px_rad)   # highest representable frequency
    n = 1
    while n < p.octaves and p.frequency * p.lacunarity ** n <= limit:
        n += 1
    return n


def detail_amplitude(slope, drainage_km2, land_mask, p: AmplifyParams = None,
                     sphere_km2=None):
    """Per-position detail amplitude, in metres.

    Rough where it is steep, smooth where water collects, muted under the sea.

    `sphere_km2` sets the valley-smoothing scale; pass `grid.area.sum()`. It defaults to
    the Earth-radius sphere purely so the bare three-argument call still means what it
    used to.
    """
    p = p or AmplifyParams()
    if sphere_km2 is None:
        from terrain.grid import R_EARTH_KM
        sphere_km2 = 4.0 * np.pi * R_EARTH_KM ** 2
    smoothing = p.valley_smoothing_sphere_fraction * float(sphere_km2)
    s = np.clip(np.asarray(slope, dtype=np.float64) / p.slope_reference, 0.0, 1.0)
    s = p.flat_floor + (1.0 - p.flat_floor) * s
    valley = np.exp(-np.asarray(drainage_km2, dtype=np.float64) / smoothing)
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

    Inverse-distance weighting, kept for nearest-neighbour lookups (k=1) and for callers
    that want it. Do NOT use it to resample elevation -- see `barycentric_weights`.
    """
    from scipy.spatial import cKDTree
    tree = tree or cKDTree(grid.xyz)
    dist, idx = tree.query(points, k=k, workers=-1)
    if k == 1:
        return idx, None
    w = 1.0 / np.maximum(dist, 1e-12) ** 2
    w /= w.sum(axis=1, keepdims=True)
    return idx, w


@njit(cache=True, parallel=True)
def _barycentric(points, nearest, xyz, nbr, out_idx, out_w):
    """Locate each point in the triangle fan around its nearest cell and weight linearly."""
    for q in prange(points.shape[0]):
        i = nearest[q]
        px, py, pz = points[q, 0], points[q, 1], points[q, 2]
        ax, ay, az = xyz[i, 0], xyz[i, 1], xyz[i, 2]

        deg = 0
        for s in range(nbr.shape[1]):
            if nbr[i, s] >= 0:
                deg += 1

        # Fall back to the cell's own value; overwritten if a triangle claims the point.
        out_idx[q, 0] = i
        out_idx[q, 1] = i
        out_idx[q, 2] = i
        out_w[q, 0] = 1.0
        out_w[q, 1] = 0.0
        out_w[q, 2] = 0.0

        for s in range(deg):
            j = nbr[i, s]
            k = nbr[i, (s + 1) % deg]
            if j < 0 or k < 0:
                continue
            bx, by, bz = xyz[j, 0], xyz[j, 1], xyz[j, 2]
            cx, cy, cz = xyz[k, 0], xyz[k, 1], xyz[k, 2]

            # Cramer's rule on [A B C] w = p, using scalar triple products.
            nx, ny, nz = by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx
            det = ax * nx + ay * ny + az * nz
            if det == 0.0:
                continue
            wa = (px * nx + py * ny + pz * nz) / det
            nx, ny, nz = cy * az - cz * ay, cz * ax - cx * az, cx * ay - cy * ax
            wb = (px * nx + py * ny + pz * nz) / det
            nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
            wc = (px * nx + py * ny + pz * nz) / det

            if wa >= -1e-9 and wb >= -1e-9 and wc >= -1e-9:
                tot = wa + wb + wc
                if tot > 0.0:
                    out_idx[q, 0] = i
                    out_idx[q, 1] = j
                    out_idx[q, 2] = k
                    out_w[q, 0] = wa / tot
                    out_w[q, 1] = wb / tot
                    out_w[q, 2] = wc / tot
                break


def barycentric_weights(grid, points, tree=None):
    """Linear interpolation over the mesh's own triangles. Use this for elevation.

    Inverse-distance weighting is what the export used to do, and it manufactures terrain
    that is not there. Shepard's method has ZERO GRADIENT at every data point, so a field
    sampled onto a finer raster acquires a dimple at each cell centre -- and hillshading,
    being a derivative, turns those dimples into a pervasive mesh-scale crosshatch. The
    effect is invisible in the elevations and glaring in the render.

    Measured on a perfectly smooth analytic field (a low-order spherical harmonic, no
    ridges and no noise) resampled onto the export raster, mean |second difference| along
    a row -- which is what hillshading amplifies:

        IDW k=3 (was)      0.345 m      visible crosshatch over the whole patch
        IDW k=12           0.274 m      a wider stencil does not help; the flaw is the kernel
        barycentric        0.0002 m     nothing visible

    Barycentric is also exact at cell centres and adds no blur, which a smoothing kernel
    would. The returned (idx, w) has the same shape as `neighbour_weights`, with the
    nearest cell in column 0, so `apply_weights` and the categorical `idx[:, 0]` lookup
    both work unchanged.
    """
    from scipy.spatial import cKDTree
    tree = tree or cKDTree(grid.xyz)
    _, nearest = tree.query(points, k=1, workers=-1)
    pts = np.ascontiguousarray(points, dtype=np.float64)
    idx = np.empty((pts.shape[0], 3), dtype=np.int32)
    w = np.empty((pts.shape[0], 3), dtype=np.float64)
    _barycentric(pts, np.ascontiguousarray(nearest, dtype=np.int64),
                 np.ascontiguousarray(grid.xyz, dtype=np.float64),
                 np.ascontiguousarray(grid.neighbours), idx, w)
    return idx, w


def apply_weights(field, idx, w):
    if w is None:
        return np.asarray(field)[idx]
    return (np.asarray(field, dtype=np.float64)[idx] * w).sum(axis=1)


@njit(cache=True, parallel=True)
def _cell_gradients(xyz, nbr, field, out):
    """Least-squares gradient of a field in each cell's own tangent plane.

    Coordinates are tangent-projected chords on the unit sphere, so the gradient is in
    field units per unit chord and needs no radius to use.
    """
    n = xyz.shape[0]
    kk = nbr.shape[1]
    for i in prange(n):
        ax, ay, az = xyz[i, 0], xyz[i, 1], xyz[i, 2]

        # Tangent basis: start from whichever axis is least aligned with the normal.
        if abs(ax) <= abs(ay) and abs(ax) <= abs(az):
            tx, ty, tz = 1.0, 0.0, 0.0
        elif abs(ay) <= abs(az):
            tx, ty, tz = 0.0, 1.0, 0.0
        else:
            tx, ty, tz = 0.0, 0.0, 1.0
        d = ax * tx + ay * ty + az * tz
        e1x, e1y, e1z = tx - ax * d, ty - ay * d, tz - az * d
        nrm = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        if nrm < 1e-12:
            out[i, 0] = out[i, 1] = out[i, 2] = 0.0
            continue
        e1x, e1y, e1z = e1x / nrm, e1y / nrm, e1z / nrm
        e2x = ay * e1z - az * e1y
        e2y = az * e1x - ax * e1z
        e2z = ax * e1y - ay * e1x

        suu = suv = svv = sdu = sdv = 0.0
        for s in range(kk):
            j = nbr[i, s]
            if j < 0:
                continue
            dvx, dvy, dvz = xyz[j, 0] - ax, xyz[j, 1] - ay, xyz[j, 2] - az
            dot = ax * dvx + ay * dvy + az * dvz
            dtx, dty, dtz = dvx - ax * dot, dvy - ay * dot, dvz - az * dot
            u = dtx * e1x + dty * e1y + dtz * e1z
            v = dtx * e2x + dty * e2y + dtz * e2z
            dz = field[j] - field[i]
            suu += u * u
            suv += u * v
            svv += v * v
            sdu += dz * u
            sdv += dz * v

        det = suu * svv - suv * suv
        if abs(det) < 1e-20:
            out[i, 0] = out[i, 1] = out[i, 2] = 0.0
        else:
            ca = (sdu * svv - sdv * suv) / det
            cb = (sdv * suu - sdu * suv) / det
            out[i, 0] = ca * e1x + cb * e2x
            out[i, 1] = ca * e1y + cb * e2y
            out[i, 2] = ca * e1z + cb * e2z


@njit(cache=True, parallel=True)
def _c1_blend(points, idx, dist, field, grad, out):
    """Shepard blend of per-cell LINEAR polynomials, compactly supported."""
    m, k = idx.shape
    for q in prange(m):
        px, py, pz = points[q, 0], points[q, 1], points[q, 2]
        r = dist[q, k - 1]
        acc = 0.0
        wsum = 0.0
        hit = -1
        lo = np.inf
        hi = -np.inf
        for s in range(k):
            d = dist[q, s]
            if d <= 1e-12:
                hit = idx[q, s]
                break
            if d >= r:
                continue
            i = idx[q, s]
            fi = field[i]
            if fi < lo:
                lo = fi
            if fi > hi:
                hi = fi
            # grad is tangent at xyz[i], so grad . (p - xyz[i]) == grad . p.
            val = fi + grad[i, 0] * px + grad[i, 1] * py + grad[i, 2] * pz
            t = (r - d) / (r * d)
            w = t * t
            acc += w * val
            wsum += w
        if hit >= 0:
            out[q] = field[hit]
        elif wsum > 0.0:
            v = acc / wsum
            # Limiter. Extrapolating gradients across a near-discontinuity overshoots the
            # way a spline does: at the continental shelf break, where the mesh field is
            # almost a step, it undershot the abyssal plain by 471 m at n=128. Clamping to
            # the range of the contributing cells costs nothing in smooth ground -- where
            # the blend is already inside it and the interpolant stays C1 -- and stops the
            # export inventing relief the simulation never produced.
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            out[q] = v
        else:
            out[q] = field[idx[q, 0]]


RESAMPLE_MODES = ("nearest", "barycentric", "c1")


def interpolate(grid, field, points, mode="nearest", k=12, tree=None, chunk=2_000_000):
    """Sample a mesh field onto arbitrary unit-sphere points.

    DEFAULT -- `mode="nearest"`. Each point takes the value of the cell it falls in. No
    blending of any kind: the raster carries mesh values unaltered, every pixel traceable
    to one cell, and nothing in the output was manufactured on the way out. The visible
    cost is that the raster renders the hexagons, which at the default width is roughly one
    cell per pixel anyway. Both alternatives below are strictly more processing, so they
    are opt-in.

    `mode="barycentric"` -- linear over the mesh's own triangles. Exact at cell centres,
    exact for a linear field, never overshoots. Use it when smooth contours matter more
    than one-pixel-one-cell traceability. Note what it is NOT: the inverse-distance
    weighting this replaced had zero gradient at every data point, so it put a dimple at
    each cell centre and a mesh-scale crosshatch across every hillshade -- 0.345 m of
    manufactured curvature on a perfectly smooth analytic control, against 0.0007 m here.

    `mode="c1"` -- Shepard blend of per-cell LINEAR POLYNOMIALS, for elevation. Barycentric
    is linear within each triangle, so the surface is continuous but its GRADIENT is
    constant per triangle and jumps at every edge; hillshading is a gradient operator, so
    it draws those edges as a herringbone on steep ground. Rendering the same ground at 4x
    the raster width made the pattern BIGGER rather than finer -- pinned to the mesh, not
    the pixel -- which is what identified it, and is why exporting wider does not help.

    Blending `z_i + g_i . (p - x_i)`, with a least-squares gradient fitted in each cell's
    tangent plane, makes the interpolant reproduce `g_i` at cell i. That buys C1 everywhere
    while keeping exact linear precision and adding no blur: on a mountain window the facet
    tail (p99.5 / median of the second difference) falls from 8808 to 20.1, and the raster
    peak returns to the full mesh peak of 10,365 m where barycentric loses 113 m. A
    smoothing kernel at sigma = 0.85 cell spacings only halves the facet curvature and costs
    528 m of peak. It pays only once the raster resolves the mesh, so pair it with a large
    `--width`.

    The C1 weight is compactly supported, `((R - d)/(R d))^2` with R the distance to the
    k-th neighbour. A plain 1/d^2 would be discontinuous wherever the k-nearest SET changes,
    which would put creases back in; this vanishes to second order at d = R, so a neighbour
    entering or leaving the stencil changes nothing. Chunked because the k-nearest query is
    the memory peak: 20M raster points at k=12 is 1.9 GB for the distances alone.
    """
    from scipy.spatial import cKDTree

    if mode not in RESAMPLE_MODES:
        raise ValueError(f"unknown resample mode {mode!r}; expected one of {RESAMPLE_MODES}")
    tree = tree or cKDTree(grid.xyz)
    if mode == "nearest":
        idx, _ = neighbour_weights(grid, points, k=1, tree=tree)
        return np.asarray(field)[idx]
    if mode == "barycentric":
        idx, w = barycentric_weights(grid, points, tree=tree)
        return apply_weights(field, idx, w)

    xyz = np.ascontiguousarray(grid.xyz, dtype=np.float64)
    f = np.ascontiguousarray(field, dtype=np.float64)
    grad = np.empty((f.shape[0], 3), dtype=np.float64)
    _cell_gradients(xyz, np.ascontiguousarray(grid.neighbours), f, grad)

    pts = np.ascontiguousarray(points, dtype=np.float64)
    out = np.empty(pts.shape[0], dtype=np.float64)
    k = min(int(k), f.shape[0])
    for lo in range(0, pts.shape[0], chunk):
        hi = min(lo + chunk, pts.shape[0])
        block = pts[lo:hi]
        dist, idx = tree.query(block, k=k, workers=-1)
        _c1_blend(np.ascontiguousarray(block),
                  np.ascontiguousarray(idx.astype(np.int64)),
                  np.ascontiguousarray(dist, dtype=np.float64), f, grad, out[lo:hi])
    return out


def amplify(points, base_elevation, amplitude, p: AmplifyParams = None, seed=0,
            octaves=None):
    """Add modulated fractal detail to an interpolated base field.

    `octaves` defaults to the full ceiling; `amplified_raster` passes the resolution-capped
    count from `octaves_for_width`.
    """
    p = p or AmplifyParams()
    pts = np.ascontiguousarray(points, dtype=np.float64)
    out = np.empty(pts.shape[0], dtype=np.float64)
    _detail_noise(pts, noise.permutation(noise.sub_seed("detail", seed)), noise.GRAD3,
                  int(octaves or p.octaves), p.frequency, p.lacunarity, p.gain, p.ridged,
                  out)
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

    # The land mask is categorical and is always nearest -- averaging land with ocean is
    # not a value either of them can take.
    if p.resample == "barycentric" or p.resample == "c1":
        idx, w = barycentric_weights(grid, pts, tree=tree)
        nearest = idx[:, 0]
    else:
        nearest, w = neighbour_weights(grid, pts, k=1, tree=tree)
        idx = None

    def sample(field):
        return np.asarray(field)[nearest] if idx is None else apply_weights(field, idx, w)

    # Elevation is the only field whose DERIVATIVE anyone looks at -- it gets hillshaded --
    # so it is the only one offered C1. Slope and drainage only set the detail amplitude.
    z = (interpolate(grid, elevation, pts, mode="c1", tree=tree)
         if p.resample == "c1" else sample(elevation))
    land = np.asarray(land_mask)[nearest]
    if not detail:
        return z.reshape(shape).astype(np.float32), land.reshape(shape)

    s = sample(slope)
    a = sample(drainage_km2)
    amp = detail_amplitude(s, a, land, p, sphere_km2=float(grid.area.sum()))
    z = amplify(pts, z, amp, p, seed, octaves=octaves_for_width(width, p))
    return z.reshape(shape).astype(np.float32), land.reshape(shape)
