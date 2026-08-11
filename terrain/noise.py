"""3D simplex noise, fBm and domain warping, sampled at cell unit vectors.

On a sphere you sample noise in 3D at each cell's position on the unit sphere, never in 2D
over (lon, lat). It is both simpler and strictly better: seamless by construction, no wrap
seam, no pole pinching, no distortion to correct for.

Simplex rather than Perlin because Perlin's axis-aligned artifacts are visible in 3D, and
they would show up as continent edges that all favour the same few directions.

Written here rather than taken from PyPI deliberately: `noise` is a C extension that cannot
be called from nopython mode, and `opensimplex` is pure Python and far too slow for 6.2M
cells x several octaves.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# The 12 edge-midpoint gradients of a cube.
GRAD3 = np.array(
    [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
     [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
     [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]],
    dtype=np.float64,
)

_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0


def permutation(seed: int) -> np.ndarray:
    """256-entry permutation, doubled so corner lookups never need a modulo."""
    p = np.random.default_rng(seed).permutation(256).astype(np.int32)
    return np.concatenate([p, p])


@njit(cache=True)
def simplex3(x, y, z, perm, grad):
    """Scalar 3D simplex noise, roughly in [-1, 1]."""
    s = (x + y + z) * _F3
    i = int(np.floor(x + s))
    j = int(np.floor(y + s))
    k = int(np.floor(z + s))
    t = (i + j + k) * _G3
    x0 = x - (i - t)
    y0 = y - (j - t)
    z0 = z - (k - t)

    # Which of the six tetrahedra within the cell are we in?
    if x0 >= y0:
        if y0 >= z0:
            i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0
        elif x0 >= z0:
            i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1
        else:
            i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1
    else:
        if y0 < z0:
            i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1
        elif x0 < z0:
            i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1
        else:
            i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0

    x1, y1, z1 = x0 - i1 + _G3, y0 - j1 + _G3, z0 - k1 + _G3
    x2, y2, z2 = x0 - i2 + 2 * _G3, y0 - j2 + 2 * _G3, z0 - k2 + 2 * _G3
    x3, y3, z3 = x0 - 1.0 + 3 * _G3, y0 - 1.0 + 3 * _G3, z0 - 1.0 + 3 * _G3

    ii, jj, kk = i & 255, j & 255, k & 255
    g0 = perm[ii + perm[jj + perm[kk]]] % 12
    g1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
    g2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
    g3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12

    n = 0.0
    t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
    if t0 > 0.0:
        t0 *= t0
        n += t0 * t0 * (grad[g0, 0] * x0 + grad[g0, 1] * y0 + grad[g0, 2] * z0)
    t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
    if t1 > 0.0:
        t1 *= t1
        n += t1 * t1 * (grad[g1, 0] * x1 + grad[g1, 1] * y1 + grad[g1, 2] * z1)
    t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
    if t2 > 0.0:
        t2 *= t2
        n += t2 * t2 * (grad[g2, 0] * x2 + grad[g2, 1] * y2 + grad[g2, 2] * z2)
    t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
    if t3 > 0.0:
        t3 *= t3
        n += t3 * t3 * (grad[g3, 0] * x3 + grad[g3, 1] * y3 + grad[g3, 2] * z3)
    return 32.0 * n


@njit(cache=True)
def _fbm_at(x, y, z, perm, grad, octaves, frequency, lacunarity, gain):
    total = 0.0
    amp = 1.0
    freq = frequency
    norm = 0.0
    for _ in range(octaves):
        total += amp * simplex3(x * freq, y * freq, z * freq, perm, grad)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


@njit(cache=True, parallel=True)
def fbm(xyz, perm, grad, octaves, frequency, lacunarity, gain, out):
    """Fractional Brownian motion over an (N, 3) array of positions."""
    for i in prange(xyz.shape[0]):
        out[i] = _fbm_at(xyz[i, 0], xyz[i, 1], xyz[i, 2],
                         perm, grad, octaves, frequency, lacunarity, gain)
    return out


@njit(cache=True, parallel=True)
def warped_fbm(xyz, perm, grad, octaves, frequency, lacunarity, gain,
               warp_amp, warp_frequency, out):
    """fBm evaluated at domain-warped coordinates.

    The cheapest realism-per-line-of-code available: feeding noise into the *coordinates*
    of noise turns isotropic blobs into folded, geologically flowy shapes. Offsetting each
    warp component into a different region of the same noise field keeps them decorrelated
    without needing three separate permutation tables.
    """
    for i in prange(xyz.shape[0]):
        x, y, z = xyz[i, 0], xyz[i, 1], xyz[i, 2]
        wx = _fbm_at(x * warp_frequency, y * warp_frequency, z * warp_frequency,
                     perm, grad, 4, 1.0, lacunarity, gain)
        wy = _fbm_at(x * warp_frequency + 31.4, y * warp_frequency + 17.1,
                     z * warp_frequency + 5.7, perm, grad, 4, 1.0, lacunarity, gain)
        wz = _fbm_at(x * warp_frequency + 71.9, y * warp_frequency + 43.3,
                     z * warp_frequency + 89.2, perm, grad, 4, 1.0, lacunarity, gain)
        out[i] = _fbm_at(x + warp_amp * wx, y + warp_amp * wy, z + warp_amp * wz,
                         perm, grad, octaves, frequency, lacunarity, gain)
    return out
