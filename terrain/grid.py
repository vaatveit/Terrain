"""Class-I Goldberg (geodesic hex) grid on the sphere.

Cells are the vertices of a frequency-n subdivided icosahedron; the cells themselves are
the dual, i.e. the spherical Voronoi regions of those vertices. That gives 10n^2+2 cells,
all hexagons except exactly 12 pentagons at the icosahedron corners, with uniform
6-neighbour connectivity.

Production resolution is n=785 -> 6,162,252 cells at ~9.77 km spacing, which is what
"4096 cells around the equator" works out to (2*pi*R / 4096 = 9.773 km).

The connectivity is built analytically from the subdivision lattice rather than by running
a Delaunay triangulation: within each icosahedron face the subdivision is a regular
triangular lattice whose adjacency is arithmetic, and stitching across the 30 shared edges
and 12 shared corners happens for free because both faces resolve a shared point to the
same global cell id. See tests/test_grid.py, which checks this against scipy's convex hull
(the hull of points on a sphere IS their Delaunay triangulation) at low n.

Everything is a flat NumPy array. Operators take the arrays, never this object -- passing a
class into @njit means jitclass, which is fragile and slow to compile.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

R_EARTH_KM = 6371.0

# Production frequency: 4096 cells around the equator -> ~6.17e6 cells -> n = 785.
N_PRODUCTION = 785

# THE WORLD IS SPECIFIED BY ITS CELL SIZE, not by its radius.
#
# The grid is dimensionless -- radius only scales `arc_dist`, `edge_len` and `area` --
# so this changes no graph algorithm and no topology. What it changes is which physical
# processes sit above the grid scale. At 9.77 km cells every land cell already carried
# more drainage area (82.7 km^2) than a real channel head needs (0.01-1 km^2), so there
# was no hillslope domain at all and `D` acted as a numerical smoother rather than as
# soil creep. At 100 m cells one cell is 0.0087 km^2, below the channel-head threshold,
# so hillslopes exist as terrain. See terrain-algorithms.md, "Cell size".
#
# THREE SPECS, AND THEY CANNOT ALL HOLD. 100 m cells, 4096 cells around the equator and
# R = 65.19 km are mutually consistent only for a perfect hexagonal tiling. The 12
# pentagons make the measured mean spacing 0.4% larger than ideal packing:
#
#     held exact        radius       mean cell    equatorial cells
#     R = 65.190 km     65.190 km    100.40 m     4096
#     100 m cells       64.927 km    100.00 m     4080
#
# Cell size wins, because it is what every retuned length scale below is denominated
# against. Swap R_WORLD_KM to EQUATOR_CELLS * TARGET_CELL_KM / (2*pi) for the other one;
# the 0.4% is immaterial to every result.
EQUATOR_CELLS = 4096
TARGET_CELL_KM = 0.1

# Measured: n * (mean neighbour arc on the unit sphere). Converged to 6 digits by n=64
# -- 1.208926 at n=32, 1.209054 at n=256 -- so it extrapolates to n=785 safely.
MEAN_SPACING_N = 1.209055


def radius_for_cell_km(n: int = N_PRODUCTION, cell_km: float = TARGET_CELL_KM) -> float:
    """Radius giving `cell_km` MEAN neighbour spacing at frequency `n`.

    Cell size is the invariant, not the radius. Every length-scale parameter in the
    pipeline is now denominated against 100 m cells, so a prototype at low `n` has to
    shrink the sphere to keep those parameters meaningful -- otherwise the prototype is
    a coarser world rather than a smaller sample of the same one, and every threshold
    from `min_lake_area_km2` to the diffusion sub-step count is wrong for it.
    """
    return float(cell_km * n / MEAN_SPACING_N)


R_WORLD_KM = radius_for_cell_km()  # 64.927 km at n=785, 4*pi*R^2 = 52,970 km^2


# --------------------------------------------------------------------------------------
# Base icosahedron
# --------------------------------------------------------------------------------------


def _icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )

    edges = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    edges = np.array(sorted(edges), dtype=np.int64)
    assert edges.shape == (30, 2)
    return verts, faces, edges


ICO_VERTS, ICO_FACES, ICO_EDGES = _icosahedron()
_EDGE_LOOKUP = {(int(a), int(b)): i for i, (a, b) in enumerate(ICO_EDGES)}


# --------------------------------------------------------------------------------------
# Global cell numbering
#   [0, 12)                           icosahedron corners
#   [12, 12+30*(n-1))                 edge interiors, 30 edges x (n-1)
#   [12+30*(n-1), 10n^2+2)            face interiors, 20 faces x (n-1)(n-2)/2
# --------------------------------------------------------------------------------------


def _edge_interior_ids(a, b, t, n):
    """Ids of the points t steps from corner `a` toward corner `b`, 0 < t < n."""
    key = (min(a, b), max(a, b))
    e = _EDGE_LOOKUP[key]
    # Edges are stored canonically; if this face traverses one backwards, mirror t.
    tt = t if a == key[0] else n - t
    return 12 + e * (n - 1) + (tt - 1)


def _face_point_ids(f, n):
    """Global cell id for every lattice point of face f, plus its (j, k) coordinates.

    Barycentric (i, j, k) with i + j + k = n; position ~ i*A + j*B + k*C for face
    corners (A, B, C), so i=n is corner A, j=n is B, k=n is C.
    """
    a, b, c = (int(x) for x in ICO_FACES[f])
    jj, kk = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = (jj + kk) <= n
    j = jj[keep].astype(np.int64)
    k = kk[keep].astype(np.int64)
    i = n - j - k

    ids = np.empty(j.shape, dtype=np.int64)

    corner_a, corner_b, corner_c = i == n, j == n, k == n
    ids[corner_a] = a
    ids[corner_b] = b
    ids[corner_c] = c

    # Edge interiors: exactly one barycentric coordinate is zero.
    on_ab = (k == 0) & ~corner_a & ~corner_b  # A -> B, j steps from A
    on_ca = (j == 0) & ~corner_c & ~corner_a  # A -> C, k steps from A
    on_bc = (i == 0) & ~corner_b & ~corner_c  # B -> C, k steps from B
    ids[on_ab] = _edge_interior_ids(a, b, j[on_ab], n)
    ids[on_ca] = _edge_interior_ids(a, c, k[on_ca], n)
    ids[on_bc] = _edge_interior_ids(b, c, k[on_bc], n)

    # Face interior: all three coordinates strictly positive.
    interior = (i > 0) & (j > 0) & (k > 0)
    per_face = (n - 1) * (n - 2) // 2
    base = 12 + 30 * (n - 1) + f * per_face
    ids[interior] = base + np.arange(int(interior.sum()), dtype=np.int64)

    return ids, i, j, k


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------


def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _slerp(u, v, t):
    """Great-circle interpolation. u, v unit vectors; t broadcastable to (...,1)."""
    dot = np.clip(np.einsum("...j,...j->...", u, v), -1.0, 1.0)[..., None]
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    # Endpoints coincide: fall back to linear, which is exact in the limit.
    near = sin_omega < 1e-12
    safe = np.where(near, 1.0, sin_omega)
    out = (np.sin((1.0 - t) * omega) * u + np.sin(t * omega) * v) / safe
    return np.where(near, (1.0 - t) * u + t * v, out)


def _spherical_barycentric(A, B, C, i, j, k, n):
    """Place a lattice point on the sphere by equal-ARC subdivision, not equal-chord.

    Linear interpolation in R^3 followed by normalisation (the obvious construction)
    bunches cells toward the face centres: measured sigma/mean of cell area is ~13%
    and max/min ~2.0, and Lloyd relaxation barely dents it because the distortion is
    structural rather than a local imbalance.

    Instead, for each corner, walk the great circle to the opposite edge and slerp along
    it; average the three results. This is the standard "Class I method 2" construction.
    """
    i = i[:, None].astype(np.float64)
    j = j[:, None].astype(np.float64)
    k = k[:, None].astype(np.float64)

    acc = np.zeros((i.shape[0], 3), dtype=np.float64)
    for (P, Q, S, p, q, s) in (
        (A, B, C, i, j, k),
        (B, C, A, j, k, i),
        (C, A, B, k, i, j),
    ):
        rest = q + s
        ok = (rest > 0)[:, 0]
        if not ok.any():
            continue
        # Point on the edge opposite P, then slerp from P out to it.
        edge = _slerp(
            np.broadcast_to(Q, (int(ok.sum()), 3)),
            np.broadcast_to(S, (int(ok.sum()), 3)),
            s[ok] / rest[ok],
        )
        acc[ok] += _slerp(np.broadcast_to(P, edge.shape), edge, rest[ok] / n)
    return _normalize(acc)


def _spherical_triangle_area(a, b, c):
    """Solid angle of a spherical triangle of unit vectors (Van Oosterom & Strackee).

    Numerically stable where Girard's excess-of-angles form is not.
    """
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", c, a)
    return 2.0 * np.abs(np.arctan2(num, den))


# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Grid:
    """Flat arrays describing the tiling. Pass the fields, not the object, into @njit."""

    n: int
    radius_km: float
    xyz: np.ndarray        # (N, 3) f64 unit vectors, cell centres
    lat: np.ndarray        # (N,) f32 degrees
    lon: np.ndarray        # (N,) f32 degrees
    neighbours: np.ndarray  # (N, 6) i32, azimuthally ordered, -1 in the unused pentagon slot
    arc_dist: np.ndarray   # (N, 6) f32 km, centre-to-centre great-circle distance
    edge_len: np.ndarray   # (N, 6) f32 km, shared Voronoi edge length
    area: np.ndarray       # (N,) f32 km^2, spherical Voronoi cell area

    @property
    def cell_count(self) -> int:
        return self.xyz.shape[0]

    def save(self, path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, n=self.n, radius_km=self.radius_km, xyz=self.xyz, lat=self.lat,
            lon=self.lon, neighbours=self.neighbours, arc_dist=self.arc_dist,
            edge_len=self.edge_len, area=self.area,
        )

    @staticmethod
    def load(path) -> "Grid":
        d = np.load(path)
        return Grid(
            n=int(d["n"]), radius_km=float(d["radius_km"]), xyz=d["xyz"], lat=d["lat"],
            lon=d["lon"], neighbours=d["neighbours"], arc_dist=d["arc_dist"],
            edge_len=d["edge_len"], area=d["area"],
        )


# --------------------------------------------------------------------------------------


def _build_positions_and_edges(n):
    """Cell centre positions and the undirected edge list, from the subdivision lattice."""
    n_cells = 10 * n * n + 2
    xyz = np.zeros((n_cells, 3), dtype=np.float64)
    seen = np.zeros(n_cells, dtype=bool)

    edge_u = []
    edge_v = []

    for f in range(20):
        a, b, c = (int(x) for x in ICO_FACES[f])
        ids, i, j, k = _face_point_ids(f, n)

        xyz[ids] = _spherical_barycentric(
            ICO_VERTS[a], ICO_VERTS[b], ICO_VERTS[c], i, j, k, n
        )
        seen[ids] = True

        # Map (j, k) -> row within this face so we can address lattice neighbours.
        lattice = np.full((n + 1, n + 1), -1, dtype=np.int64)
        lattice[j, k] = ids

        # Three of the six directions, so each intra-face edge is emitted exactly once.
        # In (j, k): (+1, 0), (0, +1), (-1, +1).
        for dj, dk in ((1, 0), (0, 1), (-1, 1)):
            j2, k2 = j + dj, k + dk
            ok = (j2 >= 0) & (k2 >= 0) & ((j2 + k2) <= n)
            edge_u.append(ids[ok])
            edge_v.append(lattice[j2[ok], k2[ok]])

    assert seen.all(), "some cells were never assigned a position"

    u = np.concatenate(edge_u)
    v = np.concatenate(edge_v)
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    # Shared icosahedron edges are walked by two faces, so dedupe.
    packed = lo * n_cells + hi
    packed = np.unique(packed)
    lo = packed // n_cells
    hi = packed % n_cells
    return xyz, lo.astype(np.int32), hi.astype(np.int32)


def _build_neighbour_table(n_cells, lo, hi):
    """(N, 6) neighbour table, -1 padded. Unordered at this point."""
    u = np.concatenate([lo, hi])
    v = np.concatenate([hi, lo])
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]

    degree = np.bincount(u, minlength=n_cells)
    if degree.max() > 6:
        raise AssertionError(f"cell with degree {degree.max()} > 6")
    if not np.all((degree == 6) | (degree == 5)):
        raise AssertionError("every cell must have degree 5 or 6")
    if int((degree == 5).sum()) != 12:
        raise AssertionError(f"expected exactly 12 pentagons, got {(degree == 5).sum()}")

    nbr = np.full((n_cells, 6), -1, dtype=np.int32)
    slot = np.concatenate([np.arange(d) for d in degree]) if n_cells else np.empty(0, int)
    nbr[u, slot] = v
    return nbr


def _order_neighbours_and_measure(xyz, nbr, radius_km, chunk=500_000):
    """Sort each cell's neighbours by azimuth, then measure cell area and Voronoi edges.

    Ordering is what makes the dual cheap: with neighbours in angular order, the Voronoi
    polygon vertices are just the circumcentres of consecutive neighbour pairs.
    """
    n_cells = xyz.shape[0]
    ordered = np.full_like(nbr, -1)
    arc = np.zeros(nbr.shape, dtype=np.float32)
    elen = np.zeros(nbr.shape, dtype=np.float32)
    area = np.zeros(n_cells, dtype=np.float32)

    for start in range(0, n_cells, chunk):
        stop = min(start + chunk, n_cells)
        c = xyz[start:stop]                      # (m, 3)
        nb = nbr[start:stop]                     # (m, 6)
        m = stop - start
        valid = nb >= 0
        deg = valid.sum(axis=1)

        safe = np.where(valid, nb, 0)
        p = xyz[safe]                            # (m, 6, 3)

        # Tangent-plane azimuth about the cell centre.
        d = p - c[:, None, :] * np.einsum("mkj,mj->mk", p, c)[..., None]
        e1 = d[:, 0, :].copy()
        e1 = _normalize(e1)
        e2 = np.cross(c, e1)
        ang = np.arctan2(
            np.einsum("mkj,mj->mk", d, e2), np.einsum("mkj,mj->mk", d, e1)
        )
        ang = np.where(valid, ang, np.inf)       # push the pentagon's empty slot last
        perm = np.argsort(ang, axis=1)

        nb_sorted = np.take_along_axis(nb, perm, axis=1)
        ordered[start:stop] = nb_sorted
        p = xyz[np.where(nb_sorted >= 0, nb_sorted, 0)]
        valid_sorted = nb_sorted >= 0

        # Centre-to-centre great-circle distance.
        dot = np.clip(np.einsum("mkj,mj->mk", p, c), -1.0, 1.0)
        arc[start:stop] = np.where(valid_sorted, radius_km * np.arccos(dot), 0.0)

        # Voronoi vertices: circumcentre of (centre, neighbour_i, neighbour_{i+1}).
        nxt = np.where(deg[:, None] == 5, (np.arange(6) + 1) % 5, (np.arange(6) + 1) % 6)
        nxt = np.broadcast_to(nxt, (m, 6))
        p_next = np.take_along_axis(p, nxt[..., None].repeat(3, axis=2), axis=1)

        cc = np.cross(p - c[:, None, :], p_next - c[:, None, :])
        norm = np.linalg.norm(cc, axis=2, keepdims=True)
        cc = np.divide(cc, norm, out=np.zeros_like(cc), where=norm > 0)
        # Keep circumcentres on the cell's own hemisphere.
        flip = np.einsum("mkj,mj->mk", cc, c) < 0
        cc = np.where(flip[..., None], -cc, cc)

        prev = np.where(deg[:, None] == 5, (np.arange(6) - 1) % 5, (np.arange(6) - 1) % 6)
        prev = np.broadcast_to(prev, (m, 6))
        cc_prev = np.take_along_axis(cc, prev[..., None].repeat(3, axis=2), axis=1)

        # Shared Voronoi edge for neighbour i runs cc[i-1] -> cc[i].
        dot_e = np.clip(np.einsum("mkj,mkj->mk", cc_prev, cc), -1.0, 1.0)
        elen[start:stop] = np.where(valid_sorted, radius_km * np.arccos(dot_e), 0.0)

        # Cell area: fan of spherical triangles (centre, cc[i-1], cc[i]).
        flat_c = np.repeat(c, 6, axis=0)
        tri = _spherical_triangle_area(
            flat_c, cc_prev.reshape(-1, 3), cc.reshape(-1, 3)
        ).reshape(m, 6)
        area[start:stop] = np.where(valid_sorted, tri, 0.0).sum(axis=1) * radius_km**2

    return ordered, arc, elen, area


def build_goldberg(n: int = N_PRODUCTION, radius_km: float = None,
                   verbose: bool = False) -> Grid:
    """Build the frequency-n class-I Goldberg grid.

    `radius_km` defaults to whatever gives TARGET_CELL_KM spacing at this `n`, so every
    grid the pipeline builds -- production or test fixture -- has 100 m cells and the
    tuned length scales mean the same thing on all of them. Pass R_EARTH_KM explicitly
    for an Earth-sized world.

    No relaxation pass: with the slerp construction above, measured cell area is already
    sigma/mean 0.89%, max/min 1.16, and that holds constant from n=64 to n=785. Both
    relaxation schemes tried made it worse -- neighbour-centroid smoothing collapses the
    pentagons (max/min 2.05 -> 4.41 over 100 iterations), and Lloyd converges far too
    slowly to matter (13.3% -> 12.2% sigma over 40 iterations) while also degrading
    max/min once the slerp construction is in place.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if radius_km is None:
        radius_km = radius_for_cell_km(n)

    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    log(f"n={n} -> {10 * n * n + 2:,} cells")
    xyz, lo, hi = _build_positions_and_edges(n)
    log(f"positions + {lo.size:,} undirected edges")

    nbr = _build_neighbour_table(xyz.shape[0], lo, hi)
    log("neighbour table")

    nbr, arc, elen, area = _order_neighbours_and_measure(xyz, nbr, radius_km)
    log("ordered neighbours, measured areas and Voronoi edges")

    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))).astype(np.float32)
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])).astype(np.float32)

    return Grid(
        n=n, radius_km=radius_km, xyz=xyz, lat=lat, lon=lon, neighbours=nbr,
        arc_dist=arc, edge_len=elen, area=area,
    )


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Build and summarise a Goldberg grid.")
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    g = build_goldberg(args.n, verbose=True)
    dt = time.perf_counter() - t0

    sphere = 4 * np.pi * g.radius_km**2
    total = float(g.area.sum())
    a = g.area
    print(f"\nn={g.n}  cells={g.cell_count:,}  built in {dt:.1f} s")
    print(f"  area sum      {total:,.1f} km^2 vs 4*pi*R^2 {sphere:,.1f} km^2 "
          f"(rel err {abs(total - sphere) / sphere:.2e})")
    print(f"  cell area     mean {a.mean():.3f}  min {a.min():.3f}  max {a.max():.3f} km^2")
    print(f"  area spread   max/min {a.max() / a.min():.4f}  "
          f"sigma/mean {a.std() / a.mean():.4%}  "
          f"p1..p99 {np.percentile(a, 1) / a.mean():.3f}..{np.percentile(a, 99) / a.mean():.3f}")
    d = g.arc_dist[g.neighbours >= 0]
    print(f"  spacing       mean {d.mean():.3f}  min {d.min():.3f}  max {d.max():.3f} km")
    print(f"  pentagons     {(g.neighbours < 0).any(axis=1).sum()}")

    if args.save:
        g.save(args.save)
        print(f"  saved -> {args.save}")
