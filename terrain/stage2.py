"""Stage 2: landscape evolution. Stream power incision + hillslope diffusion + uplift.

This is the realism engine. Everything before it produces shape; this produces the
dendritic drainage network that makes terrain hold up when you look closely.

    dz/dt = U - K * A^m * S^n  +  D * laplacian(z)

Solved with Braun & Willett's implicit method: process cells in topological order, so each
cell's receiver already has its new elevation, and the update becomes a scalar solve per
cell. O(n), unconditionally stable, and takes million-year timesteps rather than the tiny
ones an explicit scheme would force.

The steady-state relation is what makes this testable. Setting dz/dt = 0 gives

    S = (U/K)^(1/n) * A^(-m/n)

so the slope-area concavity theta equals m/n exactly. With m=0.45, n=1 the model should
converge on theta = 0.45, right inside the 0.4-0.6 band real terrain occupies. That is a
genuine end-to-end check: it only comes out right if the solver, the flow routing, the
spherical geometry and the metrics all agree.

UNITS. Interfaces are in natural units -- area km^2, distance km, elevation m, uplift
m/Myr, time Myr -- and converted to SI internally. Getting this wrong is the easiest way
to produce a plausible-looking but meaningless landscape, so the conversions all happen in
one place, in `evolve`.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
from numba import njit, prange

from terrain import flow, noise

SEC_PER_KM2 = 1.0e6   # km^2 -> m^2
M_PER_KM = 1.0e3      # km    -> m
YR_PER_MYR = 1.0e6


@dataclasses.dataclass
class LemParams:
    # Stream power. theta = m/n at steady state, so these set the target concavity.
    # K sets relief, not shape: steady-state slope is (U/K)^(1/n), so peak elevation scales
    # as U/K while theta is untouched. K IS RESOLUTION-DEPENDENT -- at 9.8 km cells the
    # model resolves smaller headwater drainage areas than at 60 km, where steady-state
    # slope is steepest, so the same K gives far more total relief. Measured: K=1e-6 gave
    # 5.8 km peaks at n=128 but 17.6 km at n=785. Tuned here for n=785; rescale for other
    # resolutions. 2.5e-6 is well inside the published 1e-7..1e-4 range for m=0.45, n=1.
    K: float = 2.5e-6      # erodibility, yr^-1 for m~0.5 n=1
    m: float = 0.45        # drainage-area exponent
    n: float = 1.0         # slope exponent; n != 1 needs Newton, see solve_stream_power
    # Hillslope diffusion.
    D: float = 0.05        # m^2/yr
    # Deposition (Davy & Lague 2009 style). 0 = pure detachment-limited.
    G: float = 0.0
    # Time stepping.
    dt_myr: float = 0.5
    fill_every: int = 1    # priority-flood cadence; 1 is affordable, see docs
    newton_iters: int = 8


@njit(cache=True)
def solve_stream_power(z, order, rec, step_km, acc_km2, uplift_m, dt_yr,
                       K, m, n, is_fixed, newton_iters, out):
    """Implicit stream power, one pass in topological order.

    For n == 1 the discretisation is linear and solves exactly:

        z_i = (z_i + U*dt + F*z_r) / (1 + F),   F = K * A^m * dt / L

    For n != 1 the same equation is nonlinear in z_i and needs a Newton iteration per node.
    That is the cost of choosing n != 1, and it is why n should be decided before the
    solver is written rather than after.
    """
    n_cells = order.shape[0]
    for t in range(n_cells):
        i = order[t]
        r = rec[i]
        if is_fixed[i] or r == i:
            out[i] = z[i]
            continue

        zi = z[i] + uplift_m[i] * dt_yr
        zr = out[r]  # receiver already updated -- this is the whole trick
        length_m = step_km[i] * M_PER_KM
        if length_m <= 0.0:
            out[i] = zi
            continue

        f = K * (acc_km2[i] * SEC_PER_KM2) ** m * dt_yr / length_m

        if n == 1.0:
            val = (zi + f * zr) / (1.0 + f)
        else:
            val = zi if zi > zr else zr
            for _ in range(newton_iters):
                d = val - zr
                if d <= 0.0:
                    val = zr
                    break
                fx = val - zi + f * d**n
                dfx = 1.0 + f * n * d ** (n - 1.0)
                val -= fx / dfx
        # Incision cannot cut below the receiver; that would invert the flow direction.
        out[i] = val if val > zr else zr
    return out


@njit(cache=True, parallel=True)
def hillslope_diffusion(z, nbr, edge_len_km, arc_dist_km, area_km2, D, dt_yr,
                        is_fixed, dz):
    """Finite-volume Laplacian on the unstructured mesh.

        dz_i/dt = (D / A_i) * sum_j e_ij * (z_j - z_i) / L_ij

    with e_ij the shared Voronoi edge length and L_ij the centre-to-centre arc distance.
    Not a 5-point stencil -- the mesh is irregular, and the Delaunay/Voronoi dual being
    orthogonal is what keeps this form clean. Conserves mass exactly on a closed sphere.

    Two things to know about this term at continental resolution.

    It does almost no direct work. Hillslope diffusion sets the convex-hilltop-to-concave-
    valley transition, which happens over tens to hundreds of metres; at ~10 km cells that
    is far below the grid scale. Measured on evolved terrain at the default D, one step
    moves at most 0.014 m against 4 km of relief.

    But it is not therefore ignorable. Because it biases every step in the same direction,
    it eventually tips receiver choices at drainage divides, and a flipped divide re-routes
    a whole sub-basin. Measured over 100 steps: the median difference from a D=0 run is
    0.014 m, but the maximum is 720 m. A small persistent forcing is not a small effect.

    Unlike the implicit stream power solve, this is EXPLICIT and so conditionally stable:
    it needs D*dt*sum_j(e_ij/L_ij) < A_i/2. The defaults leave orders of magnitude of
    headroom, but raising D or dt far enough will diverge.
    """
    n_cells = z.shape[0]
    k = nbr.shape[1]
    for i in prange(n_cells):
        if is_fixed[i]:
            dz[i] = 0.0
            continue
        total = 0.0
        for s in range(k):
            j = nbr[i, s]
            if j < 0:
                continue
            total += edge_len_km[i, s] * (z[j] - z[i]) / arc_dist_km[i, s]
        dz[i] = D * dt_yr * total / (area_km2[i] * SEC_PER_KM2)
    return dz


@njit(cache=True)
def accumulate_sediment(order, rec, erosion_m, area_km2, dt_yr, qs):
    """Downstream sediment flux, m^3/yr, from per-cell erosion this step."""
    n_cells = order.shape[0]
    for i in range(n_cells):
        qs[i] = erosion_m[i] * area_km2[i] * SEC_PER_KM2 / dt_yr
    for t in range(n_cells - 1, -1, -1):
        c = order[t]
        r = rec[c]
        if r != c:
            qs[r] += qs[c]
    return qs


@dataclasses.dataclass
class EvolveResult:
    elevation: np.ndarray
    steps: int
    seconds: float
    history: list  # per-step (step, mean |dz| m/Myr, max elevation, relief)


def uplift_field(grid, land_mask, seed=0, u_max_m_per_myr=500.0, craton_floor=0.25,
                 frequency=1.4, octaves=4, belt_sharpness=3.0):
    """Stand-in for the tectonic uplift the real Stage 1 will derive.

    The proper source is boundary_type and boundary_stress from plate motion: broad orogens
    at continental collision, arcs offset inland from trenches, rift shoulders at divergent
    margins, decaying ~200 km from the boundary. None of that exists yet.

    Crucially this uses RIDGED noise, not plain fBm. Real uplift is belt-like -- narrow
    active margins with vast near-zero cratonic interiors -- and that distinction turned out
    to matter for more than looks. With smoothly varying uplift, U changes continuously
    along every channel profile, and since basins drain from highlands to the sea, U ends up
    structurally anti-correlated with drainage area. That lands straight in the slope-area
    regression: measured theta came out 0.77-0.97 against a true m/n of 0.45. Belt-like
    uplift leaves most basins sitting in near-uniform U, where theta = m/n holds.

    `craton_floor` sets the uplift CONTRAST (1/floor), and it is the single most
    consequential knob here -- not for how the world looks, but for whether the terrain
    metrics mean anything. Measured at n=128, holding everything else fixed:

        contrast    1x     2x     4x    10x    50x
        theta    0.453  0.452  0.512  0.690  0.893

    Steady-state slope goes as U^(1/n), so a 50x contrast varies slope 50x on its own while
    the A^-0.45 signal only spans ~22x across three decades of drainage area. The uplift
    pattern then dominates the slope-area regression and theta stops measuring m/n.

    Earth's real craton-to-orogen contrast is 50x or more, and real Earth measured this way
    would be just as inflated -- published theta values come from tectonically coherent
    regions, not whole continents. The default of 0.25 (4x) is an authoring compromise:
    clear orogenic belts, ~5-6 km peaks, and theta still inside the 0.4-0.6 band. Raise the
    contrast for more Earth-like tectonics, but then compare theta only against other
    generated worlds, never against published values.

    Uplift shape matters more than magnitude -- erosion finds equilibrium against whatever
    it is given -- which is precisely why the shape being wrong was expensive.
    """
    field = np.empty(grid.cell_count, dtype=np.float64)
    perm = noise.permutation(noise.sub_seed("uplift", seed))
    noise.fbm(np.ascontiguousarray(grid.xyz), perm, noise.GRAD3,
              octaves, frequency, 2.0, 0.5, field)

    # Ridged transform: |noise| near zero becomes a narrow crest, everything else falls away.
    belts = np.clip(1.0 - np.abs(field) / 0.6, 0.0, 1.0) ** belt_sharpness
    u = u_max_m_per_myr * np.maximum(belts, craton_floor)
    u[~land_mask] = 0.0  # ocean is base level
    return u.astype(np.float32)


def evolve(grid, z0, uplift_m_per_myr, is_ocean, params: LemParams = None,
           n_steps=400, verbose=False, report_every=50,
           base_level_m=0.0) -> EvolveResult:
    """Run the landscape evolution model to near steady state.

    `base_level_m` is the elevation ocean cells present to the land draining into them, and
    it must be SEA LEVEL, not the sea floor. Holding ocean cells at their Stage 1
    bathymetric depth instead lets coastal rivers grade toward the abyssal plain 4 km down,
    so continents incise straight past zero and drown: measured land minimum -1371 m, with
    only the orogen belts left above water. Subaerial erosion grades to sea level. The true
    bathymetry is restored on the way out, since ocean cells never erode anyway.
    """
    p = params or LemParams()
    n_cells = grid.cell_count

    z = np.ascontiguousarray(z0, dtype=np.float64).copy()
    ocean_z0 = z[np.ascontiguousarray(is_ocean)].copy()
    z[np.ascontiguousarray(is_ocean)] = base_level_m
    nbr = np.ascontiguousarray(grid.neighbours)
    arc = np.ascontiguousarray(grid.arc_dist)
    edge = np.ascontiguousarray(grid.edge_len)
    area = np.ascontiguousarray(grid.area)

    # Ocean is base level: fixed elevation, no uplift, no incision.
    is_fixed = np.ascontiguousarray(is_ocean)
    uplift_m_per_yr = np.ascontiguousarray(
        uplift_m_per_yr_from(uplift_m_per_myr), dtype=np.float64)

    dt_yr = p.dt_myr * YR_PER_MYR
    out = np.empty(n_cells, dtype=np.float64)
    dz = np.empty(n_cells, dtype=np.float64)
    qs = np.empty(n_cells, dtype=np.float64)
    history = []

    t_start = time.perf_counter()
    fr = None
    for step in range(n_steps):
        if fr is None or step % p.fill_every == 0:
            fr = flow.route(z.astype(np.float32), nbr, arc, area, is_fixed,
                            metrics=False)

        z_before = z.copy()

        solve_stream_power(z, fr.order, fr.rec, fr.step, fr.acc, uplift_m_per_yr,
                           dt_yr, p.K, p.m, p.n, is_fixed, p.newton_iters, out)
        z, out = out, z

        if p.G > 0.0:
            erosion = np.maximum(z_before + uplift_m_per_yr * dt_yr - z, 0.0)
            accumulate_sediment(fr.order, fr.rec, erosion, area, dt_yr, qs)
            deposit = p.G * qs * dt_yr / (area * SEC_PER_KM2)
            deposit[is_fixed] = 0.0
            z += deposit

        if p.D > 0.0:
            hillslope_diffusion(z, nbr, edge, arc, area, p.D, dt_yr, is_fixed, dz)
            z += dz

        z[is_fixed] = z_before[is_fixed]

        rate = np.abs(z - z_before) / p.dt_myr
        land = ~is_fixed
        history.append((step, float(rate[land].mean()), float(z[land].max()),
                        float(z[land].max() - z[land].min())))
        if verbose and (step % report_every == 0 or step == n_steps - 1):
            print(f"    step {step:4d}  t={step * p.dt_myr:7.1f} Myr  "
                  f"mean |dz| {rate[land].mean():8.2f} m/Myr  "
                  f"max z {z[land].max():7.0f} m", flush=True)

    z[is_fixed] = ocean_z0  # put the real bathymetry back
    return EvolveResult(z.astype(np.float32), n_steps,
                        time.perf_counter() - t_start, history)


def uplift_m_per_yr_from(u_m_per_myr):
    return np.asarray(u_m_per_myr, dtype=np.float64) / YR_PER_MYR
