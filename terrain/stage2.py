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
# How far the depression fill has to raise a cell before it counts as under water rather
# than as float32 round-off in the router. Matches Stage 6's tolerance, so the two stages
# agree on which cells are lake bed.
LAKE_TOL_M = 1.0e-3


@dataclasses.dataclass
class LemParams:
    # Stream power. theta = m/n at steady state, so these set the target concavity.
    #
    # K sets relief, not shape: steady-state slope is (U/K)^(1/n), so peak elevation scales
    # as U/K while theta is untouched. Two things make it a calibration rather than a
    # constant, and both are measured:
    #
    # RESOLUTION USED TO MATTER AND NO LONGER DOES. Finer grids resolve smaller headwater
    # drainage areas, where steady-state slope is steepest, so the same K yielded more
    # relief -- K=1e-6 gave 5.8 km peaks at n=128 and 17.6 km at n=785 when the sphere was
    # Earth-sized and cells coarsened with n. Cell size is now fixed at 100 m for every n,
    # so the smallest resolved drainage area is always 0.0086 km^2 and that dependence is
    # gone. What is left is a weak dependence on the LARGEST basin, which still grows with
    # the world: measured peak relief goes as L^0.156 (1289 m at n=192, 1436 m at n=384,
    # same K), so n=785 runs ~12% above n=384.
    #
    # COUPLING. Stage 4 replaces drainage area with discharge, which is a different
    # parameterisation, so K carries a different meaning there and needs its own value. The
    # coupled run holds ~1.45x the relief of the uncoupled one at the same K, because arid
    # cells sit at the discharge floor and erode more slowly -- and arid rain-shadow orogens
    # are exactly where the highest ground is. The floor caps the relief multiplier at
    # floor^(-m/n) = 0.30^-0.45 = 1.72x and 17-21% of land sits at that cap, so the measured
    # 1.45x is below the ceiling rather than equal to it. (This comment read 1.37x, which is
    # 0.5^-0.45 -- the right arithmetic for the wrong floor.) That is the behaviour the stage
    # exists to produce, so the fix is to recalibrate K rather than to suppress it.
    #
    # Measured on the 100 m world, uncoupled at n=384: relief goes as K^-0.668, giving
    # 1799 m peaks at K=8e-7 and ~2010 m extrapolated to n=785. Times the coupling factor,
    # 1.2e-6 targets ~2 km peaks for the COUPLED pipeline at n=785. That is the right
    # order for this world: 15,000 km^2 of land in landmasses ~120 km across, so 2 km of
    # relief is a mountainous island, not a plateau. It still sits inside the published
    # 1e-7..1e-4 range for m=0.45, n=1.
    K: float = 1.2e-6      # erodibility, yr^-1 for m~0.5 n=1
    m: float = 0.45        # drainage-area exponent
    n: float = 1.0         # slope exponent; n != 1 needs Newton, see solve_stream_power
    # Hillslope diffusion. NOT rescaled, and that is the point of the 100 m grid: D is in
    # m^2/yr, it was always physical, and it was the cell size that was wrong. At 9.77 km
    # cells the convex-hilltop-to-concave-valley transition it models sits two orders of
    # magnitude below the grid, so the term acted as a numerical smoother. At 100 m cells
    # it resolves, and D and the uplift rate become answerable to each other -- see
    # `u_max_m_per_myr` in plates.py. 0.05 is mid-range for published soil creep
    # (0.001-0.1 m^2/yr).
    #
    # It is EXPLICIT, so raising it raises the sub-step count in `diffusion_substeps`
    # linearly. It does not destabilise anything; it just costs more.
    #
    # 0.05 -> 0.02 IS A DELIBERATE TRADE, measured at n=384 with 400 steps. Resolving
    # hillslopes costs concavity, because ground that used to be counted as fluvial
    # genuinely is not:
    #
    #     D      channel head      theta   r2      sub-steps
    #     0.00   none (monotonic)  0.457   0.991   -
    #     0.01   0.021 km^2        0.402   0.927   10
    #     0.02   0.034 km^2        0.362   0.907   20
    #     0.05   0.056 km^2        0.267   0.748   48
    #
    # Every channel head above is inside the real 0.01-1 km^2 band, so that criterion
    # does not pick between them. 0.02 buys a hillslope regime about two cells deep --
    # enough for the convex limb to be visible in the slope-area curve -- while keeping
    # theta near the 0.4-0.6 band and the sub-step cost at 20. At D=0 there is no
    # hillslope regime at all: slope falls monotonically from the very first bin, which
    # is what the whole 9.8 km world looked like.
    D: float = 0.02        # m^2/yr
    # Deposition (Davy & Lague 2009 style). 0 = pure detachment-limited.
    G: float = 0.0
    # Mass wasting. Slopes steeper than the angle of repose are not stable ground: rock and
    # soil fail and the debris runs to the base. At 9.81 km cells this could not be modelled
    # -- 33 degrees across one cell was 6,371 m of drop against ~8 km of relief in the whole
    # world -- and at 100 m cells it is 65 m, which mountain terrain clears easily. So the
    # operator became buildable with the rescale and is built here.
    #
    # IT IS CURRENTLY A NEAR-NO-OP, AND THAT IS THE USEFUL RESULT. It was written because the
    # measured slope distribution showed p99 62.5 degrees, well past failure. Decomposing that
    # statistic on an n=128 world after Stage 2 shows where the steep ground actually is:
    #
    #     steep edges (>33 deg)          172,282
    #       touching an OCEAN cell       169,078   98.1%
    #       touching a LAKE cell           3,124    1.8%
    #       land-land, neither                80    0.05%   <- all this operator can act on
    #
    # Almost every "over-steep" edge is a cell sitting against a surface pinned at a level:
    # the shoreline, where ocean is held at base level, or a lake rim. Those are not hillslopes
    # and this operator correctly leaves them alone. Genuine hillslopes on the same world run
    # p50 0.9 degrees and p90 9.4 -- nowhere near repose, because U/D equilibrium does not put
    # them there. Measured at n=256, 2 passes x 200 steps, seed 1:
    #
    #     repose   theta   r2      Hack h   drainage   peak     z std
    #     off      0.438   0.919   0.515    0.1542     691 m    47.4 m
    #     45 deg   0.443   0.903   0.515    0.1527     691 m    47.3 m
    #     40 deg   0.460   0.878   0.517    0.1534     691 m    47.2 m
    #     33 deg   0.463   0.875   0.511    0.1545     690 m    47.4 m
    #     30 deg   0.455   0.886   0.514    0.1534     689 m    47.4 m
    #
    # Every row is the same world. Kept ON anyway, at a cost that does not show up in the run
    # time, because it is a physical constraint that ought to hold and there is one regime
    # where it bites: GLACIAL erosion creates real over-steep land, taking the land-land count
    # from 80 to 4,530. Stage 5b then floods almost all of it (back to 38), so a post-glacial
    # wasting pass would be relaxing ground that is about to become seabed and is not worth
    # adding -- but that is a property of this world's hypsometry, not a general one.
    #
    # It is a SEPARATE operator from `D`, not a nonlinear diffusivity. Roering's critical-slope
    # form (q = D*grad / (1 - (|grad|/Sc)^2)) is the standard geomorphic model and is the more
    # elegant answer, but its flux diverges as the slope approaches Sc, which would make the
    # explicit sub-stepping this file already pays for very much worse. A relaxation toward the
    # threshold costs one extra gather and cannot go unstable.
    repose_deg: float = 33.0
    # Fraction of the excess removed per pass. A cell has up to 6 edges that can all shed at
    # once, so this has to stay below ~1/3 or a pass can overshoot and invert the slope it was
    # relaxing. Three passes per timestep take 58% of the excess, which tracks the threshold
    # closely over a 400-step run without ever enforcing it exactly -- and hovering just at
    # the threshold is what real threshold hillslopes do under active uplift.
    wasting_relax: float = 0.25
    wasting_passes: int = 3
    # Total subsidence one cell may accumulate over the whole run, metres.
    #
    # NOTHING ELSE BOUNDS IT. Uplift is balanced by erosion -- 100 m/Myr over a 300 Myr run
    # is 30 km of uplift and the peaks come out at ~360 m -- but the process that balances
    # subsidence is sediment infill, and deposition is off (G = 0 above). So a rift graben
    # subsiding at 16.9 m/Myr simply keeps going: measured 5,063 m over 300 Myr, and Stage
    # 6 duly reported a lake 5,265 m deep on a world whose highest peak is 361 m.
    #
    # This was latent until closed depressions were exempted from the stream-power clamp.
    # The clamp used to snap every below-receiver cell back up to its receiver each step,
    # which cancelled the subsidence as fast as it was applied -- so grabens stayed shallow
    # for a reason that had nothing to do with tectonics, and the exemption that lets basins
    # survive is also what let this run away.
    #
    # A total is the right shape for the bound rather than a smaller rate: rift subsidence
    # IS finite. McKenzie stretching gives a basin a total subsidence set by how much the
    # crust thinned, after which it stops. 500 m puts the deepest basin floors in the range
    # real continental rifts occupy -- the Dead Sea at -430 m, Baikal's floor at -1,187 m --
    # on top of whatever Stage 1 tectonics already dug.
    max_subsidence_m: float = 500.0
    # Time stepping.
    dt_myr: float = 0.5
    fill_every: int = 1    # priority-flood cadence; 1 is affordable, see docs
    newton_iters: int = 8


@njit(cache=True)
def solve_stream_power(z, order, rec, step_km, acc_km2, uplift_m, dt_yr,
                       K, m, n, is_fixed, in_lake, newton_iters, out):
    """Implicit stream power, one pass in topological order.

    For n == 1 the discretisation is linear and solves exactly:

        z_i = (z_i + U*dt + F*z_r) / (1 + F),   F = K * A^m * dt / L

    For n != 1 the same equation is nonlinear in z_i and needs a Newton iteration per node.
    That is the cost of choosing n != 1, and it is why n should be decided before the
    solver is written rather than after.

    `in_lake` marks cells the depression fill had to raise -- lake bed rather than channel.
    THEY MUST BE SKIPPED, and the reason is the clamp at the end of this function. A cell
    inside a filled depression sits below its own receiver, and for such a cell the update
    (zi + f*zr)/(1 + f) also lies below zr, so the clamp snaps it straight up to the
    receiver in ONE step and the basin is gone. It is not a slow infill: measured on an
    n=128 world, closed depressions are 19.6% of land on the Stage 1 crust and 0.006% after
    ten steps. Skipping them costs nothing physically -- a lake bed has no channel to
    incise, and basin infill is deposition, which is the G term.
    """
    n_cells = order.shape[0]
    for t in range(n_cells):
        i = order[t]
        r = rec[i]
        if is_fixed[i] or r == i:
            out[i] = z[i]
            continue
        if in_lake[i]:
            out[i] = z[i] + uplift_m[i] * dt_yr
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
                        is_fixed, in_lake, dz):
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
    it needs D*dt*sum_j(e_ij/L_ij) < A_i/2. sum_j(e_ij/L_ij) is dimensionless (6/sqrt(3)
    = 3.46 for a hexagon) and scale-free, so the limit tracks CELL AREA exactly -- which
    means it falls as the square of cell size. At 9.77 km cells it was 239 Myr, five
    hundred times the timestep. At 100 m cells it is 25,000 yr, twenty times SHORTER than
    the timestep. Call `diffusion_substeps` and sub-cycle; do not call this raw with a
    fluvial dt.

    `in_lake` is a no-flux boundary, not a source term: cells inside a closed depression
    neither diffuse nor are diffused into, because an edge is skipped whenever EITHER end
    is in a lake. That test is symmetric, so mass is still conserved exactly over the cells
    that remain. Creep does deliver sediment into a real basin, but that is deposition and
    belongs to the G term; what this operator does instead is pull the rim down into the
    hole and lift the floor, and it is quick enough to erase a 1.4 km-deep graben inside
    ten steps. Diffusion is the SECOND of the two terms that do this -- see the `in_lake`
    note in `solve_stream_power` for the first, and note that exempting either one alone
    achieves little, because whichever is left fills the basin on its own (measured on the
    Stage 1 crust at n=128, closed depressions after 200 steps: 0.002% of land shipped,
    0.854% exempting diffusion only, 0.002% exempting the clamp only, 3.240% exempting
    both).
    """
    n_cells = z.shape[0]
    k = nbr.shape[1]
    for i in prange(n_cells):
        if is_fixed[i] or in_lake[i]:
            dz[i] = 0.0
            continue
        total = 0.0
        for s in range(k):
            j = nbr[i, s]
            if j < 0 or in_lake[j]:
                continue
            total += edge_len_km[i, s] * (z[j] - z[i]) / arc_dist_km[i, s]
        dz[i] = D * dt_yr * total / (area_km2[i] * SEC_PER_KM2)
    return dz


@njit(cache=True, parallel=True)
def mass_wasting(z, nbr, arc_dist_km, area_km2, critical_slope, relax,
                 is_fixed, in_lake, dz):
    """Relax every slope steeper than the angle of repose, conserving mass exactly.

    For an edge whose drop exceeds `critical_slope * L`, move material from the high cell to
    the low one so the height difference falls by `relax` times the excess:

        dz_high = -relax * excess * A_low  / (A_high + A_low)
        dz_low  = +relax * excess * A_high / (A_high + A_low)

    Those two are constructed so that A_high*dz_high + A_low*dz_low is exactly zero and the
    difference changes by exactly -relax*excess. The volume split is what makes it correct on
    an unstructured mesh: an equal split of the HEIGHT would quietly create or destroy
    material wherever two cells differ in area, which on this grid they always do.

    Written as a gather, like `hillslope_diffusion` and for the same reason -- each cell
    computes its own dz by looking at both ends of each of its edges, so nothing scatters and
    prange is safe. Both ends of an edge evaluate the same excess and take opposite signs.

    `is_fixed` and `in_lake` are no-flux boundaries, and the test is symmetric (an edge is
    skipped if EITHER end is excluded), so mass is still conserved exactly over the cells that
    remain. Excluding lakes follows `hillslope_diffusion` rather than physics: scree really
    does run into a real basin, but with G = 0 there is no deposition to balance it, and an
    operator that fills closed depressions deletes the endorheic lakes Stage 6 exists to find.
    """
    n_cells = z.shape[0]
    k = nbr.shape[1]
    for i in prange(n_cells):
        dz[i] = 0.0
        if is_fixed[i] or in_lake[i]:
            continue
        total = 0.0
        for s in range(k):
            j = nbr[i, s]
            if j < 0 or is_fixed[j] or in_lake[j]:
                continue
            length_m = arc_dist_km[i, s] * M_PER_KM
            if length_m <= 0.0:
                continue
            limit = critical_slope * length_m
            drop = z[i] - z[j]
            share = relax * area_km2[j] / (area_km2[i] + area_km2[j])
            if drop > limit:
                total -= share * (drop - limit)      # i is above repose: it sheds
            elif -drop > limit:
                total += share * (-drop - limit)     # j is above repose: i receives
        dz[i] = total
    return dz


def diffusion_substeps(nbr, edge_len_km, arc_dist_km, area_km2, D, dt_yr, safety=0.5):
    """Explicit sub-steps needed to diffuse stably over `dt_yr`.

    The fluvial solve is implicit and unconditionally stable, so it wants the largest
    timestep that still resolves the tectonics -- 0.5 Myr. Diffusion is explicit and at
    100 m cells wants 25 kyr. Dropping the global timestep to satisfy diffusion would
    multiply the cost of the expensive part (priority-flood, O(N log N) with bad cache
    behaviour) by 20 to fix the cheap part. Sub-cycling pays only where the constraint
    actually bites.

    Uses the WORST cell rather than the mean, because the 12 pentagons are the smallest
    cells on the grid and a scheme that is unstable on twelve cells is unstable.
    """
    w = np.where(np.asarray(nbr) >= 0,
                 np.asarray(edge_len_km) / np.maximum(np.asarray(arc_dist_km), 1e-30),
                 0.0).sum(axis=1)
    limit_yr = np.min(np.asarray(area_km2) * SEC_PER_KM2
                      / (2.0 * D * np.maximum(w, 1e-30)))
    return max(1, int(np.ceil(dt_yr / (safety * float(limit_yr)))))


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


def uplift_field(grid, land_mask, seed=0, u_max_m_per_myr=100.0, craton_floor=0.25,
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
           base_level_m=0.0, discharge_weight=None) -> EvolveResult:
    """Run the landscape evolution model to near steady state.

    `base_level_m` is the elevation ocean cells present to the land draining into them, and
    it must be SEA LEVEL, not the sea floor. Holding ocean cells at their Stage 1
    bathymetric depth instead lets coastal rivers grade toward the abyssal plain 4 km down,
    so continents incise straight past zero and drown: measured land minimum -1371 m, with
    only the orogen belts left above water. Subaerial erosion grades to sea level. The true
    bathymetry is restored on the way out, since ocean cells never erode anyway.

    `discharge_weight` replaces cell area in the flow accumulation, which is how climate
    couples back into erosion (Stage 4). Stream power is properly written against DISCHARGE,
    E = K*Q^m*S^n, and drainage area is only a stand-in for it under uniform rainfall. Pass
    `area * P/mean(P)` and Q keeps area units, so K keeps its meaning and the result is
    identical to the uncoupled run wherever precipitation is uniform. Equivalent to scaling
    erodibility as K*P^m, but it needs no change to the solver at all.
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
    # Uplift and subsidence take different paths. The stream-power solve clamps every cell
    # to stay at or above its receiver, which is right for incision -- water cannot cut a
    # closed basin -- but it would also forbid a rift graben from ever dropping below its
    # outlet. So positive uplift goes through the solver, and subsidence is applied
    # afterwards without the clamp. Without this split, tectonic basins cannot form and
    # Stage 6 has no endorheic lakes to find.
    u_all = uplift_m_per_yr_from(uplift_m_per_myr)
    uplift_m_per_yr = np.ascontiguousarray(np.maximum(u_all, 0.0), dtype=np.float64)
    subsidence_m_per_yr = np.ascontiguousarray(np.minimum(u_all, 0.0), dtype=np.float64)
    has_subsidence = bool((subsidence_m_per_yr < 0).any())
    subsided_m = np.zeros(n_cells, dtype=np.float64)

    accum = area if discharge_weight is None else np.ascontiguousarray(
        discharge_weight, dtype=np.float32)

    # tan() once, not per step. 0 or >= 90 degrees switches mass wasting off, which is what
    # the uncoupled comparisons in the docstring above were measured against.
    critical_slope = (np.tan(np.radians(p.repose_deg))
                      if 0.0 < p.repose_deg < 90.0 else 0.0)

    dt_yr = p.dt_myr * YR_PER_MYR
    n_sub = (diffusion_substeps(nbr, edge, arc, area, p.D, dt_yr) if p.D > 0.0 else 1)
    dt_sub = dt_yr / n_sub
    if verbose and n_sub > 1:
        print(f"    diffusion sub-cycled {n_sub}x "
              f"({dt_sub / 1e3:.1f} kyr per sub-step)", flush=True)
    out = np.empty(n_cells, dtype=np.float64)
    dz = np.empty(n_cells, dtype=np.float64)
    qs = np.empty(n_cells, dtype=np.float64)
    in_lake = np.zeros(n_cells, dtype=np.bool_)
    history = []

    t_start = time.perf_counter()
    fr = None
    for step in range(n_steps):
        if fr is None or step % p.fill_every == 0:
            fr = flow.route(z.astype(np.float32), nbr, arc, accum, is_fixed,
                            metrics=False)
            # Cells the fill had to raise are under water for this step. Both the fluvial
            # solve and the hillslope term destroy closed basins if allowed to touch them.
            in_lake = np.ascontiguousarray(
                fr.filled.astype(np.float64) > z + LAKE_TOL_M)

        z_before = z.copy()

        solve_stream_power(z, fr.order, fr.rec, fr.step, fr.acc, uplift_m_per_yr,
                           dt_yr, p.K, p.m, p.n, is_fixed, in_lake, p.newton_iters, out)
        z, out = out, z

        if has_subsidence:
            # Bounded by a TOTAL, not run as an open-ended rate -- see `max_subsidence_m`.
            drop = np.where(is_fixed, 0.0, subsidence_m_per_yr * dt_yr)
            room = np.maximum(p.max_subsidence_m - subsided_m, 0.0)
            drop = np.maximum(drop, -room)
            subsided_m -= drop
            z += drop

        if p.G > 0.0:
            erosion = np.maximum(z_before + uplift_m_per_yr * dt_yr - z, 0.0)
            accumulate_sediment(fr.order, fr.rec, erosion, area, dt_yr, qs)
            deposit = p.G * qs * dt_yr / (area * SEC_PER_KM2)
            deposit[is_fixed] = 0.0
            z += deposit

        if p.D > 0.0:
            for _ in range(n_sub):
                hillslope_diffusion(z, nbr, edge, arc, area, p.D, dt_sub, is_fixed,
                                    in_lake, dz)
                z += dz

        # After diffusion, not before: diffusion is what steepens the basal slope toward
        # U/D, so relaxing repose first would leave the step's own oversteepening in place.
        # Not sub-cycled with it either -- failure is instantaneous against a 0.5 Myr step,
        # and this operator has no stability limit to respect.
        if critical_slope > 0.0:
            for _ in range(p.wasting_passes):
                mass_wasting(z, nbr, arc, area, critical_slope, p.wasting_relax,
                             is_fixed, in_lake, dz)
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
