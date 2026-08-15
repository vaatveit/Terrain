"""Stage 1: crust thickness -> isostasy -> sea level, with tectonics.

TWO ENTRY POINTS, and the pipeline calls the second one. `generate_crust` is the original
throwaway: noise, isostasy and a solved sea level, with no plates at all. It exists because
most of Stage 1's careful output gets overwritten by erosion anyway, so the real version was
worth building only once Stage 2 had shown which of its outputs it actually consumes. It is
now used only by tests, as the control that tectonics is measured against.

`generate_crust_tectonic` is the full version and the one `scripts/generate_world.py` runs:
it adds plates, Euler-pole motion, boundary classification, tectonic thickening, margin
classes and the derived uplift field. See `terrain/plates.py` for that half.

What both share, because these are the load-bearing ideas:

  * Generate crustal THICKNESS, not elevation. Elevation falls out of isostasy and sea
    level is a threshold applied afterward, which makes the coastline a consequence rather
    than a drawn line.
  * Crust type comes from 3D noise, independent of any plate layout. Assigning it per plate
    is geologically wrong -- six of Earth's seven major plates carry both crust types -- and
    gives every continent a Voronoi silhouette.
  * Sea level is solved, not assumed: an area-weighted quantile search for a target land
    fraction.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit

from terrain import noise


@dataclasses.dataclass
class CrustParams:
    continental_fraction: float = 0.40   # of SURFACE, not of land; the rest drowns as shelf
    target_land_fraction: float = 0.29   # Earth value
    # Continent count and size spread. Tuned against the landmass-size diagnostic: at
    # frequency 1.6 the world came out as 340 landmasses with the largest holding only 15%
    # of land, where Earth's top three hold ~94%. Dropping to 0.7 gives 64 landmasses and a
    # top-three share of 84% -- heavy-tailed, which is what the metric asks for. Coastline
    # detail comes from the domain warp below, not from piling on octaves here.
    crust_octaves: int = 3
    crust_frequency: float = 0.7
    warp_amp: float = 0.35               # domain warp; fractal coastlines need this
    warp_frequency: float = 2.2
    transition: float = 0.12             # width of the continental-slope ramp, in noise units
    relief_octaves: int = 7              # thickness texture, so the interior is not flat
    relief_frequency: float = 6.0
    # ISOSTASY IS RADIUS-BLIND: elevation comes out of crustal thickness in km, with no
    # reference to the sphere at all. Left at Earth values it would put 4,455 m of
    # continent-to-abyss relief on a world 64.9 km in radius -- 7% of the radius, against
    # Earth's 0.07% -- which renders as a potato rather than a planet. Thickness is cut
    # ~3x to a relief budget of ~1,530 m, which is what a 15,000 km^2 landmass should
    # carry. Noise FREQUENCIES are untouched: they are sampled on the unit sphere, so the
    # continents keep their shape and only their size in km changes.
    relief_km: float = 1.0               # thickness variation amplitude, km
    t_continental_km: float = 12.0
    t_oceanic_km: float = 2.4
    rho_continental: float = 2.8
    rho_oceanic: float = 2.9             # denser -- this is why ocean floor sits low
    rho_mantle: float = 3.3
    # Horizontal AREA, so it scales as the radius squared: 1000 km^2 / 98.125^2. Still
    # the same 12 cells' worth of speckle it was at Earth radius.
    min_island_area_km2: float = 0.104


@dataclasses.dataclass
class Crust:
    crust_thickness: np.ndarray  # (N,) f32 km
    crust_density: np.ndarray    # (N,) f32 g/cm^3
    elevation: np.ndarray        # (N,) f32 m, isostatic, pre-erosion
    land_mask: np.ndarray        # (N,) bool
    sea_level_m: float
    land_fraction: float
    landmass_areas: np.ndarray   # (M,) f64 km^2, descending


@njit(cache=True)
def label_components(mask, nbr, labels, stack):
    """Connected components over the cell graph. Works on any tiling."""
    n = mask.shape[0]
    k = nbr.shape[1]
    for i in range(n):
        labels[i] = -1
    n_lab = 0
    for seed in range(n):
        if not mask[seed] or labels[seed] >= 0:
            continue
        labels[seed] = n_lab
        sp = 0
        stack[sp] = seed
        sp += 1
        while sp > 0:
            sp -= 1
            c = stack[sp]
            for s in range(k):
                j = nbr[c, s]
                if j < 0 or not mask[j] or labels[j] >= 0:
                    continue
                labels[j] = n_lab
                stack[sp] = j
                sp += 1
        n_lab += 1
    return n_lab


def separate_inland_basins(land, neighbours, area, labels, stack):
    """Ocean is the largest connected body of water, not everything below sea level.

    `land = elevation > sea_level` is a threshold with no notion of connectivity, so every
    closed depression that happens to sit below the datum was being handed to the ocean --
    including deep interior basins with no path to the sea. Measured on an n=128 world:
    41 water bodies, of which one holds 98.95% of the water and the other 40 are inland.
    They are 2.59% of land area, which is roughly Earth's lake coverage, and Stage 6 was
    finding two lakes because these forty had already been classified away from it.

    Marking them as land is the whole fix. They are closed depressions on the land surface,
    so the flow router fills them, and Stage 6's existing machinery takes it from there --
    spill point, endorheic water balance, level and depth. No lake code is needed here.

    Mutates `land` in place. Returns (basin count, basin area).
    """
    water = np.ascontiguousarray(~land)
    n_lab = label_components(water, neighbours, labels, stack)
    if n_lab <= 1:
        return 0, 0.0
    sizes = np.zeros(n_lab, dtype=np.float64)
    np.add.at(sizes, labels[water], area[water].astype(np.float64))
    inland = water & (labels != int(np.argmax(sizes)))
    land[inland] = True
    return n_lab - 1, float(area[inland].sum())


def reflood(elevation, neighbours, area, labels, stack, sea_level=0.0):
    """Re-derive the land mask from an eroded surface. Returns a fresh mask.

    THE LAND MASK IS NOT A CONSTANT OF THE PIPELINE, and treating it as one leaves the
    coastline describing a landscape that no longer exists. Stage 1 solves it, then Stages
    2 and 5 move the surface underneath it by hundreds of metres. Glacial erosion is the
    reason it matters: ice is driven by its own surface slope and will gouge below its
    outlet, which is exactly what carves a fjord, and the water then has to be let in.

    Measured on an n=128 world after the full pipeline: 20.9% of land sat below the datum,
    and 99.9% of THAT was connected to the sea -- median depth 106 m, median latitude 63
    degrees. Glacially overdeepened coastal troughs, correctly cut and left dry. Flooding
    them takes the land fraction from 29.6% to 23.4%.

    Same two rules as Stage 1, applied to the new surface, so the answer stays consistent:
    below the datum is water, and ocean is the largest connected body of it. Whatever is
    below the datum and cannot reach the sea stays land and becomes a lake candidate for
    Stage 6 -- see `separate_inland_basins`, which does that half.
    """
    land = np.ascontiguousarray(np.asarray(elevation) > np.float32(sea_level))
    separate_inland_basins(land, neighbours, area, labels, stack)
    return land


def landmass_areas(land, neighbours, area, labels, stack):
    """Areas of the connected landmasses, largest first.

    Recomputed after the basins join, not carried over from the pruning pass: a basin that
    touches two islands merges them into one landmass, so the areas from before the join do
    not describe the same set of components -- and they no longer sum to the land area.
    """
    n_lab = label_components(np.ascontiguousarray(land), neighbours, labels, stack)
    sizes = np.zeros(n_lab, dtype=np.float64)
    np.add.at(sizes, labels[land], area[land].astype(np.float64))
    return np.sort(sizes)[::-1]


def area_quantile(values, area, q):
    """Value below which fraction `q` of the total AREA lies.

    Area-weighted rather than cell-counted: cell areas vary across the grid, and on a
    lat/lon raster they vary enormously, so a plain np.quantile would put sea level in the
    wrong place.
    """
    order = np.argsort(values)
    cum = np.cumsum(area[order].astype(np.float64))
    cum /= cum[-1]
    return float(values[order[min(int(np.searchsorted(cum, q)), values.size - 1)]])


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def generate_crust(grid, params: CrustParams = None, seed: int = 0,
                   verbose: bool = False) -> Crust:
    p = params or CrustParams()
    xyz = np.ascontiguousarray(grid.xyz)
    n = xyz.shape[0]

    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    # Sub-seeds derived by NAME, not drawn sequentially from one stream. Otherwise changing
    # the relief octaves would reshuffle the continents and no A/B comparison would hold.
    perm_crust = noise.permutation(noise.sub_seed("crust", seed))
    perm_relief = noise.permutation(noise.sub_seed("relief", seed))

    # 1. Continentality from warped low-frequency noise, independent of any plate layout.
    field = np.empty(n, dtype=np.float64)
    noise.warped_fbm(xyz, perm_crust, noise.GRAD3, p.crust_octaves, p.crust_frequency,
                     2.0, 0.5, p.warp_amp, p.warp_frequency, field)
    log("continentality field")

    # Threshold at the area-weighted quantile that hits the target crust fraction, so the
    # continental share of the surface is what was asked for rather than whatever the noise
    # distribution happened to give.
    thr = area_quantile(field, grid.area, 1.0 - p.continental_fraction)
    c = _smoothstep(thr - p.transition, thr + p.transition, field)

    # 2. Thickness, plus texture so the interior is not a plateau.
    relief = np.empty(n, dtype=np.float64)
    noise.fbm(xyz, perm_relief, noise.GRAD3, p.relief_octaves, p.relief_frequency,
              2.0, 0.5, relief)
    thickness = (p.t_oceanic_km + c * (p.t_continental_km - p.t_oceanic_km)
                 + c * relief * p.relief_km)
    density = p.rho_oceanic + c * (p.rho_continental - p.rho_oceanic)
    log("thickness and density")

    # 3. Airy isostasy. K is an arbitrary datum -- sea level is solved for below.
    elevation = (thickness * (1.0 - density / p.rho_mantle) * 1000.0)

    # 4. Solve sea level for the target land fraction, then rebase the datum onto it so
    #    `elevation` means metres above sea level. Everything downstream wants that.
    datum = area_quantile(elevation.astype(np.float32), grid.area,
                          1.0 - p.target_land_fraction)
    elevation = (elevation - datum).astype(np.float32)
    sea_level = 0.0
    land = elevation > np.float32(sea_level)
    log(f"sea level datum {datum:,.0f} m above the isostatic zero")

    # 5. Prune speckle islands. Threshold is in km^2, never cells -- a cell-denominated
    #    threshold prunes differently at different n and breaks multi-resolution runs.
    labels = np.empty(n, dtype=np.int32)
    stack = np.empty(n, dtype=np.int32)
    n_lab = label_components(land, np.ascontiguousarray(grid.neighbours), labels, stack)
    sizes = np.zeros(n_lab, dtype=np.float64)
    np.add.at(sizes, labels[land], grid.area[land].astype(np.float64))
    too_small = sizes < p.min_island_area_km2
    pruned = land & too_small[np.clip(labels, 0, None)]
    land[pruned] = False
    # Drown them properly. Clearing the mask alone left the elevation above the datum, so
    # a pruned islet was ocean by mask and land by height -- and every downstream stage
    # reads one or the other. This never fired before the rescale: at Earth radius the
    # 1000 km^2 threshold was smaller than a single cell on the test fixtures, so nothing
    # was ever pruned and the inconsistency stayed latent.
    elevation[pruned] = np.float32(sea_level)
    log(f"{n_lab:,} landmasses, pruned {int(too_small.sum()):,} below "
        f"{p.min_island_area_km2:,.4g} km^2")

    # 6. Everything below the datum that cannot reach the sea is an inland basin, not ocean.
    #    After pruning, so a drowned islet cannot leave a one-cell "basin" behind.
    n_basins, basin_km2 = separate_inland_basins(
        land, np.ascontiguousarray(grid.neighbours), grid.area, labels, stack)
    log(f"{n_basins:,} inland basins reclassified as land ({basin_km2:,.4g} km^2)")

    kept = landmass_areas(land, np.ascontiguousarray(grid.neighbours), grid.area,
                          labels, stack)
    land_fraction = float(grid.area[land].sum() / grid.area.sum())

    return Crust(
        crust_thickness=thickness.astype(np.float32),
        crust_density=density.astype(np.float32),
        elevation=elevation,
        land_mask=land,
        sea_level_m=sea_level,
        land_fraction=land_fraction,
        landmass_areas=kept,
    )


@dataclasses.dataclass
class TectonicCrust(Crust):
    """Everything `Crust` carries, plus what the plates produced."""
    plates: object = None
    plate_id: np.ndarray = None
    boundary_type: np.ndarray = None
    boundary_stress: np.ndarray = None
    margin_class: np.ndarray = None
    uplift: np.ndarray = None


def generate_crust_tectonic(grid, params: CrustParams = None,
                            plate_params=None, seed: int = 0,
                            uplift_contrast_cap=None, verbose: bool = False):
    """Full Stage 1: crust from noise, tectonics from plates, elevation from isostasy.

    The ordering is the load-bearing decision. Crust type comes first and independently,
    from 3D noise; plates are laid over it afterwards and never decide where continents go.
    Assigning crust type per plate is geologically wrong -- six of Earth's seven major plates
    carry both types -- and it gives every continent a Voronoi silhouette at whole-continent
    scale.

    Sub-steps, following the spec in terrain-algorithms.md:
      1 build_thickness   warped 3D noise -> continental/oceanic thickness
      2 warp_margins      (folded into 1: the domain warp is applied to the whole field)
      3 seed_plates       Poisson-disk on the sphere, Lloyd-relaxed
      4 partition         spherical Voronoi
      5 assign_motion     Euler pole and rate per plate
      6 compute_boundaries  relative velocity -> convergent / divergent / transform
      7 apply_tectonic_thickness  crustal root under orogens, thinning at rifts
      8 isostatic_elevation
      9 resolve_sea_level  area-weighted quantile for target land fraction
     10 prune_islands
     10b separate_inland_basins  ocean is the largest CONNECTED water body
     11 classify_margins   passive vs active coast
    and derive_uplift as a separate operator, so uplift models can be swapped without
    regenerating plates.
    """
    from terrain import plates as plates_mod

    p = params or CrustParams()
    pp = plate_params or plates_mod.PlateParams()
    n = grid.cell_count
    xyz = np.ascontiguousarray(grid.xyz)

    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    perm_crust = noise.permutation(noise.sub_seed("crust", seed))
    perm_relief = noise.permutation(noise.sub_seed("relief", seed))

    # 1-2. Continentality, independent of the plates.
    field = np.empty(n, dtype=np.float64)
    noise.warped_fbm(xyz, perm_crust, noise.GRAD3, p.crust_octaves, p.crust_frequency,
                     2.0, 0.5, p.warp_amp, p.warp_frequency, field)
    thr = area_quantile(field, grid.area, 1.0 - p.continental_fraction)
    c = _smoothstep(thr - p.transition, thr + p.transition, field)
    is_continental = c > 0.5
    log(f"continentality: {is_continental.mean():.1%} of surface")

    relief = np.empty(n, dtype=np.float64)
    noise.fbm(xyz, perm_relief, noise.GRAD3, p.relief_octaves, p.relief_frequency,
              2.0, 0.5, relief)
    thickness = (p.t_oceanic_km + c * (p.t_continental_km - p.t_oceanic_km)
                 + c * relief * p.relief_km)
    density = p.rho_oceanic + c * (p.rho_continental - p.rho_oceanic)

    # 3-6. Plates.
    pl = plates_mod.build_plates(grid, pp, seed)
    n_bound = int((pl.boundary_type != plates_mod.NONE).sum())
    log(f"{pl.sites.shape[0]} plates, {n_bound:,} boundary cells")

    # 7. Crustal root under orogens; thinning at rifts.
    conv = pl.boundary_type != plates_mod.NONE
    scale = np.percentile(np.abs(pl.boundary_stress[conv]), 90) if conv.any() else 1.0
    near = pl.nearest_boundary
    near_type = np.zeros(n, dtype=np.uint8)
    near_stress = np.zeros(n, dtype=np.float32)
    ok = near >= 0
    near_type[ok] = pl.boundary_type[near[ok]]
    near_stress[ok] = pl.boundary_stress[near[ok]]
    strength = np.clip(np.abs(near_stress) / max(scale, 1e-6), 0.0, 1.5)
    # Same world-scale rescale derive_uplift does; see PlateParams.for_radius.
    decay_km = pp.for_radius(pl.radius_km).orogen_decay_km if pl.radius_km \
        else pp.orogen_decay_km
    decay = np.exp(-pl.boundary_dist_km / decay_km)

    thickness = thickness + np.where(
        near_type == plates_mod.CONVERGENT,
        pp.convergent_thickening_km * decay * strength * c, 0.0)
    thickness = thickness - np.where(
        near_type == plates_mod.DIVERGENT,
        pp.divergent_thinning_km * decay * strength * c, 0.0)
    thickness = np.maximum(thickness, 1.0)
    log("tectonic thickening applied")

    # 8-10. Isostasy, sea level, pruning -- identical to the throwaway path.
    elevation = thickness * (1.0 - density / p.rho_mantle) * 1000.0
    datum = area_quantile(elevation.astype(np.float32), grid.area,
                          1.0 - p.target_land_fraction)
    elevation = (elevation - datum).astype(np.float32)
    land = elevation > 0

    labels = np.empty(n, dtype=np.int32)
    stack = np.empty(n, dtype=np.int32)
    n_lab = label_components(land, np.ascontiguousarray(grid.neighbours), labels, stack)
    sizes = np.zeros(n_lab, dtype=np.float64)
    np.add.at(sizes, labels[land], grid.area[land].astype(np.float64))
    too_small = sizes < p.min_island_area_km2
    pruned = land & too_small[np.clip(labels, 0, None)]
    land[pruned] = False
    elevation[pruned] = np.float32(0.0)  # drown them; see generate_crust
    # See separate_inland_basins: ocean is the largest connected water body, and the rest
    # are lakes-to-be that Stage 6 will fill. After pruning, as in the throwaway path.
    n_basins, basin_km2 = separate_inland_basins(
        land, np.ascontiguousarray(grid.neighbours), grid.area, labels, stack)
    kept = landmass_areas(land, np.ascontiguousarray(grid.neighbours), grid.area,
                          labels, stack)
    log(f"{n_lab:,} landmasses, {int(too_small.sum()):,} pruned, "
        f"{n_basins:,} inland basins ({basin_km2:,.4g} km^2)")

    # 11 + uplift.
    margin = plates_mod.classify_margins(grid, land, pl)
    uplift = plates_mod.derive_uplift(pl, is_continental, land, pp,
                                      contrast_cap=uplift_contrast_cap)
    log(f"uplift {uplift[land].min():.0f}..{uplift[land].max():.0f} m/Myr")

    return TectonicCrust(
        crust_thickness=thickness.astype(np.float32),
        crust_density=density.astype(np.float32),
        elevation=elevation,
        land_mask=land,
        sea_level_m=0.0,
        land_fraction=float(grid.area[land].sum() / grid.area.sum()),
        landmass_areas=kept,
        plates=pl,
        plate_id=pl.plate_id,
        boundary_type=pl.boundary_type,
        boundary_stress=pl.boundary_stress,
        margin_class=margin,
        uplift=uplift,
    )


def isostasy_check(crust: Crust, area) -> dict:
    """Mean continental surface minus mean abyssal depth.

    Report it, but do NOT treat ~4.45 km as a correctness target. That figure is what the
    dry-column Airy formula prints; the real-world 4.5 km it gets compared against is
    water-loaded. Redo the balance with a water column at these densities and the gap is
    ~6 km, so the simplified formula and the physics disagree, and anchoring parameter
    tuning to the coincidence would be a mistake.
    """
    land, sea = crust.land_mask, ~crust.land_mask
    z = crust.elevation
    mean_cont = float(np.average(z[land], weights=area[land])) if land.any() else np.nan
    abyssal = z[sea]
    deep = abyssal <= np.percentile(abyssal, 25) if sea.any() else np.zeros(0, bool)
    mean_abyss = float(abyssal[deep].mean()) if sea.any() else np.nan
    return {
        "mean_continental_m": mean_cont,
        "mean_abyssal_m": mean_abyss,
        "gap_km": (mean_cont - mean_abyss) / 1000.0,
        "freeboard_m": mean_cont - crust.sea_level_m,
    }
