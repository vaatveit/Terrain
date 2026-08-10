# Terrain Generation Algorithms: A Survey for a Batch Generator

**Context:** whole-map batch generation, offline, optimizing for realism rather than latency.

That constraint matters more than it might seem. Chunk-on-demand generation forces you into *local, stateless* functions — you can only use algorithms where `height(x, y)` depends on nothing but `(x, y)` and a seed. That rules out essentially every physical simulation, which is why on-the-fly terrain almost always looks like noise. Batch generation lifts that restriction, so the whole class of global, iterative, physically-motivated methods opens up. Those are where realism actually comes from.

The single most useful framing: **noise gives you shape; erosion gives you realism.** Real landscapes are not fractal surfaces — they are the residue of water, ice, and gravity moving material downhill for millions of years. The signature of that process is a *dendritic drainage network* with valleys that branch like a tree and obey measurable scaling laws. No amount of octave-stacking produces one.

---

## Category 1 — Stochastic base heightfields

Fast, controllable, uncorrelated with any physical process. Use these to seed a simulation, not to be the final terrain.

| Method | Character | Notes |
|---|---|---|
| **Value / Perlin / Simplex noise** | smooth, isotropic blobs | Simplex/OpenSimplex avoids Perlin's axis-aligned artifacts in ≥3D |
| **fBm** (sum of octaves) | classic "rolling hills" | lacunarity ≈ 2, gain ≈ 0.5; the default everyone starts with |
| **Ridged multifractal** | sharp crests, mountain ranges | `1 − |noise|`, then multiply octaves by previous-octave weight |
| **Billow** | rounded, dune-like | `|noise|` |
| **Hybrid / heterogeneous multifractal** | roughness varies with altitude | Musgrave's key idea: valleys smooth, peaks rough. Much more plausible than plain fBm |
| **Domain warping** | folded, geologically "flowy" | feed noise into the *coordinates* of noise; 1–2 levels; cheapest realism-per-line-of-code in the whole list |
| **Worley / cellular** | craters, cracked flats, cell structure | also the basis for Voronoi region decomposition |
| **Diamond-square / midpoint displacement** | fractal, grid artifacts | mostly historical; superseded |
| **Fault formation** | plateaus, scarps, escarpments | repeatedly bisect the map with a random line, raise one side |
| **Particle deposition** | conical piles, volcanoes | drop particles, let them roll to rest |
| **Spectral synthesis (inverse FFT)** | direct control of the power spectrum | real terrain is roughly 1/f^β with β ≈ 2; useful for matching a target PSD exactly |

**Why these alone fail:** they are statistically isotropic and have no concept of connectivity. Water has to flow somewhere, and on a pure noise field it doesn't — you get spurious pits everywhere, valleys that dead-end, and ridgelines that don't connect into a coherent divide network.

---

## Category 2 — Tectonic / structural (the largest scale)

Determines where continents, ranges, trenches, and island arcs go. Output is usually an **uplift field** `U(x, y)` plus a crust-type mask, both fed into Category 3.

- **Plate tectonic simulation.** Voronoi-partition the domain into plates, assign each a velocity, then simulate: uplift at convergent boundaries, trenches and volcanic arcs at subduction zones, rifts and mid-ocean ridges at divergent boundaries, transform faults elsewhere. Expensive to get right but nothing else produces mountain ranges with plausible *arc shapes* and correct relationships to coastlines.
- **Hand-authored or noise-based uplift.** Much simpler: paint or generate a low-frequency uplift field directly. If you're not simulating tectonics, this is what you do instead — and it's a legitimate choice, since erosion does most of the visual work regardless.
- **Hotspots.** Point sources of uplift advected along a plate velocity field produce island chains (Hawaii).
- **Crustal isostasy / flexure.** Optional refinement: loading depresses crust, unloading rebounds. Matters for basin formation.
- **Bathymetry.** Continental shelf → slope → abyssal plain, often generated as a function of distance from the coastline plus crust age.

---

## Category 3 — Erosion and landscape evolution

**This is where realism lives.** Two broad families.

### 3a. Cellular / fluid-sim erosion (graphics tradition)

- **Thermal erosion (talus).** If the slope between two cells exceeds a talus angle (~30–35°, the angle of repose), move material downhill. Converts cliffs to scree slopes. Cheap, runs to convergence quickly, essential as a finishing pass.
- **Grid hydraulic erosion.** Each cell tracks water, dissolved sediment, and velocity. Rain → flow downhill → transport capacity ∝ slope × velocity → erode if under capacity, deposit if over → evaporate. The classic.
- **Virtual pipes model.** Cells connected by pipes with flux; solves shallow-water-ish flow explicitly. Handles standing water and lakes naturally and maps well to GPU. The standard grid method in graphics.
- **Droplet / particle erosion.** Spawn thousands of individual droplets with position, velocity, water, and sediment; each carves a path. Very simple to implement, produces convincing channels, popular in hobbyist projects. Weaknesses: stochastic noise in the output, no natural lake formation, and results depend heavily on droplet count.
- **Full shallow-water equations.** Physically accurate, expensive, usually overkill unless you also want flood/river dynamics as an output.

### 3b. Landscape evolution models (geomorphology tradition)

This family is less known in game dev and is probably the most important thing on this list for your goals.

- **Stream power incision.** The governing equation of fluvial landscape evolution:

  ```
  ∂z/∂t = U − K · A^m · S^n   (+ hillslope term)
  ```

  where `U` is tectonic uplift, `A` is upstream drainage area, `S` is local slope, `K` is erodibility, and `m/n ≈ 0.5` typically. Solved on a flow graph rather than a grid stencil.

- **Hillslope diffusion.** `∂z/∂t = D∇²z`. Rounds ridgelines and fills hollows. Combined with stream power it reproduces the characteristic real-world profile: **convex hilltops, concave valley floors**, with the transition at a scale set by `D/K`.

- **Implicit O(n) solvers.** Braun & Willett's method (the basis of "FastScape") solves stream power implicitly in linear time and is unconditionally stable — you can take million-year timesteps. This is the reason the approach is practical for large batch maps: you can run a 4096² domain to topographic steady state in seconds to minutes, not hours.

- **Glacial erosion.** Shallow-ice-approximation ice flow carving bedrock. Produces U-shaped valleys, cirques, arêtes, hanging valleys, fjords — a completely different signature from fluvial terrain, and unmistakably wrong if absent from high-latitude or high-altitude maps.

- **Aeolian.** Saltation-based sand transport and dune-field cellular automata (barchan, linear, star dunes depending on wind regime).

- **Coastal.** Wave notching at sea level, longshore drift, beach and spit deposition.

### 3c. Required plumbing

Erosion needs flow routing, and flow routing needs these:

- **Flow direction:** D8 (steepest of 8 neighbors — simple, but channels snap to 45° increments), D∞ (angular interpolation), or multiple-flow-direction (splits flow among downhill neighbors — better on hillslopes, worse in channels). A common compromise is MFD on hillslopes, D8 in channels.
- **Depression handling:** raw heightfields are full of spurious pits that trap flow. Either **fill** them (Priority-Flood is the fast, correct algorithm; Planchon–Darboux is the classic) or **breach** them (carve an outlet — often more realistic since it preserves incision). You will call this every erosion iteration, so it must be fast.
- **Flow accumulation:** topological sort of the flow graph, then accumulate upstream area in one pass.
- **Endorheic basins:** genuine closed basins (Caspian, Great Basin) should *not* be filled away. Distinguish them by whether the basin's water budget exceeds evaporation.

---

## Category 4 — Hydrology as a first-class input

Instead of letting rivers emerge from erosion, you can **construct the river network first and build terrain around it**. Génevaux et al.'s hydrology-based approach grows a river tree with realistic branching angles and Horton–Strahler ordering, then synthesizes a surface consistent with it.

Trade-off: far more authorial control and guaranteed-good drainage, but the terrain between rivers is interpolated rather than simulated, so it can look smooth. Worth having in the testbed as a distinct strategy.

Regardless of approach, the finishing hydrology pass includes: lake fill to spill elevation, river width from discharge (roughly `w ∝ Q^0.5`), meander migration for low-gradient reaches, deltas at river mouths, alluvial fans at range fronts.

---

## Category 5 — Climate and biomes

Geometry alone isn't "realistic terrain" once anything is rendered on it. And climate feeds *back* into geometry.

- **Temperature:** latitude insolation + elevation lapse rate (~6.5 °C/km) + continentality (distance to ocean) + seasonal amplitude.
- **Prevailing wind:** by latitude band — trade winds, westerlies, polar easterlies (Hadley/Ferrel/Polar cells).
- **Orographic precipitation:** advect moisture packets along the wind field; precipitate as air is forced up over terrain; dry out on the lee side. This produces **rain shadows**, which is one of the highest-value/lowest-cost realism features available. Compare the Olympics vs. eastern Washington.
- **Biome assignment:** the Whittaker diagram (mean annual temperature × precipitation → biome) is the simple approach; Köppen–Geiger is the more rigorous one.

**The feedback loop:** precipitation should modulate erodibility `K`. Wet windward slopes incise deeply; arid lee slopes retain relief. So the pipeline is properly iterative — erode, compute climate on the new topography, re-erode with updated `K`. Two or three coupled passes is usually enough.

---

## Category 6 — Discrete / graph-based decomposition

- **Voronoi + Delaunay dual graphs** (Amit Patel's polygon map approach): elevation from coastline distance, moisture routed along rivers, biomes per polygon. Excellent for game-scale strategic maps, deliberately non-photoreal.
- **Poisson-disk sampling** for feature placement (settlements, vegetation, points of interest) with minimum-distance guarantees.
- **Irregular meshes for simulation.** Landscape evolution models often run on TINs rather than regular grids — fewer nodes for the same fidelity, and no grid-aligned channel artifacts.

---

## Category 7 — Data-driven and learned

- **Example-based synthesis from real DEMs.** Patch-based texture synthesis over SRTM/Copernicus tiles, guided by a user sketch or a coarse control field. Inherits real-world statistics for free.
- **Conditional GANs / diffusion.** Sketch-to-terrain and coarse-to-fine amplification. Guérin et al.'s cGAN work is the reference point in graphics; diffusion-based heightfield super-resolution is the current direction.
- **Real DEMs as ground truth.** Even if you never synthesize from data, get a corpus of real elevation tiles (SRTM 30 m global, Copernicus GLO-30, USGS 3DEP for high-res US). You need them for Section "Validation" below.

---

## Category 8 — Amplification and detail

Simulation is expensive, so run it coarse and add detail afterward:

- Noise modulated by local slope, curvature, and drainage area — rough on steep faces, smooth on valley floors, never uniform.
- Rock strata as banded modulation aligned to a dip plane.
- Material-aware detail if you carry a bedrock/regolith layer stack.
- Overhangs, arches, and caves require leaving the heightfield representation entirely (see below).

---

## Representation

| Representation | Overhangs? | Notes |
|---|---|---|
| Regular heightfield grid | no | simplest, GPU-friendly, fine for 95% of terrain |
| Layered stacks (bedrock / sediment / water) | no | needed for erosion with differential rock hardness |
| TIN / irregular mesh | no | adaptive resolution, avoids grid artifacts in channels |
| Voxel / SDF | yes | caves, arches, sea stacks; much heavier |

For a first system: single heightfield, with a second channel for sediment depth. Add layers only when you want differential erodibility.

---

## Recommended pipeline

```
0. Domain setup            grid/TIN, resolution, planar or spherical, seed
1. Crust generation     →  plates, crustal thickness, isostatic elevation,
                           sea level, land mask, margin class, uplift field U
                           (see "Stage 1 spec" below)
2. Landscape evolution  →  iterate: stream power + hillslope diffusion + uplift
                           (this is the realism engine; run to near steady state)
3. Climate pass         →  temperature, wind, orographic precipitation
4. Coupled re-run       →  modulate K by precipitation, repeat step 2 (×2–3)
5. Secondary erosion    →  glacial above snowline, aeolian in arid zones,
                           coastal at sea level, thermal talus everywhere
6. Hydrology finalize   →  depression fill/breach, lakes, rivers, widths, deltas
7. Biomes               →  Whittaker or Köppen from temperature × precipitation
8. Amplification        →  slope/curvature-modulated detail; optional ML refine
9. Export + metrics
```

Stages 2 and 5 dominate runtime. Stage 4 is the one most implementations skip and the one that most improves plausibility at continental scale.

---

## Validation: how to actually *test* the algorithms

Since comparing approaches is the point of the program, build the metrics module early. Each of these is a scalar or curve you can compute on generated terrain and on real DEM tiles of a comparable region, then compare:

- **Slope histogram** — real terrain has a characteristic distribution; pure fBm is visibly too uniform.
- **Hypsometric curve** — elevation vs. cumulative area. Distinguishes young/steep from mature/eroded landscapes.
- **Power spectral density** — log-log slope should be roughly −2 and *scale-dependent*, not perfectly linear.
- **Hack's law** — main channel length vs. drainage area: `L ∝ A^h`, `h ≈ 0.57`. Pure noise fails this badly.
- **Horton's laws** — bifurcation, length, and area ratios between Strahler stream orders. Real networks cluster tightly around known values.
- **Slope–area relation** — in channels, `S ∝ A^−θ` with concavity `θ ≈ 0.4–0.6`. The cleanest single diagnostic that an erosion model is behaving correctly.
- **Drainage density** — total channel length per unit area; should vary with climate.

A terrain that matches on slope histogram but fails Hack's law is exactly the failure mode of noise-only generation — it looks right in a screenshot and wrong the moment you follow a valley.

---

## Testbed architecture notes

- **Operators over a field stack.** Every algorithm is `fn(FieldStack, Params) -> FieldStack`, where the stack holds named 2D arrays (`elevation`, `sediment`, `water`, `precipitation`, `uplift`, `flow_accum`, …). Pipelines become declarative config, which makes A/B comparison trivial.
- **Deterministic seeding per stage** so you can vary one stage while holding the rest fixed.
- **Checkpoint between stages** — stage 2 is slow; you don't want to re-run it to test five climate models.
- **Multi-resolution** — prototype at 512², validate at 2048²+. Note that some algorithms are not scale-invariant, so parameters won't transfer directly; erosion coefficients in particular need rescaling.
- **Metrics + side-by-side hillshade rendering** in the same report output.

---

## Key literature

- **Galin et al., "A Review of Digital Terrain Modeling" (Eurographics STAR, 2019)** — the single best entry point; start here.
- Musgrave, Kolb & Mace, "The Synthesis and Rendering of Eroded Fractal Terrains" (SIGGRAPH 1989) — origin of thermal + hydraulic erosion in graphics.
- Ebert, Musgrave, Peachey, Perlin & Worley, *Texturing & Modeling: A Procedural Approach* — the multifractal chapters.
- Mei, Decaudin & Hu, "Fast Hydraulic Erosion Simulation and Visualization on GPU" (2007) — virtual pipes model.
- Braun & Willett (2013) — O(n) implicit stream power solver.
- Cordonnier et al., "Large Scale Terrain Generation from Tectonic Uplift and Fluvial Erosion" (2016) — brings landscape evolution models into graphics.
- Génevaux et al., "Terrain Generation Using Procedural Models Based on Hydrology" (SIGGRAPH 2013).
- Barnes, Lehman & Mulla, "Priority-Flood" (2014) — depression filling.
- Guérin et al., "Interactive Example-Based Terrain Authoring with Conditional GANs" (2017).
- Cordonnier et al., "Forming Terrains by Glacial Erosion" (2023).

---

# Stage 1 spec: crust generation

The operator that produces the landmass outlines. It is the root of the pipeline DAG — it requires nothing but the domain, and everything downstream requires something it provides.

## Design principle

**Generate crustal thickness, not elevation.** Elevation falls out of isostasy; sea level is a threshold applied afterward. This matters because it makes the coastline a *consequence* rather than a drawn line, which means the continental shelf, the shelf break, and the relationship between coast and mountain all come out consistent for free. It also makes sea level a free knob afterward — the same world at a glacial lowstand costs one parameter change, not a regeneration.

## Field stack contract

```
FieldStack
  domain    { width, height, cell_size_km, wrap_x, wrap_y }
  rasters   name -> 2D array (typed)
  tables    name -> record list          (plates, later: lakes, rivers)
  scalars   name -> value                (sea_level_m, land_fraction)
  history   ordered [(operator, params, seed)]   -- provenance for repro
```

```
Operator
  name      "generate_crust"
  requires  []                     -- root operator
  provides  [see below]
  params    CrustParams
  run(stack, params, rng) -> stack
```

Declaring `requires` / `provides` lets the harness topologically validate a pipeline config *before* spending twenty minutes on erosion, and lets it know which stages to invalidate when you change one parameter.

## Outputs

| Field | Type | Unit | Purpose |
|---|---|---|---|
| `plate_id` | u16 raster | — | which plate owns each cell |
| `plates` | table | — | per plate: site, is_continental, velocity |
| `crust_thickness` | f32 raster | km | the master field; everything else derives from it |
| `crust_density` | f32 raster | g/cm³ | continental ≠ oceanic; needed for correct isostasy |
| `boundary_type` | u8 raster | enum | none / convergent / divergent / transform |
| `boundary_stress` | f32 raster | mm/yr | signed normal component of relative plate motion |
| `elevation` | f32 raster | m | isostatic, pre-erosion |
| `sea_level_m` | scalar | m | solved, not assumed |
| `land_mask` | u8 raster | — | derived from elevation vs. sea level |
| `margin_class` | u8 raster | enum | inland / passive / active — drives coastal realism downstream |
| `uplift` | f32 raster | m/Myr | the `U` term for stage 2 |

`uplift` is arguably a separate operator (`derive_uplift`) that consumes `boundary_type` and `boundary_stress`. Splitting it is worthwhile: it lets you swap uplift models without regenerating plates.

## Sub-steps

| # | Sub-step | What it does |
|---|---|---|
| 1 | `seed_plates` | Poisson-disk sample sites, Lloyd-relax 2–3× for even plate sizes |
| 2 | `partition` | Voronoi rasterize → `plate_id` |
| 3 | `classify_plates` | continental/oceanic to hit target fraction; spatially correlated; border plates forced oceanic |
| 4 | `assign_motion` | velocity per plate |
| 5 | `compute_boundaries` | per boundary cell, decompose relative velocity into normal/tangential → type + stress |
| 6 | `build_thickness` | base thickness by type, smooth transition across boundaries, thicken at convergent, thin at divergent |
| 7 | `warp_boundary` | domain-warped fBm added to thickness near transitions; amplitude keyed to margin type |
| 8 | `isostatic_elevation` | thickness + density → elevation |
| 9 | `enforce_edge_ocean` | margin falloff, then flood-fill from border and demote any land touching it |
| 10 | `resolve_sea_level` | quantile search on the elevation histogram for target land fraction |
| 11 | `prune_islands` | remove components below minimum area |
| 12 | `classify_margins` | coastal cells → passive/active from nearest boundary type |

Each is independently testable, which is the point of the decomposition — sub-step 7 is where most of the tuning happens and you don't want to re-run 1–6 to iterate on it.

## Isostasy

Airy balance over a compensation depth gives:

```
z = K + t · (1 − ρ_crust / ρ_mantle)
```

With ρ_mantle = 3.3, continental crust at t = 35 km / ρ = 2.8 gives 5.30 km; oceanic at t = 7 km / ρ = 2.9 gives 0.85 km. The 4.45 km difference matches the real gap between mean continental surface and abyssal plain closely enough that you don't need to fudge it. `K` is a datum offset — set it arbitrarily and let sub-step 10 find sea level.

*Refinement worth adding later:* water loading depresses submerged columns further, which deepens ocean basins relative to this simple form. Ignoring it costs you a slightly-too-shallow ocean, which the sea-level quantile search partly absorbs.

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `plate_count` | 18 | 12–30; fewer = bigger, blockier continents |
| `lloyd_iterations` | 3 | more = more uniform plate sizes, less natural variety |
| `continental_fraction` | 0.40 | of *surface*, not of land — the rest drowns as shelf |
| `clustering` | 0.6 | 0 = confetti continents, 1 = one supercontinent |
| `border_margin` | 0.12 | fraction of map width; keep wide or you chop continents |
| `transition_width_km` | 150 | continental slope width |
| `t_continental_km` | 35 | |
| `t_oceanic_km` | 7 | |
| `rho_continental` | 2.8 | |
| `rho_oceanic` | 2.9 | denser — this is why ocean floor sits low |
| `rho_mantle` | 3.3 | |
| `warp_octaves` | 6 | fewer octaves reads as fake immediately |
| `warp_amp_passive_km` | 40 | smoother coasts |
| `warp_amp_active_km` | 90 | rougher, more indented |
| `target_land_fraction` | 0.29 | Earth value |
| `min_island_area_cells` | 12 | below this it's speckle, not archipelago |
| `convergent_thickening_km` | +15 | crustal root under orogens |
| `divergent_thinning_km` | −8 | rift and ridge thinning |

**Sub-seed derivation matters.** Derive each sub-step's RNG from `hash(stage_seed, substep_name)` rather than drawing sequentially from one stream. Otherwise changing `warp_octaves` reshuffles the plate layout and you can't A/B anything.

## Uplift derivation

Uplift is what stage 2 consumes, so getting the shape right matters more than getting the magnitude right — erosion will find equilibrium against whatever you give it.

| Boundary case | Signature |
|---|---|
| Continental–continental convergent | Broad high orogen, uplift peaked on the boundary, decay ~200 km |
| Oceanic–continental convergent | Trench offshore, arc uplift on the continental side offset ~150 km inland |
| Oceanic–oceanic convergent | Island arc on the overriding plate, offset from the trench |
| Continental divergent | Subsiding graben flanked by uplifted rift shoulders |
| Oceanic divergent | Mid-ocean ridge; thermal subsidence with distance from axis |
| Transform | Little vertical motion; localized transpressional bumps at restraining bends |

Magnitude range: 500–5000 m/Myr for active orogens, ~0 for cratonic interiors. Modulate by `boundary_stress` and decay with distance from the boundary.

## Postconditions to assert

- No land cell within `border_margin` of the map edge.
- `land_fraction` within tolerance of target.
- No land component smaller than `min_island_area_cells`.
- `crust_thickness > 0` everywhere; `plate_id` covers the full domain.
- Every coastal cell has a `margin_class` other than inland.

## Stage-specific metrics

Beyond the global metrics: **landmass size distribution** (should be heavy-tailed — one or two dominant continents, then a steep drop, then many small islands; a flat distribution means `clustering` is too low), **coastline fractal dimension by box-counting, reported per margin class** (target roughly 1.05–1.15 passive, 1.2–1.3 active), and **shelf width distribution by margin class** (passive should be several times wider than active).
