# Terrain Generation Algorithms: A Survey for a Batch Generator

**Context:** whole-world batch generation on a **sphere**, run offline on one desktop machine, optimizing for realism rather than latency. Output is a static file.

That constraint matters more than it might seem. Chunk-on-demand generation forces you into *local, stateless* functions — you can only use algorithms where `height(p)` depends on nothing but the position and a seed. That rules out essentially every physical simulation, which is why on-the-fly terrain almost always looks like noise. Batch generation lifts that restriction, so the whole class of global, iterative, physically-motivated methods opens up. Those are where realism actually comes from.

The sphere is the second decision that shapes everything downstream, and it mostly *removes* work. A planar map needs an entire subsystem to guarantee every landmass is surrounded by ocean — margin falloffs, seed constraints, flood-fill demotion from the border — and that subsystem does nothing but paper over the fact that the domain has an edge. A globe has no edge, so all of it goes away. Several things also become physically correct rather than approximated: plate motion is genuinely rotation about an Euler pole, and noise sampled in 3D is seamless by construction with no pole pinching and no wrap seam.

See **Domain and representation** below for the grid, which is built and validated.

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

**On a sphere, sample 3D noise at each cell's unit vector** — `noise(x, y, z)` on the sphere's surface, not 2D noise over (lon, lat). This is both simpler and strictly better: no wrap seam, no pole pinching, no distortion to correct for. Domain warping warps the 3D position. Spectral synthesis is the one entry above that does *not* carry over directly; its spherical analogue is synthesis from a spherical-harmonic power spectrum.

**Why these alone fail:** they are statistically isotropic and have no concept of connectivity. Water has to flow somewhere, and on a pure noise field it doesn't — you get spurious pits everywhere, valleys that dead-end, and ridgelines that don't connect into a coherent divide network.

---

## Category 2 — Tectonic / structural (the largest scale)

Determines where continents, ranges, trenches, and island arcs go. Output is usually an **uplift field** `U(x, y)` plus a crust-type mask, both fed into Category 3.

- **Plate tectonic simulation.** Voronoi-partition the domain into plates, assign each a velocity, then simulate: uplift at convergent boundaries, trenches and volcanic arcs at subduction zones, rifts and mid-ocean ridges at divergent boundaries, transform faults elsewhere. Expensive to get right but nothing else produces mountain ranges with plausible *arc shapes* and correct relationships to coastlines.

  **On a sphere this is the real model, not an approximation.** Plate motion *is* rotation about an Euler pole: an axis unit vector plus an angular rate, with relative velocity at a boundary from the cross product. A planar velocity field is a hack for something that is genuinely rotational. Spherical Voronoi partitioning is likewise trivial — nearest seed by maximum dot product of unit vectors.
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

  **This form is detachment-limited: it erodes and never deposits.** That matters more than it sounds, because deltas, alluvial fans and basin fill are the visible output of sediment flux — they are not cosmetic finishing touches to be added in a later stage. Plan for a transport-limited or hybrid formulation (a Davy & Lague 2009 style sediment-flux term) inside this stage. Note also that with `n ≠ 1` the Braun–Willett linear solve needs a per-node Newton iteration; decide `n` before writing the solver.

- **Hillslope diffusion.** `∂z/∂t = D∇²z`. Rounds ridgelines and fills hollows. Combined with stream power it reproduces the characteristic real-world profile: **convex hilltops, concave valley floors**, with the transition at a scale set by `D/K`.

- **Implicit O(n) solvers.** Braun & Willett's method (the basis of "FastScape") solves stream power implicitly in linear time and is unconditionally stable — you can take million-year timesteps. This is the reason the approach is practical for large batch maps: you can run a 6M-cell global domain to topographic steady state in minutes, not hours.

- **Glacial erosion.** Shallow-ice-approximation ice flow carving bedrock. Produces U-shaped valleys, cirques, arêtes, hanging valleys, fjords — a completely different signature from fluvial terrain, and unmistakably wrong if absent from high-latitude or high-altitude maps.

- **Aeolian.** Saltation-based sand transport and dune-field cellular automata (barchan, linear, star dunes depending on wind regime).

- **Coastal.** Wave notching at sea level, longshore drift, beach and spit deposition.

### 3c. Required plumbing

Erosion needs flow routing, and flow routing needs these:

- **Flow direction:** on a square raster this is a real problem — D8 snaps channels to 45° increments, D∞ interpolates angularly, MFD splits flow among downhill neighbors (better on hillslopes, worse in channels), and the usual compromise is MFD on hillslopes with D8 in channels. **On the hex grid most of this evaporates.** Six evenly spaced neighbors, no orthogonal-versus-diagonal asymmetry, no 45° preference to correct for. Steepest descent over the six is enough. This is the single strongest practical argument for the hex tiling.
- **Depression handling:** raw heightfields are full of spurious pits that trap flow. Either **fill** them (Priority-Flood is the fast, correct algorithm; Planchon–Darboux is the classic) or **breach** them (carve an outlet — often more realistic since it preserves incision). You will call this every erosion iteration, so it must be fast. On a globe there is no map border to seed the flood from; seed it from the ocean cells instead. **Measured** at production resolution: 1.0 s of a 1.13 s iteration, i.e. 88% of flow-routing cost — dominant, but small enough in absolute terms that no every-*k*-iterations workaround is needed.
- **Flow accumulation:** topological sort of the flow graph, then accumulate upstream area in one pass. **Accumulate real cell *areas*, never cell counts.** Cell area varies ~0.9% (σ/mean) across the grid, and counting cells instead quietly biases Hack's law and the slope–area relation — the exact diagnostics this testbed exists to produce.
- **Endorheic basins:** genuine closed basins (Caspian, Great Basin) should *not* be filled away. Distinguish them by whether the basin's water budget exceeds evaporation.
- **Hillslope diffusion on an unstructured mesh** is a finite-volume sum, not a 5-point stencil: `∂z_i/∂t = (D/A_i) · Σ_j (z_j − z_i)/L_ij · e_ij`, with `e_ij` the shared Voronoi edge length, `L_ij` the centre-to-centre arc distance, and `A_i` the cell area. The Delaunay/Voronoi dual is orthogonal, which is what keeps this clean.

---

## Category 4 — Hydrology as a first-class input

Instead of letting rivers emerge from erosion, you can **construct the river network first and build terrain around it**. Génevaux et al.'s hydrology-based approach grows a river tree with realistic branching angles and Horton–Strahler ordering, then synthesizes a surface consistent with it.

Trade-off: far more authorial control and guaranteed-good drainage, but the terrain between rivers is interpolated rather than simulated, so it can look smooth. Worth having in the testbed as a distinct strategy.

Regardless of approach, the finishing hydrology pass includes: lake fill to spill elevation, river width from discharge (roughly `w ∝ Q^0.5`), and meander migration for low-gradient reaches. **Deltas and alluvial fans are not on this list** — they are depositional landforms that should emerge from the sediment-flux term in stage 2. Painting them on at the end is the "looks right in a screenshot" failure this document warns about elsewhere.

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
- **Poisson-disk sampling** for feature placement (settlements, vegetation, points of interest) with minimum-distance guarantees. On the sphere, measure the minimum distance as a great-circle arc.
- **Irregular meshes for simulation.** Landscape evolution models often run on TINs rather than regular grids — fewer nodes for the same fidelity, and no grid-aligned channel artifacts. **This is the path taken here**: the Goldberg tiling is an unstructured mesh with a fixed neighbour table, which is why grid-aligned channel artifacts never arise.

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

## Domain and representation

### The tiling — decided and built

**Class-I Goldberg grid (geodesic hexagons), frequency n = 785.** Cells are the spherical Voronoi regions of the vertices of a frequency-785 subdivided icosahedron: all hexagons except exactly 12 pentagons at the icosahedron corners, with uniform 6-neighbour connectivity.

`2πR / 4096 = 9.773 km` is what "4096 cells around the equator" means as a resolution, and packing hexagons of that spacing over the sphere needs ~6.17M cells — hence `10n²+2` at `n = 785`.

| Property | Measured |
|---|---|
| Cells | 6,162,252 |
| Mean spacing | 9.81 km (target 9.773) |
| Cells around the equator | 4094 (target 4096) |
| Cell area σ/mean | **0.89%** |
| Cell area max/min | 1.16 |
| Pentagons | 12 |
| Build time / storage | 34 s → 650 MB cached, 0.3 s to load |

Why hexagons rather than the alternatives:

- **Compactness.** Hexagons are the optimum. HEALPix pixels are diamonds and its polar-cap pixels are noticeably more distorted; cubed-sphere cells carry 1.3–1.5× area variation and eight corner singularities.
- **Uniform 6-neighbour connectivity**, which is what kills D8's 45° channel artifact (see Category 3c).
- **`n` is freely tunable**, so it hits 9.77 km exactly. HEALPix `NESTED` indexing needs a power-of-two `Nside`, which forces 3.1M cells (12.7 km) or 12.6M (6.4 km). H3 is worse still — resolution steps of 7× can't land near 6.2M, and its cells vary ~2× in area.
- **Proven for this workload.** MPAS, ICON and DYNAMICO all solve PDEs on icosahedral hex meshes for these reasons.

What you give up: no closed-form `(i, j)` index, so cells are addressed by a flat integer plus an explicit neighbour table. That is a non-issue here — Stage 2 is receiver-array/flow-graph based and wants exactly this shape. The one real loss is HEALPix's isolatitude rings and free spherical-harmonic transforms; see Validation.

**Two construction findings worth recording, because both were counter-intuitive:**

1. **Build the connectivity analytically, not by triangulation.** Within each icosahedron face the subdivision is a regular triangular lattice whose adjacency is arithmetic; stitching across the 30 shared edges and 12 shared corners happens for free once both faces resolve a shared point to the same global cell id. No Delaunay needed — which matters, because running Qhull on 6.2M points is slow and memory-hungry. Validate the analytic result against `scipy.spatial.ConvexHull` at low `n` (the hull of points on a sphere *is* their Delaunay triangulation), then trust it at production scale.
2. **Placement, not relaxation, is what controls uniformity.** Linear subdivision followed by normalisation bunches cells toward face centres: σ/mean 13.3%, max/min 2.05. That distortion is *structural*, and relaxation barely touches it — Lloyd moved σ/mean only 13.3% → 12.2% over 40 iterations, and naive neighbour-centroid smoothing made things actively worse (max/min 2.05 → 4.41 over 100 iterations) because equalising edge lengths collapses the 12 pentagons. Switching to **slerp (equal-arc) subdivision** fixed it in one change: σ/mean 13.3% → 0.89%, scale-invariant from n=64 to n=785. No relaxation pass is used.

### Field representation

| Representation | Overhangs? | Notes |
|---|---|---|
| Per-cell scalar fields over the tiling | no | the default here; flat `(N,)` arrays, one per named field |
| Layered stacks (bedrock / sediment / water) | no | needed for erosion with differential rock hardness |
| Voxel / SDF | yes | caves, arches, sea stacks; much heavier |

For a first system: a single elevation field with a second channel for sediment depth. Add layers only when you want differential erodibility.

### Output

Two artifacts, not one. **Native:** the flat per-cell field arrays plus `n` and the seed — the grid is fully regenerable from `n`, so nothing else need be stored. **Static upload file:** resample to an equirectangular GeoTIFF (8192 × 4096 is the natural match for 4096-at-the-equator). GeoTIFF rather than `.npy` because real DEMs arrive as GeoTIFF, so the metrics code runs unmodified against both generated and real terrain.

> **Superseded — the GeoTIFF was built, measured and removed.** The "metrics code runs unmodified on both" argument held literally and bought nothing usable; see Stage 9 below. The export is the native archive alone.

---

## Recommended pipeline

```
0. Domain setup            Goldberg grid at frequency n, seed  (built; cached to disk)
1. Crust generation     →  crustal thickness, isostatic elevation, plates,
                           sea level, land mask, margin class, uplift field U
                           (see "Stage 1 spec" below)
2. Landscape evolution  →  iterate: stream power + sediment flux + hillslope
                           diffusion + uplift  (the realism engine; run to near
                           steady state. Deposition belongs HERE, not in stage 6)
3. Climate pass         →  temperature, wind, orographic precipitation
4. Coupled re-run       →  modulate K by precipitation, repeat step 2 (×2–3)
5. Secondary erosion    →  glacial above snowline, aeolian in arid zones,
                           coastal at sea level, thermal talus everywhere
5b. Reflood             →  re-derive the coastline; ice cuts below its outlet and
                           the sea has to be let into what it carved
6. Hydrology finalize   →  depression fill/breach, lakes, rivers, channel widths
7. Biomes               →  Whittaker or Köppen from temperature × precipitation
8. Amplification        →  slope/curvature-modulated detail; optional ML refine
9. Export + metrics     →  native field arrays + manifest
```

Stages 2 and 5 dominate runtime. Stage 4 is the one most implementations skip and the one that most improves plausibility at continental scale.

**Build them in a different order than you run them.** The thesis of this document is that erosion produces realism, so the schedule should reflect that rather than working left to right — most of Stage 1's careful output gets overwritten by Stage 2 anyway. Order: grid ✅ → **metrics module** ✅ → throwaway Stage 1 ✅ → **Stage 2** ✅ → **real Stage 1** ✅ (plates, Euler poles, derived uplift; per-basin θ = 0.452, peaks 8.4 km) → **Stage 3 climate** ✅ (temperature, wind bands, orographic precipitation; 11.8 s at n=785) → **Stage 4 coupled loop** ✅ (discharge-weighted erosion; wet ground incises 52 m deeper, arid ground holds 228 m more relief) → **Stage 7 biomes** ✅ → **Stage 5 glacial** ✅ (the rest of secondary erosion is sub-grid) → **Stage 6 hydrology** ✅ (lakes, endorheic basins, rivers) → **Stage 8 amplification** ✅ → **Stage 9 export** ✅. **The pipeline is complete** — `scripts/generate_world.py` runs all nine stages end to end and writes a georeferenced world. See **[README.md](README.md)** for how to run it, what every parameter does, and what the outputs contain.

### Cell size: why the sphere shrank to 64.9 km

The grid is dimensionless. Radius scales `arc_dist`, `edge_len` and `area` and nothing else, so moving to 100 m cells changed no graph algorithm and no topology. It changed which processes sit above the grid scale — and the decisive number is not the angle of repose but **the channel head**.

Real channels initiate at drainage areas of 0.01–1 km². A 9.77 km cell is **82.7 km²**, so every land cell already carried more drainage than a channel head needs. There was no hillslope domain anywhere in the world; the entire landscape was channel network, and `D` was a numerical smoother wearing the name of soil creep. That single fact explains the whole cluster of symptoms recorded elsewhere in this document: drainage density of 1.0 cells against SRTM's 3–4 px, 4.40% of land cells as local maxima against SRTM's 0.14–0.26%, and hillslope diffusion moving 0.014 m against 4 km of relief. A 100 m cell is **0.0086 km²**, below the threshold, so hillslopes exist as terrain.

**Shrinking the planet is not the same as adding cells, and only one of them was affordable.** Global 100 m on Earth is `510.1×10⁶ km² / 0.0086 km²` ≈ **5.9×10¹³ cells** (`n ≈ 2.4M`), seven orders of magnitude past the 6.16M budget. Shrinking the sphere keeps the cell count and buys the resolution by giving up world size: 4πR² = 52,973 km², about the area of Croatia, with landmasses ~120 km across. That is the trade, stated plainly — this is a large island rendered properly, not a planet.

**What had to move with it, and what emphatically did not.** Three classes:

| class | examples | rule |
|---|---|---|
| Cell-scale | `min_island_area_km2`, `min_lake_area_km2`, `channel_threshold_km3_per_yr`, `slope_baseline_km` | absolute; **not** rescaled — cell size no longer varies with `n` |
| World-scale | `orogen_decay_km`, `arc_*`, `rift_shoulder_km`, `continentality_scale_km`, `valley_smoothing_sphere_fraction`, boundary reach | fractions of the world; rescaled by `PlateParams.for_radius` and equivalents |
| Vertical / physical | `D`, lapse rate, densities, `width_coeff_m`, PET constants | untouched; they never referred to the sphere |

Putting a parameter in the wrong class is silent. `valley_smoothing_km2 = 60000` *looked* constant but was already tracking sphere area — 19 cells of drainage at n=128 and 725 at n=785 — and only looked fixed because the sphere was. It is now stored as the fraction it always was.

**Two findings that only appeared once the scale changed.** Both were latent bugs, not consequences:

- `boundary_distance` was called with a hardcoded 1200 km reach, which is 0.188 R on Earth but *larger than the whole world* at R = 64.9 km. Every cell would have read as boundary-adjacent and the uplift field would have flattened to a constant.
- `prune_islands` cleared `land_mask` but left the elevation above the datum, so a drowned islet was ocean by mask and land by height. It never fired before because at Earth radius the 1000 km² threshold was smaller than a single cell on every test fixture, so nothing was ever pruned.

**`U` and `D` became answerable to each other.** Steady-state hilltop curvature is exactly `U/D`. The old `U = 1200 m/Myr` with `D = 0.05 m²/yr` implies 0.024 1/m — 120 m of convexity *per 100 m cell*, i.e. spikes. That inconsistency was always in the parameters and was invisible only while hillslopes were sub-grid. Uplift came down 12× rather than raising `D`, because matching `U = 1200` would need `D = 0.6 m²/yr`, well outside the published 0.001–0.1 range for soil creep.

**Diffusion had to be sub-cycled, and this is the real cost of the rescale.** The explicit stability limit is `A_i / (2 D Σ(e_ij/L_ij))`. `Σ(e_ij/L_ij)` is dimensionless (6/√3 = 3.46 for a hexagon) and scale-free, so the limit tracks *cell area* and falls as the square of cell size: **239 Myr at 9.77 km, 25 kyr at 100 m**, against a fluvial timestep of 0.5 Myr. Dropping the global timestep to satisfy it would multiply the cost of the expensive part — priority-flood, `O(N log N)` with poor cache behaviour — by 20 to fix the cheap part. `diffusion_substeps` computes the count from the grid and sub-cycles only the diffusion. Stability is tested by the discrete maximum principle rather than a threshold: a stable scheme cannot carry a cell outside the range the field already occupied, and the raw call at a fluvial `dt` does exactly that.

**The measurement that says it worked, and its price.** At n=384 over 400 steps, binning slope against drainage area: with `D=0` slope falls monotonically from the first bin — no hillslope regime, which is what the whole 9.8 km world looked like. With `D` on, the curve develops a hump, slope rising to a peak at the channel head before falling as `A^-θ`. That rising limb is the convex hillslope, and it is the thing the rescale was for.

| D | channel head | θ (fluvial limb) | r² | sub-steps |
|---|---|---|---|---|
| 0.00 | none — monotonic | 0.457 | 0.991 | — |
| 0.01 | 0.021 km² | 0.402 | 0.927 | 10 |
| **0.02** | **0.034 km²** | **0.362** | 0.907 | 20 |
| 0.05 | 0.056 km² | 0.267 | 0.748 | 48 |

Every channel head there sits inside the real 0.01–1 km² band, so that criterion does not discriminate. `D = 0.02` is the chosen trade: about two cells of hillslope and 20 sub-steps.

**The θ column above understates the production world twice over**, and the correction matters more than the table. That fixture's basins are 4× smaller than production's, and its θ is fitted close to the channel head. On the n=785 world, fitted above the head, θ is **0.467 at 40 cells (r² 0.90) and 0.556 at 400 cells (r² 0.85)** — inside the band, not below it.

**This resolved two findings previously recorded here as unfixed.** The section on straight-line artifacts closed by conceding that some mountain flanks still combed with detail off, because the mesh was genuinely rough at cell scale, and called it "a Stage 2 question about drainage density at 9.8 km cells, not an export question." That diagnosis was right and the grid was the answer:

| measurement | 9.77 km cells | 100 m cells | real terrain |
|---|---|---|---|
| land cells that are local maxima | 4.40% | **0.502%** | 0.14–0.26% (SRTM) |
| median drainage area | 1.0 cells | **5.98 cells** | 3–4 px (SRTM) |

Neither needed a change to the drainage code. Both were the same symptom: a landscape with no hillslope domain, where half of all land had no upstream contribution at all.

**One threshold had to move with the grid, and it is an easy trap.** `metrics.report` takes a single `channel_threshold_km2` and uses it both to define the channel mask and to select cells for the slope–area regression — but that regression is only valid on the fluvial limb. At 9.77 km cells, 6 cells of drainage was 500 km², far above any channel head, so the two uses never conflicted. At 100 m cells, 6 cells **is** the channel head, so fitting from there mixes hillslope and fluvial ground and θ reads 0.392. `generate_world.py` now reports at 40 cells. Any θ quoted from this pipeline should carry its threshold.

### Stage 5: secondary erosion — sub-grid at 9.81 km, viable at 100 m

**This section is superseded by the grid rescale and kept because the reasoning still holds — only the verdict changed.** The world is now 100 m cells on a 64.9 km sphere (see "Cell size" below). At the old 9.81 km cells, three of the four processes could not act, and it was worth stating that with numbers rather than shipping operators that do nothing:

| process | feature scale | at 9.81 km cells | at 100 m cells |
|---|---|---|---|
| Thermal talus | needs slope > angle of repose | **impossible** — 33° across one cell is 6,371 m of drop, against ~8 km of total world relief | **viable** — 65 m of drop, which mountain terrain clears |
| Aeolian dunes | 0.05–2 km | sub-grid | resolved |
| Wave-cut notches | 1–50 m | sub-grid by 2–4 orders of magnitude | marginal |
| Cirques | 0.5–3 km | sub-grid | resolved |
| **Glacial troughs** | 1–10 km | marginal | resolved |
| **Fjords** | 20–200 km | resolved | resolved (world-spanning) |
| **Ice sheets** | 500–4,000 km | resolved | larger than the world |

None of the newly viable operators is implemented. What changed is that writing one is now worth doing, and `tests/test_glacial.py` pins the geometry so a future grid change cannot quietly take it back. Note the last two rows moved the other way: a 64.9 km sphere is smaller than an Earth ice sheet, so continental glaciation is no longer a thing this world can express.

So Stage 5 is glacial erosion, driven from the Stage 3 temperature field (which already carries the lapse rate, so latitude and altitude are handled together — a latitude-only ELA put ice on 1.2% of land and missed low-lying polar shields entirely).

**What makes glacial terrain glacial, at this resolution, is that ice can cut below its own outlet.** Water cannot: it needs a downhill path, so the fluvial solver forbids it and every river valley drains monotonically to the sea. Ice is driven by its *surface* slope and will gouge a closed basin then flow up out of it — which is what overdeepenings, fjord sills and glaciated lake districts are.

Getting that to happen took one non-obvious change. Simply removing the "don't cut below the receiver" clamp achieves **nothing**: with `E ∝ S`, erosion vanishes exactly as a cell approaches its receiver's elevation, so the bed asymptotes flat and never crosses (measured: minimum drop-to-receiver +0.0 m and zero overdeepenings at every erosion rate tried). The driving slope has to be **regional**, measured over a long flow path, which is also the physically right choice since ice responds to surface not bed slope.

**A negative result worth recording.** The textbook signature — glaciation flattens channel long profiles — did not survive measurement. Concavity in glaciated channels went 0.507 → 0.408 at n=256 (flattening, as predicted) but 0.559 → 0.631 at n=128 (steepening). **The sign flips with resolution**, so concavity is not a usable glacial diagnostic at continental cell sizes, and no test asserts it. What does hold is that erosion is cleanly confined to cold ground: 36.9 m of mean lowering under ice against 0.03 m off it, a ratio of 0.00084.

Glacier tongues correctly extend past the freezing line — mean temperature under ice is −11.3 °C but the warmest glaciated cell is +13.6 °C, because a glacier flows downhill past its own equilibrium line and keeps eroding until it melts out. An assertion that every glaciated cell is below freezing would be asserting a bug.

### Stage 6: lakes, endorheic basins, rivers

Depression filling inside the erosion loop is a numerical device — it exists so water has somewhere to go. This stage is where depressions become geography instead.

**The distinction that matters is exorheic versus endorheic,** and it needs a water balance rather than geometry, which is why this stage comes after climate. A basin whose catchment delivers more water than its lake surface can evaporate fills to its spill point and drains onward; one in an arid enough setting settles far below the rim where evaporation exactly balances inflow, and never overflows. Filling every depression to the brim — what the erosion loop does internally — turns every Caspian into an overflowing freshwater lake and deletes a whole class of landscape.

Verified on constructed basins, holding geometry and rainfall fixed and varying only temperature: at 1,200 mm and 2 °C the basin fills to spill and overflows; at 12 °C it settles 150 m below the rim and shrinks. Drier or warmer pushes it further down (700 mm / 8 °C settles 339 m below spill).

**Choosing that test climate needs care, and the trap is instructive.** Push the temperature too high and land runoff drops to *exactly zero* — PET exceeds rainfall, nothing reaches the basin at all, and the answer is "no lake" rather than "endorheic lake". The endorheic window needs positive runoff that lake-surface evaporation can still outpace, roughly `P < 2·PET`.

**That trap was not confined to the test fixture — it was the shape of the water balance itself.** `AET = min(P, PET)` puts a kink at `P = PET` and hands back exactly zero runoff everywhere drier than that, which on a generated world was **59% of all land**: 92% of the 15–30° belt, 84% of 30–45°, and 65% of the tropics. Not thin rivers across the subtropics — no rivers at all. Real catchments do not work that way, because rain arrives in events and some of it runs off before the atmosphere can take it back, however arid the annual means look. Turc–Pike, `AET = P / sqrt(1 + (P/PET)²)`, is the standard smooth alternative and has no free parameter; it approaches both physical limits correctly but reaches them asymptotically, so a desert yields a trickle rather than nothing. Zero-runoff land 59% → 0%, mean land runoff 530 → 600 mm. Note the direction: mean runoff *rises*, because this pipeline already produces more land runoff than Earth (600 mm against ~315). What it has is the wrong distribution, and that is Stage 3's problem, not this stage's.

**This stage exposed a bug two stages upstream.** The first run found four tiny coastal lakes and not one endorheic basin, because the world had essentially no closed depressions — the fluvial solver forbids them by construction, and nothing else was making any. The cause was in Stage 1: the boundary table specifies *"continental divergent → subsiding graben flanked by uplifted rift shoulders"*, `derive_uplift` computed exactly that, and then `np.clip(u, 0.0, 1.0)` threw the subsidence away. Tectonic subsidence is where the Dead Sea, Baikal and the East African rift lakes come from. Restoring it took a matching change in Stage 2, because the stream-power solver clamps every cell to stay at or above its receiver — right for incision, since water cannot cut a closed basin, but it would equally forbid a graben from dropping below its outlet. Uplift now goes through the solver and subsidence is applied outside it. Lakes went from 4 to 24, with the first endorheic basin appearing.

Closed basins remain **rare** in generated worlds — 0.03% of land against Earth's roughly 1% lake cover — because fluvial erosion efficiently removes depressions and continental rifts are uncommon. The endorheic machinery is therefore unit-tested on constructed basins rather than on emergent ones.

**And this stage exposed a second Stage 1 bug, of the same shape as the first: the lakes were being classified away before hydrology ever saw them.** Sea level is solved as a quantile and the mask was then `land = elevation > sea_level` — a pure threshold, with no notion of connectivity. Every closed depression below the datum was therefore handed to the ocean, however far inland it sat. Measured on an n=128 world: **41 bodies of water, one holding 98.95% of it and forty inland**, together 2.59% of land area, which is roughly Earth's lake cover. Stage 6 was finding two lakes because the forty basins where lakes belong had already been declared sea.

The fix is a flood fill: ocean is the largest connected body of water, and everything else below the datum is land. The basins are then closed depressions on the land surface, so the flow router fills them and this stage's existing machinery takes over — spill point, water balance, level and depth. **No lake code was needed.** A lake is marked by biome code `14`, not by elevation, because an inland basin floor sits below sea level exactly as the abyssal plain does.

Two consequences worth recording. **`target_land_fraction` now overshoots**, because basins move to the land side after the datum is fixed; chasing it would be a fixed point, since moving the datum changes which basins connect to the sea. The overshoot is bounded by the basin area — well under a point at the 29% target, though a 50%-land world runs 4 points over as the water fragments. And **the landscape got markedly more sensitive to perturbation**: those forty basins had been *pinned* at base level, fixed-elevation anchors punched into the continents. Removing them, on the same crust, took max flow path 1.84 → 2.56 km, mean top-1% drainage area 0.503 → 0.675 km², and the divide-reorganisation figure 0.26% → 17.85%. The old stability was an artifact of the anchors.

**What did not follow is a lake-covered world.** The forty basins are 2.59% of land at Stage 1 and the surviving lakes were 0.03%; the deep basins persisted as *dry* depressions below sea level.

**That was blamed on Stage 3's precipitation, and the attribution was wrong.** The claim recorded here was that the endorheic water balance was doing its job against rainfall too meagre to fill anything — land p25 about 15 mm against Earth's ~250, the same dry tail that pushes temperate desert to 20% of land. The dry tail is real. It is not what limited the lakes, and the test that settles it is to substitute the suspected cause and see whether the lakes move:

| | lakes | % of land |
|---|---|---|
| as shipped | 2 (1 endorheic) | 0.029% |
| Turc–Pike runoff — land with zero runoff **59% → 0%** | 2 | 0.031% |
| precipitation compressed to Earth's spread (log₁₀ sd 0.83 → 0.42, median 298 → 750 mm) | 2 | 0.031% |
| both | 2 | 0.031% |

Giving every cell on the planet positive runoff changes the lake count by nothing at all. **The binding constraint was that erosion had already destroyed the basins**, and the numbers were never ambiguous once anyone counted them: closed depressions are 19.6% of land on the Stage 1 crust, 0.016% after the coupled loop, and 0.084% after glacial erosion puts a few back. Two of those clear the minimum lake area, and those two were the two lakes.

**Two terms erase them and each is sufficient on its own,** which is exactly why this hid for so long — exempting either one alone looks like it barely helps, because whichever is left finishes the job. The stream-power update snaps any cell below its receiver straight up to the receiver in a single step (that clamp is right for incision, and catastrophic for a cell that is lake bed rather than channel), and hillslope diffusion drags the rim down into the hole. It is not slow: a 1,392 m graben is filled to +11 m within ten of six hundred steps, floor rising rather than rim breaching — the eight largest Stage 1 basins gain +237 to +1,402 m of floor over 100 steps while their spill points fall only 12–112 m. Measured after 200 steps at n=128:

| exemption | closed depressions | over minimum lake area |
|---|---|---|
| none (as shipped) | 0.002% of land | 0 |
| hillslope diffusion only | 0.854% | 26 |
| stream-power clamp only | 0.002% | 0 |
| **both** | **3.240%** | **30** |

Exempting both is what ships. Land relief rises with it — standard deviation 45 → 106 m on the same crust — because the basins were being planed away along with everything else.

**It also uncovered a runaway that the clamp had been hiding.** Erosion balances uplift, and nothing balances subsidence: deposition is off, so a graben subsiding at 16.9 m/Myr over a 300 Myr run reaches 5,063 m, and the first run after the exemption duly reported a lake **5,265 m deep on a world whose highest peak is 361 m**. The clamp had been cancelling the subsidence as fast as it was applied, so grabens stayed shallow for a reason that had nothing to do with tectonics. Rift subsidence is finite in nature — McKenzie stretching gives a basin a total, set by how far the crust thinned, after which it stops — so `max_subsidence_m` bounds the total rather than trimming the rate. At 500 m the deepest basin floor lands at −1,710 m, in the range real continental rifts occupy.

### Stage 5b: reflood — the coastline is not a constant of the pipeline

Stage 1 solves the land mask and Stages 2 and 5 then move the surface underneath it by hundreds of metres, with nothing re-deriving it. Glacial erosion is why that matters: ice is driven by its own surface slope and will gouge below its own outlet, which is precisely what carves a fjord — and the water then has to be let in.

Measured after a full n=128 run: **20.9% of land sat below the datum, and 99.9% of that was connected to the sea**, median depth 106 m at median latitude 63°. Glacially overdeepened coastal troughs, correctly cut and left dry. The fix reuses Stage 1's two rules verbatim on the new surface — below the datum is water, ocean is the largest connected body of it — so a trough joined to the sea floods and a hole that is not joined to the sea stays land and becomes a lake candidate. Land goes 29.6% → 23.3%.

The climate is then re-solved on the coastline that ships (**Stage 5c**). Everything downstream reads absolute millimetres and degrees against the new mask, and the old fields were solved for a continent that is now part sea — coast distance, the ocean moisture source and the sea-surface anomaly all move when a fifth of the land floods. It does not feed back into erosion; Stages 2 and 5 have run. Stage 5 keeps the *pre*-reflood temperature, which is correct: that is the coastline the ice grew on. It is also what makes the precipitation anchor land exactly on its target rather than near it, since the median is now taken over the mask it is normalising for.

**The reflood immediately reported a Stage 5 calibration error it had previously been concealing** — see the recalibration below. Once that was fixed, the four changes together take **lakes from 0.029% of land to 1.278%**, against Earth's roughly 1%, river cells from 4.96% to 12.6% of land, and endorheic basins from one to two.

### Stage 5 recalibrated: "inside the published band" was true and not enough

Glacial `K` was 2.0e-2, cutting 148 m of mean lowering under ice over the 300 kyr run. That is 0.49 mm/yr, comfortably inside the published 0.1–10 mm/yr band, and the old comment said so. The band spans two orders of magnitude, and **which end of it belongs here is a question about this world's relief, not about glaciers in general.** The pre-glacial relief budget is 432 m against several kilometres on Earth's glaciated terrain, so at 0.49 mm/yr the ice was removing 34% of the world's entire relief in 300 kyr, where Quaternary glaciation took 5–10% off Earth's shields.

This was invisible before Stage 5b, because the overdeepened ground stayed dry — it read as odd elevations rather than as a drowned continent.

It was also cheap to fix, because `K` turns out to be a pure incision-*depth* knob. Across a twentyfold change the glaciated fraction (21.8–22.1%), the overdeepening count (404–417) and the lakes (11–15, ~1.3% of land) barely move, so lowering it costs none of the glacial signature this stage exists to produce:

| `K` | lowering | rate | % of relief | drowned | Köppen E |
|---|---|---|---|---|---|
| 2.0e-2 | 148 m | 0.49 mm/yr | 34.3% | 21.4% | 4.7% |
| 1.0e-2 | 95 m | 0.32 mm/yr | 21.9% | 18.7% | 7.8% |
| 5.0e-3 | 57 m | 0.19 mm/yr | 13.1% | 16.0% | 10.8% |
| **3.0e-3** | **37 m** | **0.12 mm/yr** | **8.6%** | **14.2%** | **12.7%** |
| 2.0e-3 | 25 m | 0.08 mm/yr | 5.9% | 12.7% | 14.2% |
| 1.0e-3 | 13 m | 0.04 mm/yr | 3.0% | 10.0% | 16.8% |

3.0e-3 satisfies all three criteria at once — rate inside the published band, share of relief inside Earth's 5–10%, Köppen polar share 12.7% against Earth's 14%. 2.0e-3 scores better on the last two but falls below the published rate floor, which is not a trade worth making for a classification share.

**What it does not fix is that 14.2% of land still floods.** With the ice switched off entirely only 1.0% does, so the drowning genuinely is glacial — but it is amplified out of all proportion by hypsometry. Median land elevation on these worlds is about 10 m, so ground that is lowered at all tends to go under. That is the relief shortfall recorded elsewhere in this document turning up somewhere new, not a glacial parameter left mistuned.

**Net effect of the four changes, measured on the same n=128 world:**

| | before | after | Earth |
|---|---|---|---|
| lakes | 0.029% of land | **1.278%** | ~1% |
| endorheic lakes | 1 of 2 | 2 of 12 | |
| river cells | 4.96% of land | 12.55% | |
| land precipitation median | 298 mm | 600 mm | ~600 mm |
| log₁₀ sd of precipitation | 0.833 | 0.761 | ~0.45 |
| rain in the wettest 5% of land | 47.5% | 45.4% | ~15% |
| Köppen mean error vs Earth | 9.1% | **7.3%** | |
| Köppen B / E | 41.0% / 24.4% | 36.1% / 12.7% | 30% / 14% |
| land fraction | 29.6% | 25.4% | |

The remaining large Köppen gap is **D at 7.1% against Earth's 24%** — continental climates, which need the winters that a world this size and this flat does not produce. That is the relief and continentality shortfall again, not anything these four changes touch.

Meander migration (0.1–10 km wavelength) was sub-grid and deliberately absent at 9.8 km cells; at 100 m it is resolved and simply not implemented. River *width* is carried as a per-cell attribute (`w ∝ Q^0.5`, the standard hydraulic-geometry relation) because it is wanted for rendering. `width_coeff_m` is **not** rescaled with the radius: the relation is physical, and `Q` already carries the smaller basins — a 100 km² catchment delivers what a 100 km² catchment delivers, whatever planet it is on.

### Stage 8: amplification

Simulate coarse, add detail afterwards — which only means something if there is a finer target. The mesh is 100 m at n=785 and the default export raster is 6280×3140 (~65 m/px at the equator), so detail lives in that gap — now about 1.5 px per cell, against 2 px before the rescale. `max_amplitude_m` is denominated against the drop across one cell at `slope_reference` (100 m × 0.06 = 6 m), not against world relief, which is why it fell from 260 m to 3 m.

**Two separate problems, worth keeping apart.** Nearest-cell sampling of a hex mesh renders the hexagons — flat-shaded cells with visible seams, which reads as a rendering bug rather than as terrain. Inverse-distance weighting over the three nearest cells cuts measured high-frequency energy by 3× before any detail is considered. Only then does fractal detail go on top.

**Modulation is the whole point; uniform noise over a landscape is the most obvious tell of a procedural map.** Amplitude is driven by slope and drainage area, giving (measured, in metres) 107 on steep ground, 35 on plains, 1 on trunk valley floors, 16 under the sea. A valley floor is smooth because sediment is filling it, not because nothing is happening there.

Two calibration traps, both of which zeroed the detail almost everywhere on the first attempt:

- **The valley-smoothing scale must be a large-river drainage area, not a few cells.** At 3,000 km² it was roughly four cells of drainage at n=256, and it smoothed the entire channel network flat.
- **The slope term needs a floor.** Without one, amplitude reaches exactly zero on plains, on valley floors, *and across the whole sea bed* — base level is flat, so its slope is zero. Real plains have texture.

**Never feed amplification back onto the simulation mesh.** Detail perturbs elevations, which changes steepest-descent receivers, which rewires the drainage network the previous five stages exist to get right. Measured: slope–area concavity moves **0.512 → 0.603** and ~2,300 spurious channel cells appear. Amplification is an output-side operation and the metrics belong on the pre-amplification field. A test pins this so the pipeline does not get "simplified" by amplifying in place.

### Stage 9: export

**One artifact, after a correction.** The survey specified two — a native archive and an equirectangular GeoTIFF — and the second has since been removed. **Native:** field arrays plus `n` and the seed, at under 40 bytes per cell; the grid is regenerable from `n`, so storing geometry alongside would be paying 650 MB for nothing. A manifest records parameters, seed and measured metrics beside the data, because a world whose provenance is lost is not reproducible even when the generator is deterministic.

**Why the GeoTIFF went.** It was justified by "the metrics code runs unmodified on both generated and real terrain". That was true and it never paid out. The export did load, route and measure through the same path as the SRTM tiles with no code change — but the numbers were not comparable to the SRTM references, for a reason that is a resolution gap rather than a defect: the export was ~5 km per pixel against SRTM's 30 m, so no window was simultaneously wide enough to span three decades of drainage area and small enough to avoid serious equirectangular distortion. The conclusion was already **judge a generated world by the mesh metrics**, which is what the pipeline does. Nothing else consumed the rasters either: the renderer samples the mesh directly. Real DEMs still arrive as GeoTIFF and `terrain/dem.py` still reads them; that never required the export to be one.

Two hazards found while the round-trip existed are worth keeping, because anything that rasterises a globe meets them. Both gave a concavity of **−0.081** — slope rising with drainage area, which is not a landscape:

- **A window must avoid the poles.** A window tall enough to reach one is measuring a coordinate singularity; equirectangular cells collapse to zero width there and the D8 graph across the pole is meaningless.
- **The sea must be declared as base level.** A raster carrying bathymetry is otherwise treated as one continuous land surface, and rivers incise happily across the abyssal plain. Real SRTM tiles never need this because they are land-only — which is exactly why the generated case exposed it.

### Stage 7: biomes

Whittaker classification from the temperature and precipitation Stage 3 already produces, so the stage is nearly free — and it is worth doing early precisely because it turns the climate fields from numbers into something you can judge at a glance. Land shares against Earth's:

| | generated | Earth | | generated | Earth |
|---|---|---|---|---|---|
| temperate grassland | 23.4% | 9% | boreal forest | 8.8% | 12% |
| savanna | 17.8% | 15% | woodland/shrubland | 4.9% | 6% |
| subtropical desert | 11.8% | 14% | tropical rainforest | 4.3% | 6% |
| tundra | 11.6% | 8% | tropical seasonal forest | 3.0% | 6% |
| ice | 9.0% | 10% | temperate seasonal forest | 2.8% | 7% |

Close on desert, savanna and ice; grassland is over-represented and every forest type under-represented, which is the climate model's dry bias showing through rather than a classification error.

### Zonal banding: what fixed it, and what did not

The biome map came out in ruler-straight latitude bands. Measuring first — *zonality*, the share of a field's variance explained by latitude alone — located the cause precisely:

| field | zonality |
|---|---|
| temperature | **0.962** |
| log precipitation | 0.187 |
| elevation | 0.058 |

Precipitation was already varied; temperature was almost purely latitudinal, and biomes threshold on temperature. Four candidate fixes, measured against how far a desert belt's mean latitude wanders with longitude:

| change | result |
|---|---|
| Continental cooling on mean temperature | zonality 0.962 → 0.959; band dominance *worse*, 59% → 64% |
| Wind meander 0° → 28° | belt wander 4.62° → 3.72° |
| Lower arid thresholds (200 → 80 mm) | temperate desert 20% → 16% of land |
| **Arid biome classes outside the tropics** | **belt wander 4.41° → 9.87°** |

**The cause was in the classification, not the climate.** `SUBTROPICAL_DESERT` requires T ≥ 20 °C, so aridity could only be expressed where it was also hot — making the desert class a latitude band *by construction*, and collapsing every arid temperate region into grassland (23% of land against Earth's 9%). Earth's Gobi sits at 43°N and Patagonia at 45°S; neither is a warm desert, and neither could exist in the output. Adding temperate and cold desert classes lets aridity appear at any latitude.

**Two things kept despite not fixing the banding.** Continentality on mean annual temperature is real physics that was simply missing — Siberia averages −10 °C where the Norwegian coast at the same latitude averages +5 °C — and the wind meander does help a little.

**And an honest limit: the bands are still visible.** Deserts now span more latitudes but are not much patchier, and a wide uniform band reads much like a narrow one. Breaking zonality properly needs mechanisms this model does not have: **ocean-current heat transport** (the Gulf Stream is why Norway is not Labrador) and a **seasonal cycle** (monsoons, and the prerequisite for Köppen classification). Both are stage-scale additions rather than tuning.

### Ocean heat transport: built, correct, and not enough

The first of those two was built. Surface currents come from the wind — the three-cell circulation already alternates easterly and westerly, so a constant Ekman deflection closes it into a gyre in each basin without anything needing to know where the basins are — and the flow is made tangential at coasts, which is what gives a gyre its poleward western boundary. A temperature anomaly advects along it, and then along the *wind* to reach inland.

**It works, and it moves the headline number almost not at all: temperature zonality 0.9926 → 0.9898.** Both halves of that are worth recording.

| non-zonal contribution to land temperature | std |
|---|---|
| continental cooling | 1.07 °C |
| **maritime anomaly** | **0.66 °C** |
| latitude profile (residual) | 0.35 °C |
| lapse rate | 0.28 °C |
| *total* | *1.46 °C* |

The mechanism is the second-largest source of non-zonal temperature in the world. The aggregate does not move because the latitude term has a standard deviation of **13.7 °C** against 1.46 °C for everything else combined — no horizontal process can compete with that ratio.

**So the aggregate is the wrong measure, and the right one is directional.** The Norway–Labrador contrast is not that one coast sits beside warmer water; it is that the westerlies carry maritime air *into* Europe while at the same latitude the wind leaves Labrador for the sea. Measured: onshore-wind coasts carry **1.9×** the maritime anomaly of offshore-wind coasts. An earlier version decaying the anomaly as `exp(−distance)` from the coast scored **0.64×** — offshore coasts came out *more* maritime — because a geometric decay carries no direction and was really re-deriving continentality under another name. It scored better on zonality while getting the physics backwards, which is the whole argument for testing the mechanism rather than the statistic.

**What the temperature field is actually short of is relief, not distance.** The horizontal length scales are already Earth-equivalent by construction — `continentality_scale_km = 15.3` on a 64.927 km world *is* 1500 km on Earth, and scaling both distances and parameters up changes nothing, since only the ratio enters. Vertically it is a different story.

**This section previously said the shortfall was ×13 and called it a Stage 1/2 vertical-scale decision. The diagnosis was right and the prescription does not exist**, and both halves needed measuring rather than asserting. Two things were wrong with the original numbers. They were taken on a prototype **three times smaller in radius than the production world**, and they were taken over Stage-1 land *including cells below the datum* — which is not what Earth's ~800 m is measured over, and which inverted the trend badly enough to hide the whole effect: including sub-datum land, elevation standard deviation appears to **fall** with world size as R^−0.265, because a small world's bounded-at-500 m subsidence grabens are a large share of its relief.

Measured properly — current code, seed 1, `coupled.run(passes=3, steps_per_pass=200)`, area-weighted over land above the datum, **including the production world rather than extrapolating to it**:

| | n=64 (R=5.3 km) | n=128 (R=10.6) | n=256 (R=21.2) | **n=785 (R=64.9)** | local exponent, per doubling |
|---|---|---|---|---|---|
| land mean | 14.6 m | 20.3 m | 27.8 m | **47.4 m** | 0.475, 0.454, 0.476 |
| land elevation std | 23.9 m | 34.3 m | 48.2 m | **104.0 m** | 0.521, 0.491, 0.686 |
| peak | 243 m | 362 m | 685 m | **2,428 m** | 0.574, 0.918, 1.130 |
| land below the datum | 3.8% | 1.9% | 1.1% | 0.9% | |

So at production the lapse-rate signal is **0.68 °C against Earth's ~5** — a shortfall of **×7.7**, not ×13.

**The production run was worth the twenty minutes, because the extrapolation it replaced was wrong and wrong in an instructive way.** Fitted on the three small worlds alone, the exponents were R^0.465 / R^0.508 / R^0.746, predicting 47 m / 85 m / 1,580 m. The mean came in at **+1.4%** and the peak at **+54%**. Read the last column for why: **only the mean is a power law.** Its local exponent is flat at 0.47 across a twelvefold range in radius. The peak's steepens monotonically — 0.574, 0.918, 1.130 — so a straight line through log–log space was averaging a curve, and extrapolating that average under-predicted badly.

**That also retracts the check this section previously leaned on.** It read the three-point peak exponent of 0.746 against the theoretical 0.757 — steady-state stream power at `n=1` gives relief ∝ L_basin^(1−h·m), with Hack's h = 0.54 and m = 0.45 — and called the agreement evidence the fit was real. It was coincidence: a curve whose local exponent runs from 0.57 to 1.13 crosses 0.757 somewhere in the middle, and the sampled range happened to straddle it. A theoretical value can be matched by an average that describes nothing. What the production run *does* confirm is `stage2.py`'s claim that `K = 1.2e-6` targets ~2 km peaks at n=785 — measured 2,428 m, where the extrapolation said 1,580 m and would have made that comment look wrong.

The ordering of the three is the likely explanation and is not separately tested: **mean is a bulk statistic, peak is a pure extreme, and std sits between them.** Land cell count grows as R², so the maximum over the sample rises with world size on top of any physical scaling, and the deviation from a power law tracks how much of each statistic lives in the tail.

**And the ×7.7 is not available as a knob.** Three routes, all measured or blocked by something already recorded here:

| route | verdict |
|---|---|
| raise `U` | **blocked.** Hilltop curvature is `U/D` and that budget is spent — measured slopes run p99 62.5°, p99.9 76.1°, already past the angle of repose. See `plates.py`. |
| lower `K` alone | **blocked.** Channel relief rises without touching hilltop curvature, but hillslope length goes as `(D/K)^(1/(1+2m))` — ×7.7 lengthens hillslopes 3.2×, and at fixed `U/D` a longer hillslope is a *steeper* one. It also collapses the drainage density the 100 m rescale exists to produce. |
| raise `U` and `D` together | **consistent, and worse than what it replaces.** Holding `U/D` preserves hillslope form and puts the relief in the channel network, which is correct physics. Price: `D` 0.02 → 0.154, outside the published 0.001–0.1 soil-creep band; diffusion sub-steps 20 → ~154, and sub-stepping is already "the real cost of the rescale"; peaks at **18.7 km on a 64.9 km sphere**, 29% of the planet's radius. |

**Nor can world size close it, and this is the number to quote.** Extrapolating the top-end local exponent of 0.686, reaching Earth's 800 m standard deviation needs R ≈ 1,270 km. At a fixed 100 m cell that is n ≈ 15,300, or **2.35 billion cells — 382× the production budget.** The relief shortfall and the 100 m cell size are the same decision seen from two directions, and the section on why the sphere shrank already stated the trade: this is a large island rendered properly, not a planet.

**The reframing is the part worth keeping.** Relative to its own size this world is not flat — it is **already 12.7 times steeper than Earth**. Earth's land elevation standard deviation is 1.26e-4 of its radius; this world's is 1.60e-3. Giving it Earth's *absolute* 800 m would make it 98× steeper than Earth in relative terms: every continent a wall.

The lapse rate is the only piece of physics in the model denominated in **absolute metres**, which is why it alone reads the shrink as a defect. That exposes a gap in this document's own rescale bookkeeping. The three-class table above sorts every *parameter* into cell-scale, world-scale and vertical-physical, and puts the lapse rate in the third class as "untouched; never referred to the sphere". True of the parameter and false of the term — **elevation is an emergent field, not a parameter, and it shrank with the sphere as nothing in that table predicts.** The mean went as R^0.47 and the peak steepened from R^0.57 to R^1.13; no single class covers a quantity that does that, and this is the one that matters.

### Seasons, monsoons, and Köppen

The second of the two named mechanisms. Stage 3 now solves **four points through the year** and reports their annual mean, so `coupled.run`, hydrology, biomes and the export all see the same fields they always did.

**Four rather than two, for a reason that is not "more detail".** Four evenly spaced samples determine a two-harmonic Fourier series *exactly* — mean, annual and semiannual — so the twelve monthly values Köppen needs are an interpolation with no free parameters. Verified two ways: the reconstruction returns the samples to 1e-9, and the four-sample mean reproduces the tuned annual temperature profile to 0.21 °C on a 52 °C range.

**In a memoryless model the sample dates matter more than the lag.** Declination is the only thing carrying the date into a steady-state solve, so two dates with the same sun give the same world — sampling the solstices and equinoxes yields *three* distinct seasons and a duplicate. The samples are offset from them for that reason. (An earlier version of this plan credited the thermal lag with separating the equinoxes; it does not. The lag is kept because it is true — seasons trail the sun by four to eight weeks — not because it rescues the fourth sample.)

**Insolation uses the day-length term, not `cos(lat − declination)`.** The shortcut has no polar night: it gives the winter pole an ordinary winter and makes the summer pole merely warm, and Köppen's D and E classes turn on exactly that. The closed form gives the summer pole the highest daily insolation on the planet and the winter pole exactly zero.

Two corrections were needed along the way, both found by measurement:

- **The seasonal term is applied additively, not as a factor.** Scaling the annual profile by an insolation ratio leaves the poles with no seasons at all — `cos(lat)^1.5` is zero there, so no factor can move it — which deletes the one distinction the day-length term was added for. It is also left *unclipped*, so the seasons average back to the tuned profile exactly; clipping truncated the winter pole and pulled the polar annual mean about 5 °C above `t_pole_c`.
- **Land and sea had to be given different heat capacities.** Rock follows the sun, water stores it, and that contrast *is* the monsoon. Without it the June land–sea contrast measured 0.58 °C with land running **colder** than the sea, the thermal low had nothing to form on, and the monsoon wind changed nothing at all — 32.5% of tropical coasts reversed with the term on or off.

**A monsoon is a wind that turns round, and that is what is tested.** Band migration alone gives a wet season; the thermal low gives the reversal. It costs rain shadow, and the trade was measured rather than guessed:

| `monsoon_c` | 0.00 | 0.10 | **0.20** | 0.30 | 0.60 |
|---|---|---|---|---|---|
| windward/leeward | 2.15× | 2.30× | **2.32×** | 1.91× | 1.41× |
| tropical coast reversing | 32.5% | 34.3% | **38.7%** | 43.2% | 58.0% |

0.20 ships: the best shadow of the five — better than switching the monsoon off, because a little seasonal steering makes each season's wind *more* consistent — while still moving the reversal.

**One bug in this section is worth recording because it looked like a modelling failure and was not.** The rain shadow appeared to collapse to 1.08× under seasons. The cause was `compute_seasons` returning the *first season's* wind beside a *year's* rainfall, so anything asking which side of a ridge was upwind got a wind that blew for three months. Averaging every aggregate over the seasons took it to 1.48× — better than a single equinox solve. A latitude taper on the band migration was added while chasing this and did not fix it; it is kept on its own merits, since Earth's westerlies at 45° blow from the west in every month of the year.

**Köppen lands alongside Whittaker, not instead of it.** Every biome share in this document is a Whittaker share and switching would silently invalidate the lot. The classes come out recognisably Earth-like with this pipeline's known biases showing plainly — arid classes at 45% of land against Earth's 31%, ice cap at 12% against 6%, and the continental classes short (Dfc 3.4% against 11.1%) because continentality is weak on a world whose median land cell is 1.2 km from the sea.

**A resolution bug fell out of this.** Every transport constant here was a fraction lost *per advection step*, which is an e-folding length denominated in **cells, not kilometres**. The same `background_rainout = 0.012` reached 0.13 R at n=785 and **1.57 R at n=64** — moisture crossed a small test world entirely before decaying, so the fixtures were running different physics from production. They are now e-folding distances in world km, rescaled by radius like every other horizontal length, giving 0.1283 R at every resolution; the values chosen reproduce the old per-step numbers exactly at n=785, where they were tuned.

**A subtropical desert belt appears in the biomes even though it is absent from the precipitation profile.** Stage 3 declines monotonically from the equator with no dip at 30°, yet the map has clear desert bands near ±20°. Whittaker's threshold is on temperature *and* rainfall jointly (T > 20 °C with P < 250 mm), so warm-and-moderately-dry ground classifies as desert without needing a rainfall minimum. Worth knowing before reading too much into it: the belt is real on the map but arrives for a different reason than it does on Earth.

Two things the raw diagram needs: **permanent ice**, which it has no cell for, placed by a latitude-dependent snowline (~4900 m at the equator falling to sea level at the poles) — this is what makes mountains read as mountains; and a guard against `cos(lat)**1.6` returning NaN at the poles from a slightly negative cosine, which would compare false against every elevation and silently leave the ice caps unmarked.

Note what the throwaway Stage 1 does **not** produce: meaningful continental relief. Its highest land sits ~560 m above sea level and mean continental freeboard is ~220 m, against a real ~840 m. That is the expected consequence of skipping tectonics — real high topography comes from crustal thickening under orogens (sub-step 7), not from noise on an undeformed crust. Mountains are Stage 2's uplift term's job, so this is not worth fixing in the throwaway.

### Straight-line artifacts: three causes, none of them the one that looked obvious

Rendered worlds showed ruled parallel streaks fanning out from every mountain range, plus a pervasive scaly texture over flat ground. The instinct is to blame the drainage network — D6 steepest descent can only pick one of six directions, so rivers *are* quantised. Measurement said otherwise, and the order in which the suspects fell is the useful part.

**What was measured and cleared first.** Depression filling raised 0.0% of land cells, so the `+eps` fill ramp was not it. The winning downhill direction beat the runner-up by a median 40%, so it was not tie-breaking either. Downstream straightness ran 73% against SRTM's 58–65% under the same routing — elevated, but not enough to explain the picture. Lateral coherence, which separates a comb from a tree, came out at 4.22× chance against SRTM's 4.54× and 4.95×: *lower* than real terrain. And seeding the initial surface with cell-scale roughness, the standard LEM remedy, moved straightness only from 45.5% to 43.9%. Four plausible mechanisms, four negatives.

**Cause 1 — graph distance is not distance.** `boundary_distance` accumulated `arc_dist` along the cell graph. On a hex lattice a path at 30° to a lattice row has to zigzag, costing 2/√3 = 1.155× the direct route, so the level sets are hexagons, not circles. Measured against true great-circle distance at n=256: 8% long on median, 21.8% at worst. Uplift is an *exponential* in that distance, so the error became lattice-aligned ridges — and rendering the uplift field directly showed the full artifact **before a single erosion step ran**. Replaced with a KD-tree query over the boundary cells: exact, and cheaper than 400 relaxation sweeps because there are ~15k boundary cells rather than 6M cells.

**Cause 2 — stress propagated from one cell.** `derive_uplift` gives every interior cell the stress of its single nearest boundary cell, so each boundary cell owns a wedge of territory — a median of 7 cells at n=256. Adjacent boundary cells carry unrelated stress, so the wedges render as hard stripes fanning out from the belt. Averaging stress along strike first is both the fix and the physically honest move: stress on a real plate boundary varies smoothly along its length, and the cell-to-cell scatter is discretisation noise from classifying each cell's velocity jump independently.

**Cause 3 — the interpolator invented texture.** Sampling the mesh to the export raster used inverse-distance weighting, and Shepard's method has *zero gradient at every data point*: the resampled field acquires a dimple at each cell centre. That is invisible in the elevations — 0.13 m rms — and glaring once hillshaded, because hillshading is a derivative. Resampling a perfectly smooth analytic field (a low-order spherical harmonic, no ridges and no noise) and measuring the mean |second difference| it manufactures:

| interpolation | manufactured curvature |
|---|---|
| IDW, 3 neighbours (was) | 0.345 m |
| IDW, 12 neighbours | 0.274 m |
| **barycentric over mesh triangles** | **0.0007 m** |

A wider stencil does not help, because the flaw is the kernel rather than the support. Barycentric is exact for a linear field, adds no blur, and needs no extra data — the grid already stores azimuthally ordered neighbours, so the triangle fan around each cell is free.

**Shipped as opt-in, with `nearest` the default.** The measured comparison above is a reason to prefer barycentric *over IDW* if you are going to blend at all; it is not a reason to blend by default. Nearest-cell sampling gives one cell per pixel, every value traceable to a cell, and nothing manufactured on the way out — which is the right default for a data export whose whole point is that the metrics run on it unmodified. The blends are `--resample barycentric` and `--resample c1`.

**But barycentric is only C0, and hillshading finds that too.** Linear interpolation has a constant gradient within each triangle, so the gradient jumps at every edge; the render shows a herringbone on steep ground, where the jump is large, and nothing on plains, where it is not. What identified it: rendering the same ground at 4× the raster width made the pattern *bigger* rather than finer — it was pinned to the mesh, not the pixel. That also rules out the obvious remedy, since exporting wider reveals faceting rather than hiding it, and the extra detail octaves a wider raster unlocks are far too low-amplitude to mask it.

The fix is Shepard's method applied to per-cell **linear polynomials** rather than to bare values. Plain IDW dimples precisely because it blends constants; blending `z_i + g_i·(p − x_i)` with a least-squares gradient fitted in each cell's tangent plane makes the interpolant reproduce `g_i` at cell i, which gives C1 everywhere, exact linear precision, and no blur. Facet tail (p99.5 / median of the second difference) on a mountain window: **barycentric 8808, C1 20.1**. Against the alternative of simply smoothing:

| scheme | facets | peak elevation (mesh peak 10,365 m) |
|---|---|---|
| barycentric | reference | 10,252 m |
| smooth kernel, σ = 0.85 spacings | 0.53× curvature | 9,647 m (−528 m) |
| **C1 Shepard–Franke** | **438× lower facet tail** | **10,365 m** |

Two details are load-bearing. The weight must be compactly supported — `((R − d)/(R·d))²` with R the distance to the k-th neighbour — because plain `1/d²` is discontinuous wherever the k-nearest *set* changes, which puts creases straight back in. And the blend must be clamped to the range of the contributing cells: gradient extrapolation goes Gibbs-like across the continental shelf break, where the mesh field is very nearly a step, and undershot the abyssal plain by 471 m without the limiter. The clamp costs nothing in smooth ground, where the blend is already inside the range.

**What this does not fix, and the measurement that says so.** After all four fixes some mountain flanks still render as a regular comb, and it is not the export: it survives with detail off and C1 on, so it is in the mesh field itself. The generated surface is genuinely rough at the cell scale — **4.40% of land cells are local maxima among their own neighbours, against 0.14–0.26% for SRTM pixels**, a 17–31× excess — and half of all land cells have no upstream contribution at all (median drainage area 1.0 cells, against 3–4 pixels on SRTM). No interpolation can smooth away topography that is really there. That is a Stage 2 question about drainage density at 9.8 km cells, not an export question.

**Since resolved by the grid rescale, and the diagnosis above is why.** Both numbers were symptoms of a landscape with no hillslope domain — at 82.7 km² per cell, every land cell already carried more drainage than a real channel head needs, so there was nothing for hillslope diffusion to act on and nothing to smooth the cell-scale roughness. At 100 m cells the local-maxima fraction falls to **0.502%** (from 4.40%, against SRTM's 0.14–0.26%) and median drainage rises to **5.98 cells** (from 1.0, against SRTM's 3–4 px). Neither needed a change to the drainage code. See "Cell size" above.

**A fourth, smaller one: the detail noise was mostly aliasing.** Stage 8 ran 6 octaves from a base frequency of 180 regardless of raster width. At width 8192 the finest octave has 1.1 km features against a 4.9 km pixel — four of the six octaves sat below Nyquist, and energy above Nyquist folds back as long straight moiré fringes, which is the exact artifact detail exists to avoid. The octave count is now derived from the width. The honest consequence: at the default width (8×n) a pixel is only ~1.5× finer than a cell, so **one octave is all that fits**, and getting more detail means exporting a wider raster rather than turning a knob.

**The lesson worth keeping.** Every one of the three real causes sat *upstream* of the thing that looked guilty, and the drainage statistics were normal throughout — the network was never the problem. Rendering intermediate fields directly (uplift, stress, boundary distance) found in one image what four quantitative tests had failed to find, because the artifact was structural rather than statistical.

---

## Validation: how to actually *test* the algorithms

Since comparing approaches is the point of the program, build the metrics module early. Each of these is a scalar or curve you can compute on generated terrain and on real DEM tiles of a comparable region, then compare:

Most of these are graph- or area-based and carry over to the sphere unchanged. Two do not, and are flagged below.

- **Slope histogram** — real terrain has a characteristic distribution; pure fBm is visibly too uniform.
- **Hypsometric curve** — elevation vs. cumulative area. Distinguishes young/steep from mature/eroded landscapes. Weight by real cell area.
- **Power spectral density** — ⚠️ *does not transfer.* On a sphere the analogue is the spherical-harmonic power spectrum `C_ℓ`, which on a hex grid means resampling to HEALPix and running an SHT. This is the one place HEALPix would have been easier. It is also the weakest metric on this list, and slope–area concavity already catches the failure mode it targets — dropping it is defensible.
- **Hack's law** — main channel length vs. drainage area: `L ∝ A^h`, `h ≈ 0.57`. Measure length along great-circle steps and area as summed cell areas. ⚠️ *Weaker than expected as a discriminator* — see the Stage 1 baseline below; noise-only terrain scores 0.469 against a real 0.543, only 14% off.
- **Horton's laws** — bifurcation, length, and area ratios between Strahler stream orders. Real networks cluster tightly around known values.
- **Slope–area relation** — in channels, `S ∝ A^−θ` with concavity `θ ≈ 0.4–0.6`. The cleanest single diagnostic that an erosion model is behaving correctly. Slope is `Δz / (R·Δσ)` with `Δσ` the central angle — never a planar `√(Δx²+Δy²)`.
- **Drainage density** — total channel length per unit area; should vary with climate.
- **Coastline fractal dimension** — ⚠️ *box-counting does not transfer;* there are no boxes on a sphere. Replace with a divider/ruler walk along the coastline using great-circle steps at several step lengths, and regress. Same output, valid on a sphere.

Comparison against real DEMs still works: reproject SRTM/Copernicus tiles onto a patch of the same grid and compare there, rather than comparing in tile space.

A terrain that matches on slope histogram but fails the slope–area relation is exactly the failure mode of noise-only generation — it looks right in a screenshot and wrong the moment you follow a valley.

### Measured reference values

The metrics module is implemented and calibrated against two contrasting SRTM 1-arcsec tiles, each a 2048² window (~63 km across, 4.19M cells). These are the numbers generated terrain gets compared against.

| | **east** — S. Appalachians | **west** — Olympics / N. Cascades | Published |
|---|---|---|---|
| Character | mature, humid, unglaciated fluvial | young, active margin, glaciated | |
| Slope–area concavity θ | **0.516** (R² 0.89) | **0.404** (R² 0.88) | 0.4–0.6 |
| Hack exponent h | **0.543** (R² 0.88) | **0.434** (R² 0.54) | 0.5–0.62 |
| Horton Rb / Rl / Ra | 5.51 / 1.75 / 4.07 | 4.67 / 2.31 / 4.06 | 3–5 / 1.5–3.5 / 3–6 |
| Drainage density | 0.744 km/km² | 0.480 km/km² | threshold-dependent |
| Mean slope | 12.8° | 13.4° | |

**Use the east tile as the strict reference.** It is unglaciated and lands inside the published band on every fluvial metric. The west tile falls low on θ and h, and its Hack fit is visibly worse (R² 0.54 against 0.88) — that is a real geomorphic signal rather than a measurement problem. Glacial erosion violates stream-power scaling: U-shaped valleys have flat floors and steep walls, which flattens the slope–area relation, and glacial troughs are shorter and blunter relative to basin area than fluvial networks. Its much lower drainage density says the same thing.

**Three implementation traps, all of which produced badly wrong numbers before being fixed:**

1. **Measure channel gradient over a flow-path baseline, not cell to cell.** SRTM stores int16 metres, so where a river drops less than 1 m across a 30 m cell the measured gradient is exactly zero — and after depression filling, a meaningless `+eps` residual instead. Those are precisely the highest-drainage-area cells, so they dominate the high-A end of the regression. Left in, they put θ at **1.87** instead of 0.52. Walking downstream until ≥0.5 km has accumulated fixes it, and degrades gracefully to cell-to-cell on a coarse mesh where one cell already exceeds the baseline.
2. **Exclude cells lifted by the depression fill** from slope–area. They sit on an artificial ramp and cluster in valley bottoms.
3. **Define "clipped basin" by the catchment, not the outlet.** Testing whether a basin's *outlet* lies on the window border excludes everything, because border cells are the sinks — in a 63 km window every drop of water leaves through an edge. Test whether anything *upstream* is edge-contaminated instead; interior sub-basins are complete by definition, since a catchment lies entirely upstream of its outlet.

**Routing-scheme sensitivity, measured.** Re-running the east tile with D4 instead of D8 moves θ by 0.001 and h by 0.000, but Rb by 16% (5.51 → 6.40). So θ and h can be compared directly between D6-routed generated terrain and D8-routed real DEMs; **Horton ratios cannot** — they carry a routing-scheme bias and are only meaningful compared like-for-like.

**Drainage density is not scale-free.** It is channel length per unit area, so a 9.8 km global mesh and a 30 m DEM window differ by orders of magnitude by construction. Track it across runs at fixed resolution; do not compare it across resolutions.

### Stage 1 baseline: what noise-only terrain actually scores

The throwaway Stage 1 (3D warped fBm thickness → isostasy → solved sea level, no erosion) run at n=785 against the same metric suite:

| | Stage 1, noise only | SRTM east | |
|---|---|---|---|
| Slope–area concavity θ | **0.106** | 0.516 | **4.9× off** |
| Hack exponent h | **0.469** | 0.543 | 14% off |
| Horton Rb | 5.98 | 5.51 | — |
| Hypsometric integral | 0.368 | 0.217 | — |

**This is the number Stage 2 has to beat, and it sharpens one of this document's own claims.** The survey asserted above that pure noise "fails Hack's law badly." Measured, it does not: h lands within 14% of real terrain, close enough that Hack's law alone would not tell you erosion was missing. What collapses is **slope–area concavity — 0.106 against 0.516**. Noise-only channels barely steepen upstream at all, because nothing has carved them; there is no mechanism in an fBm field that makes gradient scale with contributing area.

So the ranking for the testbed is: **slope–area concavity is the primary discriminator, Hack's exponent is a secondary confirmation, and Horton ratios are near-useless for this purpose** — noise-only Rb (5.98) is no further from the published band than real Appalachian terrain (5.51).

### Stage 2 result: erosion closes the gap

400 timesteps × 0.5 Myr = 200 Myr at n=785, implicit stream power (`K=2.5e-6, m=0.45, n=1`) plus hillslope diffusion and belt-like uplift. 435 s, converged to mean |dz| = 0.39 m/Myr.

| | Stage 1 (noise) | **Stage 2 (eroded)** | SRTM east |
|---|---|---|---|
| Slope–area θ, per-basin | 0.124 | **0.464** | 0.516 |
| Slope–area θ, whole planet | 0.094 | **0.498** | 0.516 |
| Hack exponent h | 0.464 | **0.543** | 0.543 |
| Hypsometric integral | 0.366 | 0.196 | 0.217 |
| Horton Rb | 5.84 | 5.42 | 5.51 |
| Peak elevation | 623 m | 6,273 m | — |

θ moved from 0.094 to 0.498 and Hack's exponent landed on 0.543, matching the SRTM reference exactly. **The document's central claim — noise gives you shape, erosion gives you realism — is now a measured result rather than an assertion.**

**Solver validation.** With *uniform* uplift the recovered θ equals `m/n` to ±0.001 (m=0.45 → 0.451, m=0.60 → 0.600, R² = 1.000). That is the strongest available end-to-end check: it only comes out right if the implicit solver, the flow routing, the spherical cell areas and the metrics all agree.

### Four things measured while getting there

**1. Uplift contrast controls whether θ means anything.** Steady-state slope goes as `U^(1/n)`, so where uplift varies spatially the uplift pattern competes with the `A^(-m/n)` signal in the slope–area regression. Holding everything else fixed at n=128:

| craton-to-orogen contrast | 1× | 2× | 4× | 10× | 50× |
|---|---|---|---|---|---|
| measured θ (true m/n = 0.45) | 0.453 | 0.452 | **0.512** | 0.690 | 0.893 |

A 50× contrast varies slope 50× on its own, while `A^-0.45` spans only ~22× across three decades of drainage area — so the uplift pattern wins and θ stops measuring `m/n`. Earth's real contrast is 50× or more, and **real Earth measured this way would be just as inflated**: published θ values come from tectonically coherent regions, not whole continents. Two consequences: measure θ per-basin, not planet-wide; and when comparing against published values, keep the generated world's uplift contrast moderate (~4×) or compare only against other generated worlds.

**2. `K` is resolution-dependent; `θ` is not.** Steady-state slope is `(U/K)^(1/n)`, so `K` sets relief while leaving the exponent untouched. But finer grids resolve smaller headwater drainage areas, where steady-state slope is steepest, so the same `K` yields far more total relief: `K=1e-6` gave 5.8 km peaks at n=128 and **17.6 km at n=785**. Retune `K` whenever the resolution changes — this is the concrete form of the doc's earlier warning that erosion coefficients don't transfer between resolutions.

**3. Hillslope diffusion did almost no direct work at 9.8 km cells, and this is why the grid was rescaled.** At ~10 km cells the convex-hilltop scale is far below the grid, and one step at `D=0.05` moved at most 0.014 m against 4 km of relief. Yet because it biased every step in the same direction it eventually tipped receiver choices at divides, and a flipped divide re-routes a whole sub-basin: over 100 steps the median difference from a `D=0` run was 0.014 m and the maximum **720 m**. A small persistent forcing is not a small effect.

At 100 m cells the premise is gone: diffusion is a first-class term that moves the typical cell, and the amplification gap collapsed from ~50,000× to 5.5× because diffusion now acts everywhere rather than only where it happens to tip a divide. See "Cell size" above.

**4. A 1 cm perturbation permanently reorganises part of the world.** ⚠️ *This corrects an earlier claim here that transients decay to bit-identical output.* Measured on a reproducible world: after a one-off 1 cm nudge, 99.2% of land ends up bit-identical, but **0.8% of cells differ by up to 433 m, and that difference is fixed from step 100 onward** rather than decaying. Where the implicit solve contracts a perturbation smoothly it dies; where it tips a receiver choice at a drainage divide, the sub-basin re-routes and never returns. Reproducibility here means identical inputs give identical outputs — not that near-identical inputs give near-identical worlds. The parallel-reduction determinism requirement stands.

The earlier wrong conclusion is instructive: it was measured before the sub-seed bug below was found, so every process was generating a *different world*, and the one that happened to be sampled was smoothly contracting.

**5. Sub-seeds must not be derived with Python's `hash()`.** `hash()` is randomised per interpreter for `str` and for any tuple containing one, so `hash(("crust", seed))` produced **a different planet on every restart** — same seed, same code. A same-process reproducibility test cannot see this, because within one run the hash is fixed; the tell was a Stage 2 test that passed alone, failed in a suite, and reported a different number each time. Use a stable digest (`zlib.crc32` over a byte string). Derive by *name* rather than sequentially, so that changing one stage's parameters does not reshuffle every other stage.

### Stage 1 proper: plates, Euler poles, derived uplift

Implemented per the spec, with crust type still generated from noise independently of the plate layout. At n=785, 400 × 0.5 Myr:

| | throwaway Stage 1 | **tectonic Stage 1** | SRTM east |
|---|---|---|---|
| Slope–area θ, per-basin | 0.464 | **0.452** (IQR 0.450–0.454) | 0.516 |
| Hack exponent h | 0.543 | **0.529** | 0.543 |
| Horton Rb | 5.42 | 5.44 | 5.51 |
| Peak elevation | 6,273 m | 8,369 m | — |
| Pre-erosion relief | 623 m | 3,835 m | — |

The pre-erosion figure is the point of adding tectonics: crustal roots under orogens (thickness reaching 51 km) give real isostatic relief before a single erosion step, where the throwaway had only ~600 m of noise.

**Four things this stage got wrong before it got them right**, all now pinned by tests:

- **Rivers must grade to sea level, not to the sea floor.** Holding ocean cells at their Stage 1 bathymetric depth let coastal rivers incise toward the abyssal plain 4 km down. Continents drowned: land minimum −1,371 m, with only the orogen belts left above water while every metric still read perfectly. The metrics cannot catch this — it is a base-level error, not a network error.
- **Uplift terms must be clamped before scaling.** The collision and arc contributions overlap, so summing them unbounded pushed peak uplift to 4,943 m/Myr against a 3,000 cap, and peaks to 19.3 km.
- **Random Euler poles produce too much strike-slip.** Classifying transform whenever tangential motion exceeds normal gave 44% transform against Earth's ~20%: a real plate system's motion is organised by the ridges and trenches that create the plates, and random poles are not. Requiring tangential to exceed 2.5× normal restores a plausible 39/37/25 convergent/divergent/transform mix.
- **Warp the plate boundaries, not just the crust field.** Raw spherical Voronoi boundaries are exact great-circle arcs, and since orogens track boundaries, mountain ranges rendered as dead-straight lines. Domain-warping the Voronoi *query position* bends them into arcs without changing the partition's topology.

### Stage 3: climate

Temperature, three-cell wind, and precipitation as a steady-state advection-reaction fixed point rather than time-stepped moisture packets. 11.8 s at n=785.

**What works.** Temperature lands at 26 °C at the equator and −23 °C at the poles with the 6.5 °C/km lapse rate applied exactly, and a land mean of 9.4 °C against Earth's ~8.5. Wind bands come out easterly / westerly / easterly with the circulation going slack at 0°, 30°, 60° and 90° — the doldrums, horse latitudes and polar front. Precipitation falls monotonically from wet tropics to dry poles, and there is a real coast-to-interior gradient.

**Three fixes without which it does not work at all:**

- **Cap moisture at saturation.** Incoming advection weights sum to more than one wherever wind converges, so the fixed point is not a contraction and moisture grows without bound: **1.3 × 10⁷ mm/yr** before the cap. The Clausius–Clapeyron ceiling bounds the operator *and* supplies the mechanism that makes convergence zones rain heavily rather than diverge.
- **Evaporation fills the saturation deficit, it does not pump into saturated air.** Adding it unconditionally and raining off the overflow made the ocean rain out its own evaporation on the spot — over 10,000 mm/yr across most of the sea — leaving only a trickle to reach land.
- **Blend a little isotropic mixing into the advection.** Zonal wind is exactly zero at 0°, 30°, 60° and 90°, so on pure advection those cells never export moisture. They became traps that filled and dumped, producing 129,000 mm/yr spikes along four latitude lines.

**Measuring a rain shadow needs a control, and without one the number is nearly all continentality.** Walking upwind and downwind from ridge cells gives a raw windward/leeward ratio of ~2.4×, which looks like a strong rain shadow. Flattening the terrain and re-measuring *at the same points* gives ~2.5×. Almost the entire raw signal is the coast-to-interior gradient — walking upwind from a mountain heads toward the sea, downwind heads inland. The isolated orographic contribution is the ratio of the two, and at the tuned settings it is **1.34×**. An earlier iteration of this work reported 1.9–2.9× "rain shadows" that were nothing of the kind.

Also note what does *not* work as a test: comparing the p90/p10 spread of precipitation with and without relief. Flat terrain gives a **wider** spread, because without orographic rain the interiors go bone dry and p10 collapses.

**The circulation must meander.** Perfectly zonal bands put the four zero-wind latitudes on exact circles, and they showed up in the output as ruler-straight stripes of anomalous rainfall — plainly visible in the biome map as a dark line along the equator. Perturbing the band latitudes with low-frequency noise (standing in for Rossby waves) drops the equatorial stripe from 2.05× its neighbours to 1.61×, and a wet ITCZ at that ratio is physically correct anyway. It costs some of the orographic signal, because a less consistent wind direction makes the upwind/downwind comparison noisier.

**The output distribution is twice as wide as Earth's, the obvious diagnosis is wrong, and no knob fixes it.** Land rainfall comes out with a log₁₀ standard deviation of 0.83 against Earth's ~0.45, and the wettest 5% of land takes **47.5% of all land rain** where Earth is nearer 15%.

The natural suspect is the convergence zones — the saturation cap above exists precisely because convergent flow concentrates moisture — and it is not them. The wettest 5% of land has a median incoming advection weight of **1.001** against **0.999** for the other 95%: identical, and only 11.9% of those cells are even saturated. A `convective_efficiency` factor scaling the overflow was written and measured before being reverted; a tenfold change moved log₁₀ sd from 0.837 to 0.823.

What the tail is instead is the two *intended* mechanisms multiplying. The wettest cells hold 5.3× the moisture of the rest **and** sit where the ascent factor is ~1 rather than at `subsidence_floor`. Wet-and-ascending against dry-and-subsiding is a product, not a sum, so the spread is the direct consequence of the three-cell circulation being the only thing shaping rainfall. Every knob on it trades one Earth comparison for another:

| | log₁₀ sd | land median | rain in wettest 5% | |
|---|---|---|---|---|
| shipped | 0.833 | 321 mm | 47.5% | |
| `background_rainout_km` ×2 | 1.048 | 264 mm | 55.0% | drier median, fatter tail |
| `background_rainout_km` ×0.5 | 1.072 | 321 mm | 42.1% | |
| `eddy` 0.07 → 0.70 | 0.916 | 231 mm | 51.9% | narrower, median falls |
| `land_recycling` 0.85 | 1.047 | 357 mm | 42.6% | mild, and fights the shadow |
| `iterations` 400 → 1200 | 1.055 | 317 mm | 47.0% | converged; not a solver artifact |

(The single-solve figures run wider than the shipped 0.833 because averaging four seasons compresses the distribution.) This is the same limit as the missing desert belt below: a surface-only model has no aloft.

**What can be fixed is the anchor, and it was set on the wrong statistic.** Absolute depths are not derivable from this model, so the anchor is a free choice — and the mean was the wrong free choice, because with a tail that heavy, pinning the mean at Earth's 1000 mm divided everyone else down to a median of 321 mm where Earth's is ~600. Half the world was arid by construction. Judged on Köppen's arid share, which is what actually reads absolute millimetres:

| anchor | mean 1000 | median 450 | median 600 | median 750 |
|---|---|---|---|---|
| Köppen B (Earth 30%) | 41.3% | 35.4% | **30.5%** | 27.0% |
| tropical rainforest (Earth 11%) | ~7% | 11% | 15% | 18% |
| runoff ratio R/P (Earth ~0.40) | 0.59 | 0.66 | 0.72 | 0.76 |

600 mm ships because it is Earth's figure, not because it scored best. The costs are the shape showing through: the wet tail overshoots the rainforest share, and a heavy-tailed rainfall runs off more efficiently than an even one, so R/P drifts further from Earth. One wrinkle worth knowing — the seasonal path normalises each of its four solves and the median is not linear, so averaging four fields that each read 600 mm gave a year reading 803. The year is re-anchored after averaging and the same factor carried into the season stack, which leaves the seasonal *shape* Köppen reads untouched.

**Two limitations, recorded rather than papered over:**

1. **No subtropical desert belt.** Earth's great deserts sit near 25–30° because air there descends from the top of the Hadley cell already wrung dry over the ITCZ — the horse latitudes are dry because their air *arrives* dry. A surface-only model has no aloft, so the profile declines monotonically from the equator instead of dipping at 30°. Three surrogates were tried and measured; all failed, one of them making the belt *wetter*. Fixing it properly needs a second vertical level.
2. **Orographic strength and interior wetness are in direct tension.** Evapotranspiration recycling is what keeps continental interiors from being absolute desert, but recycled moisture refills the lee side and cancels the very contrast the orographic term creates. No setting of the two reaches Earth on both counts at once.

### Stage 4: the coupled loop

Erode, recompute climate on the new topography, re-erode against the new discharge, repeat. The doc called this the stage most implementations skip and the one that most improves plausibility; it is also the cheapest of the four to implement, because the coupling needs no new solver.

**Couple through discharge, not through a fudged erodibility.** Stream power is properly `E = K·Q^m·S^n`; drainage area is only a stand-in for discharge under uniform rainfall. Accumulating `area × P/mean(P)` instead of area restores the real variable, keeps `Q` in area units so `K` retains its calibrated meaning, and is algebraically identical to scaling `K` by `P^m`. It also degenerates *exactly* to the uncoupled run when precipitation is uniform — bit-identical, which makes a strong regression test.

**It does what it is supposed to do.** Against an uncoupled run of the same total duration, so the comparison is about *where* erosion went rather than how much there was:

| | uncoupled | coupled | change |
|---|---|---|---|
| Wettest quartile | 314 m | 262 m | **−52 m** |
| Driest quartile | 106 m | 334 m | **+228 m** |
| corr(log P, elevation) | +0.165 | −0.053 | |

Wet ground incises deeper, arid ground holds its relief, and rain shadows stop being a property of the precipitation map and start being a property of the topography.

**The coupling amplifies itself.** Measured while building Stage 6's tests: the isolated orographic ratio is 0.92 on uncoupled terrain — no signal at all — against 1.27 after coupling. Wet windward slopes get incised, which deepens the relief that drives the orographic rainfall in the first place. So Stage 3's rain shadows are weak in isolation and only become real once Stage 4 has run; the climate tests are therefore run against coupled terrain, since uncoupled is a state the pipeline never emits.

Worth noting for the testbed: **the drainage metrics are blind to this stage.** θ and Hack's exponent sit at 0.454 and 0.535 whether the run is coupled or not, and stayed there even while a normalisation bug was inflating peaks to 15 km. They measure network *geometry*, which climate does not change — it changes where the erosion goes, not how the channels organise. Judge Stage 4 by the wet/dry elevation contrast, not by the scaling exponents.

**Normalise on the quantity relief responds to, which is `w^(-m/n)` — not on the weight.** This took three attempts, and every wrong version hid behind healthy-looking metrics, because θ and Hack's exponent describe the drainage *network* and all that was wrong was its vertical scale.

| normalisation | what went wrong | n=785 peaks |
|---|---|---|
| mean weight = 1 | skew pushes the typical cell below 1 and against the floor, so *every* cell erodes ~1.7× too slowly | 15.4 km |
| median weight = 1 | typical cell correct, but relief responds to `w^(-m/n)` and the skew makes *its* mean exceed 1 | 13.7 km |
| **mean `w^(-m/n)` = 1** | — | **9.1 km** |

Solve for the scale by bisection; it is self-calibrating and needs no per-resolution tuning. That matters because the precipitation skew *sharpens* with resolution, so the earlier normalisations drifted: the coupled/uncoupled peak ratio ran 1.16× at n=256 rising to 1.63× at n=785, and is now 1.43× / 1.50× — flat.

**The residual ~1.45× is not a bug and should not be removed.** Relief goes as `Q^(-m/n)`, so cells at the discharge floor erode `0.5^(-0.45) = 1.37×` slower — and arid rain-shadow orogens are precisely where the highest ground is. The Altiplano and Tibet are high *because* they are dry. Suppressing this would suppress the effect the stage exists to produce.

**So the level is fixed by recalibrating `K`, and that is principled rather than a fudge.** Stage 4 replaces drainage area with discharge, which is a different parameterisation — `K` genuinely carries a different meaning there, and calibrating it against the uncoupled model was the actual mistake. Production `K` is 3.7e-6 coupled against ~2.5e-6 uncoupled at n=785; both sit well inside the published 1e-7–1e-4 range. A useful side effect: the faster erosion converges, so peaks now settle (9,108 m at every pass) where before they were still climbing at the end of the run.

The discharge floor still trades peak height against contrast, measured at n=256 over three passes against an uncoupled peak of 6,688 m:

| floor | 0.15 | 0.30 | 0.50 |
|---|---|---|---|
| peak elevation | 11,950 m | 9,531 m | **7,735 m** |
| wet−dry contrast | −293 m | −246 m | **−220 m** |

**Two or three passes is indeed enough.** Mean elevation change between successive passes ran 356 → 107 → 55 m at n=128; the loop settles quickly because most of the topography is set by uplift and only the *distribution* of erosion responds to climate.

**And a refinement to the uplift-contrast finding above.** The tectonic uplift field has a craton-to-orogen contrast of ~250×, far past the 4× where smooth-noise uplift wrecked the concavity measurement — yet per-basin θ comes out at 0.452 with an IQR of 0.004. The reason is that what matters is uplift variation *within* a basin, not the global ratio. Tectonic uplift clamps most of the continent to a flat cratonic floor with sharp belt transitions, so most basins sit in genuinely uniform `U`; smooth fBm varied continuously everywhere, contaminating every basin. **Realistic tectonics is better for the metric than a gentler but smoother uplift field** — so the `contrast_cap` knob is available but not needed by default.

---

## Implementation stack

**Python + Numba.** One language for both the compute core and the analysis harness — no PyO3/maturin bridge, no two-language build, and the whole scientific ecosystem (`scipy.ndimage`, `scikit-image`, `rasterio`, matplotlib, Jupyter/marimo) is in reach for the metrics work, which is where most of your time will actually go.

Rust was the earlier recommendation, argued partly from deploying the generator as a service. With the service gone that half of the case went with it, and the remaining performance gap is smaller than it looks: expect roughly 1.5–3× on the sequential graph kernels, not the 100–1000× that applies to *interpreted* Python — `@njit` code goes through LLVM the same as Rust does. The no-GIL parallelism argument is also mostly moot here, because priority-flood and the topological sort are inherently sequential in either language.

**Measured at production scale** (6,162,252 cells), which settles the feasibility question:

| Kernel | Time | Share |
|---|---|---|
| Priority-flood depression fill | 1.00 s | 88% |
| Topological ordering | 0.07 s | 6% |
| Receivers (steepest descent) | 0.03 s | 3% |
| Flow accumulation | 0.03 s | 3% |
| **Per erosion iteration** | **1.13 s** | |

That is ~38 minutes for 2000 timesteps. Comfortably fast enough; no algorithmic workaround for the fill is needed. Worth knowing: cost drops as the landscape organises — a coherent terrain has far fewer pits than a noisy one, and the fill measured 3× slower on white noise than on a realistic field.

**The traps, in the order they will bite:**

1. **float32 discipline.** NumPy defaults to `float64`, which doubles a 6.2M-cell field from 25 MB to 50 MB and halves SIMD width. Worse, inside `@njit` a bare Python float literal promotes a float32 array to float64 mid-expression. Declare `dtype=np.float32` everywhere and wrap hot-loop literals as `np.float32(0.5)`. This is the quiet 2× you will not otherwise notice.
2. **Never pass a grid or stack *object* into jitted code.** `@jitclass` is fragile and slow to compile. Pass the raw arrays; keep a thin Python wrapper that unpacks.
3. **Hand-roll the binary heap for priority-flood.** Numba supports `heapq`, but over typed lists, with per-operation overhead that matters when you push millions of times. A heap over two preallocated arrays is ~40 lines and reliably fast.
4. **`cache=True` on every `@njit`.** Compilation costs tens of seconds for large kernels; without disk caching you pay it every restart, which is corrosive in a testbed.
5. **Write your own 3D noise inside `@njit`.** The PyPI `noise` and `opensimplex` packages are either C extensions you cannot call from nopython mode or pure Python and far too slow. Simplex + fBm + domain warp is ~100 lines of scalar math.
6. **Determinism under `prange`.** Parallel float reductions have no fixed summation order, and in a chaotic system that compounds into visibly different worlds. Use fixed chunk sizes with a deterministic combine, and test it explicitly: same seed, one thread versus N, assert bit-identical output.

`fastscapelib` and `landlab` are Python-native and worth having as **reference implementations to check your slope–area exponents against** — but both are built for planar grids, so neither is a foundation here.

## Testbed architecture notes

- **Operators over a field stack.** Every algorithm is `fn(FieldStack, Params) -> FieldStack`, where the stack holds named flat `(N,)` arrays (`elevation`, `sediment`, `water`, `precipitation`, `uplift`, `flow_accum`, …) over the tiling. Pipelines become declarative config (TOML via stdlib `tomllib`), which makes A/B comparison trivial.
- **Deterministic seeding per stage** so you can vary one stage while holding the rest fixed.
- **Checkpoint between stages** — stage 2 is slow; you don't want to re-run it to test five climate models.
- **Multi-resolution** — prototype at `n ≈ 100–200` (100k–400k cells), validate at `n = 785`. Note that some algorithms are not scale-invariant, so parameters won't transfer directly; erosion coefficients in particular need rescaling. Any parameter denominated in *cells* rather than physical units silently breaks this — see `min_island_area_km2` below.
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
- Davy & Lague (2009) — sediment-flux formulation; what you need for deposition, deltas and fans.
- Górski et al. (2005) — HEALPix; the equal-area alternative to the tiling chosen here.
- Heikes & Randall (1995) — icosahedral geodesic grids for solving PDEs on the sphere.
- Guérin et al., "Interactive Example-Based Terrain Authoring with Conditional GANs" (2017).
- Cordonnier et al., "Forming Terrains by Glacial Erosion" (2023).

---

# Stage 1 spec: crust generation

The operator that produces the landmass outlines. It is the root of the pipeline DAG — it requires nothing but the domain, and everything downstream requires something it provides.

## Design principle

**Generate crustal thickness, not elevation.** Elevation falls out of isostasy; sea level is a threshold applied afterward. This matters because it makes the coastline a *consequence* rather than a drawn line, which means the continental shelf, the shelf break, and the relationship between coast and mountain all come out consistent for free. It also makes sea level a free knob afterward — the same world at a glacial lowstand costs one parameter change, not a regeneration.

**Crust type is independent of the plates.** An earlier version of this spec assigned continental-versus-oceanic per *plate*. That is geologically wrong and it shows: six of Earth's seven major plates carry both crust types — the African, North American, South American, Eurasian, Antarctic and Indo-Australian plates each hold a continent plus a large tract of ocean floor, and only the Pacific is near-purely oceanic. Assigning per plate makes every continent inherit a Voronoi silhouette at whole-continent scale. The straight rifted margins that motivated Voronoi in the first place (Africa's west coast, South America's east coast) are single *edges* of a coastline, not the shape of a whole landmass.

So: build `crust_thickness` from low-frequency domain-warped 3D noise, independent of the plates, and use plates only for `boundary_type`, `boundary_stress` and `uplift`. This is strictly less code than the per-plate version — it deletes a sub-step and a parameter.

## Field stack contract

```
FieldStack
  domain    { n, cell_count, radius_km }          -- grid regenerable from n alone
  fields    name -> flat (N,) array (typed)
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

All fields are flat `(N,)` arrays over the tiling.

| Field | Type | Unit | Purpose |
|---|---|---|---|
| `plate_id` | u16 | — | which plate owns each cell |
| `plates` | table | — | per plate: site, Euler pole axis, angular rate |
| `crust_thickness` | f32 | km | the master field; everything else derives from it |
| `crust_density` | f32 | g/cm³ | continental ≠ oceanic; needed for correct isostasy |
| `boundary_type` | u8 | enum | none / convergent / divergent / transform |
| `boundary_stress` | f32 | mm/yr | signed normal component of relative plate motion |
| `elevation` | f32 | m | isostatic, pre-erosion |
| `sea_level_m` | scalar | m | solved, not assumed |
| `land_mask` | u8 | — | derived from elevation vs. sea level |
| `margin_class` | u8 | enum | inland / passive / active — drives coastal realism downstream |
| `uplift` | f32 | m/Myr | the `U` term for stage 2 |

`uplift` is arguably a separate operator (`derive_uplift`) that consumes `boundary_type` and `boundary_stress`. Splitting it is worthwhile: it lets you swap uplift models without regenerating plates.

## Sub-steps

| # | Sub-step | What it does |
|---|---|---|
| 1 | `build_thickness` | low-frequency domain-warped 3D noise → continental vs. oceanic thickness, to hit target crust fraction |
| 2 | `warp_margins` | multi-octave domain-warped fBm added to thickness near the transition; amplitude keyed to margin type |
| 3 | `seed_plates` | Poisson-disk sample sites on the sphere, Lloyd-relax 2–3× for even plate sizes |
| 4 | `partition` | spherical Voronoi (nearest seed by dot product) → `plate_id` |
| 5 | `assign_motion` | Euler pole axis + angular rate per plate |
| 6 | `compute_boundaries` | per boundary cell, decompose relative velocity into normal/tangential → type + stress |
| 7 | `apply_tectonic_thickness` | thicken at convergent, thin at divergent, modulating the field from sub-step 1 |
| 8 | `isostatic_elevation` | thickness + density → elevation |
| 9 | `resolve_sea_level` | quantile search on the area-weighted elevation histogram for target land fraction |
| 10 | `prune_islands` | remove components below minimum area |
| 11 | `classify_margins` | coastal cells → passive/active from nearest boundary type |

Each is independently testable, which is the point of the decomposition — sub-step 2 is where most of the tuning happens and you don't want to re-run 1 to iterate on it.

**Two sub-steps from the planar version are gone.** `classify_plates` disappears with the per-plate crust bug (see Design principle). `enforce_edge_ocean` — margin falloff, then flood-fill from the border demoting any land that touches it — exists only to give a planar map an ocean rim, and a globe has no border. The `border_margin` parameter goes with it, which also recovers the ~42% of a square map that a 0.12 margin was sterilising.

## Isostasy

Airy balance over a compensation depth gives:

```
z = K + t · (1 − ρ_crust / ρ_mantle)
```

With ρ_mantle = 3.3, continental crust at t = 35 km / ρ = 2.8 gives 5.30 km; oceanic at t = 7 km / ρ = 2.9 gives 0.85 km, a difference of 4.45 km. `K` is a datum offset — set it arbitrarily and let sub-step 9 find sea level.

**Do not use that 4.45 km as a correctness check.** An earlier version of this spec claimed it matched the ~4.5 km real gap between mean continental surface and abyssal plain "closely enough that you don't need to fudge it." That comparison is invalid: 4.45 km is a **dry-column** figure, while the real-world number is **water-loaded**. Redo the balance with a water column (ρ_w = 1.03) at these densities and the gap is ~6 km — too deep, not slightly too shallow. The consequence is that a faithful implementation of the formula above fails the check, and a faithful implementation of the *physics* misses the target. Either treat 4.45 km as "what this simplified formula prints" and nothing more, or add the water term and retune `t_oceanic` / `rho_oceanic` against a real target. What you must not do is let the number silently anchor your parameter tuning.

*Refinement worth adding later:* water loading depresses submerged columns further, which deepens ocean basins relative to this simple form. The sea-level quantile search partly absorbs the difference in surface appearance, but not in the bathymetric profile.

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `plate_count` | 18 | 12–30; drives orogen layout, no longer continent shape |
| `lloyd_iterations` | 3 | more = more uniform plate sizes, less natural variety |
| `continental_fraction` | 0.40 | of *surface*, not of land — the rest drowns as shelf |
| `crust_octaves` | 4 | low-frequency thickness field; controls continent count and size spread |
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
| `min_island_area_km2` | 1000 | below this it's speckle, not archipelago |
| `convergent_thickening_km` | +15 | crustal root under orogens |
| `divergent_thinning_km` | −8 | rift and ridge thinning |

**Sub-seed derivation matters.** Derive each sub-step's RNG from `hash(stage_seed, substep_name)` rather than drawing sequentially from one stream. Otherwise changing `warp_octaves` reshuffles the plate layout and you can't A/B anything.

**Express every parameter in physical units, never in cells.** `min_island_area_km2` was originally `min_island_area_cells = 12`, which breaks the multi-resolution workflow this testbed depends on — the same world would prune different islands at `n = 200` and `n = 785`. On a sphere it is worse still, since cell area itself varies across the grid. Audit the table for any other cell-denominated value.

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

- `land_fraction` within tolerance of target, computed area-weighted.
- No land component smaller than `min_island_area_km2`.
- `crust_thickness > 0` everywhere; `plate_id` covers the full domain.
- Every coastal cell has a `margin_class` other than inland.
- Every landmass is surrounded by ocean — which on a globe is now automatic rather than enforced, so this is an assertion about connected components, not a construction step.

## Stage-specific metrics

Beyond the global metrics: **landmass size distribution** (should be heavy-tailed — one or two dominant continents, then a steep drop, then many small islands; a flat distribution means the crust field's low-frequency content is too weak), **coastline fractal dimension per margin class** (target roughly 1.05–1.15 passive, 1.2–1.3 active — measured by great-circle ruler walk, not box-counting; see Validation), and **shelf width distribution by margin class** (passive should be several times wider than active).
