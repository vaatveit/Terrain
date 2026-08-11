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
6. Hydrology finalize   →  depression fill/breach, lakes, rivers, channel widths
7. Biomes               →  Whittaker or Köppen from temperature × precipitation
8. Amplification        →  slope/curvature-modulated detail; optional ML refine
9. Export + metrics     →  native field arrays + equirectangular GeoTIFF
```

Stages 2 and 5 dominate runtime. Stage 4 is the one most implementations skip and the one that most improves plausibility at continental scale.

**Build them in a different order than you run them.** The thesis of this document is that erosion produces realism, so the schedule should reflect that rather than working left to right — most of Stage 1's careful output gets overwritten by Stage 2 anyway. Order: grid ✅ → **metrics module** ✅ → throwaway Stage 1 ✅ → **Stage 2, tuned until slope–area concavity lands in 0.4–0.6** ← *next* → *then* the real Stage 1, by which point you know which of its outputs Stage 2 actually consumes.

Note what the throwaway Stage 1 does **not** produce: meaningful continental relief. Its highest land sits ~560 m above sea level and mean continental freeboard is ~220 m, against a real ~840 m. That is the expected consequence of skipping tectonics — real high topography comes from crustal thickening under orogens (sub-step 7), not from noise on an undeformed crust. Mountains are Stage 2's uplift term's job, so this is not worth fixing in the throwaway.

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
