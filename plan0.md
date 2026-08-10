# Terrain project: spherical grid decision + doc revision

**Context:** `C:\Projects\Terrain\terrain-algorithms.md` was written assuming a planar map with forced-ocean borders, and a Rust core justified partly by a future map-generation *service*. Two things have now changed: (1) no service — generation runs locally on Windows and emits a static file that gets uploaded; (2) the world is a **globe**, with the equator divided into 4096 cells, all distances spherical.

Both changes cut scope. The service change removes the job-queue/deployment section and weakens the case for Rust. The sphere change deletes an entire subsystem the doc spent a lot of ink on (`enforce_edge_ocean`, `border_margin`, margin falloff, flood-fill demotion, and the postcondition asserting no land near the edge) — the doc itself predicted this: *"on a sphere or a torus this whole constraint evaporates."* It also makes several things physically *more* correct, not just different.

This plan covers the grid investigation you asked for, plus the doc revisions the two changes force. It carries forward the findings from the earlier critique that are still live.

---

## Part 1 — Grid choice

### What "4096 cells around the equator" pins down

| Quantity | Value |
|---|---|
| Mean Earth radius `R` | 6371 km |
| Equatorial circumference `2πR` | 40 030 km |
| Target cell spacing `d = C/4096` | **9.77 km** |
| Target cell area (hexagon, `(√3/2)d²`) | 82.7 km² |
| Sphere area `4πR²` | 510.06 × 10⁶ km² |
| **Implied global cell count** | **≈ 6.17 million** |

That is the real budget number: ~6.2M cells, ~25 MB per `f32` field, ~370 MB for a 15-field stack. Comfortable on a desktop. Note this is *not* 4096 × 2048 = 8.4M — a lon/lat raster wastes cells at the poles and is neither equal-area nor compact, which is exactly why you asked the question.

### Candidates

| Tiling | Equal area | Compactness | Neighbours | Hits 6.2M? | Verdict |
|---|---|---|---|---|---|
| **Goldberg / geodesic hex** (dual of subdivided icosahedron) | ±few % after relaxation | **best possible** (hexagons) | **uniform 6** (12 pentagons) | yes, `n=785` → 6 162 252 | **recommended** |
| **HEALPix** | **exact, by construction** | good (diamonds; worse in polar caps) | 8, irregular at 8 corner pixels | not at power-of-2 `Nside` | strong runner-up |
| Cubed sphere | 1.3–1.5× variation (equiangular) | square-ish, corner distortion | 4, singular at 8 corners | tunable | simplest indexing, worst uniformity |
| Fibonacci lattice + spherical Voronoi | ±few % | near-hex, no singularities at all | irregular (~6) | any count exactly | good fallback, no hierarchy |
| H3 / `h3o` | ~2× variation | hexagons | uniform 6 | **no** — res 5 = 2.0M, res 6 = 14.1M | ruled out |
| Lon/lat raster | no (cos φ) | no (slivers at poles) | 4/8 | — | ruled out |

### Recommendation: class-I Goldberg grid, frequency `n = 785`

Subdivide each icosahedron edge into 785 segments, project to the sphere, take the dual → 6 162 252 cells, all hexagons except exactly 12 pentagons at the icosahedron vertices. Then run a few iterations of spring/Lloyd relaxation (Heikes–Randall style) to pull area variation down to a few percent.

Why this over HEALPix, given HEALPix is *exactly* equal-area:

- **Compactness was an explicit requirement and hexagons are the optimum.** HEALPix pixels are diamonds, and polar-cap pixels are noticeably more distorted than equatorial ones.
- **Uniform 6-neighbour connectivity is a direct realism win for the thing the doc says produces realism.** The survey flags D8's channel-snapping to 45° increments as a known artifact. On a hex grid that artifact largely disappears — six evenly-spaced directions, no orthogonal-vs-diagonal asymmetry, no need for D∞ or MFD/D8 hybridisation. This is the single strongest argument.
- **`n` is freely tunable**, so it hits 9.77 km exactly. HEALPix `NESTED` indexing needs `Nside` to be a power of two, which forces you to 3.1M cells (12.7 km) or 12.6M cells (6.4 km) — you overshoot or undershoot by ~1.5×. `Nside = 1024` does give *literally* 4096 pixels on the equatorial ring, but that's the diamonds' diagonal spacing, so it's a finer grid than you asked for at 2× the compute.
- **Proven for exactly this workload.** MPAS, ICON and DYNAMICO all use icosahedral hex meshes for solving PDEs on a sphere, for these reasons.

Cost of the choice: no closed-form `(i,j)` index, so cells are addressed by a flat integer with an explicit neighbour array (`6 × 6.2M × u32` ≈ 148 MB, or derive on the fly). That is a non-issue here — the doc's Stage 2 design is already receiver-array/flow-graph based, which wants exactly this shape. HEALPix's isolatitude rings and free spherical-harmonic transforms are the one thing you give up; see Part 2, item 5.

**Keep the choice swappable.** Define a `SphericalGrid` trait — `cell_count`, `neighbours(i)`, `area(i)`, `edge_length(i,j)`, `arc_distance(i,j)`, `lat_lon(i)`, `unit_vector(i)` — and write every operator against it. Almost all the pipeline is then grid-agnostic and HEALPix becomes a swap rather than a rewrite. Build the grid generator parameterised by `n`; prototype at `n ≈ 100–200` (100k–400k cells) and run production at 785.

### What the sphere makes *better*, not just different

- **Plate motion becomes correct physics.** Real plate motion is rotation about an Euler pole. On a plane that's a hack; on a sphere it's the actual model — axis unit vector + angular rate per plate, relative velocity at a boundary from the cross product. Simpler *and* right.
- **Noise gets simpler.** Sample 3D noise at each cell's unit vector. Seamless everywhere by construction, no pole pinching, no wrap seam. Domain warping warps the 3D position. This removes a whole class of planar-map artifacts for free.
- **Spherical Voronoi for plates is trivial** — nearest seed by maximum dot product of unit vectors.
- **Latitude comes free per cell**, so the Category 5 climate model (Hadley/Ferrel/polar wind bands, insolation, Coriolis) drops in without the contortions a torus would have forced.

### Spherical math to get right

- **Distance:** great-circle. Precompute the actual per-edge arc lengths once at grid build — they vary a few percent and using a constant `d` will bias erosion.
- **Slope:** `Δz / (R · Δσ)` with `Δσ` the central angle. Never a planar `√(Δx²+Δy²)`.
- **Hillslope diffusion `D∇²z`:** finite-volume on the mesh — `∂z_i/∂t = (D/A_i) · Σ_j (z_j − z_i)/L_ij · e_ij`, where `e_ij` is the shared Voronoi edge length, `L_ij` the centre-to-centre arc distance, `A_i` the cell area. The Delaunay/Voronoi dual is orthogonal, which is what makes this discretisation clean — a further argument for the hex grid.
- **Drainage area:** accumulate real cell **areas**, not cell counts. With ±few% area variation, counting cells quietly corrupts your Hack's-law and slope–area metrics — the exact diagnostics the whole testbed exists to produce.
- **Flow routing:** steepest descent over 6 neighbours. Priority-flood is graph-based and works unchanged. Handle the 12 pentagons via the neighbour array — do not special-case them in operator code.

---

## Part 2 — Doc revisions forced by the two changes

Ranked by value-for-cost.

### 1. Delete the map-edge subsystem. (S)
Remove sub-step 9 `enforce_edge_ocean`, the `border_margin` parameter, and the postcondition "no land cell within `border_margin` of the map edge." This also recovers the ~42% of a square map that constraint was sterilising. Sub-step 11 `prune_islands` stays.

### 2. Delete the service section. (S)
The job-queue/202-polling design, object storage, instance memory sizing, and the Docker/Linux cross-build advice are all dead. Replace with: run locally, checkpoint stages to disk, export a static file.

### 3. Revisit the language recommendation — it was half-argued from the service. (S, decision only)
Rust is still defensible on the compute argument alone: priority-flood over 6M cells and a topologically-ordered stream-power solve are pointer/graph work that NumPy cannot vectorise, and `rayon` has no GIL. But "static binary in a scratch container" is no longer a reason, and it was a real part of the case. If you'd rather not spend the Rust learning budget, **Python + Numba is now a materially stronger option than it was** — the deployment drawback vanished, and Numba compiles exactly the tight index-loop code these two algorithms are made of. My recommendation is still Rust core + Python analysis harness, but it's now a closer call and worth a deliberate decision rather than inheriting the old one.

Note this also weakens the earlier suggestion to prototype against `fastscapelib`/`landlab`: both are built for planar rasters. `fastscapelib` does have an unstructured `TriMesh` grid you could feed a spherical mesh into, but you'd be off their well-trodden path, and Landlab's hex grid is planar. Treat them as a **reference implementation to validate your slope–area exponents against**, not as a foundation.

### 4. Crust type is assigned per plate. Still the biggest realism bug. (S)
Carried forward from the earlier review and unaffected by the sphere change. Sub-step 3 `classify_plates` marks each *plate* continental or oceanic. Six of Earth's seven major plates carry both — only the Pacific is near-purely oceanic. As specified, every continent inherits a Voronoi silhouette at whole-continent scale. Fix is *less* code: generate `crust_thickness` from low-frequency 3D domain-warped noise independent of plates, and use plates only for `boundary_type` / `boundary_stress` / `uplift`. Drops sub-step 3 and the `clustering` parameter.

### 5. Rework the metrics for a sphere. (M)
- Slope–area, Hack's law, Horton ratios, drainage density, hypsometry: all graph- or area-based, work unchanged. Good.
- **Power spectral density does not.** On a sphere the analogue is the spherical-harmonic power spectrum `C_ℓ`. On a hex grid that means resampling to HEALPix and using an SHT — this is HEALPix's one real advantage and the reason to keep the grid trait swappable. Alternatively drop PSD; it's the weakest metric on the list and slope–area concavity already catches the failure mode it targets.
- **Coastline fractal dimension by box-counting doesn't transfer** — there are no boxes on a sphere. Replace with a divider/ruler-walk along the coastline polyline using great-circle steps at several step lengths, and regress. Same output, valid on a sphere.
- Comparison against real DEMs still works: reproject SRTM/Copernicus tiles onto a patch of the same grid rather than comparing in tile space.

### 6. The isostasy sanity check is a false test. (S)
Carried forward. `z = K + t(1 − ρ_c/ρ_m)` is the **dry-column** form; its 4.45 km is being compared against ~4.5 km of real continental-surface-to-abyssal-plain relief, which is a **water-loaded** figure. Add water loading (ρ_w = 1.03) at the spec'd densities and the gap is ~6 km — too deep. A correct implementation of the stated formula fails the doc's check, and correct physics misses the doc's target. Either relabel 4.45 as "what this simplified formula prints, not a physical validation," or add the water term and retune `t_oceanic`/`rho_oceanic` against a real target.

### 7. Build order is still inverted. (S, reorder only)
The doc's thesis is "erosion gives you realism," then it specifies 12 sub-steps and 18 parameters of crust generation before a single erosion iteration exists — output that erosion will largely overwrite. Suggested order, now with the grid first:
1. **Grid generator + `SphericalGrid` trait**, validated (see below).
2. Metrics module against real DEM tiles.
3. Throwaway Stage 1: 3D warped fBm thickness → isostasy → sea-level quantile. ~50 lines on top of the grid.
4. Stage 2 Braun–Willett + hillslope diffusion, tuned until slope–area concavity lands in 0.4–0.6.
5. *Then* the real Stage 1.

### 8. Resolution-dependent parameters. (S)
`min_island_area_cells = 12` must become km² — it already broke multi-resolution prototyping, and now cell areas vary across the sphere too. Audit the parameter table for anything else denominated in cells.

### 9. Add sediment deposition to Stage 2. (M — the one place to *add* scope)
Carried forward. `∂z/∂t = U − K·A^m·S^n` is detachment-limited and never deposits. The doc lists deltas, alluvial fans and basin fill under Stage 6 "hydrology finalize" as if cosmetic; they're the visible output of sediment flux, and bolting them on afterward is the "looks right in a screenshot" failure the doc warns about elsewhere. Plan for a transport-limited or hybrid formulation (Davy & Lague 2009 flux term) inside Stage 2. Also unresolved: with `n ≠ 1` the Braun–Willett linear solve needs a per-node Newton iteration.

### 10. Output format. (M)
Two artifacts, not one. **Native:** the flat per-cell field arrays plus `n` and the seed — the grid is fully regenerable from `n`, so nothing else need be stored. **Static upload file:** resample to an equirectangular GeoTIFF (8192 × 4096 is a natural match for 4096-at-the-equator). Keeping GeoTIFF is still right — real DEMs arrive as GeoTIFF and your metrics code then runs unmodified on both.

---

## Not worth doing

- Rewriting Categories 1–8. I checked the load-bearing numbers and all ten citations; they're correct.
- H3/`h3o` — resolution steps of 7× can't hit 6.2M, and ~2× area variation fails your equal-area requirement.
- A cubed sphere. Simpler indexing isn't worth the corner singularities and 1.3–1.5× area variation.
- Exact hierarchical nesting for multi-resolution. Checkpointing is per-stage, not per-level; separate grids at `n = 98 / 196 / 393` are fine and cost nothing.
- `clustering` parameter — dropped by item 4.

---

## Verification

**Grid generator** — assert before anything is built on top:
- Cell count is exactly `10n² + 2`; exactly 12 cells have 5 neighbours and the rest have 6.
- Σ cell areas = `4πR²` to within float tolerance; report min/max/σ of cell area — target < ±5% after relaxation, and record it as a known bias in the metrics.
- Neighbour relation is symmetric (`j ∈ N(i) ⟺ i ∈ N(j)`); no duplicates; no self-references.
- Every cell's neighbour arc distances are within a few percent of 9.77 km at `n = 785`.
- Round-trip `cell → lat/lon → cell` is the identity for a large random sample.
- Walk a great circle from a random cell and confirm it crosses a pentagon without a discontinuity in step length.

**Spherical math** — a planar-assumption regression test: place two cells at the same longitude separation at the equator and at 60° N, and assert the computed distance differs by ≈ cos(60°) = 0.5. Anything treating x/y as planar fails this.

**Pipeline, once Stage 2 runs** — slope–area concavity in 0.4–0.6 and Hack exponent ≈ 0.57, measured on the sphere with real cell areas, compared against the same metrics computed on a real DEM tile.