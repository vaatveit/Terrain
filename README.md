# Terrain

A batch world generator on a sphere. Nine stages — tectonics, landscape evolution, climate,
glaciation, hydrology, biomes — producing a georeferenced elevation model and its climate.

The design rationale, the measurements behind every parameter, and an honest record of what
does and does not reproduce real terrain are in **[terrain-algorithms.md](terrain-algorithms.md)**.
This file is the operational reference: how to run it, what every knob does, and what comes out.

**[pipeline-review.html](pipeline-review.html)** is a third view and the one to open first if
you are new to the code: a stage-by-stage walk through what each operator does, with every
tuned constant listed at the point it acts and sorted into the cell-scale / world-scale /
vertical classes the rescale turns on. It ends with the findings from the review that produced
it — including two bugs it found, and two claims of its own that measurement later corrected.
Self-contained HTML; open it from a clone, no build step. (A hosted copy exists but is private
to the author's account, so the file in the repo is the canonical one.)

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy numba scipy rasterio matplotlib pytest
```

Generate a world:

```bash
.venv\Scripts\python.exe scripts\generate_world.py --n 256 --seed 1 --out data/world01
```

Render it from the exported rasters:

```bash
.venv\Scripts\python.exe scripts\render_world.py data/world01
```

Run the tests:

```bash
.venv\Scripts\python.exe -m pytest tests -q
```

### Resolution and runtime

`--n` is the geodesic grid frequency; the grid has `10n²+2` cells. Production is **n=785**,
which is what "4096 cells around the equator" works out to.

**Cell size is the invariant of this world, not the radius.** Every grid the pipeline
builds — production or test fixture — has **100 m cells**, and the radius follows from `n`
(`radius_for_cell_km`). Lower `n` therefore gives a *smaller sample of the same world*, not
a coarser version of it, so every tuned length scale keeps its meaning at every resolution.

| n | cells | radius | surface | cell |
|---|---|---|---|---|
| 128 | 163,842 | 10.6 km | 1,409 km² | 100 m |
| 384 | 1,474,562 | 31.8 km | 12,676 km² | 100 m |
| 785 | 6,162,252 | 64.9 km | 52,973 km² | 100 m |

Three specs were asked of the production grid — 100 m cells, 4096 cells around the equator,
and `R = 65.19 km` — and the 12 pentagons mean they cannot all hold at once. Measured,
`n × mean angular spacing` converges to 1.209055, so holding the radius exact gives 100.40 m
cells and holding cell size exact gives 4080 equatorial cells. **Cell size wins**, because it
is what every length scale below is denominated against. The 0.4% is immaterial to results.

Why 100 m at all: at the old 9.77 km cells every land cell already carried more drainage
area (82.7 km²) than a real channel head needs (0.01–1 km²), so there was no hillslope
domain and `D` acted as a numerical smoother. See **[Cell size](#cell-size)** below.

The grid is cached to `data/grid_n<N>_r<R>m.npz` on first build (34 s and 650 MB at n=785)
and reused. The radius is in the filename because a grid cached before the rescale has the
same `n` and a 98× larger sphere, and would otherwise be loaded silently.

### Command-line options

| flag | default | meaning |
|---|---|---|
| `--n` | 256 | grid frequency |
| `--seed` | 1 | world seed; same seed gives the same world on any machine |
| `--steps` | 600 | total erosion timesteps, split across passes |
| `--passes` | 3 | climate–erosion coupling passes |
| `--out` | `data/world_n<N>_s<S>` | output directory |
| `--no-glacial` | off | skip stage 5 |

`render_world.py` takes `--width` (default 4096) and `--resample` (`nearest` \| `barycentric` \| `c1`) — the raster is a rendering concern and lives there now, not in the export.

---

## Output

`generate_world.py` writes two files: the mesh and its provenance.

### `world.npz` — the native archive

Field arrays over the mesh, plus the little needed to rebuild it. **The grid is not stored**
— it regenerates from `n`, which is why the archive is 79 MB rather than 730 MB at n=785. The twelve months of
temperature and rainfall do not share that economy: 24 float32 rows that barely compress,
about 85% of the archive. **Measured at n=785, seed 42: 646 MB.** That figure read "near
600 MB (estimated, not yet measured)" until a production world was run with everything in it.
The ten tectonic fields are 31.0 MB of that and the three resource fields 0.37 MB.

```python
from terrain.export import load_native
grid, fields, meta = load_native("data/world01/world.npz")
# grid   : a rebuilt Grid (geometry, neighbours, areas)
# fields : dict of (N,) arrays, keys below
# meta   : {"n": 785, "seed": 1, "params": {...}}
```

| key | dtype | units | meaning |
|---|---|---|---|
| `field_elevation` | float32 | m | metres relative to sea level; negative is sea floor |
| `field_land_mask` | uint8 | — | 1 = land, 0 = ocean. Ocean is the largest **connected** body of water, so an inland basin is land even where its floor is below sea level |
| `field_temperature_c` | float32 | °C | mean annual surface temperature |
| `field_precipitation_mm` | float32 | mm/yr | mean annual precipitation |
| `field_temperature_monthly_c` | float32 | °C | **(12, N)**, one row per month; averages to `field_temperature_c` |
| `field_precipitation_monthly_mm` | float32 | mm | **(12, N)**, rainfall *in* each month; sums to `field_precipitation_mm`. The same months Köppen classifies on |
| `field_biome` | uint8 | code | Whittaker; see the biome table below |
| `field_koppen` | int8 | code | Köppen–Geiger; −1 at sea. Alongside `field_biome`, not instead of it |
| `field_slope` | float32 | m/m | gradient toward the flow receiver |
| `field_drainage_km2` | float32 | km² | upstream drainage area |
| `field_flow_receiver` | int32 | — | **cell id** the water flows into, always a neighbour. `rec[i] == i` terminates the tree (ocean or outlet). Chain it to walk downstream |
| `field_discharge_km3_per_yr` | float32 | km³/yr | water volume routed through the cell (Stage 6) |
| `field_river_width_m` | float32 | m | channel width, `90·√Q`; **0 means no river**. There is no separate river mask — the channel threshold puts a 0.65 m floor under every river cell, so `> 0` is exactly that test |
| `field_margin_class` | uint8 | code | `0` inland, `1` passive coast, `2` active coast. Re-derived **after** the reflood, on the coastline that ships — the Stage 1 margins describe a coast that no longer exists |
| `field_plate_id` | int16 | — | which plate the cell rides on |
| `field_boundary_type` | uint8 | code | `0` not a boundary, `1` convergent, `2` divergent, `3` transform. Set only on cells that touch another plate |
| `field_boundary_stress` | float32 | mm/yr | **normal** component of relative plate motion on a boundary cell, smoothed along strike; positive is closing. 0 off-boundary |
| `field_boundary_speed` | float32 | mm/yr | **total** relative plate speed on a boundary cell; 0 off-boundary. Exact, where `boundary_stress` is an estimate — do not difference the two to get slip; use the next field |
| `field_boundary_slip` | float32 | mm/yr | the part of that speed **along** the boundary; 0 off-boundary. `slip / speed` is obliquity: 0 head-on, 1 pure strike-slip |
| `field_boundary_dist_km` | float32 | km | great-circle distance to the nearest boundary cell. Beyond the uplift reach (0.188 R) it is a sentinel of 10× the reach, not a distance |
| `field_nearest_boundary` | int32 | — | **cell id** of that nearest boundary cell; −1 beyond the reach. Index the three boundary fields with it to carry them inland |
| `field_crust_thickness` | float32 | km | Stage 1 crust, after tectonic thickening and thinning |
| `field_uplift` | float32 | m/Myr | the uplift erosion ran on; negative in rift grabens, 0 over the Stage 1 ocean |
| `field_is_continental` | uint8 | — | 1 = continental crust, 0 = oceanic. Not the land mask: continental crust is 40% of the surface and the excess drowns as shelf |
| `field_resource_copper` | uint8 | 0–255 | prospectivity for copper, 255 = 1.0. See [Mineral prospectivity](#mineral-prospectivity) |
| `field_resource_iron` | uint8 | 0–255 | the same for iron. Rests on **invented** cratons; see below |
| `field_resource_gold` | uint8 | 0–255 | the same for gold |
| `field_lake_mask` | uint8 | — | 1 where a lake stands |
| `field_lake_level_m` | float32 | m | water surface elevation; NaN off-lake |
| `field_lake_depth_m` | float32 | m | surface minus bed; 0 off-lake |
| `field_lake_id` | int32 | — | index into `manifest.json` → `lakes`; −1 off-lake |
| `field_endorheic` | uint8 | — | 1 where the lake is terminal (no outlet to the sea) |
| `_n` | int64 | — | grid frequency; rebuilds the geometry |
| `_radius_km` | float64 | km | planet radius (64.927) |
| `_seed` | int64 | — | world seed |
| `_params_json` | str | — | every parameter dataclass, as JSON |

All field arrays are length `10n²+2` and share cell indexing with `grid`; the two monthly
fields are `(12, 10n²+2)`, month-major, so `field[k]` is month `k` everywhere.

The months are rebuilt from Stage 3's four solves through the year by a two-harmonic fit
(`koppen.monthly_from_seasons`), which four evenly spaced samples determine exactly — they
are an interpolation, not twelve independent solves. `manifest.json` → `months` dates each
row: `year_fraction` is where it sits in the year, and `declination_deg` is the (lagged)
sun it was solved under, so a positive value is northern summer.

### Which edge does the water cross?

`field_flow_receiver` is a cell id, not an edge. The edge is one step away, and nothing extra
needs to be stored to get it — `grid.neighbours` is azimuthally ordered, so the slot the
water leaves by is just where the receiver sits in that list, and the cells draining *in* are
the neighbours whose own receiver is this cell:

```python
exit_slot   = np.flatnonzero(grid.neighbours[i] == rec[i])          # empty at ocean/outlets
entry_slots = [s for s, j in enumerate(grid.neighbours[i])
               if j >= 0 and rec[j] == i]                           # >1 is a confluence
```

D6 steepest descent picks exactly one receiver, so there is always exactly one exit and up to
five entries. The crossing point on a shared edge is `normalize(xyz[i] + xyz[j])`: a spherical
Voronoi edge lies on the perpendicular bisector of the centre-to-centre arc, so that arc's
midpoint is on the edge by construction. Note the tree is built on the depression-**filled**
surface, so flow directions stay defined across lake interiors rather than stopping at the
shore.

### Mineral prospectivity

The three `field_resource_*` fields say where a sim should look for copper, iron and gold.
Each is a **score, not a deposit**: it marks a belt or a district, and a single deposit is far
smaller than one cell. To place resources, sample cells in proportion to the field times cell
area, with the sim's own seed:

```python
w = fields["resource_gold"] / 255.0 * grid.area
cells = rng.choice(grid.cell_count, size=20, replace=False, p=w / w.sum())
```

Scores are absolute, not rescaled per world, so a field can be zero everywhere: a world with
no continental rift has no rift copper. Check `w.sum() > 0` before sampling.

**The iron field is mostly invented, and so is part of the copper.** Most of the world's iron
sits in old cratons, and this model has no crustal age and no mantle to derive a craton
from. `minerals.invent_cratons` draws them from noise instead: 30% of continental crust, with
iron formation in 15% of that. Because those are fixed shares, iron covers about the same
4% of land on every world, where copper and gold vary with the plates. A rift also scores
most of its copper only near a craton margin. Gold owes nothing to them.

They are a post-process on the tectonic fields above, so a saved world can be re-scored
without regenerating it. `terrain/minerals.py` explains how each belt is scored, and
[mineral-distribution.md](mineral-distribution.md) holds the research and what is still to
build. `manifest.json` → `params.minerals` records the weights that were used.

Those weights are tuned, not chosen: `scripts/tune_mineral_weights.py` measures how much
ground each deposit type covers on production worlds and sets each weight so the type is
sampled as often as intended. A weight is therefore not a share. Placer gold is weighted 0.60
to be picked 15% of the time, because it covers so little ground. Re-run the script on
production worlds only; the result does not transfer from small ones.

```python
from terrain import minerals
cratons = minerals.invent_cratons(grid, meta["seed"], fields["is_continental"])
keys = ("land_mask", "is_continental", "boundary_stress", "boundary_speed", "boundary_slip",
        "boundary_dist_km", "nearest_boundary", "uplift", "precipitation_mm", "temperature_c",
        "flow_receiver", "slope", "river_width_m")
ore = minerals.resources(radius_km=grid.radius_km,
                         cell_km=float(grid.arc_dist[grid.neighbours >= 0].mean()),
                         craton_margin_km=cratons["craton_margin_km"],
                         iron_formation=cratons["iron_formation"],
                         **{k: fields[k] for k in keys})
```

### `manifest.json`

Grid, seed, land fraction, elevation range, every parameter dataclass, the measured
metrics, the lake table that `field_lake_id` indexes into, and the dates of the twelve
monthly rows.

### No rasters

The export used to also write three equirectangular GeoTIFFs. It no longer does. Nothing
consumed them: `render_world.py` samples the mesh directly, and the metrics that judge the
world run on the mesh graph. The one claim that justified GeoTIFF — one metrics
implementation serving generated and real terrain — never actually paid out, because the
export was ~5 km per pixel against SRTM's 30 m and no window was both wide enough for three
decades of drainage area and small enough to avoid equirectangular distortion. Real DEMs
still arrive as GeoTIFF and `terrain/dem.py` still reads them; that never needed the export
to be one too.

To get a raster, sample the mesh — `amplify.equirect_points` plus `amplify.interpolate`,
which is what `render_world.py` does in five lines.

Biome codes: `0` ocean, `1` ice, `2` tundra, `3` boreal forest, `4` temperate grassland,
`5` woodland/shrubland, `6` temperate seasonal forest, `7` temperate rainforest,
`8` subtropical desert, `9` savanna, `10` tropical seasonal forest, `11` tropical rainforest,
`12` temperate desert, `13` cold desert, `14` lake.

Köppen codes: `0` Af, `1` Am, `2` Aw, `3` BWh, `4` BWk, `5` BSh, `6` BSk, `7` Csa, `8` Cfa,
`9` Cfb, `10` Dfa, `11` Dfb, `12` Dfc, `13` ET, `14` EF, and `−1` at sea.

**Both classifications ship, and they answer different questions.** Whittaker reads the
annual mean temperature and rainfall, so a summer-dry Mediterranean climate and a summer-wet
monsoon one with the same annual totals are the same biome to it. Köppen reads *when* the
rain falls and how cold the winter gets, which is why it needed Stage 3 to grow a seasonal
cycle first. Whittaker stays the renderer's colour source and the classification every tuned
share in `terrain-algorithms.md` is measured against.

`0` and `14` in the biome table are surface types rather than Whittaker classes, and they are
how fresh water is told from salt. Elevation cannot do it: an inland basin floor sits below sea level exactly as
the abyssal plain does. A lake is land by `field_land_mask` — it is a body of water enclosed
by land — so code `14` is the only thing that marks it.

---

## Tunable parameters

Every stage takes a dataclass. Construct one with overrides and pass it in — anything not
named keeps its default.

```python
from terrain import climate, coupled, stage1, stage2
from terrain.grid import build_goldberg

grid  = build_goldberg(256)
crust = stage1.generate_crust_tectonic(
    grid, params=stage1.CrustParams(target_land_fraction=0.40), seed=7)

result = coupled.run(
    grid, crust.elevation, crust.uplift, ~crust.land_mask,
    lem=stage2.LemParams(K=5e-6, m=0.5),
    clim=climate.ClimateParams(t_equator_c=32.0),
    passes=3, steps_per_pass=200)
```

`scripts/generate_world.py` exposes only `--target-land-fraction`,
`--continental-fraction` and `--crust-frequency` and takes defaults for the rest, so **edit
the dataclass defaults or call the modules directly** to change anything not exposed as a
CLI flag.

### Stage 1 — crust · `stage1.CrustParams`

| parameter | default | effect |
|---|---|---|
| `target_land_fraction` | 0.29 | share of the globe above sea level; sea level is solved to hit it |
| `continental_fraction` | 0.40 | share of *surface* that is continental crust; the excess drowns as shelf |
| `crust_frequency` | 0.7 | **continent size.** Lower = fewer, larger continents. At 1.6 the world was 340 landmasses with the largest holding 15% of land |
| `crust_octaves` | 3 | detail in the continent outline |
| `warp_amp` / `warp_frequency` | 0.35 / 2.2 | domain warp; what makes coastlines fractal rather than blobby |
| `transition` | 0.12 | width of the continental-slope ramp |
| `relief_octaves/frequency/km` | 7 / 6.0 / 1.0 | pre-erosion thickness texture. The 1.0 km is **vertical** — cut ~3× from the Earth value with the rest of the isostasy |
| `t_continental_km` / `t_oceanic_km` | 12 / 2.4 | crustal thickness by type. Isostasy is **radius-blind** — Earth values would put 4,455 m of relief on a 64.9 km sphere |
| `rho_continental` / `rho_oceanic` / `rho_mantle` | 2.8 / 2.9 / 3.3 | isostasy densities |
| `min_island_area_km2` | 0.104 | below this an island is speckle. Cell-scale (~12 cells), so it does **not** rescale with the radius |

**The two fractions are not independent, and the reported land fraction is neither of
them.** Sea level is solved so that `target_land_fraction` of the surface is above the
datum — that part is exact. But `land_mask` is not `elevation > 0`: ocean is the *largest
connected* water body, and every other below-datum pool is an inland sea for Stage 6 to
fill. The gap `continental_fraction − target_land_fraction` is shelf that has to drown, and
the more continental crust there is, the more of that shelf is landlocked rather than joined
to the sea. Measured at n=128, seed 42:

| flags | above datum | inland seas | reported `land_fraction` |
|---|---|---|---|
| 0.29 / 0.40 (default) | 29.0% | 1.6% | **30.5%** |
| 0.50 / 0.65 | 49.9% | 15.0% | **64.9%** |

**There is also a ceiling, and it is the climate that sets it.** Past roughly 45–50% land
the continents are wider than a surface-only moisture model can carry water across, the
precipitation anchor divides by a median that has underflowed, and `compute_seasons` raises
rather than emitting a scaled-noise world — see `climate._annual_anchor_scale`. Measured at
n=128: 0.40/0.50 (43% land) runs; 0.45/0.55 (62% land) is refused. The limit is about the
width of the continents rather than the flags as such, so a higher `crust_frequency` buys
back room by breaking them up.

It does so unevenly, not past a threshold. At n=128, seed 42, 0.50/0.60, steps 1200, passes
6: `--crust-frequency` 0.7 and 0.8 are refused, 0.9, 1.0, 1.1, 1.4 and 1.6 run, and 1.2 is
refused again. What tracked the outcome was the land fraction Stage 1 reported: every run
above ~63% was refused and every run at or below 62% ran, so the frequency matters through
how much of the outline lands as inland sea. The worlds that ran end at 42–48% land after
the reflood. A pass at n=128 does not carry to n=785: at 0.7 the median/mean ratio there was
1.3e-5 against 0.0095 at n=128.

Passing that check is not a certificate, but the thing it used to warn about is now largely
fixed. The dead continental interior turned out to be two problems stacked, and neither was
the one the docs named:

1. **The moisture sweep was a reach in cells**, so it stopped at 0.62 R at n=785 and the
   interiors were unfinished rather than dry. See `advection_span_km`.
2. **Every moisture source depended on local rain.** `land_recycling` returns a fraction of
   what just fell, so its gain is under one and a far enough interior converges to zero
   however it is tuned. `river_evaporation` is the fix: open water inland evaporates, and a
   river carries water routed by the *topography* rather than by the wind, so it is a source
   the interior did not have to earn.

Measured on the full coupled pipeline at n=785, seed 1, river evaporation off against on:

| | <1 mm/yr | log₁₀ sd | wettest 5% holds | 4–8 km | 8–16 km | 16+ km |
|---|---|---|---|---|---|---|
| off | 13.30% | 2.146 | 64.1% | 453 | 127 | 111 |
| on | **0.02%** | **0.681** | 53.1% | 512 | 424 | 887 |

The second column is the larger result and was not the goal: the rainfall distribution being
twice Earth's width is a defect recorded in `_advect_step` as the one that six swept knobs
could not move, and this moves it to 0.681 against Earth's ~0.45. **The cost is the tropics**
— Köppen Af goes 11.1% → 18.1% of land against Earth's ~5%, since the water the interiors
gain lands mostly where it is already warm. The arid classes come back toward Earth in
exchange (BWk 18.3% → 13.4%).

### Stage 1 — plates · `plates.PlateParams`

| parameter | default | effect |
|---|---|---|
| `plate_count` | 18 | 12–30. Drives orogen layout, **not** continent shape — crust type comes from noise, independently |
| `min_separation_deg` | 18 | Poisson-disk spacing; without it plate sizes are lottery-distributed |
| `lloyd_iterations` | 3 | higher = more uniform plates, less natural variety |
| `max_speed_mm_per_yr` | 80 | plate speed; Earth runs 10–100 |
| `transform_ratio` | 2.5 | a boundary is transform when tangential motion exceeds this × normal. At 1.0 you get 44% transform against Earth's ~20% — 51% now that the type is read off the boundary's own normal (`plates.boundary_types`), and 23–24% at 2.5 either way |
| `boundary_warp` / `_frequency` | 0.10 / 2.5 | bends plate boundaries. **Set to 0 and mountain ranges render as dead-straight lines** |
| `stress_smoothing_passes` | 8 | averages boundary stress along strike before it is propagated inland. **At 0, each boundary cell hands its stress to its own wedge of interior cells and orogens render as hard stripes fanning out from the belt** |
| `u_max_m_per_myr` | 100 | peak orogenic uplift. Sets relief with `LemParams.K`, and hilltop curvature (`U/D`) with `LemParams.D` |
| `craton_floor_m_per_myr` | 2.0 | uplift on stable interiors |
| `orogen_decay_km` | 2.04 | how far uplift reaches from a collision boundary. **World-scale** — rescaled by `PlateParams.for_radius` |
| `arc_offset_km` / `arc_width_km` | 1.53 / 1.02 | volcanic arc position inland of a trench. **World-scale** — 150 / 100 km ÷ 98.125 |
| `transform_decay_km` | 1.02 | restraining-bend reach at a transform boundary. **World-scale.** Was a bare `100.0` inside `derive_uplift` rather than a field, so `for_radius` never saw it: over a 12.2 km uplift reach `exp(−d/100)` runs 1.000 to 0.885, painting all 17.6% of transform land with a uniform slab instead of a ridge |
| `rift_shoulder_km` | 0.82 | rift flank width. **World-scale** — 80 km ÷ 98.125. The graben **subsides**, which is what creates tectonic lake basins |
| `convergent_thickening_km` / `divergent_thinning_km` | 5.0 / 2.7 | crustal root and rift thinning. **Vertical**, so not a radius rescale — cut with the relief budget, as `CrustParams` was |

### Stages 2 & 4 — erosion · `stage2.LemParams`

| parameter | default | effect |
|---|---|---|
| `m` / `n` | 0.45 / 1.0 | **stream-power exponents. Steady-state concavity θ = m/n exactly**, so these set the primary realism target (0.4–0.6). `n ≠ 1` costs a Newton iteration per node |
| `K` | 1.2e-6 | **erodibility — sets relief, not shape.** Peak elevation scales as `U/K`. No longer resolution-dependent (cell size is fixed), but still different for coupled runs; see below |
| `D` | 0.02 | hillslope diffusion, m²/yr. **Now a first-class term** — it produces a real hillslope regime, and must be sub-cycled (`diffusion_substeps`) because the explicit stability limit fell from 239 Myr to 25 kyr |
| `G` | 0.0 | sediment deposition (Davy–Lague). 0 = pure detachment-limited |
| `repose_deg` | 33.0 | angle of repose for `mass_wasting`, run after diffusion each step. A **separate operator from `D`**, not a nonlinear diffusivity, because Roering's critical-slope flux diverges at `Sc` and would worsen the sub-stepping. 0 disables it. A near-no-op on this world — see Known limitations |
| `wasting_relax` / `wasting_passes` | 0.25 / 3 | share of the excess removed per pass, and passes per timestep. Must stay under ~1/3: a cell has six edges that can all shed at once, and above that a pass overshoots and inverts the slope it was relaxing |
| `max_subsidence_m` | 500 | **total** subsidence one cell may accumulate over the run. Erosion balances uplift; nothing balances subsidence with `G = 0`, so a constant rate over 300 Myr drove a graben to 5,063 m. Rift subsidence is finite in nature, so the bound is a total rather than a smaller rate |
| `dt_myr` | 0.5 | timestep. The implicit solver is unconditionally stable, so this can be large |
| `fill_every` | 1 | priority-flood cadence. 1 is affordable |

**`K` needs recalibrating for coupling, and no longer for resolution:**

- **Coupling.** Stage 4 substitutes discharge for drainage area, a different
  parameterisation in which `K` means something different. Use **1.2e-6 coupled** and
  **~2.5e-6 uncoupled** at n=785.
- **Resolution — this used to matter and does not now.** `K=1e-6` gave 5.8 km peaks at
  n=128 and 17.6 km at n=785 *when the sphere was Earth-sized and cells coarsened with `n`*.
  Cell size is fixed at 100 m for every `n`, so the smallest resolved drainage area is
  always 0.0086 km² and that dependence is gone. What remains is a weak dependence on the
  *largest* basin, which still grows with the world: peak relief goes as `L^0.156`, so
  n=785 runs about 12% above n=384 at the same `K`.

To retarget peak elevation, scale `K` by the ratio you want to remove: relief ∝ 1/K exactly
for `n=1`.

### Stage 4 — coupling · `coupled.run`

| parameter | default | effect |
|---|---|---|
| `passes` | 3 | climate–erosion alternations. Two or three is enough; change between passes ran 356 → 107 → 55 m |
| `steps_per_pass` | 200 | erosion steps per pass |
| `discharge_floor` | 0.30 | minimum discharge as a fraction of typical. **Caps the relief multiplier at `floor^(-m/n)` = 1.72×**, and the cap binds — 17–21% of land sits exactly at the floor. The measured coupled/uncoupled *peak* ratio is 1.43–1.50, below the cap. Lower = more relief in deserts and more contrast |

### Stage 3 — climate · `climate.ClimateParams`

| parameter | default | effect |
|---|---|---|
| `t_equator_c` / `t_pole_c` | 27 / −25 | sea-level temperature range |
| `lapse_rate_c_per_km` | 6.5 | standard environmental lapse rate. Physics — leave it alone; it does not care how big the planet is |
| `climate_relief_gain` | **1.0 (off)** | vertical exaggeration on the elevation the lapse rate reads. Would correct the fact that the rescale redenominated every world-scale *length* and missed elevation, which is emergent rather than a parameter — but **at the 7.7 that would close the gap it breaks the moisture model at n=785 and the pipeline raises.** Kept, disabled, and documented below rather than deleted |
| `continental_cooling_c` | 15 | annual-mean cooling deep inside continents, weighted by latitude. The main non-zonal signal in temperature |
| `continentality_c` / `_scale_km` | 6 / 15.3 | seasonal amplitude inland, and its decay length (1500 km ÷ 98.125) |
| `wind_speed` / `meridional_fraction` | 1.0 / 0.3 | only the ratio matters; direction comes from the three-cell bands |
| `meander_deg` / `_frequency` | 7 / 1.8 | Rossby-like waviness. **At 0 the zero-wind latitudes render as ruler-straight rainfall stripes** |
| `orographic_coeff` | 0.004 | rainout per metre of forced ascent. Raising it strengthens rain shadows and dries interiors |
| `land_recycling` | 0.70 | evapotranspiration returned to the air. **In direct tension with `orographic_coeff`** — recycled moisture refills the lee |
| `background_rainout_km` | 8.33 | distance over which airborne moisture decays to 1/e (818 km on Earth) |
| `subsidence_floor` | 0.10 | rainout suppression where Hadley air descends |
| `eddy` | 0.07 | isotropic mixing. **Must be > 0** or the zero-wind latitudes become moisture traps |
| `saturation_gain` | 0.06 | how fast warm air holds more water |
| `evaporation` | 1.0 | ocean source strength |
| `river_evaporation` | 1.0 | open water inland evaporates too, at the ocean's strength per unit area. **The only moisture source that does not depend on the rain falling where it stands** — `land_recycling` returns a fraction of local rain, so its gain is under one and a far interior converges to zero however it is tuned. Set to 0.0 to restore the ocean-only model |
| `river_full_area_km2` | 40000 | drainage area at which a channel fills a 100 m cell, from hydraulic geometry (W ≈ 0.5·√A metres). Nothing on this world approaches it: the biggest basin at n=256 drains 270 km², giving an 8.2 m channel |
| `advection_span_km` | 200 | how far the fixed-point sweep carries moisture, as a distance rather than a sweep count. **Replaces `iterations` = 400, which was a reach in cells** — 40 km, so 1.9 R at n=256 but 0.62 R at n=785, and the big worlds were left unfinished in exactly the interiors that were the complaint |
| `target_median_precip_mm` | 600 | final scaling, on Earth's land **median**; the model gives a pattern, not absolute depths, and its wet tail makes the mean a statistic of the tail rather than of the world |
| `ekman_deg` | 45 | deflection of the surface current from the wind; right in the north, left in the south |
| `current_scale_km` | 10 | distance a current carries its heat anomaly (981 km on Earth) |
| `maritime_scale_km` | 5 | how far inland maritime air keeps that anomaly (491 km on Earth) |
| `ocean_heat_c` | 1.0 | gain on the anomaly where it comes ashore. **0 disables ocean heat transport**; zeroing `wind_speed` does not |
| `strip_zonal_mean` | true | keep only the departure from the latitude mean, so poleward transport is not counted twice against `t_pole_c` |
| `axial_tilt_deg` | 23.44 | drives the declination, and so the seasons |
| `seasonal_lag_days` | 30 | seasons trail the sun, because the ocean stores the difference |
| `seasonal_gain` | 1.0 | swing per unit of insolation departure: 1.4 °C at the equator, 14.6 at 45°, 20.7 at the pole. **0 removes seasons** |
| `seasonal_continentality` / `sea_seasonal_damping` | 0.8 / 0.35 | land swings up to 1.8× the baseline, sea only 0.35×. **This differential heating is the monsoon's engine** — without it the June land–sea contrast is 0.58 °C with land *colder* than sea |
| `itcz_follow` / `itcz_taper_deg` | 0.4 / 45 | how far the circulation follows the sun, and where that dies away. Tapered so the mid-latitude westerlies do not reverse |
| `monsoon_c` | 0.2 | pull toward summer-heated land. **In direct tension with `orographic_coeff`** — see the table in `climate.py`; 0.6 buys a stronger monsoon for a third of the rain shadow |

**Every `_km` above is world km at the reference radius (64.927 km) and is rescaled by the
actual radius**, so it denotes a fixed fraction of the sphere. `background_rainout_km`,
`current_scale_km` and `maritime_scale_km` were fractions lost *per advection step* until
that turned out to be a length in cells rather than in kilometres: the same 0.012 rainout
reached 0.13 R at n=785 but 1.57 R at n=64, so a small test world had moisture crossing it
entirely before decaying. The values above reproduce the old per-step numbers exactly at
n=785, where they were tuned.

### Stage 5 — glacial · `glacial.GlacialParams`

| parameter | default | effect |
|---|---|---|
| `freeze_c` | −2.0 | ice accumulates below this mean annual temperature |
| `balance_per_degc` | 0.35 | metres of ice per year per degree below freezing |
| `K` / `m` / `n` | 0.003 / 0.30 / 1.0 | abrasion law on ice flux. `m` is below the fluvial value on purpose. `K` was 0.02 — inside the published 0.1–10 mm/yr band, and still removing 34% of the world's total relief in 300 kyr, because the band spans two orders of magnitude and this world's relief budget is 432 m, not several km |
| `slope_baseline_km` | 1.22 | **driving slope is regional, not local.** With a local slope, overdeepenings are impossible rather than merely rare. Cell-scale (12 cells of flow path) |
| `max_balance_m_per_yr` | 3.0 | ceiling on accumulation |
| `min_flux_km3_per_yr` | 1.04e-6 | below this it is snow, not a glacier. Cell-scale — 1e-2 ÷ 98.125², a volume so R² |
| `steps` / `dt_yr` | 60 / 5000 | 300 kyr of glaciation |
| `max_incision_per_step_m` | 8.0 | explicit-scheme guard, and **not a dead one**: 7,417 land cells hit it at n=128 and the deepest total incision comes out at exactly 8 × 60 = 480.00 m, so wherever it binds the local rate is set by the clamp rather than by `K·Q^m·S^n`. `K` and this are not independent knobs |

### Stages 5b/5c — reflood and re-solve · no parameters

**The coastline is not a constant of the pipeline.** Stage 1 solves the land mask; Stages 2
and 5 then move the surface underneath it by hundreds of metres. Ice is driven by its own
surface slope and gouges below its outlet — that is what carves a fjord — and the water has
to be let in afterwards or the map carries dry ground below sea level. `stage1.reflood`
re-applies Stage 1's two rules to the eroded surface: below the datum is water, and ocean is
the largest connected body of it. A trough joined to the sea floods; a hole that is not
stays land and becomes a lake candidate for Stage 6.

**How much it takes depends on world size**, so quote the resolution with the figure:
29.4% → 22.7% at n=64, 29.6% → 25.6% at n=128, and at n=785 the shipped worlds export
0.295 ± 0.003 against a 0.29 target — i.e. by production the reflood removes *less* than
the inland basins add back. Drowning after erosion is a small-world effect: median land
elevation is about 10 m on a small relief budget, so ground lowered at all tends to go
under, and a larger world stands higher above its datum.

Stage 5c then re-solves the seasonal climate on that coastline, because everything
downstream reads absolute millimetres and degrees against the new mask. It does not feed
back into erosion — Stages 2 and 5 have already run — and Stage 5 deliberately keeps the
*pre*-reflood temperature, since that is the coastline the ice grew on. Cost is one extra
`compute_seasons` (four solves).

### Stage 6 — hydrology · `hydrology.HydrologyParams`

| parameter | default | effect |
|---|---|---|
| `pet_base_mm` / `pet_linear` / `pet_cubic` | 300 / 25 / 0.05 | Thornthwaite-style potential evapotranspiration from temperature. Runoff is `P − AET` on the **Turc–Pike** curve, `AET = P/sqrt(1 + (P/PET)²)`; the obvious `min(P, PET)` kinks at `P = PET` and gave 59% of land exactly zero runoff |
| `endorheic_margin_m` | 2 | how far below spill a lake must settle to count as terminal |
| `min_lake_area_km2` | 0.0415 | below this a depression is not a lake (~5 cells) |
| `channel_threshold_km3_per_yr` | 5.2e-5 | discharge defining a river (~6 cells of drainage) |
| `width_coeff_m` / `width_exponent` | 90 / 0.5 | `w = c·Q^0.5`, standard hydraulic geometry |

### Stage 8 — mesh sampling · `amplify.AmplifyParams`

> **The detail half of Stage 8 is off, and the rescale to 100 m cells is why.** Detail only
> means something when the target is finer than the mesh. At Earth radius that held — a
> ~9.8 km mesh against ~4.9 km pixels. At 100 m cells the render raster is ~1.5 px per cell,
> `octaves_for_width` caps the noise at **one octave**, and that octave has a ~2.3 km
> wavelength — **23 cells across**. That is not sub-grid texture, it is a coarse ripple laid
> over ground the landscape evolution already resolved. Nothing calls `amplify` with
> `detail=True` any more. The machinery is kept because it is correct and because rendering
> at 4× the mesh resolution would want it back.
>
> What Stage 8 is still *for* is the sampling below: `render_world.py` reads the mesh
> archive and has to get it onto a raster to draw it.

| parameter | default | effect |
|---|---|---|
| `max_amplitude_m` | 3.0 | detail height on the roughest ground. Denominated against the drop across **one cell** at `slope_reference`, not against world relief |
| `slope_reference` | 0.06 | slope at which detail is full strength |
| `flat_floor` | 0.18 | detail retained on flat ground. **At 0 the sea bed and every plain go glass-smooth** |
| `valley_smoothing_sphere_fraction` | 1.18e-4 | drainage area over which alluvial floors smooth out, as a **fraction of the sphere** (6.2 km² at production). **Must be a large-river area** — too small and it flattens the whole channel network |
| `ocean_scale` | 0.35 | sub-sea detail, muted by sediment |
| `octaves` / `frequency` / `ridged` | 6 / 180 / 0.35 | fractal character; `ridged` blends toward crests over blobs. `octaves` is a **ceiling** — the number actually used is capped against the raster width |
| `min_px_per_feature` | 3.0 | pixels the finest octave must span. **Detail below the pixel is not detail, it is aliasing** — it folds back as long straight moiré fringes across the terrain |
| `resample` | `"nearest"` | `nearest` \| `barycentric` \| `c1` — see below. **The default blends nothing** |

The first six rows only bite when detail is switched back on. `resample` is live.

> **`--resample` — the default blends nothing.** Both smoothing modes are opt-in, so a
> default raster contains no number the simulation did not produce.

| `--resample` | what it does | cost |
|---|---|---|
| `nearest` *(default)* | each pixel takes the value of the cell it falls in — mesh data unaltered, one cell per pixel, traceable | renders the hexagons |
| `barycentric` | linear over the mesh triangles; exact at cell centres and for a linear field, never overshoots | only C0 — its gradient jumps at triangle edges, which hillshading draws as a herringbone on steep ground |
| `c1` | Shepard blend of per-cell *linear polynomials* for elevation; facet tail (p99.5 / median of the second difference) on a mountain window falls **8808 → 20.1** | one extra k-nearest query; only pays above ~2 px per mesh cell, so pair it with a larger `--width` on `render_world.py` |

> Neither blend is the old inverse-distance weighting, which had zero gradient at every data
> point and so dimpled at each cell centre — 0.345 m of manufactured curvature on a
> perfectly smooth analytic control, against 0.0007 m for barycentric. `c1` also adds no
> blur: it recovers the full mesh peak (10,365 m against barycentric's 10,252 m), where a
> smoothing kernel at σ=0.85 cell spacings cost 528 m. Elevation is the only field offered
> `c1` — it is the only one whose derivative anyone looks at. The land mask and biome codes
> are always nearest, categorical values being unaverageable.

> **Do not apply amplification to the simulation mesh.** It perturbs elevations, which
> changes flow receivers, which rewires the drainage network — measured as θ moving
> 0.512 → 0.603. It is an output-side operation; metrics belong on the pre-amplification field.

---

## Validating a world

`terrain.metrics.report` computes the diagnostics, and the same code runs on real DEM
tiles via `terrain.dem.load_window`. Reference values are measured over **nine** SRTM
1-arcsec tiles — Appalachians, Olympics, Pyrenees, Sahara, Andes, Tien Shan, Amazon,
Mindanao, Uintas — each a 63 km window:

| metric | SRTM, 9 tiles | typical generated |
|---|---|---|
| slope–area concavity θ, per basin | 0.293 ± 0.131 (102 basins) | **0.285 ± 0.010** at n=785 |
| slope–area concavity θ, whole window | 0.311 ± 0.139 (range 0.130–0.516) | planet-wide 0.257 ± 0.056 — too seed-noisy to quote |
| Hack exponent h | 0.513 ± 0.057 | 0.559 ± 0.004 |
| Horton bifurcation Rb | 5.87 (range 4.67–6.95) | ~5.1 |

The per-basin row is at a **40-cell (0.344 km²)** channel threshold, matching the generated
figure it sits beside; the whole-window row is at the **1.0 km²** `validate_metrics.py` uses
on SRTM. θ is threshold-sensitive, so quote the threshold with it — at 0.344 km² the
whole-window mean falls to 0.236, because on a 30 m DEM that threshold drags the hillslope
limb into the fit. **Compare per-basin against
per-basin and use more than one tile** — this table used to read "SRTM east 0.516" against
the generated 0.285 and call the difference the largest open gap in the project. It was
neither: about 0.12 of it was per-basin-vs-whole-window, and the rest was that east is the
highest-θ tile of the nine. **Measure θ per basin, not planet-wide** — uplift varies across a
world, and that variation lands in a global regression. **Judge worlds by the mesh metrics,
not by the exported raster**: at ~5 km/px against SRTM's 30 m, no raster window is both wide
enough for three decades of drainage area and small enough to avoid equirectangular
distortion.

## Cell size

The grid is dimensionless. Radius only scales `arc_dist`, `edge_len` and `area`, so moving
from 9.77 km cells to 100 m changed no graph algorithm and no topology. What it changed is
**which processes sit above the grid scale**.

The number that decides that is not the angle of repose — it is the channel head. Real
channels initiate at drainage areas of 0.01–1 km². A cell at 9.77 km was **82.7 km²**, so
every land cell already carried more drainage than a channel head needs: there was no
hillslope domain anywhere in the world, and `D` was a numerical smoother rather than soil
creep. A cell at 100 m is **0.0086 km²**, below the threshold, so hillslopes exist.

**Measured, at n=384 over 400 steps.** With `D=0` slope falls monotonically with drainage
area from the very first bin — no hillslope regime, which is what the whole 9.77 km world
looked like. With `D=0.02` the slope–area curve has a hump: slope *rises* over 1→4 cells,
peaks at the channel head, then falls as `A^-θ`. That rising limb is the convex hillslope.

| D | channel head | θ (fluvial limb) | r² | sub-steps |
|---|---|---|---|---|
| 0.00 | none — monotonic | 0.457 | 0.991 | — |
| 0.01 | 0.021 km² | 0.402 | 0.927 | 10 |
| **0.02** | **0.034 km²** | **0.362** | 0.907 | 20 |
| 0.05 | 0.056 km² | 0.267 | 0.748 | 48 |

Every channel head there is inside the real 0.01–1 km² band, so that criterion does not
pick between them. `D=0.02` is the chosen trade.

**The drainage structure at production is good.** Measured on the n=785 world:

| measurement | 9.77 km cells | 100 m cells | real terrain, 9 SRTM tiles |
|---|---|---|---|
| land cells that are local maxima | 4.40% | **0.502%** | 0.287% mean, range 0.090–0.508% |
| median drainage area | 1.0 cells | **5.98 cells** | 3 cells median, range 2–13 |
| θ per basin, 40 cells | — | **0.285 ± 0.010** | **0.290** median, 102 basins |

That last row read 0.467 (r² 0.90) until it was re-measured on three shipped n=785 worlds.
The old figure predates the climate work — precipitation feeds discharge weight, which feeds
erosion — and it also asserted that θ *rises* with the channel threshold, where it is in fact
flat from 6 cells to 120 and falls beyond. Hack *h* is 0.559 ± 0.004 against a nine-tile
0.513 ± 0.057.

**The real-terrain column is measured, and it did not used to be.** It read 0.14–0.26% and
3–4 px "(SRTM)" for the first two rows and a published 0.4–0.6 band for θ — and this section
concluded from that θ row that "the concavity is not good". Measured per basin at the same
threshold over nine tiles, real terrain gives 0.290 against this world's 0.285. The local
maxima row is quoted as a range rather than a target because the DEM has eight neighbours to
the mesh's six, which biases the generated figure up on its own.

**Quote the per-basin figure, not the planet-wide one.** Across three seeds at n=785,
planet-wide θ at 40 cells came out 0.271 / 0.195 / 0.304 — a coefficient of variation of
21.7%, rising to 55% at a 400-cell threshold — while per-basin θ held 0.293 / 0.288 / 0.274
at 3.6%. A global regression pools basins with different uplift, and how many of each a world
happens to grow is a property of the seed. An earlier version of this section read a
0.444 / 0.346 / 0.271 decline across n=256 / 512 / 785 as a trend; those are planet-wide
figures with one seed each, against a seed sd of 0.056, so the trend is not shown by them.

The first two were recorded in terrain-algorithms.md as *unfixed* — a surface genuinely
rough at the cell scale, with half of all land having no upstream contribution at all. Both
were symptoms of having no hillslope domain, and both resolved with the grid rather than
with any change to the drainage code.

**The channel threshold had to move with it.** `metrics.report` uses one threshold both to
define channels and to fit the slope–area regression, and the regression is only valid on
the fluvial limb. 6 cells of drainage was safely fluvial at 9.77 km cells (500 km²); at
100 m cells 6 cells *is* the channel head, so fitting from there mixes the two regimes and
θ reads 0.392. `generate_world.py` now uses 40 cells.

Three consequences worth knowing before changing anything:

- **Diffusion must be sub-cycled.** The explicit limit is `A_i / (2 D Σ(e_ij/L_ij))`, and
  `Σ(e_ij/L_ij)` is dimensionless, so the limit tracks *cell area* and falls as the square
  of cell size — 239 Myr at 9.77 km, 25 kyr at 100 m, against a 0.5 Myr fluvial timestep.
  `diffusion_substeps` computes the count from the grid; a raw call at a fluvial `dt`
  violates the discrete maximum principle.
- **`U` and `D` are now answerable to each other.** Steady-state hilltop curvature is
  exactly `U/D`. The old `U=1200 m/Myr` with `D=0.05` implies 120 m of convexity *per cell*
  at 100 m — spikes. That inconsistency was always in the parameters; it was invisible only
  while hillslopes were sub-grid.
- **Two classes of length scale, and they scale oppositely.** *Cell-scale* parameters
  (`min_island_area_km2`, `min_lake_area_km2`, `channel_threshold_km3_per_yr`,
  `slope_baseline_km`) are absolute and must **not** be rescaled — cell size no longer
  varies with `n`. *World-scale* parameters (`orogen_decay_km`, `arc_*`, `rift_shoulder_km`,
  `continentality_scale_km`, `valley_smoothing_sphere_fraction`) are fractions of the world
  and **are** rescaled, by `PlateParams.for_radius` and equivalents. Getting one into the
  wrong class is silent: it just makes a fixture at low `n` test the wrong thing.

What this unlocks, with the drop needed for a 33° talus slope as the yardstick: it was
**6,371 m per cell** at 9.77 km, against 8 km of relief in the entire world, so no cell pair
ever qualified. At 100 m it is **65 m**, which ordinary mountain terrain clears. Thermal
talus, cirques, dunes and meanders all move from impossible to viable — none is implemented,
but `tests/test_glacial.py` now pins the geometry so a future grid change cannot quietly
take it back.

---

## Known limitations

Recorded with measurements in [terrain-algorithms.md](terrain-algorithms.md):

- **No subtropical desert belt from circulation.** Needs a vertical dimension the
  surface-only climate model does not have.
- **Precipitation is over-dispersed** — log₁₀ sd **0.681** against Earth's ~0.45, with the
  wettest 5% of land taking **53.1%** of all land rain against roughly 15%. It is *not*
  convergence (the wettest cells have the same incoming advection weight as everywhere
  else); it is the ascent factor and the moisture field multiplying, which is the same
  missing vertical dimension as the item above. **No knob moves it and one source did** —
  all six knobs swept against it are coefficients on transport that was already present,
  where `river_evaporation` took it from 2.146 to 0.681 at n=785. The anchor is set on
  Earth's land **median** rather than its mean to make the best of what is left.
- **Biomes remain visibly zonal**, and this is now understood rather than merely observed.
  Adding arid classes outside the tropics widened the desert belt's latitude wander from
  4.4° to 9.9°. Ocean-current heat transport and a seasonal cycle were then both built —
  the two things this entry used to name as the fix — and temperature zonality moved
  0.9926 → 0.9898, which is nothing. **The cause is that the world has no vertical
  dimension to spend.** Land elevation std is 104 m at production against Earth's ~800, so
  the lapse rate can contribute 0.68 °C where Earth gives ~5, against a latitude term with
  a std of 13.7 °C. It is not a mistuning and there is no knob: raising uplift is blocked
  by hilltop curvature, which is exactly `U/D` and already implies a 45° basal gradient,
  and reaching Earth's relief by growing the world would take 2.35 billion cells, 382× the
  production budget. (That block used to be argued from the measured slope percentiles
  instead — p99 62.5°, "already past the angle of repose". It should not be: 98.1% of the
  over-steep edges are a land cell against ocean held at base level, so the percentiles
  describe the coastline rather than any hillslope. See `plates.py`.) Relative to
  its own size this world is already **12.7× steeper** than Earth; what it lacks is
  absolute metres, which is precisely what the 100 m cell decision traded away.
- **No meanders, dunes, wave-cut notches or cirques.** All were sub-grid at 9.77 km cells;
  at 100 m all are now above the grid scale, and none is implemented. See
  [Cell size](#cell-size) for the measured thresholds. **Thermal talus is the exception —
  it is implemented and it does almost nothing**, which is the more useful result:
  `stage2.mass_wasting` relaxes any edge past `repose_deg`, and every repose angle from 45°
  to 30° gives the same world. 98.1% of the over-steep edges on this world are land against
  ocean at base level and 1.8% are lake rim, leaving 0.05% that are land-to-land; genuine
  hillslopes run p50 0.9° because `U/D` equilibrium does not put them near failure. It is
  kept on because glacial erosion does create real over-steep land (80 → 4,530 edges),
  almost all of which Stage 5b then floods.
- **Concavity is fine, and this bullet used to say it was the largest open gap in the
  project.** Per-basin θ is **0.285 ± 0.010** across three n=785 seeds, against **0.290**
  measured the same way over 102 basins in nine SRTM tiles. The gap it reported was against
  SRTM east's 0.516 — a whole-window fit on the highest tile of nine, compared to a per-basin
  median. Correcting either half shrinks it; correcting both closes it. Hack *h* is 0.559 ±
  0.004 against a nine-tile 0.513 ± 0.057. Planet-wide θ is still *not* a usable statistic
  here — 0.257 ± 0.056 at 40 cells, ±0.106 at 400 — so state per-basin, and state the
  threshold.
- **The temperature field is short of relief, and the obvious fix does not work.** This
  world's 104 m elevation sd gives the lapse rate 0.68 °C of influence where Earth's ~800 m
  gives ~5. `climate_relief_gain` exists to exaggerate the elevation the lapse rate reads and
  **ships at 1.0, disabled**, because at the 7.7 that would close the gap the pipeline raises
  at n=785: land median precipitation collapses to 7.4×10⁻⁶ of the land mean.

  The arithmetic is short enough that it should have come first. Moisture capacity is
  `clip(exp(0.06·(T−15)), 0.05, 4.0)`, which floors at **T < −34.9 °C**. A gain of 7.7 makes
  the lapse term 50 °C per km, so land near 700 m floors and the n=785 peak at 2,289 m is
  cooled by 115 °C. Capacity multiplies the **river-evaporation** source — the only land
  moisture recycling in the model — so it collapses and the continents cannot carry water
  inland.
- **It was validated at n=128, which structurally could not detect that.** The n=128 peak is
  504 m, so the worst cooling was 25 °C and nothing floored; drowning was unmoved and the only
  symptom was the raw windward/leeward ratio slipping from above 1.5 to 1.48, which was read
  as a tolerable cost. **The damage scales with absolute relief, and relief grows steeply with
  world size** — the peak's local exponent runs R^0.57 to R^1.13. Anything that reads elevation
  in metres has to be tested at n=785 or not at all.
- **What survives.** The bookkeeping argument stands: the rescale redenominated every
  world-scale length and missed elevation because it is an emergent field rather than a
  parameter. Every consumer except precipitation took the gain without complaint — Köppen,
  biomes, the glacial ELA and the export were all fine, and drowning did not move at either
  resolution, so `glacial.K` never needed recalibrating. Decoupling the one path
  temperature → capacity → river evaporation would make a large gain viable, at the cost of
  two temperature fields. Nobody should simply raise the number.
- **Continental interiors are too wet, by a seed-dependent amount.** The interior/coast
  rainfall ratio is 2.35 / 1.29 / 0.67 across three seeds where Earth is ≈0.33, so the bias
  is consistent even though only one of the three actually inverts. It tracks the largest
  basin in exact rank order, and max drainage area has a 75% coefficient of variation across
  seeds — the noisiest quantity in the model. See `climate.river_evaporation`.
- **Endorheic basins are rare** — two of twelve lakes at n=128 — and Turc–Pike runoff
  makes them rarer than the kinked balance did, by giving every catchment some inflow. The
  logic is unit-tested on constructed basins.
- **Glacial erosion drowns 14% of land** when the reflood pass lets the sea into what it
  carved, against 1% with the ice switched off. `K` has been recalibrated down (0.49 →
  0.12 mm/yr, 34% → 8.6% of the world's relief), which is most of the way, but the residual
  is hypsometry rather than the glacial rate: median land elevation is about 10 m, so ground
  lowered at all tends to go under.
- **Köppen D is 7% of land against Earth's 24%.** Continental climates need winters this
  world does not produce — the same relief and continentality shortfall as above.
- **An Earth-sized ice sheet is larger than this world.** Continental glaciation is no
  longer something the model can express; 21% of land glaciates as alpine/polar ice.
