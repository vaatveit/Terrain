# Terrain

A batch world generator on a sphere. Nine stages — tectonics, landscape evolution, climate,
glaciation, hydrology, biomes — producing a georeferenced elevation model and its climate.

The design rationale, the measurements behind every parameter, and an honest record of what
does and does not reproduce real terrain are in **[terrain-algorithms.md](terrain-algorithms.md)**.
This file is the operational reference: how to run it, what every knob does, and what comes out.

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
which is what "4096 cells around the equator" works out to (`2πR/4096 = 9.77 km`).

| n | cells | spacing | full pipeline |
|---|---|---|---|
| 128 | 163,842 | 60 km | ~20 s |
| 256 | 655,362 | 30 km | ~70 s |
| 785 | 6,162,252 | 9.8 km | ~8 min |

The grid itself is cached to `data/grid_n<N>.npz` on first build (34 s and 650 MB at n=785)
and reused. It is regenerable from `n` alone, so it is never stored inside a world.

### Command-line options

| flag | default | meaning |
|---|---|---|
| `--n` | 256 | grid frequency |
| `--seed` | 1 | world seed; same seed gives the same world on any machine |
| `--steps` | 600 | total erosion timesteps, split across passes |
| `--passes` | 3 | climate–erosion coupling passes |
| `--width` | `8n` | export raster width in pixels (height is half) |
| `--out` | `data/world_n<N>_s<S>` | output directory |
| `--no-glacial` | off | skip stage 5 |
| `--verify` | off | check export fidelity and re-read the GeoTIFF through the DEM path |

---

## Output

`generate_world.py` writes five files.

### `world.npz` — the native archive

Field arrays over the mesh, plus the little needed to rebuild it. **The grid is not stored**
— it regenerates from `n`, which is why the archive is 79 MB rather than 730 MB at n=785.

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
| `field_land_mask` | uint8 | — | 1 = land, 0 = ocean (as decided by Stage 1) |
| `field_temperature_c` | float32 | °C | mean annual surface temperature |
| `field_precipitation_mm` | float32 | mm/yr | mean annual precipitation |
| `field_biome` | uint8 | code | see the biome table below |
| `field_slope` | float32 | m/m | gradient toward the flow receiver |
| `field_drainage_km2` | float32 | km² | upstream drainage area |
| `_n` | int64 | — | grid frequency; rebuilds the geometry |
| `_radius_km` | float64 | km | planet radius (6371) |
| `_seed` | int64 | — | world seed |
| `_params_json` | str | — | every parameter dataclass, as JSON |

All field arrays are length `10n²+2` and share cell indexing with `grid`.

### Rasters

Equirectangular, EPSG:4326, covering the full globe, `width × width/2`.

| file | bands | dtype | contents |
|---|---|---|---|
| `elevation.tif` | 1 | float32 | elevation in metres, with Stage 8 sub-grid detail |
| `climate.tif` | 2 | float32 | `temperature_c`, `precipitation_mm` |
| `biomes.tif` | 1 | uint8 | biome code, nearest-sampled (never interpolated) |
| `manifest.json` | — | — | grid, seed, raster spec, land fraction, parameters, metrics |

Biome codes: `0` ocean, `1` ice, `2` tundra, `3` boreal forest, `4` temperate grassland,
`5` woodland/shrubland, `6` temperate seasonal forest, `7` temperate rainforest,
`8` subtropical desert, `9` savanna, `10` tropical seasonal forest, `11` tropical rainforest,
`12` temperate desert, `13` cold desert.

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

`scripts/generate_world.py` uses defaults throughout, so **edit the dataclass defaults or
call the modules directly** to change anything not exposed as a CLI flag.

### Stage 1 — crust · `stage1.CrustParams`

| parameter | default | effect |
|---|---|---|
| `target_land_fraction` | 0.29 | share of the globe above sea level; sea level is solved to hit it |
| `continental_fraction` | 0.40 | share of *surface* that is continental crust; the excess drowns as shelf |
| `crust_frequency` | 0.7 | **continent size.** Lower = fewer, larger continents. At 1.6 the world was 340 landmasses with the largest holding 15% of land |
| `crust_octaves` | 3 | detail in the continent outline |
| `warp_amp` / `warp_frequency` | 0.35 / 2.2 | domain warp; what makes coastlines fractal rather than blobby |
| `transition` | 0.12 | width of the continental-slope ramp |
| `relief_octaves/frequency/km` | 7 / 6.0 / 3.0 | pre-erosion thickness texture |
| `t_continental_km` / `t_oceanic_km` | 35 / 7 | crustal thickness by type |
| `rho_continental` / `rho_oceanic` / `rho_mantle` | 2.8 / 2.9 / 3.3 | isostasy densities |
| `min_island_area_km2` | 1000 | below this an island is speckle. **Never express this in cells** — it breaks multi-resolution runs |

### Stage 1 — plates · `plates.PlateParams`

| parameter | default | effect |
|---|---|---|
| `plate_count` | 18 | 12–30. Drives orogen layout, **not** continent shape — crust type comes from noise, independently |
| `min_separation_deg` | 18 | Poisson-disk spacing; without it plate sizes are lottery-distributed |
| `lloyd_iterations` | 3 | higher = more uniform plates, less natural variety |
| `max_speed_mm_per_yr` | 80 | plate speed; Earth runs 10–100 |
| `transform_ratio` | 2.5 | a boundary is transform when tangential motion exceeds this × normal. At 1.0 you get 44% transform against Earth's ~20% |
| `boundary_warp` / `_frequency` | 0.10 / 2.5 | bends plate boundaries. **Set to 0 and mountain ranges render as dead-straight lines** |
| `u_max_m_per_myr` | 1200 | peak orogenic uplift. Sets relief together with `LemParams.K` |
| `craton_floor_m_per_myr` | 20 | uplift on stable interiors |
| `orogen_decay_km` | 200 | how far uplift reaches from a collision boundary |
| `arc_offset_km` / `arc_width_km` | 150 / 100 | volcanic arc position inland of a trench |
| `rift_shoulder_km` | 80 | rift flank width. The graben **subsides**, which is what creates tectonic lake basins |
| `convergent_thickening_km` / `divergent_thinning_km` | 15 / 8 | crustal root and rift thinning |

### Stages 2 & 4 — erosion · `stage2.LemParams`

| parameter | default | effect |
|---|---|---|
| `m` / `n` | 0.45 / 1.0 | **stream-power exponents. Steady-state concavity θ = m/n exactly**, so these set the primary realism target (0.4–0.6). `n ≠ 1` costs a Newton iteration per node |
| `K` | 3.7e-6 | **erodibility — sets relief, not shape.** Peak elevation scales as `U/K`. Resolution-dependent *and* different for coupled runs; see below |
| `D` | 0.05 | hillslope diffusion, m²/yr. Nearly inert at ~10 km cells but not ignorable — it tips drainage divides |
| `G` | 0.0 | sediment deposition (Davy–Lague). 0 = pure detachment-limited |
| `dt_myr` | 0.5 | timestep. The implicit solver is unconditionally stable, so this can be large |
| `fill_every` | 1 | priority-flood cadence. 1 is affordable |

**`K` needs recalibrating in two situations, and both are measured:**

- **Resolution.** Finer grids resolve smaller headwater areas where slope is steepest, so
  the same `K` gives more relief. `K=1e-6` gave 5.8 km peaks at n=128 and 17.6 km at n=785.
- **Coupling.** Stage 4 substitutes discharge for drainage area, a different
  parameterisation in which `K` means something different. Use **3.7e-6 coupled** and
  **~2.5e-6 uncoupled** at n=785.

To retarget peak elevation, scale `K` by the ratio you want to remove: relief ∝ 1/K exactly
for `n=1`.

### Stage 4 — coupling · `coupled.run`

| parameter | default | effect |
|---|---|---|
| `passes` | 3 | climate–erosion alternations. Two or three is enough; change between passes ran 356 → 107 → 55 m |
| `steps_per_pass` | 200 | erosion steps per pass |
| `discharge_floor` | 0.30 | minimum discharge as a fraction of typical. **Sets the coupled/uncoupled peak ratio at `floor^(-m/n)`.** Lower = more relief in deserts and more contrast |

### Stage 3 — climate · `climate.ClimateParams`

| parameter | default | effect |
|---|---|---|
| `t_equator_c` / `t_pole_c` | 27 / −25 | sea-level temperature range |
| `lapse_rate_c_per_km` | 6.5 | standard environmental lapse rate |
| `continental_cooling_c` | 15 | annual-mean cooling deep inside continents, weighted by latitude. The main non-zonal signal in temperature |
| `continentality_c` / `_scale_km` | 6 / 1500 | seasonal amplitude inland, and its decay length |
| `wind_speed` / `meridional_fraction` | 1.0 / 0.3 | only the ratio matters; direction comes from the three-cell bands |
| `meander_deg` / `_frequency` | 7 / 1.8 | Rossby-like waviness. **At 0 the zero-wind latitudes render as ruler-straight rainfall stripes** |
| `orographic_coeff` | 0.004 | rainout per metre of forced ascent. Raising it strengthens rain shadows and dries interiors |
| `land_recycling` | 0.70 | evapotranspiration returned to the air. **In direct tension with `orographic_coeff`** — recycled moisture refills the lee |
| `background_rainout` | 0.012 | fraction lost per advection step |
| `subsidence_floor` | 0.10 | rainout suppression where Hadley air descends |
| `eddy` | 0.07 | isotropic mixing. **Must be > 0** or the zero-wind latitudes become moisture traps |
| `saturation_gain` | 0.06 | how fast warm air holds more water |
| `evaporation` | 1.0 | ocean source strength |
| `iterations` | 400 | fixed-point sweeps |
| `target_mean_precip_mm` | 1000 | final scaling; the model gives a pattern, not absolute depths |

### Stage 5 — glacial · `glacial.GlacialParams`

| parameter | default | effect |
|---|---|---|
| `freeze_c` | −2.0 | ice accumulates below this mean annual temperature |
| `balance_per_degc` | 0.35 | metres of ice per year per degree below freezing |
| `K` / `m` / `n` | 0.02 / 0.30 / 1.0 | abrasion law on ice flux. `m` is below the fluvial value on purpose |
| `slope_baseline_km` | 120 | **driving slope is regional, not local.** With a local slope, overdeepenings are impossible rather than merely rare |
| `min_flux_km3_per_yr` | 0.01 | below this it is snow, not a glacier |
| `steps` / `dt_yr` | 60 / 5000 | 300 kyr of glaciation |
| `max_incision_per_step_m` | 25 | explicit-scheme guard |

### Stage 6 — hydrology · `hydrology.HydrologyParams`

| parameter | default | effect |
|---|---|---|
| `pet_base_mm` / `pet_linear` / `pet_cubic` | 300 / 25 / 0.05 | Thornthwaite-style potential evapotranspiration from temperature |
| `endorheic_margin_m` | 5 | how far below spill a lake must settle to count as terminal |
| `min_lake_area_km2` | 400 | below this a depression is not a lake |
| `channel_threshold_km3_per_yr` | 0.5 | discharge defining a river |
| `width_coeff_m` / `width_exponent` | 90 / 0.5 | `w = c·Q^0.5`, standard hydraulic geometry |

### Stage 8 — amplification · `amplify.AmplifyParams`

| parameter | default | effect |
|---|---|---|
| `max_amplitude_m` | 260 | detail height on the roughest ground |
| `slope_reference` | 0.06 | slope at which detail is full strength |
| `flat_floor` | 0.18 | detail retained on flat ground. **At 0 the sea bed and every plain go glass-smooth** |
| `valley_smoothing_km2` | 60000 | drainage area over which alluvial floors smooth out. **Must be a large-river area** — at 3000 it flattened the whole channel network |
| `ocean_scale` | 0.35 | sub-sea detail, muted by sediment |
| `octaves` / `frequency` / `ridged` | 6 / 180 / 0.35 | fractal character; `ridged` blends toward crests over blobs |
| `neighbours` | 3 | IDW interpolation. **At 1 the raster renders the hexagons** |

> **Do not apply amplification to the simulation mesh.** It perturbs elevations, which
> changes flow receivers, which rewires the drainage network — measured as θ moving
> 0.512 → 0.603. It is an output-side operation; metrics belong on the pre-amplification field.

---

## Validating a world

`terrain.metrics.report` computes the diagnostics, and the same code runs on real DEM
tiles via `terrain.dem.load_window`. Reference values from SRTM (southern Appalachians,
unglaciated):

| metric | SRTM east | typical generated |
|---|---|---|
| slope–area concavity θ | 0.516 | 0.46–0.47 |
| Hack exponent h | 0.543 | 0.546 |
| Horton bifurcation Rb | 5.51 | ~5.1 |

**Measure θ per basin, not planet-wide** — uplift varies across a world, and that variation
lands in a global regression. **Judge worlds by the mesh metrics, not by the exported
raster**: at ~5 km/px against SRTM's 30 m, no raster window is both wide enough for three
decades of drainage area and small enough to avoid equirectangular distortion.

## Known limitations

Recorded with measurements in [terrain-algorithms.md](terrain-algorithms.md):

- **No subtropical desert belt from circulation.** Needs a vertical dimension the
  surface-only climate model does not have.
- **Precipitation is dry-tailed** — land p25 is ~15 mm against Earth's ~250 — which makes
  arid biomes over-represented (~30% of land against ~18%).
- **Biomes remain visibly zonal.** Adding arid classes outside the tropics widened the
  desert belt's latitude wander from 4.4° to 9.9°, but breaking the bands properly needs
  ocean-current heat transport and a seasonal cycle (for monsoons).
- **Endorheic basins are rare** in generated worlds; the logic is unit-tested on
  constructed basins instead.
- **No meanders, dunes, wave-cut notches or cirques** — all sub-grid at ~10 km cells.
