# Mineral distribution: what the geology says, and what this pipeline can place

**Status: the build order is implemented, all seven items.** `world.npz` carries
`field_resource_copper`, `field_resource_iron` and `field_resource_gold`. What follows is the
research they rest on and a record of what was measured, dropped and left to tune. Written
2026-10-05 from a literature search plus a read of the code as it stood at commit `3231ff2`;
[What the pipeline lacks](#what-the-pipeline-lacks) describes that commit, and the build
order marks what has changed since.

**Goal:** have terrain generation identify where mineral deposits could plausibly occur, from
the same processes that already shape the world: plate tectonics, erosion and weathering, and
sedimentation.

**Short answer:** most of it can be a post-process on fields the pipeline already computes.
The output is a *prospectivity* score per cell for each of copper, iron and gold, not a claim
that a deposit exists; see [Decisions](#decisions). Three things cannot come out of the current model at all, because it has one
plate snapshot and no history: old sutures, cratons, and deposits restricted to one period of
Earth history. Iron is the awkward member of that last group.

---

## How deposits are distributed

Economic geology treats a deposit as the outcome of a **mineral system**, not as an isolated
object [1][2]. A system needs, in order:

1. a tectonic trigger that opens a time window,
2. a fertile source of metal and of the fluid or magma that carries it,
3. a pathway, usually a large fault,
4. a trap where the metal drops out,
5. preservation: exhumed enough to be reachable, not so much that it is eroded away.

Groves and co-authors compress the same idea to four terms: geodynamics, fertility,
architecture, preservation [2]. Two consequences matter for a generator.

- **Each deposit type sits in a specific tectonic niche**, mostly near plate boundaries, where
  thermal and mechanical energy are anomalously high [3]. Boundary type and distance therefore
  carry most of the signal.
- **What survives is as important as what forms.** The distribution of deposits through time
  reflects formation and preservation together [4]. Porphyry deposits form in fast-eroding arcs
  and are mostly Cenozoic for that reason [2]; orogenic gold has the best preservation record
  of any gold deposit type [5].

Deposit sizes within a province follow power laws, and mineralising events are short and rare
[1]. A favourable setting is necessary and far from sufficient, which is why the output here
should be a score to sample from and not a deposit map.

One study does almost exactly what this project would do for one deposit type: it runs a global
landscape-evolution model (goSPL) through the Cenozoic and uses the simulated erosion to
estimate which porphyry copper deposits are exhumed and which are preserved [10]. A second
builds lead-zinc prospectivity on a discrete global grid from proxies for each part of the
mineral system [15]. With regional geophysics alone it reached an AUC of 0.787 to 0.870; with
richer proxies and gradient boosting, 0.983 to 0.991, cutting the search area by more than 94%.

---

## Deposit groups and their signals

"Signal" is what this pipeline already computes that could stand in for the control. Field and
function names are current; see [What the pipeline lacks](#what-the-pipeline-lacks) for which
of them are exported.

### Tectonic and magmatic

| Deposit | Where and why | Signal in this pipeline |
|---|---|---|
| Porphyry Cu-Mo-Au, epithermal Au-Ag, skarn | Subduction arcs. Intrusions emplaced at roughly 1 to 7 km [7], typically 2 to 5 km [10]. Crust thicker than 40 km favours giant Cu deposits; thinner crust favours Au-rich ones [7]. Formation coincides with fast exhumation, hundreds to thousands of m/Myr [8] | Convergent boundary with continental crust on the overriding side and oceanic on the other, in the `arc_offset_km` band; `crust_thickness` |
| Orogenic gold | Accretionary and collisional belts alike, on lithosphere-scale faults [2][74]. Best-preserved gold deposit type [5]. See [Gold](#gold) | Any convergent boundary with continental crust on at least one side; transform boundaries and boundary bends as the fault proxy; deeper exhumation |
| Sn-W granites, Li pegmatites | Peraluminous granites in post-collisional settings [11]; Li pegmatites in post-orogenic extension [12]. Both sources are regional (China) | Core of collision belts with thick crust |
| Volcanic-hosted massive sulphide Cu-Zn | Extensional arcs and back-arcs; formation peaks during supercontinent assembly, not breakup [6] | Oceanic divergent boundaries and ocean-ocean convergence. Mostly offshore, so low value on a land map |
| Sediment-hosted stratiform Cu (with Co, Ag) | Failed rifts that became intracratonic basins [42]. See [Copper](#copper) | Divergent boundary in continental crust where uplift is negative, arid, with a lake in the graben |
| Iron oxide copper-gold | Extending continental arcs, and craton margins [47][48]. See [Copper](#copper) | Arc band on weakly convergent segments; craton-margin type has no signal |
| Carbonatite rare earths | Craton and thick-lithosphere margins [53][12] | **None** without cratons |
| Diamonds, Ni-Cu-PGE, many others | Within about 100 km of craton and thick-lithosphere margins [2][53]. Deep faults there are the conduits for the mafic intrusions that host Ni-Cu-PGE [53] | **None.** The model has no cratons, crustal age or plumes |
| Banded iron formation | Restricted to the early Precambrian; does not recur in younger basins [4]. See [Iron](#iron) | **None** for the host rock. The upgrade to ore is a weathering overlay |
| Iron oxide-apatite (Kiruna type) | Back-arc and extending continental-arc settings [66][67]. See [Iron](#iron) | Arc band on weakly convergent segments, at a deeper level than iron oxide copper-gold |

### Sedimentary and basin

| Deposit | Where and why | Signal in this pipeline |
|---|---|---|
| Carbonate-hosted Pb-Zn (Mississippi Valley type) | Platform carbonates inboard of orogenic belts [13], in the forelands and thrust belts of continent-continent collisions [14]. Ore sits on extensional faults produced by foreland flexure [16]. Brines come from low- to mid-latitude carbonate platforms and evaporites [15] | Distance band on the foreland side of collision belts, weighted by warm low latitude |
| Clastic-hosted Pb-Zn | Intracontinental or failed rifts and rifted continental margins [13]; rifted edges of cratonic lithosphere [15]; favoured during supercontinent breakup [6] | Continental divergent boundaries; passive `margin_class` |
| Marine evaporites, potash | Centred on the horse latitudes. Need a hydrographically isolated depression below sea level, which collision sutures and incipient rifts provide [17] | Arid land cells with elevation below zero, near convergent or rift boundaries |
| Lithium brines, borates, soda ash | Continental, not marine-fed [17]. Six shared traits: arid climate, closed basin with a salt lake, nearby igneous or geothermal activity, tectonic subsidence, a lithium source, and time [18] | `endorheic` × arid Köppen × proximity to an arc or rift |
| Coal | Wetness is the essential control: coal formed while a plate sat in the equatorial rainy belt and stopped in the subtropical arid belt [19][21]. Temperature is not essential: major coals formed in cool temperate climates [20] and at up to 75°S, though high-latitude seams are thin [25]. Foreland and intermontane basins beside orogens, and adjacent cratonic basins [19]. Thick, extensive seams sit on the stable craton-side margin of a foreland; the active margin gives thin, shaly, high-ash coal [20]. High clastic input suppresses peat [22]. Thick seams need accommodation to roughly balance peat growth [22][23], though one basin found subsidence rate did not predict coal content [25]. Delta plains, alluvial valleys, lake and shoreline margins; river-dominated deltas give the most continuous seams [24][21] | Precipitation minus evapotranspiration (`hydrology.runoff_mm`); low slope; low elevation; near river mouths and `lake_mask` margins; far side of the foreland band |
| Phosphate | Continental margins under upwelling and high productivity [27]. The Namibian shelf deposit covers 24,700 km² under the Benguela system, at 190 to 350 m water depth [26]. Subtropical to temperate latitudes on western continental margins [29]. Needs low detrital input, which an arid hinterland provides [28][26]. Passive margins and shallow platform seas [31][30]. Winnowing and reworking concentrate it, with the best grades around shallow topographic highs [30] | Offshore wind-driven transport at the coast (from `Climate.wind`); arid coast with low `discharge`; passive `margin_class`; shallow ocean cells |

### Weathering

| Deposit | Where and why | Signal in this pipeline |
|---|---|---|
| Ni laterite | Rainfall above 1,000 mm/yr, coldest-month mean 15 to 27 °C, warmest-month mean 22 to 31 °C [32]. Parent rock is obducted ophiolite [33] | Climate window straight from `temperature_monthly_c` and `precipitation_mm`; parent rock from suture position |
| Bauxite | Continental margins and plateaus [33]. Long stability and minimal erosion matter more than heat: bauxite formed at 57.5°S in a wet, cool climate over more than 0.5 Myr [34]. About 11% of bauxite is karst bauxite, eroded from a laterite and redeposited downstream on carbonate [36] | Wetness, low slope, low uplift. Temperature as a weak term only |
| Lateritic iron and gold | Residual iron ore develops on banded iron formation, plus channel iron deposits in old river channels. Gold enriches within the weathering profile, differently in savanna, rainforest and semi-arid settings [35] | An overlay that upgrades other primary scores, not a source of its own |
| Supergene copper enrichment | Needs more than 120 mm/yr of rain and exhumation of tens to hundreds of m/Myr, slow enough to leach but fast enough to refresh sulphide [8]. In northern Chile the enrichment blankets survived only where later exhumation stayed below about 0.01 km/Myr [9] | Porphyry score × semi-arid climate × low uplift |

### Erosion and transport

| Deposit | Where and why | Signal in this pipeline |
|---|---|---|
| Placer gold (and by extension tin, platinum, diamonds) | Needs 10³ to 10⁶ times concentration. Best where a channel sits near the threshold between detachment-limited and transport-limited flow: more transport capacity strips the bed, less buries the gold in sediment. Fault damage zones help [37]. Coarse gold stays in steep bedrock valleys within the first few km of its source [40]; grain shape changes fastest in the first 5 km [38]. Fine gold travels more than 100 km and reconcentrates at mountain-front gorges and on beaches [39] | Primary-source scores pushed down `flow_receiver`; `discharge` and `slope` for transport capacity. Two classes: proximal coarse, distal fine |
| Heavy-mineral beach sands (Ti, Zr) | Beaches fed by large rivers (general knowledge, not sourced) | Coastal cells near high-`discharge` river mouths; passive `margin_class` |

### Copper

Copper cuts across the groups above, so it is collected here. Three deposit types carry most
of it, and the pipeline can place them to very different degrees.

| Type | Share and setting | Controls | What the pipeline can do |
|---|---|---|---|
| Porphyry | About 60% of world production [8]. Subduction arcs; almost exclusively Phanerozoic [51] | See the tectonic table: arc position, crust thickness [7], exhumation window [8][10], and supergene upgrading in semi-arid, slowly eroding ground [8][9] | **Fully placeable** once boundaries are sub-classified and an exhumation proxy exists |
| Sediment-hosted stratiform | About 23% of production and reserves, plus much cobalt and silver. Occurrences are very common, economic ones rare: supergiants are known in only three basins [41][42] | An oxidised, saline brine leaches copper from rift-filling red beds and drops it on crossing into reduced, carbonaceous grey beds or petroleum [41][43]. Sulphur comes from evaporites [41]. The brine is driven by meteoric recharge from adjacent rift-margin highlands [43]. Ore focuses at basin margins where the red beds thin, on faults, on buried topographic highs and beside salt diapirs [41]. The big basins were failed rifts at low latitude, hydrologically closed, with tens to hundreds of Myr available [42]. 85% of sediment-hosted base metal, and every giant, lies within 200 km of the step between thick and thin lithosphere [44] | **Partly placeable.** The model already builds rifts as uplifted shoulders flanking a subsiding graben, and puts lakes in the graben. Graben margin × arid climate × low latitude × lake present covers the basin-scale controls. The lithosphere-edge control, which is the strongest single predictor, has no signal |
| Iron oxide copper-gold | Individually very large: resources above 100 Mt and up to more than 1,000 Mt, at grades above most porphyries [52] | Three settings: orogenic belts switching from compression to extension, post-orogenic extension, and continental arcs under extension [47]. In the Andes these sit in a Jurassic to Cretaceous arc stretched by slab rollback, on arc-parallel faults, with mafic magmatism [49]; they formed during back-arc transtension where porphyries formed during normal contraction [50]. Deposits localise on fault bends and intersections [52]. The largest are within about 100 km of craton margins and are mainly Precambrian [48] | **Andean type placeable** as arc segments with weak or oblique convergence. **Giant type not placeable** without cratons |

Minor sources: volcanic-hosted massive sulphide (tectonic table), and magmatic Ni-Cu-PGE at
craton margins [53].

Two things follow for this project.

- **Cratons stop being optional if copper matters.** The lithosphere edge controls
  sediment-hosted copper [44], the giant iron oxide copper-gold deposits [48], Ni-Cu-PGE,
  carbonatites and iron formation [53]. One paper puts most of these within 100 km of Archean
  craton margins, widening to 200 km for sediment-hosted Pb-Zn-Cu, and estimates that this
  confines the search to 10 to 35% of continental crust [53].
- **Exceptions exist and the score should not be a hard gate.** White Pine is a giant
  sediment-hosted deposit with no evaporites, fed by copper-rich basaltic detritus instead
  [45]. In Iran the same deposit type occurs in post-collisional extension and back-arc basins
  as well as rifts [46].

### Iron

Iron is the hardest major commodity for this model, because its dominant source needs invented
geology twice over. It is also the one where the *upgrading* step, which is what makes ore out
of iron-rich rock, is something the pipeline models well.

| Type | Importance | Controls | What the pipeline can do |
|---|---|---|---|
| High-grade ore hosted in banded iron formation | The bulk of world production and reserves [55]; the largest and highest-grade districts [54] | The host rock is Archean to Proterozoic [54] and does not recur [4]. It becomes ore by losing its silica: along km-scale normal and strike-slip faults, to deep-sourced fluids, and then to weathering that reaches 200 m down [54]. In most districts a geologically recent deep-weathering overprint is what turned moderate bodies into giants [56]. The Hamersley endowment is put down to three things coinciding: an iron-rich source, sustained gentle uplift and a seasonally humid climate [58] | **Host not placeable.** It needs an invented craton and, inside it, an invented iron-formation province. **Upgrade placeable** as an overlay: weathering score × fault proximity × low, steady uplift |
| Channel iron deposits | Large in the Hamersley, rare elsewhere [58] | Iron-rich detritus from a lateritised iron-formation surface, washed into incised valleys and cemented there [58][35] | Downstream of an iron-formation province along the flow tree, in low-gradient valleys. Same machinery as placers |
| Iron oxide-apatite (Kiruna type) | A key iron source for modern industry [69]; Kiruna supplies a significant share of European iron ore [66] | Emplaced in intracontinental back-arc basins under extension, later inverted [66][67]. Tied to andesitic and dioritic subvolcanic intrusions [70]. Sits on deep crustal faults, and grades upward and outward into iron oxide copper-gold [68]. The largest examples are Proterozoic, 1.1 to 1.8 Ga [71] | **Placeable.** The same score as Andean-type iron oxide copper-gold, at a deeper exhumation level. This is the one major iron source the tectonics can derive |
| Ooidal ironstone | Historically important ores [65]; common through the Phanerozoic, with no clear modern analogue [61] | Low-energy shallow sea: tidal flats, embayments, broad coastal plains [61]. Needs a break in clastic supply, a continuing iron supply and physical reworking; a transgression flooding lateritic soils provides all three [59]. Found in foreland basins, cratonic margins and intracratonic basins during high sea level [60]. The iron source is disputed: humid tropical weathering on land [61][59], or upwelling of iron-rich deep water onto favourably oriented margins [62][63][64] | **Partly placeable.** Quiet shallow coast with low `discharge`, next to laterite-scored lowland or an upwelling coast. Shares phosphate's problem: no sea-level history, so no emerged shelf |
| Iron skarn | Regionally important | Hot saline fluids from intrusions reacting with limestone and volcanic rocks [72] | Arc band only. The model has no rock types, so the limestone control has no signal |
| Magmatic Fe-Ti-V; bog iron | General knowledge, not sourced. Ilmenite in anorthosite is tied to craton margins [53] | Layered intrusions and plumes; cool, wet lowland mires | None for the first. Bog iron could reuse the coal wetness term |

Three things follow.

- **The invented craton needs a second invented layer.** A craton edge places copper and
  Pb-Zn; iron ore needs an iron-formation province *inside* the craton. That is one more noise
  threshold, and it should be labelled as invented.
- **The upgrade step is honest work.** Deep weathering under a seasonally humid climate, on
  faulted ground with slow steady uplift, is the same overlay as bauxite and laterite. The
  supergene model is contested for the deepest deposits [57], so treat it as an upgrade to a
  host that already exists, not as a source.
- **Ironstone and phosphate are the same build.** Both sit on quiet shallow margins, both have
  an upwelling model, and one ironstone studied is itself phosphatic [63]. If the upwelling
  score is built for either, the other costs little.

### Gold

Gold is the commodity this model fits best. Its main sources all hang off three things the
pipeline has or can cheaply get: convergent boundaries, how deeply the ground has been eroded,
and the drainage network.

| Type | Importance | Controls | What the pipeline can do |
|---|---|---|---|
| Orogenic gold | The main source of gold for humanity [75]. About 23,000 to 25,000 t in Precambrian shields and 22,000 t in younger Cordilleran-style belts [73] | Subduction-related accretionary to collisional belts, late in the orogen, as compression turns to transpression [74]. Beside first-order crustal faults hundreds of km long, with ore in second- and third-order structures at jogs and changes of strike [73]; world-class deposits sit at bends of 10 to 25 degrees [74]. Forms across 2 to 20 km depth, mostly in greenschist-facies rock [73], and deeper than 4 km [75]. Major provinces are often marginal to cratons [76] | **Placeable.** Band along convergent boundaries, raised at bends in the boundary trace, weighted to deeper exhumation |
| Placer gold | 15,500 t from the younger belts, against 22,000 t of lode gold in the same belts [73] | See the erosion table: downstream of lode sources, near the transport threshold [37] | **Placeable** from the flow tree. Not a footnote: placers are about 40% of the gold in those belts |
| Epithermal, high-sulfidation | The large Andean examples together hold more than 130 Moz [78] | Shallow, and so short-lived: nearly all are middle Miocene or younger, and older ones survive only in the Atacama Desert [78]. Contractional arc segments, alongside porphyries [78][80]. In the Andes the ore sits 200 to 500 m below high, low-relief surfaces, at the heads of valleys cutting into them; incision itself helped deposit the ore [78] | **Placeable**, and partly from terrain: arc band × very low exhumation × arid × valley heads incising a high, low-relief surface |
| Epithermal, low- and intermediate-sulfidation | Large silver endowment in the extensional subtype [77] | About 0.3 to 1 km depth or a little more; most are Cenozoic [77]. Extensional arcs, back-arc rifts with bimodal volcanism, and post-subduction settings [79][80]; older and lower-lying than the high-sulfidation type [78] | **Placeable** as arc and back-arc segments with weak or oblique convergence, and continental rifts |
| Porphyry gold | Co-product of copper | Thinner crust, below 40 km, favours gold-rich porphyries [7] | **Placeable** with the porphyry score, weighted to thinner `crust_thickness` |
| Carlin type | About 5% of world annual production from Nevada alone [81]; the second-largest gold concentration on Earth [82] | Four shared regional controls: the rifted margin of a craton, the carbonate slope of a passive margin, the lower plate of a regional thrust, and a brief window when contraction switched to extension [81]. Sits on deep structures inherited from the rifting [83][84]. Forms shallower than 2 to 3 km [81]. Known in very few provinces [83][84] | **Not placeable.** Needs a craton margin, rock types and an older thrust event, none of which exist |
| Witwatersrand | The largest gold concentration on Earth [82] | Archean basin preserved over buoyant cratonic lithosphere [5]. Its placer origin is disputed [86] | **Not placeable** |

Three things follow.

- **The exhumation proxy does the most work here.** The types sort by depth: epithermal within
  about 1 km of the surface [77], Carlin within 2 to 3 km [81], porphyry at 1 to 7 km [7],
  orogenic at 2 to 20 km [73]. Along one eroding belt that gives a sequence from epithermal
  where little has been removed to orogenic where a lot has.
- **Two cheap additions to the boundary work pay for themselves.** Bends in the boundary
  trace are computable, since the plate boundaries are already warped, and mark where orogenic
  gold concentrates [74]. Keeping the tangential component of plate motion separates
  transpression (orogenic gold [74]) from extension (low-sulfidation epithermal [80], iron
  oxide copper-gold [50]) from head-on contraction (porphyry, high-sulfidation [78]).
- **An earlier row was too narrow.** Orogenic gold forms on accretionary margins as well as in
  continent-continent collisions [74], so it belongs on arc margins too. The tectonic table
  now says so.

---

## What the pipeline lacks

Found by reading the code, not by running it.

- **The tectonic fields never reach the archive.** `boundary_type`, `boundary_stress`,
  `boundary_dist_km`, `crust_thickness` and `uplift` are computed in Stage 1 and
  `export.export_world` takes none of them. `is_continental` is a local variable in
  `stage1.generate_crust_tectonic` and is not kept on the result at all. `margin_class` had
  the same problem until it was exported.
- **Arc and collision are not distinguished.** `plates.derive_uplift` applies the collision
  term and the volcanic-arc term to the same mask (`conv & is_continental`). For terrain that
  is a simplification; for minerals it erases the difference between an Andes and a Himalaya,
  which is the single most useful split in the tables above.
- **Boundary obliquity is discarded.** `plates._classify_boundaries` keeps only the normal
  component of relative motion in `boundary_stress`; the tangential component is used to label
  transforms and then dropped. Extending or oblique arc segments, where iron oxide copper-gold
  forms instead of porphyry [50], can only be approximated by weak convergence. The same
  component separates the transpression that triggers orogenic gold [74] from head-on
  contraction, so it is worth keeping.
- **There is no deposition.** `LemParams.G` is 0, so Stage 2 is purely detachment-limited and
  `accumulate_sediment` never runs. Sedimentary basins can only be inferred from low slope,
  lakes and subsidence. This also means the placer threshold in [37] cannot be read from the
  model and has to be estimated.
- **Subsidence exists only in rift grabens.** Foreland flexure, which controls both coal and
  carbonate-hosted Pb-Zn, is not modelled. The foreland can only be a distance band.
- **There is no plate history.** One snapshot of plates means no old sutures, no cratons, no
  age-restricted deposits. It also means coal, phosphate, evaporites and laterites can only be
  placed where they would be forming *today*. On Earth they record the climate belt the plate
  sat in when they formed [19][21][29].
- **The climate fields phosphate needs are not exported.** `Climate.wind` and
  `Climate.sst_anomaly_c` exist in `climate.py` and stop there. Whether `sst_anomaly_c` shows
  cold water on western margins has not been checked.
- **Two known climate limitations bite here.** The README records that endorheic basins are
  rare and that there is no circulation-driven subtropical desert belt. Those are the main
  controls on evaporites and brines, so that group will be sparse.
- **Relief is small.** Land elevation has a standard deviation of 104 m. Depth windows such as
  "1 to 5 km of exhumation" must be applied as relative ranks, not absolute metres.

### Why there are no cratons

A craton is old continental lithosphere, typically more than 2 billion years old, with a
thick, cold, buoyant mantle root that has kept it stable ever since. Two properties define it:
age, and the thickness of the lithosphere beneath the crust. The model has neither, because a
craton is a product of history and the model has none.

- **One plate snapshot.** Stage 1 lays out the plates once, gives each an Euler pole and
  classifies the boundaries. Nothing ever collides, sutures or breaks up, so no crust is older
  than any other. There is no age field.
- **Crust only, no mantle.** `stage1.py` tracks `crust_thickness` and `crust_density` for
  isostasy. Nothing represents the lithospheric root.
- **Continents come from noise.** Continental versus oceanic crust is a thresholded noise
  field, deliberately independent of the plate layout. It says where continents are, not which
  parts of them are ancient.
- **"Craton" in the code means an uplift floor.** `PlateParams.craton_floor_m_per_myr` gives
  rising ground a minimum uplift of 2 m/Myr, which is what ground beyond the reach of any
  boundary ends up with. A comment in `plates.py` measures 70.9% to 76.3% of land at that
  floor (n=256, seed 1).

So the model has craton-like *behaviour*, a quiet and flat interior, but as one
undifferentiated region. Nothing separates old, thick-rooted crust from younger crust that
merely happens to be far from a boundary today. Terrain never needed the distinction, since
both look the same at the surface. Minerals do, because deposits follow the craton *edge*
[44][53], and the model has no such line.

The obvious derived proxy does not help. "Continental crust beyond the uplift reach of any
boundary" has an edge that is just a contour parallel to today's plate boundaries, so it adds
nothing that boundary distance does not already give. Real craton margins are old boundaries,
often deep inside plates and unrelated to the current ones. A noise field restricted to the
quiet interior gives edges with that independence, which is why item 5 of the build order
invents them. Deriving them properly would mean simulating plate history.

---

## Proposed build order

Scoped to copper, iron and gold, per [Decisions](#decisions). Ranked by value for cost; effort
is S, M or L. Items 1 to 3 are prerequisites, item 4 delivers most of the copper and gold, and
item 5 delivers most of the iron. The "signal" columns in the tables above were written before
anything was measured; where an item below reports a measurement, the item is right.

1. **Export the Stage 1 tectonic fields (S). Done.** `world.npz` now carries `plate_id`,
   `boundary_type`, `boundary_stress`, `boundary_speed`, `boundary_dist_km`,
   `nearest_boundary`, `crust_thickness`, `uplift` and `is_continental`; the README field
   table documents each. `boundary_speed` is new: the total relative plate speed on a
   boundary cell, which is exact where `boundary_stress` is an estimate.
2. **Boundary obliquity (S). Done, and narrower than planned.** `world.npz` now carries
   `boundary_slip`, the part of the relative plate speed that runs along the boundary;
   `slip / speed` is obliquity. It estimates the boundary normal itself. The first plan was to
   difference speed and `boundary_stress`, and measured, that does not work: on two
   counter-rotating hemispheres at n=48 a purely head-on boundary read slip/speed 0.68 where
   the answer is 0. The new field reads a median of 0.018 there and 1.000 on a purely
   strike-slip boundary. `boundary_stress` is untouched, since uplift is calibrated on it.
   On built plates half of all convergent boundary cells have slip/speed above 0.7, so
   obliquity is a strong signal in these worlds, not a correction.

   Two parts of the original item were dropped after measurement:

   - **Splitting arc from collision by crust type on each side.** Measured at n=128 over
     three seeds, only about 5% of convergent boundary cells have continental crust on one
     side and oceanic on the other; 42 to 47% are continental on both sides and 47 to 53%
     oceanic on both. Crust type comes from noise that is independent of the plates, so a
     boundary follows a continental edge only by chance. The pairing this document calls the
     most useful split barely occurs. See
     [Decided: arc and collision share belts](#decided-arc-and-collision-share-belts).
   - **Bends in the boundary trace.** The control in [74] is a district-scale jog of 10 to 25
     degrees, which is a few cells at this world's scale. Boundaries here are smooth warped
     arcs plus cell-scale jaggedness, and the jaggedness is discretisation, not structure.
     There is nothing at the right scale to measure.
3. **Exhumation proxy (S). Done.** Uplift over its ceiling, as a 0 to 1 rank, split into
   shallow, mid and deep windows that sum to one. Run time is the same everywhere in a world,
   so uplift × time ranks cells exactly as uplift does. Two things to know about it. It is
   built from the same inputs as distance and boundary stress, so it reshapes a belt across
   strike and adds no independent information; real cumulative erosion would, by telling
   valleys from ridges. And most land is outside it: measured at n=128 over three seeds, 86
   to 87% of land sits below a rank of 0.1.
4. **Score and export the derived deposit types (M). Done.** `terrain/minerals.py`, run as a
   post-process, scores seven types and combines them into `field_resource_copper`,
   `field_resource_iron` and `field_resource_gold`. An arc band is split by obliquity and by
   depth of erosion into six types, with no parameters beyond the existing plate lengths:

   | Erosion | Head-on arc | Oblique arc |
   |---|---|---|
   | Shallow | High-sulfidation epithermal | Low-sulfidation epithermal |
   | Mid | Porphyry | Iron oxide copper-gold |
   | Deep | (stripped) | Kiruna-type iron |

   Orogenic gold takes the belt axis where erosion is deep and the plates both close and
   slip. Sediment-hosted copper takes continental rifts. Measured at n=128 over three seeds:
   each type scores above 0.1 on 1 to 5% of land; copper peaks at 0.26 to 0.28, gold at 0.45
   and iron at 0.10 to 0.12. Re-scoring a saved world from its archive alone reproduces the shipped
   fields exactly.

   Left out on purpose, to be picked up later:

   - Sediment-hosted copper has no climate terms yet, so every continental rift carries it.
     The arid, low-latitude and lake controls [42] belong with item 6.
   - Crust thickness is not used. It would shift porphyries between copper-rich and
     gold-rich [7], and it is exported if that is wanted.
   - Fields do not reach 1. Copper tops out near 0.28 because the mid window peaks at one
     half and porphyry carries 0.60 of copper. That costs resolution in a byte, not
     correctness, since the sim samples in proportion.
5. **Invented cratons and iron-formation provinces (S). Done.** `minerals.invent_cratons`
   draws both from noise on sub-seeds of the world seed: cratons take 30% of continental
   crust by area, and iron formation takes 15% of craton. Iron formation scores wherever the
   ground is still quiet, and a rift now earns most of its copper only within 200 km (Earth)
   of a craton margin [44], keeping 15% elsewhere. The manifest says which fields rest on
   this. Measured at n=128 over three seeds, the iron field peaks at 0.70 and iron formation
   carries 96 to 98% of it.

   Two things measurement showed:

   - **The giant iron oxide copper-gold type was built and removed.** Those bodies sit within
     about 100 km of craton margins [48], so a ring was scored along every margin. At a weight
     of only 0.07 that ring carried 67 to 72% of all the copper and about 30% of the gold. The
     sim samples in proportion to score times area, so **a weight is not a share**: the belt
     types multiply three factors that each cut the ground down, and a ring multiplies none.
     Without it copper splits 64 to 84% porphyry, 7 to 25% sediment-hosted and 8 to 11% iron
     oxide copper-gold, against weights of 60, 23 and 17. A test now requires the type the
     weights put first to come first by area as well.
   - **Iron is nearly the same on every world.** The two shares are fixed fractions of area,
     so iron formation covers 4.3 to 4.5% of land regardless of seed, while copper and gold
     move with the plates. Making it vary would mean drawing the shares themselves from the
     seed.
6. **Climate overlay (S). Done.** Three types now read the shipped rainfall and temperature,
   and each only scales a type the plates or the cratons already put there:

   - `supergene_copper` is a porphyry with rain enough to leach it and not enough to erode
     it: full from 120 mm/yr [8], fading out between 500 and 1000.
   - `enriched_iron` is iron formation weathered to ore where it is warm and wet [58]: above
     1000 mm/yr and 20 °C mean, fading out below 500 mm and 15 °C.
   - `sediment_copper` keeps only a quarter of its score unless the rift is dry enough for
     evaporites [42]: full below 250 mm/yr, the quarter above 750.

   Only the 120 mm is sourced; the other thresholds are a judgement. Each upgrade splits a
   weight with its host: porphyry's 0.60 became 0.40 plus 0.20, iron formation's 0.70 became
   0.40 plus 0.30. Measured on three n=128 worlds run through the whole pipeline, supergene
   enrichment reaches 28 to 46% of porphyry ground and iron enrichment 30 to 43% of
   iron-formation ground.

   Three things to know:

   - **Rift copper is keyed to dryness alone,** not to warmth, latitude or lakes, which the
     research also names. This model has almost no hot dry land (4% at n=128, having no
     subtropical desert belt) and few lakes, so asking for either would remove the type for
     a known defect of the climate model and not for a geological reason.
   - **Rift copper came out small.** It carries 4 to 9% of the area-weighted copper against
     a weight of 0.23, and the climate is the main reason: on the same three worlds 38 to 70%
     of rift ground gets more than 750 mm/yr and only 9 to 18% less than 250. The craton term
     costs less, with 39 to 48% of rift ground inside its reach. Iron oxide copper-gold now
     outweighs rift copper. This is the first thing to look at when the weights are tuned.
   - **"Seasonally" humid is not tested.** The enrichment term reads annual rain and mean
     temperature. The monthly fields are exported if a dry season should be required.
7. **Flow-tree routing (M). Done.** `placer_gold` is orogenic gold, and `channel_iron` is
   enriched iron formation, carried down `flow_receiver` and dropped where the river
   slackens. Each cell passes its load to its receiver and loses a little per cell, fading
   over 100 km on Earth [39], so what arrives is a mean of the sources upstream that favours
   the near ones. The drop is how much gentler a reach is than the river above it, and only
   river cells score. Proximal placers fall inside the lode's own cell at this scale.

   This had to be judged on a production world, and was: n=785, seed 42, generated end to
   end in 2,408 s. The reach is a world-scale length and a river is a cell-scale thing, so
   on an n=128 fixture the reach is 1.7 cells against 10.2 at production, and the routed
   types all but vanish there (placer gold 0.0 to 0.2% of the gold). At n=785:

   | | cells above 0.05 | above 0.2 | peak | share of its resource by area |
   |---|---|---|---|---|
   | `placer_gold` | 10,315 | 1,335 | 0.67 | 1.2% of gold, at a weight of 0.10 |
   | `channel_iron` | 6,756 | 2,901 | 0.90 | 0.6% of iron, at a weight of 0.15 |

   Three things to know:

   - **The transport threshold of [37] is not what was built.** That needs sediment supply
     against capacity, and the model is detachment-limited with no sediment at all. A reach
     gentler than the river above it is the stand-in.
   - **Placers are real features and a small share.** Thousands of river reaches score, and
     together they carry 1.2% of the gold, where placers are about 40% of the gold in real
     belts of this kind [73]. A river reach is little ground, and the sim samples by area.
     Raising the weight is the lever, and it is a judgement about how rich a placer reach is
     beside the lode, not something the sources settle.
   - **Rivers here are not always lines.** In places the shipped river field forms patches
     a kilometre or two across, and placers inherit them.

**What the production world showed about the whole thing.** Scoring a world takes 6.7 s at
n=785 (4.0 s for the cratons, 2.7 s for the scores) and re-scoring from the archive reproduces
the shipped fields exactly. By area copper is 70% porphyry, 9% supergene, 8% sediment-hosted
and 13% iron oxide copper-gold; gold is 54% orogenic, 16% each epithermal type, 11% porphyry,
2% iron oxide copper-gold and 1% placer; iron is 79% iron formation, 17% enriched, 3% Kiruna
and 1% channel. The fields peak at 0.27, 0.48 and 0.81.

**The belts are blocky, and that is inherited.** Close up at production size, a gold belt is
broken into hard-edged diagonal blocks a kilometre or two across. The uplift field has the
same blocks and so do the mountains. `plates._classify_boundaries` labels a boundary cell by
the direction to one neighbour, so a boundary that is almost pure strike-slip (obliquity 0.85
to 1.0 in the window examined) comes out as alternating "convergent" and "transform"
segments, and `derive_uplift` gives the first full-strength orogens and the second none.
World-wide, on land within 1 km of a boundary, uplift has a median of 45 m/Myr in the
first kind of segment and 2 in the second. The mineral scores read uplift as depth of erosion, so they
follow the terrain there, which is the right thing for them to do. The cause is upstream of
this work and fixing it would move every orogen in the world.

**Dropped by the commodity decision:** coal, bauxite, nickel laterite, evaporites and brines,
lead-zinc, phosphate, ironstone, tin-tungsten, lithium and beach sands. The research on them
stays above; none is scheduled. `Climate.wind` no longer needs exporting.

**Leave alone:** turning on deposition (`G > 0`) and simulating plate history. Both are L,
both would disturb calibrated stages, and neither buys much mineral realism that the proxies
above do not.

---

## Decisions

### Decided: the generator exports fields, the sim places resources

The sim that consumes the terrain decides where resources actually go, so the generator stops
at fields and does not sample deposits. The export is three fields, one per resource:

- **`field_resource_copper`, `field_resource_iron`, `field_resource_gold`**, indexed by cell
  like every other field in `world.npz`, stored as uint8 (0 to 255). At n=785 that is about
  6 MB each before compression.
- **How each is built.** The generator scores the deposit types listed below internally, then
  combines them per resource as a weighted sum. The weights are parameters. Copper has
  sourced starting values (porphyry about 60% [8], sediment-hosted about 23% [41]); gold and
  iron have only orderings from the sources (orogenic gold first [75], iron formation first
  [55]), so their weights are a judgement to tune.
- **What the manifest records.** The weights used, under `params` with every other
  parameter, and a note that the iron field and part of the copper field rest on invented
  cratons.
- **How the sim uses it.** Multiply the resource's field by cell area and sample cells in
  proportion, with its own seed.

Keying by deposit type was considered and set aside. It would have kept distinctions such as
placer gold at the surface against orogenic gold at depth, at the cost of ten fields and a
lookup table on the sim side.

### Decided: distances shrink with the world

This world is 1/98.125 of Earth's radius with 100 m cells. Every horizontal distance in this
document is an Earth value and is divided by 98.125 before use, the same way
`PlateParams.for_radius` treats the uplift geometry. In the README's terms they are
world-scale lengths, not cell-scale ones, and must be rescaled again on a fixture at a
different `n`.

What that does to the numbers quoted above:

| Earth distance | Used for | On this world |
|---|---|---|
| 200 km | Sediment-hosted metals from the lithosphere edge [44] | 2.04 km, about 20 cells |
| 100 km | Most deposit types from a craton margin [2][53]; fine placer gold transport [39] | 1.02 km, about 10 cells |
| 5 km | Proximal, coarse placer gold [38] | 51 m, inside the source cell |
| 1 to 2 km | Footprint of a single large deposit | 10 to 20 m, far below one cell |

Two consequences. A single deposit is always smaller than a cell, so a field marks a belt or
a district and the sim places point resources inside it. And proximal placers coincide with
their lode source at this resolution: only the distal, fine-gold class travels far enough
down the flow tree to appear as its own pattern.

Vertical quantities are not covered by this. Depth windows stay relative ranks, as noted
under [What the pipeline lacks](#what-the-pipeline-lacks).

### Decided: copper, iron and gold

These are the commodities that matter. They are built from ten internal deposit-type scores,
which are not exported:

| Score | Yields | Basis |
|---|---|---|
| `porphyry` | Copper, gold | Derived; weathering upgrade for copper |
| `sediment_copper` | Copper | Rift and climate derived; craton-edge term invented |
| `iocg` | Copper, gold | Arc type derived; craton-margin type invented |
| `orogenic_gold` | Gold | Derived |
| `epithermal_hs` | Gold | Derived |
| `epithermal_ls` | Gold | Derived |
| `placer_gold` | Gold | Derived, from the gold fields above |
| `kiruna_iron` | Iron | Derived |
| `bif_iron` | Iron | Host invented; upgrade to ore derived |
| `channel_iron` | Iron | Source invented; routing derived |

Gold and most copper come out of the model. Iron mostly does not: the only derived iron is
the Kiruna type, a small share of real supply, so the two iron-formation scores carry it.

### Decided: cratons are invented

Item 5 of the build order goes ahead. Iron cannot be common without it, and it supplies the
craton terms for copper. See [Why there are no cratons](#why-there-are-no-cratons) for why
they cannot be derived.

### Decided: a resource need not exist on every world

Scores are absolute, not rank-normalised per world. A world with no continental rift has no
sediment-hosted copper, and a world whose noise puts no iron-formation province on land has
little iron. Worlds therefore differ in what they are rich in, and the sim has to tolerate a
field that is zero everywhere.

### Decided: arc and collision share belts

The model has almost no ocean-against-continent convergent boundaries (build order, item 2),
so arc cannot be split from collision by crust pairing. The scores follow the model's own
convention instead: `plates.derive_uplift` already gives every convergent boundary in
continental crust both a collision orogen and a volcanic arc offset inland. Arc deposits and
collision deposits therefore share belts and separate by distance from the boundary, depth of
erosion and obliquity. Copper and gold end up in the same mountain belts, a little offset.

The alternative was to split by distance to oceanic crust, reading a convergent belt near a
continental edge as an arc and one deep inside a continent as a collision. It is invented,
but it would give belts different characters, some copper-rich and some gold-rich. It needs
no new export, since that distance is computable from `is_continental`, so it can be layered
on later if the maps show belts that are too alike.

### Open

Nothing. The weights that combine deposit types into each resource field are the main thing
left to tune, and that is done against generated worlds, not decided in advance.

---

## Evidence quality

- Sources are the abstracts returned by a Consensus literature search. Full texts were not
  read. Thresholds quoted here should be re-checked against the paper before they are
  hard-coded.
- Two rows are marked "general knowledge, not sourced" and have no citation: heavy-mineral
  beach sands, and magmatic Fe-Ti-V with bog iron.
- The tectonic setting of iron oxide copper-gold deposits is disputed in the literature
  itself [52]; the three-setting scheme used here is one review's classification [47].
- Two iron questions are open in the sources. Whether weathering alone can turn iron
  formation into high-grade ore is contested [57] against [54][56]. Whether ironstone iron
  comes from the land or from upwelled deep water is contested [61] against [62].
- Two gold questions are open as well. The source of the gold in Carlin-type deposits is
  argued as magmatic [82] and as sedimentary [85]. Whether the Witwatersrand is a placer at
  all is challenged [86]. Neither changes the placement, since the model can place neither.
- Several citations are single-region studies used as examples of a general pattern: [11] and
  [12] (China), [20] (Karoo), [26] (Namibia), [46] (Iran), [49] and [50] (Andes). [25] is a conference abstract with one citation
  and is the only source for the claim that subsidence rate did not matter.
- Uplift rates and exhumation depths in the sources are Earth values. None has been tested
  against this world's compressed relief.

---

## References

Framework and tectonic setting

1. Hagemann et al. (2016). [Mineral system analysis: Quo vadis](https://consensus.app/papers/details/659a1795063159ebbc49ccd1cb174088/?utm_source=claude_desktop). *Ore Geology Reviews.*
2. Groves et al. (2021). [Mineral systems: Their advantages in terms of developing holistic genetic models and for target generation in global mineral exploration](https://consensus.app/papers/details/25c84527c57454b9b9b21ddbcd6591a3/?utm_source=claude_desktop). *Geosystems and Geoenvironment.*
3. Groves et al. (2007). [Geodynamic settings of mineral deposit systems](https://consensus.app/papers/details/0d9a4475ec7a5f3f9996a099573ead62/?utm_source=claude_desktop). *Journal of the Geological Society.*
4. Groves et al. (2005). [Controls on the heterogeneous distribution of mineral deposits through time](https://consensus.app/papers/details/fdda0f74b8825d70a070e7444654fa7b/?utm_source=claude_desktop). *Geological Society, London, Special Publications.*
5. Groves et al. (2005). [100th Anniversary Special Paper: Secular Changes in Global Tectonic Processes and Their Influence on the Temporal Distribution of Gold-Bearing Mineral Deposits](https://consensus.app/papers/details/e12a9c9a529957719f7d3808013bcae2/?utm_source=claude_desktop). *Economic Geology.*
6. Huston et al. (2010). [The Geology and Metallogeny of Volcanic-Hosted Massive Sulfide Deposits: Variations through Geologic Time and with Tectonic Setting](https://consensus.app/papers/details/24b782ba00ca59bfb07209ba0036612d/?utm_source=claude_desktop). *Economic Geology.*
7. Park et al. (2021). [Crustal magmatic controls on the formation of porphyry copper deposits](https://consensus.app/papers/details/1a23d005aa6952a7ab4ea7f005a6442e/?utm_source=claude_desktop). *Nature Reviews Earth & Environment.*
8. Evenstar et al. (2024). [Global constraints on exhumation rates during porphyry copper formation and supergene enrichment: applications to exploration as illustrated from the Central Andes](https://consensus.app/papers/details/dcb07af65dc85e388b0336a36c9b2785/?utm_source=claude_desktop). *Mineralium Deposita.*
9. Dahlström et al. (2022). [Pluton Exhumation in the Precordillera of Northern Chile (17.8°–24.2°S): Implications for the Formation, Enrichment, and Preservation of Porphyry Copper Deposits](https://consensus.app/papers/details/3513b6e5cf2157bb80f6f2254770896a/?utm_source=claude_desktop). *Economic Geology.*
10. Hadler Boggiani et al. (2025). [The roles of surface processes in porphyry copper deposit preservation](https://consensus.app/papers/details/5024539b2289563b87121a5eedf1ec6f/?utm_source=claude_desktop). *Earth Surface Dynamics.*
11. Mao et al. (2013). [Major types and time–space distribution of Mesozoic ore deposits in South China and their geodynamic settings](https://consensus.app/papers/details/cb7bef7b53215893a3db096b5cab4006/?utm_source=claude_desktop). *Mineralium Deposita.*
12. Chen et al. (2024). [Phanerozoic Tectonic Evolution, Metallogenesis and Formation of Mineral Systems in China](https://consensus.app/papers/details/149a54ad5a8052a9ab870542d1fcc6fc/?utm_source=claude_desktop). *Acta Geologica Sinica (English Edition).*

Sedimentary and basin

13. Leach et al. (2005). [Sediment-hosted lead-zinc deposits: A global perspective](https://consensus.app/papers/details/fcdb0296bc8f5e3fbcfed7a55b91e53a/?utm_source=claude_desktop). *Economic Geology.*
14. Song et al. (2019). [Sediment-hosted Pb–Zn deposits in the Tethyan domain from China to Iran: Characteristics, tectonic setting, and ore controls](https://consensus.app/papers/details/3f921573684154a6930d3795ef209bbe/?utm_source=claude_desktop). *Gondwana Research.*
15. Lawley et al. (2021). [Data–driven prospectivity modelling of sediment–hosted Zn–Pb mineral systems and their critical raw materials](https://consensus.app/papers/details/4e5079c2b6ab5dafa63250b3cb62be0f/?utm_source=claude_desktop). *Ore Geology Reviews.*
16. Song et al. (2023). [Mississippi Valley-type Zn–Pb deposits in orogenic thrust belts: ore formation in response to synorogenic crustal transpression or extension](https://consensus.app/papers/details/e8e802da23ce5e098a1b7a6e4b69c559/?utm_source=claude_desktop). *Mineralium Deposita.*
17. Warren (2010). [Evaporites through time: Tectonic, climatic and eustatic controls in marine and nonmarine deposits](https://consensus.app/papers/details/34be405033e6591da8cf59d5b6c46dc9/?utm_source=claude_desktop). *Earth-Science Reviews.*
18. Munk et al. (2016). [Lithium Brines: A Global Perspective](https://consensus.app/papers/details/84d82212af81530184a5c0b6e40d7adf/?utm_source=claude_desktop).
19. Calder et al. (1994). [The Euramerican Coal Province: controls on Late Paleozoic peat accumulation](https://consensus.app/papers/details/c4327d0c2be3596dba2b965d2476f5da/?utm_source=claude_desktop). *Palaeogeography, Palaeoclimatology, Palaeoecology.*
20. Cadle et al. (1993). [The Karoo Basin of South Africa: type basin for the coal-bearing deposits of southern Africa](https://consensus.app/papers/details/7cc529a2c21b5e458dd483a56a17a290/?utm_source=claude_desktop). *International Journal of Coal Geology.*
21. Liu (1990). [Permo-Carboniferous paleogeography and coal accumulation and their tectonic control in the North and South China continental plates](https://consensus.app/papers/details/0542c293e27c5f49bcf649a7950be4bb/?utm_source=claude_desktop). *International Journal of Coal Geology.*
22. Opluštil et al. (2013). [Climatic vs. tectonic controls on peat accretion in non-marine setting; an example from the Žacléř Formation (Yeadonian–Bolsovian) in the Intra-Sudetic Basin (Czech Republic)](https://consensus.app/papers/details/ac029fe1781f53059b94a06ff2a34f07/?utm_source=claude_desktop). *International Journal of Coal Geology.*
23. Su et al. (2026). [Sequence Stratigraphic Framework and Coal Accumulation of the Lopingian Longtan Formation in the Sichuan Basin, South China](https://consensus.app/papers/details/b6e4afa458ac589f90acb19801c1e2b6/?utm_source=claude_desktop). *Geological Journal.*
24. Shen et al. (2023). [The role of fluvial and tidal currents on coal accumulation in a mixed‐energy deltaic setting: Pinghu Formation, Xihu Depression, East China Sea Shelf Basin](https://consensus.app/papers/details/23ba5bec8eb956b1ad6dfa18d27eef7a/?utm_source=claude_desktop). *Sedimentology.*
25. Wainman et al. (2016). [Tectonic and climatic controls on facies distribution in the Jurassic Walloon coal measures, Surat basin, Australia](https://consensus.app/papers/details/01d4d37f05b95606b6f3cd9aba4266b7/?utm_source=claude_desktop). Conference abstract.
26. Compton et al. (2016). [Phosphorite deposits on the Namibian shelf](https://consensus.app/papers/details/359c57a0a7255a018201aa1d4265e551/?utm_source=claude_desktop). *Marine Geology.*
27. Lumiste et al. (2021). [Constraining the conditions of phosphogenesis: Stable isotope and trace element systematics of Recent Namibian phosphatic sediments](https://consensus.app/papers/details/21980c38b1b453c986f4c6f0ba1a5ab8/?utm_source=claude_desktop). *Geochimica et Cosmochimica Acta.*
28. Schöllhorn et al. (2020). [Enhanced upwelling and phosphorite formation in the northeastern Pacific during the late Oligocene: Depositional mechanisms, environmental conditions, and the impact of glacio-eustacy](https://consensus.app/papers/details/431947de250e57c2aaa7d73c60949822/?utm_source=claude_desktop). *GSA Bulletin.*
29. Álvaro et al. (2016). [Ediacaran–Cambrian phosphorites from the western margins of Gondwana and Baltica](https://consensus.app/papers/details/17dfb13befd75a54bf8ac528eee1a647/?utm_source=claude_desktop). *Sedimentology.*
30. Zhang et al. (2019). [Economic phosphorite from the Ediacaran Doushantuo Formation, South China, and the Neoproterozoic-Cambrian Phosphogenic Event](https://consensus.app/papers/details/4a752815e4a65e80a32d624ec6ab77ea/?utm_source=claude_desktop). *Sedimentary Geology.*
31. Xu et al. (2024). [Oxidative weathering on the continent and seawater upwelling along the passive continental margin promoted widespread phosphorite formation at the Neoproterozoic-Cambrian boundary in South China](https://consensus.app/papers/details/c1795a3ab6005c13ba42b7b0129282d0/?utm_source=claude_desktop). *Chemical Geology.*

Weathering

32. Thorne et al. (2012). [Climate change and the formation of nickel laterite deposits](https://consensus.app/papers/details/0b01837f76ca5866b1e4c4313cf3d69e/?utm_source=claude_desktop). *Geology.*
33. Herrington et al. (2016). [Bauxite and Nickel-Cobalt Lateritic Deposits of the Tethyan Belt](https://consensus.app/papers/details/0ffd3e49d54e59bdb2eacba988fc503a/?utm_source=claude_desktop).
34. Taylor et al. (1992). [Cool Climate Lateritic and Bauxitic Weathering](https://consensus.app/papers/details/d5f91878a13b54a181a83561a43472ab/?utm_source=claude_desktop). *The Journal of Geology.*
35. Freyssinet et al. (2005). [Ore-Forming Processes Related to Lateritic Weathering](https://consensus.app/papers/details/82082145417c57d1adc8d0b655222a6d/?utm_source=claude_desktop).
36. Chanvry et al. (2020). [Tectonic and climate control on allochthonous bauxite deposition. Example from the mid-Cretaceous Villeveyrac basin, southern France](https://consensus.app/papers/details/d3a69f8adacc5413a93787a691c660bb/?utm_source=claude_desktop). *Sedimentary Geology.*

Erosion and transport

37. Roy et al. (2018). [Gold in the hills: patterns of placer gold accumulation under dynamic tectonic and climatic conditions](https://consensus.app/papers/details/2080d9ed94ae58a99824eabad9681f01/?utm_source=claude_desktop). *Mineralium Deposita.*
38. Knight et al. (1999). [The Relationship between Placer Gold Particle Shape, Rimming, and Distance of Fluvial Transport as Exemplified by Gold from the Klondike District, Yukon Territory, Canada](https://consensus.app/papers/details/7a8bffe9676d52038d38384b6ff6b0fe/?utm_source=claude_desktop). *Economic Geology.*
39. Palmer et al. (2024). [Redistribution of fine gold from hydrothermal sources to sedimentary sinks, Rakaia River, Canterbury, New Zealand](https://consensus.app/papers/details/3d6edbe756d65da99adf7ce6a98f66c0/?utm_source=claude_desktop). *New Zealand Journal of Geology and Geophysics.*
40. Craw (1999). [Variation in placer style, gold morphology, and gold particle behavior down gravel bed-load rivers; an example from the Shotover/Arrow-Kawarau-Clutha River system, Otago, New Zealand](https://consensus.app/papers/details/06a625127a4556caae71ab4f18f686b5/?utm_source=claude_desktop). *Economic Geology.*

Copper and craton margins

41. Hitzman et al. (2005). [The sediment-hosted stratiform copper ore system](https://consensus.app/papers/details/b37da35fa8325970b0da618291e1b533/?utm_source=claude_desktop). *Economic Geology.*
42. Hitzman et al. (2010). [Formation of Sedimentary Rock-Hosted Stratiform Copper Deposits through Earth History](https://consensus.app/papers/details/faa24a3bd17a563c97c1e3acea6fbe62/?utm_source=claude_desktop). *Economic Geology.*
43. Brown (2005). [Refinements for footwall red-bed diagenesis in the sediment-hosted stratiform copper deposits model](https://consensus.app/papers/details/215d8914c1c852f78066035977ab7f03/?utm_source=claude_desktop). *Economic Geology.*
44. Hoggard et al. (2019). [Global distribution of sediment-hosted metals controlled by craton edge stability](https://consensus.app/papers/details/b635db430ab455658afe1d3323acfa13/?utm_source=claude_desktop). *Nature Geoscience.*
45. Jones et al. (2023). [Fluid Flow, Alteration, and Timing of Cu-Ag Mineralization at the White Pine Sediment-Hosted Copper Deposit, Michigan, USA](https://consensus.app/papers/details/4faa3a56b2275ec3b249e17651db12aa/?utm_source=claude_desktop). *Economic Geology.*
46. Maghfouri et al. (2020). [Metallogeny and temporal–spatial distribution of sediment-hosted stratabound copper (SSC-type) deposits in Iran; implications for future exploration](https://consensus.app/papers/details/a23881aaadf45cb6903042a846c6382b/?utm_source=claude_desktop). *Ore Geology Reviews.*
47. Skirrow (2021). [Iron oxide copper-gold (IOCG) deposits – a review (part 1): settings, mineralogy, ore geochemistry, and classification within the Cu-Au-Fe (± Co, REE) deposit family](https://consensus.app/papers/details/24cccc02374f57ffb1d6464442d1e8fb/?utm_source=claude_desktop). *Ore Geology Reviews.*
48. Groves et al. (2010). [Iron Oxide Copper-Gold (IOCG) Deposits through Earth History: Implications for Origin, Lithospheric Setting, and Distinction from Other Epigenetic Iron Oxide Deposits](https://consensus.app/papers/details/0e3b5bd95bd055c387b3a2657fed0606/?utm_source=claude_desktop). *Economic Geology.*
49. Sillitoe (2003). [Iron oxide-copper-gold deposits: an Andean view](https://consensus.app/papers/details/108f95ee367f584a94cb24af8d5675ef/?utm_source=claude_desktop). *Mineralium Deposita.*
50. Richards et al. (2017). [Contrasting Tectonic Settings and Sulfur Contents of Magmas Associated with Cretaceous Porphyry Cu ± Mo ± Au and Intrusion-Related Iron Oxide Cu-Au Deposits in Northern Chile](https://consensus.app/papers/details/04823f1ed28754258a22b68330ebc322/?utm_source=claude_desktop). *Economic Geology.*
51. Richards et al. (2013). [Magmatic-hydrothermal processes within an evolving Earth: Iron oxide-copper-gold and porphyry Cu ± Mo ± Au deposits](https://consensus.app/papers/details/0d4b4c6a9e555a63aee3aa0257921602/?utm_source=claude_desktop). *Geology.*
52. Williams et al. (2005). [Iron oxide copper-gold deposits: geology, space-time distribution, and possible modes of origin](https://consensus.app/papers/details/c130d02a7a1252febbfa381c9b0b228b/?utm_source=claude_desktop).
53. Groves et al. (2020). [Craton and thick lithosphere margins: The sites of giant mineral deposits and mineral provinces](https://consensus.app/papers/details/791d29f7c158545383adff2862089ec1/?utm_source=claude_desktop). *Gondwana Research.*

Iron

54. Hagemann et al. (2016). [BIF-hosted iron mineral system: A review](https://consensus.app/papers/details/6c5a8f1099ff510daef36eec134580a1/?utm_source=claude_desktop). *Ore Geology Reviews.*
55. Smith et al. (2016). [Palaeoproterozoic Banded Iron formationhosted High-Grade Hematite Iron Ore Deposits of the Transvaal Supergroup, South Africa](https://consensus.app/papers/details/0b306323d78156cb89f37849586a7fb0/?utm_source=claude_desktop). *Episodes.*
56. Hagemann et al. (2008). [Banded Iron Formation-Related High-Grade Iron Ore](https://consensus.app/papers/details/30c4929a37795338ac2944cce7cab743/?utm_source=claude_desktop).
57. Lascelles (2012). [Banded iron formation to high-grade iron ore: a critical review of supergene enrichment models](https://consensus.app/papers/details/f966d8a5c7305f249203d3add13087e4/?utm_source=claude_desktop). *Australian Journal of Earth Sciences.*
58. Perring et al. (2022). [Phanerozoic history of the Pilbara region: implications for iron mineralisation](https://consensus.app/papers/details/056c6632b828584eaf260a730b1f45b7/?utm_source=claude_desktop). *Australian Journal of Earth Sciences.*
59. Young (1989). [Phanerozoic ironstones: an introduction and review](https://consensus.app/papers/details/5743d3fcdc05557daf24617923e5c33d/?utm_source=claude_desktop). *Geological Society, London, Special Publications.*
60. Arthur (1989). [Temporal patterns among Phanerozoic oolitic ironstones and oceanic anoxia](https://consensus.app/papers/details/75a758ca236b556ea51e0bf785187a82/?utm_source=claude_desktop). *Geological Society, London, Special Publications.*
61. Lechte et al. (2024). [Oolitic ironstones, continental iron flux and reverse weathering in the Proterozoic Eon: Insights from the Tonian Katherine Group, Yukon](https://consensus.app/papers/details/99125eae624b58c9b796aacfffbc0300/?utm_source=claude_desktop). *Earth-Science Reviews.*
62. Matheson et al. (2022). [Ironstone as a proxy of Paleozoic ocean oxygenation](https://consensus.app/papers/details/c0d38c25bf0d58b8ade76652216ce2a5/?utm_source=claude_desktop). *Earth and Planetary Science Letters.*
63. Matheson et al. (2024). [Late Ordovician ironstone and its relation to ocean redox instability, climate and glaciation](https://consensus.app/papers/details/2ad2649cb2305f24a1111f3dc8017f34/?utm_source=claude_desktop). *Sedimentology.*
64. Luan et al. (2025). [Upwelling-related ferruginous ooids, microbialites, and the Darriwilian tipping point of Ordovician climate](https://consensus.app/papers/details/a01aaec520d35ddebd816cbc3c6ef574/?utm_source=claude_desktop). *Geology.*
65. Bhattacharyya (1982). [Phanerozoic Oolitic Ironstones--Geologic Record and Facies Model](https://consensus.app/papers/details/58241592d93156a5a7d7027f4b23acfa/?utm_source=claude_desktop). *Annual Review of Earth and Planetary Sciences.*
66. Andersson et al. (2021). [Structural Evolution of the Central Kiruna Area, Northern Norrbotten, Sweden: Implications on the Geologic Setting Generating Iron Oxide-Apatite and Epigenetic Iron and Copper Sulfides](https://consensus.app/papers/details/8d3aae51a84b525e84d6b20a7601fb1b/?utm_source=claude_desktop). *Economic Geology.*
67. Sarlus et al. (2020). [Timing and origin of the host rocks to the Malmberget iron oxide-apatite deposit, Sweden](https://consensus.app/papers/details/bd9e0a00f3c754c1b32046356380ac66/?utm_source=claude_desktop). *Precambrian Research.*
68. Simon et al. (2018). [Kiruna-Type Iron Oxide-Apatite (IOA) and Iron Oxide Copper-Gold (IOCG) Deposits Form by a Combination of Igneous and Magmatic-Hydrothermal Processes: Evidence from the Chilean Iron Belt](https://consensus.app/papers/details/d57baa1005995526b86b02b8f57ebf76/?utm_source=claude_desktop).
69. Troll et al. (2019). [Global Fe–O isotope correlation reveals magmatic origin of Kiruna-type apatite-iron-oxide ores](https://consensus.app/papers/details/c55e0149ca125cdcb55a40ddea19bf0c/?utm_source=claude_desktop). *Nature Communications.*
70. Zeng et al. (2024). [The role of iron-rich hydrosaline liquids in the formation of Kiruna-type iron oxide–apatite deposits](https://consensus.app/papers/details/5dd7620e6c1d5422b2d16af2946aa365/?utm_source=claude_desktop). *Science Advances.*
71. Hitzman et al. (1992). [Geological characteristics and tectonic setting of proterozoic iron oxide (CuUAuREE) deposits](https://consensus.app/papers/details/2eed7d5f91ac5f1d85873714e5b9e749/?utm_source=claude_desktop). *Precambrian Research.*
72. Smith et al. (2025). [Trace element geochemistry of magnetite from iron oxide-apatite (Kiruna type) and magnetite skarn deposits: A discriminant of deposit type and genesis?](https://consensus.app/papers/details/363f1b83d6105ad1b6f7753eb2733f14/?utm_source=claude_desktop). *Mineralogical Magazine.*

Gold

73. Goldfarb et al. (2005). [Distribution, character and genesis of gold deposits in metamorphic terranes](https://consensus.app/papers/details/a5ee53faf42d5fffbea960f18758824c/?utm_source=claude_desktop).
74. Groves et al. (2018). [Structural geometry of orogenic gold deposits: Implications for exploration of world-class and giant deposits](https://consensus.app/papers/details/dc9a857362dc5052ba1a57a13e156442/?utm_source=claude_desktop). *Geoscience Frontiers.*
75. Gaboury (2019). [Parameters for the formation of orogenic gold deposits](https://consensus.app/papers/details/f10b5f723c9351679dda9715b101d011/?utm_source=claude_desktop). *Applied Earth Science.*
76. Wang et al. (2021). [Towards a universal model for orogenic gold systems: A perspective based on Chinese examples with geodynamic, temporal, and deposit-scale structural and geochemical diversity](https://consensus.app/papers/details/945be0b5051355a2af4fc324e22ac51e/?utm_source=claude_desktop). *Earth-Science Reviews.*
77. Wang et al. (2019). [A review of intermediate sulfidation epithermal deposits and subclassification](https://consensus.app/papers/details/c6fb0f1805f85c53a0582bf702a8c847/?utm_source=claude_desktop). *Ore Geology Reviews.*
78. Bissig et al. (2015). [Physiographic and tectonic settings of high-sulfidation epithermal gold–silver deposits of the Andes and their controls on mineralizing processes](https://consensus.app/papers/details/9831234d934b5473b1f900bbc49083dd/?utm_source=claude_desktop). *Ore Geology Reviews.*
79. Prihatmoko et al. (2020). [Low-sulfidation epithermal gold deposits in Java, Indonesia: Characteristics and linkage to the volcano-tectonic setting](https://consensus.app/papers/details/ee66b85c7c1e5c10a4a1eec42904a1d5/?utm_source=claude_desktop). *Ore Geology Reviews.*
80. Rhys et al. (2020). [Chapter 3: Structural Controls on Ore Localization in Epithermal Gold-Silver Deposits: A Mineral Systems Approach](https://consensus.app/papers/details/59988951565758e3bbaa70b23a535cfc/?utm_source=claude_desktop).
81. Muntean (2020). [Chapter 36: Carlin-Type Gold Deposits in Nevada: Geologic Characteristics, Critical Processes, and Exploration](https://consensus.app/papers/details/8ea6f3b9dc2e5820ab1af18ac89d256e/?utm_source=claude_desktop). *Geology of the World's Major Gold Deposits and Provinces.*
82. Muntean et al. (2011). [Magmatic–hydrothermal origin of Nevada's Carlin-type gold deposits](https://consensus.app/papers/details/6f0fa11398c351c08a9bbf47803a88f0/?utm_source=claude_desktop). *Nature Geoscience.*
83. Hofstra et al. (2000). [Characteristics and models for carlin-type gold deposits](https://consensus.app/papers/details/29d1d1a090685bd88756adf6979ec16b/?utm_source=claude_desktop).
84. Su et al. (2018). [Carlin-Type Gold Deposits in the Dian-Qian-Gui "Golden Triangle" of Southwest China](https://consensus.app/papers/details/20a1057a5f685f2b82ce6ba9ff5149ff/?utm_source=claude_desktop).
85. Chai et al. (2026). [The source of the gold in Carlin-type gold deposits](https://consensus.app/papers/details/ea16f9d561795c2fa86dec2f3843b594/?utm_source=claude_desktop). *Communications Earth & Environment.*
86. Phillips et al. (2024). [The Temporal Distribution of the Host Rocks to Gold, the Archean Witwatersrand Basin, South Africa](https://consensus.app/papers/details/c7a24e55b2bf5c5abdfc5807407409f9/?utm_source=claude_desktop). *Minerals.*
