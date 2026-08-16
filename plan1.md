# The relief shortfall: measured at production scale, and what it actually forbids

> **Status: all three items are DONE.** Kept in the repo for the reasoning, not as a task
> list. Item 3 shipped at `climate_relief_gain = 7.7`; its measured result, including the
> one claim below that turned out to be false, is at the end of this file.
>
> **Item 2's production run changed the answer, and the extrapolation this plan is built on
> did not survive it.** Measured at n=785 rather than fitted from three smaller worlds:
>
> | | plan's extrapolation | **measured** | error |
> |---|---|---|---|
> | land mean | 47 m | **47.4 m** | +1.4% |
> | land elevation std | 85 m | **104.0 m** | −18% |
> | peak | 1,580 m | **2,428 m** | **−54%** |
>
> So the lapse-rate shortfall is **×7.7, not the ×9 below and not the ×13 it replaced**, and
> the world is **12.7× steeper than Earth** relative to its own radius, not 10×. The reason
> the fit failed is worth more than the fit was: **only the land mean is a power law.** Its
> local exponent is flat at 0.47 across a twelvefold range in radius, while the peak's
> steepens monotonically — 0.574, 0.918, 1.130 per doubling — so a straight line through
> log–log space was averaging a curve. This plan's own success criterion anticipated it
> ("if peak comes in near 2 km … the exponent needs rechecking"); it came in at 2,428 m.
>
> The current numbers live in `terrain-algorithms.md` under "What the temperature field is
> actually short of is relief" and in `climate.py:55-70`. **Read those, not the tables below.**
> Everything in "Why ×13 (or ×9) is not a knob" and "The reframing that matters" still holds
> — those arguments are about ratios and constraints, and a larger measured relief makes the
> case against raising `U` stronger, not weaker.

## Context

A note in `notes.txt` and in `terrain-algorithms.md:484-492` records that the temperature
field's real shortfall is relief: land elevation std **61 m** against Earth's ~800, so the
lapse rate contributes **0.28 °C** where Earth gives ~5, and that "scaling elevation ×13"
would take zonality below 0.94. It calls this a Stage 1/2 vertical-scale decision.

I investigated it. **The diagnosis is right and the prescription is not available.** Three
things came out of measuring it:

1. The numbers were taken on a prototype **3× smaller in radius** than the production world.
2. Relief does not scale the way the note assumes, and the correction is not a constant.
3. The ×13 is blocked by a constraint the rescale deliberately introduced — the hillslope
   budget it would spend is already exactly spent, by arithmetic rather than by measurement.

### What I measured

Current code, seed 1, three cached grids, `coupled.run(passes=3, steps_per_pass=200)`,
statistics area-weighted over land **above the datum** (which is what Earth's ~800 m is
measured over — the note's figure was contaminated by sub-datum land, and that inverted the
apparent trend):

| | n=64 (R=5.3 km) | n=128 (R=10.6) | n=256 (R=21.2) | fitted | ~~extrapolated n=785 (R=64.9)~~ |
|---|---|---|---|---|---|
| land mean | 14.6 m | 20.3 m | 27.8 m | R^+0.465 | ~~47 m~~ → **47.4 measured** |
| land std | 23.9 m | 34.3 m | 48.2 m | R^+0.508 | ~~85 m~~ → **104.0 measured** |
| peak | 243 m | 362 m | 685 m | R^+0.746 | ~~1,580 m~~ → **2,428 measured** |

*(The three measured columns stand. The extrapolated one was replaced by item 2's production
run — see the status note at the top. The paragraph below argues from the fitted exponents,
and its conclusion survives, but its R^0.746 peak exponent does not: the peak is not a power
law in radius.)*

The peak exponent **R^0.746** lands on the theoretical **R^0.757** — steady-state stream
power with `n=1` gives relief ∝ L_basin^(1−h·m), with Hack h=0.54 and m=0.45. That agreement
is the check that says the fit is real and not curve-drawing, and it independently confirms
`stage2.py:73`'s claim that `K=1.2e-6` "targets ~2 km peaks at n=785".

So at production resolution the lapse-rate signal is **0.55 °C, not 0.28** — the shortfall is
**×9, not ×13**. Still large. But note the two exponents differ: **peaks grow as R^0.75 while
typical land grows as R^0.51**. Bigger worlds get mountains faster than they get elevated
ground, so world size cannot close this — reaching Earth's 800 m std would need Earth's
radius, which is precisely the trade the 100 m cell decision already made and documented.

### Why ×13 (or ×9) is not a knob

**Uplift cannot rise.** `plates.py:103-132` records that `U` came down 12× at the rescale
because steady-state hilltop curvature is exactly `U/D`, and hillslopes only became resolved
at 100 m cells. That budget is now spent, and the argument is arithmetic: at `U` = 100 m/Myr,
`C = U/D = 5×10⁻³ 1/m` gives **100 m of convex relief over a 200 m hillslope and a basal
gradient of 1.0 — 45°, right at the angle of repose**. Raising `U` ×9 multiplies every
hillslope gradient by 9. Raising `D` to absorb it instead needs 0.6 m²/yr, well outside the
published 0.001–0.1 band for soil creep.

**This paragraph used to carry a second support, and that one is withdrawn.** It read
"measured slope distribution on the shipped n=256 world: p99 = 62.5°, p99.9 = 76.1°, max
83.7° — the steep ground is already past the angle of repose". The percentiles are real; the
reading of them was wrong. Decomposed while building `stage2.mass_wasting`, **98.1% of the
edges past repose have an ocean cell at one end** and 1.8% a lake cell, leaving 0.05% that
are genuinely land-to-land — it is a statistic about the shoreline, a land cell 100 m from
water held at base level, not about hillslopes, which run p50 0.9° and p90 9.4°. Nine SRTM
tiles now say the same from the other side: real terrain's p99 slope runs 31.6–55.1° across
every window measured, so 62.5° was never a plausible hillslope figure to begin with. The
retraction is recorded at `plates.py:121-132`.

**The conclusion is unchanged.** One of its two supports was never load-bearing, and the one
that carries it is the analytic `U/D` result above, not a measured distribution sitting at
failure.

**`K` cannot fall alone.** Lowering `K` raises channel relief without touching hilltop
curvature, which looks like the way out. It is not: hillslope length goes as
`(D/K)^(1/(1+2m))`, so ×9 in `K` lengthens hillslopes ~3.4× — and at fixed `U/D` a longer
hillslope is a *steeper* one (basal gradient `U·L/D`). It also collapses drainage density,
undoing the two metrics the 100 m rescale was undertaken to fix (local maxima 4.40% → 0.502%
against a nine-tile SRTM range of 0.090–0.508%; median drainage 1.0 → 5.98 cells against a
nine-tile median of 3). The "SRTM 0.14–0.26%" this cited reproduced nothing and has since
been replaced by measured values — it does not change the argument, which is about the
*direction* these two metrics would move.

**Raising `U` and `D` together is the only self-consistent terrain route, and it is not worth
it.** Holding `U/D` preserves hillslope form and puts the new relief in the channel network,
which is correct physics. The price: `D` goes 0.02 → ~0.19, outside the published 0.001–0.1
soil-creep band that `stage2.py:85` and `plates.py:82` both cite as the binding argument;
diffusion sub-steps scale linearly with `D`, so 20 → ~190, and `terrain-algorithms.md:252`
already calls sub-stepping "the real cost of the rescale"; and peaks reach ~15 km on a 65 km
sphere — 23% of the planet's radius.

### The reframing that matters

Relative to its own size this world is **not flat — it is already 12.7 times steeper than
Earth** (this read "ten times" from the extrapolated 85 m std; measured at 104.0 m it is
12.7×). Earth's land elevation std is 1.26e-4 of its radius; at n=785 this world's is 1.60e-3.
Giving it Earth's *absolute* 800 m would make it **98× steeper than Earth in relative terms**:
every continent a wall.

The lapse rate is the only piece of physics in the model denominated in **absolute metres**
(°C per km of altitude), which is why it alone reads the shrink as a defect. And there is a
real gap in the rescale's own bookkeeping here. `terrain-algorithms.md:237-241` sorts every
parameter into cell-scale / world-scale / vertical-physical, and puts the lapse rate in the
third class as "untouched; never referred to the sphere". That is true of the *parameter* and
false of the *term*: elevation is an emergent field, not a parameter, and it silently shrank
as R^0.5. Nothing in that three-class table covers emergent fields, and this is the one that
matters.

---

## Scope

**This plan is items 1 and 2 only.** Item 3 is recorded below as the option it is, and is
explicitly *deferred* — nothing in items 1 or 2 commits to it, and item 2's production run is
what decides whether it is worth doing at all.

### 1. Correct the record — **S** — **DONE**

The wrong numbers were written down as settled fact in two places and are the kind that get
acted on.

- ~~`terrain-algorithms.md:484-492`~~ **DONE** — the table (61 m / 0.28 °C / ×13) is replaced,
  under "What the temperature field is actually short of is relief". It carries the measured
  scaling with the production row rather than an extrapolation, the per-doubling local
  exponents, and the steeper-than-Earth reframing. It also records why the original std was
  wrong: measured over Stage-1 land *including cells below the datum*, which inverted the
  trend outright — std appeared to **fall** with world size as R^−0.265, because a small
  world's grabens are a large share of its relief. `climate.py:55-70` carries the same
  figures.
- `notes.txt` line 4 — **yours to decide.** Superseded by the above, but that file is your
  own and nothing here edits it.
- ~~**`plates.py:79-81` has a stale arithmetic error found on the way.**~~ **DONE.** It stated
  "C = U/D = 2e-3 1/m … 40 m of convex relief over a 200 m hillslope and a basal slope of 0.4",
  computed with the pre-trade `D = 0.05` and never updated when `stage2.py` took `D` to 0.02.
  With the shipped `D` the value is `1e-4/0.02 = 5e-3`, giving **100 m and a basal slope of
  1.0 (45°)**. Corrected in place; `plates.py:134` records the old figures. This mattered more
  than it looked: with the slope-percentile support withdrawn, this arithmetic is now the
  *only* thing holding the uplift ceiling up.

### 2. Confirm the extrapolation before anything is built on it — **M** — **DONE, and it failed**

Everything above is a two-decade fit extrapolated 3× past its largest point. One production
`n=785` run settles it. Worth doing before item 3, not before item 1.

Success criterion was: land std within ~15% of 85 m and peak within ~15% of 1,580 m; and
"if peak comes in near 2 km, `stage2.py:73`'s target was calibrated on something I have not
reproduced and the exponent needs rechecking."

**The peak came in at 2,428 m — 54% above the extrapolation, and the std 18% above.** The
criterion fired exactly as written. Rechecking it gave the reason: the peak is not a power
law in radius at all (local exponent 0.574 → 0.918 → 1.130 per doubling), so no single
fitted exponent could have extrapolated it. The land mean, whose exponent *is* flat at 0.47,
came in at +1.4%. **This is the argument for having run it rather than shipping the fit.**

### 3. ~~DEFERRED~~ — **DONE, shipped at 7.7**

If the climatic relief is wanted, buy it in the climate, not the terrain — **S to build, M to
re-verify**.

The cheapest honest route is a single gain on the elevation the **lapse rate** reads, in
`climate.temperature()` — not a change to `lapse_rate_c_per_km`, which should stay 6.5 °C/km
because that genuinely is atmospheric physics and does not care how big the planet is.

```
climate_relief_gain: float = 7.7   # elevation is an emergent field and shrank as R^0.5
```

**7.7, not the 9.0 this originally proposed** — the gain is the measured shortfall, and the
production run put that at ×7.7 (0.68 °C against Earth's ~5). Anything downstream of this
that says "×9 relief gain" means ×7.7.

One parameter, one call site, ~~**zero geomorphic cost** — Stage 2 and Stage 5 are untouched,
so θ, Hack's h, drainage density and the freshly-calibrated glacial `K` all keep their
validation.~~

**"Zero geomorphic cost" was false, and it is the one claim in this plan that was wrong on
its own terms rather than superseded by better measurement.** Temperature feeds `capacity` in
`climate.precipitation`, capacity feeds the river-evaporation source, and precipitation feeds
discharge weight and therefore erosion. Ocean cells genuinely are untouched — their lapse term
is zero at any gain, since elevation is clamped at zero — so the ocean moisture source does
not move. Land does. The claim was made by reading the call site and not the consumers.

Expected to move: temperature zonality 0.992 → ~0.94; Köppen D (7.1% against Earth's 24%);
the forest classes, currently all under-represented.

**The knock-on that must be checked, not assumed:** temperature feeds the glacial ELA, and
`glacial.K` was calibrated against a 14.2% drowning figure. A ×7.7 relief gain in
the temperature field will move the glaciated fraction and could re-open exactly the drowning
problem that recalibration closed. Re-run the `K` sweep table (`terrain-algorithms.md:370`)
before accepting the gain.

**State the tradeoff plainly in the docs if this ships:** it is a vertical exaggeration
applied to one consumer of the elevation field. It is defensible on the same grounds the
horizontal rescale was — every world-scale length was already redenominated, and this is the
one that was missed because it is emergent rather than a parameter — but it is not physics,
and the doc should say so in those words. **Done** — `README.md` known limitations,
`terrain-algorithms.md` "What was done about it, and what it cost", and the `ClimateParams`
docstring all say it in those words.

#### Measured result — n=128, seed 42

| | gain 1.0 | gain 7.7 |
|---|---|---|
| lapse term contribution, sd | 0.26 °C | **2.07 °C** |
| temperature zonality | 0.9935 | 0.9658 |
| **drowned land** | **17.36%** | **17.35%** |
| glaciated share | 20.6% | 21.7% |
| Köppen D total | 5.1% | 5.0% |
| Köppen ET tundra | 4.4% | 5.5% |
| raw windward/leeward | above 1.5 | **1.48** |

**The gate held.** Drowning did not move, so the `K` sweep did not need re-running — the
knock-on this plan insisted on checking turned out not to exist. The mechanism does its job:
relief contributes 2.07 °C of temperature structure instead of 0.26.

**Both of the benefits predicted above are wrong.** Zonality was expected at ~0.94 and reached
0.966 — and that prediction contradicted something already documented in `climate.py`, namely
that the latitude term's sd is 13.7 °C against ~1.5 for everything else combined, so no term
can move the aggregate. Köppen D was expected to improve and is flat at ~5%; the classes that
grew are ET tundra and the cold-desert/steppe pair, not the continental forests. The forest
classes this plan named are still under-represented.

**The cost is a small real one.** Colder ridges hold less water, so the raw windward/leeward
ratio fell from above 1.5 to 1.48 and the test bar moved to 1.45 with the reason recorded. The
isolated orographic test, which controls for the coast-to-interior gradient, still passes.

### Not worth doing

- **Raising `U`.** Blocked by hilltop curvature `C = U/D`, which already implies a 45° basal
  gradient. *Not* by the measured slope percentiles this line used to cite — those describe
  the coastline; see above.
- **Lowering `K` alone.** Collapses the drainage density the 100 m rescale exists to produce.
- **Raising `U` and `D` together.** The only consistent terrain route; costs `D` outside its
  published band, ~190 diffusion sub-steps, and 15 km peaks on a 65 km sphere. **L**, and the
  result is worse than what it replaces.
- **A larger world.** std grows only as R^0.51; Earth's std needs Earth's radius.

## Verification (items 1 and 2)

- `pytest` — **220** tests (206 when this plan was written). Item 1 touches only comments and
  prose and must not move any of them.
  If a test moves, item 1 was not the change I think it is.
- Item 2 is `scripts/generate_world.py --n 785`, reporting land mean / std / peak over
  above-datum land, area-weighted. Success: within ~15% of 47 m / 85 m / 1,580 m.
  **Result: 47.4 m / 104.0 m / 2,428 m — the mean passed, the std and peak did not.**
- Item 2 writes its measured row back into the scaling table from item 1, replacing the
  extrapolated column with a measured one. **The extrapolation must not survive in the doc
  once a real number exists**, and if the run disagrees with the fit, the disagreement is what
  gets written down rather than the fit. **This is what happened**, and the disagreement is
  recorded in `terrain-algorithms.md` rather than smoothed over: the fitted exponents and
  their predictions are still there next to the measured row, with the reason the peak fit
  failed.
