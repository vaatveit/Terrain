"""Mineral prospectivity: where copper, iron and gold could plausibly be.

A post-process. It reads fields the archive already carries and changes nothing upstream,
so a saved world can be re-scored without regenerating it. The research behind every choice
here, and the choices that were measured and dropped, are in mineral-distribution.md.

The output is a SCORE, not a deposit. Each resource field is 0 to 1 per cell and marks a
belt or a district; a single deposit is far smaller than one cell at this world's scale. The
sim that consumes the world samples cells in proportion to field times cell area.

Scores are absolute, not rank-normalised per world: a world with no continental rift has no
sediment-hosted copper. Worlds are meant to differ in what they are rich in.

How a belt is scored. Three things are read off the nearest plate boundary, and they are
the only three, because they are what this model's tectonics actually contain:

  where      an ARC BAND offset inland, and the belt AXIS on the boundary itself. Following
             `plates.derive_uplift`, every convergent boundary in continental crust carries
             both. Splitting arc from collision by the crust on each side was the plan and
             was measured away: only ~5% of convergent boundary cells have ocean against
             continent, because crust type is noise laid down independently of the plates.
  how        OBLIQUITY, slip/speed from `plates.boundary_slip`. Its square is the share of
             the motion that is slip, so 1 - o^2 and o^2 split a belt into a head-on part
             and an oblique part that sum to one.
  how deep   EROSION, as uplift over its ceiling. At steady state erosion equals uplift, so
             where uplift is high the shallow levels of the crust are gone and the deep
             ones are at the surface. (1-e)^2, 2e(1-e) and e^2 are shallow, mid and deep
             windows that sum to one. This is a rank, not metres: the world's relief is
             compressed and Earth's depth figures do not transfer.

The arc band then sorts into six deposit types with no further parameters:

                 head-on                  oblique
    shallow      epithermal_hs            epithermal_ls
    mid          porphyry                 iocg
    deep         (stripped; nothing)      kiruna_iron

Orogenic gold takes the belt axis, deep erosion, and motion that both closes and slips.
Sediment-hosted copper takes continental rifts.

CRATONS ARE INVENTED. Most of the world's iron, and a quarter of its copper, follow the
edges and interiors of cratons: old continental crust with a thick, cold root. A craton is
defined by its age and by the lithosphere under it, and this model has one snapshot of
plates and no mantle, so it has neither. `invent_cratons` draws them from noise instead:
a share of continental crust, and inside it a smaller share that holds iron formation. They
are laid down independently of the plates on purpose -- real craton margins are OLD
boundaries, unrelated to the present ones, and a margin derived from today's boundaries
would just be a contour parallel to them. Nothing else in this file is made up this way,
and two things rest on it: `bif_iron`, and the share of `sediment_copper` that a rift only
earns near a craton margin.

A third was built and removed. The giant iron oxide copper-gold bodies sit within ~100 km
of craton margins, so a ring was scored along every margin. A sim samples in proportion to
score times AREA, and a ring has no selectivity along its length or with depth, so at a
weight of 0.07 it carried 67-72% of all the copper, measured at n=128 over three seeds. A
weight is not a share. The belt types multiply three factors that each cut the ground down;
anything that does not will swamp them, whatever weight it is given.

CLIMATE UPGRADES AND THROTTLES; IT MAKES NOTHING. Three types read the shipped climate,
and each only scales a type the plates or the cratons already put there:
`supergene_copper` is a porphyry where there is rain enough to leach it and not enough to
erode it away; `enriched_iron` is iron formation weathered to ore where it is warm and wet;
and a rift keeps only a share of its `sediment_copper` unless it is dry enough for
evaporites.

RIVERS CARRY TWO TYPES DOWNSTREAM. `placer_gold` is orogenic gold, and `channel_iron` is
enriched iron formation, picked up by the river below it and dropped where the river
slackens. Each cell passes its load to its receiver and loses a little per cell, so what
arrives is a mean of the sources upstream that favours the near ones; the drop is how much
gentler this reach is than the river above it. Only river cells score. The placer in a
lode's own cell is not a separate thing at this scale -- coarse gold travels ~5 km on
Earth, which is half a cell here -- so this is the distal, fine-gold class only.

The reach is a WORLD-SCALE length and a river is a CELL-SCALE thing, so the routed types
cannot be judged on a small fixture: at n=128 the reach is 1.7 cells and at n=785 it is
10. Like anything that reads elevation in metres, they are measured at production size.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numba import njit

from terrain import flow, noise, plates
from terrain.stage1 import area_quantile


@dataclasses.dataclass
class MineralParams:
    # Closing or opening rate at which a boundary counts as fully active, mm/yr. Below it
    # the score falls linearly to zero: no motion, no arc and no rift.
    full_rate_mm_per_yr: float = 40.0
    # Weight of each deposit type in each resource field. THESE ARE NOT SHARES. A sim
    # samples in proportion to score times area, so a type that covers provinces outweighs
    # one confined to river reaches whatever the two are weighted. They are the output of
    # scripts/tune_mineral_weights.py, which sets each weight to the type's intended share
    # over the ground it covers; the shares, and which of them are sourced, are in that
    # script. Each dict sums to at most 1, so a field cannot exceed 1.
    #
    # Tuned on three production worlds, n=785, seeds 42 / 1 / 149. Share each type then
    # carries, against its target:
    #
    #     copper   porphyry      40%   37 / 51 / 34      gold   orogenic      40%   39 / 38 / 44
    #              supergene     20%   18 / 20 / 22             placer        15%   15 / 17 / 13
    #              sediment      23%   29 / 12 / 27             epithermal_hs 15%   15 / 16 / 14
    #              iocg          17%   17 / 18 / 17             epithermal_ls 15%   17 / 14 / 14
    #     iron     enriched      50%   46 / 49 / 54             porphyry      10%   10 / 10 / 10
    #              formation     35%   40 / 34 / 32             iocg           5%    5 /  4 /  6
    #              kiruna        10%   10 / 10 / 10
    #              channel        5%    4 /  6 /  4
    #
    # Before tuning the weights WERE the shares, and the worlds did not agree: porphyry
    # carried 67-78% of the copper, placers 1.0-1.4% of the gold and iron formation 72-79%
    # of the iron. What moved most is what covers least ground. Placer gold needs 0.60 to
    # be picked 15% of the time, so a placer reach now outscores the lode beside it.
    #
    # Do not re-tune on a small world. The routed and the climatic types do not transfer:
    # see the script.
    copper: dict = dataclasses.field(default_factory=lambda: {
        "porphyry": 0.1272, "supergene_copper": 0.2435, "sediment_copper": 0.5011,
        "iocg": 0.1279})
    gold: dict = dataclasses.field(default_factory=lambda: {
        "orogenic_gold": 0.1545, "placer_gold": 0.6039, "epithermal_hs": 0.0689,
        "epithermal_ls": 0.0750, "porphyry": 0.0446, "iocg": 0.0528})
    iron: dict = dataclasses.field(default_factory=lambda: {
        "enriched_iron": 0.3224, "bif_iron": 0.0819, "kiruna_iron": 0.1871,
        "channel_iron": 0.4084})

    # How far a river carries what it picks up, as a fraction of the sphere's radius:
    # 100 km on Earth, the distance fine gold is known to travel. The load a cell passes
    # to its receiver falls by 1/e over this length.
    placer_reach_r: float = 100.0 / 6371.0

    # Climate. Each pair is a ramp from 0 at the first value to 1 at the second, in mm/yr
    # of rain or degrees C of mean annual temperature. Only the 120 mm is sourced; the rest
    # are judgement, and they cut a rainfall field the README records as over-dispersed.
    #
    # Supergene copper wants enough rain to leach and not enough to erode: it rises to
    # full by 120 mm, the published minimum, and fades out between 500 and 1000.
    supergene_rain_mm: tuple = (0.0, 120.0)
    supergene_too_wet_mm: tuple = (1000.0, 500.0)
    # Iron formation is enriched to ore by deep weathering where it is warm and wet.
    enrichment_rain_mm: tuple = (500.0, 1000.0)
    enrichment_warmth_c: tuple = (15.0, 20.0)
    # Rift copper wants evaporites, which is a dry climate. Dryness ALONE, not warmth and
    # not latitude: a cold desert makes evaporites too, and this model has almost no hot
    # dry land to ask for (4% at n=128), having no subtropical desert belt.
    evaporite_dry_mm: tuple = (750.0, 250.0)
    # What a rift keeps without them. Not zero: one giant deposit has no evaporites.
    rift_share_without_evaporites: float = 0.25

    # Invented cratons; see the module docstring. Both shares are of AREA, and both
    # frequencies are in the unit-sphere units the crust noise uses, where continents are
    # drawn at 0.7. Cratons at 2.5 come out a few to a continent; iron-formation provinces
    # at 8 come out a few to a craton.
    craton_fraction: float = 0.30          # of continental crust
    craton_frequency: float = 2.5
    iron_formation_fraction: float = 0.15  # of craton
    iron_formation_frequency: float = 8.0
    # How far a craton margin reaches, as a fraction of the sphere's radius: 100 km on
    # Earth, the distance within which most craton-margin deposit types are found.
    # Sediment-hosted copper, the one type scored against it, fades over twice that, the
    # 200 km of its own source.
    craton_margin_r: float = 100.0 / 6371.0
    # 85% of sediment-hosted base metal lies within that 200 km; a rift with no craton
    # margin in range keeps the rest.
    rift_share_off_craton: float = 0.15


def invent_cratons(grid, seed, is_continental, params: MineralParams = None):
    """Draw cratons and iron-formation provinces from noise. See the module docstring.

    Returns `craton` and `iron_formation`, both (N,) bool, and `craton_margin_km`, (N,)
    float32: great-circle distance to the nearest craton margin from either side, with the
    1e30 sentinel of `plates.boundary_distance` beyond twelve margin lengths. Deterministic
    in `seed`, on sub-seeds of their own, so a saved world's cratons can be drawn again.
    """
    p = params or MineralParams()
    n = grid.cell_count
    xyz = np.ascontiguousarray(grid.xyz)

    def province(name, frequency, within, fraction):
        field = np.empty(n, dtype=np.float64)
        noise.fbm(xyz, noise.permutation(noise.sub_seed(name, seed)), noise.GRAD3, 3,
                  frequency, 2.0, 0.5, field)
        return within & (field > area_quantile(field[within], grid.area[within],
                                               1.0 - fraction))

    craton = province("craton", p.craton_frequency, np.asarray(is_continental, dtype=bool),
                      p.craton_fraction)
    iron = province("iron_formation", p.iron_formation_frequency, craton,
                    p.iron_formation_fraction)

    on_margin = np.zeros(n, dtype=bool)
    for s in range(grid.neighbours.shape[1]):
        j = grid.neighbours[:, s]
        ok = j >= 0
        on_margin[ok] |= craton[ok] != craton[j[ok]]
    dist = np.empty(n, dtype=np.float32)
    plates.boundary_distance(on_margin, grid.xyz, grid.radius_km, dist,
                             np.empty(n, dtype=np.int32),
                             12.0 * p.craton_margin_r * grid.radius_km)
    return {"craton": craton, "iron_formation": iron, "craton_margin_km": dist}


@njit(cache=True)
def _carry_downstream(order, rec, keep, load):
    """Pass each cell's load, scaled by `keep`, to its receiver, donors before receivers.

    `load` is (N, K) and is accumulated in place, so K quantities ride one traversal.
    `order` is `flow.build_order`'s, receivers first, so it is walked backwards.
    """
    for k in range(order.shape[0] - 1, -1, -1):
        c = order[k]
        r = rec[c]
        if r != c:
            for q in range(load.shape[1]):
                load[r, q] += keep * load[c, q]
    return load


def _ramp(x, lo_hi):
    """0 at `lo`, 1 at `hi`, linear between; `lo` may be the larger, for a falling ramp."""
    lo, hi = lo_hi
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)


def deposit_scores(*, land_mask, is_continental, boundary_stress, boundary_speed,
                   boundary_slip, boundary_dist_km, nearest_boundary, uplift,
                   precipitation_mm, temperature_c, flow_receiver, slope, river_width_m,
                   craton_margin_km, iron_formation, cell_km, radius_km,
                   params: MineralParams = None, plate_params: plates.PlateParams = None):
    """Score per deposit type, each (N,) float32 in 0..1. See the module docstring.

    The array arguments are the archive's fields under the same names, so a loaded world
    can be passed straight in, with `craton_margin_km` and `iron_formation` from
    `invent_cratons`.
    """
    p = params or MineralParams()
    pp = (plate_params or plates.PlateParams()).for_radius(radius_km)

    near = np.asarray(nearest_boundary)
    ground = np.asarray(land_mask, dtype=bool) & np.asarray(is_continental, dtype=bool)
    ok = (near >= 0) & ground
    src = np.maximum(near, 0)
    speed = np.asarray(boundary_speed, dtype=np.float64)[src]
    slip = np.asarray(boundary_slip, dtype=np.float64)[src]
    stress = np.asarray(boundary_stress)[src]
    d = np.asarray(boundary_dist_km, dtype=np.float64)

    moving = speed > 0.0
    slip_share = np.where(moving, np.clip(slip / np.where(moving, speed, 1.0), 0.0, 1.0) ** 2, 0.0)
    rate = np.clip(speed * np.sqrt(1.0 - slip_share) / p.full_rate_mm_per_yr, 0.0, 1.0)
    rate = np.where(ok, rate, 0.0)
    # The sign of the normal motion is the one thing still taken from `boundary_stress`.
    # It is unreliable only where the normal component is near zero, and `rate` is too.
    closing = np.where(stress > 0.0, rate, 0.0)
    opening = np.where(stress < 0.0, rate, 0.0)

    arc = closing * np.exp(-((d - pp.arc_offset_km) / pp.arc_width_km) ** 2)
    axis = closing * np.exp(-d / pp.orogen_decay_km)
    rift = opening * np.exp(-(d / pp.rift_shoulder_km) ** 2)

    e = np.clip(np.asarray(uplift, dtype=np.float64) / pp.u_max_m_per_myr, 0.0, 1.0)
    shallow, mid, deep = (1.0 - e) ** 2, 2.0 * e * (1.0 - e), e ** 2
    head_on = 1.0 - slip_share
    # sin(2 theta) for motion at angle theta to the boundary normal: zero when the plates
    # only close or only slide, one when they do both equally.
    both = 2.0 * np.sqrt(slip_share * head_on)

    # The craton terms. Quiet ground is the shallow window again: a craton that has since
    # been thrown up into an orogen is no longer one.
    to_margin = np.asarray(craton_margin_km, dtype=np.float64)
    near_margin = np.exp(-(to_margin / (2.0 * p.craton_margin_r * radius_km)) ** 2)
    quiet = np.where(ground, shallow, 0.0)

    # Climate. It upgrades or throttles a type that is already there and makes none.
    semi_arid = _ramp(precipitation_mm, p.supergene_rain_mm) * _ramp(
        precipitation_mm, p.supergene_too_wet_mm)
    warm_wet = _ramp(precipitation_mm, p.enrichment_rain_mm) * _ramp(
        temperature_c, p.enrichment_warmth_c)
    dry = _ramp(precipitation_mm, p.evaporite_dry_mm)

    def share(off, term):
        return off + (1.0 - off) * term

    porphyry = arc * head_on * mid
    bif_iron = quiet * np.asarray(iron_formation, dtype=bool)
    scores = {
        "porphyry": porphyry,
        "supergene_copper": porphyry * semi_arid,
        "epithermal_hs": arc * head_on * shallow,
        "epithermal_ls": arc * slip_share * shallow,
        "iocg": arc * slip_share * mid,
        "kiruna_iron": arc * slip_share * deep,
        "orogenic_gold": axis * both * deep,
        "sediment_copper": (rift * share(p.rift_share_off_craton, near_margin)
                            * share(p.rift_share_without_evaporites, dry)),
        "bif_iron": bif_iron,
        "enriched_iron": bif_iron * warm_wet,
    }

    # Routed types: what a river carries off a source and drops where it slackens. Each
    # cell passes its load to its receiver, losing a little per cell, so what reaches a
    # cell is a mean of the sources upstream that favours the near ones. Carrying a 1 the
    # same way gives the weight to divide by, and carrying the slope gives the gradient of
    # the river above, against which this reach is slack or not.
    n = near.shape[0]
    rec = np.ascontiguousarray(flow_receiver, dtype=np.int32)
    order = np.empty(n, dtype=np.int32)
    flow.build_order(rec, np.empty(n, dtype=np.int32), np.empty(n + 1, dtype=np.int64),
                     np.empty(n, dtype=np.int32), order)
    here = np.asarray(slope, dtype=np.float64)
    load = np.stack([np.ones(n), here, scores["orogenic_gold"], scores["enriched_iron"]],
                    axis=1)
    _carry_downstream(order, rec, np.exp(-cell_km / (p.placer_reach_r * radius_km)), load)
    weight, above = load[:, 0], load[:, 1] / load[:, 0]
    slack = np.where(above > 0.0, np.clip(1.0 - here / np.where(above > 0.0, above, 1.0),
                                          0.0, 1.0), 0.0)
    trap = slack * (np.asarray(river_width_m) > 0.0)
    scores["placer_gold"] = trap * load[:, 2] / weight
    scores["channel_iron"] = trap * load[:, 3] / weight
    return {name: s.astype(np.float32) for name, s in scores.items()}


def weights_for_shares(mass, shares):
    """Weights under which each deposit type carries its share of a resource, by area.

    `mass` is each type's score summed over area, and `shares` is what each should carry.
    A sim samples in proportion to score times area, so a type's weight has to be its share
    divided by the ground it covers: a type confined to a few river reaches needs a far
    larger weight than one spread over whole provinces to be picked as often. The result
    sums to 1, so the field built from it cannot exceed 1.

    Used by scripts/tune_mineral_weights.py, which is where `MineralParams`' weights come
    from.
    """
    raw = {t: shares[t] / mass[t] for t in shares}
    total = sum(raw.values())
    return {t: w / total for t, w in raw.items()}


def resources(*, params: MineralParams = None, **fields):
    """The three resource fields, each (N,) float32 in 0..1: copper, iron, gold.

    Each is the weighted sum of the deposit types that yield it; see `MineralParams`.
    Takes the same arguments as `deposit_scores`.
    """
    p = params or MineralParams()
    scores = deposit_scores(params=p, **fields)
    return {name: sum(w * scores[t] for t, w in weights.items()).astype(np.float32)
            for name, weights in (("copper", p.copper), ("iron", p.iron), ("gold", p.gold))}
