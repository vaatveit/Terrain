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

Not here yet, and the reason iron is thin: banded iron formation needs cratons, which this
model does not have and which will be invented, and placers need routing down the flow tree.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from terrain import plates


@dataclasses.dataclass
class MineralParams:
    # Closing or opening rate at which a boundary counts as fully active, mm/yr. Below it
    # the score falls linearly to zero: no motion, no arc and no rift.
    full_rate_mm_per_yr: float = 40.0
    # Share of each resource carried by each deposit type. Copper's are sourced: porphyry
    # is ~60% of world production and sediment-hosted ~23%, with the iron oxide family
    # taking the rest here. Gold's and iron's are a judgement -- the sources give an
    # ordering (orogenic gold first, iron formation first) and no shares -- and are meant to
    # be tuned against generated worlds. Each sums to at most 1 so a field cannot exceed 1;
    # gold and iron leave room for the types not built yet.
    copper: dict = dataclasses.field(default_factory=lambda: {
        "porphyry": 0.60, "sediment_copper": 0.23, "iocg": 0.17})
    gold: dict = dataclasses.field(default_factory=lambda: {
        "orogenic_gold": 0.45, "epithermal_hs": 0.15, "epithermal_ls": 0.15,
        "porphyry": 0.10, "iocg": 0.05})
    iron: dict = dataclasses.field(default_factory=lambda: {"kiruna_iron": 0.15})


def deposit_scores(*, land_mask, is_continental, boundary_stress, boundary_speed,
                   boundary_slip, boundary_dist_km, nearest_boundary, uplift, radius_km,
                   params: MineralParams = None, plate_params: plates.PlateParams = None):
    """Score per deposit type, each (N,) float32 in 0..1. See the module docstring.

    The array arguments are the archive's fields under the same names, so a loaded world
    can be passed straight in.
    """
    p = params or MineralParams()
    pp = (plate_params or plates.PlateParams()).for_radius(radius_km)

    near = np.asarray(nearest_boundary)
    ok = (near >= 0) & np.asarray(land_mask, dtype=bool) & np.asarray(is_continental, dtype=bool)
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

    scores = {
        "porphyry": arc * head_on * mid,
        "epithermal_hs": arc * head_on * shallow,
        "epithermal_ls": arc * slip_share * shallow,
        "iocg": arc * slip_share * mid,
        "kiruna_iron": arc * slip_share * deep,
        "orogenic_gold": axis * both * deep,
        "sediment_copper": rift,
    }
    return {name: s.astype(np.float32) for name, s in scores.items()}


def resources(*, params: MineralParams = None, **fields):
    """The three resource fields, each (N,) float32 in 0..1: copper, iron, gold.

    Each is the weighted sum of the deposit types that yield it; see `MineralParams`.
    Takes the same arguments as `deposit_scores`.
    """
    p = params or MineralParams()
    scores = deposit_scores(params=p, **fields)
    return {name: sum(w * scores[t] for t, w in weights.items()).astype(np.float32)
            for name, weights in (("copper", p.copper), ("iron", p.iron), ("gold", p.gold))}
