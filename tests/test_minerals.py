"""Mineral prospectivity: deposit-type scores and the three resource fields.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import export, flow, minerals, plates, stage1  # noqa: E402
from terrain.grid import R_WORLD_KM, build_goldberg  # noqa: E402

# The authored lengths, which is what a production-radius world sees.
ARC_KM = plates.PlateParams().arc_offset_km
FAR_KM = 6.0  # several arc widths and decay lengths out


def belt(dist_km, *, speed=40.0, slip=0.0, closing=True, uplift=50.0,
         continental=True, land=True, reached=True, craton_margin_km=1e3,
         iron_formation=False, precipitation_mm=200.0, temperature_c=25.0,
         downstream=False, slope=0.1, river=False):
    """Cells at the given distances from ONE boundary cell, which is cell 0.

    Every cell takes its plate motion from that boundary cell, exactly as a world's cells
    take theirs through `nearest_boundary`. Defaults: closing head-on at 40 mm/yr, which is
    full rate, with uplift at half its ceiling, far from any craton, in a hot dry climate,
    with no water flowing between the cells. `downstream=True` strings them into one river
    course, each cell draining to the next and the last to nowhere.
    """
    d = np.asarray(dist_km, dtype=np.float32)
    n = d.size
    on_boundary = np.zeros(n, dtype=np.float32)
    on_boundary[0] = 1.0
    return dict(
        land_mask=np.full(n, land), is_continental=np.full(n, continental),
        boundary_stress=on_boundary * (1.0 if closing else -1.0),
        boundary_speed=on_boundary * speed, boundary_slip=on_boundary * slip,
        boundary_dist_km=d,
        nearest_boundary=np.full(n, 0 if reached else -1, dtype=np.int32),
        uplift=np.broadcast_to(np.float32(uplift), (n,)).copy(),
        craton_margin_km=np.broadcast_to(np.float32(craton_margin_km), (n,)).copy(),
        iron_formation=np.full(n, iron_formation),
        precipitation_mm=np.full(n, precipitation_mm, dtype=np.float32),
        temperature_c=np.full(n, temperature_c, dtype=np.float32),
        flow_receiver=(np.minimum(np.arange(n) + 1, n - 1) if downstream
                       else np.arange(n)).astype(np.int32),
        slope=np.broadcast_to(np.float32(slope), (n,)).copy(),
        river_width_m=np.broadcast_to(np.float32(river), (n,)) * np.float32(5.0),
        cell_km=0.1, radius_km=R_WORLD_KM)


# A lode at the head of a river: orogenic gold on a transpressive, deeply eroded belt axis,
# and nothing in the country downstream of it, which lies well clear of the belt.
LODE = dict(uplift=95.0, speed=60.0, slip=60.0 * np.sqrt(0.5), downstream=True, river=True)
# What one cell passes to the next: the routed load fades over 100 km on Earth.
KEEP = np.exp(-0.1 / (100.0 / (6371.0 / R_WORLD_KM)))
COUNTRY_KM = 20.0


def test_a_placer_forms_where_the_river_below_a_lode_slackens():
    s = minerals.deposit_scores(**belt([0.0, COUNTRY_KM], slope=[0.2, 0.0], **LODE))
    lode = s["orogenic_gold"][0]
    assert lode > 0.9 and s["orogenic_gold"][1] < 1e-3
    # The flat cell holds its own barren ground and what the lode cell passed down.
    assert s["placer_gold"][1] == pytest.approx(KEEP * lode / (1.0 + KEEP), rel=1e-3)
    # Not at the head: nothing upstream is steeper than the lode's own reach.
    assert s["placer_gold"][0] == 0.0


def test_no_placer_on_a_reach_as_steep_as_the_river_above_it():
    """A river that keeps its gradient keeps its load. The gold has to be dropped."""
    s = minerals.deposit_scores(**belt([0.0, COUNTRY_KM, COUNTRY_KM], slope=0.2, **LODE))
    assert np.all(s["placer_gold"] == pytest.approx(0.0, abs=1e-6))


def test_placer_gold_fades_with_distance_below_the_lode():
    """Fine gold travels about 100 km on Earth, which is ten cells here."""
    near = [0.0] + [COUNTRY_KM] * 2
    far = [0.0] + [COUNTRY_KM] * 31
    flat_at_the_end = lambda n: [0.2] * (n - 1) + [0.0]
    just_below = minerals.deposit_scores(**belt(near, slope=flat_at_the_end(3), **LODE))
    far_below = minerals.deposit_scores(**belt(far, slope=flat_at_the_end(32), **LODE))
    assert just_below["placer_gold"][-1] > 0.1
    assert far_below["placer_gold"][-1] < 0.1 * just_below["placer_gold"][-1]


def test_no_placer_without_a_river_or_without_a_lode_upstream():
    off_river = minerals.deposit_scores(
        **belt([0.0, COUNTRY_KM], slope=[0.2, 0.0], **{**LODE, "river": False}))
    barren = minerals.deposit_scores(
        **belt([COUNTRY_KM, COUNTRY_KM], slope=[0.2, 0.0], **LODE))
    assert np.all(off_river["placer_gold"] == 0.0)
    assert np.all(barren["placer_gold"] < 1e-3)


def test_channel_iron_is_washed_off_enriched_iron_formation_into_a_slack_valley():
    """The detritus comes off a lateritised surface, so it needs the enrichment upstream:
    iron formation that the climate never weathered sheds none."""
    province = dict(uplift=2.0, reached=False, downstream=True, river=True,
                    slope=[0.2, 0.0], iron_formation=[True, False])
    tropical = minerals.deposit_scores(
        **belt([0.0, 0.0], precipitation_mm=1500.0, temperature_c=25.0, **province))
    arid = minerals.deposit_scores(
        **belt([0.0, 0.0], precipitation_mm=200.0, temperature_c=25.0, **province))
    source = tropical["enriched_iron"][0]
    assert source > 0.9
    assert tropical["channel_iron"][1] == pytest.approx(KEEP * source / (1.0 + KEEP), rel=1e-3)
    assert np.all(arid["channel_iron"] == 0.0)



def test_porphyry_sits_in_the_arc_band_not_on_the_boundary():
    s = minerals.deposit_scores(**belt([0.0, ARC_KM, FAR_KM]))
    on_boundary, in_arc, far = s["porphyry"]
    assert in_arc > on_boundary > far
    # Full closing rate, head-on, erosion at mid depth: the mid window peaks at one half.
    assert in_arc == pytest.approx(0.5)
    assert far < 0.01


def test_depth_of_erosion_sorts_what_a_head_on_arc_exposes():
    """Little erosion keeps the shallow epithermal level; more strips it to the porphyry
    level; more again strips that too."""
    shallow, mid, deep = (minerals.deposit_scores(**belt([0.0, ARC_KM], uplift=u))
                          for u in (5.0, 50.0, 95.0))
    arc = 1
    assert shallow["epithermal_hs"][arc] > shallow["porphyry"][arc]
    assert mid["porphyry"][arc] > mid["epithermal_hs"][arc]
    assert deep["porphyry"][arc] < 0.2 and deep["epithermal_hs"][arc] < 0.01


def test_an_oblique_arc_carries_the_iron_oxide_family_instead():
    """Where the plates slide as much as they close, the arc hosts low-sulfidation veins,
    iron oxide copper-gold and Kiruna iron, shallow to deep, in place of the head-on trio."""
    oblique = dict(speed=80.0, slip=80.0 * 0.9)  # mostly slip, still closing
    shallow, mid, deep = (minerals.deposit_scores(**belt([0.0, ARC_KM], uplift=u, **oblique))
                          for u in (5.0, 50.0, 95.0))
    arc = 1
    assert shallow["epithermal_ls"][arc] > shallow["epithermal_hs"][arc]
    assert mid["iocg"][arc] > mid["porphyry"][arc]
    assert deep["kiruna_iron"][arc] > deep["iocg"][arc] > 0.0
    # And none of it without the slip.
    head_on = minerals.deposit_scores(**belt([0.0, ARC_KM], uplift=95.0))
    assert head_on["kiruna_iron"][arc] == 0.0


def test_orogenic_gold_follows_the_axis_of_a_deeply_eroded_transpressive_belt():
    at_45_deg = dict(speed=60.0, slip=60.0 * np.sqrt(0.5))
    s = minerals.deposit_scores(**belt([0.0, ARC_KM, FAR_KM], uplift=95.0, **at_45_deg))
    on_boundary, in_arc, far = s["orogenic_gold"]
    assert on_boundary > in_arc > far
    assert on_boundary == pytest.approx(0.95 ** 2, rel=1e-3)  # full rate, 45 deg, deep


def test_orogenic_gold_needs_both_closing_and_slip():
    head_on = minerals.deposit_scores(**belt([0.0], uplift=95.0))
    sliding = minerals.deposit_scores(**belt([0.0], uplift=95.0, speed=60.0, slip=60.0))
    assert head_on["orogenic_gold"][0] == 0.0
    assert sliding["orogenic_gold"][0] == pytest.approx(0.0, abs=1e-6)


def test_a_continental_rift_carries_sediment_copper_and_nothing_from_the_arc():
    s = minerals.deposit_scores(**belt([0.0, 0.4, FAR_KM], closing=False,
                                       craton_margin_km=0.0))
    axis, margin, far = s["sediment_copper"]
    assert axis > margin > far and axis == pytest.approx(1.0)
    for name in ("porphyry", "epithermal_hs", "epithermal_ls", "iocg", "kiruna_iron",
                 "orogenic_gold"):
        assert np.all(s[name] == 0.0), f"{name} scored on a rift"
    # And a closing boundary carries none of the rift's.
    closing = belt([0.0, 0.4], craton_margin_km=0.0)
    assert np.all(minerals.deposit_scores(**closing)["sediment_copper"] == 0.0)


def test_a_rift_far_from_any_craton_keeps_only_a_small_share_of_its_copper():
    """85% of sediment-hosted base metal lies within 200 km of the step between thick and
    thin lithosphere. A rift with no craton edge near it keeps the other 15%."""
    on_edge = minerals.deposit_scores(**belt([0.0], closing=False, craton_margin_km=0.0))
    far = minerals.deposit_scores(**belt([0.0], closing=False, craton_margin_km=1e3))
    assert on_edge["sediment_copper"][0] == pytest.approx(1.0)
    assert far["sediment_copper"][0] == pytest.approx(0.15)


def test_iron_formation_scores_in_quiet_ground_whether_or_not_a_boundary_is_near():
    """It is the one type that owes nothing to the plates: an old province inside a craton,
    scored wherever the ground has not since been thrown up into an orogen."""
    quiet = belt([0.0, FAR_KM], uplift=2.0, iron_formation=True, reached=False, speed=0.0)
    s = minerals.deposit_scores(**quiet)
    assert np.all(s["bif_iron"] == pytest.approx(0.98 ** 2))
    orogen = minerals.deposit_scores(**belt([0.0], uplift=100.0, iron_formation=True))
    assert orogen["bif_iron"][0] == 0.0
    assert np.all(minerals.deposit_scores(**belt([0.0, FAR_KM]))["bif_iron"] == 0.0)


def test_supergene_copper_needs_some_rain_and_not_much():
    """Weathering enriches a porphyry where there is water to leach the copper downward and
    not so much that the ground erodes out from under it. Below 120 mm/yr there is too
    little; it is an upgrade to a porphyry, never a source of its own."""
    at_arc = lambda mm: {k: v[1] for k, v in minerals.deposit_scores(
        **belt([0.0, ARC_KM], precipitation_mm=mm)).items()}
    semi_arid, hyper_arid, wet = at_arc(300.0), at_arc(60.0), at_arc(2000.0)
    assert semi_arid["supergene_copper"] == pytest.approx(semi_arid["porphyry"])
    assert hyper_arid["supergene_copper"] == pytest.approx(0.5 * hyper_arid["porphyry"])
    assert wet["supergene_copper"] == 0.0
    # The porphyry itself does not care what the weather is.
    assert semi_arid["porphyry"] == hyper_arid["porphyry"] == wet["porphyry"] > 0
    no_arc = minerals.deposit_scores(**belt([FAR_KM], precipitation_mm=300.0))
    assert no_arc["supergene_copper"][0] < 0.01


def test_iron_formation_is_enriched_to_ore_only_where_it_is_warm_and_wet():
    province = dict(uplift=2.0, iron_formation=True, reached=False)
    score = lambda mm, c: {k: v[0] for k, v in minerals.deposit_scores(
        **belt([0.0], precipitation_mm=mm, temperature_c=c, **province)).items()}
    tropical, dry, cold = score(1500.0, 25.0), score(300.0, 25.0), score(1500.0, 5.0)
    assert tropical["enriched_iron"] == pytest.approx(tropical["bif_iron"])
    assert dry["enriched_iron"] == 0.0 and cold["enriched_iron"] == 0.0
    # The formation is there whatever the weather; only the upgrade is climatic.
    assert tropical["bif_iron"] == dry["bif_iron"] == cold["bif_iron"] > 0.9
    off = minerals.deposit_scores(**belt([0.0], precipitation_mm=1500.0, uplift=2.0))
    assert off["enriched_iron"][0] == 0.0


def test_a_wet_rift_keeps_only_a_share_of_its_copper():
    """The ore needs an oxidised brine and the sulphur of evaporites, which is a dry
    climate. It is not a hard gate: one giant deposit has no evaporites at all."""
    rift = dict(closing=False, craton_margin_km=0.0)
    dry = minerals.deposit_scores(**belt([0.0], precipitation_mm=200.0, **rift))
    wet = minerals.deposit_scores(**belt([0.0], precipitation_mm=2000.0, **rift))
    cold_dry = minerals.deposit_scores(
        **belt([0.0], precipitation_mm=200.0, temperature_c=-5.0, **rift))
    assert dry["sediment_copper"][0] == pytest.approx(1.0)
    assert wet["sediment_copper"][0] == pytest.approx(0.25)
    # Dryness alone: a cold desert makes evaporites too.
    assert cold_dry["sediment_copper"][0] == pytest.approx(1.0)


def test_the_craton_term_fades_over_twice_the_margin_length():
    """200 km on Earth, which is 2.04 km on this world."""
    reach_km = 200.0 / (6371.0 / R_WORLD_KM)
    s = minerals.deposit_scores(**belt([0.0, 0.0], closing=False,
                                       craton_margin_km=[reach_km, 20 * reach_km]))
    one_length_off, far = s["sediment_copper"]
    assert one_length_off == pytest.approx(0.15 + 0.85 * np.exp(-1.0), rel=1e-3)
    assert far == pytest.approx(0.15)


@pytest.mark.parametrize("off", [dict(continental=False), dict(land=False)])
def test_nothing_scores_off_continental_land(off):
    world = belt([0.0, ARC_KM], craton_margin_km=0.0, iron_formation=True, **off)
    for closing in (True, False):
        world["boundary_stress"] = np.abs(world["boundary_stress"]) * (1 if closing else -1)
        for name, score in minerals.deposit_scores(**world).items():
            assert np.all(score == 0.0), f"{name} scored with {off}"


@pytest.mark.parametrize("off", [dict(reached=False), dict(speed=0.0)])
def test_plate_types_need_a_boundary_in_reach_and_motion_on_it(off):
    for closing in (True, False):
        s = minerals.deposit_scores(**belt([0.0, ARC_KM], closing=closing, **off))
        for name in ("porphyry", "epithermal_hs", "epithermal_ls", "iocg", "kiruna_iron",
                     "orogenic_gold", "sediment_copper"):
            assert np.all(s[name] == 0.0), f"{name} scored with {off}"


def test_resources_are_the_weighted_sum_of_the_deposit_types():
    world = belt([0.0, 0.4, ARC_KM, FAR_KM], speed=60.0, slip=30.0)
    p = minerals.MineralParams()
    s = minerals.deposit_scores(**world)
    r = minerals.resources(**world)
    assert set(r) == {"copper", "iron", "gold"}
    for name, weights in (("copper", p.copper), ("iron", p.iron), ("gold", p.gold)):
        assert np.allclose(r[name], sum(w * s[t] for t, w in weights.items()))
        assert r[name].dtype == np.float32
        assert sum(weights.values()) <= 1.0 + 1e-9, "a field could exceed 1"


@pytest.fixture(scope="module")
def world():
    g = build_goldberg(72)
    return g, stage1.generate_crust_tectonic(g, seed=1)


def scored(world, fn):
    """`fn` over a Stage 1 world, under one mild climate everywhere: 300 mm and 20 C.
    Stage 1 has no climate of its own, and these tests are about the plates and cratons.
    Rivers are routed on its uneroded surface and drawn wherever 20 cells drain."""
    g, c = world
    pl = c.plates
    inv = minerals.invent_cratons(g, 1, c.is_continental)
    fr = flow.route(c.elevation, g.neighbours, g.arc_dist, g.area, ~c.land_mask)
    river = c.land_mask & (fr.acc > 20 * np.median(g.area))
    return fn(
        flow_receiver=fr.rec, slope=fr.slope, river_width_m=river.astype(np.float32),
        cell_km=float(np.mean(g.arc_dist[g.neighbours >= 0])),
        land_mask=c.land_mask, is_continental=c.is_continental,
        boundary_stress=pl.boundary_stress, boundary_speed=pl.boundary_speed,
        boundary_slip=pl.boundary_slip, boundary_dist_km=pl.boundary_dist_km,
        nearest_boundary=pl.nearest_boundary, uplift=c.uplift, radius_km=g.radius_km,
        craton_margin_km=inv["craton_margin_km"], iron_formation=inv["iron_formation"],
        precipitation_mm=np.full(g.cell_count, 300.0, dtype=np.float32),
        temperature_c=np.full(g.cell_count, 20.0, dtype=np.float32))


def test_invented_cratons_take_their_set_share_of_continental_crust(world):
    g, c = world
    p = minerals.MineralParams()
    inv = minerals.invent_cratons(g, 1, c.is_continental)
    craton, iron = inv["craton"], inv["iron_formation"]
    assert not craton[~c.is_continental].any(), "a craton on oceanic crust"
    share = g.area[craton].sum() / g.area[c.is_continental].sum()
    assert share == pytest.approx(p.craton_fraction, abs=0.02)
    assert not iron[~craton].any(), "iron formation outside a craton"
    assert g.area[iron].sum() / g.area[craton].sum() == pytest.approx(
        p.iron_formation_fraction, abs=0.02)


def test_craton_margin_distance_is_zero_on_the_margin_and_grows_either_side(world):
    g, c = world
    inv = minerals.invent_cratons(g, 1, c.is_continental)
    craton, d = inv["craton"], inv["craton_margin_km"]
    on_margin = np.zeros(g.cell_count, dtype=bool)
    for s in range(g.neighbours.shape[1]):
        j = g.neighbours[:, s]
        ok = j >= 0
        on_margin[ok] |= craton[ok] != craton[j[ok]]
    assert on_margin.any()
    assert np.all(d[on_margin] == 0.0) and d[~on_margin].min() > 0.0
    # Both sides: the margin band reaches into the craton and out of it.
    spacing = float(np.median(g.arc_dist[g.neighbours >= 0]))
    near = (d > 0) & (d < 3 * spacing)
    assert (near & craton).any() and (near & ~craton).any()


def test_cratons_are_fixed_by_the_seed_and_drawn_apart_from_the_crust(world):
    g, c = world
    a = minerals.invent_cratons(g, 1, c.is_continental)
    b = minerals.invent_cratons(g, 1, c.is_continental)
    other = minerals.invent_cratons(g, 2, c.is_continental)
    assert np.array_equal(a["craton"], b["craton"])
    assert not np.array_equal(a["craton"], other["craton"])


def test_a_generated_world_has_all_three_resources(world):
    g, c = world
    r = scored(world, minerals.resources)
    for name, field in r.items():
        assert field.shape == (g.cell_count,)
        assert np.isfinite(field).all() and field.min() >= 0.0 and field.max() <= 1.0
        assert np.all(field[~c.land_mask] == 0.0), f"{name} scored at sea"
        # Present, not merely non-zero. How high a field peaks is set by the tuned weights
        # and by which types this fixture's flat 300 mm climate lets score, so the bar is
        # deliberately low: it is here to catch a resource going missing.
        assert field.max() > 0.05, f"no {name} anywhere"


def test_weights_for_shares_make_each_type_carry_its_share_by_area():
    """A sim samples in proportion to score times AREA, so a weight is not a share. A type
    that scores over a great deal of ground outweighs one that scores on a few river
    reaches, whatever the two are weighted. A copper-gold ring drawn along every craton
    margin carried 67 to 72% of the copper at a weight of 0.07, and was removed for it.
    The weights that give each type its share are its share over the ground it covers."""
    mass = {"broad": 100.0, "belt": 10.0, "reach": 0.5}  # score summed over area
    shares = {"broad": 0.5, "belt": 0.3, "reach": 0.2}
    w = minerals.weights_for_shares(mass, shares)
    assert sum(w.values()) == pytest.approx(1.0)
    carried = {t: w[t] * mass[t] for t in w}
    total = sum(carried.values())
    for t in shares:
        assert carried[t] / total == pytest.approx(shares[t])
    # The thinnest type needs by far the largest weight to carry its share.
    assert w["reach"] > w["belt"] > w["broad"]


def test_resource_fields_ship_as_bytes(tmp_path):
    """Three fields at a byte each, not three floats: the archive is already large."""
    g = build_goldberg(24)
    land = g.lat > 0
    score = np.linspace(0.0, 1.0, g.cell_count).astype(np.float32)
    ex = export.export_world(tmp_path, g, np.zeros(g.cell_count, np.float32), land, seed=1,
                             resource_copper=score, resource_iron=score * 0.5,
                             resource_gold=score[::-1])
    _, fields, _ = export.load_native(ex.files["native"], grid=g)
    for name in ("resource_copper", "resource_iron", "resource_gold"):
        assert fields[name].dtype == np.uint8
    assert np.array_equal(fields["resource_copper"], np.round(score * 255))
    assert fields["resource_copper"].max() == 255 and fields["resource_iron"].max() == 128
