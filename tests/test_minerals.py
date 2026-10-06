"""Mineral prospectivity: deposit-type scores and the three resource fields.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import export, minerals, plates, stage1  # noqa: E402
from terrain.grid import R_WORLD_KM, build_goldberg  # noqa: E402

# The authored lengths, which is what a production-radius world sees.
ARC_KM = plates.PlateParams().arc_offset_km
FAR_KM = 6.0  # several arc widths and decay lengths out


def belt(dist_km, *, speed=40.0, slip=0.0, closing=True, uplift=50.0,
         continental=True, land=True, reached=True):
    """Cells at the given distances from ONE boundary cell, which is cell 0.

    Every cell takes its plate motion from that boundary cell, exactly as a world's cells
    take theirs through `nearest_boundary`. Defaults: closing head-on at 40 mm/yr, which is
    full rate, with uplift at half its ceiling.
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
        uplift=np.broadcast_to(np.float32(uplift), (n,)).copy(), radius_km=R_WORLD_KM)


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
    s = minerals.deposit_scores(**belt([0.0, 0.4, FAR_KM], closing=False))
    axis, margin, far = s["sediment_copper"]
    assert axis > margin > far and axis == pytest.approx(1.0)
    for name in ("porphyry", "epithermal_hs", "epithermal_ls", "iocg", "kiruna_iron",
                 "orogenic_gold"):
        assert np.all(s[name] == 0.0), f"{name} scored on a rift"
    # And a closing boundary carries none of the rift's.
    assert np.all(minerals.deposit_scores(**belt([0.0, 0.4]))["sediment_copper"] == 0.0)


@pytest.mark.parametrize("off", [dict(continental=False), dict(land=False),
                                 dict(reached=False), dict(speed=0.0)])
def test_nothing_scores_without_continental_land_a_boundary_in_reach_and_motion(off):
    for closing in (True, False):
        s = minerals.deposit_scores(**belt([0.0, ARC_KM], closing=closing, **off))
        for name, score in s.items():
            assert np.all(score == 0.0), f"{name} scored with {off}"


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


def test_a_generated_world_has_copper_and_gold_on_its_convergent_belts():
    g = build_goldberg(72)
    c = stage1.generate_crust_tectonic(g, seed=1)
    pl = c.plates
    r = minerals.resources(
        land_mask=c.land_mask, is_continental=c.is_continental,
        boundary_stress=pl.boundary_stress, boundary_speed=pl.boundary_speed,
        boundary_slip=pl.boundary_slip, boundary_dist_km=pl.boundary_dist_km,
        nearest_boundary=pl.nearest_boundary, uplift=c.uplift, radius_km=g.radius_km)
    for name, field in r.items():
        assert field.shape == (g.cell_count,)
        assert np.isfinite(field).all() and field.min() >= 0.0 and field.max() <= 1.0
        assert np.all(field[~c.land_mask] == 0.0), f"{name} scored at sea"
    assert r["copper"].max() > 0.2 and r["gold"].max() > 0.1
    # Iron has only its Kiruna share until the invented cratons exist.
    assert 0.0 < r["iron"].max() <= minerals.MineralParams().iron["kiruna_iron"]


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
