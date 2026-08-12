"""Stage 9: export.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import biomes, coupled, export, flow, hydrology, stage1  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    g = build_goldberg(96)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=2, steps_per_pass=150)
    fr = flow.route(res.elevation, g.neighbours, g.arc_dist, g.area, ~crust.land_mask)
    b = biomes.classify(res.climate.temperature_c, res.climate.precipitation_mm,
                        crust.land_mask, res.elevation, g.lat)
    # This fixture tests the EXPORT PLUMBING and needs a lake to exist to test it with,
    # so one is carved rather than hoped for. At Earth radius the hope was safe: the
    # 400 km^2 minimum was under a single cell at n=96, so any one-cell dimple counted.
    # At 100 m cells the minimum is ~5 real cells, closed basins are rare (0.03% of land),
    # and hillslope diffusion now actively fills the shallow ones -- so a 792 km^2 fixture
    # world often has none at all. Lake GENESIS is covered in tests/test_hydrology.py.
    z = res.elevation.copy()
    interior = crust.land_mask.copy()
    for s in range(g.neighbours.shape[1]):
        j = g.neighbours[:, s]
        interior &= (j >= 0) & crust.land_mask[np.maximum(j, 0)]
    pit = int(np.flatnonzero(interior)[0])
    z[pit] = z[g.neighbours[pit][g.neighbours[pit] >= 0]].min() - 60.0

    hyd = hydrology.analyse(g, z, ~crust.land_mask,
                            res.climate.precipitation_mm, res.climate.temperature_c,
                            params=hydrology.HydrologyParams(min_lake_area_km2=0.0))
    out = tmp_path_factory.mktemp("world")
    ex = export.export_world(
        out, g, res.elevation, crust.land_mask, seed=1, slope=fr.slope,
        drainage_km2=fr.acc, temperature_c=res.climate.temperature_c,
        precipitation_mm=res.climate.precipitation_mm, biome=b,
        flow_receiver=fr.rec, discharge=hyd.discharge,
        river_width_m=hyd.river_width_m, lake_mask=hyd.lake_mask,
        lake_level_m=hyd.lake_level_m, lake_depth_m=hyd.lake_depth_m,
        lake_id=hyd.lake_id, endorheic=hyd.endorheic, lakes=hyd.lakes,
        params={"note": "test"}, metrics_summary={"theta": 0.45})
    return g, crust, res, b, ex, hyd


# --------------------------------------------------------------------------------------
# Native archive
# --------------------------------------------------------------------------------------


def test_the_archive_is_the_only_artifact(world):
    """The export writes the mesh and its provenance, and nothing else.

    It used to also write three equirectangular GeoTIFFs. Nothing consumed them: the
    renderer now samples the mesh directly, and the metrics that judge the world run on the
    mesh graph. The one claim that justified them -- one metrics implementation for
    generated and real terrain -- was already conceded in `verify_roundtrip`'s own
    docstring, which had to state that its numbers were not SRTM-comparable.
    """
    g, crust, res, b, ex, hyd = world
    assert set(ex.files) == {"native", "manifest"}
    assert not list(pathlib.Path(ex.directory).glob("*.tif"))


def test_native_archive_round_trips_exactly(world, tmp_path):
    g, crust, res, b, ex, hyd = world
    grid2, fields, meta = export.load_native(ex.files["native"])
    assert meta["n"] == g.n and meta["seed"] == 1
    assert grid2.cell_count == g.cell_count
    assert np.array_equal(fields["elevation"], res.elevation)
    assert np.array_equal(fields["biome"], b)


def test_hydrology_reaches_the_archive(world):
    """Stage 6 used to be computed, printed, and thrown away -- the rivers never left the
    process that made them. Discharge is not recoverable from drainage area alone: the two
    differ by the precipitation field, which is the whole point of the coupled stage."""
    g, crust, res, b, ex, hyd = world
    _, fields, _ = export.load_native(ex.files["native"])

    assert np.array_equal(fields["discharge_km3_per_yr"], hyd.discharge)
    assert np.array_equal(fields["river_width_m"], hyd.river_width_m)

    # Width in metres is the whole encoding: zero means no river, and there is no boolean
    # companion that could drift out of step with it.
    assert "river_mask" not in fields
    rivers = fields["river_width_m"] > 0
    assert rivers.any(), "no rivers survived the export"
    assert (fields["river_width_m"][rivers] > 0.6).all(), "channel threshold not respected"


def test_the_flow_tree_survives_as_cell_ids_and_stays_edge_derivable(world):
    """The archive carries the receiver as a CELL id, and the exit EDGE is derived from it.
    That only works if every receiver is a neighbour -- so it is asserted rather than
    assumed, since a consumer drawing rivers has no way to recover from it being false."""
    g, crust, res, b, ex, hyd = world
    _, fields, _ = export.load_native(ex.files["native"])
    rec = fields["flow_receiver"]

    assert rec.dtype == np.int32
    terminal = rec == np.arange(rec.size)
    assert terminal.any(), "nothing terminates the tree"

    # Exactly one exit slot per flowing cell, none at a terminator. This is the derivation
    # README documents; if it stops holding, the documented recipe silently returns nothing.
    hit = (g.neighbours == rec[:, None]) & (g.neighbours >= 0)
    assert (hit.sum(axis=1) == ~terminal).all(), "a receiver is not a neighbour"

    # And the tree terminates: follow the ids down and you reach a sink, never a cycle.
    walk = np.flatnonzero(~terminal)[::97]
    for start in walk:
        i, steps = int(start), 0
        while rec[i] != i:
            i, steps = int(rec[i]), steps + 1
            assert steps < rec.size, f"flow from {start} does not terminate"


def test_lakes_reach_the_archive(world):
    g, crust, res, b, ex, hyd = world
    _, fields, _ = export.load_native(ex.files["native"])

    assert np.array_equal(fields["lake_mask"].astype(bool), hyd.lake_mask)
    assert np.array_equal(fields["lake_depth_m"], hyd.lake_depth_m)
    assert np.array_equal(fields["lake_id"], hyd.lake_id)
    assert np.array_equal(fields["endorheic"].astype(bool), hyd.endorheic)
    # lake_level_m is NaN off-lake by design, so compare it where it means something.
    wet = hyd.lake_mask
    assert np.array_equal(fields["lake_level_m"][wet], hyd.lake_level_m[wet])
    assert np.isnan(fields["lake_level_m"][~wet]).all()

    assert wet.any(), "no lakes survived the export"
    assert (fields["lake_depth_m"][wet] > 0).all()
    assert (fields["lake_depth_m"][~wet] == 0).all()


def test_lake_ids_index_the_manifest_table(world):
    """field_lake_id is a foreign key. If the table it points into is not written, the
    per-lake quantities -- spill point, inflow, whether it is terminal -- are gone."""
    g, crust, res, b, ex, hyd = world
    manifest = json.loads(pathlib.Path(ex.files["manifest"]).read_text())
    _, fields, _ = export.load_native(ex.files["native"])

    table = manifest["lakes"]
    assert len(table) == len(hyd.lakes)
    ids = fields["lake_id"]
    assert set(np.unique(ids).tolist()) - {-1} == set(range(len(table)))
    assert (ids[~fields["lake_mask"].astype(bool)] == -1).all()

    for row in table:
        cells = ids == row["id"]
        assert cells.sum() == row["cells"]
        assert row["endorheic"] == bool(fields["endorheic"][cells].all())


def test_native_archive_rebuilds_the_grid_rather_than_storing_it(world):
    """Geometry is regenerable from n, so it must not be paying to store it: the cached
    grid at n=785 alone is 650 MB."""
    g, crust, res, b, ex, hyd = world
    size = pathlib.Path(ex.files["native"]).stat().st_size
    per_cell = size / g.cell_count
    assert per_cell < 40, f"{per_cell:.0f} bytes/cell suggests geometry is being stored"
    grid2, _, _ = export.load_native(ex.files["native"])
    assert np.allclose(grid2.xyz, g.xyz)


def test_manifest_records_provenance(world):
    g, crust, res, b, ex, hyd = world
    m = json.loads(pathlib.Path(ex.files["manifest"]).read_text())
    assert m["seed"] == 1
    assert m["grid"]["n"] == g.n
    assert m["params"]["note"] == "test"
    assert m["metrics"]["theta"] == 0.45
    assert 0.0 < m["land_fraction"] < 1.0


def test_the_renderer_can_rebuild_its_raster_from_the_archive(world):
    """render_world.py is now the only consumer, and it samples the mesh itself. If the
    archive cannot produce a raster, the world cannot be looked at."""
    from terrain import amplify
    g, crust, res, b, ex, hyd = world
    _, fields, _ = export.load_native(ex.files["native"], grid=g)

    pts, shape = amplify.equirect_points(256)
    nearest, _ = amplify.neighbour_weights(g, pts, k=1)
    z = np.asarray(fields["elevation"])[nearest].reshape(shape)
    codes = set(np.unique(np.asarray(fields["biome"])[nearest]).tolist())

    assert z.shape == (128, 256)
    assert np.isfinite(z).all()
    # Nearest sampling manufactures nothing: every pixel is some cell's own value.
    assert set(np.unique(z).tolist()).issubset(set(np.unique(res.elevation).tolist()))
    assert codes.issubset(set(biomes.NAMES)), f"invented biome codes: {codes}"
