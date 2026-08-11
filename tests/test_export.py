"""Stage 9: export.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import biomes, coupled, dem, export, flow, hydrology, stage1  # noqa: E402
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
    hyd = hydrology.analyse(g, res.elevation, ~crust.land_mask,
                            res.climate.precipitation_mm, res.climate.temperature_c)
    out = tmp_path_factory.mktemp("world")
    ex = export.export_world(
        out, g, res.elevation, crust.land_mask, seed=1, slope=fr.slope,
        drainage_km2=fr.acc, temperature_c=res.climate.temperature_c,
        precipitation_mm=res.climate.precipitation_mm, biome=b,
        discharge=hyd.discharge, river_mask=hyd.river_mask,
        river_width_m=hyd.river_width_m, lake_mask=hyd.lake_mask,
        lake_level_m=hyd.lake_level_m, lake_depth_m=hyd.lake_depth_m,
        lake_id=hyd.lake_id, endorheic=hyd.endorheic, lakes=hyd.lakes, width=512,
        params={"note": "test"}, metrics_summary={"theta": 0.45})
    return g, crust, res, b, ex, hyd


# --------------------------------------------------------------------------------------
# GeoTIFF
# --------------------------------------------------------------------------------------


def test_geotiff_is_georeferenced_equirectangular(world):
    import rasterio
    g, crust, res, b, ex, hyd = world
    with rasterio.open(ex.files["elevation"]) as ds:
        assert ds.crs.to_string() == "EPSG:4326"
        assert (ds.width, ds.height) == (512, 256)
        assert ds.bounds.left == pytest.approx(-180.0)
        assert ds.bounds.right == pytest.approx(180.0)
        assert ds.bounds.top == pytest.approx(90.0)
        assert ds.bounds.bottom == pytest.approx(-90.0)
        assert ds.dtypes[0] == "float32"


def test_climate_bands_are_labelled(world):
    import rasterio
    g, crust, res, b, ex, hyd = world
    with rasterio.open(ex.files["climate"]) as ds:
        assert ds.count == 2
        assert ds.descriptions == ("temperature_c", "precipitation_mm")
        t = ds.read(1)
        assert -60 < t.min() and t.max() < 60


def test_biome_raster_preserves_codes(world):
    """Categorical data must be sampled nearest, never interpolated -- the average of
    'tundra' and 'savanna' is not a biome."""
    import rasterio
    g, crust, res, b, ex, hyd = world
    with rasterio.open(ex.files["biomes"]) as ds:
        assert ds.dtypes[0] == "uint8"
        codes = set(np.unique(ds.read(1)).tolist())
    assert codes.issubset(set(biomes.NAMES)), f"invented biome codes: {codes}"
    assert codes.issubset(set(np.unique(b).tolist()))


def test_raster_carries_the_world_faithfully(world):
    """The export's own correctness test: pixels against the mesh field at the same coords.

    Error is non-zero by design -- Stage 8 adds sub-grid detail on the way out -- but it
    must stay small against the relief, and unbiased enough not to shift sea level.
    """
    g, crust, res, b, ex, hyd = world
    f = export.verify_fidelity(ex.files["elevation"], g, res.elevation, samples=40_000)
    assert f["mean_abs_error_m"] < 0.02 * f["relief_m"]
    assert f["p99_abs_error_m"] < 0.05 * f["relief_m"]
    assert abs(f["bias_m"]) < 30.0


# --------------------------------------------------------------------------------------
# Native archive
# --------------------------------------------------------------------------------------


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
    assert np.array_equal(fields["river_mask"].astype(bool), hyd.river_mask)
    assert np.array_equal(fields["river_width_m"], hyd.river_width_m)

    rivers = fields["river_mask"].astype(bool)
    assert rivers.any(), "no rivers survived the export"
    assert (fields["river_width_m"][~rivers] == 0).all()
    assert (fields["river_width_m"][rivers] > 0).all()


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
    assert m["raster"]["crs"] == "EPSG:4326"
    assert m["params"]["note"] == "test"
    assert m["metrics"]["theta"] == 0.45
    assert 0.0 < m["land_fraction"] < 1.0


# --------------------------------------------------------------------------------------
# The claim that motivated choosing GeoTIFF
# --------------------------------------------------------------------------------------


def test_the_metrics_module_reads_the_export_unmodified(world):
    """GeoTIFF was chosen over .npy so one metrics implementation serves generated and real
    terrain. This exercises that path rather than assuming it.

    The numbers are NOT expected to match the SRTM references -- the export is ~5 km per
    pixel against SRTM's 30 m, so no window is both wide enough for three decades of
    drainage area and small enough to avoid serious equirectangular distortion. What is
    asserted is that it loads, routes and measures without a code change.
    """
    g, crust, res, b, ex, hyd = world
    # The channel threshold has to suit the raster. The production default assumes an 8192
    # export; on a 512-pixel test raster it leaves too few channel cells to fit a slope-area
    # regression at all, and theta comes back nan.
    import rasterio
    with rasterio.open(ex.files["elevation"]) as ds:
        cell_km2 = (360.0 / ds.width) * (180.0 / ds.height) * 111.0 * 111.0
    r = export.verify_roundtrip(ex.files["elevation"], window=128,
                                channel_threshold_km2=cell_km2 * 8)
    assert r["cells"] == 128 * 128
    assert r["channel_cells"] > 0
    assert np.isfinite(r["concavity_theta"]), "the regression should at least be fittable"
    assert np.isfinite(r["hack_h"])


def test_sea_must_be_declared_as_base_level(world):
    """Without sink_below, a raster carrying bathymetry is treated as one land surface and
    rivers incise across the abyssal plain -- measured concavity of -0.081, i.e. slope
    rising with drainage area."""
    g, crust, res, b, ex, hyd = world
    row0, col0 = export.best_land_window(ex.files["elevation"], 128)
    with_sea = dem.load_window(ex.files["elevation"], size=128, row0=row0, col0=col0,
                               sink_below=0.0)
    without = dem.load_window(ex.files["elevation"], size=128, row0=row0, col0=col0)
    assert with_sea.is_sink.sum() > without.is_sink.sum()
    assert with_sea.is_sink[with_sea.z <= 0].all()


def test_window_selection_avoids_the_poles(world):
    """Equirectangular cells collapse to zero width at the poles, so a window reaching one
    is measuring a coordinate singularity rather than terrain."""
    import rasterio
    g, crust, res, b, ex, hyd = world
    size = 128
    row0, col0 = export.best_land_window(ex.files["elevation"], size, max_abs_lat=55.0)
    with rasterio.open(ex.files["elevation"]) as ds:
        dlat = (ds.bounds.top - ds.bounds.bottom) / ds.height
        lat_hi = ds.bounds.top - row0 * dlat
        lat_lo = ds.bounds.top - (row0 + size) * dlat
    assert max(abs(lat_hi), abs(lat_lo)) <= 56.0
