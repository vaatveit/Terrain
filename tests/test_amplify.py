"""Stage 8: sub-grid detail and mesh-to-raster sampling.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from terrain import amplify, coupled, flow, metrics, stage1  # noqa: E402
from terrain.grid import build_goldberg  # noqa: E402


@pytest.fixture(scope="module")
def world():
    g = build_goldberg(128)
    crust = stage1.generate_crust_tectonic(g, seed=1)
    res = coupled.run(g, crust.elevation, crust.uplift, ~crust.land_mask,
                      passes=2, steps_per_pass=200)
    fr = flow.route(res.elevation, g.neighbours, g.arc_dist, g.area, ~crust.land_mask)
    return g, crust, res, fr


# --------------------------------------------------------------------------------------
# Amplitude modulation
# --------------------------------------------------------------------------------------


def test_detail_is_strongest_on_steep_ground(world):
    g, crust, res, fr = world
    land = crust.land_mask
    a = amplify.detail_amplitude(fr.slope, fr.acc, land)
    steep = land & (fr.slope > np.percentile(fr.slope[land], 90))
    flat = land & (fr.slope < np.percentile(fr.slope[land], 25))
    assert a[steep].mean() > 2.0 * a[flat].mean()


def test_trunk_valleys_are_smoothed(world):
    """Alluvial floors are smooth because sediment fills them."""
    g, crust, res, fr = world
    land = crust.land_mask
    a = amplify.detail_amplitude(fr.slope, fr.acc, land)
    valleys = land & (fr.acc > np.percentile(fr.acc[land], 98))
    elsewhere = land & (fr.acc < np.percentile(fr.acc[land], 50))
    assert a[valleys].mean() < 0.2 * a[elsewhere].mean()


def test_flat_ground_keeps_some_texture(world):
    """Without a floor on the slope term, amplitude hits exactly zero on plains, on valley
    floors and across the whole sea bed -- base level is flat, so its slope is zero."""
    g, crust, res, fr = world
    land = crust.land_mask
    a = amplify.detail_amplitude(fr.slope, fr.acc, land)
    flat = land & (fr.slope < np.percentile(fr.slope[land], 10))
    assert a[flat].mean() > 5.0, "plains should not be billiard tables"
    assert a[~land].mean() > 0.0, "the sea bed has slope zero but should not be glass"


def test_amplitude_never_exceeds_its_cap():
    p = amplify.AmplifyParams()
    a = amplify.detail_amplitude(np.full(50, 10.0), np.zeros(50),
                                 np.ones(50, dtype=bool), p)
    assert a.max() <= p.max_amplitude_m + 1e-9


def test_ocean_detail_is_muted(world):
    g, crust, res, fr = world
    p = amplify.AmplifyParams()
    a_land = amplify.detail_amplitude(np.full(10, 0.05), np.zeros(10),
                                      np.ones(10, dtype=bool), p)
    a_sea = amplify.detail_amplitude(np.full(10, 0.05), np.zeros(10),
                                     np.zeros(10, dtype=bool), p)
    assert a_sea.mean() == pytest.approx(a_land.mean() * p.ocean_scale)


# --------------------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------------------


def _high_freq(x):
    lap = 4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:]
    return float(np.abs(lap).mean())


def test_interpolation_removes_hex_blockiness(world):
    """Nearest-cell sampling renders the mesh cells themselves, which reads as a bug."""
    g, crust, res, fr = world
    args = (g, res.elevation, fr.slope, fr.acc, crust.land_mask)
    nearest, _ = amplify.amplified_raster(
        *args, width=1024, detail=False, p=amplify.AmplifyParams(neighbours=1))
    smooth, _ = amplify.amplified_raster(*args, width=1024, detail=False)
    assert _high_freq(smooth) < 0.5 * _high_freq(nearest)


def test_interpolation_preserves_the_large_scale_field(world):
    g, crust, res, fr = world
    raw, land = amplify.amplified_raster(g, res.elevation, fr.slope, fr.acc,
                                         crust.land_mask, width=1024, detail=False)
    assert raw.min() >= res.elevation.min() - 1.0
    assert raw.max() <= res.elevation.max() + 1.0


def test_detail_perturbs_but_does_not_replace_the_base(world):
    g, crust, res, fr = world
    args = (g, res.elevation, fr.slope, fr.acc, crust.land_mask)
    raw, land = amplify.amplified_raster(*args, width=1024, detail=False)
    amp, _ = amplify.amplified_raster(*args, width=1024, detail=True)
    d = np.abs(amp - raw)
    relief = float(raw[land].max() - raw[land].min())
    assert d.mean() > 0.5, "detail should actually do something"
    assert d.max() < 0.1 * relief, "detail must be texture, not new landforms"


def test_raster_is_reproducible(world):
    g, crust, res, fr = world
    args = (g, res.elevation, fr.slope, fr.acc, crust.land_mask)
    a, _ = amplify.amplified_raster(*args, width=512, seed=3)
    b, _ = amplify.amplified_raster(*args, width=512, seed=3)
    c, _ = amplify.amplified_raster(*args, width=512, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_equirect_points_are_unit_vectors_covering_the_sphere():
    pts, shape = amplify.equirect_points(64)
    assert shape == (32, 64)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0)
    assert pts[:, 2].max() > 0.98 and pts[:, 2].min() < -0.98


# --------------------------------------------------------------------------------------
# The constraint that matters
# --------------------------------------------------------------------------------------


def test_amplification_must_not_be_fed_back_onto_the_mesh(world):
    """Detail on the simulation mesh reorganises the drainage the pipeline exists to build.

    Adding detail perturbs elevations, which changes steepest-descent receivers, which
    rewires the network. Measured: slope-area concavity moves 0.512 -> 0.603 and ~2300
    spurious channel cells appear. Amplification is an output-side operation, and the
    metrics belong on the pre-amplification field. This test pins that so nobody
    "simplifies" the pipeline by amplifying in place.
    """
    g, crust, res, fr = world
    land = crust.land_mask
    thr = float(np.median(g.area) * 6)
    amp_mesh = amplify.amplify(
        g.xyz, res.elevation,
        amplify.detail_amplitude(fr.slope, fr.acc, land), seed=1).astype(np.float32)

    def report(z):
        return metrics.report(np.ascontiguousarray(z, np.float32), g.neighbours,
                              g.arc_dist, g.area, is_sink=~land,
                              channel_threshold_km2=thr)

    base, amped = report(res.elevation), report(amp_mesh)
    assert abs(amped["concavity_theta"] - base["concavity_theta"]) > 0.03, (
        "if this stops being true the warning in terrain/amplify.py can be relaxed")
    assert amped["channel_cells"] > base["channel_cells"]
