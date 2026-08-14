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
    a = amplify.detail_amplitude(fr.slope, fr.acc, land,
                                 sphere_km2=float(g.area.sum()))
    steep = land & (fr.slope > np.percentile(fr.slope[land], 90))
    flat = land & (fr.slope < np.percentile(fr.slope[land], 25))
    assert a[steep].mean() > 2.0 * a[flat].mean()


def test_trunk_valleys_are_smoothed(world):
    """Alluvial floors are smooth because sediment fills them."""
    g, crust, res, fr = world
    land = crust.land_mask
    a = amplify.detail_amplitude(fr.slope, fr.acc, land,
                                 sphere_km2=float(g.area.sum()))
    valleys = land & (fr.acc > np.percentile(fr.acc[land], 98))
    elsewhere = land & (fr.acc < np.percentile(fr.acc[land], 50))
    assert a[valleys].mean() < 0.2 * a[elsewhere].mean()


def test_flat_ground_keeps_some_texture(world):
    """Without a floor on the slope term, amplitude hits exactly zero on plains, on valley
    floors and across the whole sea bed -- base level is flat, so its slope is zero."""
    g, crust, res, fr = world
    land = crust.land_mask
    a = amplify.detail_amplitude(fr.slope, fr.acc, land,
                                 sphere_km2=float(g.area.sum()))
    flat = land & (fr.slope < np.percentile(fr.slope[land], 10))
    # As a FRACTION of the cap, not an absolute height: the cap is denominated against the
    # drop across one cell, so it moved from 260 m to 3 m with the 100 m grid.
    floor = 0.019 * amplify.AmplifyParams().max_amplitude_m
    assert a[flat].mean() > floor, "plains should not be billiard tables"
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
    """What the opt-in blends buy: the default renders the mesh cells themselves.

    That is the trade the modes exist to expose -- `nearest` is the mesh data untouched and
    shows the hexagons; the blends smooth them and no longer carry one cell per pixel.
    """
    g, crust, res, fr = world
    args = (g, res.elevation, fr.slope, fr.acc, crust.land_mask)
    nearest, _ = amplify.amplified_raster(*args, width=1024, detail=False)
    for m in ("barycentric", "c1"):
        smooth, _ = amplify.amplified_raster(*args, width=1024, detail=False,
                                             p=amplify.AmplifyParams(resample=m))
        assert _high_freq(smooth) < 0.5 * _high_freq(nearest), m


def test_interpolation_invents_no_texture_of_its_own(world):
    """Resample a PERFECTLY SMOOTH analytic field and nothing should appear.

    Inverse-distance weighting fails this badly, and that failure was the pervasive
    mesh-scale crosshatch in every hillshaded render. Shepard's method has zero gradient at
    every data point, so it puts a dimple at each cell centre; the elevations look fine and
    the derivative -- which is what hillshading shows -- does not. Barycentric over the
    mesh triangles is exact for a linear field and near-exact for a smooth one.
    """
    g, crust, res, fr = world
    xyz = g.xyz.astype(np.float64)
    smooth_field = 2000.0 * (xyz[:, 0] * xyz[:, 1] + 0.5 * xyz[:, 2] ** 2)

    pts, shape = amplify.equirect_points(512)
    exact = 2000.0 * (pts[:, 0] * pts[:, 1] + 0.5 * pts[:, 2] ** 2)

    def manufactured(idx, w):
        err = (amplify.apply_weights(smooth_field, idx, w) - exact).reshape(shape)
        # Second difference along a row: the curvature hillshading amplifies.
        return float(np.abs(np.diff(err, 2, axis=1)).mean())

    bary = manufactured(*amplify.barycentric_weights(g, pts))
    idw = manufactured(*amplify.neighbour_weights(g, pts, k=3))
    assert bary < 0.1 * idw, f"barycentric {bary:.4f} vs IDW {idw:.4f}"


def test_c1_interpolation_reproduces_a_linear_field_exactly(world):
    """Linear precision is the whole reason to blend polynomials rather than values.

    If the field is linear, every cell's linear Taylor expansion already equals it
    everywhere, so any convex combination of them is exact.
    """
    g, crust, res, fr = world
    xyz = g.xyz.astype(np.float64)
    pts, _ = amplify.equirect_points(256)
    for coeffs in ((1000.0, 0.0, 0.0), (300.0, -450.0, 120.0)):
        a, b, c = coeffs
        field = a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2]
        exact = a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2]
        got = amplify.interpolate(g, field, pts, mode="c1")
        assert np.abs(got - exact).max() < 0.05, f"{coeffs}: {np.abs(got-exact).max():.4f} m"


def test_c1_interpolation_is_interpolating_at_cell_centres(world):
    """A cell centre must return its own value, or this is smoothing rather than
    interpolation -- which is what it exists to avoid."""
    g, crust, res, fr = world
    rng = np.random.default_rng(0)
    sample = rng.choice(g.cell_count, 4000, replace=False)
    got = amplify.interpolate(g, res.elevation, np.ascontiguousarray(g.xyz[sample]),
                              mode="c1")
    assert np.abs(got - np.asarray(res.elevation, np.float64)[sample]).max() < 1e-6


def test_c1_interpolation_does_not_overshoot_the_mesh(world):
    """Gradient extrapolation goes Gibbs-like across a near-discontinuity.

    The continental shelf break is very nearly a step in the mesh field, and without the
    limiter the blend undershot the abyssal plain by 471 m -- relief the simulation never
    produced, invented by the export.
    """
    g, crust, res, fr = world
    pts, _ = amplify.equirect_points(512)
    z = np.asarray(res.elevation, np.float64)
    got = amplify.interpolate(g, z, pts, mode="c1")
    assert got.min() >= z.min() - 1e-6
    assert got.max() <= z.max() + 1e-6


def test_c1_beats_barycentric_on_gradient_continuity(world):
    """Barycentric is linear per triangle, so its GRADIENT jumps at every triangle edge.

    Hillshading is a gradient operator, so it draws those edges as a herringbone on steep
    ground. Facets are sparse, large jumps in the second difference, so the tail of that
    distribution is what separates the two -- the mean does not, because a C1 surface
    legitimately carries more smooth curvature than a set of flat planes.
    """
    g, crust, res, fr = world
    pts, shape = amplify.equirect_points(1024)
    z = np.asarray(res.elevation, np.float64)

    def facet_tail(v):
        d2 = np.abs(np.diff(v.reshape(shape), 2, axis=1))
        return float(np.percentile(d2, 99.5) / max(np.median(d2), 1e-9))

    bary = facet_tail(amplify.apply_weights(z, *amplify.barycentric_weights(g, pts)))
    c1 = facet_tail(amplify.interpolate(g, z, pts, mode="c1"))
    assert c1 < bary, f"C1 tail {c1:.1f} vs barycentric {bary:.1f}"


def test_the_default_blends_nothing(world):
    """The default export must carry mesh values unaltered.

    Every pixel is exactly one cell's value -- nothing averaged, nothing manufactured. Both
    blends are opt-in, so a default raster contains no number the simulation did not.
    """
    g, crust, res, fr = world
    assert amplify.AmplifyParams().resample == "nearest"

    pts, shape = amplify.equirect_points(512)
    z = np.asarray(res.elevation, np.float64)
    default = amplify.interpolate(g, z, pts)
    nearest_idx, _ = amplify.neighbour_weights(g, pts, k=1)
    assert np.array_equal(default, z[nearest_idx])
    # Every value present must be an actual cell value, not a blend of several.
    assert np.isin(default, z).all()

    raster, _ = amplify.amplified_raster(g, res.elevation, fr.slope, fr.acc,
                                         crust.land_mask, width=512, detail=False)
    assert np.array_equal(raster, z[nearest_idx].reshape(shape).astype(np.float32))


def test_resample_mode_is_honoured_end_to_end(world):
    """Each mode must actually change the raster, and an unknown one must fail loudly."""
    g, crust, res, fr = world
    args = (g, res.elevation, fr.slope, fr.acc, crust.land_mask)
    out = {m: amplify.amplified_raster(*args, width=512, detail=False,
                                       p=amplify.AmplifyParams(resample=m))[0]
           for m in amplify.RESAMPLE_MODES}
    for a, b in (("nearest", "barycentric"), ("barycentric", "c1"), ("nearest", "c1")):
        assert not np.array_equal(out[a], out[b]), f"{a} and {b} produced the same raster"

    pts, _ = amplify.equirect_points(256)
    with pytest.raises(ValueError):
        amplify.interpolate(g, res.elevation, pts, mode="cubic")


def test_barycentric_weights_are_a_convex_combination(world):
    """Anything else is not linear interpolation, and would over- or undershoot."""
    g, crust, res, fr = world
    pts, _ = amplify.equirect_points(512)
    idx, w = amplify.barycentric_weights(g, pts)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w >= -1e-12).all(), "negative weight means the point is outside its triangle"
    assert (idx >= 0).all() and (idx < g.cell_count).all()
    # Interpolating a constant must give exactly that constant.
    const = np.full(g.cell_count, 7.5)
    assert np.allclose(amplify.apply_weights(const, idx, w), 7.5)


def test_detail_is_capped_to_what_the_raster_can_carry(world):
    """Octaves finer than the pixel are not detail, they are aliasing.

    At the 6-octave ceiling from a base frequency of 180, the finest octave has 1.1 km
    features against a ~4.9 km pixel at width 8192 -- energy that folds back as long
    straight moire fringes across the terrain.
    """
    p = amplify.AmplifyParams()
    counts = [amplify.octaves_for_width(w, p) for w in (2048, 8192, 32768)]
    assert counts == sorted(counts), "a finer raster must not carry fewer octaves"
    assert all(c <= p.octaves for c in counts)
    for width, c in zip((2048, 8192, 32768), counts):
        finest = p.frequency * p.lacunarity ** (c - 1)
        px_per_feature = (1.0 / finest) / (2.0 * np.pi / width)
        assert px_per_feature >= p.min_px_per_feature - 1e-9 or c == 1, (
            f"width {width}: finest octave spans {px_per_feature:.2f} px")


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
    # Against the CAP, not an absolute height. The cap is denominated against the drop
    # across one cell, so it went 260 m -> 3 m with the 100 m grid and a fixed 0.5 m bar
    # would silently become a demand for 17% of full amplitude rather than 0.19%.
    cap = amplify.AmplifyParams().max_amplitude_m
    assert d.mean() > 0.0019 * cap, "detail should actually do something"
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
    rewires the network. Measured: slope-area concavity moves 0.512 -> 0.603 and the
    channel count shifts by thousands of cells. Amplification is an output-side
    operation, and the metrics belong on the pre-amplification field. This test pins that
    so nobody "simplifies" the pipeline by amplifying in place.

    The two assertions carry different weight. THETA is the load-bearing one: it moves
    0.024 on a base of ~0.36, a 7% shift in the project's headline realism metric, and
    that is what the warning in terrain/amplify.py is about. The channel count is a
    secondary corroboration and is asserted as a MAGNITUDE, not a direction -- on the
    Earth-radius grid detail added ~2300 spurious channels; here it moves ~68 either way,
    because 3 m of texture on a 100 m cell rarely flips a receiver. Which way it goes is
    a property of the amplitude-to-cell-relief ratio and is not the point.
    """
    g, crust, res, fr = world
    land = crust.land_mask
    thr = float(np.median(g.area) * 6)
    amp_mesh = amplify.amplify(
        g.xyz, res.elevation,
        amplify.detail_amplitude(fr.slope, fr.acc, land,
                                 sphere_km2=float(g.area.sum())),
        seed=1).astype(np.float32)

    def report(z):
        return metrics.report(np.ascontiguousarray(z, np.float32), g.neighbours,
                              g.arc_dist, g.area, is_sink=~land,
                              channel_threshold_km2=thr)

    base, amped = report(res.elevation), report(amp_mesh)
    # 0.01, having been 0.02 and before that 0.03. Detail amplitude is 3 m here against 260
    # at Earth radius, by design, so the perturbation keeps shrinking in absolute terms; the
    # latest step is the coupled loop moving to seasonal precipitation, which erodes a
    # different world and this one is less sensitive -- measured 0.016 against 0.024. Still
    # 5% of a theta near 0.31, which is what the warning in terrain/amplify.py is about.
    assert abs(amped["concavity_theta"] - base["concavity_theta"]) > 0.01, (
        "if this stops being true the warning in terrain/amplify.py can be relaxed")
    moved = abs(amped["channel_cells"] - base["channel_cells"])
    assert moved > 0.002 * base["channel_cells"], "the network should be visibly rewired"
