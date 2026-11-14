"""Batched marks (``fast=``, ``theme.fast_threshold``): ggm-bam.3."""

import time
import warnings

import numpy as np
import pandas as pd
import pytest
from manim import Camera, Create, FadeIn, VGroup, VMobject

from ggmanim import (
    aes,
    arrow,
    coord_cartesian,
    facet_wrap,
    geom_bar,
    geom_col,
    geom_point,
    geom_raster,
    geom_rect,
    geom_segment,
    geom_tile,
    ggplot,
    theme,
)
from ggmanim.animate import GrowFromBaseline, attach_geoms
from ggmanim.geoms._batch import (
    DEFAULT_FAST_THRESHOLD,
    batched_group,
    is_batched,
    stack_layers,
    use_fast,
)
from ggmanim.theme import theme_default


def _scatter(n, seed=0, **cols):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    for k, v in cols.items():
        df[k] = v
    return df


_LIMITS = coord_cartesian(xlim=(-1, 1), ylim=(-1, 1))


def _layer(spec, i=0):
    return spec.build().layers[i]


def _built(spec):
    return attach_geoms(spec.build(), spec)


def _pixels(mob) -> np.ndarray:
    cam = Camera(pixel_width=640, pixel_height=360)
    cam.capture_mobject(mob)
    return cam.pixel_array.astype(np.int16)


def _differing(a, b, tol=16) -> float:
    """Share of pixels whose largest channel difference is above ``tol``."""
    return float((np.abs(a - b).max(axis=2) > tol).mean())


# -- dispatch -------------------------------------------------------------------------


def test_default_threshold_is_documented_on_the_theme():
    assert theme_default().fast_threshold == DEFAULT_FAST_THRESHOLD == 2000
    assert "fast_threshold" in type(theme_default()).__doc__


def test_use_fast_auto_threshold_and_overrides():
    t = theme_default()
    small, big = pd.DataFrame({"x": range(2000)}), pd.DataFrame({"x": range(2001)})
    assert not use_fast(None, small, t, "g")
    assert use_fast(None, big, t, "g")
    assert use_fast(True, small, t, "g")
    assert not use_fast(False, big, t, "g")
    assert not use_fast(None, big, theme(fast_threshold=None), "g")
    assert use_fast(None, pd.DataFrame({"x": [1]}), theme(fast_threshold=0), "g")


def test_mapped_key_turns_auto_mode_off_and_warns_when_forced():
    t = theme_default()
    keyed = pd.DataFrame({"x": range(3000), "key": range(3000)})
    assert not use_fast(None, keyed, t, "g")
    with pytest.warns(UserWarning, match="key"):
        assert use_fast(True, keyed, t, "g")


def test_below_threshold_layers_are_unchanged():
    lay = _layer(ggplot(_scatter(100), aes("x", "y")).add(geom_point()))
    assert not is_batched(lay)
    assert len(lay.marks) == 100 and lay.keys == list(range(100))


def test_auto_mode_batches_above_the_theme_threshold():
    df = _scatter(300)
    spec = ggplot(df, aes("x", "y")).add(geom_point())
    assert not is_batched(_layer(spec))
    lay = _layer(spec.add(theme(fast_threshold=250)))
    assert is_batched(lay) and lay.keys is None and len(lay.marks) == 1


# -- batched output ---------------------------------------------------------------------


def test_batched_group_attributes():
    a, b = VMobject(), VMobject()
    g = batched_group([a, b])
    assert isinstance(g, VGroup) and g.marks == [a, b] and g.keys is None and g.batched


def test_points_batch_by_style_with_one_subpath_per_mark():
    df = _scatter(90, g=["a", "b", "c"] * 30)
    lay = _layer(ggplot(df, aes("x", "y", color="g")).add(geom_point(fast=True)))
    assert is_batched(lay)
    # opaque points of different colours that overlap keep their data order, so a
    # colour can need more than one batch; every point is a closed 8-curve circle
    assert len({m.get_fill_color().to_hex() for m in lay.marks}) == 3
    assert sum(len(m.get_subpaths()) for m in lay.marks) == 90
    assert all(len(m.points) == 32 * len(m.get_subpaths()) for m in lay.marks)


def test_single_opaque_style_is_one_batch_even_when_marks_overlap():
    df = pd.DataFrame({"x": np.zeros(50), "y": np.linspace(0, 0.01, 50)})
    spec = ggplot(df, aes("x", "y")).add(geom_point(size=0.3, fast=True), _LIMITS)
    assert len(_layer(spec).marks) == 1


def test_translucent_overlapping_points_are_split_into_stacking_layers():
    df = pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.8], "y": [0.0, 0.001, 0.002, 0.8]})
    spec = ggplot(df, aes("x", "y")).add(geom_point(size=0.2, alpha=0.3, fast=True), _LIMITS)
    lay = _layer(spec)
    # three stacked points need three batches; the far one shares the first
    assert sorted(len(m.get_subpaths()) for m in lay.marks) == [1, 1, 2]


def test_shapes_sizes_and_strokes_batch_by_style():
    df = _scatter(60, v=np.tile([0.05, 0.1], 30))
    spec = ggplot(df, aes("x", "y", size="v")).add(geom_point(shape="circle_open", fast=True))
    lay = _layer(spec)
    assert is_batched(lay)
    # outlines are as wide as the marker's size makes them: one style per size
    widths = {round(m.get_stroke_width(), 3) for m in lay.marks}
    assert len(widths) == 2 and 0 not in widths
    assert all(m.get_fill_opacity() == 0 for m in lay.marks)


def test_points_cut_by_the_panel_edge_are_clipped_individually():
    df = pd.DataFrame({"x": [0.0, 1.0, 5.0], "y": [0.0, 0.0, 0.0]})
    spec = (
        ggplot(df, aes("x", "y"))
        .add(geom_point(size=0.3, fast=True))
        .add(coord_cartesian(xlim=(-1, 1), ylim=(-1, 1), expand=False))
    )
    lay = _layer(spec)
    assert is_batched(lay)
    # the centre point is batched; the one on the edge goes through clip_marker
    assert len(lay.marks) == 2


def test_tiles_batch_and_seal_like_the_slow_path():
    rows = [(i, j, str((i + j) % 3)) for i in range(10) for j in range(10)]
    g = pd.DataFrame(rows, columns=["x", "y", "v"])
    lay = _layer(ggplot(g, aes("x", "y", fill="v")).add(geom_tile(fast=True)))
    assert is_batched(lay) and len(lay.marks) == 3
    assert all(m.get_stroke_width() > 0 for m in lay.marks)  # seal_seams hairline
    assert sum(len(m.get_subpaths()) for m in lay.marks) == 100


def test_rect_dashed_outlines_are_drawn_per_row():
    df = pd.DataFrame({"xmin": [0, 2], "xmax": [1, 3], "ymin": [0, 0], "ymax": [1, 1]})
    spec = ggplot(df).add(
        geom_rect(
            aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
            color="red",
            linetype="dashed",
            fast=True,
        )
    )
    lay = _layer(spec)
    assert is_batched(lay) and len(lay.marks) == 2
    assert all(isinstance(m, VGroup) for m in lay.marks)  # VGroup(fill, dashes)


def test_bars_batch_and_grow_from_the_baseline():
    df = pd.DataFrame({"x": range(40), "y": np.linspace(1, 4, 40)})
    spec = ggplot(df, aes("x", "y")).add(geom_col(fast=True), name="bars")
    plot = _built(spec)
    lay = plot.layers["bars"]
    assert is_batched(lay) and len(lay.marks) == 1
    anim = plot.reveal_layer("bars")
    leaves = getattr(anim, "animations", [anim])
    assert any(isinstance(a, GrowFromBaseline) for a in leaves)


def test_geom_bar_accepts_fast():
    df = pd.DataFrame({"g": list("aabbbc")})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lay = _layer(ggplot(df, aes("g")).add(geom_bar(fast=True)))
    assert is_batched(lay)


def test_segments_batch_unless_dashed_or_with_arrows():
    df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 0, 0], "xend": [1, 2, 3], "yend": [1, 1, 1]})
    seg = aes("x", "y", xend="xend", yend="yend")
    lay = _layer(ggplot(df, seg).add(geom_segment(fast=True)))
    assert is_batched(lay) and len(lay.marks) == 1
    assert len(lay.marks[0].get_subpaths()) == 3
    lay = _layer(ggplot(df, seg).add(geom_segment(linetype="dashed", fast=True)))
    assert is_batched(lay) and len(lay.marks) == 3
    assert all(isinstance(m, VGroup) for m in lay.marks)  # dashed rows: drawn per row
    with pytest.warns(UserWarning, match="arrow"):
        lay = _layer(ggplot(df, seg).add(geom_segment(arrow=arrow(), fast=True)))
    assert not is_batched(lay) and len(lay.marks) == 3
    # auto mode with arrows: per-row drawing, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = ggplot(df, seg).add(geom_segment(arrow=arrow()), theme(fast_threshold=0))
        assert not is_batched(_layer(spec))


def test_raster_is_flagged_batched():
    g = pd.DataFrame(
        {"x": np.repeat([1, 2, 3], 3), "y": np.tile([1, 2, 3], 3), "v": list("pqr") * 3}
    )
    assert is_batched(_layer(ggplot(g, aes("x", "y", fill="v")).add(geom_raster())))


def test_fast_is_a_known_param():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for layer in (geom_point, geom_tile, geom_rect, geom_col, geom_segment):
            layer(fast=True)


# -- animation ------------------------------------------------------------------------------


def test_mark_refuses_batched_layers_including_faceted_ones():
    df = _scatter(40, f=["a", "b"] * 20)
    spec = ggplot(df, aes("x", "y")).add(geom_point(fast=True), name="pts")
    with pytest.raises(TypeError, match="batches"):
        _built(spec).mark("pts", 0)
    with pytest.raises(TypeError, match="batches"):
        _built(spec.add(facet_wrap("f"))).mark("pts", 0)


def test_whole_layer_reveals_work_on_batches():
    df = _scatter(50)
    df["xend"], df["yend"] = df.x + 0.1, df.y
    spec = (
        ggplot(df, aes("x", "y"))
        .add(geom_point(fast=True), name="pts")
        .add(geom_segment(aes(xend="xend", yend="yend"), fast=True), name="seg")
    )
    plot = _built(spec)
    assert isinstance(plot.reveal_layer("pts"), FadeIn)
    assert isinstance(plot.reveal_layer("seg"), Create)


# -- stacking layers ------------------------------------------------------------------------


def test_stack_layers_touching_boxes_do_not_overlap():
    lo = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert stack_layers(lo, lo + 1.0).tolist() == [0, 0, 0, 0]


def test_stack_layers_keeps_overlapping_pairs_in_order():
    lo = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [5.0, 5.0]])
    assert stack_layers(lo, lo + 1.0).tolist() == [0, 1, 2, 0]


def test_stack_layers_same_style_overlaps_can_be_ignored():
    lo = np.array([[0.0, 0.0], [0.5, 0.0], [0.25, 0.0]])
    style = np.array([0, 0, 1])
    # style 0 is opaque (no self overlap): 0 and 1 share a layer, 2 goes above both
    assert stack_layers(lo, lo + 1.0, style, np.array([False, False])).tolist() == [0, 0, 1]
    assert stack_layers(lo, lo + 1.0, style, np.array([True, True])).tolist() == [0, 1, 2]


def test_stack_layers_thin_boxes():
    # zero-height boxes (horizontal segments) still overlap along their length
    lo = np.array([[0.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
    hi = np.array([[1.0, 0.0], [1.5, 0.0], [4.0, 0.0]])
    assert stack_layers(lo, hi).tolist() == [0, 1, 0]


# -- visual parity --------------------------------------------------------------------------


def _parity_spec(case, fast):
    rng = np.random.default_rng(3)
    df = _scatter(300, seed=3, g=rng.choice(list("abc"), 300))
    df["s"] = rng.choice(["circle", "triangle", "square_open", "cross", "diamond_filled"], 300)
    df["xend"] = df.x + rng.normal(0, 0.3, 300)
    df["yend"] = df.y + rng.normal(0, 0.3, 300)
    rows = [(i, j, np.sin(i / 3) + np.cos(j / 4)) for i in range(30) for j in range(20)]
    grid = pd.DataFrame(rows, columns=["x", "y", "z"])
    bars = pd.DataFrame({"x": np.arange(60), "y": rng.uniform(1, 5, 60), "g": list("ab") * 30})
    if case == "points":
        return ggplot(df, aes("x", "y", color="g")).add(geom_point(fast=fast))
    if case == "points_alpha":
        return ggplot(df, aes("x", "y", color="g")).add(geom_point(size=0.12, alpha=0.4, fast=fast))
    if case == "shapes":
        return ggplot(df, aes("x", "y", shape="s")).add(geom_point(fill="red", fast=fast))
    if case == "tiles":
        return ggplot(grid, aes("x", "y", fill="z")).add(geom_tile(fast=fast))
    if case == "bars":
        bar = geom_col(color="white", alpha=0.6, width=2, position="identity", fast=fast)
        return ggplot(bars, aes("x", "y", fill="g")).add(bar)
    seg = aes("x", "y", xend="xend", yend="yend", color="g")
    return ggplot(df, seg).add(geom_segment(fast=fast))


@pytest.mark.parametrize("case", ["points", "points_alpha", "shapes", "tiles", "bars", "segments"])
def test_fast_renders_like_slow(case):
    small = theme(panel_width=5, panel_height=3)
    slow = _parity_spec(case, False).add(small).build()
    fast = _parity_spec(case, True).add(small).build()
    assert not is_batched(slow.layers[0]) and is_batched(fast.layers[0])
    assert _differing(_pixels(slow), _pixels(fast)) < 0.005


def test_10k_scatter_builds_fast():
    spec = ggplot(_scatter(10_000), aes("x", "y")).add(geom_point(size=0.02))
    spec.build()  # warm-up (prototypes, fonts)
    t = time.perf_counter()
    plot = spec.build()
    elapsed = time.perf_counter() - t
    assert is_batched(plot.layers[0])
    assert elapsed < 3.0  # ~0.2 s on a laptop; generous for loaded CI machines
