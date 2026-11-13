"""Point markers are clipped to the panel border (ggm-sxu)."""

import numpy as np
import pandas as pd
import pytest
from manim import Dot, VGroup

from ggmanim import aes, ggplot
from ggmanim.geoms._marker_clip import clip_marker, scene_box
from ggmanim.geoms._markers import make_marker

SHAPES = [
    "circle",
    "circle_open",
    "circle_filled",
    "square",
    "triangle_open",
    "diamond",
    "star",
    "star_open",
    "cross",
    "plus",
]


def _panel_box(m):
    pp = m.panel_params
    corners = m.panel.transform([pp.x_range[0], pp.x_range[1]], [pp.y_range[0], pp.y_range[1]])
    return corners.min(axis=0), corners.max(axis=0)


def _points(mob):
    pts = [np.asarray(m.points) for m in mob.family_members_with_points()]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _border_plot(shape="circle", clip="on", size=0.15):
    # x = 0 and x = 10 / y = 0 lie exactly on the panel border (expand=False)
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0, 10.0], "y": [5.0, 5.0, 0.0, 10.0]})
    return (
        ggplot(df, aes("x", "y"))
        .geom_point(shape=shape, size=size, fill="red")
        .coord_cartesian(xlim=(0, 10), ylim=(0, 10), expand=False, clip=clip)
        .build()
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_border_markers_stay_inside_panel(shape):
    m = _border_plot(shape)
    lo, hi = _panel_box(m)
    layer = m.layers[0]
    assert len(layer) == 4
    for mob in layer:
        pts = _points(mob)
        assert np.all(pts[:, :2] >= lo[:2] - 1e-6), shape
        assert np.all(pts[:, :2] <= hi[:2] + 1e-6), shape


def test_interior_marker_untouched():
    on, off = _border_plot(clip="on"), _border_plot(clip="off")
    a, b = on.layers[0][1], off.layers[0][1]  # the centre point
    assert type(a) is type(b) is Dot
    # (the layouts differ by a shift: clip="off" markers widen the plot's bbox)
    np.testing.assert_allclose(a.points - a.get_center(), b.points - b.get_center(), atol=1e-9)
    # border points are rebuilt, interior ones are the plain marker
    assert isinstance(on.layers[0][0], VGroup) and not isinstance(on.layers[0][0], Dot)


def test_clip_off_unchanged():
    m = _border_plot(clip="off")
    lo, hi = _panel_box(m)
    assert all(type(d) is Dot for d in m.layers[0])
    pts = _points(m.layers[0])
    assert pts[:, 0].min() < lo[0] - 0.1 and pts[:, 0].max() > hi[0] + 0.1


def test_filled_keeps_fill_open_has_none():
    solid = _border_plot("circle").layers[0][0]
    assert all(p.get_fill_opacity() > 0 for p in solid)
    assert all(p.get_stroke_width() == 0 for p in solid)
    hollow = _border_plot("circle_open").layers[0][0]
    assert len(hollow) == 1 and hollow[0].get_fill_opacity() == 0
    assert hollow[0].get_stroke_width() > 0
    both = _border_plot("circle_filled").layers[0][0]
    assert sorted(p.get_fill_opacity() > 0 for p in both) == [False, True]


def test_half_circle_on_border():
    dot = make_marker("circle", [1.0, 0.0, 0.0], radius=0.5, color="#ffffff")
    out = clip_marker(dot, (0.0, 1.0, -1.0, 1.0))
    (fill,) = out
    v = np.asarray(fill.get_anchors())[:, :2]
    area = 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], 1)) - np.dot(v[:, 1], np.roll(v[:, 0], 1)))
    assert area == pytest.approx(np.pi * 0.25 / 2, rel=0.02)
    assert v[:, 0].max() == pytest.approx(1.0)


def test_open_outline_does_not_stroke_the_border():
    m = make_marker("square_open", [1.0, 0.0, 0.0], radius=0.5, color="#ffffff")
    out = clip_marker(m, (0.0, 1.0, -1.0, 1.0))
    (stroke,) = out
    assert stroke.get_fill_opacity() == 0
    pts = np.asarray(stroke.get_anchors())
    # an open path (two corners + one edge), not a closed polygon along the border
    assert len(stroke.get_subpaths()) == 1
    assert not np.allclose(pts[0], pts[-1])


def test_plus_on_corner_keeps_arms_along_border():
    m = make_marker("plus", [1.0, 0.0, 0.0], radius=0.5, color="#ffffff")
    out = clip_marker(m, (0.0, 1.0, 0.0, 1.0))
    pts = _points(out)
    assert len(pts) and np.all(pts[:, 0] <= 1 + 1e-9) and np.all(pts[:, 1] >= -1e-9)
    assert pts[:, 0].min() < 0.6 and pts[:, 1].max() > 0.4  # both inward arms visible


@pytest.mark.parametrize("eps", [0.0, 1e-15, -1e-15])
@pytest.mark.parametrize(
    ("centre", "axis"),
    [((0.0, 0.5), 0), ((1.0, 0.5), 0), ((0.5, 0.0), 1), ((0.5, 1.0), 1)],
    ids=["left", "right", "bottom", "top"],
)
def test_plus_on_border_keeps_arm_along_border(centre, axis, eps):
    # a plus centred on a panel edge: one arm runs along the border (up to
    # rounding); it must be kept whole, not cut at an arbitrary point (ggm-sol)
    c = np.array([*centre, 0.0])
    c[axis] += eps
    m = make_marker("plus", c, radius=0.2, color="#ffffff")
    out = clip_marker(m, (0.0, 1.0, 0.0, 1.0))
    pts = _points(out)[:, :2]
    assert np.all(pts >= -1e-9) and np.all(pts <= 1 + 1e-9)
    other = 1 - axis
    on_border = pts[np.abs(pts[:, axis] - centre[axis]) <= 1e-9]
    lo, hi = on_border[:, other].min(), on_border[:, other].max()
    assert lo < 0.5 - 0.15 and hi > 0.5 + 0.15  # the full arm along the border
    assert np.abs(pts[:, axis] - centre[axis]).max() > 0.15  # the inward arm


def test_clip_marker_fast_path_and_invisible():
    dot = make_marker("circle", [0.5, 0.0, 0.0], radius=0.1, color="#ffffff")
    assert clip_marker(dot, (0.0, 1.0, -1.0, 1.0)) is dot
    assert clip_marker(dot, (5.0, 6.0, 5.0, 6.0)) is None


def test_centre_just_outside_is_drawn_partially():
    df = pd.DataFrame({"x": [10.02, 5.0], "y": [5.0, 5.0]})
    m = (
        ggplot(df, aes("x", "y"))
        .geom_point(size=0.15)
        .coord_cartesian(xlim=(0, 10), ylim=(0, 10), expand=False)
        .build()
    )
    lo, hi = _panel_box(m)
    assert len(m.layers[0]) == 2
    assert np.all(_points(m.layers[0])[:, 0] <= hi[0] + 1e-6)


def test_scene_box_coord_flip():
    df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 4.0]})
    m = ggplot(df, aes("x", "y")).geom_point().coord_flip(expand=False).build()
    pp = m.panel_params
    box = scene_box(m.panel, (*pp.x_range, *pp.y_range))
    lo, hi = _panel_box(m)
    np.testing.assert_allclose(box, (lo[0], hi[0], lo[1], hi[1]))
