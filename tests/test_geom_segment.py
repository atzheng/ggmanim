"""geom_segment / geom_curve: endpoints, clipping, arrowheads."""

import numpy as np
import pandas as pd
import pytest
from manim import ArcBetweenPoints, Polygon, VGroup

import ggmanim
from ggmanim import aes, arrow, coord_cartesian, geom_curve, geom_segment, ggplot
from ggmanim.geoms import ArrowSpec, GeomCurve, GeomSegment
from ggmanim.registry import COMPONENTS

DF = pd.DataFrame({"x": [0.0, 2.0], "y": [0.0, 1.0], "xend": [4.0, 2.0], "yend": [2.0, 5.0]})
MAP = aes("x", "y", xend="xend", yend="yend")


def test_registered_and_exported():
    for n in ("geom_segment", "geom_curve", "arrow"):
        assert n in ggmanim.__all__
    assert "geom_segment" in COMPONENTS and "geom_curve" in COMPONENTS
    assert isinstance(geom_segment().geom, GeomSegment)
    assert isinstance(geom_curve().geom, GeomCurve)
    with pytest.raises(TypeError, match="arrow"):
        geom_segment(arrow="yes")


def test_arrow_validation():
    a = arrow()
    assert isinstance(a, ArrowSpec) and a.ends == "last" and a.type == "open"
    with pytest.raises(ValueError):
        arrow(ends="middle")
    with pytest.raises(ValueError):
        arrow(type="fancy")
    with pytest.raises(ValueError):
        arrow(angle=120)


def test_segment_endpoints_and_training():
    m = ggplot(DF, MAP).geom_segment().build()
    marks = m.layers[0].marks
    assert len(marks) == 2 and m.layers[0].keys == [0, 1]
    for line, row in zip(marks, DF.itertuples()):
        np.testing.assert_allclose(line.get_start(), m.panel.transform(row.x, row.y)[0])
        np.testing.assert_allclose(line.get_end(), m.panel.transform(row.xend, row.yend)[0])
    # xend/yend train the scales
    lo, hi = m.panel.panel_params.y_range
    assert lo <= 0 and hi >= 5


def test_segment_clipped_and_no_head_when_cut():
    m = (
        ggplot(DF, MAP)
        .geom_segment(arrow=arrow(ends="both"))
        .add(coord_cartesian(xlim=(0, 2), ylim=(0, 5), expand=False))
        .build()
    )
    first, second = m.layers[0].marks
    # first segment leaves the panel at x = 2: its line ends there, only the start head
    assert isinstance(first, VGroup) and len(first) == 2
    line = first[0]
    np.testing.assert_allclose(line.get_end(), m.panel.transform(2.0, 1.0)[0], atol=1e-9)
    # the second is fully inside: line + 2 heads
    assert len(second) == 3


def test_arrow_heads_open_and_closed():
    m = ggplot(DF.iloc[:1], MAP).geom_segment(arrow=arrow(length=0.2, angle=30)).build()
    line, head = m.layers[0].marks[0]
    tip = m.panel.transform(4.0, 2.0)[0]
    verts = head.get_anchors()
    assert any(np.allclose(v, tip, atol=1e-9) for v in verts)  # head point at the end
    barbs = [v for v in verts if not np.allclose(v, tip)]
    for v in barbs:
        assert np.linalg.norm(v - tip) == pytest.approx(0.2)
    m = ggplot(DF.iloc[:1], MAP).geom_segment(arrow=arrow(type="closed")).build()
    _, head = m.layers[0].marks[0]
    assert isinstance(head, Polygon) and head.get_fill_opacity() > 0


def test_zero_length_segment_skipped():
    df = pd.DataFrame({"x": [1.0, 0.0], "y": [1.0, 0.0], "xend": [1.0, 1.0], "yend": [1.0, 2.0]})
    m = ggplot(df, MAP).geom_segment(arrow=arrow()).build()
    assert m.layers[0].keys == [1]


def test_curve_is_arc_between_endpoints():
    m = ggplot(DF, MAP).geom_curve(curvature=0.5).build()
    arc = m.layers[0].marks[1]  # (2, 1) -> (2, 5) stays inside the panel
    assert isinstance(arc, ArcBetweenPoints)
    a = m.panel.transform(2.0, 1.0)[0]
    b = m.panel.transform(2.0, 5.0)[0]
    np.testing.assert_allclose(arc.get_start(), a, atol=1e-6)
    np.testing.assert_allclose(arc.get_end(), b, atol=1e-6)
    # positive curvature bends to the right of the travel direction
    mid = arc.point_from_proportion(0.5)
    d = b - a
    assert np.cross(d[:2], (mid - a)[:2]) < 0
    m2 = ggplot(DF, MAP).geom_curve(curvature=-0.5).build()
    mid2 = m2.layers[0].marks[1].point_from_proportion(0.5)
    assert np.cross(d[:2], (mid2 - a)[:2]) > 0


def test_curve_clipped_and_arrows():
    m = (
        ggplot(DF, MAP)
        .geom_curve(curvature=1, arrow=arrow(ends="both"))
        .add(coord_cartesian(xlim=(0, 4), ylim=(0, 5), expand=False))
        .build()
    )
    for mark in m.layers[0].marks:
        pts = np.vstack([s.points for s in mark.get_family() if len(s.points)])
        # every mark stays (almost: heads have extent) inside the panel's scene box
        lo = m.panel.transform(0.0, 0.0)[0]
        hi = m.panel.transform(4.0, 5.0)[0]
        assert pts[:, 0].min() >= lo[0] - 0.25 and pts[:, 0].max() <= hi[0] + 0.25
    assert len(m.layers[0].marks) == 2
