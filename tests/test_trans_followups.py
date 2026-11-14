"""coord_trans / coord_polar follow-ups (ggm-78e): munched segmented paths,
abline, segment and curve; facet_grid(space="free") in transformed space;
values at a transform's open domain end drawn at the panel edge."""

import warnings

import numpy as np
import pandas as pd
import pytest
from manim import DL, UR

from ggmanim import (
    aes,
    arrow,
    coord_polar,
    coord_trans,
    facet_grid,
    geom_abline,
    geom_area,
    geom_curve,
    geom_path,
    geom_point,
    geom_segment,
    ggplot,
)
from ggmanim.coords_trans import TransPanelParams
from ggmanim.facets._layout import space_ranges
from ggmanim.geoms._segments import munch_segments
from ggmanim.geoms.segment import bend_along


def _pts(m) -> np.ndarray:
    return np.asarray(m.get_all_points())[:, :2]


def _anchors(m) -> np.ndarray:
    return np.asarray(m.get_start_anchors())[:, :2]


def _bend(pts: np.ndarray) -> float:
    """Largest distance of ``pts`` from the chord through its ends."""
    a, b = pts[0], pts[-1]
    d = b - a
    n = np.array([-d[1], d[0]]) / np.hypot(*d)
    return float(np.max(np.abs((pts - a) @ n)))


def _inverse_y(panel, pts: np.ndarray) -> np.ndarray:
    """Scene y -> input-space y on a coord_trans(y="log10") panel."""
    (y0, y1) = panel.trans_range[1]
    lo, hi = panel.axes.get_corner(DL)[1], panel.axes.get_corner(UR)[1]
    return 10 ** (y0 + (pts[:, 1] - lo) / (hi - lo) * (y1 - y0))


def _inverse_x(panel, pts: np.ndarray) -> np.ndarray:
    (x0, x1) = panel.trans_range[0]
    lo, hi = panel.axes.get_corner(DL)[0], panel.axes.get_corner(UR)[0]
    return x0 + (pts[:, 0] - lo) / (hi - lo) * (x1 - x0)


# --------------------------------------------------------------------------- segmented paths
def _varying():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 1000.0, 1.0], "v": [0.0, 1.0, 2.0]})


def test_segmented_path_munched_under_coord_trans():
    p = ggplot(_varying(), aes("x", "y", colour="v")).add(geom_path()).add(coord_trans(y="log10"))
    segs = p.build().layers[0].marks[0]
    assert len(segs) == 2  # one run per segment style, as without munching
    colours = {m.get_stroke_color().to_hex() for m in segs}
    assert len(colours) == 2
    panel = p.build().panel
    ends = [((1.0, 1.0), (2.0, 1000.0)), ((2.0, 1000.0), (3.0, 1.0))]
    for m, ((xa, ya), (xb, yb)) in zip(segs, ends):
        pts = _anchors(m)
        assert len(pts) > 10
        # every vertex lies on the straight data-space segment
        x, y = _inverse_x(panel, pts), _inverse_y(panel, pts)
        np.testing.assert_allclose(x, xa + (y - ya) / (yb - ya) * (xb - xa), atol=1e-6)
        assert _bend(np.asarray(m.get_all_points())[:, :2]) > 0.3


def test_munch_segments_keeps_segment_style_rows():
    df = _varying().assign(group=1)
    panel = ggplot(df, aes("x", "y")).add(geom_point()).add(coord_trans(y="log10")).build().panel
    out = munch_segments(panel, df)
    assert len(out) > len(df)
    # vertices of the first segment carry row 0's style, the last vertex row 2's
    first = out[out["y"] < 1000.0]
    assert out["v"].iloc[0] == 0.0 and out["v"].iloc[-1] == 2.0
    assert set(out["v"]) == {0.0, 1.0, 2.0}
    assert (first["v"].iloc[: len(first) // 2] == 0.0).all()
    # linear coord: untouched
    lin = ggplot(df, aes("x", "y")).add(geom_point()).build().panel
    assert munch_segments(lin, df) is df


def test_segmented_path_under_polar_is_spiral():
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [5.0, 5.0, 5.0], "v": [0.0, 1.0, 2.0]})
    p = ggplot(df, aes("x", "y", colour="v")).add(geom_path()).add(coord_polar()).build()
    segs = p.layers[0].marks[0]
    centre = p.panel.axes.c2p(0.5, 0.5)[:2]
    for m in segs:
        r = np.hypot(*(_anchors(m) - centre).T)
        assert len(r) > 10
        assert np.ptp(r) < 1e-6  # constant y: an arc of a circle


# --------------------------------------------------------------------------- abline
def test_abline_bends_under_coord_trans_and_follows_data_space_line():
    df = pd.DataFrame({"x": [1.0, 10.0], "y": [1.0, 1000.0]})
    p = (
        ggplot(df, aes("x", "y"))
        .add(geom_point())
        .add(geom_abline(slope=100, intercept=0))
        .add(coord_trans(y="log10"))
        .build()
    )
    line = p.layers[1].marks[0]
    pts = _anchors(line)
    assert len(pts) >= 20
    x, y = _inverse_x(p.panel, pts), _inverse_y(p.panel, pts)
    np.testing.assert_allclose(y, 100 * x, rtol=1e-6)
    assert _bend(pts) > 0.3


def test_abline_unchanged_under_linear_coord():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    p = ggplot(df, aes("x", "y")).add(geom_point()).add(geom_abline()).build()
    assert len(p.layers[1].marks[0].get_all_points()) == 4  # one Line


def test_abline_slope_zero_is_circle_under_polar():
    df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    p = (
        ggplot(df, aes("x", "y"))
        .add(geom_point())
        .add(geom_abline(slope=0, intercept=5))
        .add(coord_polar())
        .build()
    )
    pts = _anchors(p.layers[1].marks[0])
    r = np.hypot(*(pts - p.panel.axes.c2p(0.5, 0.5)[:2]).T)
    assert len(pts) > 20
    assert np.ptp(r) < 1e-6


# --------------------------------------------------------------------------- segment / curve
def _seg_df():
    return pd.DataFrame(
        {"x": [1.0, 5.0], "y": [1.0, 5.0], "xend": [10.0, 5.0], "yend": [1000.0, 5.0]}
    )


def test_segment_munched_with_arrow_along_tangent():
    p = (
        ggplot(_seg_df(), aes("x", "y", xend="xend", yend="yend"))
        .add(geom_segment(arrow=arrow(type="closed")))
        .add(coord_trans(y="log10"))
        .build()
    )
    marks = p.layers[0].marks
    assert len(marks) == 1  # zero-length row dropped
    line, head = marks[0].submobjects
    pts = _anchors(line)
    assert len(pts) >= 20 and _bend(pts) > 0.3
    tip = p.panel.transform(10.0, 1000.0)[0, :2]
    np.testing.assert_allclose(np.asarray(line.get_all_points())[-1, :2], tip, atol=1e-9)
    assert any(np.allclose(v[:2], tip) for v in head.get_vertices())


def test_segment_fast_ignored_under_nonlinear_coord():
    p = (
        ggplot(_seg_df(), aes("x", "y", xend="xend", yend="yend"))
        .add(geom_segment(fast=True))
        .add(coord_trans(y="log10"))
        .build()
    )
    assert len(_anchors(p.layers[0].marks[0])) >= 20


def test_curve_zero_curvature_matches_segment_and_curve_ends_fixed():
    base = ggplot(_seg_df().iloc[:1], aes("x", "y", xend="xend", yend="yend"))
    trans = coord_trans(y="log10")
    seg = base.add(geom_segment()).add(trans).build().layers[0].marks[0]
    flat = base.add(geom_curve(curvature=0)).add(trans).build()
    np.testing.assert_allclose(_pts(flat.layers[0].marks[0]), _pts(seg), atol=1e-9)
    bent = base.add(geom_curve(curvature=0.4)).add(trans).build()
    pts = _anchors(bent.layers[0].marks[0])
    np.testing.assert_allclose(pts[0], flat.panel.transform(1.0, 1.0)[0, :2], atol=1e-9)
    end = np.asarray(bent.layers[0].marks[0].get_all_points())[-1, :2]
    np.testing.assert_allclose(end, flat.panel.transform(10.0, 1000.0)[0, :2], atol=1e-9)
    flat_pts = _anchors(flat.layers[0].marks[0])
    gap = np.min(np.linalg.norm(pts[:, None] - flat_pts[None], axis=2), axis=1)
    assert gap.max() > 0.05  # bent away from the munched segment


def test_bend_along_straight_path_is_identity():
    pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.0], [1.0, 0.0, 0.0]])
    path = np.column_stack([np.linspace(0, 1, 11), np.zeros(11), np.zeros(11)])
    np.testing.assert_allclose(bend_along(pts, path), pts, atol=1e-12)
    # an L-shaped path: the midpoint (offset 0.3 left of the chord) moves to the corner's normal
    lpath = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    out = bend_along(np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.0, 0.0]]), lpath)
    np.testing.assert_allclose(out[[0, -1]], lpath[[0, -1]])


# --------------------------------------------------------------------------- facet space
def test_space_free_uses_transformed_ranges():
    df = pd.DataFrame(
        {"x": [1.0, 10.0, 1.0, 1000.0], "y": [1.0, 2.0, 1.0, 2.0], "f": ["a", "a", "b", "b"]}
    )
    p = (
        ggplot(df, aes("x", "y"))
        .add(geom_point())
        .add(facet_grid(cols="f", scales="free_x", space="free_x"))
        .add(coord_trans(x="log10"))
        .build()
    )
    pa, pb = p.panels[(1, 1)], p.panels[(1, 2)]
    assert isinstance(pa.panel_params, TransPanelParams)
    wa = pa.axes.get_corner(UR)[0] - pa.axes.get_corner(DL)[0]
    wb = pb.axes.get_corner(UR)[0] - pb.axes.get_corner(DL)[0]
    ta, tb = pa.trans_range[0], pb.trans_range[0]
    assert wa / wb == pytest.approx((ta[1] - ta[0]) / (tb[1] - tb[0]), rel=1e-6)
    assert wa / wb == pytest.approx(1 / 3, rel=0.05)  # 1 decade vs 3 decades
    h, v = space_ranges(pa.panel_params)
    assert h == pytest.approx(ta) and v == pytest.approx(pa.trans_range[1])


# --------------------------------------------------------------------------- domain edge
@pytest.mark.parametrize("clip", ["on", "off"])
def test_area_from_zero_reaches_panel_edge_under_log(clip):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 1000.0, 100.0]})
    p = (
        ggplot(df, aes("x", "y"))
        .add(geom_area())
        .add(geom_segment(aes(xend="x", yend=0)))
        .add(coord_trans(y="log10", ylim=(1, 1000), clip=clip))
        .build()
    )
    bottom = p.panel.axes.get_corner(DL)[1]
    for layer in (0, 1):
        for m in p.layers[layer].marks:
            pts = np.asarray(m.get_all_points())
            assert len(pts) and np.isfinite(pts).all()
            assert pts[:, 1].min() == pytest.approx(bottom, abs=1e-9)


def test_transform_squishes_domain_edge_and_infinities():
    df = pd.DataFrame({"x": [1.0, 100.0], "y": [1.0, 100.0]})
    panel = (
        ggplot(df, aes("x", "y")).add(geom_point()).add(coord_trans(y="log10", clip="off")).build()
    ).panel
    lo, hi = panel.axes.get_corner(DL)[1], panel.axes.get_corner(UR)[1]
    pts = panel.transform([1.0, 1.0, 1.0, 1.0], [0.0, np.inf, -1.0, -np.inf])
    assert pts[0, 1] == pytest.approx(lo)  # log(0) = -inf: bottom edge
    assert pts[1, 1] == pytest.approx(hi)  # +inf: top edge
    assert np.isnan(pts[2, 1]) and np.isnan(pts[3, 1])  # log of a negative: not drawn
    x, y = panel.squish_input([1.0, 2.0], [0.0, 5.0])
    np.testing.assert_allclose(y, [panel.panel_params.y_range[0], 5.0])


def test_reverse_log_domain_edge_goes_to_the_right_end():
    from ggmanim.coords_trans import _squished
    from ggmanim.scales.continuous import get_transform

    class NegLog:  # decreasing transform with an open lower end at 0
        domain = get_transform("log10").domain

        def transform(self, v):
            return -get_transform("log10").transform(v)

        def inverse(self, v):
            return get_transform("log10").inverse(-np.asarray(v, dtype=float))

    out = _squished(NegLog(), np.array([0.0, 10.0]), (-3.0, 0.5))
    np.testing.assert_allclose(out, [0.5, -1.0])


def test_points_at_domain_edge_drawn_at_edge_with_clip_off():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 10.0, 100.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = (
            ggplot(df, aes("x", "y"))
            .add(geom_point())
            .add(coord_trans(y="log10", ylim=(1, 100), clip="off"))
            .build()
        )
    ys = sorted(m.get_center()[1] for m in p.layers[0].marks)
    assert len(ys) == 3
    assert ys[0] == pytest.approx(p.panel.axes.get_corner(DL)[1], abs=1e-6)


def test_polar_panel_has_munch_distance():
    df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    panel = ggplot(df, aes("x", "y")).add(geom_point()).add(coord_polar()).build().panel
    d = panel.munch_distance([0.0, 10.0], [5.0, 5.0])
    assert d.shape == (1,) and d[0] > 0
