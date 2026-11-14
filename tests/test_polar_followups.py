"""coord_polar follow-ups (ggm-bzi): munched reference lines, closed paths,
theta labels clear of the radius axis on small panels."""

import numpy as np
import pandas as pd
import pytest
from manim import VMobject

from ggmanim import (
    aes,
    coord_polar,
    geom_hline,
    geom_line,
    geom_path,
    geom_vline,
    ggplot,
    scale_y_continuous,
    theme,
    ylim,
)
from ggmanim.coords_polar import R_MAX

WAVE = pd.DataFrame({"x": np.arange(1, 9), "y": [3, 5, 4, 7, 6, 8, 5, 4]})
SKILLS = ["speed", "power", "range", "defence", "stamina"]
RADAR = pd.DataFrame({"skill": pd.Categorical(SKILLS, categories=SKILLS), "score": [8, 5, 7, 4, 6]})


def _pts(m: VMobject) -> np.ndarray:
    pts = np.vstack([m.get_start_anchors(), m.get_end_anchors()[-1:]])
    return pts[:, :2]  # every vertex, the last end included


def _polar(p):
    panel = p.panel
    centre = panel.polar_to_scene(0.0, 0.0)[0][:2]
    return panel, centre


def test_hline_is_a_circle_under_polar():
    p = ggplot(WAVE, aes("x", "y")).add(geom_line(), geom_hline(yintercept=5)).add(coord_polar())
    p = p.build()
    panel, centre = _polar(p)
    (circle,) = p.layers[1].marks
    pts = _pts(circle)
    assert len(pts) > 50  # munched, not a chord
    r = np.hypot(*(pts - centre).T)
    side = panel.axes.x_length
    np.testing.assert_allclose(r, float(panel.radius(5.0)) * side, atol=1e-6)
    # a full turn: the points go all round the centre
    ang = np.arctan2(*(pts - centre).T[::-1])
    assert np.ptp(ang) > 1.9 * np.pi


def test_vline_is_a_ray_under_polar():
    p = ggplot(WAVE, aes("x", "y")).add(geom_line(), geom_vline(xintercept=[2, 6]))
    p = p.add(coord_polar()).build()
    panel, centre = _polar(p)
    marks = p.layers[1].marks
    assert len(marks) == 2
    side = panel.axes.x_length
    for m, x in zip(marks, (2, 6)):
        d = _pts(m) - centre
        frac = (x - panel.theta_range[0]) / np.ptp(panel.theta_range)
        a = float(panel.angle(frac))
        u = np.array([np.sin(a), np.cos(a)])
        # every point on the ray through the centre at x's angle
        np.testing.assert_allclose(d[:, 0] * u[1] - d[:, 1] * u[0], 0.0, atol=1e-6)
        r = d @ u
        assert r.min() == pytest.approx(0.0, abs=1e-6)
        assert r.max() == pytest.approx(R_MAX * side)


def test_hline_with_inner_radius_and_theta_y():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 2.0]})
    p = ggplot(df, aes("x", "y")).add(geom_line(), geom_vline(xintercept=2))
    p = p.add(coord_polar(theta="y", inner_radius=0.3)).build()
    panel, centre = _polar(p)
    # theta="y": a vline (constant x = radius) is a circle
    r = np.hypot(*(_pts(p.layers[1].marks[0]) - centre).T)
    np.testing.assert_allclose(r, float(panel.radius(2.0)) * panel.axes.x_length, atol=1e-6)


def test_reference_lines_stay_straight_on_cartesian():
    p = ggplot(WAVE, aes("x", "y")).add(geom_hline(yintercept=5), geom_vline(xintercept=3))
    p = p.build()
    for layer in p.layers.values():
        assert len(layer.marks) == 1
        assert len(_pts(layer.marks[0])) == 2  # one straight Line


def test_reference_line_label_under_polar():
    p = ggplot(WAVE, aes("x", "y")).add(geom_hline(yintercept=5, label="mean"))
    p = p.add(coord_polar()).build()
    assert len(p.layers[0].labels) == 1


def test_closed_path_returns_to_start():
    df = pd.DataFrame({"x": [0.0, 1.0, 1.0], "y": [0.0, 0.0, 1.0]})
    m = ggplot(df, aes("x", "y")).add(geom_path(closed=True)).build().layers[0].marks[0]
    anchors = np.asarray(m.get_start_anchors())
    assert len(anchors) == 3  # 3 edges, the last one closing
    np.testing.assert_allclose(m.points[-1], m.points[0])
    open_ = ggplot(df, aes("x", "y")).add(geom_path()).build().layers[0].marks[0]
    assert len(open_.get_start_anchors()) == 2


def test_closed_line_single_row_group_is_harmless():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "g": ["a", "b", "b"]})
    p = ggplot(df, aes("x", "y", group="g")).add(geom_line(closed=True)).build()
    assert len(p.layers[0].marks) == 2


def test_closed_radar_edge_runs_forward_round_the_circle():
    p = ggplot(RADAR, aes("skill", "score", group=1)).add(geom_line(closed=True))
    p = p.add(coord_polar(), ylim(0, 10)).build()
    panel, centre = _polar(p)
    pts = _pts(p.layers[0].marks[0])
    np.testing.assert_allclose(pts[-1], pts[0], atol=1e-9)  # closed
    # clockwise angles from 12 o'clock, unwrapped: they only ever increase, so
    # the closing edge (stamina -> speed) does not sweep back across the circle
    d = pts - centre
    ang = np.unwrap(np.arctan2(d[:, 0], d[:, 1]))
    assert np.all(np.diff(ang) > -1e-6)
    assert ang[-1] - ang[0] == pytest.approx(2 * np.pi, abs=1e-6)


def test_closed_varying_colour_path():
    df = pd.DataFrame({"x": [0.0, 1.0, 1.0], "y": [0.0, 0.0, 1.0], "c": [0.0, 1.0, 2.0]})
    p = ggplot(df, aes("x", "y", color="c")).add(geom_path(closed=True)).build()
    segs = p.layers[0].marks[0]
    assert len(segs) == 3  # one stroke per edge, the closing one included


def _small(**kw):
    return (
        ggplot(WAVE, aes("x", "y"))
        .add(geom_line(), coord_polar(), theme(panel_width=1.8, panel_height=1.8))
        .add(*kw.get("extra", ()))
        .build()
    )


def test_theta_labels_clear_of_radius_axis_on_small_panels():
    panel = _small().panel
    from manim import Line

    axis = [m for m in panel.axes.submobjects if isinstance(m, Line)][-1]
    ax_x = axis.get_start()[0]
    lo = min(axis.get_start()[1], axis.get_end()[1])
    hi = max(axis.get_start()[1], axis.get_end()[1])
    half = panel.y_tick_labels[0].height / 2
    for lab in panel.x_tick_labels:
        if lab.get_bottom()[1] < hi + half and lab.get_top()[1] > lo - half:
            assert lab.get_left()[0] >= ax_x + 0.2  # tick_length + tick_label_buff
    # the radius labels stay left of the axis and of every theta label beside it
    for t in panel.y_tick_labels:
        assert t.get_right()[0] < ax_x


def test_radius_breaks_thinned_on_small_panels():
    big = ggplot(WAVE, aes("x", "y")).add(geom_line(), coord_polar()).build()
    small = _small()
    nb, ns = len(big.panel.panel_params.y_breaks), len(small.panel.panel_params.y_breaks)
    assert ns < nb
    assert len(small.panel.y_tick_labels) == ns
    tops = sorted((t.get_bottom()[1], t.get_top()[1]) for t in small.panel.y_tick_labels)
    assert all(b[0] > a[1] for a, b in zip(tops, tops[1:]))  # no overlaps


def test_user_radius_breaks_are_kept():
    small = _small(extra=(scale_y_continuous(breaks=[3, 4, 5, 6, 7, 8]),))
    assert small.panel.panel_params.y_breaks == (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
