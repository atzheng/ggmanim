"""coord_polar / coord_radar: pie, donut, rose and radar charts (ggm-aji.2)."""

import numpy as np
import pandas as pd
import pytest
from manim import VMobject

from ggmanim import (
    aes,
    coord_polar,
    coord_radar,
    facet_wrap,
    geom_bar,
    geom_col,
    geom_line,
    geom_polygon,
    ggplot,
    labs,
    theme_void,
    xlim,
)
from ggmanim.coords import coord_munch, is_linear
from ggmanim.coords_polar import R_LABEL, R_MAX, CoordPolar, PolarPanel
from ggmanim.registry import COMPONENTS

FRUIT = pd.DataFrame({"fruit": ["a"] * 6 + ["b"] * 3 + ["c"] * 2 + ["d"]})


def _pie(**kw):
    return (
        ggplot(FRUIT, aes(x=1, fill="fruit"))
        .add(geom_bar(width=1))
        .add(coord_polar(theta="y", **kw))
        .build()
    )


def _xy(m: VMobject) -> np.ndarray:
    return np.asarray(m.get_start_anchors())[:, :2]


def _polar(panel: PolarPanel, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scene points -> (radius in panel units, clockwise angle from 12 o'clock)."""
    c = panel.polar_to_scene(0.0, 0.0)[0, :2]
    side = panel.axes.x_length
    d = (pts - c) / side
    return np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 0], d[:, 1]) % (2 * np.pi)


def _arc_span(panel: PolarPanel, wedge: VMobject, r: float) -> float:
    """Angle swept by the wedge's vertices at radius ``r`` (in ring order)."""
    rad, ang = _polar(panel, _xy(wedge))
    on = np.isclose(rad, r, atol=1e-6)
    a = np.unwrap(ang[on])
    return float(abs(a[-1] - a[0]))


def test_registered_and_validated():
    assert {"coord_polar", "coord_radar"} <= set(COMPONENTS)
    p = ggplot().coord_polar(theta="y", start=0.5)
    assert isinstance(p.coord, CoordPolar)
    assert not is_linear(p.coord)
    assert repr(p.coord) == "coord_polar(theta='y', start=0.5, direction=1)"
    assert is_linear(coord_radar())
    for bad in ({"theta": "z"}, {"direction": 0}, {"clip": "maybe"}, {"inner_radius": 1.0}):
        with pytest.raises(ValueError):
            coord_polar(**bad)


def test_pie_wedge_angles_proportional_to_counts():
    p = _pie()
    panel = p.panel
    assert isinstance(panel, PolarPanel) and not panel.is_linear
    wedges = p.layers[0].marks
    counts = FRUIT["fruit"].value_counts().sort_index().to_numpy()
    spans = [_arc_span(panel, w, R_MAX) for w in wedges]
    np.testing.assert_allclose(spans, 2 * np.pi * counts / counts.sum(), atol=1e-6)
    # the wedges cover the full turn
    assert sum(spans) == pytest.approx(2 * np.pi)


def test_pie_wedges_are_munched_arcs():
    p = _pie()
    wedge = p.layers[0].marks[0]  # "a": half of the pie
    rad, _ = _polar(p.panel, _xy(wedge))
    assert np.isclose(rad, R_MAX).sum() > 20  # the outer edge is an arc, not a chord
    assert rad.min() == pytest.approx(0.0, abs=1e-9)  # the tip is the centre
    assert rad.max() == pytest.approx(R_MAX)


def test_theta_mapping_start_and_direction():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    for kw, quarter in (
        ({}, np.pi / 2),  # clockwise from 12 o'clock: a quarter turn is 3 o'clock
        ({"direction": -1}, 3 * np.pi / 2),
        ({"start": np.pi / 2}, np.pi),
    ):
        panel = ggplot(df, aes("x", "y")).geom_line().add(coord_polar(**kw)).build().panel
        assert panel.theta_range == (0.0, 1.0)  # continuous theta: no expansion
        assert panel.r_range == (0.0, 1.0)  # radius: no expansion
        pts = panel.transform([0.25, 0.25], [1.0, 0.5])[:, :2]
        rad, ang = _polar(panel, pts)
        np.testing.assert_allclose(rad, [R_MAX, R_MAX / 2])
        np.testing.assert_allclose(ang, quarter)


def test_donut_via_xlim_and_inner_radius():
    shares = pd.DataFrame({"x": 2, "k": list("abc"), "v": [5, 3, 2]})
    base = ggplot(shares, aes("x", "v", fill="k")).add(geom_col(width=1))
    p = base.add(coord_polar(theta="y"), xlim(0.5, 2.5)).build()
    assert p.panel.r_range == (0.5, 2.5)
    rad, _ = _polar(p.panel, _xy(p.layers[0].marks[0]))
    assert rad.min() == pytest.approx(R_MAX / 2)  # bars from x=1.5: half way out
    q = base.add(coord_polar(theta="y", inner_radius=0.5)).build()
    rad, _ = _polar(q.panel, _xy(q.layers[0].marks[0]))
    assert rad.min() == pytest.approx(R_MAX / 2)
    assert rad.max() == pytest.approx(R_MAX)


def test_rose_discrete_theta_equal_sectors():
    months = list("ABCDEFGHIJKL")
    df = pd.DataFrame({"m": pd.Categorical(months, categories=months), "n": np.arange(1, 13)})
    p = ggplot(df, aes("m", "n")).add(geom_col(width=1), coord_polar()).build()
    panel = p.panel
    assert panel.theta_range == (0.5, 12.5)  # half a level each side
    spans = [_arc_span(panel, w, float(panel.radius(n))) for w, n in zip(p.layers[0].marks, df.n)]
    np.testing.assert_allclose(spans, 2 * np.pi / 12, atol=1e-6)
    # theta labels: one per level, around the circle
    labels = panel.x_tick_labels
    assert len(labels) == 12
    centre = panel.polar_to_scene(0, 0)[0]
    dist = [np.hypot(*(lab.get_center() - centre)[:2]) for lab in labels]
    assert min(dist) > R_LABEL * panel.axes.x_length


def test_munch_distance_is_arc_length():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    panel = ggplot(df, aes("x", "y")).geom_line().add(coord_polar()).build().panel
    # a full turn at the outer radius; a straight on-screen measure would say 0
    d = panel.munch_distance([0.0, 1.0], [1.0, 1.0])
    assert d[0] == pytest.approx(2 * np.pi * R_MAX)
    mx, my = coord_munch(panel, [0.0, 1.0], [1.0, 1.0])
    assert len(mx) > 100
    # radial segment: its length in panel units
    assert panel.munch_distance([0.5, 0.5], [0.0, 1.0])[0] == pytest.approx(R_MAX)


def test_line_becomes_spiral_and_radar_stays_straight():
    df = pd.DataFrame({"x": [0.0, 0.5], "y": [0.0, 1.0]})
    line = ggplot(df, aes("x", "y")).add(geom_line(), coord_polar()).build()
    assert len(_xy(line.layers[0].marks[0])) > 10
    tri = pd.DataFrame({"k": pd.Categorical(list("abcde")), "v": [3, 5, 2, 4, 1]})
    radar = ggplot(tri, aes("k", "v")).add(geom_polygon(fill=None, color="red"), coord_radar())
    poly = radar.build().layers[0].marks[0]
    assert len(_xy(poly)) == 5  # one vertex per spoke: straight edges


def test_continuous_theta_end_labels_merge():
    p = _pie()
    texts = [lab.text for lab in p.panel.x_tick_labels]
    assert texts[0] == "0/12"
    assert "12" not in texts


def test_titles_and_r_axis_placement():
    p = (
        ggplot(FRUIT, aes(x=1, fill="fruit"))
        .add(geom_bar(width=1), coord_polar(theta="y"), labs(x="radius", y="angle"))
        .build()
    )
    panel = p.panel
    centre = panel.polar_to_scene(0, 0)[0]
    assert panel.x_title.text == "angle"  # theta title below
    assert panel.x_title.get_top()[1] < panel.axes.get_bottom()[1]
    assert panel.y_title.text == "radius"  # radius title on the left
    assert panel.y_title.get_right()[0] < panel.axes.get_left()[0]
    # r axis ticks at the radius breaks, above the centre, left of the theta labels
    tick_y = [t.get_center()[1] for t in panel.y_ticks]
    assert len(tick_y) == len(panel.panel_params.x_breaks) and min(tick_y) >= centre[1]
    assert max(t.get_right()[0] for t in panel.y_ticks) <= panel.axes.get_left()[0] + 1e-9


def test_theme_void_draws_only_the_data():
    q = (
        ggplot(FRUIT, aes(x=1, fill="fruit"))
        .add(geom_bar(width=1), coord_polar(theta="y"), theme_void())
        .build()
    )
    panel = q.panel
    assert len(panel.grid) == 0 and len(panel.x_tick_labels) == 0
    assert len(panel.y_ticks) == 0 and panel.x_title is None and panel.y_title is None


def test_panel_is_square_and_grid_has_circles_and_spokes():
    p = _pie()
    panel = p.panel
    assert panel.axes.x_length == pytest.approx(panel.axes.y_length)
    n_r = len(panel.panel_params.x_breaks)
    n_t = len(panel.panel_params.y_breaks)
    # spokes at every theta break, circles at the radius breaks + the outer one
    assert len(panel.grid_major) == n_t + n_r + 1


def test_facets_build_polar_panels():
    df = FRUIT.assign(g=["p", "q"] * 6)
    p = (
        ggplot(df, aes(x=1, fill="fruit"))
        .add(geom_bar(width=1), coord_polar(theta="y"), facet_wrap("g"))
        .build()
    )
    assert all(isinstance(pan, PolarPanel) for pan in p.panels.values())
