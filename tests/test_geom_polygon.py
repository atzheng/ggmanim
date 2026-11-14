"""geom_polygon: vertices in data order, one mark per group, holes, clipping, flip."""

import numpy as np
import pandas as pd
import pytest
from manim import VMobject

from ggmanim import aes, coord_cartesian, coord_flip, geom_polygon, ggplot
from ggmanim.geoms import GeomPolygon
from ggmanim.geoms._rect import orient_rings, signed_area
from ggmanim.registry import COMPONENTS

TRI = pd.DataFrame({"x": [0, 4, 2, 5, 7, 7, 5], "y": [0, 0, 3, 0, 0, 2, 2], "id": list("aaabbbb")})


def _rings(m: VMobject) -> list[np.ndarray]:
    return [np.asarray(sp)[:-1] for sp in m.get_subpaths()]


def _anchors(m: VMobject) -> np.ndarray:
    return np.asarray(m.get_start_anchors())


def test_registered():
    assert "geom_polygon" in COMPONENTS


def test_vertices_in_data_order_one_mark_per_group():
    g = ggplot(TRI, aes("x", "y", group="id")).add(geom_polygon()).build()
    mob = g.layers[0]
    assert len(mob.marks) == 2 and mob.keys == [1, 2]
    for m, (_, gdf) in zip(mob.marks, TRI.groupby("id")):
        want = g.panel.transform(gdf["x"].to_numpy(float), gdf["y"].to_numpy(float))
        got = _anchors(m)
        # the ring may be reversed to run counter-clockwise; same vertices, same cycle
        if signed_area(want) < 0:
            want = want[::-1]
            k = int(np.argmin(np.linalg.norm(got - want[0], axis=1)))
            got = np.roll(got, -k, axis=0)
        assert np.allclose(got, want, atol=1e-6)
        assert m.get_fill_opacity() == pytest.approx(1.0)
        assert m.get_stroke_width() == 0


def test_style_from_first_row_and_alpha_fill_only():
    g = (
        ggplot(TRI, aes("x", "y", group="id", fill="id"))
        .add(geom_polygon(color="#FFFFFF", alpha=0.4, linewidth=2))
        .build()
    )
    a, b = g.layers[0].marks
    assert a.get_fill_color().to_hex() != b.get_fill_color().to_hex()
    assert a.get_fill_opacity() == pytest.approx(0.4)
    assert a.get_stroke_opacity() == pytest.approx(1.0) and a.get_stroke_width() == 2


def test_subgroup_holes_have_opposite_orientation():
    t = np.linspace(0, 2 * np.pi, 9)[:-1]
    outer = pd.DataFrame({"x": 4 * np.cos(t), "y": 4 * np.sin(t), "s": "outer"})
    hole = pd.DataFrame({"x": np.cos(t), "y": np.sin(t), "s": "hole"})  # same direction
    df = pd.concat([outer, hole], ignore_index=True).assign(id=1)
    g = ggplot(df, aes("x", "y", group="id", subgroup="s")).add(geom_polygon()).build()
    mob = g.layers[0]
    assert len(mob.marks) == 1
    rings = _rings(mob.marks[0])
    assert len(rings) == 2
    assert signed_area(rings[0]) > 0 and signed_area(rings[1]) < 0
    # subgroups in order of appearance: the outer ring comes first
    assert np.ptp(rings[0][:, 0]) > np.ptp(rings[1][:, 0])


def test_orient_rings_even_odd():
    def sq(s):
        return np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=float)

    rings = orient_rings([sq(3)[::-1], sq(2), sq(1)[::-1], sq(1) + 10])
    assert signed_area(rings[0]) > 0  # depth 0: CCW
    assert signed_area(rings[1]) < 0  # depth 1 (hole): CW
    assert signed_area(rings[2]) > 0  # depth 2 (island in the hole): CCW
    assert signed_area(rings[3]) > 0  # separate island


def test_clipped_to_panel():
    g = (
        ggplot(TRI, aes("x", "y", group="id"))
        .add(geom_polygon(), coord_cartesian(xlim=(0, 6), ylim=(0, 3), expand=False))
        .build()
    )
    m = g.layers[0].marks[1]  # square 5..7 x 0..2 cut at x = 6
    x0, x1 = g.panel.transform([5, 6], [0, 0])[:, 0]
    xs = _anchors(m)[:, 0]
    assert xs.min() == pytest.approx(x0) and xs.max() == pytest.approx(x1)
    g = (
        ggplot(TRI, aes("x", "y", group="id"))
        .add(geom_polygon(), coord_cartesian(xlim=(-3, -1), expand=False))
        .build()
    )
    assert g.layers[0].marks == [] and g.layers[0].keys == []


def test_coord_flip():
    g = ggplot(TRI, aes("x", "y", group="id")).add(geom_polygon(), coord_flip()).build()
    pts = _anchors(g.layers[0].marks[0])
    want = g.panel.transform([0, 4, 2], [0, 0, 3])
    for p in want:
        assert np.linalg.norm(pts - p, axis=1).min() < 1e-6


def test_draw_key():
    key = GeomPolygon().draw_key({"fill": "#123456"})
    assert key.get_fill_color().to_hex().upper() == "#123456"
