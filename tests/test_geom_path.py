"""geom_path: data-order paths (vs geom_line's x order), labels, clipping, dashing,
per-segment colour; per-segment geom_smooth line and label colours."""

import numpy as np
import pandas as pd
import pytest
from manim import BLUE, GREEN, RED, VGroup, VMobject

import ggmanim
from ggmanim import aes, after_stat, geom_path, ggplot, scale_color_viridis_c
from ggmanim.geoms import GeomPath
from ggmanim.layer import Layer
from ggmanim.registry import COMPONENTS

H = {c: c.to_hex().upper() for c in (RED, GREEN, BLUE)}

# a spiral-ish path whose x is not monotone
SPIRAL = pd.DataFrame({"x": [0.0, 2.0, 1.0, 3.0, 0.5], "y": [0.0, 1.0, 2.0, 3.0, 4.0]})


def _hex(m: VMobject) -> str:
    return m.get_stroke_color().to_hex().upper()


def _verts(m: VMobject) -> np.ndarray:
    return np.vstack([m.get_start_anchors(), m.get_end_anchors()[-1:]])


def _panel_points(m, x, y) -> np.ndarray:
    return m.panel.transform(np.asarray(x, float), np.asarray(y, float))


def test_registered_and_exported():
    assert "geom_path" in COMPONENTS and "geom_path" in ggmanim.__all__
    layer = geom_path(color=RED, name="p")
    assert isinstance(layer, Layer) and isinstance(layer.geom, GeomPath)
    assert type(layer.geom) is GeomPath  # not GeomLine
    assert hasattr(ggplot(SPIRAL, aes("x", "y")), "geom_path")


def test_order_preserved_not_sorted():
    m = ggplot(SPIRAL, aes("x", "y")).geom_path().build()
    (path,) = m.layers[0].marks
    np.testing.assert_allclose(_verts(path), _panel_points(m, SPIRAL.x, SPIRAL.y), atol=1e-9)
    # geom_line sorts the same data by x
    m2 = ggplot(SPIRAL, aes("x", "y")).geom_line().build()
    s = SPIRAL.sort_values("x")
    np.testing.assert_allclose(_verts(m2.layers[0].marks[0]), _panel_points(m2, s.x, s.y))


def test_label_at_last_point_in_data_order():
    df = pd.concat([SPIRAL.assign(g="a"), SPIRAL.assign(y=SPIRAL.y + 5, g="b")])
    m = ggplot(df, aes("x", "y", color="g")).geom_path(label=["A", "B"]).build()
    layer = m.layers[0]
    assert [t.text for t in layer.labels] == ["A", "B"]
    for path, lab in zip(layer.marks, layer.labels):
        assert _hex(lab) == _hex(path)
        end = path.get_end()  # last data row (x=0.5), not the max-x point
        assert lab.get_left()[0] > end[0]
        assert abs(lab.get_center()[1] - end[1]) < 0.3


def test_clipped_by_coord_limits():
    m = (
        ggplot(SPIRAL, aes("x", "y"))
        .geom_path()
        .coord_cartesian(ylim=(0.0, 2.5), expand=False)
        .build()
    )
    (path,) = m.layers[0].marks
    v = _verts(path)
    top = _panel_points(m, [0.0], [2.5])[0, 1]
    assert v[:, 1].max() <= top + 1e-9
    # starts at the first data row and keeps data order up to the cut
    np.testing.assert_allclose(v[:3], _panel_points(m, SPIRAL.x[:3], SPIRAL.y[:3]), atol=1e-9)


def test_dashed():
    m = ggplot(SPIRAL, aes("x", "y")).geom_path(linetype="dashed").build()
    (dashes,) = m.layers[0].marks
    assert isinstance(dashes, VGroup) and len(dashes) > 3


def test_per_segment_colour_in_data_order_and_label_uses_last_segment():
    df = SPIRAL.assign(c=["r", "r", "g", "b", "b"])
    pal = {"r": RED, "g": GREEN, "b": BLUE}
    m = ggplot(df, aes("x", "y", color="c")).add(
        geom_path(aes(group=1), label="end"), ggmanim.scale_color_manual(values=pal)
    )
    layer = m.build().layers[0]
    (segs,) = layer.marks
    assert [_hex(s) for s in segs] == [H[RED], H[GREEN], H[BLUE]]
    (lab,) = layer.labels
    assert lab.get_color().to_hex().upper() == H[BLUE]


def test_per_segment_label_colour_is_last_visible_segment():
    df = SPIRAL.assign(c=["r", "r", "g", "b", "b"])
    pal = {"r": RED, "g": GREEN, "b": BLUE}
    m = (
        ggplot(df, aes("x", "y", color="c", group=1))
        .geom_path(label="end")
        .scale_color_manual(values=pal)
        .coord_cartesian(ylim=(0.0, 2.5), expand=False)
        .build()
    )
    layer = m.layers[0]
    (segs,) = layer.marks
    assert [_hex(s) for s in segs] == [H[RED], H[GREEN]]  # BLUE segment fully outside
    assert layer.labels[0].get_color().to_hex().upper() == H[GREEN]


def test_continuous_colour_spiral_end_to_end():
    t = np.linspace(0, 4 * np.pi, 41)
    df = pd.DataFrame({"x": t * np.cos(t), "y": t * np.sin(t), "t": t})
    m = ggplot(df, aes("x", "y", color="t")).add(geom_path(), scale_color_viridis_c()).build()
    (segs,) = m.layers[0].marks
    assert len(segs) == 40 and len({_hex(s) for s in segs}) == 40


# -- geom_smooth ------------------------------------------------------------------


@pytest.fixture
def lin():
    x = np.linspace(0, 10, 21)
    return pd.DataFrame({"x": x, "y": 2 * x + np.sin(x)})


def test_smooth_constant_colour_unchanged(lin):
    m = ggplot(lin, aes("x", "y")).geom_smooth(se=False, label="fit").build()
    layer = m.layers[0]
    (line,) = layer.marks
    assert type(line) is VMobject
    assert _hex(line) == layer.labels[0].get_color().to_hex().upper()


def test_smooth_per_segment_colour_and_label(lin):
    m = (
        ggplot(lin, aes("x", "y"))
        .geom_smooth(aes(color=after_stat("y")), se=True, n=11, label="fit")
        .scale_color_viridis_c()
        .build()
    )
    layer = m.layers[0]
    (grp,) = layer.marks
    ribbon, segs = grp
    assert isinstance(segs, VGroup) and len(segs) == 10
    assert len({_hex(s) for s in segs}) == 10
    assert all(s.get_stroke_opacity() == pytest.approx(1.0) for s in segs)  # alpha is the ribbon's
    assert layer.labels[0].get_color().to_hex().upper() == _hex(segs[-1])
