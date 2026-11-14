"""geom_ribbon / geom_area: polygons per group, outlines, NA breaks, clipping, stacking."""

import numpy as np
import pandas as pd
import pytest
from manim import VGroup, VMobject

from ggmanim import aes, coord_cartesian, coord_flip, geom_area, geom_ribbon, ggplot
from ggmanim.build import BuildError
from ggmanim.geoms import GeomArea, GeomRibbon
from ggmanim.geoms.area import _DeferredPosition
from ggmanim.positions import POSITIONS, Position
from ggmanim.registry import COMPONENTS

RIB = pd.DataFrame({"x": [2.0, 0.0, 1.0], "lo": [0.0, 1.0, 0.5], "hi": [2.0, 3.0, 2.5]})


def _ring(m: VMobject) -> np.ndarray:
    return np.asarray(m.get_start_anchors())


def _fill(mark):
    return mark[0] if isinstance(mark, VGroup) else mark


@pytest.mark.parametrize("name", ["geom_ribbon", "geom_area"])
def test_registered(name):
    assert name in COMPONENTS


def test_ribbon_polygon_sorted_by_x():
    g = ggplot(RIB, aes("x", ymin="lo", ymax="hi")).add(geom_ribbon()).build()
    mob = g.layers[0]
    assert len(mob.marks) == 1 and mob.keys == [-1]
    want = g.panel.transform([0, 1, 2, 2, 1, 0], [3, 2.5, 2, 0, 0.5, 1])
    assert np.allclose(_ring(mob.marks[0]), want, atol=1e-6)
    m = mob.marks[0]
    assert m.get_stroke_width() == 0 and m.get_fill_opacity() == pytest.approx(1.0)


def test_ribbon_groups_one_mark_each():
    df = pd.concat([RIB.assign(g="a"), (RIB + 5).assign(g="b")], ignore_index=True)
    g = ggplot(df, aes("x", ymin="lo", ymax="hi", fill="g")).add(geom_ribbon(alpha=0.3)).build()
    mob = g.layers[0]
    assert mob.keys == [1, 2]
    a, b = mob.marks
    assert a.get_fill_opacity() == pytest.approx(0.3)
    assert a.get_fill_color().to_hex() != b.get_fill_color().to_hex()


@pytest.mark.parametrize(
    "outline_type, n_lines",
    [("both", 2), ("upper", 1), ("lower", 1), ("full", 1)],
)
def test_ribbon_outline_types(outline_type, n_lines):
    g = (
        ggplot(RIB, aes("x", ymin="lo", ymax="hi"))
        .add(geom_ribbon(outline_type=outline_type, color="#FFFFFF", alpha=0.5))
        .build()
    )
    mark = g.layers[0].marks[0]
    assert isinstance(mark, VGroup) and len(mark) == 2
    fill, outline = mark
    assert fill.get_stroke_width() == 0
    assert outline.get_fill_opacity() == 0
    assert outline.get_stroke_opacity() == pytest.approx(1.0)  # alpha is the fill's only
    subs = outline.get_subpaths()
    assert len(subs) == n_lines
    upper = g.panel.transform([0, 1, 2], [3, 2.5, 2])
    if outline_type in ("both", "upper"):
        assert np.allclose(np.asarray(subs[0])[[0, 4, -1]], upper, atol=1e-6)
    if outline_type == "full":
        anchors = np.asarray(subs[0])[::4]
        assert len(anchors) == 6  # closed: back to the start


def test_ribbon_outline_type_invalid():
    with pytest.raises(ValueError, match="outline_type"):
        geom_ribbon(outline_type="x")
    with pytest.raises(ValueError, match="outline_type"):
        geom_area(outline_type="both sides")
    with pytest.raises(ValueError, match="outline_type"):
        GeomRibbon().draw_panel(RIB.rename(columns={"lo": "ymin", "hi": "ymax"}), None, None,
                                outline_type="side")  # fmt: skip


def test_ribbon_dashed_outline():
    g = (
        ggplot(RIB, aes("x", ymin="lo", ymax="hi"))
        .add(geom_ribbon(color="#FFFFFF", linetype="dashed"))
        .build()
    )
    fill, outline = g.layers[0].marks[0]
    assert isinstance(outline, VGroup) and len(outline) > 2


def test_ribbon_missing_values_break_the_band():
    df = pd.DataFrame(
        {"x": [0, 1, 2, 3, 4, 5], "lo": [0, 0, np.nan, 0, 0, 0], "hi": [1, 1, 1, 1, 1, 1]}
    )
    g = ggplot(df, aes("x", ymin="lo", ymax="hi")).add(geom_ribbon()).build()
    m = g.layers[0].marks[0]
    assert len(m.get_subpaths()) == 2  # x 0..1 and x 3..5
    assert len(_ring(m)) == 4 + 6


def test_ribbon_clipped_to_panel():
    g = (
        ggplot(RIB, aes("x", ymin="lo", ymax="hi"))
        .add(geom_ribbon(), coord_cartesian(ylim=(1, 2), expand=False))
        .build()
    )
    ys = _ring(g.layers[0].marks[0])[:, 1]
    y0, y1 = g.panel.transform([0, 0], [1, 2])[:, 1]
    assert ys.min() == pytest.approx(y0) and ys.max() == pytest.approx(y1)


def test_ribbon_coord_flip():
    g = ggplot(RIB, aes("x", ymin="lo", ymax="hi")).add(geom_ribbon(), coord_flip()).build()
    want = g.panel.transform([0, 1, 2, 2, 1, 0], [3, 2.5, 2, 0, 0.5, 1])
    assert np.allclose(_ring(g.layers[0].marks[0]), want, atol=1e-6)


def test_draw_key_filled_square():
    key = GeomRibbon().draw_key({"fill": "#ABCDEF", "alpha": 0.5})
    assert key.get_fill_opacity() == pytest.approx(0.5)
    assert key.get_fill_color().to_hex().upper() == "#ABCDEF"


# -- geom_area ------------------------------------------------------------------------

AREA = pd.DataFrame({"x": [0, 1, 2] * 2, "y": [1.0, 2.0, 1.0, 2.0, 1.0, 3.0], "g": list("aaabbb")})


def test_area_setup_data_from_zero():
    out = GeomArea().setup_data(AREA, {})
    assert list(out["ymin"]) == [0] * 6 and list(out["ymax"]) == list(AREA["y"])


def test_area_identity_position():
    g = ggplot(AREA, aes("x", "y", fill="g")).add(geom_area(position="identity")).build()
    mob = g.layers[0]
    assert len(mob.marks) == 2
    want = g.panel.transform([0, 1, 2, 2, 1, 0], [1, 2, 1, 0, 0, 0])
    assert np.allclose(_ring(_fill(mob.marks[0])), want, atol=1e-6)
    # y scale includes 0
    assert g.panel.panel_params.y_range[0] <= 0


def test_area_upper_outline_by_default():
    g = (
        ggplot(AREA, aes("x", "y", fill="g"))
        .add(geom_area(position="identity", color="#FFFFFF"))
        .build()
    )
    fill, outline = g.layers[0].marks[0]
    assert len(outline.get_subpaths()) == 1
    top = g.panel.transform([0, 1, 2], [1, 2, 1])
    assert np.allclose(np.asarray(outline.get_subpaths()[0])[[0, 4, -1]], top, atol=1e-6)


def test_area_default_position_is_stack():
    layer = geom_area()
    pos = layer.position
    if isinstance(pos, _DeferredPosition):
        assert pos.target == "stack"
    else:
        assert type(pos) is POSITIONS["stack"]
    assert GeomArea.default_position == "stack"
    assert type(geom_area(position="identity").position).__name__ == "PositionIdentity"


@pytest.mark.skipif("stack" in POSITIONS, reason="position_stack is registered")
def test_area_stack_unavailable_fails_at_build_not_construction():
    p = ggplot(AREA, aes("x", "y", fill="g")).add(geom_area())  # constructing is fine
    with pytest.raises(BuildError, match="stack"):
        p.build()


class _FakeStack(Position):
    """Minimal stacker for the test: ymax cumulated per x in group order."""

    def compute_panel(self, data, params, scales=None):
        data = data.sort_values(["x", "group"], kind="stable").copy()
        height = data["ymax"] - data["ymin"]
        data["ymax"] = height.groupby(data["x"]).cumsum()
        data["ymin"] = data["ymax"] - height
        return data.sort_index()


def test_area_deferred_stack_resolved_at_build(monkeypatch):
    monkeypatch.delitem(POSITIONS, "stack", raising=False)
    layer = geom_area()
    assert isinstance(layer.position, _DeferredPosition)
    monkeypatch.setitem(POSITIONS, "stack", _FakeStack)  # registered later
    g = ggplot(AREA, aes("x", "y", fill="g")).add(layer).build()
    data = g.layer_data[0]
    top = data[data["group"] == 2]
    assert list(top["ymin"]) == [1, 2, 1] and list(top["ymax"]) == [3, 3, 4]
    want = g.panel.transform([0, 1, 2, 2, 1, 0], [3, 3, 4, 1, 2, 1])
    assert np.allclose(_ring(_fill(g.layers[0].marks[1])), want, atol=1e-6)
    assert g.panel.panel_params.y_range[1] >= 4  # scales retrained on stacked extents


@pytest.mark.skipif("stack" not in POSITIONS, reason="position_stack not registered yet")
def test_area_stacked_with_position_stack():
    g = ggplot(AREA, aes("x", "y", fill="g")).add(geom_area()).build()
    data = g.layer_data[0]
    tops = data.groupby("x")["ymax"].max().to_numpy()
    assert np.allclose(tops, [3, 3, 4])
    assert len(g.layers[0].marks) == 2
