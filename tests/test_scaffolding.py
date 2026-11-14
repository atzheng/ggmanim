"""Parallel-work hooks: positions registry/pipeline, attach_marks and the key aesthetic."""

import numpy as np
import pandas as pd
import pytest
from manim import VGroup

from ggmanim import aes, geom_col, geom_hline, geom_point, geom_smooth, ggplot
from ggmanim.build import BuildError, build_plot
from ggmanim.geoms import Geom, GeomPoint, attach_marks
from ggmanim.layer import Layer
from ggmanim.positions import POSITIONS, Position, PositionIdentity, get_position
from ggmanim.positions import position_identity as pos_identity

# -- positions --------------------------------------------------------------------


def test_get_position_resolution():
    assert isinstance(get_position(None), PositionIdentity)
    assert isinstance(get_position("identity"), PositionIdentity)
    assert isinstance(get_position("position_identity"), PositionIdentity)
    assert isinstance(get_position(PositionIdentity), PositionIdentity)
    p = pos_identity()
    assert get_position(p) is p
    assert POSITIONS["identity"] is PositionIdentity
    with pytest.raises(ValueError, match="Unknown position 'nope'"):
        get_position("nope")
    with pytest.raises(TypeError):
        get_position(3)
    with pytest.raises(TypeError):
        get_position(p, width=1)


def test_layers_resolve_positions():
    assert isinstance(geom_point().position, PositionIdentity)
    assert isinstance(geom_point(position=None).position, PositionIdentity)
    assert isinstance(Layer(geom=GeomPoint()).position, PositionIdentity)
    assert Geom.default_position == "identity"
    with pytest.raises(ValueError, match="Unknown position"):
        geom_point(position="stack_not_yet")


def test_subclass_registers_and_runs_after_setup_data():
    seen = {}

    class PositionShift(Position):
        name = "test_shift"

        def setup_params(self, data):
            return {"by": self.params.get("by", 1.0)}

        def compute_panel(self, data, params, scales):
            seen["cols"] = set(data.columns)
            seen["scales"] = set(scales or {})
            by = params["by"]
            return data.assign(x=data.x + by, xmin=data.xmin + by, xmax=data.xmax + by)

    try:
        assert POSITIONS["test_shift"] is PositionShift
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        m = ggplot(df, aes("x", "y")).add(geom_col(position=PositionShift(by=10.0))).build()
        # the position saw geom_col's setup_data output (bar extents) ...
        assert {"xmin", "xmax", "ymin", "ymax", "PANEL", "group"} <= seen["cols"]
        assert seen["scales"] == {"x", "y"}
        # ... and the position scales were retrained on the shifted data
        assert m.scales["x"].get_limits()[0] > 10
        np.testing.assert_allclose(m.layer_data[0]["x"], [11.0, 12.0, 13.0])
    finally:
        POSITIONS.pop("test_shift", None)


def test_position_errors_name_the_layer():
    class PositionBoom(Position):
        def compute_panel(self, data, params, scales):
            raise RuntimeError("boom")

    df = pd.DataFrame({"x": [1.0], "y": [1.0]})
    with pytest.raises(BuildError, match="layer 0 .*PositionBoom failed: boom"):
        build_plot(ggplot(df, aes("x", "y")).add(geom_point(position=PositionBoom())))
    assert "PositionBoom" not in POSITIONS  # no ``name`` in the class body: not registered


def test_compute_layer_splits_by_panel():
    calls = []

    class PositionCount(Position):
        def compute_panel(self, data, params, scales):
            calls.append(sorted(data["PANEL"].unique()))
            return data

    df = pd.DataFrame({"x": [1, 2, 3, 4], "PANEL": [2, 1, 2, 1]}, index=[10, 11, 12, 13])
    out = PositionCount().compute_layer(df, {}, None)
    assert calls == [[1], [2]]
    pd.testing.assert_frame_equal(out, df)  # original row order restored


# -- marks / keys -----------------------------------------------------------------


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 1.0, 4.0, 3.0],
            "g": ["a", "a", "b", "b"],
            "id": ["p", "q", "r", "s"],
        }
    )


def test_point_marks_default_to_row_index(df):
    layer = ggplot(df, aes("x", "y")).geom_point().build().layers[0]
    assert layer.marks == list(layer.submobjects)
    assert layer.keys == [0, 1, 2, 3]


def test_point_marks_use_key_aesthetic(df):
    layer = ggplot(df, aes("x", "y", key="id")).geom_point().build().layers[0]
    assert layer.keys == ["p", "q", "r", "s"]
    # key is never scaled and does not split groups
    m = ggplot(df, aes("x", "y", key="id")).geom_line().build()
    assert len(m.layers[0].marks) == 1
    assert "key" not in m.scales


def test_clipped_points_drop_their_keys(df):
    from ggmanim import coord_cartesian

    p = ggplot(df, aes("x", "y", key="id")).geom_point().add(coord_cartesian(xlim=(0.5, 2.5)))
    layer = p.build().layers[0]
    assert len(layer.marks) == len(layer.keys) < 4
    assert set(layer.keys) <= {"p", "q", "r"}


def test_group_marks(df):
    m = ggplot(df, aes("x", "y", color="g")).geom_line(label="lab").build()
    layer = m.layers[0]
    assert layer.keys == [1, 2]
    assert len(layer.marks) == 2 and len(layer.labels) == 2
    assert all(mk not in layer.labels for mk in layer.marks)
    keyed = ggplot(df.assign(k=["A", "A", "B", "B"]), aes("x", "y", color="g", key="k"))
    assert keyed.geom_line().build().layers[0].keys == ["A", "B"]


def test_key_survives_stats(df):
    data = pd.concat([df, df.assign(y=df.y + 1)], ignore_index=True)
    data["k"] = data["g"].map({"a": "first", "b": "second"})
    m = ggplot(data, aes("x", "y", color="g", key="k")).add(geom_smooth()).build()
    assert m.layers[0].keys == ["first", "second"]
    assert "key" in m.layer_data[0].columns


def test_col_and_refline_marks(df):
    bars = ggplot(df, aes("x", "y", key="id")).geom_col().build().layers[0]
    assert bars.keys == ["p", "q", "r", "s"] and len(bars.marks) == 4
    lines = ggplot(df, aes("x", "y")).add(geom_hline(yintercept=[1.5, 2.5])).build().layers[0]
    assert lines.keys == [0, 1] and len(lines.marks) == 2


def test_attach_marks_contract():
    from manim import Dot

    g = VGroup(Dot(), Dot())
    data = pd.DataFrame({"group": [2, 2, 5]})
    with pytest.raises(ValueError, match="2 mark"):
        attach_marks(g, data, per="row")
    assert attach_marks(g, data, per="group") is g
    assert g.keys == [2, 5] and g.marks == [g[0], g[1]]
    with pytest.raises(ValueError, match="per must be"):
        attach_marks(g, data, per="panel")
    empty = attach_marks(VGroup(), pd.DataFrame(), per="group")
    assert empty.marks == [] and empty.keys == []


def test_default_draw_panel_attaches_group_marks():
    from manim import Dot

    class GeomDot(Geom):
        required_aes = ("x", "y")

        def draw_group(self, data, panel_params, coord, theme, **params):
            return Dot()

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "group": [1, 1, 2]})
    out = GeomDot().draw_panel(df, None, None)
    assert out.keys == [1, 2] and len(out.marks) == 2
