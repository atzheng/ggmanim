"""facet_wrap (ggm-913.2): layout table, grid shape, axes, strips, computed facets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from manim import config

from ggmanim import (
    aes,
    facet_wrap,
    geom_hline,
    geom_point,
    ggplot,
    label_both,
    labeller,
)
from ggmanim.build import BuildError
from ggmanim.facets import FacetWrap, wrap_dims


def seven(n_per: int = 3) -> pd.DataFrame:
    """Seven groups ``a``..``g``; group ``i`` has y in ``[10 i, 10 i + 2]``."""
    rows = []
    for i, g in enumerate("abcdefg"):
        for k in range(n_per):
            rows.append({"x": float(k), "y": 10.0 * i + k, "g": g})
    return pd.DataFrame(rows)


def layout_of(facet, *frames):
    return facet.compute_layout(list(frames) or [seven()])


def cells(layout):
    return [(int(r.ROW), int(r.COL)) for r in layout.itertuples(index=False)]


# --------------------------------------------------------------------------- dims
@pytest.mark.parametrize(
    "n, expected",
    [(1, (1, 1)), (2, (1, 2)), (3, (1, 3)), (4, (2, 2)), (5, (2, 3)), (7, (3, 3)),
     (10, (3, 4)), (13, (4, 4)), (17, (4, 5))],
)  # fmt: skip
def test_wrap_dims_default(n, expected):
    assert wrap_dims(n) == expected


def test_wrap_dims_given():
    assert wrap_dims(7, nrow=2) == (2, 4)
    assert wrap_dims(7, ncol=2) == (4, 2)
    assert wrap_dims(7, 3, 3) == (3, 3)
    with pytest.raises(ValueError, match="at least"):
        wrap_dims(7, 2, 3)


# --------------------------------------------------------------------------- layout
def test_layout_table_seven_panels():
    layout = layout_of(facet_wrap("g"))
    assert list(layout.columns) == ["PANEL", "ROW", "COL", "SCALE_X", "SCALE_Y", "g"]
    assert list(layout["PANEL"]) == list(range(1, 8))
    assert list(layout["g"]) == list("abcdefg")
    assert cells(layout) == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1)]
    assert set(layout["SCALE_X"]) == {1} and set(layout["SCALE_Y"]) == {1}


def test_layout_scale_ids():
    free_y = layout_of(facet_wrap("g", scales="free_y"))
    assert set(free_y["SCALE_X"]) == {1}
    assert list(free_y["SCALE_Y"]) == list(range(1, 8))
    free = layout_of(facet_wrap("g", scales="free"))
    assert list(free["SCALE_X"]) == list(free["SCALE_Y"]) == list(range(1, 8))


def test_layout_dir_v_fills_columns():
    layout = layout_of(facet_wrap("g", dir="v"))
    assert cells(layout) == [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3)]
    layout = layout_of(facet_wrap("g", dir="v", nrow=2))
    assert cells(layout) == [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3), (1, 4)]


def test_layout_nrow_ncol_and_as_table():
    assert cells(layout_of(facet_wrap("g", ncol=2)))[-1] == (4, 1)
    assert cells(layout_of(facet_wrap("g", nrow=2)))[:5] == [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1)]
    flipped = layout_of(facet_wrap("g", as_table=False))
    assert cells(flipped) == [(3, 1), (3, 2), (3, 3), (2, 1), (2, 2), (2, 3), (1, 1)]


def test_layout_multiple_vars_and_drop():
    df = pd.DataFrame({"a": ["p", "p", "q"], "b": [1, 2, 1], "x": 0.0, "y": 0.0})
    layout = layout_of(facet_wrap(["a", "b"]), df)
    assert list(zip(layout["a"], layout["b"], strict=True)) == [("p", 1), ("p", 2), ("q", 1)]
    full = layout_of(facet_wrap("~ a + b", drop=False), df)
    assert len(full) == 4 and list(full["b"]) == [1, 2, 1, 2]
    cat = df.assign(a=pd.Categorical(df["a"], categories=["p", "q", "r"]))
    assert list(layout_of(facet_wrap("a", drop=False), cat)["a"]) == ["p", "q", "r"]
    assert list(layout_of(facet_wrap("a"), cat)["a"]) == ["p", "q"]


def test_categorical_order_respected():
    df = seven().assign(g=lambda d: pd.Categorical(d["g"], categories=list("gfedcba")))
    assert list(layout_of(facet_wrap("g"), df)["g"]) == list("gfedcba")


def test_bad_arguments():
    with pytest.raises(ValueError, match="dir"):
        facet_wrap("g", dir="x")
    with pytest.raises(ValueError, match="strip_position"):
        facet_wrap("g", strip_position="middle")
    with pytest.raises(ValueError, match="nrow"):
        facet_wrap("g", nrow=0)
    with pytest.raises(ValueError, match="scales"):
        facet_wrap("g", scales="loose")
    with pytest.raises(TypeError):
        facet_wrap(3)
    with pytest.raises(BuildError, match="nope"):
        ggplot(seven(), aes("x", "y")).add(geom_point()).add(facet_wrap("nope")).build()


# --------------------------------------------------------------------------- build
def test_seven_panels_axes_and_strips():
    plot = ggplot(seven(), aes("x", "y")).add(geom_point()).add(facet_wrap("g")).build()
    assert set(plot.panels) == {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1)}
    has_x = {rc for rc, p in plot.panels.items() if len(p.x_tick_labels)}
    has_y = {rc for rc, p in plot.panels.items() if len(p.y_tick_labels)}
    # bottom panel of each column: (3,1) and, above the gap, (2,2) and (2,3)
    assert has_x == {(3, 1), (2, 2), (2, 3)}
    assert has_y == {(1, 1), (2, 1), (3, 1)}
    assert len(plot.strips) == 7
    assert all(s.side == "top" for s in plot.strips)
    assert plot.width <= config.frame_width + 1e-6
    # each panel holds its own rows
    assert all(len(plot.layers[0].panel(*rc).marks) == 3 for rc in plot.panels)


def test_free_y_ranges_differ_and_axes_everywhere():
    plot = (
        ggplot(seven(), aes("x", "y"))
        .add(geom_point())
        .add(facet_wrap("g", scales="free_y"))
        .build()
    )
    lims = [plot.panel_scales[rc]["y"].get_limits() for rc in sorted(plot.panels)]
    assert lims[0] == pytest.approx((0, 2)) and lims[-1] == pytest.approx((60, 62))
    assert len({tuple(lim) for lim in lims}) == 7
    assert plot.panel_scales[(1, 1)]["x"] is plot.panel_scales[(1, 2)]["x"]
    assert all(len(p.y_tick_labels) for p in plot.panels.values())
    has_x = {rc for rc, p in plot.panels.items() if len(p.x_tick_labels)}
    assert has_x == {(3, 1), (2, 2), (2, 3)}


def test_free_x_axes_on_every_panel():
    plot = (
        ggplot(seven(), aes("x", "y"))
        .add(geom_point())
        .add(facet_wrap("g", scales="free_x"))
        .build()
    )
    assert all(len(p.x_tick_labels) for p in plot.panels.values())
    assert not len(plot.panels[(1, 2)].y_tick_labels)


def test_label_both_strips():
    plot = (
        ggplot(seven(), aes("x", "y"))
        .add(geom_point())
        .add(facet_wrap("g", labeller=label_both))
        .build()
    )
    facet = facet_wrap("g", labeller=label_both)
    layout = layout_of(facet)
    specs = facet.strips(layout)
    assert specs[1] == [("top", ["g: a"])] and specs[7] == [("top", ["g: g"])]
    assert len(plot.strips) == 7


def test_labeller_and_multiple_vars_strips():
    df = pd.DataFrame({"a": ["p", "q"], "b": [1, 2], "x": [0.0, 1.0], "y": [0.0, 1.0]})
    facet = facet_wrap(["a", "b"], labeller=labeller(a={"p": "Pea"}, b=label_both))
    specs = facet.strips(layout_of(facet, df))
    assert specs[1] == [("top", ["Pea", "b: 1"])]
    assert specs[2] == [("top", ["q", "b: 2"])]


@pytest.mark.parametrize("side", ["top", "bottom", "left", "right"])
def test_strip_position(side):
    plot = (
        ggplot(seven(), aes("x", "y"))
        .add(geom_point())
        .add(facet_wrap("g", strip_position=side))
        .build()
    )
    assert {s.side for s in plot.strips} == {side}
    panel = plot.panels[(1, 1)]
    strip = plot.panel_grid.panel_strips[(1, 1)][0]
    area = panel.axes
    if side == "top":
        assert strip.get_bottom()[1] >= area.get_top()[1] - 1e-6
    elif side == "bottom":
        assert strip.get_top()[1] <= area.get_bottom()[1] + 1e-6
    elif side == "left":
        assert strip.get_right()[0] <= area.get_left()[0] + 1e-6
    else:
        assert strip.get_left()[0] >= area.get_right()[0] - 1e-6


def test_layers_without_facet_var_repeat():
    plot = (
        ggplot(seven(), aes("x", "y"))
        .add(geom_point())
        .add(geom_hline(yintercept=5))
        .add(facet_wrap("g"))
        .build()
    )
    assert all(len(plot.layers[1].panel(*rc).marks) == 1 for rc in plot.panels)


def test_dir_v_build_grid():
    plot = (
        ggplot(seven(), aes("x", "y"))
        .add(geom_point())
        .add(facet_wrap("g", dir="v", nrow=2))
        .build()
    )
    assert (1, 4) in plot.panels and (2, 4) not in plot.panels
    # panel (1,4) sits above a gap: it draws its own x axis
    assert len(plot.panels[(1, 4)].x_tick_labels)
    assert not len(plot.panels[(1, 3)].x_tick_labels)
    x14 = plot.panels[(1, 4)].axes.get_center()[0]
    x13 = plot.panels[(1, 3)].axes.get_center()[0]
    assert x14 > x13


# --------------------------------------------------------------------------- computed
def test_computed_facet_dict():
    df = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)})
    plot = (
        ggplot(df, aes("x", "y"))
        .add(geom_point())
        .add(facet_wrap({"big": lambda d: d["x"] >= 4}, labeller=label_both))
        .build()
    )
    assert set(plot.panels) == {(1, 1), (1, 2)}
    assert len(plot.layers[0].panel(1, 1).marks) == 4
    assert len(plot.layers[0].panel(1, 2).marks) == 6
    assert "big" not in df.columns  # user data untouched


def test_computed_facet_callable_and_layer_data():
    df = pd.DataFrame({"x": np.arange(6.0), "y": np.arange(6.0)})
    other = pd.DataFrame({"x": [0.5, 4.5], "y": [1.0, 1.0]})

    def half(d):
        return np.where(d["x"] < 3, "low", "high")

    facet = facet_wrap(half)
    assert isinstance(facet, FacetWrap) and facet.vars() == ["half"]
    plot = (
        ggplot(df, aes("x", "y"))
        .add(geom_point())
        .add(geom_point(data=other))
        .add(geom_hline(yintercept=2))
        .add(facet)
        .build()
    )
    # levels sorted: "high" then "low"
    assert len(plot.layers[1].panel(1, 1).marks) == 1
    assert len(plot.layers[1].panel(1, 2).marks) == 1
    assert len(plot.layers[2].panel(1, 1).marks) == 1  # hline has no x: repeated


def test_registered_method():
    plot = ggplot(seven(), aes("x", "y")).geom_point().facet_wrap("g", ncol=4).build()
    assert max(c for _, c in plot.panels) == 4
