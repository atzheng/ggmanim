"""Facet base contract, panel layout and the multi-panel build pipeline (ggm-913.1).

facet_wrap / facet_grid are separate beads, so multi-panel builds use small
test-only Facet subclasses that just return a layout table.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from manim import DL, UR, VGroup, config

from ggmanim import (
    aes,
    coord_flip,
    facet_null,
    geom_col,
    geom_function,
    geom_hline,
    geom_point,
    ggplot,
    label_both,
    label_value,
    labeller,
    labs,
    theme,
)
from ggmanim.build import BuildError
from ggmanim.facets import Facet, FacetNull, PanelGrid, PanelLayers, _layout


class FacetBy(Facet):
    """Minimal test facet: one panel per level of ``var``, ``ncol`` per row."""

    def __init__(self, var="g", ncol=None, scales="fixed", labeller=None, side="top"):
        super().__init__(scales=scales, labeller=labeller)
        self.var, self.ncol, self.side = var, ncol, side

    def vars(self):
        return [self.var]

    def compute_layout(self, data):
        combos = self.combinations(data)
        n = len(combos)
        ncol = self.ncol or math.ceil(math.sqrt(n))
        layout = combos.assign(
            PANEL=range(1, n + 1),
            ROW=[i // ncol + 1 for i in range(n)],
            COL=[i % ncol + 1 for i in range(n)],
        )
        return self.scale_ids(layout)

    def strips(self, layout):
        return {
            int(p): [(self.side, self.strip_labels(layout, int(p), [self.var]))]
            for p in layout["PANEL"]
        }


class TwoPanels(FacetNull):
    """A FacetNull subclass forced to two side-by-side panels (no facet var)."""

    is_null = False

    def compute_layout(self, data):
        return pd.DataFrame(
            {"PANEL": [1, 2], "ROW": [1, 1], "COL": [1, 2], "SCALE_X": [1, 1], "SCALE_Y": [1, 1]}
        )

    def map_data(self, data, layout):
        return Facet.map_data(self, data, layout)


def df2():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "g": ["a", "a", "a", "b", "b", "b"],
        }
    )


def box(m):
    return m.get_corner(DL), m.get_corner(UR)


# --------------------------------------------------------------------------- null
def test_facet_null_is_default_and_identical():
    df = df2()
    base = ggplot(df, aes("x", "y")).add(geom_point())
    a, b = base.build(), base.add(facet_null()).build()
    assert len(a.submobjects) == len(b.submobjects)
    assert np.allclose(a.get_all_points(), b.get_all_points())
    assert b.panels == {(1, 1): b.panel}
    assert len(b.strips) == 0 and b.panel_grid is None
    assert b.axes is b.panel.axes
    assert isinstance(FacetNull().compute_layout([]), pd.DataFrame)


def test_facet_null_repr_and_add():
    p = ggplot(df2(), aes("x", "y")).facet_null()
    assert isinstance(p.facet, FacetNull)
    assert repr(p.facet) == "facet_null()"


# --------------------------------------------------------------------------- two panels
def test_two_forced_panels_share_scales():
    plot = ggplot(df2(), aes("x", "y")).add(geom_point()).add(TwoPanels()).build()
    assert set(plot.panels) == {(1, 1), (1, 2)}
    p1, p2 = plot.panels[(1, 1)], plot.panels[(1, 2)]
    assert plot.panel is p1 and plot.axes is p1.axes
    # one shared scale object, identical panel params
    assert plot.panel_scales[(1, 1)]["x"] is plot.panel_scales[(1, 2)]["x"]
    assert p1.panel_params == p2.panel_params
    # same plotting-area size, p2 to the right of p1, aligned vertically
    (l1, h1), (l2, h2) = box(p1.axes), box(p2.axes)
    assert np.allclose(h1 - l1, h2 - l2)
    assert l2[0] > h1[0] and np.isclose(l1[1], l2[1])
    # inner y axis suppressed, x axis on both (bottom row)
    assert len(p1.y_tick_labels) > 0 and len(p2.y_tick_labels) == 0
    assert len(p1.x_tick_labels) == len(p2.x_tick_labels) > 0
    # no facet vars: every row is repeated in both panels
    layer = plot.layers[0]
    assert isinstance(layer, PanelLayers)
    assert len(layer.panel(1, 1).marks) == len(layer.panel(1, 2).marks) == 6
    assert len(layer.marks) == 12
    assert sorted(plot.layer_data[0]["PANEL"].unique()) == [1, 2]


def test_grid_structure_and_titles():
    plot = (
        ggplot(df2(), aes("x", "y", colour="g"))
        .add(geom_point())
        .add(FacetBy())
        .add(labs(x="X title", y="Y title", title="T"))
        .build()
    )
    grid = plot.panel_grid
    assert isinstance(grid, PanelGrid)
    assert grid in plot.submobjects
    assert len(plot.strips) == 2
    assert [s.text.text for s in plot.strips] == ["a", "b"]
    # axis titles drawn once, below / left of the grid, not on the panels
    assert all(p.x_title is None and p.y_title is None for p in plot.panels.values())
    assert grid.x_title is not None and grid.y_title is not None
    assert grid.x_title.get_top()[1] < min(p.get_bottom()[1] for p in plot.panels.values())
    assert grid.y_title.get_right()[0] < min(p.get_left()[0] for p in plot.panels.values())
    # strips sit right above each panel, as wide as the plotting area
    for (_r, _c), panel in plot.panels.items():
        strip = grid.panel_strips[(_r, _c)][0]
        lo, hi = box(panel.axes)
        assert np.isclose(strip.get_bottom()[1], hi[1], atol=1e-6)
        assert np.isclose(strip.width, hi[0] - lo[0], atol=1e-6)
    # the legend sits right of the whole grid
    assert plot.legend.get_left()[0] > grid.get_right()[0] - 1e-6
    # whole plot fits the frame
    assert plot.width <= config.frame_width + 1e-6
    assert plot.height <= config.frame_height + 1e-6
    assert plot["panel"] is plot.panel


def test_facet_rows_split_by_variable():
    plot = ggplot(df2(), aes("x", "y")).add(geom_point()).add(FacetBy()).build()
    layer = plot.layers[0]
    assert len(layer.panel(1, 1).marks) == 3 and len(layer.panel(1, 2).marks) == 3
    assert layer.keys == [0, 1, 2, 3, 4, 5]
    assert "g" not in plot.layer_data[0].columns


def test_free_scales_train_per_panel():
    plot = ggplot(df2(), aes("x", "y")).add(geom_point()).add(FacetBy(scales="free_x")).build()
    s1, s2 = plot.panel_scales[(1, 1)], plot.panel_scales[(1, 2)]
    assert s1["x"] is not s2["x"] and s1["y"] is s2["y"]
    assert s1["x"].get_limits() == pytest.approx((1, 3))
    assert s2["x"].get_limits() == pytest.approx((10, 30))
    assert s1["y"].get_limits() == pytest.approx((1, 6))
    # free x: x axis on every panel; fixed y: y axis on the first column only
    p2 = plot.panels[(1, 2)]
    assert len(p2.x_tick_labels) and not len(p2.y_tick_labels)

    free = ggplot(df2(), aes("x", "y")).add(geom_point()).add(FacetBy(scales="free")).build()
    f1, f2 = free.panel_scales[(1, 1)], free.panel_scales[(1, 2)]
    assert f1["y"].get_limits() == pytest.approx((1, 3))
    assert f2["y"].get_limits() == pytest.approx((4, 6))
    assert len(free.panels[(1, 2)].y_tick_labels) > 0


def test_free_discrete_scales_per_panel():
    df = pd.DataFrame({"k": ["p", "q", "r", "s"], "v": [1, 2, 3, 4], "g": ["a", "a", "b", "b"]})
    plot = ggplot(df, aes("k", "v")).add(geom_col()).add(FacetBy(scales="free_x")).build()
    assert list(plot.panel_scales[(1, 1)]["x"].get_limits()) == ["p", "q"]
    assert list(plot.panel_scales[(1, 2)]["x"].get_limits()) == ["r", "s"]
    fixed = ggplot(df, aes("k", "v")).add(geom_col()).add(FacetBy()).build()
    assert list(fixed.panel_scales[(1, 2)]["x"].get_limits()) == ["p", "q", "r", "s"]


def test_layers_without_facet_vars_repeat():
    plot = (
        ggplot(df2(), aes("x", "y"))
        .add(geom_point())
        .add(geom_hline(yintercept=[2, 4]))
        .add(geom_function(fun=lambda x: x / 10))
        .add(FacetBy())
        .build()
    )
    hl = plot.layers[1]
    assert len(hl.panel(1, 1).marks) == len(hl.panel(1, 2).marks) == 2
    fn = plot.layers[2]
    assert len(fn.panel(1, 1).marks) == len(fn.panel(1, 2).marks) == 1
    assert sorted(plot.layer_data[2]["PANEL"].unique()) == [1, 2]


def test_layer_data_with_partial_facet_vars():
    other = pd.DataFrame({"x": [5.0], "y": [5.0], "g": ["b"]})
    plot = (
        ggplot(df2(), aes("x", "y"))
        .add(geom_point())
        .add(geom_point(data=other))
        .add(FacetBy())
        .build()
    )
    assert len(plot.layers[1].panel(1, 1).marks) == 0
    assert len(plot.layers[1].panel(1, 2).marks) == 1


def test_grid_with_gap_draws_axes_above_hole():
    df = pd.DataFrame({"x": [1.0, 2, 3], "y": [1.0, 2, 3], "g": ["a", "b", "c"]})
    plot = ggplot(df, aes("x", "y")).add(geom_point()).add(FacetBy(ncol=2)).build()
    assert set(plot.panels) == {(1, 1), (1, 2), (2, 1)}
    assert len(plot.panels[(1, 2)].x_tick_labels) > 0  # nothing below it
    assert len(plot.panels[(1, 1)].x_tick_labels) == 0
    assert len(plot.panels[(2, 1)].y_tick_labels) > 0
    # rows / columns aligned
    (a, _), (b, _) = box(plot.panels[(1, 1)].axes), box(plot.panels[(2, 1)].axes)
    assert np.isclose(a[0], b[0])


def test_coord_flip_facets():
    plot = (
        ggplot(df2(), aes("x", "y"))
        .add(geom_point())
        .add(coord_flip())
        .add(FacetBy(scales="free_x"))
        .build()
    )
    p2 = plot.panels[(1, 2)]
    assert p2.flipped
    # x is vertical under flip: free x -> vertical axis on every panel
    assert len(p2.y_tick_labels) > 0
    assert len(p2.x_tick_labels) > 0  # bottom row


def test_strip_sides_and_placement():
    base = ggplot(df2(), aes("x", "y")).add(geom_point())
    inside = base.add(FacetBy(side="bottom")).build()
    p = inside.panels[(1, 1)]
    strip = inside.panel_grid.panel_strips[(1, 1)][0]
    # inside: strip touches the panel, tick labels move below it
    assert np.isclose(strip.get_top()[1], p.axes.get_bottom()[1], atol=1e-6)
    assert p.x_tick_labels.get_top()[1] < strip.get_bottom()[1]
    outside = base.add(FacetBy(side="bottom")).add(theme(strip_placement="outside")).build()
    p = outside.panels[(1, 1)]
    strip = outside.panel_grid.panel_strips[(1, 1)][0]
    assert strip.get_top()[1] < p.x_tick_labels.get_bottom()[1]
    for side in ("left", "right"):
        built = base.add(FacetBy(side=side)).build()
        s = built.panel_grid.panel_strips[(1, 1)][0]
        assert s.height > s.width
    with pytest.raises(ValueError, match="strip_placement"):
        base.add(FacetBy()).add(theme(strip_placement="nope")).build()


def test_theme_facet_fields():
    t = theme(panel_spacing=1.0, strip_text_size=12, strip_background=False)
    plot = ggplot(df2(), aes("x", "y")).add(geom_point()).add(FacetBy()).add(t).build()
    (_, h1), (l2, _) = box(plot.panels[(1, 1)].axes), box(plot.panels[(1, 2)].axes)
    assert l2[0] - h1[0] >= 1.0 - 1e-6
    assert plot.strips[0].background is None


def test_inside_legend_uses_whole_grid():
    plot = (
        ggplot(df2(), aes("x", "y", colour="g"))
        .add(geom_point())
        .add(FacetBy())
        .add(theme(legend_position=(1, 1)))
        .build()
    )
    right = plot.panels[(1, 2)].axes.get_right()[0]
    assert np.isclose(plot.legend.get_right()[0], right, atol=1e-6)


# --------------------------------------------------------------------------- errors
def test_unknown_facet_var_is_build_error():
    with pytest.raises(BuildError, match="nope"):
        ggplot(df2(), aes("x", "y")).add(geom_point()).add(FacetBy("nope")).build()


def test_bad_layout_is_build_error():
    class Bad(Facet):
        def compute_layout(self, data):
            return pd.DataFrame({"PANEL": [1]})

    with pytest.raises(BuildError, match="ROW"):
        ggplot(df2(), aes("x", "y")).add(geom_point()).add(Bad()).build()


def test_base_compute_layout_abstract_and_scales_arg():
    with pytest.raises(NotImplementedError):
        Facet().compute_layout([])
    with pytest.raises(ValueError):
        Facet(scales="loose")
    assert Facet(scales="free_y").free == {"x": False, "y": True}


# --------------------------------------------------------------------------- units
def test_map_data_matches_and_repeats():
    layout = pd.DataFrame(
        {
            "PANEL": [1, 2, 3, 4],
            "ROW": [1, 1, 2, 2],
            "COL": [1, 2, 1, 2],
            "SCALE_X": 1,
            "SCALE_Y": 1,
            "a": ["u", "u", "v", "v"],
            "b": [1, 2, 1, 2],
        }
    )

    class F(Facet):
        def vars(self):
            return ["a", "b"]

    f = F()
    data = pd.DataFrame({"x": [0, 1, 2], "a": ["v", "u", "v"], "b": [2, 1, 1]})
    out = f.map_data(data, layout)
    assert list(out["PANEL"]) == [1, 3, 4]
    assert list(out["x"]) == [1, 2, 0]
    only_a = f.map_data(pd.DataFrame({"x": [7], "a": ["u"]}), layout)
    assert list(only_a["PANEL"]) == [1, 2]
    none = f.map_data(pd.DataFrame({"x": [7, 8]}), layout)
    assert list(none["PANEL"]) == [1, 1, 2, 2, 3, 3, 4, 4]
    assert f.map_data(pd.DataFrame(), layout).empty


def test_combinations_order_and_nan():
    f = FacetBy("g")
    data = [pd.DataFrame({"g": ["b", None, "a", "b"]})]
    assert list(f.combinations(data)["g"].iloc[:2]) == ["a", "b"]
    assert pd.isna(f.combinations(data)["g"].iloc[2])
    cat = pd.Categorical(["lo", "hi"], categories=["lo", "hi"])
    assert list(f.combinations([pd.DataFrame({"g": cat})])["g"]) == ["lo", "hi"]


def test_init_scales_shares_or_clones():
    from ggmanim.scales.continuous import ScaleContinuousPosition

    x, y = ScaleContinuousPosition(["x"]), ScaleContinuousPosition(["y"])
    layout = pd.DataFrame(
        {"PANEL": [1, 2, 3], "ROW": 1, "COL": [1, 2, 3], "SCALE_X": [1, 2, 2], "SCALE_Y": 1}
    )
    s = Facet().init_scales(layout, x, y)
    assert s[1]["y"] is y and s[3]["y"] is y
    assert s[2]["x"] is s[3]["x"] and s[1]["x"] is not s[2]["x"]


def test_axis_visibility_rules():
    layout = pd.DataFrame(
        {"PANEL": [1, 2, 3], "ROW": [1, 1, 2], "COL": [1, 2, 1], "SCALE_X": 1, "SCALE_Y": 1}
    )
    vis = _layout.axis_visibility(layout)
    assert vis == {1: (False, True), 2: (True, False), 3: (True, True)}
    free = layout.assign(SCALE_X=[1, 2, 3], SCALE_Y=[1, 2, 3])
    assert all(v == (True, True) for v in _layout.axis_visibility(free).values())
    top = _layout.axis_visibility(layout, h_side="top", v_side="right")
    assert top == {1: (True, False), 2: (True, True), 3: (False, True)}


def test_track_sizes_fill_total():
    layout = pd.DataFrame(
        {"PANEL": [1, 2], "ROW": [1, 1], "COL": [1, 2], "SCALE_X": 1, "SCALE_Y": 1}
    )
    ext = {1: (0.0, 0.0, 0.5, 0.7), 2: (0.0, 0.0, 0.5, 0.2)}
    w, h = _layout.track_sizes(layout, ext, (10.0, 5.0), (0.3, 0.3))
    # outer left (0.7) and bottom are not counted; inner left of col 2 is
    assert sum(w) == pytest.approx(10.0 - 0.2 - 0.3)
    assert h == [pytest.approx(5.0)]
    w2, _ = _layout.track_sizes(layout, ext, (10.0, 5.0), (0.3, 0.3), col_weights=[1, 3])
    assert w2[1] == pytest.approx(3 * w2[0])


def test_labellers():
    assert label_value({"a": 1.0, "b": "x"}) == ["1", "x"]
    assert label_value({"a": 1, "b": "x"}, multi_line=False) == ["1, x"]
    assert label_both({"cyl": 4}) == ["cyl: 4"]
    lab = labeller(am={0: "auto", 1: "manual"}, cyl=label_both)
    assert lab({"cyl": 6, "am": 0}) == ["cyl: 6", "auto"]
    assert labeller(g=str.upper)({"g": "a"}) == ["A"]
    assert labeller(multi_line=False)({"a": 1, "b": 2}) == ["1, 2"]
    plot = ggplot(df2(), aes("x", "y")).add(geom_point()).add(FacetBy(labeller=label_both)).build()
    assert [s.text.original_text for s in plot.strips] == ["g: a", "g: b"]


def test_panel_grid_accessors():
    plot = ggplot(df2(), aes("x", "y")).add(geom_point()).add(FacetBy()).build()
    grid = plot.panel_grid
    assert grid.panel(1, 2) is plot.panels[(1, 2)]
    assert list(grid.panel_ids) == [1, 2]
    assert isinstance(grid.axes, VGroup) and len(grid.axes) == 2
    assert grid.panel_params == plot.panel_params
