"""facet_grid (ggm-913.3): layout, strips, switch, free scales / space, margins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from manim import DL, UR, config

from ggmanim import aes, facet_grid, geom_col, geom_hline, geom_point, ggplot, label_both
from ggmanim.facets import FacetGrid
from ggmanim.facets.grid import MARGIN_LABEL, parse_formula


def df6():
    """2 row levels x 3 col levels, two points per cell; ranges differ per cell."""
    rows = []
    for i, r in enumerate(["a", "b"]):
        for j, c in enumerate([4, 6, 8]):
            for k in range(2):
                rows.append((float(j * 10 + k), float(i * 100 + k), r, c))
    return pd.DataFrame(rows, columns=["x", "y", "r", "c"])


def box(m):
    return m.get_corner(DL), m.get_corner(UR)


def build(facet, df=None, *extra):
    p = ggplot(df6() if df is None else df, aes("x", "y")).add(geom_point()).add(facet)
    for e in extra:
        p = p.add(e)
    return p.build()


# --------------------------------------------------------------------------- arguments
@pytest.mark.parametrize(
    "args, rows, cols",
    [
        (("r", "c"), ["r"], ["c"]),
        (("r ~ c",), ["r"], ["c"]),
        ((". ~ c",), [], ["c"]),
        (("r + c ~ .",), ["r", "c"], []),
        ((["r", "c"], None), ["r", "c"], []),
        ((None, "a + b"), [], ["a", "b"]),
        ((".", "c"), [], ["c"]),
    ],
)
def test_rows_cols_forms(args, rows, cols):
    f = facet_grid(*args)
    assert isinstance(f, FacetGrid)
    assert (f.rows, f.cols) == (rows, cols)
    assert f.vars() == rows + cols


def test_bad_arguments():
    with pytest.raises(ValueError, match="both"):
        facet_grid("r", "r")
    with pytest.raises(ValueError, match="formula"):
        facet_grid("r ~ c", cols="d")
    with pytest.raises(ValueError, match="formula"):
        parse_formula("a ~ b ~ c")
    with pytest.raises(ValueError, match="space"):
        facet_grid("r", space="loose")
    with pytest.raises(ValueError, match="switch"):
        facet_grid("r", switch="z")
    with pytest.raises(ValueError, match="margins"):
        facet_grid("r", margins=["nope"])
    with pytest.raises(TypeError):
        facet_grid(rows=3)
    assert "FacetGrid" in repr(facet_grid("r ~ c"))


# --------------------------------------------------------------------------- layout
def test_layout_2x3():
    layout = facet_grid("r ~ c", scales="free").compute_layout([df6()])
    assert list(layout.columns) == ["PANEL", "ROW", "COL", "SCALE_X", "SCALE_Y", "r", "c"]
    assert list(layout["PANEL"]) == [1, 2, 3, 4, 5, 6]
    assert list(layout["ROW"]) == [1, 1, 1, 2, 2, 2]
    assert list(layout["COL"]) == [1, 2, 3, 1, 2, 3]
    assert list(layout["r"]) == ["a", "a", "a", "b", "b", "b"]
    assert list(layout["c"]) == [4, 6, 8, 4, 6, 8]
    # free scales are shared along the grid: x per column, y per row
    assert list(layout["SCALE_X"]) == [1, 2, 3, 1, 2, 3]
    assert list(layout["SCALE_Y"]) == [1, 1, 1, 2, 2, 2]
    fixed = facet_grid("r ~ c").compute_layout([df6()])
    assert set(fixed["SCALE_X"]) == set(fixed["SCALE_Y"]) == {1}


def test_layout_full_grid_even_for_empty_cells():
    df = df6().iloc[:-2]  # drop the (b, 8) cell
    layout = facet_grid("r ~ c").compute_layout([df])
    assert len(layout) == 6


def test_layout_multiple_vars_and_no_vars():
    df = df6().assign(s=lambda d: np.where(d["x"] % 2 == 0, "even", "odd"))
    layout = facet_grid(cols=["c", "s"]).compute_layout([df])
    assert list(layout["ROW"]) == [1] * 6
    assert list(zip(layout["c"], layout["s"])) == [
        (4, "even"),
        (4, "odd"),
        (6, "even"),
        (6, "odd"),
        (8, "even"),
        (8, "odd"),
    ]
    one = facet_grid().compute_layout([df])
    assert len(one) == 1 and list(one["PANEL"]) == [1]


def test_drop_false_keeps_unused_levels():
    df = df6()
    df["c"] = pd.Categorical(df["c"], categories=[4, 6, 8, 10])
    assert len(facet_grid(cols="c").compute_layout([df])) == 3
    layout = facet_grid(cols="c", drop=False).compute_layout([df])
    assert list(layout["c"]) == [4, 6, 8, 10]


def test_categorical_order_respected():
    df = df6()
    df["r"] = pd.Categorical(df["r"], categories=["b", "a"])
    layout = facet_grid("r ~ c").compute_layout([df])
    assert list(layout["r"])[:3] == ["b", "b", "b"]


def test_unknown_var_is_error():
    with pytest.raises(ValueError, match="nope"):
        facet_grid("nope ~ c").compute_layout([df6()])


# --------------------------------------------------------------------------- build
def test_2x3_build_strips_and_axes():
    plot = build(facet_grid("r ~ c", labeller=label_both))
    assert set(plot.panels) == {(r, c) for r in (1, 2) for c in (1, 2, 3)}
    grid = plot.panel_grid
    texts = [(s.side, s.text.text.replace(" ", "")) for s in plot.strips]
    assert texts == [
        ("top", "c:4"),
        ("top", "c:6"),
        ("top", "c:8"),
        ("right", "r:a"),
        ("right", "r:b"),
    ]
    # col strips only above row 1, row strips only right of the last column
    assert len(grid.panel_strips[(2, 1)]) == 0
    assert [s.side for s in grid.panel_strips[(1, 3)]] == ["top", "right"]
    # right strips are rotated (tall) and touch the plotting area
    for r in (1, 2):
        s = grid.panel_strips[(r, 3)][-1]
        lo, hi = box(plot.panels[(r, 3)].axes)
        assert s.height > s.width
        assert np.isclose(s.get_left()[0], hi[0], atol=1e-6)
        assert np.isclose(s.height, hi[1] - lo[1], atol=1e-6)
    # fixed scales: x axis on the bottom row, y axis on the first column
    for (r, c), p in plot.panels.items():
        assert (len(p.x_tick_labels) > 0) == (r == 2)
        assert (len(p.y_tick_labels) > 0) == (c == 1)
    # data split by cell
    layer = plot.layers[0]
    assert all(len(layer.panel(*cell).marks) == 2 for cell in plot.panels)
    assert plot.width <= config.frame_width + 1e-6


def test_switch_moves_strips():
    plot = build(facet_grid("r ~ c", switch="y"))
    grid = plot.panel_grid
    sides = {s.side for s in plot.strips}
    assert sides == {"top", "left"}
    left = grid.panel_strips[(1, 1)]
    assert [s.side for s in left] == ["top", "left"]
    assert len(grid.panel_strips[(1, 3)]) == 1  # top only
    # inside placement: strip touches the panel, y tick labels moved past it
    p = plot.panels[(1, 1)]
    strip = left[1]
    assert np.isclose(strip.get_right()[0], p.axes.get_left()[0], atol=1e-6)
    assert p.y_tick_labels.get_right()[0] < strip.get_left()[0]

    both = build(facet_grid("r ~ c", switch="both"))
    assert {s.side for s in both.strips} == {"bottom", "left"}
    bottom = both.panel_grid.panel_strips[(2, 2)]
    assert [s.side for s in bottom] == ["bottom"]
    assert len(both.panel_grid.panel_strips[(1, 2)]) == 0

    x_only = build(facet_grid("r ~ c", switch="x"))
    assert {s.side for s in x_only.strips} == {"bottom", "right"}


def test_free_scales_per_row_and_column():
    plot = build(facet_grid("r ~ c", scales="free"))
    sc = plot.panel_scales
    assert sc[(1, 1)]["x"] is sc[(2, 1)]["x"]
    assert sc[(1, 1)]["x"] is not sc[(1, 2)]["x"]
    assert sc[(1, 1)]["y"] is sc[(1, 3)]["y"]
    assert sc[(1, 1)]["y"] is not sc[(2, 1)]["y"]
    assert sc[(1, 2)]["x"].get_limits() == pytest.approx((10, 11))
    assert sc[(2, 1)]["y"].get_limits() == pytest.approx((100, 101))
    # axes once per column / row
    for (r, c), p in plot.panels.items():
        assert (len(p.x_tick_labels) > 0) == (r == 2)
        assert (len(p.y_tick_labels) > 0) == (c == 1)


def _items():
    rows = []
    for kind, names in {"big": list("pqrst"), "small": ["u", "v"]}.items():
        for n in names:
            rows.append((kind, n, 1.0))
    return pd.DataFrame(rows, columns=["kind", "item", "v"])


def test_space_free_widths_proportional_to_levels():
    base = ggplot(_items(), aes("item", "v")).add(geom_col())
    plot = base.add(facet_grid(cols="kind", scales="free_x", space="free_x")).build()
    (l1, h1), (l2, h2) = box(plot.panels[(1, 1)].axes), box(plot.panels[(1, 2)].axes)
    w1, w2 = h1[0] - l1[0], h2[0] - l2[0]
    r1 = plot.panels[(1, 1)].panel_params.x_range
    r2 = plot.panels[(1, 2)].panel_params.x_range
    assert w1 / w2 == pytest.approx((r1[1] - r1[0]) / (r2[1] - r2[0]), rel=1e-6)
    assert w1 > 2 * w2
    # same bar width in both panels
    b1 = plot.layers[0].panel(1, 1).marks[0].width
    b2 = plot.layers[0].panel(1, 2).marks[0].width
    assert b1 == pytest.approx(b2, rel=1e-6)
    # space fixed: equal widths
    fixed = base.add(facet_grid(cols="kind", scales="free_x")).build()
    (l1, h1), (l2, h2) = box(fixed.panels[(1, 1)].axes), box(fixed.panels[(1, 2)].axes)
    assert h1[0] - l1[0] == pytest.approx(h2[0] - l2[0])


def test_space_free_y_heights():
    df = _items().rename(columns={"item": "y_item"})
    plot = (
        ggplot(df, aes("v", "y_item"))
        .add(geom_point())
        .add(facet_grid(rows="kind", scales="free_y", space="free_y"))
        .build()
    )
    (l1, h1), (l2, h2) = box(plot.panels[(1, 1)].axes), box(plot.panels[(2, 1)].axes)
    assert (h1[1] - l1[1]) > 2 * (h2[1] - l2[1])
    f = facet_grid("kind", space="free")
    layout = f.compute_layout([df])

    class P:
        def __init__(self, xr, yr):
            self.x_range, self.y_range = xr, yr

    cw, rw = f.panel_weights(layout, {1: P((0, 2), (0, 5)), 2: P((0, 2), (3, 3))})
    assert cw == [2.0] and rw == [5.0, 1.0]  # zero-length ranges fall back to 1
    assert facet_grid("kind").panel_weights(layout, {}) == (None, None)


def test_margins():
    f = facet_grid("r ~ c", margins=True)
    assert sorted(map(sorted, f.margin_sets())) == [["c"], ["c", "r"], ["r"]]
    plot = build(f)
    layout = plot.panel_grid.layout
    assert len(layout) == 12  # (2 + 1) x (3 + 1)
    assert list(layout["c"])[:4] == [4, 6, 8, MARGIN_LABEL]
    assert list(layout.loc[layout["ROW"] == 3, "r"]) == [MARGIN_LABEL] * 4
    layer = plot.layers[0]
    assert len(layer.panel(1, 1).marks) == 2
    assert len(layer.panel(1, 4).marks) == 6  # row a, all columns
    assert len(layer.panel(3, 1).marks) == 4  # column 4, all rows
    assert len(layer.panel(3, 4).marks) == 12  # everything
    texts = [s.text.text for s in plot.strips]
    assert texts.count(MARGIN_LABEL) == 2


def test_margins_on_one_variable_and_nested():
    assert facet_grid("r ~ c", margins=["c"]).margin_sets() == [{"c"}]
    nested = facet_grid(rows=["a", "b"], margins="b").margin_sets()
    assert nested == [{"b"}]
    outer = facet_grid(rows=["a", "b"], margins=["a"]).margin_sets()
    assert outer == [{"a", "b"}]
    assert facet_grid("r ~ c").margin_sets() == []
    df = df6()
    out = facet_grid("r ~ c", margins=True).add_margins(df)
    assert len(out) == 4 * len(df)
    # layers lacking the facet vars are not duplicated (they repeat anyway)
    no_vars = pd.DataFrame({"x": [1.0]})
    assert facet_grid("r ~ c", margins=True).add_margins(no_vars) is no_vars


def test_layer_without_facet_vars_repeats_in_every_cell():
    plot = (
        ggplot(df6(), aes("x", "y"))
        .add(geom_point())
        .add(geom_hline(yintercept=50))
        .add(facet_grid("r ~ c", margins=True))
        .build()
    )
    hl = plot.layers[1]
    assert all(len(hl.panel(*cell).marks) == 1 for cell in plot.panels)
