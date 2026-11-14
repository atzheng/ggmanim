"""Phase 4 regression tests: facets end to end through the public API
(``ggplot(...).build()``), plus a smoke test that every plot and scene in
``examples/gallery_facets.py`` builds.

The per-feature behaviour is covered by ``test_facet_wrap.py`` /
``test_facet_grid.py`` / ``test_facets_base.py``; these tests pin down what the
gallery shows and the seams between facets and the rest of the plot.
"""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from manim import Scene

from ggmanim import (
    aes,
    facet_grid,
    facet_wrap,
    geom_function,
    geom_hline,
    geom_point,
    ggplot,
    label_both,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_phase4_{name}", EXAMPLES / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gallery():
    return _load("gallery_facets")


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x": np.arange(12.0),
            "y": np.arange(12.0) ** 1.5,
            "g": list("aabbccddeeff"),
            "h": ["p", "q"] * 6,
        }
    )


def _overlap(labels, axis: str) -> bool:
    """Do any two neighbouring tick labels' bounding boxes overlap along ``axis``?"""
    i = 0 if axis == "x" else 1
    spans = sorted((m.get_corner([-1, -1, 0])[i], m.get_corner([1, 1, 0])[i]) for m in labels)
    return any(b[0] < a[1] for a, b in zip(spans, spans[1:]))


# -- gallery smoke test ----------------------------------------------------------------


def test_gallery_plots_build(gallery):
    makers = [f for n, f in vars(gallery).items() if n.startswith("make_") and callable(f)]
    assert len(makers) >= 5
    for make in makers:
        plot = make().build()
        assert len(plot.panels) > 1, make.__name__
        assert len(plot.strips) > 0, make.__name__


def test_gallery_scenes(gallery):
    scenes = {
        n
        for n, c in vars(gallery).items()
        if inspect.isclass(c) and issubclass(c, Scene) and c.__module__ == gallery.__name__
    }
    assert scenes == {
        "WrapFixed",
        "WrapFreeY",
        "GridSwitch",
        "GridSpace",
        "ReferenceLayer",
        "PanelReveal",
    }


# -- what the gallery shows ------------------------------------------------------------


def test_wrap_fixed_shares_scales_and_places_axes(gallery):
    plot = gallery.make_wrap_fixed().build()
    assert list(plot.panels) == [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3)]
    ranges = {p.panel_params.x_range for p in plot.panels.values()}
    ranges |= {p.panel_params.y_range for p in plot.panels.values()}
    assert len(ranges) == 2  # one x range and one y range for every panel
    with_x = {rc for rc, p in plot.panels.items() if len(p.x_tick_labels)}
    with_y = {rc for rc, p in plot.panels.items() if len(p.y_tick_labels)}
    # (1, 4) has no panel below it, so it carries its column's x axis
    assert with_x == {(1, 4), (2, 1), (2, 2), (2, 3)}
    assert with_y == {(1, 1), (2, 1)}


def test_wrap_free_y_every_panel_has_its_own_y(gallery):
    plot = gallery.make_wrap_free_y().build()
    panels = plot.panels.values()
    assert len({p.panel_params.y_range for p in panels}) == 4
    assert len({p.panel_params.x_range for p in panels}) == 1
    assert all(len(p.y_tick_labels) for p in panels)
    labels = [s.text.original_text for s in plot.strips]
    assert labels == ["day: Mon", "day: Thu", "day: Sat", "day: Sun"]


def test_grid_switch_both_moves_strips(gallery):
    plot = gallery.make_grid_switch().build()
    sides = {rc: [s.side for s in st] for rc, st in plot.panel_grid.panel_strips.items()}
    assert sides[(1, 1)] == ["left"]
    assert sides[(2, 1)] == ["bottom", "left"]
    assert sides[(2, 3)] == ["bottom"]
    assert sides[(1, 2)] == []
    # strips are drawn outside the axes: the column strips below the x tick labels
    bottom_strip = plot.panel_grid.panel_strips[(2, 1)][0]
    assert bottom_strip.get_top()[1] <= plot.panels[(2, 1)].x_tick_labels.get_bottom()[1] + 1e-6


def test_grid_space_free_rows_proportional_to_range(gallery):
    plot = gallery.make_grid_space().build()
    per_unit = [
        p.axes.height / (p.panel_params.y_range[1] - p.panel_params.y_range[0])
        for p in plot.panels.values()
    ]
    np.testing.assert_allclose(per_unit, per_unit[0], rtol=1e-6)
    # five, three and two items: tallest row first
    heights = [p.axes.height for p in plot.panels.values()]
    assert heights == sorted(heights, reverse=True)


def test_reference_layer_repeated_in_every_panel(gallery):
    plot = gallery.make_reference_layer().build()
    n = len(gallery.cars())
    assert list(plot.layers["all"].parts) == list(plot.panels)
    assert all(len(part) == n for part in plot.layers["all"].parts.values())
    assert len(plot.layers["all"].marks) == n * len(plot.panels)
    assert all(len(part) == 1 for part in plot.layers["mean"].parts.values())
    # the faceted layer is split, not repeated
    assert len(plot.layers["points"].marks) == n


# -- seams -----------------------------------------------------------------------------


def test_unfaceted_plot_has_one_panel(df):
    plot = ggplot(df, aes("x", "y")).add(geom_point(), name="pts").build()
    assert plot.panels == {(1, 1): plot.panel}
    assert len(plot.strips) == 0
    assert plot.panel_grid is None


def test_panels_layers_strips_line_up(df):
    plot = (
        ggplot(df, aes("x", "y"))
        .add(geom_point(), name="pts")
        .add(facet_grid(rows="h", cols="g", labeller=label_both))
        .build()
    )
    assert list(plot.layers["pts"].parts) == list(plot.panels)
    assert len(plot.panels) == 2 * 6
    for rc, panel in plot.panels.items():
        part = plot.layers["pts"].panel(*rc)
        # every mark sits inside its own panel's axes
        for mark in part:
            c = mark.get_center()
            assert panel.axes.get_left()[0] - 1e-6 <= c[0] <= panel.axes.get_right()[0] + 1e-6


def test_build_twice_gives_equal_layouts(gallery):
    spec = gallery.make_wrap_free_y()
    a, b = spec.build(), spec.build()
    assert list(a.panels) == list(b.panels)
    for rc in a.panels:
        assert a.panels[rc].panel_params.y_range == b.panels[rc].panel_params.y_range


def test_function_and_hline_in_every_wrap_panel(df):
    plot = (
        ggplot(df, aes("x", "y"))
        .add(geom_point(), name="pts")
        .add(geom_function(lambda x: x), name="fn")
        .add(geom_hline(yintercept=5), name="ref")
        .add(facet_wrap("g"))
        .build()
    )
    assert len(plot.panels) == 6
    assert all(len(p) > 0 for p in plot.layers["fn"].parts.values())
    assert all(len(p) == 1 for p in plot.layers["ref"].parts.values())


# -- known issue: ggm-rnm --------------------------------------------------------------


def test_small_free_panels_tick_labels_do_not_overlap():
    wrap = _load("facet_wrap")
    plot = wrap.make_plot(scales="free_y", labeller=label_both).build()
    assert not any(_overlap(p.y_tick_labels, "y") for p in plot.panels.values())
