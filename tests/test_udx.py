"""Follow-ups to ggm-rnm (ggm-udx): break thinning under ``coord_trans``, the
``SCALE_X``/``SCALE_Y`` swap under ``coord_flip`` and the crowded-discrete-labels
warning."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from manim import DL, UR

from ggmanim import (
    aes,
    coord_cartesian,
    coord_flip,
    coord_trans,
    facet_grid,
    facet_wrap,
    geom_point,
    ggplot,
    theme,
    theme_default,
)
from ggmanim.build import assign_panels, resolve_layers
from ggmanim.coords_trans import CoordTrans
from ggmanim.scales import ScaleContinuousPosition, ScaleDiscretePosition


def _scale(aes_name: str, lo: float, hi: float, **kw) -> ScaleContinuousPosition:
    s = ScaleContinuousPosition(aes_name, **kw)
    s.train([lo, hi])
    return s


def _overlap(labels, axis: str) -> bool:
    i = 0 if axis == "x" else 1
    spans = sorted((m.get_corner(DL)[i], m.get_corner(UR)[i]) for m in labels)
    return any(b[0] < a[1] for a, b in zip(spans, spans[1:]))


def _trans_panel(width: float, trans: str = "sqrt", lo: float = 0.0, hi: float = 9000.0, **kw):
    th = theme_default().replace(panel_width=width, panel_height=3.0)
    return CoordTrans(x=trans).build(_scale("x", lo, hi, **kw), _scale("y", 0.0, 1.0), th)


# --------------------------------------------------------------------------- coord_trans
def test_coord_trans_wide_panel_keeps_default_breaks():
    p = _trans_panel(8.0)
    assert p.panel_params.x_breaks == (0.0, 2000.0, 4000.0, 6000.0, 8000.0)
    assert len(p.x_tick_labels) == 5


def test_coord_trans_narrow_panel_thins_breaks():
    p = _trans_panel(1.5)
    pp = p.panel_params
    assert 2 <= len(pp.x_breaks) < 5
    assert len(p.x_tick_labels) == len(pp.x_breaks) == len(pp.x_labels)
    assert not _overlap(p.x_tick_labels, "x")
    # panel_params stay in input space, consistent with the drawn ticks
    assert all(float(lab) == b for lab, b in zip(pp.x_labels, pp.x_breaks))
    xs = [float(p.transform(b, 0.5)[0, 0]) for b in pp.x_breaks]
    ticks = sorted(float(m.get_center()[0]) for m in p.x_tick_labels)
    np.testing.assert_allclose(sorted(xs), ticks, atol=0.2)
    # minor breaks follow the new majors, in input space
    assert pp.x_minor_breaks and not set(pp.x_minor_breaks) & set(pp.x_breaks)
    assert all(pp.x_range[0] <= b <= pp.x_range[1] for b in pp.x_minor_breaks)


def test_coord_trans_log_breaks_are_clean_input_values():
    # linear pretty() breaks bunch up at the top of a log axis: thinned to two
    p = _trans_panel(8.0, "log10", 1.0, 100000.0)
    pp = p.panel_params
    assert len(pp.x_breaks) == len(p.x_tick_labels) == 2
    assert all(b == float(f"{b:.6g}") for b in pp.x_breaks + pp.x_minor_breaks)


def test_coord_trans_user_breaks_untouched():
    p = _trans_panel(1.0, breaks=[0, 1000, 2000, 3000, 4000, 5000])
    assert p.panel_params.x_breaks == (0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0)


def test_coord_trans_discrete_axis_untouched():
    xs = ScaleDiscretePosition("x")
    xs.train(list("abcdef"))
    th = theme_default().replace(panel_width=6.0, panel_height=3.0)
    p = CoordTrans().build(xs, _scale("y", 1.0, 1000.0), th)
    assert len(p.panel_params.x_breaks) == 6


def test_coord_trans_faceted_plot_thins():
    df = pd.DataFrame(
        {"x": np.tile([0.0, 3000.0, 9000.0], 6), "y": 1.0, "g": np.repeat(list("abcdef"), 3)}
    )
    plot = (
        ggplot(df, aes("x", "y"))
        .add(geom_point(), facet_wrap("g", ncol=6), coord_trans(x="sqrt"))
        .build()
    )
    for panel in plot.panels.values():
        if len(panel.x_tick_labels):
            assert len(panel.panel_params.x_breaks) < 5
            assert not _overlap(panel.x_tick_labels, "x")


# --------------------------------------------------------------------------- coord_flip layout
def _cars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wt": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            "mpg": [34.0, 30.0, 26.0, 21.0, 20.0, 18.0, 16.0, 14.0, 10.0],
            "cyl": [4, 4, 4, 6, 6, 6, 8, 8, 8],
        }
    )


@pytest.mark.parametrize(
    "facet", [facet_grid(cols="cyl", scales="free_x"), facet_wrap("cyl", scales="free_x")]
)
def test_coord_flip_swaps_scale_ids(facet):
    # ggplot2 4.0.3, ggplot_build(...)$layout$layout: SCALE_X 1 1 1, SCALE_Y 1 2 3
    plot = ggplot(_cars(), aes("wt", "mpg")).add(geom_point(), facet, coord_flip())
    layout = assign_panels(resolve_layers(plot), plot)
    assert list(layout["SCALE_X"]) == [1, 1, 1]
    assert list(layout["SCALE_Y"]) == [1, 2, 3]
    plain = ggplot(_cars(), aes("wt", "mpg")).add(geom_point(), facet)
    layout = assign_panels(resolve_layers(plain), plain)
    assert list(layout["SCALE_X"]) == [1, 2, 3]
    assert list(layout["SCALE_Y"]) == [1, 1, 1]


def test_coord_flip_free_x_frees_horizontal_axis():
    built = (
        ggplot(_cars(), aes("wt", "mpg"))
        .add(geom_point(), facet_grid(cols="cyl", scales="free_x"), coord_flip())
        .build()
    )
    pps = [built.panels[(1, c)].panel_params for c in (1, 2, 3)]
    # x aesthetic (vertical) shared, y aesthetic (horizontal, mpg) per column
    assert pps[0].x_range == pps[1].x_range == pps[2].x_range
    assert len({pp.y_range for pp in pps}) == 3
    # cyl 4 panel: mpg 26..34 expanded (ggplot2: 20.775 34.525 on mtcars; here 25.6 34.4)
    assert pps[0].y_range == pytest.approx((25.6, 34.4))
    # every column draws its own horizontal axis, only the first the vertical one
    assert all(len(built.panels[(1, c)].x_tick_labels) for c in (1, 2, 3))
    assert len(built.panels[(1, 2)].y_tick_labels) == 0


def test_coord_cartesian_layout_unchanged():
    layout = pd.DataFrame({"PANEL": [1, 2], "SCALE_X": [1, 2], "SCALE_Y": [1, 1]})
    assert coord_cartesian().setup_layout(layout) is layout
    out = coord_flip().setup_layout(layout)
    assert list(out["SCALE_X"]) == [1, 1] and list(out["SCALE_Y"]) == [1, 2]
    assert list(layout["SCALE_X"]) == [1, 2]  # not mutated


# --------------------------------------------------------------------------- crowded discrete
def _discrete_plot(n: int):
    df = pd.DataFrame({"x": [f"category {i}" for i in range(n)], "y": np.arange(n, dtype=float)})
    return ggplot(df, aes("x", "y")).add(geom_point())


def test_crowded_discrete_axis_warns():
    with pytest.warns(UserWarning, match=r"12 discrete x-axis labels overlap"):
        _discrete_plot(12).build()


def test_discrete_axis_with_room_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _discrete_plot(3).build()
        _discrete_plot(12).add(theme(axis_text_x_angle=45)).build()
        _discrete_plot(12).add(theme(axis_text_x=False)).build()


def test_flipped_crowded_discrete_names_screen_axis():
    df = pd.DataFrame({"x": [f"c{i}" for i in range(40)], "y": np.arange(40.0)})
    with pytest.warns(UserWarning, match=r"discrete y-axis labels overlap"):
        ggplot(df, aes("x", "y")).add(geom_point(), coord_flip()).build()


def test_measuring_pass_does_not_warn():
    # space="free_y": rows are short only in the nominal measuring pass
    df = pd.DataFrame(
        {
            "item": [f"item {i}" for i in range(8)],
            "v": np.arange(8.0),
            "kind": ["a"] * 6 + ["b"] * 2,
        }
    )
    plot = ggplot(df, aes("v", "item")).add(
        geom_point(), facet_grid(rows="kind", scales="free_y", space="free_y")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot.build()
