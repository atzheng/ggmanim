"""Phase 2 regression tests: the seams Phase 1 left open, each checked through the
public API (``ggplot(...).build()``), plus a smoke test that every plot in
``examples/gallery.py`` builds.

Phase 1 raised on a categorical x, raised on a numeric colour, passed mapped
size/alpha through unscaled, and ignored mapped shape/linetype when drawing.
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from manim import Dot, ManimColor, Square, Triangle, VGroup, VMobject

from ggmanim import (
    aes,
    coord_cartesian,
    geom_col,
    geom_line,
    geom_point,
    ggplot,
    guide_legend,
    guides,
    label_math,
    scale_x_date,
    scale_x_log10,
    theme,
    theme_bw,
    theme_dark,
    theme_gray,
    theme_void,
    xlim,
)
from ggmanim.guides import Legend
from ggmanim.guides_colourbar import Colourbar

GALLERY = Path(__file__).resolve().parent.parent / "examples" / "gallery.py"


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
            "g": ["a", "a", "b", "b", "c", "c"],
            "z": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["p", "q", "r", "s", "t", "u"],
        }
    )


def _hex(c) -> str:
    return ManimColor(c).to_hex().upper()


# -- discrete x no longer raises ------------------------------------------------------


def test_discrete_x_bars(df):
    m = ggplot(df, aes("cat", "y")).add(geom_col(), name="bars").build()
    assert m.scales["x"].get_limits() == ["p", "q", "r", "s", "t", "u"]
    assert tuple(m.panel_params.x_labels) == ("p", "q", "r", "s", "t", "u")
    assert len(m.layers["bars"]) == 6
    # bars sit at the integer positions 1..n, in level order
    centres = [b.get_center()[0] for b in m.layers["bars"]]
    assert centres == sorted(centres)


def test_discrete_x_points_and_rotated_labels(df):
    m = (
        ggplot(df, aes("g", "y"))
        .add(geom_point(), name="pts")
        .add(theme(axis_text_x_angle=45))
        .build()
    )
    assert m.layer_data["pts"]["x"].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    labels = m.panel.x_tick_labels
    assert len(labels) == 3
    # rotated labels are taller than wide-ish horizontal text of the same string
    flat = ggplot(df, aes("g", "y")).geom_point().build().panel.x_tick_labels
    assert labels[0].height > flat[0].height


def test_discrete_y(df):
    m = ggplot(df, aes("x", "g")).add(geom_point(), name="pts").build()
    assert m.scales["y"].get_limits() == ["a", "b", "c"]


# -- numeric colour no longer raises --------------------------------------------------------


def test_numeric_colour_gradient_and_colourbar(df):
    m = ggplot(df, aes("x", "y", color="z")).add(geom_point(), name="pts").build()
    pts = m.layers["pts"]
    assert _hex(pts[0].get_fill_color()) == "#132B43"  # ggplot2's default low
    assert _hex(pts[-1].get_fill_color()) == "#56B1F7"  # ... and high
    assert len(m.legends) == 1 and isinstance(m.legends[0], Colourbar)


def test_numeric_fill_bars(df):
    m = ggplot(df, aes("cat", "y", fill="z")).add(geom_col(), name="bars").build()
    fills = {_hex(b.get_fill_color()) for b in m.layers["bars"]}
    assert len(fills) == 6
    assert isinstance(m.legends[0], Colourbar)


# -- mapped size / alpha are scaled, not passed through ------------------------------------


def test_mapped_size_and_alpha_scaled(df):
    m = ggplot(df, aes("x", "y", size="z", alpha="z")).add(geom_point(), name="pts").build()
    d = m.layer_data["pts"]
    # raw data 0..5 would give radii up to 5 scene units and opacities > 1
    assert d["size"].min() > 0 and d["size"].max() < 0.5
    assert d["alpha"].between(0, 1).all()
    assert d["size"].is_monotonic_increasing and d["alpha"].is_monotonic_increasing
    radii = [p.radius for p in m.layers["pts"]]
    assert radii == pytest.approx(d["size"].tolist())
    opac = [p.get_fill_opacity() for p in m.layers["pts"]]
    assert opac == pytest.approx(d["alpha"].tolist())
    # binned legends for both (merged: same title and breaks)
    assert m.legends and all(isinstance(lg, Legend) for lg in m.legends)


def test_mapped_linewidth_scaled(df):
    d = df.assign(w=df["z"] * 100)  # raw 0..500 would be absurd stroke widths
    m = ggplot(d, aes("x", "y", linewidth="w", group="g")).add(geom_line(), name="l").build()
    lw = m.layer_data["l"]["linewidth"]
    assert lw.min() == pytest.approx(1.0) and lw.max() == pytest.approx(6.0)  # default range


# -- shape / linetype drawn ---------------------------------------------------------------------


def test_mapped_shape_drawn(df):
    m = ggplot(df, aes("x", "y", shape="g")).add(geom_point(), name="pts").build()
    kinds = [type(p) for p in m.layers["pts"]]
    assert kinds == [Dot, Dot, Triangle, Triangle, Square, Square]


def test_mapped_linetype_drawn(df):
    m = ggplot(df, aes("x", "y", linetype="g")).add(geom_line(), name="l").build()
    groups = list(m.layers["l"])
    assert len(groups) == 3
    # "solid" is one path; "dashed"/"dotted" are split into many dash segments
    assert type(groups[0]) is VMobject
    assert all(isinstance(g, VGroup) and len(g) > 2 for g in groups[1:])


def test_shape_and_colour_legends_merge(df):
    m = ggplot(df, aes("x", "y", color="g", shape="g")).geom_point().build()
    assert len(m.legends) == 1


# -- guides, coords, labels, dates, themes: smoke checks of the phase 2 surface ---------------


def test_inside_legend_is_inside_panel(df):
    m = (
        ggplot(df, aes("x", "y", color="g"))
        .geom_point()
        .add(guides(color=guide_legend(ncol=2)), theme(legend_position=(1, 0)))
        .build()
    )
    panel = m.panel.axes
    assert m.legend.get_right()[0] <= panel.get_right()[0] + 1e-6
    assert m.legend.get_bottom()[1] >= panel.get_bottom()[1] - 1e-6


def test_zoom_keeps_rows_limits_censor(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        zoom = ggplot(df, aes("x", "y")).geom_point().add(coord_cartesian(xlim=(1, 3))).build()
        censor = ggplot(df, aes("x", "y")).geom_point().add(xlim(1, 3)).build()
    assert zoom.layer_data[0]["x"].notna().sum() == 6
    assert len(censor.layers[0]) == 3


def test_label_math_on_log_axis():
    d = pd.DataFrame({"x": [1.0, 10.0, 100.0, 1000.0], "y": [1.0, 2.0, 3.0, 4.0]})
    m = ggplot(d, aes("x", "y")).geom_point().add(scale_x_log10(labels=label_math())).build()
    assert tuple(m.panel_params.x_labels) == ("$10^{0}$", "$10^{1}$", "$10^{2}$", "$10^{3}$")
    assert len(m.panel.grid_minor) > 0


def test_monthly_date_axis():
    d = pd.DataFrame(
        {"month": pd.date_range("2022-01-01", periods=12, freq="MS"), "v": np.arange(12.0)}
    )
    m = (
        ggplot(d, aes("month", "v"))
        .geom_line()
        .add(scale_x_date(date_breaks="3 months", date_labels="%b %Y"))
        .build()
    )
    assert "Jan 2022" in m.panel_params.x_labels
    assert "Apr 2022" in m.panel_params.x_labels


@pytest.mark.parametrize("th", [theme_gray, theme_bw, theme_dark, theme_void])
def test_complete_themes_build(df, th):
    m = ggplot(df, aes("x", "y", color="g")).geom_point().add(th()).build()
    assert isinstance(m, VGroup)


# -- the gallery ----------------------------------------------------------------------------------


def _load_gallery():
    spec = importlib.util.spec_from_file_location("ggmanim_gallery", GALLERY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gallery_plots_build():
    gallery = _load_gallery()
    makers = [getattr(gallery, n) for n in dir(gallery) if n.startswith("make_")]
    assert len(makers) >= 10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for make in makers:
            plots = make()
            for p in plots if isinstance(plots, (list, tuple)) else [plots]:
                assert isinstance(p.build(), VGroup)
    assert len(gallery.SCENES) == 10
