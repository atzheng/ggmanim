"""Panel-size-aware break density and ``space="free"`` under ``coord_flip`` (ggm-rnm)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from manim import DL, UR

from ggmanim import (
    aes,
    coord_flip,
    facet_grid,
    geom_col,
    geom_point,
    ggplot,
    scale_y_continuous,
    theme_default,
)
from ggmanim.coords import CoordCartesian, PanelParams, fit_breaks
from ggmanim.scales import ScaleContinuousPosition, ScaleDiscretePosition


def _y_scale(lo=22.0, hi=36.0, **kw) -> ScaleContinuousPosition:
    s = ScaleContinuousPosition("y", **kw)
    s.train([lo, hi])
    return s


def _x_scale(lo=0.0, hi=1.0) -> ScaleContinuousPosition:
    s = ScaleContinuousPosition("x")
    s.train([lo, hi])
    return s


def _overlap(labels, axis: str) -> bool:
    i = 0 if axis == "x" else 1
    spans = sorted((m.get_corner(DL)[i], m.get_corner(UR)[i]) for m in labels)
    return any(b[0] < a[1] for a, b in zip(spans, spans[1:]))


def _panel(x_scale, y_scale, w, h):
    theme = theme_default().replace(panel_width=w, panel_height=h)
    return CoordCartesian().build(x_scale, y_scale, theme)


# --------------------------------------------------------------------------- scales
def test_with_n_breaks_is_a_copy():
    s = _y_scale()
    thin = s.with_n_breaks(3)
    assert thin is not s and s.n_breaks == 5 and thin.n_breaks == 3
    assert len(thin.get_breaks()) < len(s.get_breaks())
    assert s.has_default_breaks()
    assert not _y_scale(breaks=[25, 30]).has_default_breaks()
    assert not _y_scale(breaks=lambda r: [25, 30]).has_default_breaks()


# --------------------------------------------------------------------------- coarsening
def test_large_panel_keeps_default_breaks():
    xs, ys = _x_scale(), _y_scale()
    panel = _panel(xs, ys, 6.0, 4.0)
    pp = CoordCartesian().setup_panel_params(xs, ys)
    assert panel.panel_params == pp


def test_small_panel_coarsens_default_breaks():
    xs, ys = _x_scale(), _y_scale()
    default = CoordCartesian().setup_panel_params(xs, ys)
    assert len(default.y_breaks) >= 7  # 22, 24, ..., 36
    panel = _panel(xs, ys, 1.0, 0.8)
    pp = panel.panel_params
    assert 2 <= len(pp.y_breaks) < len(default.y_breaks)
    assert set(pp.y_breaks) <= set(default.y_breaks) | set(np.arange(20.0, 41.0, 5.0))
    assert len(pp.y_labels) == len(pp.y_breaks) == len(panel.y_tick_labels)
    assert not _overlap(panel.y_tick_labels, "y")
    # minor breaks follow the new majors (none of them is a major break)
    assert not set(pp.y_minor_breaks) & set(pp.y_breaks)
    assert len(pp.y_minor_breaks) >= len(pp.y_breaks) - 1


def test_narrow_x_axis_thins_when_pretty_cannot():
    # 1000..9000: pretty() steps 2000 (four breaks) or 5000 (one break inside)
    xs = _x_scale(1000.0, 9000.0)
    panel = _panel(xs, _y_scale(), 2.2, 3.0)
    assert panel.panel_params.x_breaks == (4000.0, 8000.0)  # every 2nd, the round ones
    assert not _overlap(panel.x_tick_labels, "x")
    assert 2000.0 in panel.panel_params.x_minor_breaks  # dropped majors become minor
    # far too narrow for any two labels: still two breaks (never fewer)
    assert len(_panel(xs, _y_scale(), 0.3, 3.0).panel_params.x_breaks) == 2


def test_user_breaks_never_change():
    ys = _y_scale(breaks=list(range(22, 37, 2)))
    panel = _panel(_x_scale(), ys, 1.0, 0.8)
    assert panel.panel_params.y_breaks == tuple(float(b) for b in range(22, 37, 2))
    fn = _y_scale(breaks=lambda r: np.arange(22, 37, 1))
    assert len(_panel(_x_scale(), fn, 1.0, 0.8).panel_params.y_breaks) == 15


def test_discrete_labels_are_never_dropped():
    xs = ScaleDiscretePosition("x")
    xs.train(["Fair", "Good", "Very Good", "Premium", "Ideal"])
    panel = _panel(xs, _y_scale(), 0.8, 2.0)
    assert len(panel.panel_params.x_breaks) == 5
    assert len(panel.x_tick_labels) == 5


def test_fit_breaks_is_pure():
    xs, ys = _x_scale(), _y_scale()
    pp = CoordCartesian().setup_panel_params(xs, ys)
    out = fit_breaks(pp, xs, ys, 1.0, 0.8, theme_default())
    assert out != pp and ys.n_breaks == 5
    assert fit_breaks(pp, xs, ys, 8.0, 6.0, theme_default()) is pp
    assert fit_breaks(pp, None, None, 0.1, 0.1, theme_default()) is pp


def test_facet_column_panels_share_breaks():
    """Panels sharing a scale get the same breaks whether or not their axis is drawn."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"x": rng.uniform(0, 1, 60), "y": rng.uniform(22, 36, 60), "g": np.repeat(list("abc"), 20)}
    )
    plot = ggplot(df, aes("x", "y")).add(geom_point()).add(facet_grid(rows="g")).build()
    ys = {p.panel_params.y_breaks for p in plot.panels.values()}
    assert len(ys) == 1


# --------------------------------------------------------------------------- coord_flip
def test_panel_params_flipped_flag():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0]})
    p = ggplot(df, aes("x", "y")).add(geom_col())
    assert not p.build().panel_params.flipped
    assert p.add(coord_flip()).build().panel_params.flipped


def test_panel_weights_follow_drawn_axes():
    f = facet_grid("kind", space="free")
    layout = f.compute_layout([pd.DataFrame({"kind": ["a", "b"]})])
    common = dict(x_breaks=(), y_breaks=(), x_labels=(), y_labels=())
    plain = {
        1: PanelParams(x_range=(0, 2), y_range=(0, 5), **common),
        2: PanelParams(x_range=(0, 2), y_range=(0, 1), **common),
    }
    flipped = {
        1: PanelParams(x_range=(0, 5), y_range=(0, 2), flipped=True, **common),
        2: PanelParams(x_range=(0, 1), y_range=(0, 2), flipped=True, **common),
    }
    assert f.panel_weights(layout, plain) == ([2.0], [5.0, 1.0])
    assert f.panel_weights(layout, flipped) == ([2.0], [5.0, 1.0])


def test_space_free_under_coord_flip_uses_horizontal_range():
    rows = [("big", n, 1.0) for n in "pqrst"] + [("small", n, 3.0) for n in "uv"]
    df = pd.DataFrame(rows, columns=["kind", "item", "v"])
    plot = (
        ggplot(df, aes("item", "v"))
        .add(geom_col())
        .add(facet_grid(cols="kind", scales="free", space="free_x"), coord_flip())
        .build()
    )
    a1, a2 = plot.panels[(1, 1)].axes, plot.panels[(1, 2)].axes
    w1 = a1.get_corner(UR)[0] - a1.get_corner(DL)[0]
    w2 = a2.get_corner(UR)[0] - a2.get_corner(DL)[0]
    # columns follow the drawn (horizontal = y aesthetic) range, not the item counts
    r1 = plot.panels[(1, 1)].panel_params.y_range
    r2 = plot.panels[(1, 2)].panel_params.y_range
    assert w1 / w2 == pytest.approx((r1[1] - r1[0]) / (r2[1] - r2[0]), rel=1e-6)
    # the y scale is shared along the row, so equal widths (before ggm-rnm: 5:2)
    assert w1 == pytest.approx(w2)


def test_scale_y_continuous_n_breaks_respected_when_room():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [22.0, 36.0]})
    p = ggplot(df, aes("x", "y")).add(geom_point(), scale_y_continuous(n_breaks=8)).build()
    assert len(p.panel_params.y_breaks) >= 7
