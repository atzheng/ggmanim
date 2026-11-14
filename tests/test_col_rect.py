"""geom_col/geom_bar drawn through the shared rectangle helper (ggm-5iz)."""

import numpy as np
import pandas as pd
import pytest
from manim import VGroup, VMobject

from ggmanim import aes, geom_bar, geom_col, geom_rect, ggplot
from ggmanim.geoms import GeomCol
from ggmanim.geoms._batch import is_batched
from ggmanim.theme import theme_default

DF = pd.DataFrame({"x": ["a", "b", "c"], "y": [2.0, 5.0, 3.0]})


def _bars(*layers, data=DF):
    return ggplot(data, aes("x", "y")).add(*layers).build().layers[0]


def test_alpha_applies_to_the_fill_only_like_geom_rect():
    bars = _bars(geom_col(alpha=0.4, color="#ff0000", linewidth=2))
    for b in bars:
        assert b.get_fill_opacity() == pytest.approx(0.4)
        assert b.get_stroke_opacity() == pytest.approx(1.0)
        assert b.get_stroke_width() == pytest.approx(2.0)
    rects = pd.DataFrame({"xmin": [0.0], "xmax": [1.0], "ymin": [0.0], "ymax": [1.0]})
    rect = (
        ggplot(rects)
        .add(geom_rect(aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"), alpha=0.4))
        .build()
        .layers[0][0]
    )
    assert rect.get_fill_opacity() == bars[0].get_fill_opacity()


def test_no_outline_by_default_and_no_seal():
    for b in _bars(geom_col()):
        assert type(b) is VMobject
        assert b.get_stroke_width() == 0 and b.get_fill_opacity() == pytest.approx(1.0)


def test_dashed_outline_is_fill_plus_dashes():
    bars = _bars(geom_col(color="black", linetype="dashed"))
    assert len(bars) == 3
    for b in bars:
        assert isinstance(b, VGroup) and len(b) == 2
        fill, dashes = b
        assert fill.get_stroke_width() == 0 and len(dashes) > 1


def test_seal_class_flag_seals_opaque_bars():
    class Sealed(GeomCol):
        seal = True

    from ggmanim.geoms.base import make_layer

    layer = make_layer(Sealed(), None, None, stat="identity", params={})
    b = _bars(layer)[0]
    assert b.get_stroke_width() > 0
    assert b.get_stroke_color().to_hex() == b.get_fill_color().to_hex()


def test_fast_path_matches_slow_path_style_and_geometry():
    df = pd.DataFrame({"x": np.arange(30.0), "y": np.arange(1.0, 31.0)})
    kw = {"alpha": 0.5, "color": "#0000ff", "fill": "#ff0000"}
    slow = _bars(geom_col(fast=False, **kw), data=df)
    fast = _bars(geom_col(fast=True, **kw), data=df)
    assert is_batched(fast) and len(fast) == 1
    batch = fast[0]
    assert batch.get_fill_opacity() == pytest.approx(0.5)
    assert batch.get_stroke_opacity() == pytest.approx(1.0)  # alpha is the fill's
    np.testing.assert_allclose(np.concatenate([b.points for b in slow]), batch.points, atol=1e-9)


def test_fast_path_draws_dashed_bars_per_row():
    df = pd.DataFrame({"x": np.arange(5.0), "y": np.arange(1.0, 6.0)})
    fast = _bars(geom_col(fast=True, color="black", linetype="dotted"), data=df)
    assert is_batched(fast)
    assert sum(isinstance(m, VGroup) for m in fast) == 5


def test_geom_bar_shares_the_drawing():
    bars = ggplot(DF, aes("x")).add(geom_bar(alpha=0.3, color="black")).build().layers[0]
    assert len(bars) == 3
    assert all(b.get_fill_opacity() == pytest.approx(0.3) for b in bars)
    assert all(b.get_stroke_opacity() == pytest.approx(1.0) for b in bars)


def test_key_is_the_filled_square():
    key = GeomCol().draw_key({"fill": "#ff0000", "alpha": 0.5, "color": "black"}, theme_default())
    assert key.get_fill_opacity() == pytest.approx(0.5)
    assert key.get_stroke_opacity() == pytest.approx(1.0)
