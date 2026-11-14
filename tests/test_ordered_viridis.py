"""Ordered categoricals mapped to colour/fill default to viridis_d (ggplot2's
``scale_colour_ordinal``); unordered ones keep the hue palette."""

import numpy as np
import pandas as pd

from ggmanim import aes, geom_col, geom_density_2d_filled, ggplot
from ggmanim.palettes import hue_pal, viridis_pal
from ggmanim.scales import ScaleDiscrete, default_scale_for


def _hexes(colours):
    return [c.to_hex().lower() for c in colours]


def test_default_scale_ordered_is_viridis():
    vals = pd.Series(pd.Categorical(["lo", "mid", "hi"], ["lo", "mid", "hi"], ordered=True))
    for aes_name in ("fill", "colour", "color"):
        sc = default_scale_for(aes_name, vals)
        assert isinstance(sc, ScaleDiscrete)
        sc.train(vals)
        assert _hexes(sc.map(vals)) == _hexes(viridis_pal()(3))


def test_default_scale_unordered_stays_hue():
    vals = pd.Series(pd.Categorical(["a", "b", "c"]))
    sc = default_scale_for("fill", vals)
    sc.train(vals)
    assert _hexes(sc.map(vals)) == _hexes(hue_pal()(3))
    sc = default_scale_for("fill", pd.Series(["a", "b", "c"]))
    sc.train(pd.Series(["a", "b", "c"]))
    assert _hexes(sc.map(pd.Series(["a", "b", "c"]))) == _hexes(hue_pal()(3))


def test_ordered_shape_unchanged():
    vals = pd.Series(pd.Categorical(["a", "b"], ordered=True))
    sc = default_scale_for("shape", vals)
    sc.train(vals)
    plain = default_scale_for("shape", pd.Series(["a", "b"]))
    plain.train(pd.Series(["a", "b"]))
    assert list(sc.map(vals)) == list(plain.map(pd.Series(["a", "b"])))


def test_build_ordered_fill_uses_viridis():
    df = pd.DataFrame(
        {
            "g": pd.Categorical(["small", "medium", "large"], ["small", "medium", "large"], True),
            "v": [1.0, 2.0, 3.0],
        }
    )
    g = ggplot(df, aes("g", "v", fill="g")).add(geom_col()).build()
    got = {m.get_fill_color().to_hex().lower() for m in g.layers[0].marks}
    assert got == set(_hexes(viridis_pal()(3)))


def test_density_2d_filled_levels_use_viridis():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    g = ggplot(df, aes("x", "y")).add(geom_density_2d_filled(bins=4)).build()
    got = {m.get_fill_color().to_hex().lower() for m in g.layers[0].marks}
    assert got <= set(_hexes(viridis_pal()(4)))
