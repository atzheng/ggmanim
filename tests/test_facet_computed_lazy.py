"""Computed facets are added at build time (Facet.facet_data hook, ggm-7v3)."""

import numpy as np
import pandas as pd

from ggmanim import aes, facet_grid, facet_wrap, geom_point, ggplot
from ggmanim.facets import Facet


def _half(d):
    return np.where(d["x"] < 3, "low", "high")


def test_layer_added_after_the_facet_is_split():
    df = pd.DataFrame({"x": np.arange(6.0), "y": np.arange(6.0)})
    other = pd.DataFrame({"x": [0.5, 4.5, 5.0], "y": [1.0, 1.0, 2.0]})
    plot = (
        ggplot(df, aes("x", "y"))
        .add(geom_point())
        .add(facet_wrap({"half": _half}))
        .add(geom_point(data=other))
        .build()
    )
    assert set(plot.panels) == {(1, 1), (1, 2)}
    # levels sorted: "high" then "low"
    assert len(plot.layers[1].panel(1, 1).marks) == 2
    assert len(plot.layers[1].panel(1, 2).marks) == 1
    assert "half" not in other.columns and "half" not in df.columns


def test_spec_is_not_rewritten_and_callable_layer_data_is_split():
    df = pd.DataFrame({"x": np.arange(6.0), "y": np.arange(6.0)})
    spec = (
        ggplot(df, aes("x", "y"))
        .add(geom_point(data=lambda d: d[d["x"] > 1]))
        .add(facet_wrap({"half": _half}))
    )
    assert spec.data is df and "half" not in spec.data.columns
    plot = spec.build()
    assert len(plot.layers[0].panel(1, 1).marks) == 3  # x = 3, 4, 5
    assert len(plot.layers[0].panel(1, 2).marks) == 1  # x = 2


def test_facet_data_hook_default_and_grid():
    df = pd.DataFrame({"x": [1.0, 2.0], "g": ["a", "b"]})
    assert Facet.facet_data(facet_grid(cols="g"), df) is df
    assert facet_wrap("g").facet_data(df) is df
