"""Live-plot follow-ups (ggm-dpj): spec on any build, faceted live updates."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, ggplot, xlim, ylim
from ggmanim.build import BuildError
from ggmanim.live import update_data


def _df():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "g": ["a", "a", "a", "b", "b", "b"],
        }
    )


def _spec(df):
    return ggplot(df, aes("x", "y")).geom_point().facet_wrap("g").add(xlim(0, 5), ylim(0, 5))


def test_build_sets_spec():
    spec = ggplot(_df(), aes("x", "y")).geom_point().add(xlim(0, 5), ylim(0, 5))
    plot = spec.build()
    assert plot.spec is spec
    plot.update_data(_df().assign(y=4.0))  # no build_live needed
    assert next(iter(plot.layer_data.values()))["y"].tolist() == [4.0] * 6


def test_faceted_spec_set_and_update_in_place():
    df = _df()
    spec = _spec(df)
    plot = spec.build()
    assert plot.spec is spec
    layer = next(iter(plot.layers.values()))
    marks = list(layer.marks)
    ids = [id(m) for m in marks]
    family = [id(m) for m in plot.panel_grid.get_family()]

    plot.update_data(df.assign(y=[4.0, 4.0, 4.0, 0.5, 0.5, 0.5]))

    layer2 = next(iter(plot.layers.values()))
    assert layer2 is layer
    assert [id(m) for m in layer2.marks] == ids
    assert [id(m) for m in plot.panel_grid.get_family()] == family
    p11, p12 = plot.panels[(1, 1)], plot.panels[(1, 2)]
    assert np.allclose(marks[0].get_center(), p11.transform(1.0, 4.0)[0], atol=1e-6)
    assert np.allclose(marks[3].get_center(), p12.transform(1.0, 0.5)[0], atol=1e-6)


def test_faceted_row_count_change_replaces_layer():
    df = _df()
    plot = _spec(df).build()
    update_data(plot, df.iloc[[0, 1, 3, 4, 5]])
    layer = next(iter(plot.layers.values()))
    assert len(layer.marks) == 5
    assert any(layer is m for m in plot.get_family())


def test_faceted_new_level_raises():
    df = _df()
    plot = _spec(df).build()
    with pytest.raises(BuildError, match="facet layout"):
        plot.update_data(df.assign(g=["a", "a", "a", "b", "b", "c"]))


def test_faceted_needs_fixed_limits():
    plot = ggplot(_df(), aes("x", "y")).geom_point().facet_wrap("g").build()
    with pytest.raises(BuildError, match="fixed"):
        plot.update_data(_df())
