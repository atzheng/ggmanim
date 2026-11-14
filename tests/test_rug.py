"""geom_rug."""

import numpy as np
import pandas as pd
import pytest

import ggmanim
from ggmanim import aes, coord_flip, geom_point, geom_rug, ggplot
from ggmanim.geoms import GeomRug
from ggmanim.registry import COMPONENTS

DF = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 6.0, 8.0]})


def test_registered_and_exported():
    assert "geom_rug" in COMPONENTS and "geom_rug" in ggmanim.__all__
    assert isinstance(geom_rug().geom, GeomRug)
    with pytest.raises(ValueError, match="sides"):
        geom_rug(sides="bx")


def test_ticks_on_bottom_and_left():
    p = ggplot(DF, aes("x", "y")).add(geom_point(), geom_rug()).build()
    marks = p.layers[1].marks
    assert len(marks) == 3
    pp = p.panel.panel_params
    for (x, y), m in zip(DF.to_numpy(), marks):
        paths = m.get_subpaths()
        assert len(paths) == 2
        bottom = p.panel.transform(np.array([x]), np.array([pp.y_range[0]]))[0]
        left = p.panel.transform(np.array([pp.x_range[0]]), np.array([y]))[0]
        h = p.panel.transform(np.array([x]), np.array([pp.y_range[1]]))[0][1] - bottom[1]
        np.testing.assert_allclose(paths[0][0], bottom, atol=1e-9)
        assert paths[0][-1][1] - bottom[1] == pytest.approx(0.03 * h)
        np.testing.assert_allclose(paths[1][0], left, atol=1e-9)


def test_sides_length_outside_and_x_only():
    p = ggplot(DF, aes("x")).add(geom_rug(sides="tb", length=0.2, outside=True)).build()
    m = p.layers[0].marks[0]
    top, bottom = m.get_subpaths()  # in "tb" order
    assert bottom[-1][1] - bottom[0][1] == pytest.approx(-0.2)
    assert top[-1][1] - top[0][1] == pytest.approx(0.2)
    p = ggplot(DF, aes("x", "y")).add(geom_rug(sides="trbl"), coord_flip()).build()
    assert all(len(m.get_subpaths()) == 4 for m in p.layers[0].marks)
    assert GeomRug().draw_key({}).height > 0


def test_missing_values_skip_ticks():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [1.0, 2.0, np.nan]})
    p = ggplot(df, aes("x", "y")).add(geom_rug()).build()
    assert [len(m.get_subpaths()) for m in p.layers[0].marks] == [2, 1, 1]
