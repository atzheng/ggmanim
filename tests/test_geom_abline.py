"""geom_abline: spans the panel, trains no scales, clipping, defaults."""

import numpy as np
import pandas as pd
import pytest

import ggmanim
from ggmanim import aes, coord_cartesian, geom_abline, ggplot, scale_x_log10
from ggmanim.geoms import GeomAbline
from ggmanim.registry import COMPONENTS

DF = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})


def test_registered_and_exported():
    assert "geom_abline" in COMPONENTS and "geom_abline" in ggmanim.__all__
    layer = geom_abline(slope=2)
    assert isinstance(layer.geom, GeomAbline) and not layer.inherit_aes
    np.testing.assert_allclose(layer.data["intercept"], [0.0])  # missing intercept -> 0
    np.testing.assert_allclose(geom_abline().data["slope"], [1.0])  # y = x
    with pytest.raises(TypeError):
        geom_abline(aes(slope="s"), slope=1)


def test_spans_panel_and_trains_nothing():
    base = ggplot(DF, aes("x", "y")).geom_point().build()
    m = ggplot(DF, aes("x", "y")).geom_point().geom_abline(slope=0.5, intercept=1).build()
    assert m.panel.panel_params.x_range == base.panel.panel_params.x_range
    assert m.panel.panel_params.y_range == base.panel.panel_params.y_range
    (line,) = m.layers[1].marks
    x0, x1 = m.panel.panel_params.x_range
    np.testing.assert_allclose(line.get_start(), m.panel.transform(x0, 1 + 0.5 * x0)[0])
    np.testing.assert_allclose(line.get_end(), m.panel.transform(x1, 1 + 0.5 * x1)[0])


def test_spans_panel_on_zoom():
    m = (
        ggplot(DF, aes("x", "y"))
        .geom_point()
        .geom_abline(slope=0.1, intercept=4)
        .add(coord_cartesian(xlim=(2, 6)))
        .build()
    )
    (line,) = m.layers[1].marks
    x0, x1 = m.panel.panel_params.x_range
    assert x0 < 2 and x1 > 6  # expanded zoom range
    assert line.get_start()[0] == pytest.approx(m.panel.transform(x0, 0)[0][0])
    assert line.get_end()[0] == pytest.approx(m.panel.transform(x1, 0)[0][0])


def test_steep_line_clipped_and_outside_dropped():
    m = (
        ggplot(DF, aes("x", "y"))
        .geom_point()
        .geom_abline(slope=[10.0, 0.0], intercept=[0.0, 100.0])
        .build()
    )
    layer = m.layers[1]
    assert layer.keys == [0]  # the horizontal line at y = 100 is outside
    (line,) = layer.marks
    y0, y1 = m.panel.panel_params.y_range
    ys = sorted([line.get_start()[1], line.get_end()[1]])
    assert ys[0] == pytest.approx(m.panel.transform(0, y0)[0][1])
    assert ys[1] == pytest.approx(m.panel.transform(0, y1)[0][1])


def test_transformed_units_on_log_scale():
    df = pd.DataFrame({"x": [1.0, 1000.0], "y": [0.0, 3.0]})
    m = ggplot(df, aes("x", "y")).geom_point().geom_abline(slope=1).add(scale_x_log10()).build()
    (line,) = m.layers[1].marks
    x0, x1 = m.panel.panel_params.x_range  # log10 space
    np.testing.assert_allclose(line.get_end(), m.panel.transform(x1, x1)[0])


def test_mapped_slopes_with_colour():
    lines = pd.DataFrame({"s": [0.5, 1.0], "i": [0.0, 1.0], "kind": ["a", "b"]})
    m = (
        ggplot(DF, aes("x", "y"))
        .geom_point()
        .add(geom_abline(aes(slope="s", intercept="i", color="kind"), lines))
        .build()
    )
    marks = m.layers[1].marks
    assert len(marks) == 2
    assert marks[0].get_stroke_color().to_hex() != marks[1].get_stroke_color().to_hex()
