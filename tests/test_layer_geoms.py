"""The build records ``GGPlot.layer_geoms``; geoms record ``batched`` on their output."""

import numpy as np
import pandas as pd
import pytest
from manim import VGroup

from ggmanim import aes, facet_wrap, geom_col, geom_point, geom_raster, ggplot
from ggmanim.animate import GrowFromBaseline, attach_geoms
from ggmanim.geoms import GeomCol, GeomPoint, attach_marks


def _df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 1.0, 3.0, 1.0], "f": list("aabb")})


def _leaves(anim):
    subs = getattr(anim, "animations", None)
    if not subs:
        return [anim]
    return [leaf for a in subs for leaf in _leaves(a)]


def test_build_records_layer_geoms():
    spec = ggplot(_df(), aes("x", "y")).add(geom_col(), name="bars").add(geom_point())
    plot = spec.build()
    assert list(plot.layer_geoms) == ["bars", 1]
    assert isinstance(plot.layer_geoms["bars"], GeomCol)
    assert isinstance(plot.layer_geoms[1], GeomPoint)
    # bars grow without attach_geoms
    assert all(isinstance(a, GrowFromBaseline) for a in _leaves(plot.reveal_layer("bars")))


def test_faceted_build_records_layer_geoms():
    spec = ggplot(_df(), aes("x", "y")).add(geom_col(), name="bars").add(facet_wrap("f"))
    plot = spec.build()
    assert isinstance(plot.layer_geoms["bars"], GeomCol)
    assert all(isinstance(a, GrowFromBaseline) for a in _leaves(plot.reveal_layer("bars")))


def test_attach_geoms_still_overrides():
    spec = ggplot(_df(), aes("x", "y")).add(geom_col(), name="bars")
    plot = spec.build()
    other = ggplot(_df(), aes("x", "y")).add(geom_point(), name="bars")
    assert attach_geoms(plot, other) is plot
    assert isinstance(plot.layer_geoms["bars"], GeomPoint)


def test_attach_marks_records_batched():
    g = attach_marks(VGroup(VGroup()), pd.DataFrame({"x": [1]}))
    assert g.batched is False
    g = attach_marks(VGroup(VGroup()), pd.DataFrame({"x": [1]}), batched=True)
    assert g.batched is True


def _grid():
    return pd.DataFrame(
        {
            "x": np.repeat([1, 2, 3], 3),
            "y": np.tile([1, 2, 3], 3),
            "v": ["p", "q", "r"] * 3,
            "f": ["a"] * 9,
        }
    )


def test_raster_output_is_batched():
    plot = ggplot(_grid(), aes("x", "y", fill="v")).add(geom_raster(), name="r").build()
    assert plot.layers["r"].batched is True
    with pytest.raises(TypeError, match="batches"):
        plot.mark("r", 0)


def test_faceted_raster_mark_raises():
    spec = ggplot(_grid(), aes("x", "y", fill="v")).add(geom_raster(), name="r")
    plot = spec.add(facet_wrap("f")).build()
    with pytest.raises(TypeError, match="batches"):
        plot.mark("r", 0)
