"""annotate(): layers from constant vectors, without a data frame."""

import numpy as np
import pandas as pd
import pytest
from manim import BLUE, RED

from ggmanim import aes, annotate, geom_point, ggplot
from ggmanim.geoms import GeomPoint, GeomText, GeomVline
from ggmanim.layer import Layer
from ggmanim.registry import COMPONENTS

DF = pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 10.0], "g": ["u", "v"]})


def _plot(*layers):
    p = ggplot(DF, aes("a", "b", color="g")).add(geom_point())
    for layer in layers:
        p = p.add(layer)
    return p


def test_text_layer_is_one_row_and_does_not_inherit():
    layer = annotate("text", x=5, y=5, label="peak", color=RED, size=5)
    assert isinstance(layer, Layer) and isinstance(layer.geom, GeomText)
    assert layer.inherit_aes is False
    assert list(layer.data.columns) == ["x", "y", "label"]
    assert len(layer.data) == 1
    assert layer.params["size"] == 5
    m = _plot(layer).build()
    (mob,) = m.layers[1].marks
    assert mob.text == "peak"
    assert mob.get_fill_color().to_hex().upper() == RED.to_hex().upper()
    np.testing.assert_allclose(mob.get_center(), m.panel.transform(5.0, 5.0)[0], atol=1e-6)


def test_vectors_make_n_rows_and_scalars_recycle():
    layer = annotate("text", x=[1, 2, 3], y=4, label=["a", "b", "c"])
    assert len(layer.data) == 3
    assert list(layer.data["y"]) == [4, 4, 4]
    m = _plot(layer).build()
    assert [t.text for t in m.layers[1].marks] == ["a", "b", "c"]


def test_annotations_train_position_scales():
    m = _plot(annotate("text", x=20, y=-5, label="far")).build()
    pp = m.panel.panel_params
    assert pp.x_range[1] >= 20 and pp.y_range[0] <= -5


def test_point_with_vector_colours_used_as_is():
    layer = annotate("point", x=[1, 2], y=[1, 2], color=[RED, BLUE], size=0.1)
    assert isinstance(layer.geom, GeomPoint)
    m = _plot(layer).build()
    dots = m.layers[1].marks
    assert [d.get_fill_color().to_hex().upper() for d in dots] == [
        RED.to_hex().upper(),
        BLUE.to_hex().upper(),
    ]
    assert len(m.legends) == 1  # only the plot's own colour legend


def test_vline():
    layer = annotate("vline", xintercept=[2, 4], linetype="dashed")
    assert isinstance(layer.geom, GeomVline) and len(layer.data) == 2
    assert len(_plot(layer).build().layers[1].marks) == 2


@pytest.mark.parametrize(
    "geom,kw",
    [
        ("rect", {"xmin": 1, "xmax": 3, "ymin": 2, "ymax": 4, "alpha": 0.2}),
        ("segment", {"x": 1, "y": 1, "xend": 4, "yend": 6}),
    ],
)
def test_rect_and_segment(geom, kw):
    if "geom_" + geom not in COMPONENTS:
        pytest.skip(f"geom_{geom} not merged yet")
    layer = annotate(geom, **kw)  # pragma: no cover - until geom_rect/segment land
    assert layer.inherit_aes is False and len(layer.data) == 1
    m = _plot(layer).build()
    assert len(m.layers[1].marks) == 1


def test_errors():
    with pytest.raises(ValueError, match="unknown geom"):
        annotate("nope", x=1, y=1)
    with pytest.raises(ValueError, match="unequal"):
        annotate("text", x=[1, 2], y=[1, 2, 3], label="a")
    with pytest.raises(ValueError, match="position"):
        annotate("text", label="a")


def test_name_and_geom_prefix():
    layer = annotate("geom_point", x=1, y=1, name="marker")
    assert layer.name == "marker"
    m = _plot(layer).build()
    assert len(m.layers["marker"].marks) == 1
    assert geom_point is COMPONENTS["geom_point"]
