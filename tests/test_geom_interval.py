"""geom_errorbar / errorbarh / linerange / pointrange / crossbar and draw_crossbar."""

import numpy as np
import pandas as pd
import pytest
from manim import Polygon, VGroup

import ggmanim
from ggmanim import (
    aes,
    coord_cartesian,
    coord_flip,
    geom_crossbar,
    geom_errorbar,
    geom_errorbarh,
    geom_linerange,
    geom_pointrange,
    ggplot,
)
from ggmanim.geoms import (
    GeomCrossbar,
    GeomErrorbar,
    GeomErrorbarh,
    GeomLinerange,
    GeomPointrange,
    draw_crossbar,
)
from ggmanim.registry import COMPONENTS

DF = pd.DataFrame(
    {"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 2.5], "lo": [1.0, 2.0, 1.5], "hi": [3.0, 4.5, 3.0]}
)
MAP = aes("x", "y", ymin="lo", ymax="hi")


def _subpaths(m):
    return [np.asarray(s) for s in m.get_subpaths()]


def _tf(m, x, y):
    return m.panel.transform(np.asarray(x, float), np.asarray(y, float))


@pytest.mark.parametrize(
    "ctor, cls",
    [
        (geom_errorbar, GeomErrorbar),
        (geom_errorbarh, GeomErrorbarh),
        (geom_linerange, GeomLinerange),
        (geom_pointrange, GeomPointrange),
        (geom_crossbar, GeomCrossbar),
    ],
)
def test_registered_and_exported(ctor, cls):
    name = ctor.__name__
    assert name in COMPONENTS and name in ggmanim.__all__
    assert isinstance(ctor().geom, cls)


def test_errorbar_positions_and_width_training():
    m = ggplot(DF, MAP).geom_errorbar(width=0.4).build()
    marks = m.layers[0].marks
    assert len(marks) == 3 and m.layers[0].keys == [0, 1, 2]
    top, stem, bottom = _subpaths(marks[1])
    np.testing.assert_allclose(top[[0, -1]], _tf(m, [1.8, 2.2], [4.5, 4.5]), atol=1e-9)
    np.testing.assert_allclose(stem[[0, -1]], _tf(m, [2, 2], [2.0, 4.5]), atol=1e-9)
    np.testing.assert_allclose(bottom[[0, -1]], _tf(m, [1.8, 2.2], [2.0, 2.0]), atol=1e-9)
    lo, hi = m.panel.panel_params.x_range
    assert lo < 0.8 and hi > 3.2  # whiskers train the x scale


def test_errorbar_default_width_is_09_resolution():
    data = GeomErrorbar().setup_data(DF.rename(columns={"lo": "ymin", "hi": "ymax"}), {})
    np.testing.assert_allclose(data["xmax"] - data["xmin"], 0.9)
    assert not data["flipped_aes"].any()


def test_errorbar_horizontal_orientation_and_errorbarh():
    h = pd.DataFrame({"y": [1.0, 2.0], "xmin": [0.0, 1.0], "xmax": [2.0, 4.0]})
    for layer in (
        geom_errorbar(aes(y="y", xmin="xmin", xmax="xmax"), h, width=0.5),
        geom_errorbarh(aes(y="y", xmin="xmin", xmax="xmax"), h, height=0.5),
    ):
        m = ggplot().add(layer).build()
        left, stem, right = _subpaths(m.layers[0].marks[1])
        np.testing.assert_allclose(stem[[0, -1]], _tf(m, [1.0, 4.0], [2, 2]), atol=1e-9)
        np.testing.assert_allclose(left[[0, -1]], _tf(m, [4.0, 4.0], [1.75, 2.25]), atol=1e-9)
        lo, hi = m.panel.panel_params.y_range
        assert lo < 0.75 and hi > 2.25  # height trains y


def test_errorbar_explicit_orientation_y():
    h = pd.DataFrame({"y": [1.0], "xmin": [0.0], "xmax": [2.0], "ymin": [0.5], "ymax": [1.5]})
    m = ggplot(h, aes(y="y", xmin="xmin", xmax="xmax")).geom_errorbar(orientation="y").build()
    _, stem, _ = _subpaths(m.layers[0].marks[0])
    assert stem[0][1] == pytest.approx(stem[-1][1])  # horizontal stem


def test_linerange_and_clipping():
    m = ggplot(DF, MAP).geom_linerange().add(coord_cartesian(ylim=(2.5, 4), expand=False)).build()
    marks = m.layers[0].marks
    assert m.layers[0].keys == [0, 1, 2]
    np.testing.assert_allclose(marks[1].get_start(), _tf(m, 2, 2.5)[0], atol=1e-9)
    np.testing.assert_allclose(marks[1].get_end(), _tf(m, 2, 4)[0], atol=1e-9)
    m = ggplot(DF, MAP).geom_linerange().add(coord_cartesian(ylim=(3.5, 4), expand=False)).build()
    assert m.layers[0].keys == [1]  # the others are entirely below the panel


def test_pointrange_point_size_fatten():
    m = ggplot(DF, MAP).geom_pointrange().build()
    line, dot = m.layers[0].marks[0]
    np.testing.assert_allclose(dot.get_center(), _tf(m, 1, 2)[0], atol=1e-9)
    assert dot.width / 2 == pytest.approx(0.02 * 4)
    m = ggplot(DF, MAP).geom_pointrange(fatten=2, size=0.03).build()
    _, dot = m.layers[0].marks[0]
    assert dot.width / 2 == pytest.approx(0.06)


def test_crossbar_box_and_middle():
    m = ggplot(DF, MAP).geom_crossbar(width=0.5, fill="#FF0000", fatten=3).build()
    box, mid = m.layers[0].marks[0]
    assert isinstance(box, Polygon)
    corners = box.get_vertices()
    np.testing.assert_allclose(
        corners, _tf(m, [0.75, 1.25, 1.25, 0.75], [1.0, 1.0, 3.0, 3.0]), atol=1e-9
    )
    assert box.get_fill_opacity() == pytest.approx(1.0)
    np.testing.assert_allclose(mid.get_start(), _tf(m, 0.75, 2.0)[0], atol=1e-9)
    assert mid.get_stroke_width() == pytest.approx(box.get_stroke_width() * 3)


def test_crossbar_no_fill_by_default_and_dashed():
    m = ggplot(DF, MAP).geom_crossbar(linetype="dashed").build()
    box, _ = m.layers[0].marks[0]
    assert isinstance(box, VGroup)  # fill polygon + dashes
    assert box[0].get_fill_opacity() == 0


def test_draw_crossbar_reusable_and_flipped():
    m = ggplot(DF, MAP).geom_point().build()
    rows = pd.DataFrame(
        {"xmin": [0.95], "xmax": [1.2], "ymin": [1.0], "ymax": [3.0], "y": [2.0], "key": ["k"]}
    )
    out, drawn = draw_crossbar(rows, m.panel, None, m.panel.panel_params)
    assert len(out) == 1 and list(drawn["key"]) == ["k"]
    box, mid = out[0]
    np.testing.assert_allclose(mid.get_start(), _tf(m, 0.95, 2.0)[0], atol=1e-9)
    # flipped: the rows' x is drawn on the y axis
    out, _ = draw_crossbar(rows.assign(xmin=2.0, xmax=2.5, ymin=1.0, ymax=2.0, y=1.5), m.panel,
                           None, m.panel.panel_params, flipped=True)  # fmt: skip
    _, mid = out[0]
    np.testing.assert_allclose(mid.get_start(), _tf(m, 1.5, 2.0)[0], atol=1e-9)
    # entirely outside the panel: dropped
    out, drawn = draw_crossbar(rows.assign(ymin=50.0, ymax=60.0), m.panel, None,
                               m.panel.panel_params)  # fmt: skip
    assert len(out) == 0 and len(drawn) == 0


def test_coord_flip_draws_horizontally():
    m = ggplot(DF, MAP).geom_errorbar(width=0.4).add(coord_flip()).build()
    _, stem, _ = _subpaths(m.layers[0].marks[0])
    assert stem[0][1] == pytest.approx(stem[-1][1])


def test_missing_aes_error():
    with pytest.raises(Exception, match="ymin|ymax|missing"):
        ggplot(DF, aes("x", "y")).geom_errorbar().build()


def test_draw_keys():
    for geom in (GeomErrorbar(), GeomLinerange(), GeomPointrange(), GeomCrossbar()):
        key = geom.draw_key({"color": "#FF0000"})
        assert key.width <= 0.35 and key.height <= 0.35
