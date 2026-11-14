"""ggm-pgf: jitter seeded per layer, line labels keyed by group value, path geoms
squish +-inf to the panel edges, stat_align(orientation="y")."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    geom_hline,
    geom_jitter,
    geom_line,
    geom_path,
    geom_step,
    ggplot,
    stat_align,
)
from ggmanim.geoms import GeomPath
from ggmanim.stats import StatAlign

# -- jitter seed from the layer index -----------------------------------------------


def _jittered(plot, key):
    return plot.layer_data[key][["x", "y"]].to_numpy()


def test_identical_jitter_layers_differ_but_builds_repeat():
    df = pd.DataFrame({"x": [1.0, 1.0, 2.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0]})
    spec = ggplot(df, aes("x", "y")).add(geom_jitter(), name="a").add(geom_jitter(), name="b")
    p1, p2 = spec.build(), spec.build()
    assert not np.allclose(_jittered(p1, "a"), _jittered(p1, "b"))
    np.testing.assert_array_equal(_jittered(p1, "a"), _jittered(p2, "a"))
    np.testing.assert_array_equal(_jittered(p1, "b"), _jittered(p2, "b"))


def test_explicit_seed_ignores_layer_index():
    df = pd.DataFrame({"x": [1.0, 1.0, 2.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0]})
    spec = (
        ggplot(df, aes("x", "y"))
        .add(geom_jitter(seed=3), name="a")
        .add(geom_jitter(seed=3), name="b")
    )
    p = spec.build()
    np.testing.assert_array_equal(_jittered(p, "a"), _jittered(p, "b"))


# -- line labels keyed by raw group value ---------------------------------------------


def _df():
    return pd.DataFrame({"x": [1, 2, 1, 2], "y": [1.0, 2.0, 2.0, 3.0], "s": ["b", "b", "a", "a"]})


def _texts(layer):
    return [t.text for t in layer.labels]


def test_label_dict_by_group_value():
    spec = ggplot(_df(), aes("x", "y", color="s"))
    p = spec.add(geom_line(label={"b": "BEE", "a": "AY"}), name="l").build()
    assert _texts(p.layers["l"]) == ["AY", "BEE"]  # group order: a, b
    p = spec.add(geom_line(label={"b": "BEE"}), name="l").build()
    assert _texts(p.layers["l"]) == ["BEE"]


def test_label_dict_by_group_id_still_works():
    p = ggplot(_df(), aes("x", "y", color="s")).add(geom_line(label={2: "two"}), name="l").build()
    assert _texts(p.layers["l"]) == ["two"]


def test_label_dict_by_tuple_of_values():
    df = _df().assign(t=["u", "u", "v", "v"])
    spec = ggplot(df, aes("x", "y", color="s", linetype="t"))
    p = spec.add(geom_path(label={("a", "v"): "A"}), name="l").build()
    assert _texts(p.layers["l"]) == ["A"]


def test_hline_label_dict_stays_positional():
    df = pd.DataFrame({"yintercept": [1.0, 2.0], "c": ["q", "p"]})
    base = ggplot(_df(), aes("x", "y"))
    hl = aes(yintercept="yintercept", color="c")
    p = base.add(geom_hline(hl, data=df, label={0: "first", "p": "P"})).build()
    assert _texts(p.layers[0]) == ["first"]  # keyed by row position, not group value


# -- +-inf on the panel edges ---------------------------------------------------------


def test_path_handle_na_keeps_inf():
    data = pd.DataFrame({"x": [-np.inf, 1.0, np.nan, 2.0], "y": [0.0, 1.0, 1.0, np.inf]})
    with pytest.warns(UserWarning, match="removed 1 row"):
        kept = GeomPath().handle_na(data)
    assert kept["x"].tolist() == [-np.inf, 1.0, 2.0]


@pytest.mark.parametrize("geom", [geom_path, geom_line, geom_step])
def test_path_geoms_draw_inf_on_panel_edges(geom):
    df = pd.DataFrame({"x": [-np.inf, 1.0, 2.0, np.inf], "y": [0.0, 1.0, 2.0, 3.0]})
    p = ggplot(df, aes("x", "y")).add(geom(), name="l").build()
    pts = p.layers["l"].marks[0].get_points()
    pp = p.panel.panel_params
    left = p.panel.transform([pp.x_range[0]], [0.0])[0][0]
    right = p.panel.transform([pp.x_range[1]], [0.0])[0][0]
    assert pts[:, 0].min() == pytest.approx(left)
    assert pts[:, 0].max() == pytest.approx(right)


# -- stat_align(orientation="y") ------------------------------------------------------


def test_stat_align_orientation_y_is_the_transpose():
    df = pd.DataFrame({"y": [1, 2, 3, 1.5, 2.5], "x": [1, 2, 1, 3, 2], "group": [1, 1, 1, 2, 2]})
    flipped = StatAlign(orientation="y").compute(df)
    plain = StatAlign().compute(df.rename(columns={"x": "y", "y": "x"}))
    assert flipped["flipped_aes"].all()
    np.testing.assert_allclose(flipped["y"], plain["x"])
    np.testing.assert_allclose(flipped["x"], plain["y"])


def test_stat_align_orientation_validated_and_layer_builds():
    with pytest.raises(ValueError, match="orientation"):
        StatAlign(orientation="z")
    df = pd.DataFrame({"y": [1, 2, 3, 1.5, 2.5], "x": [1, 2, 1, 3, 2], "g": list("aaabb")})
    layer = stat_align(orientation="y", geom="path", position="identity")
    p = ggplot(df, aes("x", "y", group="g")).add(layer, name="a").build()
    assert len(p.layers["a"].marks) == 2
