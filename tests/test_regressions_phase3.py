"""Phase 3 regression tests: the seams Phase 2 left open, each checked through the
public API (``ggplot(...).build()``), plus a smoke test that every plot in
``examples/gallery_geoms.py`` builds.

Phase 2 raised in ``apply_positions`` for anything but ``position_identity``,
drew ``geom_bar`` with identity heights only (no counting), and had no
``stat_summary``, ``stat_align`` or parameterised position strings.
"""

import importlib.util
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from manim import Scene, VGroup

from ggmanim import (
    aes,
    after_stat,
    geom_area,
    geom_bar,
    geom_col,
    geom_histogram,
    geom_point,
    geom_text,
    ggplot,
    position_dodge,
    position_stack,
    stat_summary,
)
from ggmanim.positions import POSITIONS

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

DF = pd.DataFrame(
    {
        "x": ["a", "a", "b", "b", "b", "c"],
        "y": [1.0, 2.0, 3.0, 1.0, 2.0, 4.0],
        "g": ["u", "v", "u", "v", "u", "v"],
    }
)


# -- positions ------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(POSITIONS))
def test_every_registered_position_builds(name):
    # Phase 2: apply_positions raised NotImplementedError for non-identity positions.
    m = ggplot(DF, aes("x", "y", fill="g")).add(geom_col(position=name)).build()
    assert isinstance(m, VGroup)
    assert len(m.layer_data[0]) == len(DF)


def test_stack_fill_dodge_move_bars():
    stacked = ggplot(DF, aes("x", "y", fill="g")).add(geom_col()).build().layer_data[0]
    assert stacked.groupby("x")["ymax"].max().tolist() == [3.0, 6.0, 4.0]
    filled = ggplot(DF, aes("x", "y", fill="g")).add(geom_col(position="fill")).build()
    np.testing.assert_allclose(filled.layer_data[0].groupby("x")["ymax"].max(), 1.0)
    dodged = ggplot(DF, aes("x", "y", fill="g")).add(geom_col(position="dodge")).build()
    d = dodged.layer_data[0]
    assert (d["ymin"] == 0).all()
    two = d[d["x"] < 2.5]  # a and b have two groups each: side by side at half width
    np.testing.assert_allclose(two["xmax"] - two["xmin"], 0.45)


def test_position_strings_take_arguments():
    a = ggplot(DF, aes("x", "y", fill="g")).add(geom_col(position="dodge(width=0.5)")).build()
    b = ggplot(DF, aes("x", "y", fill="g")).add(geom_col(position=position_dodge(width=0.5)))
    pd.testing.assert_frame_equal(a.layer_data[0], b.build().layer_data[0])


def test_geom_point_jitter_moves_points_reproducibly():
    df = pd.DataFrame({"x": ["a"] * 20, "y": np.zeros(20)})
    one = ggplot(df, aes("x", "y")).add(geom_point(position="jitter")).build().layer_data[0]
    two = ggplot(df, aes("x", "y")).add(geom_point(position="jitter")).build().layer_data[0]
    assert one["x"].nunique() > 1
    pd.testing.assert_frame_equal(one, two)


# -- geom_bar counts ------------------------------------------------------------------


def test_geom_bar_counts_rows():
    # Phase 2: geom_bar() needed a y and drew identity heights.
    d = ggplot(DF, aes("x")).add(geom_bar()).build().layer_data[0]
    assert d["count"].tolist() == [2, 3, 1]
    assert d["ymax"].tolist() == [2, 3, 1]


def test_geom_bar_identity_still_available():
    d = ggplot(DF, aes("x", "y")).add(geom_bar(stat="identity")).build().layer_data[0]
    assert d.groupby("x")["ymax"].max().tolist() == [3.0, 6.0, 4.0]


def test_count_labels_stack_with_bars():
    p = (
        ggplot(DF, aes("x", fill="g"))
        .add(geom_bar())
        .add(geom_text(aes(label=after_stat("count")), stat="count", position=position_stack(0.5)))
    )
    built = p.build()
    bars, labels = built.layer_data[0], built.layer_data[1]
    mid = (bars["ymin"] + bars["ymax"]) / 2
    np.testing.assert_allclose(np.sort(labels["y"]), np.sort(mid))


def test_geom_histogram_counts_into_bins():
    df = pd.DataFrame({"v": [0.1, 0.2, 1.1, 1.2, 1.3, 2.5]})
    d = ggplot(df, aes("v")).add(geom_histogram(binwidth=1, boundary=0)).build().layer_data[0]
    assert d["count"].tolist() == [2, 3, 1]


# -- stats ----------------------------------------------------------------------------


def test_stat_summary_mean_se():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the NaN interval is dropped
        d = ggplot(DF, aes("x", "y")).add(stat_summary()).build().layer_data[0]
    np.testing.assert_allclose(d["y"], [1.5, 2.0, 4.0])
    se = d.dropna(subset=["ymin", "ymax"])  # one row in "c": no standard error
    assert len(se) == 2
    assert (se["ymin"] < se["y"]).all() and (se["y"] < se["ymax"]).all()


def test_geom_area_stacks_on_aligned_x():
    # stat_align interpolates each group onto the union of x, so groups with
    # different x stack without gaps.
    df = pd.DataFrame(
        {"x": [1, 2, 3, 1.5, 2.5, 3.5], "y": [1, 2, 3, 1, 1, 1.0], "g": list("aaabbb")}
    )
    d = ggplot(df, aes("x", "y", fill="g")).add(geom_area()).build().layer_data[0]
    top = d[d["x"] == 2.0].set_index("group")
    assert top["ymax"].max() == pytest.approx(3.0)  # a(2) = 2 on top of b(2) = 1


# -- the gallery ----------------------------------------------------------------------


@pytest.fixture
def gallery(monkeypatch):
    # gallery_geoms imports its sibling examples, as manim's loader allows.
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(
        "ggmanim_gallery_geoms", EXAMPLES / "gallery_geoms.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gallery_geoms_plots_build(gallery):
    makers = [getattr(gallery, n) for n in dir(gallery) if n.startswith("make_")]
    assert len(makers) >= 10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for make in makers:
            plots = make()
            for p in plots if isinstance(plots, list) else [plots]:
                assert isinstance(p.build(), VGroup)


def test_gallery_geoms_scenes_are_its_own(gallery):
    # ``manim -a`` renders every Scene whose module is this file; imported example
    # scenes would be rendered twice, and an abstract base would fail.
    own = [
        obj
        for _, obj in inspect.getmembers(gallery, inspect.isclass)
        if issubclass(obj, Scene) and obj.__module__ == gallery.__name__
    ]
    assert sorted(c.__name__ for c in own) == sorted(c.__name__ for c in gallery.SCENES)
    assert len(gallery.SCENES) == 10
    assert len(gallery.PANELS) == 9
