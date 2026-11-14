"""stat_align: shared x grid, zero crossings, padding rows, stacking via geom_area."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, geom_area, ggplot, stat_align
from ggmanim.geoms import GeomArea
from ggmanim.registry import COMPONENTS
from ggmanim.stats import STATS, StatAlign, get_stat
from ggmanim.stats.align import align_grid


def test_registered():
    assert STATS["align"] is StatAlign
    assert "stat_align" in COMPONENTS
    assert isinstance(get_stat("align"), StatAlign)
    assert GeomArea.default_stat == "align"
    assert isinstance(geom_area().stat, StatAlign)


def test_grid_and_adjust():
    df = pd.DataFrame({"x": [0.0, 10.0, 5.0, 10.0], "y": [1.0, 1, 2, 2], "group": [1, 1, 2, 2]})
    loc, adjust = align_grid(df)
    assert adjust == pytest.approx(0.01)  # 0.1% of the range
    np.testing.assert_allclose(loc, [-0.01, 0, 0.01, 4.99, 5, 5.01, 9.99, 10, 10.01])


def test_adjust_capped_by_smallest_gap():
    df = pd.DataFrame({"x": [0.0, 0.003, 10.0], "y": [1.0, 1, 1], "group": 1})
    _, adjust = align_grid(df)
    assert adjust == pytest.approx(0.001)


def test_groups_interpolated_onto_each_others_x():
    df = pd.DataFrame(
        {"x": [0.0, 10.0, 5.0, 10.0], "y": [0.0, 10.0, 2.0, 2.0], "group": [1, 1, 2, 2]}
    )
    out = StatAlign().compute(df)
    a = out[out["group"] == 1]
    # group 1 gains a row at x = 5 (from group 2), linearly interpolated
    assert a.loc[a["x"] == 5, "y"].item() == pytest.approx(5.0)
    b = out[out["group"] == 2]
    assert b["x"].min() == pytest.approx(5 - 0.01)  # padding starts just left of 5
    assert b["x"].max() == pytest.approx(10 + 0.01)


def test_padding_rows():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [3.0, 4.0, 5.0]})
    out = StatAlign().compute(df)
    pad = out["align_padding"].to_numpy()
    assert pad[0] and pad[-1] and not pad[1:-1].any()
    assert out["y"].iloc[0] == 0 and out["y"].iloc[-1] == 0
    inner = out[~out["align_padding"]]
    np.testing.assert_allclose(np.interp(inner["x"], [0, 1, 2], [3, 4, 5]), inner["y"])
    assert inner["x"].min() == 0 and inner["x"].max() == 2


def test_zero_crossing_added_to_grid():
    df = pd.DataFrame({"x": [0.0, 4.0], "y": [-1.0, 3.0]})
    loc, _ = align_grid(df.assign(group=1))
    assert np.isclose(loc, 1.0).any()
    out = StatAlign().compute(df)
    assert out.loc[np.isclose(out["x"], 1.0), "y"].item() == pytest.approx(0.0)


def test_duplicate_x_keeps_first_and_last():
    df = pd.DataFrame({"x": [0.0, 1.0, 1.0, 1.0, 2.0], "y": [1.0, 1.0, 9.0, 3.0, 3.0]})
    out = StatAlign().compute(df)
    inner = out[~out["align_padding"]]
    assert 9.0 not in inner["y"].to_numpy()
    assert inner.loc[np.isclose(inner["x"], 1.0), "y"].item() == pytest.approx(3.0)
    left = inner[inner["x"] < 1].sort_values("x")
    assert left["y"].iloc[-1] == pytest.approx(1.0)  # the jump starts just left of 1


def test_single_x_group_dropped():
    df = pd.DataFrame({"x": [1.0, 1.0, 0.0, 2.0], "y": [1.0, 2, 1, 1], "group": [1, 1, 2, 2]})
    out = StatAlign().compute(df)
    assert set(out["group"]) == {2}


def test_constant_columns_carried():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0], "fill": ["a", "a"]})
    out = StatAlign().compute(df)
    assert set(out["fill"]) == {"a"}


def test_area_stacks_groups_with_different_x():
    df = pd.DataFrame(
        {"x": [0.0, 2.0, 4.0, 1.0, 3.0], "y": [1.0, 1.0, 1.0, 2.0, 2.0], "g": list("aaabb")}
    )
    g = ggplot(df, aes("x", "y", fill="g")).add(geom_area()).build()
    data = g.layer_data[0]
    # at x = 2 group b (interpolated to 2) sits on group a's 1 (b is stacked first)
    at2 = data[np.isclose(data["x"], 2.0)].sort_values("group")
    assert at2["ymax"].max() == pytest.approx(3.0)
    # x = 1 and x = 3 exist in group a too: stacking is per shared x
    for x in (1.0, 3.0):
        assert data[np.isclose(data["x"], x)]["group"].nunique() == 2
    assert len(g.layers[0].marks) == 2


def test_stat_align_constructor_layer():
    layer = stat_align()
    assert isinstance(layer.geom, GeomArea) and isinstance(layer.stat, StatAlign)
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 1.0]})
    g = ggplot(df, aes("x", "y")).add(layer).build()
    assert len(g.layers[0].marks) == 1
