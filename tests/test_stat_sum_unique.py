"""stat_sum / geom_count and stat_unique."""

import numpy as np
import pandas as pd

from ggmanim import aes, after_stat, geom_count, geom_point, ggplot, stat_sum, stat_unique
from ggmanim.geoms.count import GeomCount
from ggmanim.stats import StatSum, StatUnique, get_stat


def test_stat_sum_counts_and_props():
    df = pd.DataFrame({"x": [1, 1, 2, 1, 2, 3], "y": [0, 0, 0, 0, 1, 1]})
    out = StatSum().compute(df)
    assert list(zip(out["x"], out["y"], out["n"], strict=True)) == [
        (1, 0, 3),
        (2, 0, 1),
        (2, 1, 1),
        (3, 1, 1),
    ]
    np.testing.assert_allclose(out["prop"], out["n"] / 6)


def test_stat_sum_weights_and_groups():
    df = pd.DataFrame(
        {"x": [1, 1, 1, 1], "y": [0, 0, 0, 0], "weight": [2.0, 3, 1, 1], "color": list("aabb")}
    )
    out = StatSum().compute(df)
    assert out["n"].tolist() == [5, 2] and out["color"].tolist() == ["a", "b"]
    assert out["prop"].tolist() == [1.0, 1.0]
    assert "weight" not in out.columns
    assert isinstance(get_stat("sum"), StatSum)


def test_geom_count_maps_size_to_n_unless_given():
    layer = geom_count()
    assert isinstance(layer.geom, GeomCount)
    assert layer.mapping["size"] == after_stat("n")
    assert "size" not in geom_count(size=0.1).mapping
    assert geom_count(aes(size=after_stat("prop"))).mapping["size"] == after_stat("prop")
    assert stat_sum().mapping["size"] == after_stat("n")


def test_geom_count_build_sizes_by_count():
    df = pd.DataFrame({"x": [1, 1, 1, 2], "y": [1, 1, 1, 2]})
    m = ggplot(df, aes("x", "y")).add(geom_count(), name="c").build()
    marks = m.layers["c"].marks
    assert len(marks) == 2
    assert marks[0].width > marks[1].width  # n = 3 vs n = 1


def test_stat_unique_drops_duplicate_rows():
    df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [1, 1, 2, 3]})
    out = StatUnique().compute(df)
    assert out[["x", "y"]].values.tolist() == [[1, 1], [2, 2], [2, 3]]
    assert isinstance(get_stat("unique"), StatUnique)
    m = ggplot(df, aes("x", "y")).add(stat_unique(), name="u").build()
    assert len(m.layers["u"].marks) == 3
    m = ggplot(df, aes("x", "y")).add(geom_point(stat="unique"), name="u").build()
    assert len(m.layers["u"].marks) == 3
