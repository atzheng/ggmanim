"""geom_step: stairstep expansion (vertex counts, positions, directions)."""

import numpy as np
import pandas as pd
import pytest
from manim import VMobject

import ggmanim
from ggmanim import aes, geom_step, ggplot
from ggmanim.geoms import GeomPath, GeomStep
from ggmanim.geoms.step import stairstep
from ggmanim.registry import COMPONENTS

DF = pd.DataFrame({"x": [3.0, 0.0, 1.0, 2.0], "y": [4.0, 1.0, 3.0, 2.0]})  # unsorted


def _verts(m: VMobject) -> np.ndarray:
    return np.vstack([m.get_start_anchors(), m.get_end_anchors()[-1:]])


def test_registered_and_exported():
    assert "geom_step" in COMPONENTS and "geom_step" in ggmanim.__all__
    layer = geom_step()
    assert isinstance(layer.geom, GeomStep) and isinstance(layer.geom, GeomPath)
    with pytest.raises(ValueError, match="direction"):
        geom_step(direction="diag")


@pytest.mark.parametrize("direction, n_vert", [("hv", 7), ("vh", 7), ("mid", 8)])
def test_vertex_count(direction, n_vert):
    m = ggplot(DF, aes("x", "y")).geom_step(direction=direction).build()
    (path,) = m.layers[0].marks
    assert len(_verts(path)) == n_vert  # 2n - 1 (hv/vh), 2n (mid)


def test_hv_positions():
    m = ggplot(DF, aes("x", "y")).geom_step().build()
    xs = [0, 1, 1, 2, 2, 3, 3]
    ys = [1, 1, 3, 3, 2, 2, 4]
    np.testing.assert_allclose(
        _verts(m.layers[0].marks[0]), m.panel.transform(np.array(xs, float), np.array(ys, float))
    )


def test_stairstep_vh_and_mid():
    vh = stairstep(DF, "vh")
    np.testing.assert_allclose(vh["x"], [0, 0, 1, 1, 2, 2, 3])
    np.testing.assert_allclose(vh["y"], [1, 3, 3, 2, 2, 4, 4])
    mid = stairstep(DF, "mid")
    np.testing.assert_allclose(mid["x"], [0, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3])
    np.testing.assert_allclose(mid["y"], [1, 1, 3, 3, 2, 2, 4, 4])


def test_attributes_follow_source_rows():
    df = DF.assign(c=["d", "a", "b", "c"])
    out = stairstep(df, "hv")
    assert list(out["c"]) == ["a", "b", "b", "c", "c", "d", "d"]


def test_groups_one_mark_each():
    df = pd.concat([DF.assign(g="a"), DF.assign(y=DF.y + 5, g="b")])
    m = ggplot(df, aes("x", "y", color="g")).geom_step().build()
    assert len(m.layers[0].marks) == 2
    assert all(len(_verts(p)) == 7 for p in m.layers[0].marks)


def test_single_point_and_clipping():
    m = ggplot(DF.iloc[:1], aes("x", "y")).geom_step().build()
    assert len(m.layers[0].marks) == 1
    m = (
        ggplot(DF, aes("x", "y"))
        .geom_step()
        .add(ggmanim.coord_cartesian(xlim=(0.5, 2.5), expand=False))
        .build()
    )
    pts = np.vstack([_verts(s) for s in m.layers[0].marks[0].get_family() if len(s.points)])
    lo = m.panel.transform(0.5, 0)[0][0]
    hi = m.panel.transform(2.5, 0)[0][0]
    assert pts[:, 0].min() >= lo - 1e-6 and pts[:, 0].max() <= hi + 1e-6
