"""stat_ecdf and the step geom that draws it."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, geom_point, ggplot, stat_ecdf
from ggmanim.geoms.count import GeomStepEdges
from ggmanim.geoms.step import GeomStep
from ggmanim.layer import Layer
from ggmanim.stats import StatEcdf, get_stat


def test_ecdf_steps_from_0_to_1():
    df = pd.DataFrame({"x": [3.0, 1, 2, 2, 5]})
    out = StatEcdf().compute(df)
    assert out["x"].tolist() == [-np.inf, 1, 2, 3, 5, np.inf]
    np.testing.assert_allclose(out["y"], [0, 0.2, 0.6, 0.8, 1.0, 1.0])
    assert (out["ecdf"] == out["y"]).all()
    assert (np.diff(out["y"]) >= 0).all()


def test_ecdf_no_pad_grid_n_and_weights():
    df = pd.DataFrame({"x": [0.0, 1, 2, 3, 4]})
    out = StatEcdf(pad=False).compute(df)
    assert out["y"].iloc[0] == pytest.approx(0.2) and out["y"].iloc[-1] == 1
    out = StatEcdf(n=3, pad=False).compute(df)
    np.testing.assert_allclose(out["x"], [0, 2, 4])
    np.testing.assert_allclose(out["y"], [0.2, 0.6, 1.0])
    w = StatEcdf(pad=False).compute(df.assign(weight=[1.0, 0, 0, 0, 3]))
    np.testing.assert_allclose(w["y"], [0.25, 0.25, 0.25, 0.25, 1.0])
    assert "weight" not in w.columns


def test_ecdf_per_group_and_registered():
    df = pd.DataFrame({"x": [1.0, 2, 10, 20, 30], "color": list("aabbb")})
    out = StatEcdf().compute(df)
    assert out.groupby("color")["y"].max().tolist() == [1.0, 1.0]
    assert len(out) == (2 + 2) + (3 + 2)
    assert isinstance(get_stat("ecdf"), StatEcdf)


def test_stat_ecdf_returns_a_step_layer():
    layer = stat_ecdf(pad=False, color="red")
    assert isinstance(layer, Layer)
    assert isinstance(layer.geom, GeomStepEdges) and isinstance(layer.geom, GeomStep)
    assert isinstance(layer.stat, StatEcdf) and layer.stat.params["pad"] is False
    assert isinstance(stat_ecdf(geom="point").geom, type(geom_point().geom))


def test_ecdf_plot_reaches_panel_edges():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"v": rng.normal(size=30), "g": ["a"] * 15 + ["b"] * 15})
    m = ggplot(df, aes("v", color="g")).add(stat_ecdf(), name="ecdf").build()
    layer = m.layers["ecdf"]
    assert len(layer.marks) == 2  # one staircase per group
    pp = m.panel.panel_params
    left = m.panel.transform([pp.x_range[0]], [0.0])[0]
    right = m.panel.transform([pp.x_range[1]], [1.0])[0]
    for path in layer.marks:
        pts = np.asarray(path.points)
        np.testing.assert_allclose(pts[0][:2], left[:2], atol=1e-6)
        np.testing.assert_allclose(pts[-1][:2], right[:2], atol=1e-6)
        # horizontal-then-vertical staircase: every segment is axis-aligned
        d = np.diff(pts[:, :2], axis=0)
        assert np.all((np.abs(d[:, 0]) < 1e-9) | (np.abs(d[:, 1]) < 1e-9))


def test_ecdf_step_direction_and_geom_step_semantics():
    df = pd.DataFrame({"v": [0.0, 1.0, 2.0]})
    for direction in ("hv", "vh", "mid"):
        m = ggplot(df, aes("v")).add(stat_ecdf(pad=False, direction=direction)).build()
        pts = np.asarray(m.layers[0].marks[0].points)[:, :2]
        d = np.diff(pts, axis=0)
        assert np.all((np.abs(d[:, 0]) < 1e-9) | (np.abs(d[:, 1]) < 1e-9))
        start = m.panel.transform([0.0], [1 / 3])[0][:2]
        if direction == "hv":
            np.testing.assert_allclose(pts[0], start, atol=1e-6)
            # first move is horizontal (to x = 1)
            assert abs(pts[1][1] - start[1]) < 1e-9


def test_ecdf_nan_rows_dropped_inf_kept():
    data = pd.DataFrame(
        {"x": [-np.inf, 1.0, np.nan, np.inf], "y": [0.0, 0.5, 0.7, 1.0], "group": [1] * 4}
    )
    with pytest.warns(UserWarning, match="removed 1 row"):
        kept = GeomStepEdges().handle_na(data)
    assert kept["x"].tolist() == [-np.inf, 1.0, np.inf]
