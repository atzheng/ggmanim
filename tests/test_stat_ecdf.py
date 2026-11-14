"""stat_ecdf and the step fallback geom that draws it."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, geom_point, ggplot, stat_ecdf
from ggmanim.geoms.count import GeomStepFallback, step_expand
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
    assert isinstance(layer.geom, GeomStepFallback)
    assert isinstance(layer.stat, StatEcdf) and layer.stat.params["pad"] is False
    assert isinstance(stat_ecdf(geom="point").geom, type(geom_point().geom))


def test_step_expand_directions():
    x, y = np.array([0.0, 1, 2]), np.array([0.0, 1, 2])
    xs, ys = step_expand(x, y, "hv")
    assert xs.tolist() == [0, 1, 1, 2, 2] and ys.tolist() == [0, 0, 1, 1, 2]
    xs, ys = step_expand(x, y, "vh")
    assert xs.tolist() == [0, 0, 1, 1, 2] and ys.tolist() == [0, 1, 1, 2, 2]
    xs, ys = step_expand(x, y, "mid")
    assert xs.tolist() == [0, 0.5, 0.5, 1.5, 1.5, 2] and ys.tolist() == [0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError, match="direction"):
        step_expand(x, y, "diag")


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
