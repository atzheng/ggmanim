"""stat_qq / stat_qq_line, the normal quantile function, geom_qq / geom_qq_line."""

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, geom_qq, geom_qq_line, ggplot, stat_qq, stat_qq_line
from ggmanim.geoms.qq import GeomQq, GeomQqLine
from ggmanim.stats import StatQq, StatQqLine, get_stat, ppoints, qnorm


def test_qnorm_matches_stdlib_to_machine_precision():
    p = np.concatenate([np.logspace(-15, -1, 40), np.linspace(0.02, 0.98, 97)])
    p = np.concatenate([p, 1 - p])
    want = np.array([NormalDist().inv_cdf(v) for v in p])
    np.testing.assert_allclose(qnorm(p), want, rtol=1e-12, atol=1e-12)
    assert qnorm(0.5) == 0.0
    assert qnorm(0.975, mean=10, sd=2) == pytest.approx(10 + 2 * 1.959963984540054, rel=1e-14)
    assert qnorm([0.0, 1.0]).tolist() == [-np.inf, np.inf]
    assert np.isnan(qnorm(1.5))


def test_ppoints_matches_r():
    np.testing.assert_allclose(ppoints(4), (np.arange(1, 5) - 3 / 8) / (4 + 1 / 4))
    np.testing.assert_allclose(ppoints(20), (np.arange(1, 21) - 0.5) / 20)


def test_qq_points_lie_near_the_line_for_normal_data():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"sample": rng.normal(5.0, 2.0, 400)})
    pts = StatQq().compute(df)
    line = StatQqLine().compute(df)
    assert (np.diff(pts["y"]) >= 0).all() and (np.diff(pts["x"]) > 0).all()
    np.testing.assert_allclose(pts["theoretical"], qnorm(ppoints(400)))
    slope, intercept = line["slope"].iloc[0], line["intercept"].iloc[0]
    assert slope == pytest.approx(2.0, rel=0.1) and intercept == pytest.approx(5.0, abs=0.2)
    resid = pts["y"] - (intercept + slope * pts["x"])
    assert np.median(np.abs(resid)) < 0.1
    # the line spans the theoretical quantiles
    assert line["x"].tolist() == pytest.approx([pts["x"].min(), pts["x"].max()])


def test_qq_line_through_quartiles_and_fullrange():
    df = pd.DataFrame({"sample": [1.0, 2, 3, 4, 5]})
    line = StatQqLine().compute(df)
    q = np.quantile(df["sample"], [0.25, 0.75])
    z = qnorm([0.25, 0.75])
    assert line["slope"].iloc[0] == pytest.approx((q[1] - q[0]) / (z[1] - z[0]))

    class Scale:
        def get_limits(self):
            return (-4.0, 4.0)

    full = StatQqLine(fullrange=True).compute(df, {"x": Scale()})
    assert full["x"].tolist() == [-4.0, 4.0]
    with pytest.raises(ValueError, match="line_p"):
        StatQqLine(line_p=(0.8, 0.2)).compute(df)


def test_qq_custom_distribution_and_groups():
    def qunif(p, lo=0.0, hi=1.0):
        return lo + (hi - lo) * np.asarray(p)

    df = pd.DataFrame({"sample": [3.0, 1, 2, 10, 30, 20], "color": list("aaabbb")})
    out = StatQq(distribution=qunif, dparams={"hi": 2.0}).compute(df)
    assert out.groupby("color").size().tolist() == [3, 3]
    np.testing.assert_allclose(out["x"].iloc[:3], 2 * ppoints(3))
    assert out["y"].tolist() == [1, 2, 3, 10, 20, 30]
    assert isinstance(get_stat("qq"), StatQq) and isinstance(get_stat("qq_line"), StatQqLine)


def test_layers_and_build():
    assert isinstance(geom_qq().geom, GeomQq) and isinstance(geom_qq_line().geom, GeomQqLine)
    assert isinstance(stat_qq().stat, StatQq) and isinstance(stat_qq_line().stat, StatQqLine)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"v": rng.normal(size=50), "g": ["a"] * 25 + ["b"] * 25})
    m = (
        ggplot(df, aes(sample="v", color="g"))
        .add(geom_qq(), name="pts")
        .add(geom_qq_line(), name="line")
        .build()
    )
    assert len(m.layers["pts"].marks) == 50
    assert len(m.layers["line"].marks) == 2
