"""Range scales: continuous size / alpha / linewidth (ggm-658)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    ggplot,
    scale_alpha,
    scale_linewidth,
    scale_radius,
    scale_size,
    scale_size_area,
)
from ggmanim.geoms.line import GeomLine
from ggmanim.geoms.point import GeomPoint
from ggmanim.guides import build_legends
from ggmanim.scales.discrete import ScaleDiscrete, default_scale_for
from ggmanim.scales.range import ScaleContinuousIdentity, ScaleContinuousRange
from ggmanim.theme import Theme


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 1.0, 4.0, 3.0, 5.0],
            "v": [1.0, 25.0, 50.0, 75.0, 100.0],
            "g": ["a", "a", "b", "b", "b"],
        }
    )


def _trained(scale, values):
    scale.train(values)
    return scale


# -- map() --------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctor, rng",
    [
        (scale_size, (0.03, 0.15)),
        (scale_radius, (0.03, 0.15)),
        (scale_alpha, (0.1, 1.0)),
        (scale_linewidth, (1.0, 6.0)),
    ],
)
def test_map_endpoints_equal_range(ctor, rng):
    s = _trained(ctor(), [3.0, 7.0, 11.0])
    out = s.map([3.0, 11.0])
    np.testing.assert_allclose(out, rng)


def test_custom_range():
    s = _trained(scale_alpha(range=(0.2, 0.6)), [0, 10])
    np.testing.assert_allclose(s.map([0, 5, 10]), [0.2, 0.4, 0.6])


def test_area_vs_radius_semantics():
    area = _trained(scale_size(range=(0.0, 1.0)), [0.0, 100.0])
    radius = _trained(scale_radius(range=(0.0, 1.0)), [0.0, 100.0])
    # area: radius ~ sqrt(value); radius: linear
    np.testing.assert_allclose(area.map([25.0]), [0.5])
    np.testing.assert_allclose(radius.map([25.0]), [0.25])


def test_scale_size_area_zero_is_zero():
    # rescaled by the largest |limit|, not the range: 10 of 40 -> sqrt(1/4) of max
    s = _trained(scale_size_area(max_size=0.2), [10.0, 40.0])
    np.testing.assert_allclose(s.map([10.0, 40.0]), [0.1, 0.2])
    s = _trained(scale_size_area(max_size=0.2), [0.0, 40.0])
    np.testing.assert_allclose(s.map([0.0]), [0.0])


def test_oob_and_nan_map_to_na_value():
    s = _trained(scale_size(limits=(0, 10)), [0, 10])
    out = s.map([5.0, 20.0, np.nan])
    assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isnan(out[2])


def test_zero_width_range_maps_to_middle():
    s = _trained(scale_alpha(), [3.0, 3.0])
    np.testing.assert_allclose(s.map([3.0]), [0.55])


def test_clone_is_untrained():
    s = _trained(scale_size(), [1, 2])
    c = s.clone()
    assert c.get_limits() is None and s.get_limits() == (1.0, 2.0)
    assert c.output_range == s.output_range


def test_log_trans():
    s = _trained(scale_radius(range=(0.0, 1.0), trans="log10"), [1.0, 100.0])
    np.testing.assert_allclose(s.map([10.0]), [0.5])


def test_bad_args():
    with pytest.raises(ValueError):
        ScaleContinuousRange("size", (1, 2, 3))
    with pytest.raises(ValueError):
        ScaleContinuousRange("size", palette="cubic")


# -- default scale selection ----------------------------------------------------


@pytest.mark.parametrize("aes_name", ["size", "alpha", "linewidth"])
def test_default_scale_numeric(aes_name):
    s = default_scale_for(aes_name, pd.Series([1.0, 2.0]))
    assert isinstance(s, ScaleContinuousRange) and s.aesthetics == (aes_name,)


def test_default_scale_ordered_factor_no_warning():
    vals = pd.Series(pd.Categorical(["lo", "hi", "mid"], ["lo", "mid", "hi"], ordered=True))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = default_scale_for("alpha", vals)
    assert isinstance(s, ScaleDiscrete)
    s.train(vals)
    np.testing.assert_allclose(list(s.map(["lo", "mid", "hi"])), [0.1, 0.55, 1.0])


def test_default_scale_unordered_factor_warns():
    with pytest.warns(UserWarning, match="Using linewidth for a discrete variable"):
        s = default_scale_for("linewidth", ["a", "b"])
    s.train(["a", "b"])
    np.testing.assert_allclose(list(s.map(["a", "b"])), [1.0, 6.0])


def test_discrete_size_spaces_areas():
    with pytest.warns(UserWarning):
        s = default_scale_for("size", ["a", "b", "c"])
    s.train(["a", "b", "c"])
    r = np.array(list(s.map(["a", "b", "c"])), dtype=float)
    np.testing.assert_allclose(np.diff(r**2), np.diff(r**2)[0])  # even areas


# -- build ----------------------------------------------------------------------


def test_build_size_within_range_with_legend(df):
    m = ggplot(df, aes("x", "y", size="v")).geom_point().build()
    radii = np.array([d.radius for d in m.layers[0]])
    assert radii.min() == pytest.approx(0.03) and radii.max() == pytest.approx(0.15)
    assert isinstance(m.scales["size"], ScaleContinuousRange)
    assert m.layer_data[0]["size"].dtype.kind == "f"
    (leg,) = m.legends
    assert leg.aesthetics == ("size",)
    assert 4 <= len(leg.keys) <= 5
    # keys drawn at the mapped size, increasing
    key_r = [k[0].radius for k in leg.keys]
    np.testing.assert_allclose(key_r, m.scales["size"].map(leg.breaks))
    assert key_r == sorted(key_r)


@pytest.mark.parametrize(
    "lo, hi", [(1, 100), (37, 1913), (0.2, 0.9), (3, 7), (10, 2000), (-5, 5), (0, 1e6)]
)
def test_legend_key_count(lo, hi):
    s = _trained(scale_size(), [lo, hi])
    (leg,) = build_legends({"size": s}, [(GeomPoint(), {"size": "v"}, {})], Theme())
    assert 3 <= len(leg.keys) <= 5


def test_build_alpha_maps_into_default_range(df):
    m = ggplot(df, aes("x", "y", alpha="v")).geom_point().build()
    op = np.array([d.get_fill_opacity() for d in m.layers[0]])
    assert op.min() == pytest.approx(0.1) and op.max() == pytest.approx(1.0)
    (leg,) = m.legends
    np.testing.assert_allclose(
        [k[0].get_fill_opacity() for k in leg.keys], m.scales["alpha"].map(leg.breaks)
    )


def test_build_linewidth(df):
    m = ggplot(df, aes("x", "y", linewidth="v", group="g")).geom_line().build()
    widths = sorted(p.get_stroke_width() for p in m.layers[0])
    assert 1.0 <= widths[0] <= widths[-1] <= 6.0
    assert len(m.legends) == 1


def test_constant_size_bypasses_scale(df):
    m = ggplot(df, aes("x", "y")).geom_point(size=0.08, alpha=0.5).build()
    assert "size" not in m.scales and "alpha" not in m.scales
    assert all(d.radius == pytest.approx(0.08) for d in m.layers[0])
    assert m.legends == []


def test_user_scale_used(df):
    m = ggplot(df, aes("x", "y", size="v")).geom_point().scale_size(range=(0.1, 0.2)).build()
    radii = [d.radius for d in m.layers[0]]
    assert min(radii) == pytest.approx(0.1) and max(radii) == pytest.approx(0.2)


def test_guide_none_hides_legend(df):
    m = ggplot(df, aes("x", "y", size="v")).geom_point().scale_size(guide="none").build()
    assert m.legends == []
    m = ggplot(df, aes("x", "y", size="v")).geom_point().guides(size="none").build()
    assert m.legends == []


def test_size_and_alpha_same_column_merge(df):
    m = ggplot(df, aes("x", "y", size="v", alpha="v")).geom_point().build()
    (leg,) = m.legends
    assert set(leg.aesthetics) == {"size", "alpha"}
    k = leg.keys[0][0]
    assert k.radius == pytest.approx(m.scales["size"].map([leg.breaks[0]])[0])
    assert k.get_fill_opacity() == pytest.approx(m.scales["alpha"].map([leg.breaks[0]])[0])


def test_size_and_colour_legends_separate(df):
    m = ggplot(df, aes("x", "y", size="v", color="g")).geom_point().build()
    assert len(m.legends) == 2


def test_build_legends_direct_with_line_glyph():
    s = _trained(scale_linewidth(), [0, 10])
    (leg,) = build_legends({"linewidth": s}, [(GeomLine(), {"linewidth": "w"}, {})], Theme())
    widths = [k[0].get_stroke_width() for k in leg.keys]
    np.testing.assert_allclose(widths, s.map(leg.breaks))


def test_identity_scale_has_no_legend():
    s = _trained(ScaleContinuousIdentity("size"), [0.1, 0.2])
    np.testing.assert_allclose(s.map([0.1, 0.5]), [0.1, 0.5])
    assert build_legends({"size": s}, [(GeomPoint(), {"size": "s"}, {})], Theme()) == []


def test_big_keys_do_not_overlap():
    s = _trained(scale_size_area(max_size=0.5), [0, 100])
    (leg,) = build_legends({"size": s}, [(GeomPoint(), {"size": "v"}, {})], Theme())
    tops = [k.get_top()[1] for k in leg.keys]
    bottoms = [k.get_bottom()[1] for k in leg.keys]
    for lower_top, upper_bottom in zip(tops[1:], bottoms[:-1]):
        assert lower_top <= upper_bottom + 1e-9
