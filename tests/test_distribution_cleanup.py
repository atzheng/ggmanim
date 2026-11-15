"""Distribution cleanup (ggm-1ii): bw rules live in stats/_kde, positions guess
orientation from a discrete y scale, ``position_dodge(orientation=)`` and the
``stroke_style`` rename."""

import warnings

import numpy as np
import pandas as pd
import pytest

import ggmanim.geoms as geoms
from ggmanim import position_dodge, position_dodge2
from ggmanim.stats import StatBin, StatCount, _kde, bw_bcv, bw_sj, bw_ucv, resolve_bw
from ggmanim.stats.boxplot import flip_names

X = np.random.default_rng(1).normal(size=80)


def test_bw_rules_registered_in_kde():
    for name in ("ucv", "bcv", "sj", "sj-ste", "sj-dpi"):
        assert name in _kde._PRECOMPUTE_BW_RULES
        with pytest.raises(ValueError, match="unknown bandwidth"):
            _kde.bandwidth(X, name)
    assert _kde.precompute_bw(X, "ucv") == bw_ucv(X)
    assert _kde.precompute_bw(X, "bcv") == bw_bcv(X)
    assert _kde.precompute_bw(X, "SJ") == bw_sj(X)
    assert _kde.precompute_bw(X, "sj-dpi") == bw_sj(X, method="dpi")
    assert resolve_bw(X, "SJ") == bw_sj(X)
    assert resolve_bw(X, "nrd0") == "nrd0"


def test_oriented_position_flag_removed_from_bin():
    assert not hasattr(StatBin, "oriented_position")
    assert not hasattr(StatCount, "oriented_position")


def _bars():
    # horizontal bars: y is the discrete position, x the continuous value
    return pd.DataFrame(
        {
            "xmin": [0.0] * 4,
            "xmax": [3.0, 2.0, 1.0, 4.0],
            "y": [1.0, 1.0, 2.0, 2.0],
            "ymin": [0.55, 0.55, 1.55, 1.55],
            "ymax": [1.45, 1.45, 2.45, 2.45],
            "group": [1, 2, 1, 2],
            "PANEL": 1,
        }
    )


def _scales(ydisc, xdisc=False):
    def sc(d):
        return type("S", (), {"is_discrete": d})()

    return {"x": sc(xdisc), "y": sc(ydisc)}


@pytest.mark.parametrize("pos", [position_dodge, position_dodge2])
def test_guess_orientation_from_discrete_y(pos):
    data = _bars()
    p = pos()
    params = p.setup_params(data)
    out = p.compute_layer(p.setup_data(data, params), params, _scales(ydisc=True))
    # dodged along y, x untouched
    assert (out["xmax"].to_numpy() == data["xmax"].to_numpy()).all()
    assert out["ymin"].nunique() > 2
    assert (out["ymax"] - out["ymin"]).max() < 0.9
    # same as the explicit flipped_aes result
    flagged = data.assign(flipped_aes=True)
    params = p.setup_params(flagged)
    ref = p.compute_layer(p.setup_data(flagged, params), params, _scales(ydisc=True))
    for c in ("ymin", "ymax", "xmin", "xmax"):
        np.testing.assert_allclose(out[c].to_numpy(), ref[c].to_numpy())


def test_no_guess_when_y_continuous():
    data = _bars()
    p = position_dodge()
    params = p.setup_params(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = p.compute_layer(p.setup_data(data, params), params, _scales(ydisc=False))
    assert (out["ymin"].to_numpy() == data["ymin"].to_numpy()).all()


def test_orientation_param():
    data = _bars()
    ref = flip_names(data)  # vertical version
    p = position_dodge(orientation="y")
    assert p.params["orientation"] == "y"
    params = p.setup_params(data)
    assert params["flipped_aes"] and params["orientation_explicit"]
    # explicit "x" disables the scale guess
    p = position_dodge(orientation="x")
    params = p.setup_params(ref)
    out = p.compute_layer(p.setup_data(ref, params), params, _scales(ydisc=True))
    # dodged along x (not guessed horizontal): y extents untouched
    assert (out["ymax"].to_numpy() == ref["ymax"].to_numpy()).all()
    assert (out["xmax"] - out["xmin"]).max() < 0.9
    with pytest.raises(ValueError):
        position_dodge2(orientation="z").setup_params(data)


def test_stroke_style_rename_keeps_deprecated_alias():
    assert "stroke_style" in geoms.__all__
    assert geoms.style is not geoms.stroke_style
    from manim import Line

    row = {"color": "red", "alpha": 1.0}
    with pytest.warns(DeprecationWarning, match="stroke_style"):
        mob = geoms.style(Line(), row, "white", 2.0)
    assert mob is not None
