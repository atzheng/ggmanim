"""Position adjustments: stack, fill, dodge, dodge2, jitter, jitterdodge, nudge."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    geom_col,
    geom_jitter,
    geom_point,
    ggplot,
    position_dodge,
    position_dodge2,
    position_fill,
    position_jitter,
    position_jitterdodge,
    position_nudge,
    position_stack,
)
from ggmanim.positions import (
    POSITIONS,
    PositionDodge,
    PositionDodge2,
    PositionFill,
    PositionIdentity,
    PositionJitter,
    PositionJitterdodge,
    PositionNudge,
    PositionStack,
    get_position,
)


@pytest.fixture
def bars():
    return pd.DataFrame(
        {"x": ["a", "a", "b", "b"], "g": ["p", "q", "p", "q"], "y": [1.0, 2.0, 3.0, -1.0]}
    )


def layer(m, i=0):
    """Layer data sorted by (x, group) for comparison with hand-computed values."""
    return m.layer_data[i].sort_values(["x", "group"]).reset_index(drop=True)


def cols(df, *names):
    return {n: df[n].to_numpy(float) for n in names}


# -- registry / resolution ----------------------------------------------------------
@pytest.mark.parametrize(
    "name, cls",
    [
        ("stack", PositionStack),
        ("fill", PositionFill),
        ("dodge", PositionDodge),
        ("dodge2", PositionDodge2),
        ("jitter", PositionJitter),
        ("jitterdodge", PositionJitterdodge),
        ("nudge", PositionNudge),
    ],
)
def test_registered(name, cls):
    assert POSITIONS[name] is cls
    assert isinstance(get_position(name), cls)
    assert isinstance(get_position("position_" + name), cls)


def test_get_position_with_args_and_objects():
    assert get_position("dodge", width=0.8) == position_dodge(width=0.8)
    assert geom_col(position=position_dodge(width=0.8)).position == PositionDodge(width=0.8)
    assert isinstance(geom_col(position="dodge").position, PositionDodge)
    assert get_position(PositionNudge, x=1) == position_nudge(x=1)
    with pytest.raises(ValueError, match="preserve"):
        position_dodge(preserve="both")


def test_default_positions():
    assert isinstance(geom_col().position, PositionStack)
    assert isinstance(geom_point().position, PositionIdentity)
    assert isinstance(geom_col(position="identity").position, PositionIdentity)


# -- stack / fill -------------------------------------------------------------------
def test_stack_mixed_sign(bars):
    d = layer(ggplot(bars, aes("x", "y", fill="g")).geom_col().build())
    c = cols(d, "xmin", "xmax", "ymin", "ymax", "y")
    # first group on top: at "a" q (2) is the bottom slice, p (1) on top of it;
    # at "b" the negative q goes below 0, independently of p
    np.testing.assert_allclose(c["ymin"], [2, 0, 0, -1])
    np.testing.assert_allclose(c["ymax"], [3, 2, 3, 0])
    np.testing.assert_allclose(c["y"], [3, 2, 3, 0])  # y = top of the slice (vjust=1)
    np.testing.assert_allclose(c["xmin"], [0.55, 0.55, 1.55, 1.55])
    np.testing.assert_allclose(c["xmax"], [1.45, 1.45, 2.45, 2.45])


def test_stack_reverse_and_vjust(bars):
    df = bars.assign(y=[1.0, 2.0, 3.0, 1.0])
    d = layer(
        ggplot(df, aes("x", "y", fill="g")).geom_col(position=position_stack(reverse=True)).build()
    )
    np.testing.assert_allclose(d["ymin"], [0, 1, 0, 3])
    np.testing.assert_allclose(d["ymax"], [1, 3, 3, 4])
    d = layer(
        ggplot(df, aes("x", "y", fill="g")).geom_point(position=position_stack(vjust=0.5)).build()
    )
    np.testing.assert_allclose(d["y"], [2.5, 1.0, 2.5, 0.5])  # centred in each slice


def test_stack_scale_retrained(bars):
    m = ggplot(bars.assign(y=[1.0, 2.0, 3.0, 4.0]), aes("x", "y", fill="g")).geom_col().build()
    assert m.scales["y"].get_limits()[1] >= 7


def test_fill_sums_to_one():
    df = pd.DataFrame({"x": [1, 1, 1, 2, 2], "g": list("pqrpq"), "y": [1.0, 3.0, 4.0, 2.0, np.nan]})
    d = layer(ggplot(df, aes("x", "y", fill="g")).geom_col(position="fill").build())
    np.testing.assert_allclose(d["ymax"][:3], [1.0, 0.875, 0.5])
    np.testing.assert_allclose(d["ymin"][:3], [0.875, 0.5, 0.0])
    assert d.groupby("x").ymax.max().tolist() == [1.0, 1.0]  # NaN ignored at x=2


# -- dodge ---------------------------------------------------------------------------
def test_dodge_bars(bars):
    d = layer(ggplot(bars, aes("x", "y", fill="g")).geom_col(position="dodge").build())
    c = cols(d, "x", "xmin", "xmax", "ymin", "ymax")
    np.testing.assert_allclose(c["x"], [0.775, 1.225, 1.775, 2.225])
    np.testing.assert_allclose(c["xmin"], [0.55, 1.0, 1.55, 2.0])
    np.testing.assert_allclose(c["xmax"], [1.0, 1.45, 2.0, 2.45])
    np.testing.assert_allclose(c["ymin"], [0, 0, 0, -1])
    np.testing.assert_allclose(c["ymax"], [1, 2, 3, 0])


def test_dodge_width_and_preserve(bars):
    df = bars.iloc[:3]  # "b" has only group p
    d = layer(ggplot(df, aes("x", "y", fill="g")).geom_col(position="dodge").build())
    np.testing.assert_allclose(d["xmin"], [0.55, 1.0, 1.55])
    np.testing.assert_allclose(d["xmax"], [1.0, 1.45, 2.45])  # alone: full width
    p = position_dodge(preserve="single")
    d = layer(ggplot(df, aes("x", "y", fill="g")).geom_col(position=p).build())
    np.testing.assert_allclose(d["xmin"], [0.55, 1.0, 1.55])
    np.testing.assert_allclose(d["xmax"], [1.0, 1.45, 2.0])  # half width, left slot
    d = layer(
        ggplot(bars, aes("x", "y", fill="g")).geom_col(position=position_dodge(width=0.5)).build()
    )
    np.testing.assert_allclose(d["x"], [0.875, 1.125, 1.875, 2.125])
    np.testing.assert_allclose(d["xmax"] - d["xmin"], 0.45)


def test_dodge_points_needs_width(bars):
    d = layer(
        ggplot(bars, aes("x", "y", color="g"))
        .geom_point(position=position_dodge(width=0.6))
        .build()
    )
    np.testing.assert_allclose(d["x"], [0.85, 1.15, 1.85, 2.15])


def test_dodge2(bars):
    d = layer(ggplot(bars.iloc[:3], aes("x", "y", fill="g")).geom_col(position="dodge2").build())
    # two bars at "a" share 0.9, padded by 10%; the lone bar at "b" keeps 0.9 * 0.9
    np.testing.assert_allclose(d["x"], [0.775, 1.225, 2.0])
    np.testing.assert_allclose(d["xmin"], [0.5725, 1.0225, 1.595])
    np.testing.assert_allclose(d["xmax"], [0.9775, 1.4275, 2.405])


def test_dodge2_variable_widths_and_no_overlap():
    df = pd.DataFrame({"x": [1.0, 1.0, 3.0], "y": [1.0, 2.0, 3.0], "g": list("pqp")})
    d = layer(
        ggplot(df, aes("x", "y", fill="g"))
        .geom_col(width=1.0, position=position_dodge2(padding=0))
        .build()
    )
    np.testing.assert_allclose(d["xmin"], [0.5, 1.0, 2.5])
    np.testing.assert_allclose(d["xmax"], [1.0, 1.5, 3.5])
    # nothing overlaps: unchanged, no padding
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    d = layer(ggplot(df, aes("x", "y")).geom_col(position="dodge2").build())
    np.testing.assert_allclose(d["xmin"], [0.55, 1.55])
    np.testing.assert_allclose(d["xmax"], [1.45, 2.45])


def test_dodge2_preserve_single():
    df = pd.DataFrame({"x": [1.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0], "g": list("pqp")})
    p = position_dodge2(preserve="single", padding=0)
    d = layer(ggplot(df, aes("x", "y", fill="g")).geom_col(position=p).build())
    np.testing.assert_allclose(d["xmax"] - d["xmin"], 0.45)
    np.testing.assert_allclose(d["x"], [0.775, 1.225, 2.0])  # lone bar centred


# -- jitter ---------------------------------------------------------------------------
@pytest.fixture
def pts():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"x": np.repeat(["a", "b", "c"], 20), "y": rng.normal(size=60), "g": ["p", "q"] * 30}
    )


def test_jitter_reproducible_and_bounded(pts):
    def run(**kw):
        return (
            ggplot(pts, aes("x", "y"))
            .geom_point(position=position_jitter(**kw))
            .build()
            .layer_data[0]
        )

    a, b = run(seed=1), run(seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["x"], run(seed=2)["x"])
    base = np.repeat([1.0, 2.0, 3.0], 20)
    dx = a["x"].to_numpy(float) - base
    assert np.all(np.abs(dx) <= 0.4) and dx.std() > 0.1  # discrete x: resolution 1
    dy = a["y"].to_numpy(float) - pts["y"].to_numpy()
    assert np.abs(dy).max() > 0  # height defaults to 0.4 * resolution(y)
    # seed=None: deterministic per build (build_plot is pure)
    pd.testing.assert_frame_equal(run(), run())
    # explicit amounts
    c = run(width=0.1, height=0, seed=3)
    assert np.abs(c["x"].to_numpy(float) - base).max() <= 0.1
    np.testing.assert_allclose(c["y"], pts["y"])


def test_jitter_continuous_resolution():
    df = pd.DataFrame({"x": [0.0, 0.5, 1.0] * 10, "y": [1.0] * 30})
    d = ggplot(df, aes("x", "y")).geom_point(position=position_jitter(height=0, seed=1)).build()
    dx = d.layer_data[0]["x"].to_numpy(float) - df["x"].to_numpy()
    assert np.abs(dx).max() <= 0.2 + 1e-12  # 0.4 * 0.5


def test_geom_jitter_alias(pts):
    lyr = geom_jitter(width=0.1, seed=5)
    assert isinstance(lyr.position, PositionJitter)
    assert lyr.position.params == {"width": 0.1, "height": None, "seed": 5}
    assert isinstance(geom_jitter().position, PositionJitter)
    a = ggplot(pts, aes("x", "y")).geom_jitter(width=0.1, seed=5).build().layer_data[0]
    b = ggplot(pts, aes("x", "y")).geom_point(position=position_jitter(width=0.1, seed=5)).build()
    pd.testing.assert_frame_equal(a, b.layer_data[0])
    with pytest.raises(ValueError):
        geom_jitter(position="identity", width=0.1)


def test_jitterdodge(pts):
    p = position_jitterdodge(seed=1)
    d = ggplot(pts, aes("x", "y", fill="g")).geom_point(position=p).build().layer_data[0]
    base = np.repeat([1.0, 2.0, 3.0], 20) + np.where(pts["g"] == "p", -0.1875, 0.1875)
    dx = d["x"].to_numpy(float) - base
    assert np.all(np.abs(dx) <= 0.1 + 1e-12) and np.abs(dx).max() > 0.05  # 0.4 / (2 + 2)
    np.testing.assert_allclose(d["y"], pts["y"])  # jitter_height=0
    with pytest.raises(Exception, match="aesthetic to dodge by"):
        ggplot(pts, aes("x", "y")).geom_point(position=p).build()


# -- nudge ---------------------------------------------------------------------------
def test_nudge(bars):
    d = layer(ggplot(bars, aes("x", "y")).geom_point(position=position_nudge(x=0.25, y=-1)).build())
    np.testing.assert_allclose(d["x"], [1.25, 1.25, 2.25, 2.25])
    np.testing.assert_allclose(sorted(d["y"]), sorted(bars["y"] - 1))
    d = layer(ggplot(bars, aes("x", "y")).geom_col(position=position_nudge(x=1)).build())
    np.testing.assert_allclose(d["xmin"], [1.55, 1.55, 2.55, 2.55])


def test_positions_do_not_mutate_input(bars):
    data = pd.DataFrame({"x": [1.0, 1.0], "y": [1.0, 2.0], "group": [1, 2], "PANEL": [1, 1]})
    data = data.assign(xmin=0.55, xmax=1.45, ymin=0.0, ymax=data.y)
    before = data.copy()
    for pos in (
        position_stack(),
        position_fill(),
        position_dodge(),
        position_dodge2(),
        position_jitter(seed=1),
        position_nudge(x=1),
    ):
        params = pos.setup_params(data)
        pos.compute_layer(pos.setup_data(data, params), params, {})
        pd.testing.assert_frame_equal(data, before)
