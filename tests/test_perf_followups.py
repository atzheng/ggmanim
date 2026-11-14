"""ggm-1gr: the vectorised / cached fast paths give the same results as the
straightforward implementations they replaced (kept here as references)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from manim import ORIGIN, ManimColor

from ggmanim.geoms import _markers
from ggmanim.geoms._markers import make_marker
from ggmanim.palettes import gradient_n_pal, oklab_to_rgb, rgb_to_oklab
from ggmanim.positions.stack import PositionFill, PositionStack

# --------------------------------------------------------------------- stack


def _ref_pos_stack(df: pd.DataFrame, vjust: float, fill: bool) -> pd.DataFrame:
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    y = np.where(np.isnan(y), 0.0, y)
    heights = np.concatenate([[0.0], np.cumsum(y)])
    if fill:
        total = abs(heights[-1])
        heights = heights / total if total > 0 else np.full_like(heights, np.nan)
    if "ymin" in df.columns and "ymax" in df.columns:
        top = pd.to_numeric(df["ymax"], errors="coerce").to_numpy(float)
        bottom = pd.to_numeric(df["ymin"], errors="coerce").to_numpy(float)
        max_is_lower = top < bottom
    else:
        max_is_lower = np.zeros(len(df), dtype=bool)
    lo = np.minimum(heights[:-1], heights[1:])
    hi = np.maximum(heights[:-1], heights[1:])
    out = df.copy()
    out["y"] = (1.0 - vjust) * lo + vjust * hi
    out["ymin"] = np.where(max_is_lower, hi, lo)
    out["ymax"] = np.where(max_is_lower, lo, hi)
    return out


def _ref_collide(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    xcol = "xmin" if "xmin" in df.columns else "x"
    key = pd.to_numeric(df[xcol], errors="coerce")
    group = pd.to_numeric(df["group"], errors="coerce") if "group" in df.columns else 0
    order = pd.DataFrame({"k": key, "g": group if params["reverse"] else -group})
    df = df.loc[order.sort_values(["k", "g"], kind="stable").index]
    out = [
        _ref_pos_stack(part, params["vjust"], params["fill"])
        for _, part in df.groupby(key.loc[df.index], sort=False, dropna=False)
    ]
    return pd.concat(out)


def _ref_compute_panel(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """The groupby implementation of ``PositionStack.compute_panel`` before ggm-1gr."""
    if params["var"] is None or "y" not in data.columns:
        return data
    index = data.index
    data = data.reset_index(drop=True)
    ymax = pd.to_numeric(data["ymax"], errors="coerce").to_numpy(float)
    negative = np.nan_to_num(ymax, nan=0.0) < 0
    pieces = [_ref_collide(data[mask], params) for mask in (negative, ~negative) if mask.any()]
    out = pd.concat(pieces).sort_index()
    out.index = index
    return out


def _stack_data(seed: int, n: int, *, ymin: bool, xmin: bool, group: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 6, n).astype(float)
    x[rng.random(n) < 0.1] = np.nan
    y = rng.normal(1.0, 3.0, n).round(2)
    y[rng.random(n) < 0.1] = np.nan
    y[rng.random(n) < 0.1] = 0.0
    df = pd.DataFrame({"x": x, "y": y, "PANEL": 1, "colour": rng.choice(list("abc"), n)})
    if group:
        g = rng.integers(1, 5, n).astype(float)
        g[rng.random(n) < 0.05] = np.nan
        df["group"] = g
    if xmin:
        df["xmin"] = df["x"] - 0.45
        df["xmax"] = df["x"] + 0.45
    if ymin:
        df["ymin"] = np.where(rng.random(n) < 0.2, df["y"], 0.0)
        df["ymax"] = np.where(df["ymin"] == 0.0, df["y"], 0.0)
    return df.set_index(pd.Index(rng.permutation(n) + 100))


@pytest.mark.parametrize(
    "cls,vjust,reverse,ymin,xmin,group",
    [
        (cls, vjust, reverse, ymin, xmin, group)
        for cls, vjust, reverse, (ymin, xmin, group) in itertools.product(
            (PositionStack, PositionFill),
            (1.0, 0.5),
            (False, True),
            ((False, False, True), (True, True, True), (False, False, False), (True, False, True)),
        )
    ],
)
def test_stack_matches_reference(cls, vjust, reverse, ymin, xmin, group):
    for seed in range(4):
        data = _stack_data(seed, 60, ymin=ymin, xmin=xmin, group=group)
        pos = cls(vjust=vjust, reverse=reverse)
        params = pos.setup_params(data)
        prepared = pos.setup_data(data, params)
        got = pos.compute_panel(prepared.copy(), params)
        want = _ref_compute_panel(prepared.copy(), params)
        pd.testing.assert_frame_equal(got, want, check_exact=True)


@pytest.mark.parametrize("cls", [PositionStack, PositionFill])
def test_stack_tall_stacks_match_reference(cls):
    """Few x positions with many rows each (the per-stack cumsum branch)."""
    rng = np.random.default_rng(7)
    n = 300
    data = pd.DataFrame(
        {
            "x": rng.integers(0, 2, n).astype(float),
            "y": rng.normal(0.5, 2.0, n),
            "group": rng.integers(1, 40, n),
            "PANEL": 1,
        }
    )
    pos = cls(vjust=0.5)
    params = pos.setup_params(data)
    prepared = pos.setup_data(data, params)
    got = pos.compute_panel(prepared.copy(), params)
    want = _ref_compute_panel(prepared.copy(), params)
    pd.testing.assert_frame_equal(got, want, check_exact=True)


def test_stack_all_zero_fill_is_nan():
    data = pd.DataFrame({"x": [1.0, 1.0], "y": [0.0, 0.0], "group": [1, 2], "PANEL": 1})
    pos = PositionFill()
    params = pos.setup_params(data)
    got = pos.compute_panel(pos.setup_data(data, params), params)
    assert got[["y", "ymin", "ymax"]].isna().all().all()


# --------------------------------------------------------------------- palettes


def _ref_gradient(colours, t):
    lab = rgb_to_oklab(np.array([ManimColor(c).to_rgb() for c in colours], dtype=float))
    pos = np.linspace(0.0, 1.0, len(colours))
    x = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    nan = ~np.isfinite(x)
    xc = np.where(nan, 0.0, x)
    interp = np.stack([np.interp(xc, pos, lab[:, k]) for k in range(3)], axis=-1)
    rgb = np.rint(oklab_to_rgb(interp) * 255).astype(int)
    return [None if bad else "#{:02X}{:02X}{:02X}".format(*c) for c, bad in zip(rgb, nan)]


def test_gradient_pal_matches_reference():
    colours = ["#132B43", "#F0E442", "#56B1F7"]
    t = np.random.default_rng(0).uniform(-0.2, 1.2, 3000)
    t[::17] = np.nan
    t[::31] = np.inf
    got = gradient_n_pal(colours)(t)
    want = _ref_gradient(colours, t)
    assert [None if c is None else c.to_hex() for c in got] == want
    assert all(c is None or isinstance(c, ManimColor) for c in got)


def test_gradient_pal_shares_equal_colours():
    pal = gradient_n_pal(["#000000", "#FFFFFF"])
    a, b, c = pal([0.25, 0.25, 0.75])
    assert a is b and a is not c
    assert pal([]) == [] and pal([np.nan]) == [None]


# --------------------------------------------------------------------- markers


def _ref_marker(shape, point, radius):
    """The geometry of ``make_marker`` before ggm-1gr: deep copy, then manim's
    ``scale`` and ``shift``."""
    from ggmanim.scales.shape import parse_shape

    base, _ = parse_shape(shape)
    radius = max(float(radius), 0.0)
    m = _markers._prototype(base).copy()
    m.scale(radius if radius > 0 else 1e-6, about_point=ORIGIN)
    m.shift(np.asarray(point, dtype=float))
    return m


SHAPES = [
    "circle",
    "square",
    "triangle",
    "diamond",
    "star",
    "cross",
    "plus",
    "circle_open",
    "square_filled",
    "triangle_open",
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("radius", [0.05, 0.0])
def test_marker_matches_reference(shape, radius):
    point = (0.3, -1.7, 0.0)
    got = make_marker(shape, point, radius, "#FF0000", fill="#00FF00", alpha=0.6)
    ref = _ref_marker(shape, point, radius)
    got_f, ref_f = got.family_members_with_points(), ref.family_members_with_points()
    assert [type(m) for m in got.get_family()] == [type(m) for m in ref.get_family()]
    assert len(got_f) == len(ref_f)
    for a, b in zip(got_f, ref_f):
        np.testing.assert_array_equal(a.points, b.points)


def test_marker_style_and_independence():
    proto_points = _markers._prototype("circle").points.copy()
    a = make_marker("circle_filled", (1, 2, 0), 0.1, "#FF0000", fill="#0000FF", alpha=0.5)
    b = make_marker("circle_filled", (1, 2, 0), 0.1, "#FF0000", fill="#0000FF", alpha=0.5)
    assert a.radius == pytest.approx(0.1) and a.marker_shape == "circle"
    assert a.get_fill_color().to_hex() == "#0000FF"
    assert a.get_stroke_color().to_hex() == "#FF0000"
    assert a.get_fill_opacity() == pytest.approx(0.5)
    a.shift((5, 0, 0))
    a.set_fill("#FFFFFF")
    np.testing.assert_array_equal(_markers._prototype("circle").points, proto_points)
    assert b.get_center()[0] == pytest.approx(1.0)
    assert b.get_fill_color().to_hex() == "#0000FF"
    cross = make_marker("cross", (0, 0, 0), 0.1, "#FF0000")
    cross.submobjects[0].shift((1, 0, 0))
    assert _markers._prototype("cross").submobjects[0].get_center()[0] == pytest.approx(0.0)
