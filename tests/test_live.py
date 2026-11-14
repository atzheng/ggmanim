"""Live plots: ggplot.with_data, GGPlot.update_data, LivePlot, always_rebuild."""

import time

import numpy as np
import pandas as pd
import pytest
from manim import ValueTracker

from ggmanim import aes, coord_cartesian, geom_line, geom_point, ggplot, xlim, ylim
from ggmanim.build import BuildError
from ggmanim.live import LivePlot, always_rebuild, build_live, check_fixed_limits


@pytest.fixture
def df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "id": ["a", "b", "c"]})


def _spec(df):
    return (
        ggplot(df, aes("x", "y", key="id"))
        .add(geom_point(), name="pts")
        .add(geom_line(), name="ln")
        .add(xlim(0, 5), ylim(0, 5))
    )


def test_with_data_is_immutable_copy(df):
    p = ggplot(df, aes("x", "y")).geom_point()
    new = df.assign(y=[3.0, 2.0, 1.0])
    q = p.with_data(new)
    assert q is not p and q.data is new and p.data is df
    assert q.layers == p.layers and q.mapping == p.mapping


def test_update_data_moves_same_dots(df):
    plot = build_live(_spec(df))
    pts = plot.layers["pts"]
    marks = list(pts.marks)
    ids = [id(m) for m in marks]
    a = plot.mark("pts", "a")
    before = a.get_center().copy()
    radius = a.width
    line = plot.layers["ln"].marks[0]
    panel_family = [id(m) for m in plot.panel.get_family()]

    out = plot.update_data(df.assign(y=[4.0, 3.0, 2.0]))

    assert out is plot
    assert plot.layers["pts"] is pts
    assert [id(m) for m in plot.layers["pts"].marks] == ids
    assert plot.mark("pts", "a") is a
    expected = plot.panel.transform(1.0, 4.0)[0]
    assert np.allclose(a.get_center(), expected, atol=1e-6)
    assert not np.allclose(a.get_center(), before)
    assert a.width == pytest.approx(radius)
    assert plot.layers["ln"].marks[0] is line  # path updated in place
    assert [id(m) for m in plot.panel.get_family()] == panel_family  # no text rebuilt
    assert plot.layer_data["pts"]["y"].tolist() == [4.0, 3.0, 2.0]


def test_update_data_keeps_size_when_plot_is_scaled(df):
    plot = build_live(_spec(df))
    plot.scale(0.5)
    a = plot.mark("pts", "a")
    w = a.width
    plot.update_data(df.assign(y=[4.0, 3.0, 2.0]))
    assert a.width == pytest.approx(w)
    assert np.allclose(a.get_center(), plot.panel.transform(1.0, 4.0)[0], atol=1e-6)


def test_update_data_new_rows_replace_layer(df):
    plot = build_live(_spec(df))
    old = plot.layers["pts"]
    bigger = pd.concat([df, pd.DataFrame({"x": [4.0], "y": [4.0], "id": ["d"]})])
    plot.update_data(bigger.reset_index(drop=True))
    new = plot.layers["pts"]
    assert new is not old
    assert new in plot.submobjects and old not in plot.submobjects
    assert list(new.keys) == ["a", "b", "c", "d"]


def test_update_data_layers_argument(df):
    plot = build_live(_spec(df))
    line_pts = plot.layers["ln"].marks[0].points.copy()
    plot.update_data(df.assign(y=[4.0, 3.0, 2.0]), layers="pts")
    assert np.allclose(plot.layers["ln"].marks[0].points, line_pts)
    with pytest.raises(KeyError):
        plot.update_data(df, layers="nope")


def test_update_data_inside_updater(df):
    plot = build_live(_spec(df))
    t = ValueTracker(0.0)
    plot.add_updater(lambda m: m.update_data(df.assign(y=df["y"] + t.get_value())))
    a = plot.mark("pts", "a")
    t.set_value(1.0)
    plot.update(0)
    assert np.allclose(a.get_center(), plot.panel.transform(1.0, 2.0)[0], atol=1e-6)


def test_update_data_requires_fixed_limits(df):
    plot = build_live(ggplot(df, aes("x", "y")).geom_point().add(xlim(0, 5)))
    with pytest.raises(BuildError, match="fixed y"):
        plot.update_data(df)
    with pytest.raises(BuildError, match="fixed"):
        check_fixed_limits(ggplot(df, aes("x", "y")).add(xlim(0, None), ylim(0, 5)))
    # coord_cartesian with both ends is enough
    p = ggplot(df, aes("x", "y")).geom_point().add(coord_cartesian(xlim=(0, 5), ylim=(0, 5)))
    check_fixed_limits(p)
    build_live(p).update_data(df.assign(y=[2.0, 2.0, 2.0]))


def test_update_data_requires_identity_stat(df):
    p = ggplot(df, aes("x", "y")).geom_point().geom_smooth(method="lm").add(xlim(0, 5), ylim(0, 5))
    plot = build_live(p)
    with pytest.raises(BuildError, match="identity"):
        plot.update_data(df)
    plot.update_data(df.assign(y=[2.0, 2.5, 3.0]), layers=0)  # the point layer is fine


def test_update_data_needs_spec(df):
    plot = _spec(df).build()
    with pytest.raises(BuildError, match="build_live"):
        plot.update_data(df)
    plot.update_data(df.assign(y=[4.0, 3.0, 2.0]), spec=_spec(df))
    assert plot.spec is not None


def test_streaming_line_appends_points(df):
    full = pd.DataFrame({"x": np.linspace(0, 4, 20), "y": np.linspace(0, 4, 20)})
    plot = build_live(ggplot(full.iloc[:5], aes("x", "y")).geom_line().add(xlim(0, 5), ylim(0, 5)))
    line = plot.layers[0].marks[0]
    n0 = len(line.points)
    plot.update_data(full.iloc[:15])
    assert plot.layers[0].marks[0] is line
    assert len(line.points) > n0


def test_always_rebuild_value_tracker():
    a = ValueTracker(1.0)
    spec = (
        ggplot()
        .geom_function(fun=lambda x, a=a: a.get_value() * np.sin(x))
        .add(xlim(0, 2 * np.pi), ylim(-3, 3))
    )
    plot = always_rebuild(spec)
    assert isinstance(plot.live, LivePlot)
    curve = plot.layers[0].marks[0]
    h1 = curve.height
    a.set_value(2.0)
    plot.update(0)
    assert plot.layers[0].marks[0] is curve
    assert curve.height == pytest.approx(2 * h1, rel=0.02)


def test_always_rebuild_60_frames_fast():
    a = ValueTracker(0.5)
    spec = (
        ggplot()
        .geom_function(fun=lambda x, a=a: a.get_value() * np.sin(x), n=200)
        .add(xlim(0, 2 * np.pi), ylim(-2, 2))
    )
    plot = always_rebuild(spec)
    start = time.perf_counter()
    for i in range(60):
        a.set_value(0.5 + i / 60)
        plot.update(1 / 15)
    assert time.perf_counter() - start < 10


def test_liveplot_spec_factory(df):
    t = ValueTracker(0.0)
    live = LivePlot(lambda: _spec(df.assign(x=df["x"] + t.get_value())))
    a = live.plot.mark("pts", "a")
    live.plot.add_updater(live.updater())
    t.set_value(1.0)
    live.plot.update(0)
    assert live.plot.mark("pts", "a") is a
    assert np.allclose(a.get_center(), live.plot.panel.transform(2.0, 1.0)[0], atol=1e-6)
    live.update_data(df.assign(y=[0.5, 0.5, 0.5]))
    assert np.allclose(a.get_center(), live.plot.panel.transform(1.0, 0.5)[0], atol=1e-6)


def test_liveplot_rejects_free_limits(df):
    with pytest.raises(BuildError):
        LivePlot(ggplot(df, aes("x", "y")).geom_point())
