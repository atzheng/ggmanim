import copy

import manim
import numpy as np
import pytest
from manim import BLACK, WHITE, ManimColor, Square, Text, tempconfig

from ggmanim.coords import CartesianPanel, CoordCartesian, PanelParams, coord_cartesian
from ggmanim.plot import ggplot
from ggmanim.registry import COMPONENTS
from ggmanim.scales import (
    scale_x_continuous,
    scale_x_log10,
    scale_y_continuous,
    scale_y_reverse,
)
from ggmanim.theme import theme, theme_classic, theme_default, theme_minimal


def _scales(x=(3, 2500), y=(-2, 7), xs=scale_x_log10, ys=scale_y_continuous):
    sx, sy = xs(), ys()
    sx.train(list(x))
    sy.train(list(y))
    return sx, sy


def _build(**kw):
    sx, sy = _scales()
    return coord_cartesian().build(sx, sy, theme_default(), {"x": "Dose", "y": "Response"}, **kw)


def _centres(group):
    return np.array([m.get_center() for m in group])


def test_registered_and_composes():
    assert "coord_cartesian" in COMPONENTS
    p = ggplot().coord_cartesian(xlim=(0, 1))
    assert isinstance(p.coord, CoordCartesian) and p.coord.xlim == (0, 1)


def test_axes_range_is_expanded_limits():
    sx, sy = _scales()
    panel = coord_cartesian().build(sx, sy, theme_default())
    assert isinstance(panel, CartesianPanel)
    assert isinstance(panel.axes, manim.Axes)
    np.testing.assert_allclose(panel.axes.x_range[:2], sx.get_expanded_limits())
    np.testing.assert_allclose(panel.axes.y_range[:2], sy.get_expanded_limits())
    np.testing.assert_allclose(panel.panel_params.x_range, sx.get_expanded_limits())
    np.testing.assert_allclose(panel.panel_params.y_range, sy.get_expanded_limits())


def test_tick_labels_are_scale_labels():
    sx, sy = _scales()
    panel = coord_cartesian().build(sx, sy, theme_default())
    xr, yr = sx.get_expanded_limits(), sy.get_expanded_limits()
    pp = panel.panel_params
    assert list(pp.x_labels) == sx.get_labels(xr) == ["10", "100", "1000"]
    np.testing.assert_allclose(pp.x_breaks, sx.get_breaks(xr))
    np.testing.assert_allclose(pp.x_breaks, [1, 2, 3])
    assert list(pp.y_labels) == sy.get_labels(yr)
    assert [t.text for t in panel.x_tick_labels] == ["10", "100", "1000"]
    assert [t.text for t in panel.y_tick_labels] == sy.get_labels(yr)
    # positioned under / left of their break
    xc = _centres(panel.x_tick_labels)
    np.testing.assert_allclose(xc[:, 0], panel.transform(pp.x_breaks, pp.y_range[0])[:, 0])
    assert (xc[:, 1] < panel.axes.x_axis.get_center()[1]).all()
    yc = _centres(panel.y_tick_labels)
    np.testing.assert_allclose(yc[:, 1], panel.transform(pp.x_range[0], pp.y_breaks)[:, 1])
    assert (yc[:, 0] < panel.axes.y_axis.get_center()[0]).all()
    assert len(panel.x_ticks) == 3 and len(panel.y_ticks) == len(pp.y_breaks)


def test_no_manim_numbers_or_logbase():
    panel = _build()
    for axis in (panel.axes.x_axis, panel.axes.y_axis):
        assert isinstance(axis.scaling, manim.LinearBase)
        assert len(getattr(axis, "numbers", [])) == 0


def test_transform_matches_per_point_c2p():
    panel = _build()
    rng = np.random.default_rng(0)
    xs = rng.uniform(*panel.panel_params.x_range, 50)
    ys = rng.uniform(*panel.panel_params.y_range, 50)
    pts = panel.transform(xs, ys)
    assert pts.shape == (50, 3)
    expected = np.array([panel.axes.c2p(x, y) for x, y in zip(xs, ys)])
    np.testing.assert_allclose(pts, expected)


def test_transform_shapes():
    panel = _build()
    assert panel.transform([], []).shape == (0, 3)
    assert panel.transform(1.0, 0.0).shape == (1, 3)
    np.testing.assert_allclose(panel.transform(2.0, [0.0, 1.0])[:, 0], panel.axes.c2p(2.0, 0)[0])
    assert np.isnan(panel.transform([np.nan], [0.0])[0, 0])


def test_transform_tracks_moved_panel():
    panel = _build()
    before = panel.transform([2.0], [1.0])
    panel.shift(manim.RIGHT * 2).scale(0.5)
    after = panel.transform([2.0], [1.0])
    np.testing.assert_allclose(after, np.array([panel.axes.c2p(2.0, 1.0)]))
    assert not np.allclose(before, after)


def test_axes_meet_bottom_left():
    panel = _build()
    (x0, _), (y0, _) = panel.panel_params.x_range, panel.panel_params.y_range
    corner = panel.transform(x0, y0)[0]
    np.testing.assert_allclose(panel.axes.x_axis.get_start(), corner, atol=1e-9)
    np.testing.assert_allclose(panel.axes.y_axis.get_start(), corner, atol=1e-9)


def test_titles_placement():
    panel = _build()
    axes = panel.axes
    assert panel.x_title.text == "Dose"
    assert panel.x_title.get_top()[1] < panel.x_tick_labels.get_bottom()[1]
    assert panel.x_title.get_center()[0] == pytest.approx(axes.x_axis.get_center()[0])
    assert panel.y_title.get_right()[0] < panel.y_tick_labels.get_left()[0]
    assert panel.y_title.get_center()[1] == pytest.approx(axes.y_axis.get_center()[1])
    # rotated: taller than wide
    assert panel.y_title.height > panel.y_title.width
    assert panel.x_title in panel.submobjects and panel.y_title in panel.submobjects


def test_scale_name_beats_labels_and_none_suppresses():
    sx, sy = _scales()
    sx = scale_x_log10(name="From scale")
    sx.train([3, 2500])
    panel = coord_cartesian().build(sx, sy, theme_default(), {"x": "labs", "y": None})
    assert panel.x_title.original_text == "From scale"
    assert panel.y_title is None
    assert coord_cartesian().build(sx, sy).y_title is None


def test_caller_mobject_title_not_mutated():
    sx, sy = _scales()
    title = Text("Y")
    t_before = title.get_center().copy(), title.width, title.height
    panel = coord_cartesian().build(sx, sy, theme_default(), {"y": title})
    assert panel.y_title is not title
    np.testing.assert_allclose(title.get_center(), t_before[0])
    assert (title.width, title.height) == pytest.approx(t_before[1:])
    assert panel.y_title.height > panel.y_title.width  # the copy was rotated


def test_build_is_pure_and_deterministic():
    sx, sy = _scales()
    sx_state, sy_state = copy.deepcopy(sx.__dict__), copy.deepcopy(sy.__dict__)
    th = theme_default()
    labels = {"x": "Dose", "y": "Response"}
    a = coord_cartesian().build(sx, sy, th, labels)
    b = coord_cartesian().build(sx, sy, th, labels)
    for ga, gb in [(a.x_tick_labels, b.x_tick_labels), (a.y_tick_labels, b.y_tick_labels),
                   (a.x_ticks, b.x_ticks), (a.grid, b.grid)]:
        np.testing.assert_array_equal(_centres(ga), _centres(gb))
    np.testing.assert_array_equal(a.x_title.get_center(), b.x_title.get_center())
    np.testing.assert_array_equal(a.y_title.get_center(), b.y_title.get_center())
    np.testing.assert_array_equal(a.axes.get_all_points(), b.axes.get_all_points())
    assert a.panel_params == b.panel_params
    assert sx.__dict__.keys() == sx_state.keys()
    assert sx.range == sx_state["_range"] and sy.range == sy_state["_range"]
    assert labels == {"x": "Dose", "y": "Response"}


def test_text_engine_text_needs_no_latex(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LaTeX used")

    for mod in (manim, manim.mobject.text.tex_mobject):
        for name in ("Tex", "MathTex", "SingleStringMathTex"):
            monkeypatch.setattr(mod, name, boom)
    monkeypatch.setenv("PATH", "")
    panel = _build()
    assert len(panel.x_tick_labels) == 3


def test_foreground_follows_background():
    sx, sy = _scales()
    for bg, fg in ((WHITE, BLACK), (BLACK, WHITE)):
        with tempconfig({"background_color": bg}):
            panel = coord_cartesian().build(sx, sy, theme_default(), {"x": "x"})
        assert panel.axes.x_axis.get_stroke_color() == ManimColor(fg)
        assert panel.x_ticks[0].get_stroke_color() == ManimColor(fg)
        assert panel.x_tick_labels[0].get_color() == ManimColor(fg)
        assert panel.x_title.get_color() == ManimColor(fg)


def test_theme_variants():
    sx, sy = _scales()
    minimal = coord_cartesian().build(sx, sy, theme_minimal())
    assert len(minimal.x_ticks) == 0 and len(minimal.grid) > 0
    assert minimal.axes.x_axis.get_stroke_opacity() == 0
    classic = coord_cartesian().build(sx, sy, theme_classic())
    assert len(classic.grid) == 0 and len(classic.x_ticks) == 3
    bare = coord_cartesian().build(sx, sy, theme_default().merge(theme(tick_labels=False)))
    assert len(bare.x_tick_labels) == 0
    sized = coord_cartesian().build(sx, sy, theme(panel_width=4, panel_height=3))
    assert sized.axes.x_axis.get_length() == pytest.approx(4)
    assert sized.axes.y_axis.get_length() == pytest.approx(3)


def test_reverse_scale():
    sx, sy = _scales(y=(1, 10), ys=scale_y_reverse)
    panel = coord_cartesian().build(sx, sy)
    labels = [t.text for t in panel.y_tick_labels]
    ys = _centres(panel.y_tick_labels)[:, 1]
    # labels ordered top-to-bottom ascending in data space (large values at the bottom)
    ordered = [lab for _, lab in sorted(zip(-ys, labels))]
    assert ordered == sorted(ordered, key=float)


def test_untrained_and_missing_scales_fall_back():
    panel = coord_cartesian().build(scale_x_continuous(), None)
    assert panel.panel_params.x_range == (0.0, 1.0)
    assert panel.panel_params.y_range == (0.0, 1.0)
    assert panel.panel_params.y_breaks == ()
    sx = scale_x_continuous()
    sx.train([5, 5])  # zero width
    lo, hi = coord_cartesian().build(sx, None).panel_params.x_range
    assert lo < 5 < hi


def test_zoom_does_not_censor():
    sx, sy = _scales(x=(0, 100), y=(0, 10), xs=scale_x_continuous)
    coord = coord_cartesian(xlim=(20, 40))
    panel = coord.build(sx, sy)
    lo, hi = panel.panel_params.x_range
    assert lo < 20 and hi > 40 and hi - lo < 25
    assert all(20 <= b <= 40 for b in panel.panel_params.x_breaks) or all(
        lo <= b <= hi for b in panel.panel_params.x_breaks)
    # scale untouched; data outside the zoom still maps (no NaN)
    assert sx.get_limits() == (0.0, 100.0)
    assert np.isfinite(sx.map([0, 100])).all()
    assert np.isfinite(panel.transform(sx.map([0, 100]), [5, 5])).all()
    exact = coord_cartesian(xlim=(20, 40), expand=False).build(sx, sy)
    assert exact.panel_params.x_range == (20.0, 40.0)
    half = coord_cartesian(xlim=(20, None)).build(sx, sy).panel_params.x_range
    assert half[0] < 20 and half[1] > 100


def test_zoom_log_scale_is_in_data_space():
    sx, sy = _scales()
    rng = coord_cartesian(xlim=(10, 1000), expand=False).build(sx, sy).panel_params.x_range
    np.testing.assert_allclose(rng, (1.0, 3.0))


def test_invalid_lim():
    with pytest.raises(ValueError):
        coord_cartesian(xlim=(1, 2, 3))


def test_panel_params_type():
    assert isinstance(_build().panel_params, PanelParams)


def test_mobject_tick_label_square_unaffected():
    # make_text passes mobjects through; titles given as mobjects are copied
    sq = Square()
    sx, sy = _scales()
    panel = coord_cartesian().build(sx, sy, theme_default(), {"x": sq})
    assert panel.x_title is not sq
    np.testing.assert_allclose(sq.get_center(), [0, 0, 0])
