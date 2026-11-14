"""GGPlot.create / reveal_layer / hide_layer / mark and the animation registry."""

import numpy as np
import pandas as pd
import pytest
from manim import Animation, Create, FadeIn, Scene, Write, tempconfig

import ggmanim.geoms as geoms_pkg
from ggmanim import (
    aes,
    coord_flip,
    facet_wrap,
    geom_col,
    geom_line,
    geom_point,
    geom_raster,
    geom_text,
    ggplot,
    labs,
)
from ggmanim.animate import (
    ANIMATIONS,
    PRESETS,
    AnimationContext,
    GrowFromBaseline,
    animation_for,
    attach_geoms,
    create_plot,
    register_animation,
)
from ggmanim.geoms import Geom, GeomPoint


def _df():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, -1.0, 3.0, 1.0],
            "id": ["a", "b", "c", "d"],
            "g": ["u", "u", "v", "v"],
        }
    )


def _spec():
    return (
        ggplot(_df(), aes("x", "y", key="id"))
        .add(geom_col(), name="bars")
        .add(geom_point(), name="pts")
        .add(geom_line(), name="ln")
        .add(geom_text(aes(label="id")), name="txt")
        .add(labs(title="Title"))
    )


def _built(spec=None):
    spec = spec if spec is not None else _spec()
    return attach_geoms(spec.build(), spec)


def _play(*anims_fn):
    """Play the animations returned by ``fn(scene)`` in a headless scene; return it."""
    with tempconfig({"dry_run": True, "quality": "low_quality", "disable_caching": True}):

        class S(Scene):
            def construct(self):
                for fn in anims_fn:
                    self.play(fn(self))

        scene = S()
        scene.render()
    return scene


def _leaves(anim):
    """Every non-group animation inside ``anim``."""
    subs = getattr(anim, "animations", None)
    if not subs:
        return [anim]
    return [leaf for a in subs for leaf in _leaves(a)]


# -- create -------------------------------------------------------------------------


def test_create_plays_and_leaves_the_plot_in_the_scene():
    plot = _built()
    anim = plot.create()
    assert isinstance(anim, Animation)
    scene = _play(lambda s: anim)
    assert scene.mobjects == [plot]
    # every layer mark is back at full opacity after the reveal
    assert plot.mark("pts", "a").get_fill_opacity() == pytest.approx(1.0)


def test_create_stage_order_and_per_geom_animations():
    plot = _built()
    anim = plot.create()
    stages = anim.animations
    # panel, axes, 4 layers, titles (no legend)
    assert len(stages) == 7
    bars, pts, ln, txt = stages[2:6]
    assert all(isinstance(a, GrowFromBaseline) for a in _leaves(bars))
    assert isinstance(pts, FadeIn)
    assert isinstance(ln, Create)
    assert isinstance(txt, Write)
    assert pts.mobject is plot.layers["pts"]


def test_create_lag_ratio_and_run_time():
    plot = _built()
    seq = plot.create(lag_ratio=1.0)
    together = plot.create(lag_ratio=0.0, run_time=2.0)
    assert seq.run_time == pytest.approx(sum(a.run_time for a in seq.animations))
    assert together.run_time == pytest.approx(2.0)


def test_create_order_subset_and_validation():
    plot = _built()
    anim = plot.create(order=("panel", "axes"))
    assert len(anim.animations) == 2
    scene = _play(lambda s: anim)
    assert plot not in scene.mobjects  # layers were not shown, so the plot is not adopted
    with pytest.raises(ValueError, match="unknown stage"):
        plot.create(order=("panel", "points"))


def test_create_plot_accepts_a_spec():
    anim = create_plot(_spec())
    assert anim.plot.layer_geoms["bars"].__class__.__name__ == "GeomCol"
    _play(lambda s: anim)


def test_animations_override():
    plot = _built()
    anim = plot.create(animations={"pts": "grow"})
    assert type(anim.animations[3]).__name__ == "AnimationGroup"


# -- layers -------------------------------------------------------------------------


def test_reveal_and_hide_layer():
    plot = _built()
    scene = _play(
        lambda s: plot.create(),
        lambda s: plot.hide_layer("ln"),
    )
    family = {id(m) for top in scene.mobjects for m in top.get_family()}
    assert id(plot.layers["ln"]) not in family
    assert id(plot.layers["pts"]) in family
    scene = _play(lambda s: plot.reveal_layer("ln", run_time=0.5))
    assert plot.layers["ln"] in scene.mobjects


def test_reveal_layer_by_position_and_unknown_layer():
    plot = _built()
    assert isinstance(plot.reveal_layer(1), FadeIn)
    with pytest.raises(KeyError, match="no layer"):
        plot.reveal_layer("nope")


def test_reveal_layer_animation_override_and_kwargs():
    plot = _built()
    anim = plot.reveal_layer("pts", animation="create", run_time=3)
    assert isinstance(anim, Create)
    assert anim.run_time == 3


# -- marks --------------------------------------------------------------------------


def test_mark_by_key_is_the_dot_at_its_data_position():
    plot = _built()
    for row in _df().itertuples():
        dot = plot.mark("pts", row.id)
        np.testing.assert_allclose(dot.get_center(), plot.axes.c2p(row.x, row.y), atol=1e-9)


def test_mark_defaults_to_row_index_and_lists_keys_on_miss():
    plot = ggplot(_df(), aes("x", "y")).add(geom_point(), name="pts").build()
    np.testing.assert_allclose(plot.mark("pts", 2).get_center(), plot.axes.c2p(3.0, 3.0))
    with pytest.raises(KeyError, match=r"keys: \[0, 1, 2, 3\]"):
        plot.mark("pts", 99)


def test_mark_in_faceted_plot():
    spec = _spec().add(facet_wrap("g"))
    plot = _built(spec)
    dot = plot.mark("pts", "c")
    assert dot in plot.layers["pts"].panel(1, 2).marks
    _play(lambda s: plot.create(panel_lag=0.5))


def test_mark_on_batched_layer_raises():
    grid = pd.DataFrame(
        {"x": np.repeat([1, 2, 3], 3), "y": np.tile([1, 2, 3], 3), "v": ["p", "q", "r"] * 3}
    )
    spec = ggplot(grid, aes("x", "y", fill="v")).add(geom_raster(), name="r")
    plot = _built(spec)
    with pytest.raises(TypeError, match="batches"):
        plot.mark("r", 0)


# -- bars grow from the baseline ------------------------------------------------------


def _start(anim):
    anim.begin()
    return anim.starting_mobject


def test_bars_grow_from_zero_line():
    plot = _built()
    anims = _leaves(plot.reveal_layer("bars"))
    zero = plot.axes.c2p(0, 0)[1]
    assert len(anims) == 4
    for anim, row in zip(anims, _df().itertuples()):
        bar = anim.mobject
        assert anim.dim == 1
        assert anim.anchor[1] == pytest.approx(zero)
        # the positive bar's bottom / the negative bar's top sits on the zero line
        edge = bar.get_bottom()[1] if row.y > 0 else bar.get_top()[1]
        assert edge == pytest.approx(zero)
        start = _start(anim)
        assert start.height == pytest.approx(0.0, abs=1e-9)
        assert start.get_center()[1] == pytest.approx(zero)
        anim.finish()
        assert bar.height > 0.1


def test_bars_grow_horizontally_under_coord_flip():
    spec = _spec().add(coord_flip())
    plot = _built(spec)
    anims = _leaves(plot.reveal_layer("bars"))
    assert all(a.dim == 0 for a in anims)
    zero = plot.panel.transform(1.0, 0.0)[0][0]
    for anim in anims:
        assert anim.anchor[0] == pytest.approx(zero)
        assert _start(anim).width == pytest.approx(0.0, abs=1e-9)


def test_bars_anchor_clamps_when_zero_is_out_of_range():
    df = pd.DataFrame({"x": [1, 2], "y": [5.0, 7.0]})
    spec = ggplot(df, aes("x", "y")).add(geom_col(), name="b")
    from ggmanim import coord_cartesian

    spec = spec.add(coord_cartesian(ylim=(4, 8)))  # zoom: bars cut at the panel bottom
    plot = _built(spec)
    for anim in _leaves(plot.reveal_layer("b")):
        assert anim.anchor[1] == pytest.approx(anim.mobject.get_bottom()[1])


# -- registry -----------------------------------------------------------------------


def _geom_classes():
    return [
        getattr(geoms_pkg, n)
        for n in geoms_pkg.__all__
        if isinstance(getattr(geoms_pkg, n), type)
        and issubclass(getattr(geoms_pkg, n), Geom)
        and getattr(geoms_pkg, n) is not Geom
    ]


@pytest.mark.parametrize("cls", _geom_classes(), ids=lambda c: c.__name__)
def test_every_exported_geom_has_an_animation(cls):
    assert animation_for(cls) is not None


@pytest.mark.parametrize(
    "name, preset",
    [
        ("GeomPoint", "fade_in"),
        ("GeomCount", "fade_in"),
        ("GeomFunction", "create"),
        ("GeomSmooth", "create"),
        ("GeomStep", "create"),
        ("GeomAbline", "create"),
        ("GeomVline", "create"),
        ("GeomCol", "grow_from_baseline"),
        ("GeomArea", "fade_in"),
        ("GeomTile", "fade_in"),
        ("GeomText", "write"),
        ("GeomLabel", "write"),
    ],
)
def test_default_presets(name, preset):
    cls = getattr(geoms_pkg, name)
    assert animation_for(cls).reveal is PRESETS[preset]


def test_register_animation_for_a_new_geom_and_subclass_inheritance():
    class GeomFancy(GeomPoint):
        pass

    assert animation_for(GeomFancy()).reveal is PRESETS["fade_in"]  # inherited
    calls = []

    def factory(mob, ctx):
        assert isinstance(ctx, AnimationContext)
        calls.append(ctx.layer)
        return FadeIn(mob)

    try:
        register_animation(GeomFancy, factory)
        assert animation_for("GeomFancy").reveal is factory
        plot = ggplot(_df(), aes("x", "y")).add(geom_point(), name="f").build()
        plot.layer_geoms = {"f": GeomFancy()}
        plot.reveal_layer("f")
        assert calls == ["f"]
    finally:
        ANIMATIONS.pop("GeomFancy", None)


def test_register_animation_rejects_unknown_preset():
    with pytest.raises(ValueError, match="unknown animation"):
        register_animation("GeomWhatever", "zoom")


def test_animation_inferred_without_layer_geoms():
    plot = _spec().build()  # no attach_geoms
    assert plot.layer_geoms == {}
    assert isinstance(plot.reveal_layer("pts"), FadeIn)
    assert isinstance(plot.reveal_layer("ln"), Create)
    assert isinstance(plot.reveal_layer("txt"), Write)
    _play(lambda s: plot.create())
