"""Main axis moved to top/right via ``position=`` honours the *_top/_right theme fields (ggm-ghs)."""

import numpy as np
import pandas as pd

from ggmanim import aes, ggplot, scale_x_continuous, scale_y_continuous, theme


def _plot(**kw):
    df = pd.DataFrame({"x": np.arange(5.0), "y": np.arange(5.0)})
    return (
        ggplot(df, aes("x", "y"))
        .geom_point()
        .add(scale_x_continuous(name="X", position="top"))
        .add(scale_y_continuous(name="Y", position="right"))
        .add(theme(**kw))
    )


def test_top_right_text_and_title_switches_apply_to_main_axis():
    pan = _plot(axis_text_x_top=False, axis_title_y_right=False).build().panel
    assert len(pan.x_tick_labels) == 0 and pan.x_title is not None
    assert len(pan.y_tick_labels) > 0 and pan.y_title is None


def test_top_right_ticks_switch():
    pan = _plot(axis_ticks_x_top=False, axis_ticks_y_right=False).build().panel
    assert len(pan.x_ticks) == 0 and len(pan.y_ticks) == 0
    assert len(pan.x_tick_labels) > 0


def test_inherits_plain_switches_and_bottom_left_unaffected():
    pan = _plot(axis_text_y=False).build().panel
    assert len(pan.y_tick_labels) == 0 and len(pan.x_tick_labels) > 0
    df = pd.DataFrame({"x": np.arange(5.0), "y": np.arange(5.0)})
    p = ggplot(df, aes("x", "y")).geom_point().add(theme(axis_text_x_top=False))
    assert len(p.build().panel.x_tick_labels) > 0


def test_explicit_true_overrides_false_parent():
    pan = _plot(axis_text_x=False, axis_text_x_top=True).build().panel
    assert len(pan.x_tick_labels) > 0
