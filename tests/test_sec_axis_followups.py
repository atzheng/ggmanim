"""sec_axis follow-ups (ggm-sa3): top/right theme switches, date/datetime scales."""

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    dup_axis,
    facet_wrap,
    ggplot,
    label_date,
    scale_x_continuous,
    scale_x_date,
    scale_x_datetime,
    scale_y_continuous,
    scale_y_date,
    sec_axis,
    theme,
)
from ggmanim.scales.date import ScaleDatePosition, to_epoch
from ggmanim.theme import Theme


def _hourly(tz=None):
    t = pd.date_range("2024-03-01", periods=48, freq="h", tz=tz)
    return pd.DataFrame({"t": t, "v": np.sin(np.arange(48) / 5.0)})


def _celsius(**kw):
    df = pd.DataFrame({"t": np.arange(10.0), "c": np.linspace(-5, 35, 10)})
    return (
        ggplot(df, aes("t", "c"))
        .geom_line()
        .add(scale_y_continuous(name="C", sec_axis=sec_axis(lambda c: c * 1.8 + 32, name="F")))
        .add(scale_x_continuous(name="t", sec_axis=dup_axis()))
        .add(theme(**kw))
    )


# ---------------------------------------------------------------- theme fields
def test_theme_fields_default_none_and_accepted_by_theme():
    t = Theme()
    for name in ("axis_text_x_top", "axis_text_y_right", "axis_title_x_top", "axis_title_y_right"):
        assert getattr(t, name) is None
        assert getattr(theme(**{name: False}), name) is False


def test_secondary_text_inherits_primary_switches():
    pan = _celsius(axis_text_y=False, axis_title_x=False).build().panel
    assert len(pan.y_sec_tick_labels) == 0 and pan.y_sec_title is not None
    assert len(pan.x_sec_tick_labels) > 0 and pan.x_sec_title is None
    assert len(pan.y_sec_ticks) > 0  # ticks are not text: unaffected


def test_top_right_switches_override_secondary_only():
    pan = _celsius(axis_text_y_right=False, axis_title_x_top=False).build().panel
    assert len(pan.y_sec_tick_labels) == 0 and len(pan.y_tick_labels) > 0
    assert pan.x_sec_title is None and pan.x_title is not None
    assert pan.y_sec_title is not None and len(pan.x_sec_tick_labels) > 0
    # and they can force text back on where the per-axis switch blanks it
    pan = _celsius(axis_text_y=False, axis_text_y_right=True).build().panel
    assert len(pan.y_tick_labels) == 0 and len(pan.y_sec_tick_labels) > 0
    pan = _celsius(axis_title=False, axis_title_y_right=True).build().panel
    assert pan.y_title is None and pan.y_sec_title is not None and pan.x_sec_title is None


def test_bottom_left_secondary_uses_per_axis_switch():
    # primary on top/right -> secondary on bottom/left: top/right fields don't apply
    df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    p = (
        ggplot(df, aes("x", "y"))
        .geom_point()
        .add(scale_y_continuous(position="right", sec_axis=dup_axis()))
        .add(theme(axis_text_y_right=False))
    )
    pan = p.build().panel
    assert pan.y_sec_side == "left" and len(pan.y_sec_tick_labels) > 0


def test_facet_secondary_title_switch():
    df = pd.DataFrame({"x": np.arange(6.0), "y": np.arange(6.0), "g": list("aabbcc")})
    base = (
        ggplot(df, aes("x", "y"))
        .geom_point()
        .add(scale_y_continuous(sec_axis=dup_axis(name="dup")))
        .add(facet_wrap("g"))
    )
    assert base.build().panel_grid.y_sec_title is not None
    off = base.add(theme(axis_title_y_right=False)).build()
    assert off.panel_grid.y_sec_title is None


# ---------------------------------------------------------------- date scales
def test_date_constructors_accept_sec_axis():
    d = dup_axis()
    for ctor in (scale_x_date, scale_y_date, scale_x_datetime):
        s = ctor(sec_axis=d)
        assert isinstance(s, ScaleDatePosition) and s.sec_axis is not None
    assert scale_x_date().sec_axis is None
    with pytest.raises(TypeError):
        scale_x_date(sec_axis=lambda x: x)


def test_datetime_dup_axis_in_other_timezone():
    p = (
        ggplot(_hourly(), aes("t", "v"))
        .geom_line()
        .add(
            scale_x_datetime(
                date_labels="%H:%M",
                sec_axis=dup_axis(name="Tokyo", labels=label_date("%H:%M", tz="Asia/Tokyo")),
            )
        )
    )
    pan = p.build().panel
    assert pan.panel_params.x_labels[:2] == ("00:00", "12:00")
    assert pan.x_sec_labels[:2] == ("09:00", "21:00")
    assert np.allclose(pan.x_sec_breaks, pan.panel_params.x_breaks)
    assert pan.x_sec_side == "top"


def test_datetime_numeric_shift_transform():
    # seconds since the epoch: +9 h, as ggplot2's ``. + 9 * 3600``
    p = (
        ggplot(_hourly(), aes("t", "v"))
        .geom_line()
        .add(scale_x_datetime(sec_axis=sec_axis(lambda t: t + 9 * 3600, name="+9h")))
    )
    pan = p.build().panel
    assert pan.x_sec_labels and all(":" in s for s in pan.x_sec_labels)
    # each secondary break sits 9 h before the instant it names
    named = to_epoch([pd.Timestamp("2024-03-01 12:00")], "s")[0]
    assert named - 9 * 3600 in np.round(pan.x_sec_breaks)


def test_date_breaks_and_transform_returning_dates():
    shift = pd.Timedelta(hours=1)
    sec = sec_axis(lambda t: pd.to_datetime(t, unit="s") + shift, breaks=["2024-03-02"])
    p = ggplot(_hourly(), aes("t", "v")).geom_line().add(scale_x_datetime(sec_axis=sec))
    pan = p.build().panel
    assert pan.x_sec_labels == ("Mar 02",)
    expected = to_epoch(["2024-03-02"], "s")[0] - 3600
    assert pan.x_sec_breaks == pytest.approx((expected,))


def test_date_scale_dup_axis_labels():
    d = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=12, freq="MS"), "v": range(12)})
    p = (
        ggplot(d, aes("d", "v"))
        .geom_line()
        .add(scale_x_date(sec_axis=dup_axis(labels=label_date("%j"))))
    )
    pan = p.build().panel
    assert pan.x_sec_labels[0] == "001"
    assert all(len(s) == 3 and s.isdigit() for s in pan.x_sec_labels)
    # primary labels: month names, inherited by a plain dup_axis
    plain = ggplot(d, aes("d", "v")).geom_line().add(scale_x_date(sec_axis=dup_axis()))
    pp = plain.build().panel
    assert pp.x_sec_labels == pp.panel_params.x_labels


def test_sec_axis_follows_data_timezone():
    # tz-aware data: the scale adopts the zone, and an unbound date-aware
    # secondary formatter is re-bound to it
    p = (
        ggplot(_hourly(tz="America/New_York"), aes("t", "v"))
        .geom_line()
        .add(scale_x_datetime(sec_axis=dup_axis(labels=label_date("%H:%M"))))
    )
    pan = p.build().panel
    assert pan.x_sec_labels[0] == "00:00"


def test_date_breaks_on_continuous_scale_rejected():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    p = (
        ggplot(df, aes("x", "y"))
        .geom_point()
        .add(scale_x_continuous(sec_axis=sec_axis(breaks=["2024-01-01"])))
    )
    with pytest.raises(TypeError, match="date"):
        p.build()
