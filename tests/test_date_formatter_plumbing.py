"""ggm-boe: date scales pass unit/time zone to date-aware formatters, multi-line
legend labels/titles via ``Theme.make_text`` (Paragraph), ``label_date_short(leading=)``."""

import datetime as dt

import pandas as pd
import pytest
from manim import Paragraph, Tex, Text

from ggmanim import (
    aes,
    ggplot,
    label_date,
    label_date_short,
    labs,
    scale_color_discrete,
    scale_x_date,
    scale_x_datetime,
    theme,
)
from ggmanim.build import build_plot
from ggmanim.labels import DateFormatter, bind_date_formatter
from ggmanim.theme import Theme, hjust_align

HOUR = 3600.0


def _hourly(tz=None, start="2021-03-01", n=48, freq="6h"):
    return pd.DataFrame({"t": pd.date_range(start, periods=n, freq=freq, tz=tz), "v": range(n)})


# -- (1) date-aware formatters --------------------------------------------------------


def test_formatters_are_date_aware_and_bind():
    f = label_date("%H:%M")
    assert isinstance(f, DateFormatter) and f._date_aware
    assert f.unit is None and f.tz is None
    assert f([0.5]) == ["12:00"]  # unbound: days
    g = f.bind(unit="s", tz="Europe/Paris")
    assert g.unit == "s" and g.tz is not None
    assert g([0.0]) == ["01:00"]
    assert f([0.5]) == ["12:00"]  # binding returns a copy


def test_explicit_args_win_over_bind():
    f = label_date("%H:%M", unit="h", tz="UTC")
    g = f.bind(unit="s", tz="Asia/Tokyo")
    assert g.unit == "h" and str(g.tz) == "UTC"
    assert g([1.0]) == ["01:00"]


def test_bind_date_formatter_leaves_plain_callables():
    def plain(x):
        return [str(v) for v in x]

    assert bind_date_formatter(plain, "s") is plain
    assert bind_date_formatter(None, "s") is None
    assert bind_date_formatter(("a", "b"), "s") == ("a", "b")


def test_custom_date_aware_protocol():
    calls = []

    class Fmt:
        _date_aware = True

        def __call__(self, x):
            return ["?"] * len(x)

        def bind(self, unit=None, tz=None):
            calls.append((unit, tz))
            return lambda x: [f"{unit}:{v:g}" for v in x]

    sc = scale_x_datetime(labels=Fmt(), breaks=[0, HOUR])
    assert calls == [("s", None)]
    assert sc.get_labels((0, HOUR)) == ["s:0", "s:3600"]


def test_datetime_scale_supplies_unit():
    df = _hourly()
    sc = build_plot(
        ggplot(df, aes("t", "v")).geom_line().add(scale_x_datetime(labels=label_date_short()))
    ).scales["x"]
    labels = sc.get_labels()
    # seconds were interpreted as seconds (as days they would be in the 6th millennium)
    assert labels[0] == "02\nMar\n2021"
    assert sc.get_labels() == label_date_short(unit="s")(sc.get_breaks())


def test_date_scale_supplies_unit_D():
    df = pd.DataFrame({"d": pd.date_range("2021-01-01", periods=24, freq="MS"), "v": range(24)})
    sc = build_plot(
        ggplot(df, aes("d", "v"))
        .geom_line()
        .add(scale_x_date(date_breaks="6 months", labels=label_date("%b %Y")))
    ).scales["x"]
    assert sc.get_labels()[:2] == ["Jan 2021", "Jul 2021"]


def test_datetime_scale_supplies_timezone_argument():
    sc = scale_x_datetime(labels=label_date("%H:%M"), timezone="America/New_York", breaks=[0])
    assert sc.get_labels((0, 1)) == ["19:00"]
    # explicit tz wins
    sc = scale_x_datetime(labels=label_date("%H:%M", tz="UTC"), timezone="America/New_York")
    assert sc.labels([0.0]) == ["00:00"]


def test_datetime_scale_supplies_data_timezone():
    df = _hourly(tz="Asia/Tokyo", start="2021-03-01 00:00", n=8, freq="h")
    sc = build_plot(
        ggplot(df, aes("t", "v")).geom_line().add(scale_x_datetime(labels=label_date("%H:%M")))
    ).scales["x"]
    breaks = sc.get_breaks()
    labs_ = sc.get_labels()
    expected = [
        (dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=float(b)))
        .astimezone(dt.timezone(dt.timedelta(hours=9)))
        .strftime("%H:%M")
        for b in breaks
    ]
    assert labs_ == expected
    assert "00:00" in labs_  # local midnight is labelled as such


def test_sequence_labels_untouched_by_set_timezone():
    sc = scale_x_datetime(breaks=[0, HOUR], labels=["a", "b"])
    c = sc.clone()
    c.set_timezone("Europe/Paris")
    assert c.labels == ("a", "b")


# -- (3) leading= ------------------------------------------------------------------------


def test_label_date_short_leading():
    days = [dt.date(2021, 3, d) for d in (1, 5, 9)]
    assert label_date_short()(days) == ["01\nMar\n2021", "05", "09"]
    assert label_date_short(leading="")(days) == ["1\nMar\n2021", "5", "9"]
    assert label_date_short(leading=" ")(days)[1] == " 5"
    times = [dt.datetime(2021, 3, 1, h) for h in (6, 9, 12)]
    assert label_date_short(leading="")(times) == ["6:00\n1\nMar\n2021", "9:00", "12:00"]
    # replacement is literal (no regex escapes)
    assert label_date_short(leading="\\1")(days)[1] == "\\15"


# -- (2) multi-line text via make_text -------------------------------------------------


def test_make_text_multiline_paragraph():
    th = Theme()
    one = th.make_text("ab")
    assert isinstance(one, Text)
    two = th.make_text("ab\ncdef", align="right")
    assert isinstance(two, Paragraph) and len(two) == 2
    assert two[0].get_right()[0] == pytest.approx(two[1].get_right()[0], abs=1e-2)
    left = th.make_text("ab\ncdef")
    assert left[0].get_left()[0] == pytest.approx(left[1].get_left()[0], abs=1e-2)
    with pytest.raises(ValueError):
        th.make_text("a\nb", align="middle")
    tex = Theme(text_engine="tex").make_text("a\nb")
    assert isinstance(tex, Tex)


def test_hjust_align():
    assert [hjust_align(h) for h in (0, 0.5, 1)] == ["left", "center", "right"]


def test_legend_multiline_title_and_labels():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "g": ["one\nline two", "b", "c"]})
    m = (
        ggplot(df, aes("x", "y", color="g"))
        .geom_point()
        .add(labs(color="Group\nname"))
        .add(scale_color_discrete())
        .build()
    )
    (leg,) = m.legends
    assert isinstance(leg.title, Paragraph) and len(leg.title) == 2
    assert leg.title[0].get_left()[0] == pytest.approx(leg.title[1].get_left()[0], abs=1e-2)
    assert isinstance(leg.labels[2], Paragraph) and len(leg.labels[2]) == 2
    assert isinstance(leg.labels[0], Text)


def test_colourbar_multiline_title():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "z": [0.0, 0.5, 1.0]})
    m = ggplot(df, aes("x", "y", color="z")).geom_point().add(labs(color="Value\n(units)")).build()
    (cb,) = m.legends
    assert isinstance(cb.title, Paragraph)


def test_tex_engine_multiline_legend_title():
    df = pd.DataFrame({"x": [1, 2], "y": [1, 2], "g": ["a", "b"]})
    m = (
        ggplot(df, aes("x", "y", color="g"))
        .geom_point()
        .add(labs(color="Two\nlines"))
        .add(theme(text_engine="tex"))
        .build()
    )
    (leg,) = m.legends
    assert isinstance(leg.title, Tex) and leg.title.height > leg.labels[0].height * 1.5
