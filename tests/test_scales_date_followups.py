"""Date scale follow-ups (ggm-e0o): continuous-scale-over-dates error, ``timezone=``
on date-time scales, :func:`label_date_short` and multi-line tick labels."""

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from manim import Paragraph

from ggmanim import (
    aes,
    ggplot,
    label_date,
    label_date_short,
    scale_x_continuous,
    scale_x_date,
    scale_x_datetime,
    scale_x_discrete,
    scale_x_log10,
    scale_y_continuous,
    scale_y_date,
    theme,
)
from ggmanim._tz import local_to_utc, resolve_tz, utc_to_local
from ggmanim.build import BuildError, build_plot
from ggmanim.scales.date import ScaleDatePosition, data_timezone, date_breaks_width, to_epoch

HOUR = 3600.0


def _monthly(n=24):
    return pd.DataFrame(
        {"d": pd.date_range("2021-01-01", periods=n, freq="MS"), "v": np.arange(n, dtype=float)}
    )


def _hourly(tz="UTC", start="2021-03-13", n=72):
    return pd.DataFrame({"t": pd.date_range(start, periods=n, freq="h", tz=tz), "v": range(n)})


# ----------------------------------------------------------------- (1) continuous over dates
@pytest.mark.parametrize("scale", [scale_x_continuous, scale_x_log10])
def test_continuous_x_scale_over_dates_raises(scale):
    with pytest.raises(BuildError, match=r"scale_x_date\(\) or scale_x_datetime\(\)"):
        build_plot(ggplot(_monthly(), aes("d", "v")).geom_line().add(scale()))


def test_continuous_y_scale_over_dates_and_date_objects_raise():
    df = pd.DataFrame({"d": [dt.date(2021, 1, i) for i in range(1, 6)], "v": range(5)})
    with pytest.raises(BuildError, match="scale_y_date"):
        build_plot(ggplot(df, aes("v", "d")).geom_point().add(scale_y_continuous()))
    with pytest.raises(BuildError, match="dates/date-times"):
        build_plot(ggplot(_hourly(), aes("t", "v")).geom_line().add(scale_x_continuous()))


def test_date_scales_and_numeric_columns_still_fine():
    g = build_plot(ggplot(_monthly(), aes("d", "v")).geom_line().add(scale_x_date()))
    assert isinstance(g.scales["x"], ScaleDatePosition)
    g = build_plot(ggplot(_monthly(), aes("v", "v")).geom_line().add(scale_x_continuous()))
    assert not isinstance(g.scales["x"], ScaleDatePosition)
    # a discrete scale over dates is allowed (one level per date)
    df = _monthly(3).assign(d=lambda f: f["d"].dt.strftime("%Y-%m"))
    build_plot(ggplot(df, aes("d", "v")).geom_col().add(scale_x_discrete()))


# ----------------------------------------------------------------- (2) time zones
def test_tz_helpers_round_trip_and_dst():
    ny = resolve_tz("America/New_York")
    assert isinstance(ny, ZoneInfo) and resolve_tz(None) is None
    utc = to_epoch(["2021-01-15 12:00", "2021-07-15 12:00"], "s")
    wall = utc_to_local(utc, ny)
    np.testing.assert_allclose(utc - wall, [5 * HOUR, 4 * HOUR])  # EST / EDT
    np.testing.assert_allclose(local_to_utc(wall, ny), utc)
    assert np.isnan(utc_to_local([np.nan], ny)[0])
    with pytest.raises(ValueError, match="unknown time zone"):
        resolve_tz("Not/AZone")


def test_timezone_moves_breaks_and_labels():
    p = ggplot(_hourly(), aes("t", "v")).geom_line()
    utc = build_plot(p.add(scale_x_datetime())).scales["x"]
    ist = build_plot(p.add(scale_x_datetime(timezone="Asia/Kolkata"))).scales["x"]
    assert utc.get_labels()[0] == "Mar 13 00:00"
    assert ist.get_labels()[0] == "Mar 13 12:00"
    # breaks sit on Kolkata half-days: UTC 06:30 / 18:30
    b = np.asarray(ist.get_breaks())
    np.testing.assert_allclose((b % (12 * HOUR)) / HOUR, 6.5)


def test_date_breaks_follow_local_days_across_dst():
    s = scale_x_datetime(date_breaks="1 day", date_labels="%d %H:%M", timezone="America/New_York")
    s.train(to_epoch(["2021-03-12 12:00", "2021-03-16 12:00"], "s"))
    b = np.asarray(s.get_breaks())
    assert set(s.get_labels()) <= {f"{d} 00:00" for d in range(12, 18)}
    gaps = np.diff(b) / HOUR
    assert 23 in gaps and set(gaps) <= {23.0, 24.0}  # the spring-forward day is 23 h
    # date_breaks_width with tz directly
    f = date_breaks_width("1 day", "s", tz="America/New_York")
    assert np.all(np.diff(f(to_epoch(["2021-03-13", "2021-03-16"], "s"))) / HOUR <= 24)


def test_data_timezone_used_by_default_and_overridden():
    df = _hourly(tz="America/New_York")
    assert data_timezone([df["t"]]).key == "America/New_York"
    assert data_timezone([_hourly(tz=None)["t"]]) is None
    assert data_timezone([df["t"], _hourly(tz="Europe/Paris")["t"]]) is None
    g = build_plot(ggplot(df, aes("t", "v")).geom_line())
    assert g.scales["x"].timezone.key == "America/New_York"
    assert g.scales["x"].get_labels()[:2] == ["Mar 13", "Mar 14"]  # local midnights
    user = scale_x_datetime(date_breaks="12 hours", date_labels="%H:%M %Z")
    g = build_plot(ggplot(df, aes("t", "v")).geom_line().add(user))
    labels = g.scales["x"].get_labels()
    assert labels[0] == "00:00 EST" and "12:00 EDT" in labels
    assert user.timezone is None  # the user's scale is not mutated by the build
    g = build_plot(ggplot(df, aes("t", "v")).geom_line().add(scale_x_datetime(timezone="UTC")))
    assert g.scales["x"].get_labels()[0] == "Mar 13 12:00"


def test_naive_data_is_utc_and_timezone_validation():
    s = scale_x_datetime(timezone="Europe/Paris", date_labels="%H:%M")
    s.train(to_epoch(["2021-01-01 00:00", "2021-01-01 06:00"], "s"))
    assert s.get_labels()[0] == "01:00"  # naive midnight = UTC midnight = 01:00 CET
    with pytest.raises(ValueError, match="date-time"):
        ScaleDatePosition(("x",), unit="D", timezone="UTC")
    with pytest.raises(ValueError, match="unknown time zone"):
        scale_x_datetime(timezone="Mars/Olympus")


def test_label_date_tz():
    sec = to_epoch(["2021-07-01 12:00"], "s")
    assert label_date("%H:%M %Z", unit="s", tz="Europe/London")(sec) == ["13:00 BST"]
    aware = dt.datetime(2021, 7, 1, 12, tzinfo=dt.timezone.utc)
    assert label_date("%H:%M", tz="Asia/Tokyo")([aware]) == ["21:00"]
    assert label_date("%H:%M", unit="s")([np.nan, np.inf]) == ["", "Inf"]


# ----------------------------------------------------------------- (3) label_date_short
def test_label_date_short_quarters_and_months():
    d = to_epoch(pd.date_range("2021-01-01", periods=8, freq="QS"))
    assert label_date_short()(d) == [
        "Jan\n2021", "Apr", "Jul", "Oct", "Jan\n2022", "Apr", "Jul", "Oct",
    ]  # fmt: skip
    years = to_epoch(["2020-01-01", "2021-01-01", "2022-01-01"])
    assert label_date_short()(years) == ["2020", "2021", "2022"]


def test_label_date_short_days_times_and_missing():
    days = to_epoch(["2021-12-30", "2021-12-31", "2022-01-01", "2022-01-02"])
    assert label_date_short(sep=" ")(days) == ["30 Dec 2021", "31", "01 Jan 2022", "02"]
    hours = to_epoch(pd.date_range("2021-03-13 18:00", periods=3, freq="6h"), "s")
    assert label_date_short(unit="s")(hours) == ["18:00\n13\nMar\n2021", "00:00\n14", "06:00"]
    assert label_date_short(unit="s", tz="Asia/Tokyo")(hours)[1] == "09:00"
    with_nan = [np.nan, *to_epoch(["2021-01-01", "2022-01-01"])]
    assert label_date_short()(with_nan) == ["", "2021", "2022"]
    with pytest.raises(ValueError, match="4 strftime"):
        label_date_short(format=("%Y",))


def test_label_date_short_on_scale():
    p = ggplot(_monthly(), aes("d", "v")).geom_line()
    x = build_plot(p.add(scale_x_date(date_breaks="6 months", labels=label_date_short()))).scales[
        "x"
    ]
    assert x.get_labels()[:3] == ["Jan\n2021", "Jul", "Jan\n2022"]


def test_multiline_tick_labels_render_centred():
    p = (
        ggplot(_monthly(), aes("d", "v"))
        .geom_line()
        .add(scale_x_date(date_breaks="6 months", labels=label_date_short()))
    )
    m = p.build()
    labs = list(m.panel.x_tick_labels)
    two = [lab for lab in labs if isinstance(lab, Paragraph)]
    assert len(two) >= 2 and all(len(lab) == 2 for lab in two)
    single = next(lab for lab in labs if not isinstance(lab, Paragraph))
    assert two[0].height > 1.5 * single.height
    for lab, tick in zip(labs, m.panel.x_ticks):  # each label centred under its tick
        assert lab.get_center()[0] == pytest.approx(tick.get_center()[0], abs=1e-3)
    # the tex engine gets a LaTeX line break instead
    m = p.add(theme(text_engine="tex")).build()
    assert not any(isinstance(lab, Paragraph) for lab in m.panel.x_tick_labels)


def test_scale_y_date_multiline_right_aligned():
    df = _monthly()
    m = (
        ggplot(df, aes("v", "d"))
        .geom_point()
        .add(scale_y_date(date_breaks="6 months", labels=label_date_short()))
        .build()
    )
    para = next(lab for lab in m.panel.y_tick_labels if isinstance(lab, Paragraph))
    month, year = para[0], para[1]
    assert year.get_right()[0] == pytest.approx(month.get_right()[0], abs=1e-2)
