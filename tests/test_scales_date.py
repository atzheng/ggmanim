"""Date / date-time position scales (ggmanim.scales.date)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ggmanim import aes, ggplot, scale_x_date, scale_x_datetime, scale_y_date
from ggmanim.build import build_plot
from ggmanim.labels import label_date
from ggmanim.scales.continuous import ScaleContinuousPosition
from ggmanim.scales.date import (
    ScaleDatePosition,
    date_breaks_auto,
    date_format_auto,
    date_kind,
    parse_date_width,
    to_epoch,
)

DAY = 86400.0


def _auto_labels(lo, hi, unit="D"):
    s = ScaleDatePosition("x", unit=unit, limits=(lo, hi))
    return s.get_labels()


# --------------------------------------------------------------------------- conversion
def test_to_epoch_accepts_many_date_types():
    out = to_epoch(
        [dt.date(1970, 1, 2), None, "1970-01-03", np.datetime64("1970-01-04"), pd.Timestamp(0)]
    )
    np.testing.assert_allclose(out, [1, np.nan, 2, 3, 0])
    np.testing.assert_allclose(to_epoch(["1970-01-01T00:01:00"], "s"), [60])
    np.testing.assert_allclose(to_epoch([np.nan, 5.0]), [np.nan, 5.0])  # numbers pass
    s = pd.Series(pd.date_range("2020-01-01", periods=2, freq="D"))
    np.testing.assert_allclose(to_epoch(s), [18262, 18263])


def test_to_epoch_tz_aware_is_converted_to_utc():
    s = pd.Series(pd.to_datetime(["2020-01-01 00:00"]).tz_localize("Etc/GMT+5"))
    np.testing.assert_allclose(to_epoch(s, "s"), to_epoch(["2020-01-01T05:00"], "s"))


def test_date_kind():
    assert date_kind([pd.Series(pd.date_range("2020", periods=3, freq="MS"))]) == "date"
    assert date_kind([pd.Series(pd.date_range("2020", periods=3, freq="h"))]) == "datetime"
    assert date_kind([pd.Series([dt.date(2020, 1, 1), None])]) == "date"
    assert date_kind([pd.Series([1.0, 2.0]), pd.Series(["a", "b"])]) is None


# --------------------------------------------------------------------------- breaks
def test_auto_breaks_three_days():
    assert _auto_labels("2020-01-01", "2020-01-04") == ["Jan 01", "Jan 02", "Jan 03", "Jan 04"]


def test_auto_breaks_three_months():
    assert _auto_labels("2020-01-01", "2020-04-01") == ["Jan", "Feb", "Mar", "Apr"]


def test_auto_breaks_five_years():
    assert _auto_labels("2020-01-01", "2025-01-01") == [
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    ]


def test_auto_breaks_hours_for_datetime():
    labs = _auto_labels("2020-01-01T00:00", "2020-01-01T06:30", unit="s")
    assert labs == ["00:00", "02:00", "04:00", "06:00"]


def test_auto_breaks_never_below_a_day_for_dates():
    b = date_breaks_auto(0, 1, n=5, unit="D")
    assert np.allclose(b, np.round(b))


def test_weekly_breaks_start_on_monday():
    lo, hi = to_epoch(["2020-01-01", "2020-01-31"])
    b = date_breaks_auto(lo, hi, unit="D")
    days = (np.asarray(b) + 3) % 7  # 1970-01-01 was a Thursday -> Monday == 0
    assert np.all(days == 0)


def test_parse_date_width():
    assert parse_date_width("1 month") == ("month", 1)
    assert parse_date_width("2 weeks") == ("week", 2)
    assert parse_date_width("year") == ("year", 1)
    assert parse_date_width("1 quarter") == ("month", 3)
    assert isinstance(parse_date_width("MS"), pd.offsets.BaseOffset)
    with pytest.raises(ValueError, match="cannot parse"):
        parse_date_width("1 fortnight")


def test_date_breaks_width_and_pandas_alias():
    s = scale_x_date(date_breaks="1 month", limits=("2020-01-01", "2020-06-15"))
    assert s.get_labels() == ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    s = scale_x_date(date_breaks="QS", limits=("2020-01-01", "2020-12-31"))
    assert s.get_labels() == ["Jan", "Apr", "Jul", "Oct"]
    s = scale_x_datetime(date_breaks="6 hours", limits=("2020-01-01", "2020-01-02"))
    assert s.get_labels() == [
        "Jan 01 00:00",
        "Jan 01 06:00",
        "Jan 01 12:00",
        "Jan 01 18:00",
        "Jan 02 00:00",
    ]


def test_date_minor_breaks():
    s = scale_x_date(
        date_breaks="1 month", date_minor_breaks="1 week", limits=("2020-01-01", "2020-02-29")
    )
    minor = s.get_minor_breaks()
    assert len(minor) >= 6
    assert not set(np.round(minor)) & set(np.round(s.get_breaks()))


def test_breaks_as_dates():
    s = scale_x_date(
        breaks=["2020-02-01", dt.date(2020, 3, 1)], limits=("2020-01-01", "2020-04-01")
    )
    np.testing.assert_allclose(s.get_breaks(), to_epoch(["2020-02-01", "2020-03-01"]))


# --------------------------------------------------------------------------- labels
def test_date_labels_strftime():
    s = scale_x_date(
        date_breaks="1 month", date_labels="%m/%Y", limits=("2020-01-01", "2020-03-01")
    )
    assert s.get_labels() == ["01/2020", "02/2020", "03/2020"]
    s = scale_x_date(labels=label_date("%d %b"), limits=("2020-01-01", "2020-01-03"))
    assert s.get_labels() == ["01 Jan", "02 Jan", "03 Jan"]


def test_date_format_auto():
    jan = to_epoch(["2020-01-01", "2021-01-01"])
    assert date_format_auto(jan) == "%Y"
    assert date_format_auto(to_epoch(["2020-01-01", "2020-07-01"])) == "%b"
    assert date_format_auto(to_epoch(["2020-07-01", "2021-01-01"])) == "%b %Y"
    assert date_format_auto(to_epoch(["2020-07-03", "2021-01-01"])) == "%Y-%m-%d"
    assert date_format_auto(to_epoch(["2020-01-01T10:30:15"], "s"), "s") == "%H:%M:%S"


# --------------------------------------------------------------------------- limits
def test_limits_as_iso_strings_and_dates():
    s = scale_x_date(limits=("2020-01-01", dt.date(2020, 12, 31)))
    assert s.get_limits() == (18262.0, 18627.0)
    s = scale_x_date(limits=("2020-01-01", None))
    s.train(pd.Series(pd.to_datetime(["2019-06-01", "2020-03-01"])))
    assert s.get_limits() == (18262.0, float(to_epoch(["2020-03-01"])[0]))
    # out-of-limits values are censored
    out = s.map(pd.Series(pd.to_datetime(["2019-06-01", "2020-02-01"])))
    assert np.isnan(out[0]) and np.isfinite(out[1])


# --------------------------------------------------------------------------- build
def _monthly(n=36):
    return pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=n, freq="MS"),
            "value": np.arange(n, dtype=float),
        }
    )


def test_datetime64_column_gets_date_scale_without_user_scale():
    g = build_plot(ggplot(_monthly(), aes("month", "value")).geom_line())
    x = g.scales["x"]
    assert isinstance(x, ScaleDatePosition) and x.unit == "D"
    assert isinstance(g.scales["y"], ScaleContinuousPosition)
    assert x.get_labels(x.get_expanded_limits()) == ["2020", "2021", "2022", "2023"]
    one_year = build_plot(ggplot(_monthly(12), aes("month", "value")).geom_line()).scales["x"]
    assert one_year.get_labels() == ["Jan", "Apr", "Jul", "Oct"]


def test_geom_line_over_datetime_x():
    df = pd.DataFrame(
        {"t": pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC"), "v": range(48)}
    )
    g = build_plot(ggplot(df, aes("t", "v")).geom_line())
    x = g.scales["x"]
    assert x.unit == "s"
    assert x.get_labels()[0] == "Jan 01 00:00"
    assert len(g.layers[0]) > 0


def test_date_objects_vline_and_y_date():
    df = pd.DataFrame(
        {"d": [dt.date(2021, 1, 1) + dt.timedelta(days=i) for i in range(10)], "v": range(10)}
    )
    g = build_plot(ggplot(df, aes("d", "v")).geom_point().geom_vline(xintercept="2021-01-05"))
    assert isinstance(g.scales["x"], ScaleDatePosition)
    g = build_plot(ggplot(df, aes("v", "d")).geom_point().add(scale_y_date(date_labels="%d")))
    assert g.scales["y"].get_labels()[0].isdigit()


def test_coord_cartesian_xlim_with_dates():
    p = ggplot(_monthly(), aes("month", "value")).geom_line()
    g = build_plot(p.coord_cartesian(xlim=("2020-06-01", "2021-06-01")))
    x = g.scales["x"]
    lo, hi = to_epoch(["2020-06-01", "2021-06-01"])
    x_lo, x_hi = g.panel_params.x_range
    assert lo - 40 < x_lo <= lo and hi <= x_hi < hi + 40
    assert "Jan 2021" in g.panel_params.x_labels
    assert len(g.layers[0]) > 0 and isinstance(x, ScaleDatePosition)
