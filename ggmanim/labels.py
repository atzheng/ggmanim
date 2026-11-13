"""Label formatters and break helpers for scales (``scales``-package style).

Every ``label_*`` function returns a *formatter*: a callable taking an array of
data-space break values and returning one string per value, usable as a scale's
``labels=``::

    ggplot(df, aes("x", "y")).geom_point().scale_y_continuous(labels=label_percent())

Every ``breaks_*`` function returns a *break function*: a callable taking the
data-space limits ``(lo, hi)`` and returning data-space break positions, usable
as a scale's ``breaks=`` (or ``minor_breaks=``)::

    scale_x_continuous(breaks=breaks_width(2.5), minor_breaks=breaks_width(0.5))

Missing values (NaN) format as ``""`` (no label drawn); infinities as ``"Inf"`` /
``"-Inf"``.

Math labels
-----------
:func:`label_math` returns LaTeX math strings wrapped in ``$...$`` (e.g.
``"$10^{3}$"``). The coord renders them according to ``theme.text_engine``: with
``"tex"`` they go to :class:`~manim.Tex` as they are; with ``"text"`` they are
converted to plain text with Unicode superscripts (``"10³"``) by
:func:`render_label`. Any label string of the form ``"$...$"`` gets this
treatment, so hand-written math labels work too.

Pure Python/numpy; no manim imports.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import textwrap
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .scales.breaks import (
    _common_decimals,
    _display_decimals,
    extended,
    log_breaks,
    pretty,
    width_breaks,
)

__all__ = [
    "label_number",
    "label_comma",
    "label_percent",
    "label_dollar",
    "label_currency",
    "label_scientific",
    "label_math",
    "label_date",
    "label_date_short",
    "DateFormatter",
    "bind_date_formatter",
    "label_wrap",
    "breaks_pretty",
    "breaks_extended",
    "breaks_width",
    "breaks_log",
    "render_label",
]

Formatter = Callable[[Any], list[str]]
BreakFunction = Callable[[Sequence[float]], np.ndarray]


def _floats(x: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float)).ravel()


def _nonfinite(v: float) -> str:
    if math.isnan(v):
        return ""
    return "Inf" if v > 0 else "-Inf"


# --------------------------------------------------------------------------- numbers
def _group(int_digits: str, big_mark: str) -> str:
    if not big_mark or len(int_digits) <= 3:
        return int_digits
    head = len(int_digits) % 3 or 3
    parts = [int_digits[:head]] + [int_digits[i : i + 3] for i in range(head, len(int_digits), 3)]
    return big_mark.join(parts)


def _decimals_for(accuracy: float | None, x: np.ndarray) -> int:
    """Decimal places: from ``accuracy`` (e.g. 0.01 -> 2, 5 -> 0) or, if ``None``,
    the fewest that show every finite value exactly, up to 7 significant digits
    (shared by all labels)."""
    if accuracy is not None:
        accuracy = float(accuracy)
        if not accuracy > 0:
            raise ValueError(f"accuracy must be positive, got {accuracy}")
        return max(0, -math.floor(math.log10(accuracy) + 1e-10))
    fin = x[np.isfinite(x)]
    if not len(fin) or not np.any(fin):
        return 0
    return _display_decimals(fin)


def _format_fixed(
    x: np.ndarray,
    accuracy: float | None,
    prefix: str,
    suffix: str,
    big_mark: str,
    decimal_mark: str,
    trim: bool,
) -> list[str]:
    d = _decimals_for(accuracy, x)
    out = []
    for v in x:
        if not math.isfinite(v):
            out.append(_nonfinite(v))
            continue
        if accuracy is not None:
            v = round(v / accuracy) * accuracy
        s = f"{abs(v):.{d}f}"
        neg = v < 0 and float(s) != 0.0
        int_part, _, frac = s.partition(".")
        body = _group(int_part, big_mark) + (decimal_mark + frac if frac else "")
        out.append(("-" if neg else "") + prefix + body + suffix)
    if not trim:
        width = max((len(s) for s in out), default=0)
        out = [s.rjust(width) if s else s for s in out]
    return out


def label_number(
    accuracy: float | None = None,
    scale: float = 1,
    prefix: str = "",
    suffix: str = "",
    big_mark: str = ",",
    decimal_mark: str = ".",
    trim: bool = True,
) -> Formatter:
    """Fixed-point numbers (``scales::label_number``).

    ``accuracy``: round to a multiple of this (``0.01`` -> 2 decimals, ``10`` -> tens);
    ``None`` picks the fewest decimals that show every value, shared by all labels.
    ``scale`` multiplies the values first; ``prefix``/``suffix`` wrap the number
    (a minus sign goes before the prefix: ``-$5``). ``big_mark`` separates thousands,
    ``decimal_mark`` replaces ``"."``. ``trim=False`` right-justifies all labels to a
    common width.

    >>> label_number()([0, 1234.5, -12345678.9])
    ['0.0', '1,234.5', '-12,345,678.9']
    """

    def fmt(x: Any) -> list[str]:
        v = _floats(x) * float(scale)
        return _format_fixed(v, accuracy, prefix, suffix, big_mark, decimal_mark, trim)

    return fmt


def label_comma(
    accuracy: float | None = None,
    scale: float = 1,
    prefix: str = "",
    suffix: str = "",
    big_mark: str = ",",
    decimal_mark: str = ".",
    trim: bool = True,
) -> Formatter:
    """Numbers with a comma thousands separator (``scales::label_comma``)."""
    return label_number(accuracy, scale, prefix, suffix, big_mark, decimal_mark, trim)


def label_percent(
    accuracy: float | None = None,
    scale: float = 100,
    prefix: str = "",
    suffix: str = "%",
    big_mark: str = ",",
    decimal_mark: str = ".",
    trim: bool = True,
) -> Formatter:
    """Percentages: values are multiplied by ``scale`` (100) and suffixed with ``%``.

    >>> label_percent()([0, 0.25, 1])
    ['0%', '25%', '100%']
    """
    return label_number(accuracy, scale, prefix, suffix, big_mark, decimal_mark, trim)


def label_currency(
    accuracy: float | None = None,
    scale: float = 1,
    prefix: str = "$",
    suffix: str = "",
    big_mark: str = ",",
    decimal_mark: str = ".",
    trim: bool = True,
    largest_with_cents: float = 100_000,
) -> Formatter:
    """Currency (``scales::label_currency``/``label_dollar``).

    With ``accuracy=None``, cents are shown (``accuracy=0.01``) when some value is
    fractional and the largest is below ``largest_with_cents``; otherwise whole
    units. Negative values are written ``-$5``.

    >>> label_dollar()([0, 1500, -20])
    ['$0', '$1,500', '-$20']
    """

    def fmt(x: Any) -> list[str]:
        v = _floats(x) * float(scale)
        acc = accuracy
        if acc is None:
            fin = v[np.isfinite(v)]
            fractional = bool(len(fin)) and bool(np.any(np.abs(fin - np.round(fin)) > 1e-9))
            small = bool(len(fin)) and float(np.max(np.abs(fin))) < largest_with_cents
            acc = 0.01 if (fractional and small) else 1.0
        return _format_fixed(v, acc, prefix, suffix, big_mark, decimal_mark, trim)

    return fmt


#: Alias of :func:`label_currency` (``scales::label_dollar``).
label_dollar = label_currency


def label_scientific(digits: int = 3, decimal_mark: str = ".") -> Formatter:
    """Scientific notation, R style (``"1.23e+07"``), with at most ``digits``
    significant digits; all labels share the fewest mantissa decimals needed.

    >>> label_scientific()([0, 12345678.9, 1e-7])
    ['0.00e+00', '1.23e+07', '1.00e-07']
    """
    digits = int(digits)
    if digits < 1:
        raise ValueError(f"digits must be >= 1, got {digits}")

    def fmt(x: Any) -> list[str]:
        v = _floats(x)
        fin = v[np.isfinite(v)]
        parts: list[tuple[float, int]] = []
        for f in fin:
            if f == 0:
                parts.append((0.0, 0))
                continue
            m_str, e_str = f"{f:.{digits - 1}e}".split("e")
            parts.append((float(m_str), int(e_str)))
        mants = np.array([p[0] for p in parts]) if parts else np.zeros(0)
        d = _common_decimals(mants, max_digits=digits - 1) if len(mants) else 0
        it = iter(parts)
        out = []
        for f in v:
            if not math.isfinite(f):
                out.append(_nonfinite(f))
                continue
            m, e = next(it)
            sign = "-" if e < 0 else "+"
            out.append(f"{m:.{d}f}".replace(".", decimal_mark) + f"e{sign}{abs(e):02d}")
        return out

    return fmt


def label_math(base: float = 10, digits: int = 3) -> Formatter:
    """Powers of ``base``: each value ``x`` becomes ``base^log_base(x)`` as LaTeX math
    (``"$10^{3}$"``); rendered with superscripts by the coord (Unicode ``10³`` with the
    Text engine, LaTeX with ``theme(text_engine="tex")``). Meant for log scales:
    ``scale_x_log10(labels=label_math())``. Non-integer exponents are shown with up
    to ``digits`` significant digits; values <= 0 give ``""``.
    """
    b = float(base)
    if not (b > 0 and b != 1):
        raise ValueError(f"base must be positive and != 1, got {base}")
    base_s = "e" if b == math.e else f"{b:g}"

    def fmt(x: Any) -> list[str]:
        out = []
        for v in _floats(x):
            if not (math.isfinite(v) and v > 0):
                out.append("" if not math.isinf(v) or v < 0 else "Inf")
                continue
            k = math.log(v) / math.log(b)
            kr = round(k)
            k_s = str(int(kr)) if abs(k - kr) < 1e-9 else f"{k:.{digits}g}"
            out.append(f"${base_s}^{{{k_s}}}$")
        return out

    return fmt


_DATE_UNITS = {"D": "days", "h": "hours", "m": "minutes", "s": "seconds", "ms": "milliseconds"}
_EPOCH = _dt.datetime(1970, 1, 1)


def _to_datetime(v: Any, unit: str, tz: _dt.tzinfo | None) -> _dt.date | None:
    """One break value -> ``date``/``datetime`` (``None`` for missing), shown in
    ``tz`` when given (numbers, naive datetimes and ``datetime64`` are UTC)."""
    if v is None:
        return None
    if isinstance(v, np.datetime64):
        if np.isnat(v):
            return None
        v = v.astype("datetime64[us]").astype(_dt.datetime)
    if isinstance(v, _dt.datetime):
        if tz is None:
            return v
        return (v if v.tzinfo else v.replace(tzinfo=_dt.timezone.utc)).astimezone(tz)
    if isinstance(v, _dt.date):
        return v
    f = float(v)
    if not math.isfinite(f):
        return None
    d = _EPOCH + _dt.timedelta(**{_DATE_UNITS[unit]: f})
    return d if tz is None else d.replace(tzinfo=_dt.timezone.utc).astimezone(tz)


def _check_unit(unit: str) -> None:
    if unit not in _DATE_UNITS:
        raise ValueError(f"unit must be one of {sorted(_DATE_UNITS)}, got {unit!r}")


class DateFormatter:
    """A date formatter that a date scale can complete with its ``unit``/``tz``.

    Returned by :func:`label_date` and :func:`label_date_short`; call it like any
    formatter. **Date-aware protocol**: an object with ``_date_aware = True`` and a
    ``bind(unit=..., tz=...)`` method returning a formatter. Date and date-time
    scales (:class:`~ggmanim.scales.date.ScaleDatePosition`) call ``bind`` with their
    own unit (``"D"``/``"s"``) and time zone, so ``scale_x_datetime(labels=
    label_date_short())`` needs no ``unit="s"``/``tz=``. Arguments the user gave
    explicitly win over the scale's. Unbound, ``unit`` defaults to ``"D"`` and ``tz``
    to none.
    """

    _date_aware = True

    def __init__(
        self, make: Callable[[str, _dt.tzinfo | None], Formatter], unit: str | None, tz: Any
    ) -> None:
        from ._tz import resolve_tz

        if unit is not None:
            _check_unit(unit)
        self._make = make
        self.unit = unit
        self.tz = resolve_tz(tz)
        self._fn = make(unit if unit is not None else "D", self.tz)

    def __call__(self, x: Any) -> list[str]:
        return self._fn(x)

    def bind(self, unit: str | None = None, tz: Any = None) -> DateFormatter:
        """A copy using ``unit``/``tz`` where this formatter left them unset."""
        return DateFormatter(
            self._make,
            self.unit if self.unit is not None else unit,
            self.tz if self.tz is not None else tz,
        )

    def __repr__(self) -> str:
        return f"<DateFormatter unit={self.unit!r} tz={self.tz!r}>"


def bind_date_formatter(fmt: Any, unit: str, tz: Any = None) -> Any:
    """``fmt.bind(unit=unit, tz=tz)`` for a date-aware formatter (see
    :class:`DateFormatter`); anything else is returned unchanged."""
    if callable(fmt) and getattr(fmt, "_date_aware", False) and hasattr(fmt, "bind"):
        return fmt.bind(unit=unit, tz=tz)
    return fmt


def label_date(format: str = "%Y-%m-%d", unit: str | None = None, tz: Any = None) -> DateFormatter:
    """Dates/times with :meth:`~datetime.datetime.strftime` ``format``.

    Accepts ``datetime``/``date`` objects, ``numpy.datetime64`` values, or numbers
    counted in ``unit`` since 1970-01-01 (``"D"`` days, as for date scales; ``"s"``
    seconds for date-times; also ``"h"``, ``"m"``, ``"ms"``). Date position scales
    choose the breaks; this only formats them.

    ``tz`` (an IANA name such as ``"America/New_York"`` or a ``tzinfo``) shows the
    times in that zone: numbers, ``datetime64`` values and naive datetimes are taken
    as UTC, aware datetimes are converted. ``None`` formats them as they are.

    Used as ``labels=`` of a date/date-time scale, unset ``unit``/``tz`` are taken
    from the scale (see :class:`DateFormatter`); elsewhere ``unit`` defaults to
    ``"D"``.
    """

    def make(u: str, zone: _dt.tzinfo | None) -> Formatter:
        def one(v: Any) -> str:
            if v is not None and not isinstance(v, (_dt.date, np.datetime64)):
                f = float(v)
                if not math.isfinite(f) and not math.isnan(f):
                    return _nonfinite(f)
            d = _to_datetime(v, u, zone)
            return "" if d is None else d.strftime(format)

        def fmt(x: Any) -> list[str]:
            return [one(v) for v in np.atleast_1d(np.asarray(x)).ravel()]

        return fmt

    return DateFormatter(make, unit, tz)


_LEADING_ZERO_RE = re.compile(r"^0(?=\d)")


def label_date_short(
    format: Sequence[str] = ("%Y", "%b", "%d", "%H:%M"),
    sep: str = "\n",
    unit: str | None = None,
    tz: Any = None,
    leading: str = "0",
) -> DateFormatter:
    """Compact date labels (like ``scales::label_date_short``): each label shows only
    the components that changed since the previous break, finest first, joined by
    ``sep`` -- e.g. ``["Jan\\n2021", "Apr", "Jul", "Oct", "Jan\\n2022"]`` for
    quarterly breaks. The year/month/day formats are ``format[0:3]``; the time
    ``format[3]`` is shown on every label unless all breaks fall at midnight (then
    days are dropped too when all are 1st of the month, and months when all are
    January). A change in a coarse component (a new year) brings in all finer ones.
    ``leading`` replaces the leading zero of each component (``"05"`` -> ``"5"``
    with ``leading=""``, or ``"\\u2007"`` for a figure space); the default ``"0"``
    keeps it. ``unit`` and ``tz`` are as for :func:`label_date` (taken from the scale
    when unset). Labels containing ``"\\n"`` are drawn over several lines.
    """
    if len(format) != 4:
        raise ValueError("format must give 4 strftime formats: year, month, day, time")
    fy, fm, fd, ft = format
    leading = str(leading)

    def strf(d: _dt.date, p: str) -> str:
        s = d.strftime(p)
        return s if leading == "0" else _LEADING_ZERO_RE.sub(lambda _m: leading, s)

    def make(u: str, zone: _dt.tzinfo | None) -> Formatter:
        def fmt(x: Any) -> list[str]:
            dts = [_to_datetime(v, u, zone) for v in np.atleast_1d(np.asarray(x)).ravel()]
            ok = [d for d in dts if d is not None]
            midnight = all(getattr(d, "hour", 0) == 0 and getattr(d, "minute", 0) == 0 for d in ok)
            f_time: str | None = None if midnight else ft
            f_day: str | None = fd
            f_month: str | None = fm
            if midnight and all(d.day == 1 for d in ok):
                f_day = None
                if all(d.month == 1 for d in ok):
                    f_month = None
            out: list[str] = []
            prev: _dt.date | None = None
            for d in dts:
                if d is None:
                    out.append("")
                    prev = None
                    continue
                new_y = prev is None or d.year != prev.year
                new_m = new_y or d.month != prev.month  # type: ignore[union-attr]
                new_d = new_m or d.day != prev.day  # type: ignore[union-attr]
                parts = [
                    f_time,
                    f_day if new_d else None,
                    f_month if new_m else None,
                    fy if new_y else None,
                ]
                out.append(sep.join(strf(d, p) for p in parts if p))
                prev = d
            return out

        return fmt

    return DateFormatter(make, unit, tz)


def label_wrap(width: int) -> Formatter:
    """Wrap long (e.g. category) labels onto lines of at most ``width`` characters."""
    width = int(width)

    def fmt(x: Any) -> list[str]:
        return [textwrap.fill(str(s), width=width) if str(s) else "" for s in x]

    return fmt


# --------------------------------------------------------------------------- breaks
def _limits(limits: Sequence[float]) -> tuple[float, float]:
    lo, hi = float(np.min(limits)), float(np.max(limits))
    return lo, hi


def breaks_pretty(n: int = 5) -> BreakFunction:
    """Break function using R's :func:`~ggmanim.scales.breaks.pretty` (1-2-5 steps,
    covering the range) with about ``n`` intervals."""

    def brk(limits: Sequence[float]) -> np.ndarray:
        return pretty(*_limits(limits), n=n)

    return brk


def breaks_extended(n: int = 5, only_loose: bool = False, **kwargs: Any) -> BreakFunction:
    """Break function using Wilkinson's extended algorithm (ggplot2's default; see
    :func:`~ggmanim.scales.breaks.extended`) targeting ``n`` breaks."""

    def brk(limits: Sequence[float]) -> np.ndarray:
        lo, hi = _limits(limits)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return np.array([], dtype=float)
        return extended(lo, hi, m=n, only_loose=only_loose, **kwargs)

    return brk


def breaks_width(width: float, offset: float = 0.0) -> BreakFunction:
    """Break function with a break every ``width`` (aligned to multiples of
    ``width``, shifted by ``offset``)."""

    def brk(limits: Sequence[float]) -> np.ndarray:
        return width_breaks(*_limits(limits), width=width, offset=offset)

    return brk


def breaks_log(n: int = 5, base: float = 10) -> BreakFunction:
    """Break function for log scales: integer powers of ``base``, with 1-2-5 style
    sub-breaks when the range spans few powers (``scales::breaks_log``)."""

    def brk(limits: Sequence[float]) -> np.ndarray:
        lo, hi = _limits(limits)
        if not lo > 0:
            lo = hi * 1e-3 if hi > 0 else 1.0
        return log_breaks(lo, hi, n=n, base=base)

    return brk


# --------------------------------------------------------------------------- rendering
_SUPERSCRIPT = dict(
    zip(
        "0123456789+-=()ni",
        "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ",
    )
)
_TEX_SYMBOLS = {
    r"\times": "×",
    r"\cdot": "·",
    r"\pi": "π",
    r"\infty": "∞",
    r"\pm": "±",
    r"\,": " ",
    r"\ ": " ",
}
_SUP_RE = re.compile(r"\^(\{([^{}]*)\}|(.))")
_TEX_SPECIAL_RE = re.compile(r"(?<!\\)([$%&#_])")


def is_math(s: str) -> bool:
    """True for a ``"$...$"`` math label."""
    return len(s) >= 2 and s.startswith("$") and s.endswith("$") and s.count("$") == 2


def _superscript(m: re.Match) -> str:
    body = m.group(2) if m.group(2) is not None else m.group(3)
    body = body.replace("−", "-")
    if body and all(c in _SUPERSCRIPT for c in body):
        return "".join(_SUPERSCRIPT[c] for c in body)
    return "^" + (body if len(body) == 1 else f"({body})")


def math_to_unicode(s: str) -> str:
    """Convert a simple ``"$...$"`` math label to plain text: ``^{..}`` becomes Unicode
    superscripts (``"$10^{-3}$"`` -> ``"10⁻³"``), a few symbols (``\\times``,
    ``\\cdot``, ``\\pi`` ...) their characters; other markup is left as is."""
    body = s[1:-1] if is_math(s) else s
    for tex, uni in _TEX_SYMBOLS.items():
        body = body.replace(tex, uni)
    body = _SUP_RE.sub(_superscript, body)
    return body.replace("{", "").replace("}", "")


def render_label(s: str, engine: str) -> str:
    """Prepare a label string for ``theme.text_engine``.

    ``"text"``: ``"$...$"`` math is converted with :func:`math_to_unicode`; other
    strings are unchanged. ``"tex"``: math strings pass through; in other strings the
    LaTeX specials ``$ % & # _`` are escaped and newlines become ``\\\\`` line breaks.
    """
    if engine == "tex":
        if is_math(s):
            return s
        return _TEX_SPECIAL_RE.sub(r"\\\1", s).replace("\n", r"\\")
    if is_math(s):
        return math_to_unicode(s)
    return s
