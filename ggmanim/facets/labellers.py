"""Strip labellers: turn a panel's facet values into strip text.

A *labeller* is a callable ``labeller(values) -> list[str]`` where ``values`` is
an ordered ``{facet variable: value}`` dict for one strip (one panel side). It
returns one string per strip line; with ``multi_line=True`` (the default, as in
ggplot2) each variable gets its own strip, otherwise they are joined into one.
Every labeller here carries ``is_labeller = True`` so :func:`labeller` can tell a
labeller (takes the dict) from a per-value function (takes one value).

>>> label_value({"cyl": 4, "am": 1})
['4', '1']
>>> label_both({"cyl": 4})
['cyl: 4']
>>> labeller(cyl={4: "four"})({"cyl": 4})
['four']
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["label_value", "label_both", "labeller", "as_labeller"]
__toplevel__ = ["label_value", "label_both", "labeller"]

Labeller = Callable[[Mapping[str, Any]], list[str]]


def _fmt(value: Any) -> str:
    """``str(value)``, with integral floats shown without ``.0`` (``4.0`` -> ``"4"``)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _mark(fn: Callable) -> Callable:
    fn.is_labeller = True  # type: ignore[attr-defined]
    return fn


def label_value(values: Mapping[str, Any], multi_line: bool = True) -> list[str]:
    """The values themselves: ``{"cyl": 4}`` -> ``["4"]``. With
    ``multi_line=False`` several variables are joined with ``", "``."""
    out = [_fmt(v) for v in values.values()]
    return out if multi_line or len(out) <= 1 else [", ".join(out)]


def label_both(values: Mapping[str, Any], multi_line: bool = True, sep: str = ": ") -> list[str]:
    """Variable and value: ``{"cyl": 4}`` -> ``["cyl: 4"]``."""
    out = [f"{k}{sep}{_fmt(v)}" for k, v in values.items()]
    return out if multi_line or len(out) <= 1 else [", ".join(out)]


_mark(label_value)
_mark(label_both)


def _one(var: str, value: Any, how: Any) -> list[str]:
    if how is None:
        return label_value({var: value})
    if isinstance(how, Mapping):
        return [_fmt(how.get(value, value))]
    if getattr(how, "is_labeller", False):
        return list(how({var: value}))
    if callable(how):
        return [_fmt(how(value))]
    raise TypeError(
        f"labeller for {var!r} must be a dict, a function of one value or a labeller, got {how!r}"
    )


def labeller(*, default: Any = None, multi_line: bool = True, **mappings: Any) -> Labeller:
    """A labeller with per-variable rules, like ggplot2's ``labeller()``.

    Each keyword names a facet variable and gives a ``dict`` (value -> label;
    unknown values are shown as is), a function of one value returning the label,
    or another labeller (e.g. ``label_both``). Variables without a rule use
    ``default`` (a labeller, default :func:`label_value`). ``multi_line=False``
    joins the variables of one strip with ``", "``.

    >>> labeller(am={0: "auto", 1: "manual"}, cyl=label_both)({"cyl": 6, "am": 0})
    ['cyl: 6', 'auto']
    """
    fallback = default if default is not None else label_value

    def _labeller(values: Mapping[str, Any]) -> list[str]:
        out: list[str] = []
        for var, value in values.items():
            out.extend(_one(var, value, mappings.get(var, fallback)))
        return out if multi_line or len(out) <= 1 else [", ".join(out)]

    return _mark(_labeller)  # type: ignore[return-value]


def as_labeller(how: Any) -> Labeller:
    """Normalise a ``labeller=`` argument: ``None`` -> :func:`label_value`;
    ``"label_value"``/``"label_both"`` -> that labeller; a labeller is returned as
    is; a dict or one-value function applies to every variable."""
    if how is None:
        return label_value
    if isinstance(how, str):
        named = {"label_value": label_value, "label_both": label_both}
        if how not in named:
            raise ValueError(f"unknown labeller {how!r}; use one of {sorted(named)}")
        return named[how]
    if getattr(how, "is_labeller", False):
        return how
    if isinstance(how, Mapping) or callable(how):

        def _all(values: Mapping[str, Any]) -> list[str]:
            out: list[str] = []
            for var, value in values.items():
                out.extend(_one(var, value, how))
            return out

        return _mark(_all)  # type: ignore[return-value]
    raise TypeError(f"labeller must be a labeller, dict or function, got {how!r}")
