"""Aesthetic mappings.

``aes(x, y, **kwargs)`` maps aesthetic names to either a column name (``str``),
a callable ``f(df) -> Series/array`` or an :func:`after_stat` reference that is
evaluated on the stat's output. ggplot2 spellings are normalised
(``colour`` -> ``color``, ``fill_color`` -> ``fill``, ...).

An explicit ``None`` value means "unset this aesthetic": when a layer mapping is
merged over the plot mapping, ``aes(color=None)`` removes an inherited
``color``. Omitted ``x``/``y`` are simply absent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "AES_ALIASES",
    "Aes",
    "AfterStat",
    "aes",
    "after_stat",
    "evaluate_aes",
    "standardise_aes_name",
    "standardise_aes_names",
]

#: ggplot2 / R / manim spellings -> canonical aesthetic name.
AES_ALIASES: dict[str, str] = {
    "colour": "color",
    "col": "color",
    "fill_colour": "fill",
    "fill_color": "fill",
    "bg": "fill",
    "fg": "color",
    "pch": "shape",
    "cex": "size",
    "lty": "linetype",
    "lwd": "linewidth",
    "stroke_width": "linewidth",
    "opacity": "alpha",
}


def standardise_aes_name(name: str) -> str:
    """Return the canonical spelling of an aesthetic name."""
    return AES_ALIASES.get(name, name)


def standardise_aes_names(d: Mapping) -> dict[str, Any]:
    """Normalise every key of ``d``; raise if two spellings collide."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        canon = standardise_aes_name(k)
        if canon in out:
            raise ValueError(f"aesthetic {canon!r} given more than once (as {k!r})")
        out[canon] = v
    return out


class AfterStat:
    """Reference to a variable computed by the stat (e.g. ``after_stat("count")``)."""

    __slots__ = ("expr",)

    def __init__(self, expr: str | Callable):
        self.expr = expr

    def evaluate(self, data: pd.DataFrame):
        return _eval_one(self.expr, data)

    def __eq__(self, other):
        return isinstance(other, AfterStat) and other.expr == self.expr

    def __hash__(self):
        return hash(("AfterStat", self.expr))

    def __repr__(self):
        return f"after_stat({self.expr!r})"


def after_stat(expr: str | Callable) -> AfterStat:
    """Map an aesthetic to a stat-computed variable."""
    return AfterStat(expr)


class Aes(Mapping):
    """Immutable mapping aesthetic -> column name / callable / after_stat."""

    __slots__ = ("_d",)
    _d: dict[str, Any]

    def __init__(self, mapping: Mapping | None = None, **kwargs):
        d = standardise_aes_names(dict(mapping or {}))
        for k, v in standardise_aes_names(kwargs).items():
            if k in d:
                raise ValueError(f"aesthetic {k!r} given more than once")
            d[k] = v
        object.__setattr__(self, "_d", d)

    # Mapping protocol
    def __getitem__(self, key: str):
        return self._d[standardise_aes_name(key)]

    def __contains__(self, key) -> bool:
        return isinstance(key, str) and standardise_aes_name(key) in self._d

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __setattr__(self, name, value):
        raise AttributeError("Aes is immutable")

    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = other if isinstance(other, Aes) else Aes(other)
            return self._d == other._d
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __reduce__(self):
        return (Aes, (self._d,))

    def __repr__(self):
        inner = ", ".join(f"{k}={v!r}" for k, v in self._d.items())
        return f"aes({inner})"

    def merge(self, other: Mapping | None) -> Aes:
        """Return ``self`` updated with ``other`` (``other`` wins).

        A ``None`` value in ``other`` removes the aesthetic.
        """
        if other is None:
            return self
        other = other if isinstance(other, Aes) else Aes(other)
        d = dict(self._d)
        for k, v in other._d.items():
            if v is None:
                d.pop(k, None)
            else:
                d[k] = v
        return Aes(d)

    def __or__(self, other: Mapping) -> Aes:
        return self.merge(other)

    def defined(self) -> Aes:
        """Drop explicitly-unset (``None``) aesthetics."""
        return Aes({k: v for k, v in self._d.items() if v is not None})

    def after_stat(self) -> Aes:
        """Only the :class:`AfterStat` aesthetics."""
        return Aes({k: v for k, v in self._d.items() if isinstance(v, AfterStat)})

    def before_stat(self) -> Aes:
        """Aesthetics evaluated on the layer data (not after_stat, not None)."""
        return Aes(
            {k: v for k, v in self._d.items() if v is not None and not isinstance(v, AfterStat)}
        )

    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        return evaluate_aes(self, data)


_MISSING = object()


def aes(x: Any = _MISSING, y: Any = _MISSING, **kwargs) -> Aes:
    """Construct an aesthetic mapping.

    >>> aes("wt", "mpg", colour="cyl")
    aes(x='wt', y='mpg', color='cyl')
    """
    d: dict[str, Any] = {}
    if x is not _MISSING:
        d["x"] = x
    if y is not _MISSING:
        d["y"] = y
    return Aes(d, **kwargs)


def _eval_one(value: Any, data: pd.DataFrame):
    if isinstance(value, str):
        if value not in data.columns:
            raise KeyError(f"column {value!r} not found in data (columns: {list(data.columns)})")
        return data[value]
    if isinstance(value, AfterStat):
        return value.evaluate(data)
    if callable(value):
        return value(data)
    return value


def evaluate_aes(mapping: Mapping, data: pd.DataFrame | None) -> pd.DataFrame:
    """Evaluate the non-after_stat aesthetics of ``mapping`` against ``data``.

    Column names select columns, callables are called with the DataFrame and
    other values (numbers, arrays) are used as-is / broadcast. ``None`` and
    :class:`AfterStat` entries are skipped. Returns a DataFrame with one column
    per aesthetic, indexed like ``data``.
    """
    mapping = mapping if isinstance(mapping, Aes) else Aes(mapping)
    if data is None:
        data = pd.DataFrame()
    out: dict[str, Any] = {}
    for k, v in mapping.before_stat().items():
        val = _eval_one(v, data)
        if isinstance(val, pd.Series):
            val = val.reindex(data.index) if len(data.index) else val
        elif np.ndim(val) > 0:
            val = np.asarray(val)
            if len(data.index) and len(val) != len(data.index):
                raise ValueError(
                    f"aesthetic {k!r} has length {len(val)}, data has {len(data.index)} rows"
                )
        out[k] = val
    if not out:
        return pd.DataFrame(index=data.index)
    if len(data.index):
        return pd.DataFrame(out, index=data.index)
    if all(np.ndim(v) == 0 for v in out.values()):
        return pd.DataFrame({k: [v] for k, v in out.items()})
    return pd.DataFrame(out)
