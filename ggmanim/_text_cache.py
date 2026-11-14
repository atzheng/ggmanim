"""Memoised text mobjects for :meth:`ggmanim.theme.Theme.make_text`.

Laying out text is the most expensive part of building a plot's decoration: every
``manim.Text``/``Paragraph``/``Tex`` writes (or finds in manim's file cache) an SVG
and then parses it into mobjects. Plots repeat the same strings (tick labels shared
by facet panels, legend keys, a plot rebuilt for another scene), so
:func:`cached_text` keeps one *prototype* per distinct request and hands out
a copy of the prototype (:func:`fast_copy`, a deep copy several times faster than
``Mobject.copy``), so callers may move, rotate or recolour what they get without
touching the prototype.

The key is everything that changes the result: the engine, the text, the font size,
the colour (as ``#RRGGBBAA``), the alignment, the remaining keyword arguments and the
parts of manim's ``config`` that text depends on (renderer, and the TeX template for
the ``"tex"`` engine). Requests whose keyword arguments or colour cannot be hashed
are built directly, uncached. The cache is a bounded LRU (:data:`MAXSIZE` entries);
:func:`clear_text_cache` empties it.
"""

from __future__ import annotations

import copy
import enum
from collections.abc import Callable, Hashable
from functools import lru_cache
from pathlib import PurePath
from typing import Any

import numpy as np
import svgelements as se
from manim import ManimColor, Mobject, config

__all__: list[str] = []  # internal helpers; nothing re-exported

#: maximum number of prototypes kept
MAXSIZE = 2048


class _Spec:
    """A cache key that also carries the factory building the prototype (the
    factory takes no part in hashing or equality)."""

    __slots__ = ("build", "key")

    def __init__(self, key: tuple, build: Callable[[], Mobject]):
        self.key = key
        self.build = build

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Spec) and other.key == self.key


@lru_cache(maxsize=MAXSIZE)
def _prototype(spec: _Spec) -> Mobject:
    return spec.build()


def _color_key(color: Any) -> Hashable:
    try:
        return ManimColor(color).to_hex(with_alpha=True)
    except Exception:  # noqa: BLE001 -- anything ManimColor rejects: let manim decide
        return None


def _config_key(engine: str) -> tuple:
    renderer = str(config.renderer)
    if engine == "tex":
        template = config.tex_template
        return (renderer, template.tex_compiler, template.output_format, template.body)
    return (renderer,)


def text_key(
    engine: str, text: str, font_size: float, color: Any, align: str, kwargs: dict[str, Any]
) -> tuple | None:
    """The cache key for one :meth:`~ggmanim.theme.Theme.make_text` request, or
    ``None`` when the request cannot be cached (unhashable arguments)."""
    color_key = _color_key(color)
    if color_key is None:
        return None
    items = tuple(sorted(kwargs.items()))
    key = (engine, text, float(font_size), color_key, align, items, _config_key(engine))
    try:
        hash(key)
    except TypeError:
        return None
    return key


def cached_text(key: tuple | None, build: Callable[[], Mobject]) -> Mobject:
    """A copy of the prototype ``build()`` makes for ``key`` (built on first use);
    ``key=None`` builds directly."""
    if key is None:
        return build()
    return fast_copy(_prototype(_Spec(key, build)))


#: values shared (not copied) by :func:`fast_copy`: immutable, or treated as values
#: (colours, which manim never changes in place, and the parsed SVG paths a text's
#: glyphs keep after generating their points from them)
_ATOMIC = (
    str,
    bytes,
    int,
    float,
    complex,
    bool,
    type(None),
    enum.Enum,
    PurePath,
    ManimColor,
    se.SVGElement,
)

_SHARED_TYPES = frozenset((str, int, float, bool, type(None), ManimColor))


def fast_copy(mob: Mobject) -> Mobject:
    """A deep copy of ``mob`` equivalent to ``mob.copy()`` for text mobjects, without
    the generic ``copy.deepcopy`` machinery: mobjects are cloned attribute by
    attribute (each clone once, so shared references stay shared), arrays, lists,
    tuples and dicts are copied, the values in :data:`_ATOMIC` are shared;
    anything else falls back to ``copy.deepcopy``."""
    memo: dict[int, Any] = {}

    def conv(v: Any) -> Any:
        t = type(v)
        if t in _SHARED_TYPES:  # fast path for the common exact types
            return v
        if isinstance(v, _ATOMIC):
            return v
        if isinstance(v, np.ndarray):
            return v.copy()
        if isinstance(v, Mobject):
            return clone(v)
        if type(v) is list:
            return [conv(x) for x in v]
        if type(v) is dict:
            return {k: conv(x) for k, x in v.items()}
        if type(v) is tuple:
            return tuple(conv(x) for x in v)
        return copy.deepcopy(v, memo)

    def clone(m: Mobject) -> Mobject:
        done = memo.get(id(m))
        if done is not None:
            return done
        cls = m.__class__
        new = cls.__new__(cls)
        memo[id(m)] = new
        attrs = m.__dict__.copy()
        for k, v in attrs.items():
            if type(v) not in _SHARED_TYPES:
                attrs[k] = conv(v)
        new.__dict__.update(attrs)
        new.original_id = str(id(m))  # as Mobject.__deepcopy__ sets it
        return new

    return clone(mob)


def clear_text_cache() -> None:
    """Drop every cached prototype."""
    _prototype.cache_clear()


def text_cache_info():
    """``functools`` cache statistics (hits, misses, maxsize, currsize)."""
    return _prototype.cache_info()
