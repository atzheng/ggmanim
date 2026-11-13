"""Discrete (categorical) non-position scales: colour, fill, shape, linetype.

A discrete scale is *trained* on the values of a mapped column (possibly across
several layers), which fixes an ordered list of levels. ``map`` then turns data
values into palette values; because the level -> value lookup is fixed after
training, every layer sharing the scale gets the same colour for the same level.

Colour aesthetics map to ``ManimColor`` instances, ready to pass to manim as
``color=`` / ``stroke_color=`` (for ``color``) or ``fill_color=`` (for ``fill``),
see :data:`AES_TO_MANIM`.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from manim import ManimColor

from ..palettes import LINETYPES, SHAPES, Palette, as_manim_color, hue_pal, resolve_palette
from ..registry import register
from .base import Scale

__all__ = [
    "AES_TO_MANIM",
    "CATEGORICAL_AESTHETICS",
    "COLOUR_AESTHETICS",
    "ScaleDiscrete",
    "ScaleDiscreteIdentity",
    "ScaleDiscreteManual",
    "default_scale_for",
    "is_discrete",
    "normalise_aesthetic",
    "scale_color_discrete",
    "scale_color_identity",
    "scale_color_manual",
    "scale_colour_discrete",
    "scale_colour_identity",
    "scale_colour_manual",
    "scale_fill_discrete",
    "scale_fill_identity",
    "scale_fill_manual",
]

CATEGORICAL_AESTHETICS: tuple[str, ...] = ("color", "fill", "shape", "linetype")
"""Aesthetics that get a discrete scale by default when mapped to a non-numeric column."""

COLOUR_AESTHETICS: tuple[str, ...] = ("color", "fill")

AES_TO_MANIM: dict[str, str] = {"color": "color", "fill": "fill_color"}
"""Which manim mobject kwarg each colour aesthetic's mapped values feed."""

NA_COLOR = ManimColor("#7F7F7F")  # ggplot2's grey50

_AES_ALIASES = {"colour": "color", "col": "color"}


def normalise_aesthetic(aes: str) -> str:
    """Normalise British/short spellings (``colour``, ``col``) to ``color``."""
    return _AES_ALIASES.get(aes, aes)


def is_discrete(values: Any) -> bool:
    """True if ``values`` should be treated as categorical (non-numeric, bool or Categorical)."""
    if isinstance(values, pd.Categorical):
        return True
    if isinstance(values, (str, bytes)) or np.ndim(values) == 0:
        values = [values]
    s = values if isinstance(values, (pd.Series, pd.Index)) else pd.Series(list(values))
    dtype = s.dtype
    if dtype == np.dtype(object):
        dtype = pd.Series(list(s.dropna())).infer_objects().dtype
    if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(dtype):
        return True
    return not pd.api.types.is_numeric_dtype(dtype)


def _is_na(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _sorted_levels(levels: Iterable[Any]) -> list[Any]:
    levels = list(levels)
    try:
        return sorted(levels)
    except TypeError:  # mixed types: sort by string representation
        return sorted(levels, key=str)


def _to_python(v: Any) -> Any:
    """Unwrap numpy scalars so levels hash/compare like plain Python values."""
    return v.item() if isinstance(v, np.generic) else v


class ScaleDiscrete(Scale):
    """A discrete scale mapping levels of a categorical variable to palette values.

    Parameters
    ----------
    aesthetics:
        Aesthetic name(s) this scale applies to, e.g. ``"color"`` or ``("color", "fill")``.
    palette:
        Palette name, list of values, or callable ``n -> list`` (default: :func:`hue_pal`).
    name:
        Legend title (``None``: guides fall back to the mapping's label).
    breaks:
        Subset/order of levels shown in the legend (default: all levels).
    labels:
        Legend labels: list (parallel to breaks), dict level -> label, or callable.
    limits:
        Fixed list of levels; overrides training.
    cycle:
        Recycle palette entries if there are more levels than palette values
        (default: raise ``ValueError``).
    na_value:
        Value used for missing data and for values outside the levels.
    drop:
        Drop unused categories of a pandas Categorical when training.
    guide:
        ``"legend"`` or ``"none"``.
    """

    is_discrete = True

    def __init__(
        self,
        aesthetics: str | Sequence[str],
        palette: Palette | str | None = None,
        *,
        name: str | None = None,
        breaks: Sequence[Any] | None = None,
        labels: Sequence[str] | Mapping[Any, str] | Callable[[Any], str] | None = None,
        limits: Sequence[Any] | None = None,
        cycle: bool = False,
        na_value: Any = None,
        drop: bool = True,
        guide: str = "legend",
    ):
        if isinstance(aesthetics, str):
            aesthetics = (aesthetics,)
        self.aesthetics: tuple[str, ...] = tuple(normalise_aesthetic(a) for a in aesthetics)
        self.palette = hue_pal() if palette is None else palette
        self.name = name
        self.breaks = None if breaks is None else list(breaks)
        self.labels = labels
        self.limits = None if limits is None else list(limits)
        self.cycle = cycle
        self.drop = drop
        self.guide = guide
        self.is_colour = any(a in COLOUR_AESTHETICS for a in self.aesthetics)
        if na_value is None and self.is_colour:
            na_value = NA_COLOR
        self.na_value = self._coerce(na_value) if na_value is not None else None
        self.reset()

    # -- training ----------------------------------------------------------
    def reset(self) -> None:
        """Forget all trained levels."""
        self._levels: list[Any] = []
        self._ordered = False  # True once a Categorical fixed the order
        self._lookup: dict[Any, Any] | None = None

    def train(self, values: Any) -> None:
        """Add the unique non-missing levels of ``values`` to the scale.

        pandas Categorical inputs contribute their categories in category order;
        other inputs contribute their unique values, and the combined levels are
        sorted unless a Categorical has fixed the order.
        """
        self._lookup = None
        if isinstance(values, (pd.Series, pd.Index)) and isinstance(
            values.dtype, pd.CategoricalDtype
        ):
            values = values.array if isinstance(values, pd.Series) else values.values
        if isinstance(values, pd.Categorical):
            cats = values.categories
            if self.drop:
                present = set(values.dropna().unique())
                cats = [c for c in cats if c in present]
            self._extend(_to_python(c) for c in cats)
            self._ordered = True
            return

        if isinstance(values, (str, bytes)) or np.ndim(values) == 0:
            values = [values]
        uniq = pd.unique(
            pd.Series(list(values) if not isinstance(values, pd.Series) else values, dtype=object)
        )
        self._extend(_to_python(v) for v in uniq if not _is_na(v))
        if not self._ordered:
            self._levels = _sorted_levels(self._levels)

    def _extend(self, new: Iterable[Any]) -> None:
        seen = set(self._levels)
        for v in new:
            if v not in seen:
                self._levels.append(v)
                seen.add(v)

    # -- levels / guides ---------------------------------------------------
    def get_limits(self) -> list[Any]:
        """The ordered levels of the scale (user ``limits`` override training)."""
        return list(self.limits) if self.limits is not None else list(self._levels)

    def get_breaks(self) -> list[Any]:
        limits = self.get_limits()
        if self.breaks is None:
            return limits
        lim = set(limits)
        return [b for b in self.breaks if b in lim]

    def get_labels(self) -> list[str]:
        breaks = self.get_breaks()
        labels = self.labels
        if labels is None:
            return [str(b) for b in breaks]
        if callable(labels):
            return [str(labels(b)) for b in breaks]
        if isinstance(labels, Mapping):
            return [str(labels.get(b, b)) for b in breaks]
        labels = list(labels)
        if self.breaks is None:
            if len(labels) != len(breaks):
                raise ValueError(f"Got {len(labels)} labels for {len(breaks)} breaks")
            return [str(x) for x in labels]
        # labels are parallel to the user-supplied breaks
        by_break = dict(zip(self.breaks, labels))
        return [str(by_break.get(b, b)) for b in breaks]

    def get_palette_values(self) -> list[Any]:
        """Palette values in level order (parallel to :meth:`get_limits`)."""
        return [self._get_lookup()[lvl] for lvl in self.get_limits()]

    # -- mapping -----------------------------------------------------------
    def _coerce(self, value: Any) -> Any:
        return as_manim_color(value) if self.is_colour else value

    def _palette_for(self, levels: list[Any]) -> list[Any]:
        return resolve_palette(self.palette, len(levels), cycle=self.cycle)

    def _get_lookup(self) -> dict[Any, Any]:
        if self._lookup is None:
            levels = self.get_limits()
            values = self._palette_for(levels)
            self._lookup = {lvl: self._coerce(v) for lvl, v in zip(levels, values)}
        return self._lookup

    def map(self, values: Any) -> Any:
        """Map data values to palette values.

        Returns a ``pd.Series`` (same index) for Series input, otherwise a 1-d
        object ``np.ndarray``. Missing / unknown values map to ``na_value``.
        """
        lookup = self._get_lookup()
        index = values.index if isinstance(values, pd.Series) else None
        if isinstance(values, (str, bytes)) or np.ndim(values) == 0:
            values = [values]
        vals = list(values)
        out = np.empty(len(vals), dtype=object)
        for i, v in enumerate(vals):
            v = _to_python(v)
            try:
                out[i] = lookup.get(v, self.na_value)
            except TypeError:  # unhashable
                out[i] = self.na_value
        if index is not None:
            return pd.Series(out, index=index, dtype=object)
        return out

    # -- plumbing ----------------------------------------------------------
    def clone(self) -> ScaleDiscrete:
        """An untrained copy with the same configuration."""
        new = copy.copy(self)
        new.reset()
        return new

    def add_to(self, plot):
        return plot._with_scale(self)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(aesthetics={self.aesthetics!r}, levels={self.get_limits()!r})"
        )


class ScaleDiscreteManual(ScaleDiscrete):
    """Discrete scale with user-supplied values (dict level -> value, or list in level order)."""

    def __init__(self, aesthetics, values: Mapping[Any, Any] | Sequence[Any], **kwargs):
        if isinstance(values, Mapping):
            self.values: dict[Any, Any] | list[Any] = dict(values)
        else:
            self.values = list(values)
        super().__init__(aesthetics, palette=None, **kwargs)
        self.palette = self.values  # type: ignore[assignment]  # dict palettes handled in _palette_for

    def _palette_for(self, levels: list[Any]) -> list[Any]:
        if isinstance(self.values, dict):
            missing = [lvl for lvl in levels if lvl not in self.values]
            if missing:
                raise ValueError(
                    f"Manual scale for {self.aesthetics} has no value for level(s) {missing!r}. "
                    f"Supplied keys: {list(self.values)!r}"
                )
            return [self.values[lvl] for lvl in levels]
        if len(self.values) < len(levels) and not self.cycle:
            raise ValueError(
                f"Insufficient values in manual scale for {self.aesthetics}: "
                f"{len(levels)} needed but only {len(self.values)} provided."
            )
        return resolve_palette(self.values, len(levels), cycle=self.cycle)


class ScaleDiscreteIdentity(ScaleDiscrete):
    """Data values are already aesthetic values (e.g. a column of hex colours). No legend by default."""

    def __init__(self, aesthetics, *, guide: str = "none", **kwargs):
        super().__init__(aesthetics, guide=guide, **kwargs)

    def _get_lookup(self) -> dict[Any, Any]:
        if self._lookup is None:
            self._lookup = {lvl: self._coerce(lvl) for lvl in self.get_limits()}
        return self._lookup

    def map(self, values: Any) -> Any:
        index = values.index if isinstance(values, pd.Series) else None
        if isinstance(values, (str, bytes)) or np.ndim(values) == 0:
            values = [values]
        vals = list(values)
        out = np.empty(len(vals), dtype=object)
        for i, v in enumerate(vals):
            out[i] = self.na_value if _is_na(v) else self._coerce(_to_python(v))
        if index is not None:
            return pd.Series(out, index=index, dtype=object)
        return out


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------


def _alias(fn: Callable, name: str) -> Callable:
    def alias(*args, **kwargs):
        return fn(*args, **kwargs)

    alias.__name__ = alias.__qualname__ = name
    alias.__doc__ = fn.__doc__
    return register(alias)


@register
def scale_color_discrete(
    palette: Palette | str | None = None, *, aesthetics="color", **kwargs
) -> ScaleDiscrete:
    """Discrete colour scale. ``palette``: name, list, or callable ``n -> colours`` (default hue)."""
    return ScaleDiscrete(aesthetics, palette=palette, **kwargs)


@register
def scale_fill_discrete(
    palette: Palette | str | None = None, *, aesthetics="fill", **kwargs
) -> ScaleDiscrete:
    """Discrete fill scale. ``palette``: name, list, or callable ``n -> colours`` (default hue)."""
    return ScaleDiscrete(aesthetics, palette=palette, **kwargs)


@register
def scale_color_manual(values, *, aesthetics="color", **kwargs) -> ScaleDiscreteManual:
    """Colour scale with explicit ``values`` (dict level -> colour, or list in level order)."""
    return ScaleDiscreteManual(aesthetics, values=values, **kwargs)


@register
def scale_fill_manual(values, *, aesthetics="fill", **kwargs) -> ScaleDiscreteManual:
    """Fill scale with explicit ``values`` (dict level -> colour, or list in level order)."""
    return ScaleDiscreteManual(aesthetics, values=values, **kwargs)


@register
def scale_color_identity(*, aesthetics="color", **kwargs) -> ScaleDiscreteIdentity:
    """Use data values directly as colours (hex strings, names, ManimColor). No legend by default."""
    return ScaleDiscreteIdentity(aesthetics, **kwargs)


@register
def scale_fill_identity(*, aesthetics="fill", **kwargs) -> ScaleDiscreteIdentity:
    """Use data values directly as fill colours. No legend by default."""
    return ScaleDiscreteIdentity(aesthetics, **kwargs)


scale_colour_discrete = _alias(scale_color_discrete, "scale_colour_discrete")
scale_colour_manual = _alias(scale_color_manual, "scale_colour_manual")
scale_colour_identity = _alias(scale_color_identity, "scale_colour_identity")


# ---------------------------------------------------------------------------
# Defaults for build.py
# ---------------------------------------------------------------------------

_DEFAULT_PALETTES: dict[str, Palette] = {
    "color": hue_pal(),
    "fill": hue_pal(),
    "shape": SHAPES,
    "linetype": LINETYPES,
}


def default_scale_for(aesthetic: str, values: Any) -> ScaleDiscrete:
    """Default (untrained) scale for a categorical ``aesthetic`` mapped to ``values``.

    Raises ``NotImplementedError`` for numeric colour/fill (continuous gradients are
    Phase 2), ``TypeError`` for numeric shape/linetype, and ``ValueError`` for
    aesthetics that have no default discrete scale.
    """
    aes = normalise_aesthetic(aesthetic)
    if aes not in CATEGORICAL_AESTHETICS:
        raise ValueError(
            f"No default discrete scale for aesthetic {aesthetic!r}; "
            f"categorical aesthetics are {CATEGORICAL_AESTHETICS}."
        )
    if not is_discrete(values):
        if aes in COLOUR_AESTHETICS:
            raise NotImplementedError(
                f"Mapping a numeric column to {aes!r} needs a continuous colour scale "
                "(gradients are not implemented yet). Convert the column to a category, "
                "e.g. df[col].astype(str) or .astype('category'), or use "
                f"scale_{aes}_identity() if it already holds colours."
            )
        raise TypeError(
            f"A continuous (numeric) variable cannot be mapped to {aes!r}; "
            "convert it to a category first, e.g. df[col].astype('category')."
        )
    return ScaleDiscrete(aes, palette=_DEFAULT_PALETTES[aes])
