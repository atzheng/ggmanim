"""Stat base class and grouping helpers.

A *stat* transforms a layer's data (already resolved to aesthetic-named columns:
``x``, ``y``, ``color``, ...) into the data the geom draws. Stats are pure
pandas/numpy and never import manim.

Pipeline contract
-----------------
* The build pipeline first calls :func:`add_group` on the layer data so every row
  carries an integer ``group`` column.
* It then calls ``stat.compute_layer(df, scales, **stat.params)``, which splits by
  ``PANEL`` (if present) and calls :meth:`Stat.compute_panel` for each panel.
* ``compute_panel`` splits by ``group``, calls :meth:`Stat.compute_group` for each
  group and concatenates the results. Columns that are constant within a group
  (``group``, ``PANEL``, ``color``, ``fill``, ...) are carried over onto the
  group's output rows if ``compute_group`` did not produce them itself.

``scales`` is a dict ``aesthetic -> scale object``. Stats only ever ask a position
scale for ``scale.get_limits()`` (continuous limits in transformed space, or
``None`` if the scale is not trained / has no limits). It may be ``None`` or lack
keys; stats fall back to the data range.

Grouping convention
-------------------
Following ggplot2, ``group`` is an integer. Groups are numbered ``1..k`` in sorted
order of the interaction of the discrete grouping aesthetics; when there is no
grouping aesthetic every row gets ``NO_GROUP == -1``.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "Stat",
    "NO_GROUP",
    "GROUPING_AES",
    "add_group",
    "is_discrete",
    "scale_limits",
    "get_stat",
]

#: Group id used when the data has no grouping aesthetic (ggplot2's convention).
NO_GROUP = -1

#: Aesthetics that, when mapped to a discrete variable, define groups.
GROUPING_AES = ("color", "colour", "fill", "fill_color", "shape", "linetype")


def is_discrete(s: pd.Series) -> bool:
    """True for categorical / string / object / boolean series."""
    return (
        isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_bool_dtype(s)
    )


def add_group(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with an integer ``group`` column.

    * If ``df`` already has a ``group`` column it is converted to integer ids:
      integer-valued columns are kept as is, anything else is re-coded ``1..k``
      in sorted order.
    * Otherwise groups are the interaction of the discrete aesthetics in
      :data:`GROUPING_AES` that are present, numbered ``1..k`` in sorted order.
    * With no grouping aesthetic every row gets :data:`NO_GROUP` (``-1``).
    """
    out = df.copy()
    if out.empty:
        out["group"] = pd.Series(dtype="int64")
        return out

    if "group" in out.columns:
        g = out["group"]
        if pd.api.types.is_integer_dtype(g) and not g.isna().any():
            out["group"] = g.astype("int64")
        else:
            codes, _ = pd.factorize(g, sort=True, use_na_sentinel=False)
            out["group"] = codes.astype("int64") + 1
        return out

    keys = [a for a in GROUPING_AES if a in out.columns and is_discrete(out[a])]
    if not keys:
        out["group"] = NO_GROUP
        return out

    ids = out.groupby(keys, sort=True, dropna=False, observed=True).ngroup()
    out["group"] = ids.to_numpy(dtype="int64") + 1
    return out


def scale_limits(scales: Mapping[str, Any] | None, aes: str):
    """``scales[aes].get_limits()`` as a float pair, or ``None`` if unavailable."""
    if not scales:
        return None
    scale = scales.get(aes)
    if scale is None or not hasattr(scale, "get_limits"):
        return None
    lim = scale.get_limits()
    if lim is None:
        return None
    try:
        lo, hi = (float(v) for v in lim)
    except (TypeError, ValueError):  # discrete limits, etc.
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return lo, hi


class Stat:
    """Base stat: subclasses override :meth:`compute_group` (or ``compute_panel``).

    Attributes
    ----------
    required_aes : tuple
        Aesthetics that must be present as columns in the layer data.
    default_aes : dict
        Default aesthetic values the stat contributes (may reference computed
        variables; interpreted by the layer/geom, not by the stat).
    params : dict
        Parameters given to the constructor; the layer passes them to
        :meth:`compute_layer` / :meth:`compute_panel`.
    """

    required_aes: tuple = ()
    default_aes: dict = {}
    #: Columns the stat drops from its output (not carried over from the input).
    dropped_aes: tuple = ()

    def __init__(self, **params):
        self.params = dict(params)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"

    # -- composition ---------------------------------------------------------
    def add_to(self, plot):
        raise NotImplementedError(
            f"{type(self).__name__} as a standalone layer requires a default geom; "
            "use geom_*(stat=...)"
        )

    # -- parameters ----------------------------------------------------------
    @classmethod
    def parameters(cls) -> set:
        """Names of keyword parameters accepted by ``compute_group``."""
        sig = inspect.signature(cls.compute_group)
        return {
            name
            for name, p in sig.parameters.items()
            if name not in ("self", "df", "data", "scales")
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }

    def _filter_params(self, params: Mapping[str, Any]) -> dict:
        allowed = self.parameters()
        return {k: v for k, v in params.items() if k in allowed}

    # -- checks --------------------------------------------------------------
    def check_required_aes(self, df: pd.DataFrame) -> None:
        missing = [a for a in self.required_aes if a not in df.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__} requires the following missing aesthetics: "
                + ", ".join(missing)
            )

    # -- computation ---------------------------------------------------------
    def compute(self, df: pd.DataFrame, scales=None) -> pd.DataFrame:
        """Convenience: ``add_group`` (if needed) + ``compute_layer`` with ``self.params``."""
        if df is None:
            df = pd.DataFrame()
        if "group" not in df.columns:
            df = add_group(df)
        return self.compute_layer(df, scales, **self.params)

    def compute_layer(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        """Split by ``PANEL`` (if present) and call :meth:`compute_panel`.

        Unknown params (e.g. geom params passed through by the layer) are ignored.
        """
        self.check_required_aes(df)
        params = self._filter_params(params)
        if "PANEL" not in df.columns:
            return self.compute_panel(df, scales, **params)
        pieces = []
        for panel, pdf in df.groupby("PANEL", sort=True, observed=True):
            res = self.compute_panel(pdf, scales, **params)
            if "PANEL" not in res.columns:
                res = res.assign(PANEL=panel)
            pieces.append(res)
        return _concat(pieces, df)

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        """Split by ``group``, compute each group, re-attach constant columns, concat."""
        if df.empty:
            return df.copy()
        if "group" not in df.columns:
            df = add_group(df)
        pieces = []
        for _, gdf in df.groupby("group", sort=True, observed=True):
            res = self.compute_group(gdf, scales, **params)
            if res is None or len(res) == 0:
                continue
            res = res.reset_index(drop=True)
            pieces.append(_carry_constant_columns(gdf, res, self.dropped_aes))
        return _concat(pieces, df)

    def compute_group(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        raise NotImplementedError


def _carry_constant_columns(src: pd.DataFrame, res: pd.DataFrame, dropped=()) -> pd.DataFrame:
    """Copy columns of ``src`` that are constant (within this group) onto ``res``."""
    extra = {}
    for col in src.columns:
        if col in res.columns or col in dropped:
            continue
        s = src[col]
        if s.nunique(dropna=False) == 1:
            extra[col] = s.iloc[0]
    if extra:
        res = res.assign(**{k: [v] * len(res) for k, v in extra.items()})
    return res


def _concat(pieces, template: pd.DataFrame) -> pd.DataFrame:
    if not pieces:
        return template.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def get_stat(stat, **params) -> Stat:
    """Resolve ``stat`` to a :class:`Stat` instance.

    Accepts a ``Stat`` instance (returned as is), a ``Stat`` subclass, or a name
    such as ``"smooth"`` / ``"stat_smooth"`` / ``"identity"``.
    """
    if isinstance(stat, Stat):
        return stat
    if isinstance(stat, type) and issubclass(stat, Stat):
        return stat(**params)
    if isinstance(stat, str):
        from . import STATS  # late import to avoid a cycle

        name = stat[5:] if stat.startswith("stat_") else stat
        try:
            return STATS[name](**params)
        except KeyError:
            raise ValueError(
                f"Unknown stat {stat!r}; available: {', '.join(sorted(STATS))}"
            ) from None
    raise TypeError(f"Cannot interpret {stat!r} as a stat")
