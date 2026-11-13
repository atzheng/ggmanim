"""stat_identity: leaves the data unchanged."""

from __future__ import annotations

import pandas as pd

from .base import Stat

try:
    from ..registry import register
except ImportError:  # registry not available yet

    def register(f):
        return f


__all__ = ["StatIdentity", "stat_identity"]


class StatIdentity(Stat):
    """Passthrough stat; skips the per-group split entirely."""

    def compute_layer(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        return df.copy()

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        return df.copy()

    def compute_group(self, df: pd.DataFrame, scales=None) -> pd.DataFrame:
        return df.copy()


@register
def stat_identity(**params) -> StatIdentity:
    """Identity stat (no transformation)."""
    return StatIdentity(**params)
