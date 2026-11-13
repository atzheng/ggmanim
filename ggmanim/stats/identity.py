"""stat_identity: leaves the data unchanged."""

from __future__ import annotations

import pandas as pd

from ..registry import register
from .base import Stat

__all__ = ["StatIdentity", "stat_identity"]


class StatIdentity(Stat):
    """Passthrough stat; skips the per-group split entirely."""

    def compute_layer(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        return df.copy()

    def compute_panel(self, df: pd.DataFrame, scales=None, **params) -> pd.DataFrame:
        return df.copy()

    def compute_group(self, df: pd.DataFrame, scales=None) -> pd.DataFrame:  # type: ignore[override]
        return df.copy()


@register
def stat_identity(**params) -> StatIdentity:
    """Identity stat (no transformation)."""
    return StatIdentity(**params)
