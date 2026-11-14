"""Statistical transformations (pure pandas/numpy; no manim imports).

One public module per stat; the export block below is generated from each
module's ``__all__`` by ``scripts/gen_exports.py``. :data:`STATS` (used by
:func:`get_stat` to resolve ``stat="smooth"``) is derived from the exported
``Stat*`` classes, so a new stat module needs no edit here.
"""

import re as _re

# -- generated exports (scripts/gen_exports.py) --
from .base import GROUPING_AES, NO_GROUP, Stat, add_group, get_stat, is_discrete, scale_limits
from .bin import StatBin, bin_breaks, bin_breaks_bins, bin_breaks_width, stat_bin
from .count import StatCount, stat_count
from .ecdf import StatEcdf, stat_ecdf
from .function import StatFunction, evaluate_function, stat_function
from .identity import StatIdentity, stat_identity
from .qq import StatQq, StatQqLine, ppoints, qnorm, stat_qq, stat_qq_line
from .smooth import StatSmooth, stat_smooth, t_quantile
from .sum import StatSum, stat_sum
from .summary import (
    StatSummary,
    StatSummaryBin,
    mean_cl_normal,
    mean_sdl,
    mean_se,
    median_hilow,
    stat_summary,
    stat_summary_bin,
    summary_function,
)
from .unique import StatUnique, stat_unique

__all__ = [
    "GROUPING_AES",
    "NO_GROUP",
    "Stat",
    "StatBin",
    "StatCount",
    "StatEcdf",
    "StatFunction",
    "StatIdentity",
    "StatQq",
    "StatQqLine",
    "StatSmooth",
    "StatSum",
    "StatSummary",
    "StatSummaryBin",
    "StatUnique",
    "add_group",
    "bin_breaks",
    "bin_breaks_bins",
    "bin_breaks_width",
    "evaluate_function",
    "get_stat",
    "is_discrete",
    "mean_cl_normal",
    "mean_sdl",
    "mean_se",
    "median_hilow",
    "ppoints",
    "qnorm",
    "scale_limits",
    "stat_bin",
    "stat_count",
    "stat_ecdf",
    "stat_function",
    "stat_identity",
    "stat_qq",
    "stat_qq_line",
    "stat_smooth",
    "stat_sum",
    "stat_summary",
    "stat_summary_bin",
    "stat_unique",
    "summary_function",
    "t_quantile",
]
# -- end generated --


def _stat_name(cls: type) -> str:
    """``StatSmooth`` -> ``"smooth"``, ``StatSummaryBin`` -> ``"summary_bin"``,
    ``StatBin2d`` -> ``"bin_2d"``; a class attribute ``name`` overrides it."""
    name = cls.__dict__.get("name")
    if isinstance(name, str):
        return name
    stem = cls.__name__[len("Stat") :]
    return _re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])", "_", stem).lower()


#: name -> Stat class, used by :func:`get_stat` to resolve ``stat="smooth"``.
STATS: dict[str, type[Stat]] = {
    _stat_name(_obj): _obj
    for _name, _obj in sorted(globals().items())
    if _name.startswith("Stat")
    and isinstance(_obj, type)
    and issubclass(_obj, Stat)
    and _obj is not Stat
}

__all__ += ["STATS"]
