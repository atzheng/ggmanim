"""Statistical transformations (pure pandas/numpy; no manim imports).

One public module per stat; the export block below is generated from each
module's ``__all__`` by ``scripts/gen_exports.py``. :data:`STATS` (used by
:func:`get_stat` to resolve ``stat="smooth"``) is derived from the exported
``Stat*`` classes, so a new stat module needs no edit here.
"""

import re as _re

# -- generated exports (scripts/gen_exports.py) --
from .base import GROUPING_AES, NO_GROUP, Stat, add_group, get_stat, is_discrete, scale_limits
from .function import StatFunction, evaluate_function, stat_function
from .identity import StatIdentity, stat_identity
from .smooth import StatSmooth, stat_smooth, t_quantile

__all__ = [
    "GROUPING_AES",
    "NO_GROUP",
    "Stat",
    "StatFunction",
    "StatIdentity",
    "StatSmooth",
    "add_group",
    "evaluate_function",
    "get_stat",
    "is_discrete",
    "scale_limits",
    "stat_function",
    "stat_identity",
    "stat_smooth",
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
