"""guide_axis(check_overlap=, n_dodge=) on position scales (ggm-2fa)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ggmanim import (
    aes,
    geom_col,
    geom_point,
    ggplot,
    guide_axis,
    scale_x_continuous,
    scale_x_discrete,
    scale_y_discrete,
)
from ggmanim.guides_axis import axis_label_priority, check_axis_overlap

DF = pd.DataFrame({"g": [f"category-{i:02d}" for i in range(12)], "v": np.arange(12.0)})


def _plot(**kw):
    return ggplot(DF, aes("g", "v")).add(geom_col(), scale_x_discrete(**kw))


def _labels(plot):
    return list(plot.panel.x_tick_labels)


def _shown(plot):
    return [len(lab.get_all_points()) > 0 for lab in _labels(plot)]


def test_priority_order():
    assert axis_label_priority(0) == []
    assert axis_label_priority(1) == [0]
    assert axis_label_priority(2) == [0, 1]
    assert axis_label_priority(5) == [0, 4, 2, 1, 3]
    assert sorted(axis_label_priority(11)) == list(range(11))
    assert axis_label_priority(11)[:3] == [0, 10, 5]


def test_check_axis_overlap_keeps_ends():
    boxes = [(i, 0, i + 1.5, 1) for i in range(5)]  # neighbours overlap
    assert check_axis_overlap(boxes) == [True, False, True, False, True]
    assert check_axis_overlap([None, (0, 0, 1, 1)]) == [True, True]
    assert all(check_axis_overlap([(i, 0, i + 1, 1) for i in range(4)]))  # touching ok


def test_guide_axis_validation():
    g = guide_axis(check_overlap=True, n_dodge=2)
    assert (g.check_overlap, g.n_dodge) == (True, 2)
    d = guide_axis(**{"check.overlap": True, "n.dodge": 3})
    assert (d.check_overlap, d.n_dodge) == (True, 3)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            guide_axis(n_dodge=bad)
    with pytest.raises(TypeError):
        guide_axis(foo=1)
    with pytest.raises(ValueError):
        scale_x_discrete(guide="legend")
    assert scale_x_discrete(guide="axis").guide is None


def test_default_warns_and_check_overlap_drops():
    with pytest.warns(UserWarning, match="discrete x-axis labels overlap"):
        _plot().build()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot = _plot(guide=guide_axis(check_overlap=True)).build()
    shown = _shown(plot)
    assert len(shown) == 12 and shown[0] and shown[-1]
    assert 2 <= sum(shown) < 12


def test_n_dodge_stacks_rows():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot = _plot(guide=guide_axis(n_dodge=2)).build()
    labs = _labels(plot)
    assert len(labs) == 12 and all(_shown(plot))
    tops = [lab.get_top()[1] for lab in labs]
    assert tops[0] == pytest.approx(tops[2])
    assert tops[1] < tops[0] - 0.1  # odd labels sit one row lower
    assert plot.panel.x_title.get_top()[1] < min(lab.get_bottom()[1] for lab in labs)


def test_continuous_and_y_guides():
    df = pd.DataFrame({"x": np.linspace(0, 1, 5), "y": np.linspace(0, 1, 5)})
    p = ggplot(df, aes("x", "y")).add(geom_point(), scale_x_continuous(guide=guide_axis(n_dodge=2)))
    p.build()
    assert scale_y_discrete(guide=guide_axis(check_overlap=True)).guide.check_overlap
