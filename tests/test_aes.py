import pickle

import numpy as np
import pandas as pd
import pytest

from ggmanim.aes import Aes, AfterStat, aes, after_stat, evaluate_aes


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "g": list("xyx")})


def test_positional_x_y():
    m = aes("a", "b")
    assert dict(m) == {"x": "a", "y": "b"}


def test_omitted_xy_absent():
    assert "x" not in aes(color="g")
    assert len(aes()) == 0


@pytest.mark.parametrize(
    "alias,canon",
    [
        ("colour", "color"),
        ("col", "color"),
        ("fill_color", "fill"),
        ("fill_colour", "fill"),
        ("pch", "shape"),
        ("lty", "linetype"),
        ("cex", "size"),
        ("lwd", "linewidth"),
    ],
)
def test_alias_normalisation(alias, canon):
    m = aes(**{alias: "g"})
    assert dict(m) == {canon: "g"}
    assert m[alias] == "g"  # lookup also normalises
    assert alias in m


def test_passthrough_aesthetics():
    m = aes(fill="g", size="a", alpha="b", shape="g", linetype="g", group="g")
    assert set(m) == {"fill", "size", "alpha", "shape", "linetype", "group"}


def test_duplicate_alias_rejected():
    with pytest.raises(ValueError):
        aes(color="a", colour="b")


def test_immutable():
    m = aes("a")
    with pytest.raises(AttributeError):
        m.foo = 1
    with pytest.raises(TypeError):
        m["x"] = "b"


def test_merge_override_and_unset():
    base = aes("a", "b", color="g")
    merged = base.merge(aes(y="a", colour=None))
    assert dict(merged) == {"x": "a", "y": "a"}
    assert dict(base) == {"x": "a", "y": "b", "color": "g"}
    assert base | {"size": "a"} == aes("a", "b", color="g", size="a")


def test_equality_with_dict():
    assert aes("a", colour="g") == {"x": "a", "color": "g"}


def test_evaluate_columns_callables_constants(df):
    m = aes("a", lambda d: d.a + d.b, color="g", size=3, label=after_stat("count"))
    out = evaluate_aes(m, df)
    assert list(out.columns) == ["x", "y", "color", "size"]
    np.testing.assert_array_equal(out["y"], [5.0, 7.0, 9.0])
    assert (out["size"] == 3).all()
    assert out.index.equals(df.index)


def test_evaluate_missing_column(df):
    with pytest.raises(KeyError, match="nope"):
        evaluate_aes(aes("nope"), df)


def test_evaluate_without_data():
    out = evaluate_aes(aes(x=1.5), None)
    assert out["x"].tolist() == [1.5]


def test_after_stat_split():
    m = aes("a", y=after_stat("count"))
    assert dict(m.after_stat()) == {"y": AfterStat("count")}
    assert dict(m.before_stat()) == {"x": "a"}
    assert repr(after_stat("count")) == "after_stat('count')"


def test_pickle_roundtrip():
    m = aes("a", colour="g")
    assert pickle.loads(pickle.dumps(m)) == m
    assert isinstance(pickle.loads(pickle.dumps(m)), Aes)
