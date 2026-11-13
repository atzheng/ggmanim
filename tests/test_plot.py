import copy
import pickle

import pandas as pd
import pytest

from ggmanim import registry
from ggmanim.aes import aes
from ggmanim.layer import Layer
from ggmanim.plot import GGManim, Labels, ggplot, labs, xlab, ylab
from ggmanim.registry import COMPONENTS, register


# -- tiny stub components (real geoms/scales/themes live elsewhere) ---------


class StubGeom:
    pass


def _stub_layer(data=None, mapping=None, name=None, **params):
    return Layer(geom=StubGeom(), stat="identity", position="identity",
                 data=data, mapping=aes(**(mapping or {})), params=params, name=name)


class StubScale:
    def __init__(self, *aesthetics, tag=None):
        self.aesthetics = aesthetics
        self.tag = tag

    def add_to(self, plot):
        return plot._with_scale(self)


class StubTheme:
    def __init__(self, complete=False, **elements):
        self.complete = complete
        self.elements = elements

    def merge(self, other):
        return StubTheme(complete=self.complete, **{**self.elements, **other.elements})

    def add_to(self, plot):
        return plot._with_theme(self)


class StubCoord:
    def add_to(self, plot):
        return plot._with_coord(self)


class TwoLayers:
    def add_to(self, plot):
        return plot._with_layer(_stub_layer())._with_layer(_stub_layer())


@pytest.fixture
def stubs():
    """Register stub constructors for the duration of one test."""
    saved = dict(COMPONENTS)

    @register
    def geom_stub(data=None, mapping=None, **params):
        return _stub_layer(data=data, mapping=mapping, **params)

    @register(name="scale_x_stub")
    def _make_scale(tag=None):
        return StubScale("x", "xmin", "xmax", tag=tag)

    @register
    class theme_stub(StubTheme):
        pass

    yield
    COMPONENTS.clear()
    COMPONENTS.update(saved)


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "g": list("xyx")})


# -- registry ----------------------------------------------------------------


def test_register_returns_unchanged_and_stores(stubs):
    def geom_foo():
        return None

    assert register(geom_foo) is geom_foo
    assert COMPONENTS["geom_foo"] is geom_foo
    assert register(name="foo_alias")(geom_foo) is geom_foo
    assert COMPONENTS["foo_alias"] is geom_foo


def test_register_rejects_private_names():
    with pytest.raises(ValueError):
        register(name="_hidden")(lambda: None)


def test_labs_xlab_ylab_registered():
    for n in ("labs", "xlab", "ylab"):
        assert COMPONENTS[n] is {"labs": labs, "xlab": xlab, "ylab": ylab}[n]


def test_chaining_method_exists_for_every_registered_component(stubs):
    p = ggplot()
    for n in COMPONENTS:
        assert callable(getattr(p, n)), n
        assert n in dir(p)


def test_component_registered_after_plot_import_gets_method(stubs):
    p = ggplot()
    with pytest.raises(AttributeError):
        p.geom_late

    @register
    def geom_late(**kw):
        return _stub_layer(**kw)

    q = p.geom_late(alpha=0.5)
    assert len(q.layers) == 1 and q.layers[0].params == {"alpha": 0.5}


def test_unknown_and_private_attrs_raise():
    p = ggplot()
    with pytest.raises(AttributeError):
        p.not_a_component
    with pytest.raises(AttributeError):
        p.__array_interface__


def test_chaining_equals_add(stubs, df):
    p = ggplot(df, aes("a", "b"))
    via_chain = p.geom_stub(alpha=0.3)
    via_add = p.add(COMPONENTS["geom_stub"](alpha=0.3))
    assert len(via_chain.layers) == len(via_add.layers) == 1
    assert via_chain.layers[0].params == via_add.layers[0].params == {"alpha": 0.3}


# -- immutability / .add -----------------------------------------------------


def test_add_returns_new_and_leaves_original(stubs, df):
    p = ggplot(df, aes("a", "b"))
    q = p.add(_stub_layer(), StubScale("x"), StubCoord(), StubTheme(), labs(title="T"))
    assert q is not p
    assert p.layers == () and p.scales == () and p.coord is None and p.theme is None
    assert dict(p.labels) == {}
    assert len(q.layers) == 1 and len(q.scales) == 1
    assert isinstance(q.coord, StubCoord) and isinstance(q.theme, StubTheme)
    assert q.labels["title"] == "T"
    assert q.data is p.data  # shared by reference
    r = q.geom_stub()
    assert len(q.layers) == 1 and len(r.layers) == 2


def test_spec_is_immutable(df):
    p = ggplot(df)
    with pytest.raises(AttributeError):
        p.data = None
    with pytest.raises(AttributeError):
        p.layers = ()
    with pytest.raises(TypeError):
        p.labels["x"] = "nope"


def test_add_rejects_non_components():
    with pytest.raises(TypeError, match="add_to"):
        ggplot().add(42)


def test_copy_and_pickle(stubs, df):
    p = ggplot(df, aes("a", "b")).geom_stub(name="pts").labs(x="A")
    for q in (copy.copy(p), copy.deepcopy(p), pickle.loads(pickle.dumps(p))):
        assert isinstance(q, ggplot)
        assert q.mapping == p.mapping
        assert [l.name for l in q.layers] == ["pts"]
        assert q.labels["x"] == "A"


def test_ggplot_mapping_only_positional():
    p = ggplot(aes("a", "b"))
    assert p.data is None and p.mapping == aes("a", "b")


def test_ggplot_accepts_plain_dict_mapping(df):
    p = ggplot(df, {"x": "a", "colour": "g"})
    assert dict(p.mapping) == {"x": "a", "color": "g"}


def test_ggmanim_alias():
    assert GGManim is ggplot


# -- layers ------------------------------------------------------------------


def test_name_stored_on_layer_via_add(df):
    p = ggplot(df).add(_stub_layer(), name="pts")
    assert p.layers[0].name == "pts"
    assert p.layer("pts") is p.layers[0]


def test_name_via_chaining_and_legacy_layer_id(stubs, df):
    p = ggplot(df).geom_stub(name="pts").geom_stub(layer_id="line")
    assert [l.name for l in p.layers] == ["pts", "line"]
    assert all("name" not in l.params and "layer_id" not in l.params for l in p.layers)


def test_name_requires_single_layer():
    with pytest.raises(ValueError):
        ggplot().add(StubScale("x"), name="s")
    with pytest.raises(ValueError):
        ggplot().add(TwoLayers(), name="two")


def test_duplicate_layer_name_rejected():
    p = ggplot().add(_stub_layer(), name="pts")
    with pytest.raises(ValueError):
        p.add(_stub_layer(), name="pts")
    with pytest.raises(ValueError):
        p.add(_stub_layer(name="pts"))


def test_layer_data_override_with_dataframe(df):
    other = pd.DataFrame({"a": [10, 20], "b": [1, 1]})
    p = ggplot(df, aes("a", "b")).add(_stub_layer(), _stub_layer(data=other))
    inherited, overridden = p.layers
    # DataFrame truthiness must never be used (legacy bug: `data or self.data`)
    assert inherited.resolve_data(p.data) is df
    assert overridden.resolve_data(p.data) is other
    # an empty DataFrame is still an override
    empty = df.iloc[:0]
    assert _stub_layer(data=empty).resolve_data(df) is empty


def test_layer_data_callable(df):
    layer = _stub_layer(data=lambda d: d[d.a > 1])
    assert layer.resolve_data(df)["a"].tolist() == [2, 3]


def test_layer_mapping_inheritance():
    plot_map = aes("a", "b", colour="g")
    layer = Layer(mapping=aes(y="c", color=None, size="s"))
    assert dict(layer.resolve_mapping(plot_map)) == {"x": "a", "y": "c", "size": "s"}
    no_inherit = layer.replace(inherit_aes=False)
    assert dict(no_inherit.resolve_mapping(plot_map)) == {"y": "c", "size": "s"}


def test_layer_coerces_dict_mapping_and_defaults():
    layer = Layer(mapping={"colour": "g"})
    assert dict(layer.mapping) == {"color": "g"}
    assert Layer().params == {} and Layer().inherit_aes is True and Layer().name is None


# -- scales / theme / coord / labels -----------------------------------------


def test_scale_with_overlapping_aesthetics_replaces(stubs):
    p = ggplot().add(StubScale("x", tag=1), StubScale("color", "fill", tag=2))
    q = p.scale_x_stub(tag=3)  # covers x/xmin/xmax → replaces the x scale
    assert [s.tag for s in q.scales] == [2, 3]
    r = q.add(StubScale("fill", tag=4))
    assert [s.tag for s in r.scales] == [3, 4]
    assert r.get_scale("xmin").tag == 3
    assert r.get_scale("colour") is None
    assert r.get_scale("fill").tag == 4


def test_theme_merge_and_replace(stubs):
    p = ggplot().theme_stub(complete=True, a=1, b=1)
    q = p.add(StubTheme(b=2, c=3))
    assert q.theme.elements == {"a": 1, "b": 2, "c": 3} and q.theme.complete
    r = q.add(StubTheme(complete=True, z=0))
    assert r.theme.elements == {"z": 0}


def test_coord_replaced():
    c1, c2 = StubCoord(), StubCoord()
    assert ggplot().add(c1, c2).coord is c2


def test_labs_xlab_ylab():
    p = ggplot().labs(x="X", colour="Group", title="T").ylab("Y")
    assert dict(p.labels) == {"x": "X", "color": "Group", "title": "T", "y": "Y"}
    q = p.xlab("X2")
    assert q.labels["x"] == "X2" and p.labels["x"] == "X"
    assert isinstance(xlab("a"), Labels)


def test_legacy_style_chaining(df):
    # .xlab/.ylab available on the GGManim alias via the registry
    p = GGManim(df, aes("a", "b")).xlab("A").ylab("B")
    assert p.labels["x"] == "A" and p.labels["y"] == "B"


def test_registry_module_contract():
    assert isinstance(registry.COMPONENTS, dict)
