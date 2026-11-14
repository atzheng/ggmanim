"""``plot._load_components`` makes chaining independent of import order (ggm-1ec).

Each check runs in a fresh interpreter so earlier imports in the test session
cannot mask a missing module in ``_load_components``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def _run(code: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_load_components_imports_positions_and_facets():
    # Evict the positions/facets packages (and anything they registered), then
    # let the chaining fallback reload them: they must come back, and every
    # position_*/facet_* component they register must be chainable.
    res = _run(
        """
        import json, sys
        from ggmanim.plot import ggplot, _COMPONENT_MODULES
        from ggmanim.registry import COMPONENTS

        pkgs = ("ggmanim.positions", "ggmanim.facets")
        for m in [m for m in sys.modules if m.split(".")[:2] in [p.split(".") for p in pkgs]]:
            del sys.modules[m]
        for k in [k for k, v in COMPONENTS.items()
                  if getattr(v, "__module__", "").startswith(pkgs)]:
            del COMPONENTS[k]

        try:
            ggplot().no_such_component_xyz
        except AttributeError:
            pass

        keys = [k for k in COMPONENTS if k.startswith(("position_", "facet_"))]
        print(json.dumps({
            "listed": [m for m in ("positions", "facets") if m in _COMPONENT_MODULES],
            "loaded": [p for p in pkgs if p in sys.modules],
            "chainable": {k: callable(getattr(ggplot(), k)) for k in keys},
        }))
        """
    )
    assert res["listed"] == ["positions", "facets"]
    assert res["loaded"] == ["ggmanim.positions", "ggmanim.facets"]
    assert all(res["chainable"].values()), res["chainable"]


def test_every_component_chainable_after_importing_only_plot():
    full = _run(
        """
        import json
        import ggmanim
        from ggmanim.registry import COMPONENTS
        print(json.dumps(sorted(COMPONENTS)))
        """
    )
    res = _run(
        """
        import json
        from ggmanim.plot import ggplot
        from ggmanim.registry import COMPONENTS

        p = ggplot()
        bad = [k for k in sorted(COMPONENTS) if not callable(getattr(p, k, None))]
        print(json.dumps({"keys": sorted(COMPONENTS), "bad": bad}))
        """
    )
    assert full, "no components registered"
    assert set(full) <= set(res["keys"])
    assert res["bad"] == []


def test_component_modules_cover_every_registered_component():
    # A component defined in a module that _load_components never imports would
    # only be chainable by accident of import order.
    res = _run(
        """
        import json
        import ggmanim
        from ggmanim.plot import _COMPONENT_MODULES
        from ggmanim.registry import COMPONENTS

        roots = tuple("ggmanim." + m for m in _COMPONENT_MODULES) + ("ggmanim.plot",)
        uncovered = {
            k: v.__module__ for k, v in COMPONENTS.items()
            if not any(v.__module__ == r or v.__module__.startswith(r + ".") for r in roots)
        }
        print(json.dumps(uncovered))
        """
    )
    assert res == {}
