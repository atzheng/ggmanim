"""Execute the README's python code blocks (building plots, not rendering)."""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCKS = re.findall(r"^```python\n(.*?)^```", README.read_text(), flags=re.S | re.M)


def test_readme_has_examples():
    assert len(BLOCKS) >= 3


def test_readme_blocks_run():
    # Blocks run in order in one namespace (later snippets may reuse earlier ones),
    # like a reader pasting them into one session.
    ns: dict = {"__name__": "readme"}
    for i, code in enumerate(BLOCKS):
        try:
            exec(compile(code, f"README.md[block {i}]", "exec"), ns)
        except Exception as e:  # pragma: no cover - failure path
            pytest.fail(f"README python block {i} failed: {type(e).__name__}: {e}\n{code}")
    from ggmanim.mobjects import GGPlot

    assert isinstance(ns["plot"], GGPlot)
    assert "DoseResponse" in ns  # the example Scene is defined
