"""The docs site's code: self-contained python blocks run, included examples exist.

A block that starts with an import is a complete snippet and is executed on its
own (building plots, not rendering). Other blocks are fragments. Blocks that
include a file with ``--8<-- "path"`` (pymdownx.snippets) must name an existing
file; ``mkdocs build --strict`` checks the same, this catches it without mkdocs.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted((ROOT / "docs").rglob("*.md"))
BLOCK = re.compile(r"^```python\n(.*?)^```", flags=re.S | re.M)
INCLUDE = re.compile(r'^--8<-- "([^"]+)"', flags=re.M)


def _blocks():
    for page in PAGES:
        for i, code in enumerate(BLOCK.findall(page.read_text())):
            yield page.relative_to(ROOT), i, code


SNIPPETS = [
    pytest.param(code, id=f"{page}[{i}]")
    for page, i, code in _blocks()
    if "--8<--" not in code and code.lstrip().startswith(("import ", "from "))
]


def test_docs_have_snippets():
    assert PAGES
    assert len(SNIPPETS) >= 10


@pytest.mark.parametrize("code", SNIPPETS)
def test_docs_snippet_runs(code):
    exec(compile(code, "<docs snippet>", "exec"), {"__name__": "docs_snippet"})


def test_docs_includes_exist():
    included = [p for page in PAGES for p in INCLUDE.findall(page.read_text())]
    assert included
    for path in included:
        assert (ROOT / path).is_file(), path
