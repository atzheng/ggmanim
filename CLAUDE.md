# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## Build & Test

```bash
uv venv .venv && uv pip install -p .venv/bin/python -e '.[dev]'   # or: pip install -e '.[dev]'
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/python -m pytest -q
.venv/bin/manim -ql -s examples/basic.py Basic                      # render one example
.venv/bin/python scripts/gen_exports.py            # regenerate export blocks (--check: verify)
MANIM=.venv/bin/manim scripts/render_examples.sh /tmp/media   # render every examples/*.py
```

All four must pass before a PR. Tests build mobjects without rendering; no LaTeX
is needed (text engine defaults to Pango `Text`).

## Architecture Overview

ggplot2's grammar on manim. The spec is immutable (`ggplot.add()` returns a new
plot); `build.py` is a pure pipeline spec → `GGPlot` mobject (see its module
docstring for the numbered steps). Components register with `registry.register`
and get chaining methods for free.

- `aes.py`, `plot.py`, `layer.py` — spec, mapping, layers
- `geoms/`, `stats/`, `positions/` — one module per geom/stat/position; `base.py` holds the contracts
  (`Geom` + `attach_marks`, `Stat`, `Position` + `get_position`/`POSITIONS`); `stats.STATS` is
  derived from the exported `Stat*` classes, `positions.POSITIONS` from `Position.name`
- `scales/` — pure numpy, no manim imports; `breaks.py`, `labels.py` for ticks
- `coords.py` — panel construction (`CoordCartesian`, flip, fixed)
- `facets/` — facet specs and panel layout (empty package until the facet beads)
- `guides*.py`, `theme.py`, `palettes.py` — legends, appearance, colours
- `mobjects.py` — `GGPlot` (the built result), `animate.py`/`live.py` — animation helpers
- `PLAN.md` — roadmap; beads carry the per-task detail (`bd show <id>`)
- `scripts/gen_exports.py` — writes the export blocks of `ggmanim/__init__.py` and the
  subpackage `__init__.py` files from each module's `__all__` (`--check` in tests/CI)
- `scripts/render_examples.sh` — renders every `examples/*.py` (`manim -ql -s -a`); used by CI

## Conventions & Patterns

- Follow ggplot2 names and argument semantics; `colour`/`color` both accepted.
- Scales/stats never import manim. Geoms draw through `coord.transform` only.
- Sizes are scene units; text goes through `theme.make_text`.
- Every geom's `draw_panel` returns a `VGroup` and calls `attach_marks` so
  animations can address marks by row/key.
- numpy-style docstrings; `ruff` (E,F,W,I,B,UP) and `mypy` clean.

## Parallel work rules

Several beads are worked at once on separate branches. To keep merges clean:

1. **One bead = one branch `bead/<id>` = one PR**, rebased on `main` before opening.
   Claim with `bd update <id> --claim`; close only after merge.
2. **Touch only the files your bead's DESIGN notes list** (`bd show <id>`). If you
   need a change elsewhere, make the smallest adapter in your own module and file
   a follow-up bead (`bd create ... --deps discovered-from:<id>`).
3. **Add, don't edit.** New geoms/stats/scales/positions/facets go in new modules
   with their own `__all__`. Never hand-edit the generated export blocks in
   `ggmanim/__init__.py` or subpackage `__init__.py` files: run
   `python scripts/gen_exports.py`. On a merge conflict there, re-run it.
4. **Region markers.** Shared files (`theme.py`, `geoms/base.py`, `mobjects.py`,
   `coords.py`, README cheat sheet) contain `# region: <name>` blocks. Add lines
   only inside the region named in your bead, at its end.
5. **`build.py` is owned by one bead at a time** (see the bead's DESIGN). Do not
   touch it otherwise.
6. **Tests and examples are per bead**: `tests/test_<feature>.py`,
   `examples/<feature>.py` (CI renders every `examples/*.py`). Edit an existing
   test only if your bead changes that behaviour, and say so in the PR.
7. **No drive-by changes**: no renames, reformatting, import reordering or
   docstring edits outside your hunks.
8. Record decisions and gotchas in the bead (`bd comment`), not in chat.
