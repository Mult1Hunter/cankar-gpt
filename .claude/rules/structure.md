---
paths:
  - "tests/structure/**"
  - "cankar/core/paths.py"
  - "pyproject.toml"
---

# Structure law (ADR 0007)

You are editing the layout's enforcement surface. Read
`docs/decisions/0007-structure-law.md` before changing any of it.

- **There is no `scripts/` directory, ever.** All logic is importable
  `cankar/<stage>/` modules; each stage owns one `cli.py`; the single entry point
  is `uv run cankar <stage> <command>`.
- `cankar/core/paths.py` is the ONLY place artifact paths are defined. A relative
  f-string path anywhere else is a bug.
- The root allowlist and the stage tuple live in `tests/structure/test_layout.py`.
  Changing them is a conscious act that cites an ADR in the same PR - **usually an
  existing one**; a new record only if the reason for the entry is itself a
  decision (ADR 0023).
- `pyproject.toml` holds the import-linter contracts and mypy strictness. Loosening
  either is an architecture change, not config tidying.
