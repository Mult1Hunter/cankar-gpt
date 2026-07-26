---
paths:
  - "cankar/**/*.py"
  - "tests/**/*.py"
---

# Code standards (ADR 0008)

- Closed sets are `StrEnum`s, not bare string literals.
- Results are typed (dataclass / pydantic model), not `dict[str, Any]`.
- Library code raises domain errors and logs. Never `SystemExit`, never `print` -
  only `cankar/<stage>/cli.py` is allowed to exit and write to stdout.
- Configs are validated models, not raw dicts parsed inline.
- mypy + ruff + import-linter gate CI. `cankar/core/` is the bottom layer; stages
  import only `core` and stay independent of each other.
- Artifact paths come from `cankar/core/paths.py` - no relative f-string paths.

## Why (absorbed from the withdrawn ADR 0008)

A code audit found closed sets as runtime-validated string sets, results as
stringly-keyed dicts, library code raising `SystemExit` and printing to stderr,
untyped TOML access, and no static type checker. Two facts worth keeping:

- mypy's first run immediately found real shadowing - a `triage` parameter
  silently shadowed by a local list.
- StrEnum serialises to the same plain strings, so committed registry JSONL stayed
  byte-compatible through the migration (tested). That is why the closed-set rule
  cost nothing to adopt.
