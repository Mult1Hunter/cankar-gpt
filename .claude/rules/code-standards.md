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
