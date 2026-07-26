# ADR 0021 - local freshness gate for snapshot reports

**Status:** merged into ADR 0022, docs/decisions/0022-ci-gates.md (2026-07-26)

## Context

The durable fact - a repo-local pre-push hook is dead under a global core.hooksPath - is a bullet there. The rest is the docstring of tests/structure/test_report_freshness.py.

## Decision

Withdrawn as a standing record. The content lives at `docs/decisions/0022-ci-gates.md`; this number
is retained so existing `ADR 0021` citations still resolve.

## Consequences

Nothing changes mechanically - see ADR 0023 for the consolidation that produced
this stub, and `docs/decisions/README.md` for where each topic now lives.
