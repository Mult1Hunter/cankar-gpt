# ADR 0020 - SessionStart hook keeps the graphify index fresh

**Status:** withdrawn - content moved, .claude/settings.json and the graphify section of CLAUDE.md (2026-07-26)

## Context

A six-line, non-fatal dev-tooling hook, trivially reversible, affecting no artifact. The settings file IS the decision. (The withdrawn text also carried an error: it claimed no ADR mentioned graphify, but ADR 0003 does.)

## Decision

Withdrawn as a standing record. The content lives at `.claude/settings.json and the graphify section of CLAUDE.md`; this number
is retained so existing `ADR 0020` citations still resolve.

## Consequences

Nothing changes mechanically - see ADR 0023 for the consolidation that produced
this stub, and `docs/decisions/README.md` for where each topic now lives.
