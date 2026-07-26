# ADR 0020 - SessionStart hook keeps the graphify index fresh

**Status:** accepted, 2026-07-26

## Context

CLAUDE.md has told every session to run `graphify update .` after code changes
since Phase 1, and a `.claude` hook-guard was noted as "off until Phase 2". Phase
2 came and went; the guard was never built and the instruction was never followed.

Measured on 2026-07-26, at Phase 4: `graphify-out/graph.json` held **96 nodes and
0 edges**, indexing **18 files - none of the 83 Python modules** and only 3 of 19
ADRs. It still listed `scripts/crawl_wikivir.py`, a path the ADR 0007 restructure
deleted and whose directory the structure law now bans. Every code query returned
`No matching nodes found.`, so both the user and the agent stopped reaching for
the tool - correctly, since it was empty. A rebuild takes **1.65s** with no LLM
and no API key, and produced **1458 nodes / 3025 edges / 126 communities**; the
same failing query then resolved to `cankar/core/paths.py:13`.

This is the "rule without a check is a wish" failure mode from CLAUDE.md, caught
in our own repo: the instruction existed, the gate did not, and nothing noticed
for four months.

## Decision

A checked-in `SessionStart` hook in `.claude/settings.json` runs
`uv run --group tooling graphify update .` at every session start (startup,
resume, and clear), silenced and non-fatal:

```
uv run --group tooling graphify update . >/dev/null 2>&1 || true
```

## Rationale

- **SessionStart, not PostToolUse.** A per-edit rebuild would add 1.65s to every
  file write; once per session is invisible and no session can begin against a
  fossil index. Intra-session staleness is bounded by one turn's edits.
- **Silenced.** SessionStart stdout is injected into the context window. The
  rebuild's six status lines would be a recurring context tax for no decision
  value - the exact anti-pattern the same-day CLAUDE.md rightsize removed.
- **Non-fatal (`|| true`).** A missing `tooling` group or a graphify bug must
  never block a session from starting. The tool is an accelerator, not a gate.
- **Checked in, not local.** graphify is a declared project dependency
  (`dependency-groups.tooling`) and CLAUDE.md documents it as the first move for
  codebase questions; the gate that keeps it honest belongs with it. Contributors
  get it behind the workspace-trust dialog.
- **Known cost accepted:** community labels still need an LLM backend to refresh,
  so `cankar/evals/style.py` currently sits in a community named after a deleted
  file. Cosmetic; `graphify cluster-only .` fixes it when a backend is available.

## Consequences

- CLAUDE.md's graphify section cites this ADR instead of carrying a bare
  instruction; the stale "hook-guard off until Phase 2 (ADR 0003)" line is gone -
  it cited an ADR that never mentioned graphify.
- `.claude/settings.json` now exists as a checked-in file; project settings are
  reviewable like any other code.
- Lesson recorded (the ratchet): a deferred gate is an un-kept promise with a
  deadline nobody owns. When a rule is written with "guard off until phase N",
  the guard is part of phase N's definition of done.
