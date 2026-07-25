# ADR 0002 - public monorepo from day one

**Status:** accepted, 2026-07

## Context

The original plan (ROADMAP v3 appendix) kept the repo private until the Phase 2.5
TinyCankar milestone, then flipped it public. Separately, the project spans several
toolchains over its lifetime - a Python ML pipeline, a FastAPI serving app, an Astro
demo site, a Laravel orchestrator - and needed a decided organization before commit #1.

## Decision

**Public from commit #1**, on a personal account (github.com/Mult1Hunter/cankar-gpt),
as a **single monorepo**. Naming aligned to **cankar-gpt / CankarGPT** everywhere.
Layout is document-first - directories are created when their phase starts:

| Path | Contents | Arrives |
|---|---|---|
| `scripts/` | Python pipeline entry points | Phase 1 (now) |
| `cankar/` | shared Python package | ~Phase 2 |
| `apps/api` | FastAPI serving (joins a uv workspace) | Phase 7 |
| `apps/web` | Astro demo + browser ONNX (own `package.json`) | Phase 7.5 |
| `apps/orchestrator` | Laravel two-tier demo (own `composer.json`) | Phase 8 |

Per-language toolchains; no cross-language build system (nx/turbo). Guardrails:
gitleaks (pre-commit + CI on every PR and push to main), enforced commit types
(`FEAT/FIX/DATA/TRAIN/DOCS/CHORE`, uppercase, via commit-msg hook), PR-only `main` after the
initial commit, lean community kit (MIT LICENSE, SECURITY.md, CONTRIBUTING.md),
committed Claude skills (`commit`, `public-hygiene`, `adr`).

## Rationale

- The private->public flip was motivation scaffolding; public-from-day-one is a
  stronger commitment device and deletes a whole risk class - the "paranoid pass
  before the flip" becomes a routine pre-push check (`public-hygiene` skill).
- Empty skeleton dirs impress nobody; a documented target layout costs nothing.
- Solo repo: heavyweight monorepo tooling adds maintenance without payoff.

## Consequences

- No private staging area, ever: anything pushed is permanently public (forks and
  caches survive force-pushes). Secrets discipline starts at the first `git add`.
- ROADMAP Phase 2.5 "flip repo public" becomes "publish TinyCankar samples"
  (promotion, not exposure).
- PR-based flow adds ceremony for a solo dev - accepted in exchange for CI gating
  and reviewable history.

## Amendment (2026-07-25): renamed to cankar-gpt

The project shipped under a slug that transposed the final two letters of GPT - a
typo that had been present since commit #1 and, worse, mechanically ratified: the
brand, this ADR, and the `public-hygiene` name guard all enforced the wrong order.
The earlier note that "the naming was deliberate" was a rationalization of the slip.

Renamed to **cankar-gpt / CankarGPT** everywhere. Concretely: the GitHub repo was
renamed (the old URL redirects); every in-repo reference was updated in one pass;
the `public-hygiene` name-consistency guard was inverted to target the old
transposed spelling instead; and `pyproject.toml`'s distribution name moved to
`cankar-gpt`. The Python import package (`cankar`) and the `uv run cankar` entry
point are unchanged, so no code imports break. The private sibling folder was
renamed to `../cankar-gpt-meta` to stay consistent. Done before any public launch,
while the audience was still zero.
