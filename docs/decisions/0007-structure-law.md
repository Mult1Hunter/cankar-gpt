# ADR 0007 - structure law: no scripts, committed ledgers, mechanical layout

**Status:** accepted, 2026-07 (supersedes parts of 0002/0005; makes 0003's
provenance rule implementable)

## Context

A dual architect review (ML-platform lens + monorepo-governance lens) was
commissioned against the fear that ten phases of accretion end in slop. The
reviews converged independently on: keep stage subpackages, freeze the root
mechanically, add import contracts, split the registry by ownership. They also
found live violations: `scripts/corpus/` held a 346-line API client and a
subprocess-chained orchestrator in a directory whose contract said "thin";
shard manifests lived only inside gitignored `data/`, making ADR 0003's
"regenerate and diff" unimplementable - Phase 3 could never have proven what
it trained on.

## Decision

1. **`scripts/` is abolished, permanently.** All logic is importable
   `cankar/<stage>/` modules; each stage owns one `cli.py`; the single console
   entry is `cankar <stage> <command>` (`[project.scripts]`). nanochat's stage
   prefixes map as `base_train -> cankar train base`.
2. **Committed provenance ledgers.** `registry/` partitions by ownership:
   `works/` (human-curated, notes sacred), `datasets/` (machine-appended shard
   manifests - the committed baseline that makes diffing real), `reports/`
   (generated, GENERATED-marked, CI drift-checked), `runs/` from Phase 2.
3. **Path policy is code.** `cankar/core/paths.py` is the only place artifact
   locations are defined; relative f-string paths are banned.
4. **The structure is tested.** `tests/structure/test_layout.py` holds the
   root allowlist, the stage tuple, python-placement, banned junk names,
   force-add guards, GENERATED markers, and <=30-line directory READMEs.
   import-linter enforces layering (core at the bottom; stages independent).
   Changing structure = editing the law file + citing an ADR, in one PR.
5. **`ops/`** absorbs `bin/` (and later runpod/, deploy/): operated, never
   imported.
6. Scheduled arrivals (each with its allowlist edit + ADR note): `evalsets/`
   (Ph2.25), `apps/web` pulled forward to Ph4 for the MVP samples page,
   `cankar/model/` extracted stage-neutral at Ph3 (serving/export need the
   model without the training stack), `cankar/prompts/` package data with
   `lock.json` content-addressing at Ph5, torch-free-base CI from Ph3
   (heavy deps in uv dependency groups).

## Supersessions

- ADR 0002: the `scripts/` layout row is abolished; `apps/web` arrival moves
  7.5 -> 4; everything else stands.
- ADR 0005: fully absorbed here (2026-07-26). Its `scripts/<stage>/` mirror leg is
  replaced by per-stage `cli.py`; "registry/ stays at root" is refined into the
  ownership partition; "the root never grows" is promoted from prose to a failing
  test. Its one surviving unique decision: **one package, one venv, one lockfile** -
  subpackages rather than a uv workspace, because workspace packaging ceremony buys
  nothing until per-stage dependency bleed (torch) actually hurts. A split stays
  possible later, and this ADR's partition boundaries already match where it would
  cut.
- ADR 0003: the provenance rule gains its missing mechanism (committed
  `registry/datasets/`).

## Consequences

- Entry points change: `uv run cankar corpus ingest --all` (README updated).
- The repo's entropy surface is two files: the root allowlist and the stage
  tuple - both change only in reviewed diffs citing ADRs.
- Every line of pipeline logic is importable and testable; subprocess
  orchestration of our own code is extinct.
