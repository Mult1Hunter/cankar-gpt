# ADR 0003 - validation ladder for agent-driven development

**Status:** accepted, 2026-07 (amended 2026-07-28 - PR granularity, register home)

## Context

Most implementation work in this repo is done by an AI agent (Claude Code), with the
maintainer acting as reviewing tech lead. Agent-produced work needs systematic
verification - human review alone doesn't scale, and "looks right" violates design
invariant #2 (eval before claims).

## Decision

Every change climbs a validation ladder as far as its risk demands:

| Level | Gate | Arrives |
|---|---|---|
| **L0** | mechanical: ruff, gitleaks, commit-msg hook, CI on PRs | now |
| **L1** | tests: pytest; golden-file tests for text cleaning (wikitext fixture -> expected text); property tests for the NFC invariant | first pipeline PR |
| **L2** | data contracts: pydantic models per JSONL stage (`CorpusDoc`, `Chunk`, `Pair`) + a validate step; stats-band checks (token counts inside expected ranges - "10x off means the listing is wrong") | Phase 1-2 |
| **L3** | eval gate: harness numbers (held-out perplexity, style classifier, meaning judge) required in any `TRAIN:` PR that claims quality | Phase 2.25 |
| **L4** | human: `main` moves only by a PR the maintainer merges; the agent never pushes to `main` (granularity: see Amendment) | now |

**Provenance rules** (invariants made assertable, not remembered):

- every dataset artifact ships a `MANIFEST.json`: generating script's git SHA, source
  snapshot date, content hash, doc/token counts - regenerate and diff to verify
- training/tokenizer configs are committed files, never CLI-only flags; seeds pinned
- the shared register prompt (invariant #1) lives in `cankar/core/register.py`; its
  content hash is stamped into the manifest of anything generated with it (see
  Amendment 2)

## Tooling adoption map

| Tool | When |
|---|---|
| Graphify (codebase knowledge graph: CLI + skill; hook-guard skipped, revisit Phase 2; no MCP in v0.9.x) | wired at scaffold; earns its keep from Phase 2 |
| CodeQL (python + actions) | active at repo creation |
| `corpus-qa` agent (`.claude/agents/`) | Phase 1, before the first crawl |
| `design-review` agent - senior-engineer pass on every PR diff | every PR (added 2026-07) |
| train-monitor watchdog (harness loop + notifications) | Phase 3 cloud runs |
| `prompts/` governance + pair-judge batch script (promptfoo if prompt iterations exceed ~3) | Phase 5 |
| `style-critic` agent - qualitative commentary only, barred from producing quality claims | Phase 6 |
| HF Hub MCP + `hf-publish` skill (model/dataset cards) | Phase 2.5+ |
| claude-code-action PR review; OpenSSF Scorecard | trial after first PRs |

## Rationale

- An invariant you can `assert` beats one you remember; gates catch what review skims.
- Data-pipeline bugs are silent - golden files and contracts make them loud.
- The ladder is phase-matched: no gate exists before the thing it validates does.

## Consequences

- PRs carry ceremony (tests, manifests) - accepted; it is the trust substrate that
  makes agent-written code mergeable on sight.
- Each phase start includes a "which gates activate now" check against this ADR.

## Amendment (2026-07-28) - L4 is per branch, not per feature

L4 read "all work lands via PR", which in practice became one PR per feature. At
this repo's working pace that is the dominant cost: a self-contained feature can
be designed, implemented, reviewed and committed in well under an hour, and
wrapping each one in a PR round-trip roughly doubled it (maintainer decision,
2026-07-28).

**What changes:** granularity only. Several thematic commits accumulate on one
feature branch and land as a single PR - e.g. Phase 5 shipped segmentation, the
de-styler, HF publication and the samples gate as 17 commits in one PR.

**What does NOT change, and is the whole point of L4:** `main` still moves only
through a PR the maintainer merges, and the agent still never pushes to `main`.
The human gate is intact; only its batch size grew.

**The cost, stated so it is not discovered later:** a larger diff is harder to
review, and the `design-review` pass (L0-adjacent, required by CI - ADR 0022)
now sees more surface at once. Mitigation is to run `design-review` when a
subsystem is finished rather than only before opening the PR, so review still
happens at feature scale even though merging does not.

## Amendment 2 (2026-07-28) - the register's home and what stamps it

The provenance rule above named `prompts/` as the register's home and said its
hash is stamped "into every generated pair". Both were written before the thing
existed; neither survived contact.

- **Home:** ADR 0007 leaves no room for a root `prompts/` directory, and the
  register is a module constant (`cankar/core/register.py`) precisely because
  `configs/` holds what varies per run and NOT varying is the property being
  bought. One-home enforcement: `tests/structure/test_register_home.py`.
- **Stamp:** the hash goes in the generated artifact's **manifest**, not on every
  pair row - `registry/datasets/pairs/pairs.manifest.json` carries
  `register_sha256`. Per-row would be 10,043 copies of one constant.
- **Scope:** the passages manifest deliberately does NOT stamp it. Segmentation
  never reads the register, so stamping there recorded an input the artifact does
  not have and would mark passages stale on an edit that cannot affect them.
  Stamp the artifacts that actually consume it.

"Checked again at inference time" remains unimplemented and correctly so: the
inference consumer is Phase 8. The ROADMAP invariant line stays unticked until
both directions import the constant.
