---
name: adr
description: Propose (not write) an architecture decision record. Use when a non-obvious, hard-to-reverse decision gets made - and to check whether it deserves a record at all.
---

# Architecture decision records

Records are **topic-based and amended in place**. The default outcome of invoking
this skill is *no new file*.

## 1. Does this deserve a record? All three must hold

1. **Hard to reverse, or surprising.** Undoing it later touches more than one file,
   or a competent engineer would guess the opposite.
2. **A rejected alternative exists** and can be named in one clause. No alternative
   means it is a result, not a decision.
3. **A cold session would otherwise redo or contradict it.** If the enforcing test's
   failure message already says so, **the test is the record.**

Fail any one -> not an ADR. Route it:

| It is | Goes to |
|---|---|
| A procedure or ritual | `.claude/skills/<name>/SKILL.md` + the PR template |
| A mechanically enforced standard | the enforcing test's module docstring; `.claude/rules/<topic>.md` for the agent-facing rule |
| A tooling choice | ROADMAP checkbox (when) + the tool's own config/README (what) |
| An event or result ("this run happened", "N docs dropped") | `registry/reports/` -> ROADMAP -> `docs/` narrative or the blog |
| A reversal or carve-out to an existing decision | a dated **Amendment** in that ADR + its index row |

## 2. Propose, then stop

State the pitch in ONE line in chat - what is hard to reverse or surprising, and
which alternative was rejected. **Do not create the file until the maintainer says
yes.** You are the typist; they are the architect of record.

## 3. Writing it, once approved

- **Number:** next after the highest `NNNN-*.md`. Numbers are never reused or
  renumbered - hundreds of citations point at them.
- **Filename:** `NNNN-kebab-topic.md`. The title carries the searchable noun; it is
  the only part `graphify` indexes besides `##` headings.
- **Shape** (25-40 lines, advisory not gated - a line-count test rewards reflowing):
  - `## Context` <=6 lines: what was tried, what broke, which alternatives existed.
  - `## Decision` 1-3 sentences plus the artifact paths implementing it.
  - `## Rationale` <=5 bullets, each carrying a **number, a rejected alternative, or
    a case study**. A bullet with none of the three is boilerplate - cut it.
  - `## Consequences` <=4 lines: what gets harder, what is now owed.
- Add a one-line row to `docs/decisions/README.md` in the SAME commit (gated).
- If it changes a rule in CLAUDE.md or ROADMAP.md, update those in the same commit.

## 4. Amending and superseding

A changed decision **appends to the existing record**: a dated `## Amendment`
section, and the reason on its index row. Prefer this over a new number.

When a record really is overtaken, **append to its Status line - never rewrite its
body**: `**Status:** accepted 2026-07; partly superseded by ADR 0007 (2026-07-22)`.
The reversal is the useful part, and a reader landing cold via grep or a code
citation must see it. Symmetry with the index is gated by
`tests/structure/test_adr_index.py`.

Demoted or withdrawn records become short tombstones keeping their number and
`Status: withdrawn, content moved to <path>` - never deletions.
