# ADR 0004 - works registry as source of truth for corpus provenance

**Status:** accepted, 2026-07 (amended 2026-07 for non-authored sources - see Amendment)

## Context

The Wikivir crawl covers 218 Cankar works; dLib.si holds 458 records for the
same author (journal originals, book editions, manuscripts, plus non-text and
in-copyright items). More sources and more authors follow. Without a canonical
list there is no way to say "we have all of X", to prevent title collisions
across authors, or to know where any document came from.

## Decision

A committed, per-author **works registry** (`registry/<author>.jsonl`, pydantic
`WorkRecord` in `cankar/corpus/registry.py`) is the source of truth for every known
work: canonical title, year, genre, aliases, and per-source status
(`ingested`, `candidate`, `skipped-quality`, `skipped-manuscript`,
`skipped-rights`, `missing`). Rules:

- **Registry-first ingestion:** seed the registry from author catalogs (Wikivir
  index + Seznam pages) before crawling; every corpus document must map to a
  registry entry; unmatched source records go to a triage report, never
  silently dropped.
- **Identity:** normalized title (NFC, casefold, punctuation stripped,
  diacritics kept) within one author; publication year disambiguates editions
  and validates plausibility against the author's lifetime.
- **Collision guard:** `cankar corpus validate` - work_id uniqueness,
  duplicate normalized titles, year ranges, and cross-author title collisions
  flagged for manual confirmation (generic titles like "Jure" collide).
- **Source preference:** hand transcription (wikivir) beats OCR (dlib); dLib
  fills gaps only. OCR ingestion passes cankar/corpus/ocr_clean quality gates.
- `registry/reports/coverage-<author>.md` is generated (`cankar corpus report`),
  committed, and answers "what do we have and from where" at a glance.

## Rationale

- Completeness becomes measurable (coverage report) instead of anecdotal.
- Provenance per work survives any pipeline rewrite - it lives in data, not in
  scripts or memory.
- The same structure serves every future author and source (PD authors next).

## Consequences

- Every new source integration starts with registry matching + status
  bookkeeping - small constant overhead per crawler.
- The registry is hand-editable (`notes` never clobbered by tooling); humans
  resolve triage and collision flags.

## Amendment (2026-07) - registry scope is authored literary works

The Wikipedia ingestion stage forced the boundary the original invariant left
implicit. "Every ingested document maps to a works-registry entry" is scoped to
**authored literary works** (the Cankar + PD-author corpus). General-reference
sources without per-work authorship - Wikipedia's ~190k articles - do NOT enter
a per-author works registry; forcing them in would be meaningless.

For such sources the sanctioned provenance path is the committed **dataset
manifest** (`registry/datasets/`, ADR 0007): input dump filename, date, sha256,
license, plus output counts. The "never silently dropped" guarantee still binds,
honored by **skip counts** (every filtered class tallied and recorded in the
manifest) rather than infeasible per-item triage over hundreds of thousands of
redirects and stubs.

CLAUDE.md's engineering-system rule is updated to match: authored-source
documents map to a registry entry; non-authored sources carry dataset-manifest
provenance with per-reason skip counts (the manifest's `skip_counts` field).

## Amendment 2 (2026-07) - the registry is a ledger, not an ingestion gate

An external methodological review argued (point 1D) that making a hand-curated
registry the ingestion gatekeeper during discovery invites confirmation bias -
"only what I already know about exists". The critique was half right, and the
half that was right had already cost us: the dLib crawl sent any record whose
title failed registry matching to a gitignored triage file, which silently
buried recoverable PD works whose dLib titles are journal-issue titles (the
1914 crtice class). The registry surfaced unknowns too (dlib-discovered
candidates), so the instrument was sound - the GATING was the bug.

Resolution, mechanized in `cankar corpus reconcile-dlib`:

- The registry remains the source of truth for what is KNOWN - but failing to
  match it must never destroy information. Unmatched-but-PD records are now
  upserted into the committed registry as `dlib-discovered` candidates with
  their `is_part_of` context, instead of dying in gitignored triage.
- Coverage against the source is audited by an explicit bucket classification
  (every DOC record lands in exactly one enumerated class) written to a
  committed snapshot report - the miss class is now measurable, permanently.
- Title matching gains one relaxation (subtitle stripping on dLib's " : "
  separator), still exact-on-normalized - no fuzzy matching, preserving the
  name-collision guarantees that motivated this ADR.

## Amendment 3 (2026-07-26) - misattribution, absorbed from ADR 0014

The ADR 0013 holdout audit surfaced texts crawled into Cankar's shard but written
by others - two Vera Albreht memoirs and a critic's essay - carrying
`author="Ivan Cankar"`. They sat in the merged Cankar TRAINING slice, poisoning
Phase 4 continued-pretraining and the style classifier.

`WorkFlag.NOT_BY_AUTHOR` extends this ADR's core claim: the human-curated registry
is the source of truth, so a misattributed work is flagged there (author field
kept, true author and rights in `notes`) and `merge.py` drops any shard doc mapping
to a flagged work, before gate and dedup, counting it as `not_by_author`.

- **Enforcement belongs at merge, not crawl.** Wikivir exposes no structured author
  metadata, so a crawl-time detector would be a fragile title heuristic - the ADR
  0006 failure class. A genitive-title sweep already false-positived on Cankar's own
  `Smrt in pogreb Jakoba Nesrece`.
- **Flag over deletion.** The records must persist so the merge can match and
  exclude them; deleting them would let the shard docs back in.
- **Enumerated, not detected.** A full-corpus sweep (genitive + person-name + text
  read) found 5 real records, not the 3 the length-banded audit saw. A future
  about-subject work with no name cue would need the same human audit - recorded,
  not mechanized, because a reliable wikivir detector does not exist.
- Two of the Albreht memoirs are also still in copyright (d. 1982, PD 2053);
  excluded on the attribution flag, rights recorded in notes.

Counts for each re-merge live in `registry/reports/merge.md`, which is
provenance-stamped and regenerates - they are not repeated here.

**Defense in depth across a stage boundary.** The eval-side url list from ADR 0013
is kept, but downgraded from a candidacy filter to an independent last-line
assertion: `cankar_docs` fails loud if a flagged work ever re-appears. `evals`
keeps its own guard because it reads the merged corpus and never the corpus
registry - stage independence means a merge regression must not silently score
seen text. `coverage.py` correspondingly stops counting flagged works as ingested.

This is distinct from cross-author works that ARE kept under their true author via
`collision_resolution.toml` - a March 1898 letter from Josip Murn sits in both
correspondents' shards and is re-attributed, not dropped.
