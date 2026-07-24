# cankar/evals - Phase 2.25 evaluation harness

The measuring stick, frozen before Phase 3 so quality claims carry numbers
(design invariant #2). ADR 0013.

- `holdout.py` - freeze the held-out Cankar set. Whole-work, Wikivir prose,
  containment-closed (a candidate whose text lives inside a kept volume or a
  cross-source twin would leak - the dominant contamination on this corpus).
  Writes the provenance-stamped `registry/evals/holdout.json` + a report.
- `bpb.py` - deterministic held-out BPB harness: each doc BOS-prepended and
  tiled into non-overlapping windows, every token scored once (nanochat's
  training loader crops and packs - wrong for eval). Model is a duck type;
  real checkpoints load in `cankar/model/` at Phase 3.
- `vendored_bpb.py` - nanochat's `evaluate_bpb` vendored verbatim (import
  blocked by torch pin), numerically drift-tested against the sibling.
- `style.py` - the style classifier (ADR 0015). Char n-gram TF-IDF + balanced
  logistic regression, Cankar prose vs PD peers (prose-vs-prose, not vs
  Wikipedia), chunk-level, group-split by content near-duplicate cluster, verse
  filtered. Freezes a `.joblib` + provenance manifest + a human-audited confound
  report. Content-only - no corpus-registry import (stage independence is real).
- `cli.py` - `cankar evals holdout-freeze`, `cankar evals style-train`.

LLM-judge is a later deliverable (ROADMAP Phase 6, needs Phase-5 pairs).
