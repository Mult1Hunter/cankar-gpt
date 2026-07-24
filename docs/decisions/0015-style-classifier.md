# ADR 0015 - style classifier: eval pillar #2

**Status:** accepted, 2026-07

## Context

Invariant #2 needs a harness signal for "is this in Cankar's prose voice" to
judge Phase-5/6 style-transfer outputs. The naive baseline (word TF-IDF, Cankar
vs Wikipedia, whole works, url-grouped) fails four ways the holdout pattern
hides: it learns TOPIC not voice, measures an encyclopedia/era confound (ROADMAP
A-1), leaks via the volume-containment structure ADR 0013 calls dominant, and
mismatches the passage-sized deploy distribution. Genre labels that would filter
verse are 74% empty; clean.py already records the ~150-poem cost of a form
confound.

## Decision

New `cankar/evals/style.py` + `cankar evals style-train`. Binary char n-gram
(3-5) TF-IDF + balanced logistic regression, Cankar prose vs 14 PD peers'
prose, wikivir-only. Passage-sized chunks (~90 words), group-split by
within-author near-duplicate CLUSTER (containment, content-based - no registry
read, so stage independence is real). Verse stripped by a short-line detector
calibrated on committed fixtures. Frozen: gitignored `.joblib` + provenance
manifest `registry/evals/style.json` (lib versions, seed, config, artifact
sha256, metrics) + human-audited report `registry/reports/style.md`.

## Rationale

- char n-grams measure voice (punctuation rhythm, morphology), word n-grams
  measure topic - the ablation and top-feature audit prove it, and are a
  REQUIRED gate before any claim rides on the score (MF-5).
- chunk-level matches the Phase-6 deployment unit (invariant-#1 spirit); doc-
  level would train on 8k-char works and score few-hundred-char passages.
- content-based grouping + verse detection keep evals independent of the corpus
  registry (import-linter) while killing the containment leak (MF-1/MF-2).
- **Orthography deviation (evidence over the critique's prescribed normalizer):**
  the data says the "solnce"-class edition-spelling confound is not load-bearing
  - it is absent from the top char features, function words alone (orthography-
  stable) separate Cankar at ROC-AUC ~0.87 (0.874), and the sonce-family
  spot-check leaves AUC unchanged (0.9933 = char_wb). Building a comprehensive
  historical normalizer to fix an
  absent confound is the ADR 0006 anti-pattern; the funcwords floor + audit are
  the neutralization instead.
- A trained scorer is not regenerable data (MF-4): the manifest pins versions +
  seed + config + artifact sha; the contract is "retrain reproduces metrics
  within tolerance" (seed-determinism test), not bit-identical weights.

## Consequences

- Ships the classifier as a Cankar-vs-PD-peers VOICE instrument. Its Phase-6
  deploy negative (modern de-styled Slovene) is unseen here, so the manifest
  stamps `deploy_validated: PENDING Phase 6` - the gap is visible, not a vibe
  (MF-3, invariant #1 analogue); the deploy-margin check lands with Phase-5 pairs.
- Adds scikit-learn/scipy/joblib. The `.joblib` is gitignored (heavy binary);
  durable publication (HF/R2) is a pre-Phase-6 task, tracked.
- SloBERTa stays deferred behind this baseline (ROADMAP), now with a measured
  AUC to beat, not a guess.
