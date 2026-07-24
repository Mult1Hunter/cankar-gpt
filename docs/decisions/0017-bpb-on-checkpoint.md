# ADR 0017 - held-out BPB on a trained checkpoint

**Status:** accepted, 2026-07

## Context

The eval harness (ADR 0013) froze a held-out set and a BPB metric but had no way
to run them on an actual model - "quality claims come from the harness" (invariant
#2) was unenforceable. To score a checkpoint, `evals` must reconstruct the model
and read the tokenizer, but it MUST NOT import the `train` stage (import-linter:
`evals` and `train` are independent siblings, ADR 0007). The checkpoint stored a
`TrainConfig`, which lacks the resolved `vocab_size` (derived from the tokenizer),
so `evals` could not rebuild the exact architecture from it.

## Decision

Make the checkpoint SELF-DESCRIBING: `save_checkpoint` also stores the resolved
`GPTConfig` (`gptconfig`). `cankar evals bpb --checkpoint` reads the `.pt` as a
file, rebuilds the GPT from `gptconfig` via a shared bottom-layer builder
(`cankar.model.build_gpt`), loads the weights, and runs the existing `holdout_bpb`
harness over `iter_holdout_texts`. A trained GPT satisfies the `BpbModel` duck
type directly (`forward(idx, targets, ..., loss_reduction)` matches), so no
adapter is needed. `holdout_bpb` moves batches + token_bytes to the model's device
(GPU-correct; no-op on CPU).

## Rationale

- Self-describing checkpoint over cross-stage import: `evals` reads a file and the
  bottom-layer `model` package, never `train` - stage independence stays real, and
  the checkpoint is reproducible-from-itself (sampling uses the same path).
- `build_gpt` in `cankar.model` DRYs the GPT construction across train and evals
  (rule of two); the `train` sampler now rebuilds from `gptconfig` too.
- `iter_holdout_texts` re-verifies each work's content sha, so a drifted corpus
  fails loud rather than silently scoring the wrong bytes - the same guard class
  as the training-data corpus-skew check (ADR 0016).

## Consequences

- Every checkpoint now yields a comparable held-out number
  (`cankar evals bpb --checkpoint ...`) - the payoff of the Phase-2.25 eval
  investment, and the mechanism Phase 4's "record numbers" needs. Verified
  end-to-end: a 3-step nano model scores BPB 3.57 over the real 50 works (high, as
  expected of an untrained model - the point is the number is real and drops as it
  trains).
- Checkpoints written before this ADR lack `gptconfig`; `bpb_on_checkpoint` rejects
  them with a clear error (they are gitignored and ephemeral).
- Deferred: a committed BPB report/manifest (the number is logged now); wire it
  when a canonical checkpoint exists.
