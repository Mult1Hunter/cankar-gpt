# ADR 0018 - Phase 3 base pretrain, run locally

**Status:** accepted, 2026-07

## Context

The ROADMAP budgeted Phase 3 (base pretrain on the full corpus) as a cloud job
(~$10-15 on a rented RunPod GPU), a holdover from when the corpus target was
200-300M tokens. The measured corpus is smaller: 142.78M training tokens over
159,973 all-source chunks (Wikipedia + 15 literary authors, held-out Cankar
works excluded). The yield analysis put the honest model size at ~15-30M params
for that budget. TinyCankar (ADR 0016) proved the train stage runs on the local
RTX 4070 Ti Super at ~264k tok/s.

The train stage was Cankar-only (`cankar_chunk_texts`): the base pretrain needs
every source, but the frozen-holdout exclusion must still apply or the BPB eval
(invariant #2) is contaminated.

## Decision

1. **Corpus scope is a config field.** `TrainConfig.corpus: CorpusScope`
   (`cankar` | `all`, StrEnum). `all` keeps every source; `cankar` keeps only
   Cankar's chunks (Phase 2.5 / Phase 4). The held-out works are dropped in
   BOTH scopes - the exclusion is not tied to the author filter (test pins it).

   **Closure verification (design-review, invariant #2):** the held-out set was
   frozen over Cankar docs, but the `all` scope trains on every source. Verified
   directly: containment of all 50 held-out works against the full 126,302-doc
   training corpus (both directions) = **max 0.0000, zero hits >= 0.80**. No
   held-out work is reproduced in any source, so the `all`-scope BPB is honest.

2. **Run Phase 3 locally, not cloud.** At 142.78M tokens a ~26M model does 3
   epochs in ~27 minutes on the 4070 Ti Super - the cloud spend buys nothing.
   `configs/train/base.toml`: n_layer 6, n_embd 384 (26.3M params), seq_len
   1024, 3 epochs, `max_hours` safety cap. Cloud stays reserved for any future
   run that outgrows 16GB (a larger model or a general-Slovene corpus expansion).

## Result

- Base model: 26.3M params, 3 epochs, loss 9.0 -> 2.8 (generalization, not the
  memorization TinyCankar's 1.2 signalled on 2.77M tokens).
- **Held-out BPB 1.5056** (50 works) vs TinyCankar's 2.2227 - a 32% improvement
  from more data + less overfit. The floor for Phase 4 to beat.
- Cost: $0 (vs the budgeted ~$10-15). ~27 min wall-clock.

## Consequences

- The `--max-hours` flag (cost discipline) is validated but not load-bearing
  locally; it earns its keep only if a future run does go to cloud.
- Phase 4 (Cankar specialization) continues pretraining `checkpoints/base.pt`
  on the `cankar` scope - the base is the foundation, not the product. The
  mechanism is `cankar train run --init-from <checkpoint>`, a reusable CLI
  contract (the Phase 5/6 styler is a plausible second user):
  - seeds ONLY the model weights (`load_state_dict`, strict=True - a shape
    mismatch fails loud, so init_from can only load a matching architecture);
  - a FRESH optimizer + step 0 + this config's schedule/scope (not a resume);
  - `--resume` takes precedence when the run's own checkpoint exists, so
    `run --init-from base.pt --resume` is an idempotent restart: seed from
    base on the first run, continue the specialization run on any restart.
- setup.sh and the cloud-ops cost-discipline items (terminate, delete volumes)
  are now lower priority: reserved for a future run that actually needs cloud.
