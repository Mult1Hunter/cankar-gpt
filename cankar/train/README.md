# cankar/train - the training stage (ADR 0016)

Trains the vendored GPT (`cankar/model`) on the Cankar slice. A stage: it imports
only the bottom layer (`core` + `model`), never a sibling stage - the tokenizer
comes via `core.encoding`, the held-out exclusion set via `core.holdout`.

- `config.py` - `TrainConfig` (validated, TOML). One config + seed fully
  determines a run. Presets: `configs/train/`.
- `data.py` - Cankar chunks, HELD-OUT works dropped (or BPB is contaminated -
  invariant #2), tokenized, low-waste packed into deterministic, resumable
  (x, y) batches.
- `loop.py` - build model -> `init_weights` -> AdamW (warmup+cosine) -> train,
  logging CE loss + tokens/sec; periodic checkpoints; sample text as it learns.
- `checkpoint.py` - reboot-safe save/load (model + optimizer + step + RNG +
  config). `step` fixes the data position, so `--resume` continues exactly.
- `sample.py` - generate from a checkpoint (rebuilds the model from its config).
- `cli.py` - `cankar train run [--config ...] [--resume]`,
  `cankar train sample --checkpoint ...`.

Checkpoints land in `checkpoints/` (gitignored). CPU runs (smoke); the real
TinyCankar run is on a GPU - the loop prints tokens/sec in the first ~minute
(the ROADMAP "calibration first" number).
