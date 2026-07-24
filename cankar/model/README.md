# cankar/model - the model architecture (ADR 0016)

The from-scratch GPT, nanochat scaled down. A **bottom-layer, non-stage package**
(like `core`): both `evals` (checkpoint-loading for BPB) and `train` import it,
so it must sit below the stages (import-linter). Imports only torch.

- `gpt.py` - `GPTConfig` + `GPT`, VENDORED-AS-PORT from nanochat @ 92d63d4
  (ADR 0011 pattern; not byte-identical - the three nanochat-internal imports are
  repointed and `setup_optimizer` returns AdamW, not Muon). The forward/decode
  path is verbatim and guarded by `tests/model/test_gpt_drift.py` (behavioral
  logit-equality vs the sibling). `forward(idx, targets, ..., loss_reduction)`
  satisfies the evals `BpbModel` duck type.
- `flash_attention.py` - FA3/SDPA switch, ported (only `COMPUTE_DTYPE` repointed).
  Falls back to torch SDPA on CPU / non-FA3 GPUs, so the model runs everywhere.
- `compute.py` - the compute shim (`COMPUTE_DTYPE`, `print0`) standing in for
  `nanochat.common`. Named `compute.py` because `common.py` is a banned basename.

Vendored code stays close to upstream: fix drift by re-porting, not by editing
the architecture in place. Build sequence: `GPT(config)` -> `.init_weights()`
(non-optional - lambdas are fake-init until then).
