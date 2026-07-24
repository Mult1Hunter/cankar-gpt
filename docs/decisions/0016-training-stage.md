# ADR 0016 - training stage: vendored GPT + lean AdamW loop

**Status:** accepted, 2026-07

## Context

Phases 3-4 (and the Phase 2.5 TinyCankar rehearsal) need to train the model.
The project is "nanochat scaled down" (ADR 0001), but nanochat is not importable
(torch pin, ADR 0011) and its trainer is 31k lines of DDP/fp8/torch.compile/Muon
machinery irrelevant to a 10-30M single-device model. We need the model
architecture and a training loop we own, without letting the trainer import the
evals stage (import-linter).

## Decision

Two additions. (1) `cankar/model/` - a bottom-layer non-stage package (like
`core`; both `evals` and `train` import it) holding the nanochat GPT
VENDORED-AS-PORT: `gpt.py` + `flash_attention.py` copied, the three
nanochat-internal imports repointed to a `compute.py` shim, and `setup_optimizer`
returning plain AdamW (Muon is premature here) while keeping the tuned per-group
LR structure. (2) `cankar/train/` - a new stage: a validated TOML config, a
Cankar-only data loader that drops the held-out works, a lean single-device
AdamW loop (warmup+cosine, CE loss + tokens/sec, reboot-safe checkpoint/resume),
a sampler, and `cankar train run/sample`. The frozen holdout contract was
promoted to `cankar/core/holdout.py` so `train` reads it without importing evals.

## Rationale

- The port is guarded behaviorally, not by byte-diff: a fixed-seed forward
  logit-equality test vs the sibling (tests/model/test_gpt_drift.py) catches
  upstream drift while allowing the necessary import edits.
- AdamW, not Muon: the "50-100x less compute" thesis is a size/token claim, not
  an optimizer one; Muon buys ~1.3-2x on matrix params only and drags in fused
  Triton kernels. The matrix LR is retuned (0.02 Muon -> 0.002 AdamW).
- Low-waste packing (concatenate BOS-prepended chunks, slice into windows) over
  nanochat's BOS-bestfit ~35% crop, which a 2.77M-token corpus cannot afford.
- Resume is exact because the batch order is a pure function of (seed, epoch);
  the checkpoint's step fixes the data position.

## Consequences

- Ships the training stage; TinyCankar (Phase 2.5) is now a `cankar train run`
  away. CPU here only smoke-tests (real run is a GPU job) - verified end-to-end
  on the real Cankar slice: holdout-excluded load, loss down, tokens/sec, sample,
  checkpoint, resume.
- Vendored files are ruff/mypy-exempt (kept close to upstream) and MIT-attributed
  (THIRD_PARTY_NOTICES.md). Muon, fp8, DDP, torch.compile, the RunPod setup.sh,
  and held-out BPB-on-checkpoint are deferred to Phase 3.
