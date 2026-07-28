# Rehearsal in style-transfer SFT - what the first sweep got wrong

Phase 6 fine-tunes `cankar-v1` on the Phase 5 pairs. The first learning-rate
sweep produced a perfectly monotonic Pareto frontier: every configuration that
improved the plain->Cankar mapping degraded held-out BPB, with no exceptions.
That was read as a capacity wall in a 26.3M-parameter model.

It was a method error. Held-out BPB was measured *after* training and nothing in
the objective defended it, so the optimizer was free to spend voice on pair
loss - and did, exactly as it should have. Rehearsal (replay) is the documented
standard mitigation for catastrophic forgetting and had been skipped entirely.
The wall was self-inflicted.

## What changed

The training set is now pairs PLUS unconditional Cankar windows, mixed by
scored-token share (`cankar/train/sft.py`, `with_rehearsal`). The replay windows
carry the pretraining objective, so held-out BPB is no longer an external metric
the optimizer can ignore - it is part of what the loss is made of.

### The knob is a token share; the gradient sees a step share

`rehearsal_frac` sets replay's share of the SCORED TOKENS. That is not what
reaches the optimizer, and the difference is about 3x.

`GPT.forward` reduces the loss with `mean`, so every batch moves the weights
equally regardless of how many tokens it scored. `iter_batches` length-sorts
before grouping, and replay windows are all exactly `seq_len + 1` while pairs sit
at p99 469 - so the two kinds land in separate batches almost perfectly. Measured
at the shipped settings: **609 pure-pair batches, 127 pure-replay, 1 mixed.**
Replay's real weight is its share of *batches*, which equals its share of
*examples*: 17.4%, not the configured 50%.

This cuts against the reasoning the mechanism was built on. The token framing was
chosen specifically to avoid row-counting, on the grounds that a window scores
512 tokens against a pair's 107 - and under this batcher, row-counting was the
accurate unit all along. It also resolves what had looked like an anomaly: a
setting of 0.5 appeared to be far above the 5-20% replay band the literature
quotes. At 17.4% effective, it is inside it.

The measured results are unaffected - the code did what the tables record. What
was wrong is the label. `sft.rehearsal_step_frac` now computes the effective
share from the same grouping the batcher emits, the loop logs both, and the
checkpoint records both.

Replay windows are built the way `train/data.py` builds pretraining batches:
BOS-prefixed documents concatenated into one permuted stream and sliced at a
fixed width, so most windows start mid-sentence. No genre filter, unlike Phase 5
segmentation - `cankar-v1` was pretrained on the drama and verse too, and BPB is
measured over all of it, so filtering to prose would rehearse a distribution the
model never learned. Held-out works stay excluded on both sides.

## Results

All cells: 2 epochs, `cankar-v1` init, held-out BPB against the frozen 50-work
set (base 1.4508). `pair` is held-out pair loss (pairs only - scoring replay
windows there would reward voice retention as if it were mapping progress).
`copy` is the longest verbatim run shared with the prompt, over output length.

| lr_scale | rehearsal | BPB | vs base | pair | copy |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.0 | 1.5554 | +0.1046 | 1.650 | 0.08 |
| 0.01 | 0.1 | 1.5422 | +0.0914 | 1.651 | 0.10 |
| 0.01 | 0.2 | 1.5286 | +0.0778 | 1.654 | 0.11 |
| 0.01 | 0.35 | 1.5120 | +0.0612 | 1.657 | 0.09 |
| 0.03 | 0.0 | 1.6210 | +0.1702 | 1.457 | 0.09 |
| 0.03 | 0.1 | 1.5689 | +0.1181 | 1.461 | 0.10 |
| 0.03 | 0.2 | 1.5385 | +0.0877 | 1.463 | 0.11 |
| 0.03 | 0.35 | 1.5090 | +0.0582 | 1.468 | 0.12 |
| 0.1 | 0.0 | 1.7366 | +0.2858 | 1.286 | 0.15 |
| 0.1 | 0.1 | 1.5960 | +0.1452 | 1.292 | 0.15 |
| 0.1 | 0.2 | 1.5505 | +0.0997 | 1.293 | 0.16 |
| 0.1 | 0.35 | 1.5214 | +0.0706 | 1.298 | 0.11 |
| 0.3 | 0.0 | 1.9049 | +0.4541 | 1.175 | 0.15 |
| 0.3 | 0.1 | 1.6419 | +0.1911 | 1.181 | 0.15 |
| 0.3 | 0.2 | 1.5948 | +0.1440 | 1.183 | 0.15 |
| 0.3 | 0.35 | 1.5645 | +0.1137 | 1.187 | 0.09 |
| 0.3 | 0.5 | 1.5450 | +0.0942 | 1.187 | 0.15 |
| 0.3 | 0.65 | 1.5216 | +0.0708 | 1.192 | 0.15 |

The `rehearsal 0.0` cells reproduce sweep 1 to within 0.0002 (1.5554/1.5553,
1.6210/1.6209, 1.7366/1.7368), which is what makes the rest of the table a
comparison rather than two unrelated runs.

**Replay trades steeply in its favour at every learning rate**, and the benefit
scales with how much forgetting there was to prevent. It is a trade, not a free
win: at fixed lr, replay improves BPB and *worsens* pair loss in all four rows.
What changes is the exchange rate.

| lr_scale | BPB damage, no replay | at 0.35 | recovered | pair loss cost |
|---:|---:|---:|---:|---|
| 0.01 | +0.1046 | +0.0612 | 41% | 1.650 -> 1.657 |
| 0.03 | +0.1702 | +0.0582 | 66% | 1.457 -> 1.468 |
| 0.1 | +0.2858 | +0.0706 | 75% | 1.286 -> 1.298 |
| 0.3 | +0.4541 | +0.1137 | 75% | 1.175 -> 1.187 |

Read against sweep 1's frontier (reproduced in full below) at equal voice
damage, replay reaches a 17-30% lower pair loss. The best cell in the study
(lr 0.3, rehearsal 0.65) strictly dominates the best cell in sweep 1 (lr 0.1,
2 epochs) on both axes at once: pair loss 1.192 vs 1.286, BPB +0.071 vs +0.286.
Dominance is shown for that comparison - across the frontier it is a better
exchange rate, not a strict beat.

### Sweep 1, for reference

The no-rehearsal frontier every claim above is measured against. Same protocol,
`rehearsal_frac` 0 throughout.

| lr_scale | epochs | BPB | vs base | pair |
|---:|---:|---:|---:|---:|
| 0.003 | 1 | 1.5005 | +0.0497 | 2.197 |
| 0.003 | 2 | 1.5163 | +0.0655 | 1.914 |
| 0.01 | 1 | 1.5291 | +0.0783 | 1.787 |
| 0.01 | 2 | 1.5553 | +0.1045 | 1.650 |
| 0.03 | 1 | 1.5740 | +0.1232 | 1.572 |
| 0.03 | 2 | 1.6209 | +0.1701 | 1.458 |
| 0.1 | 1 | 1.6522 | +0.2014 | 1.375 |
| 0.1 | 2 | 1.7368 | +0.2860 | 1.286 |

## Two things this does not show

**The runs are not compute-matched.** Replay is ADDED to the pair set rather
than displacing part of it, so every pair is seen the same number of times at
any fraction and a higher fraction means more total tokens. "It just trained
longer" is not an available explanation, though: sweep 1 varied epochs directly
and more pair compute made BPB *uniformly worse* at all four learning rates
(1.5005->1.5163, 1.5291->1.5553, 1.5740->1.6209, 1.6522->1.7368). Extra pair
compute damages voice; extra replay compute restores it. Opposite signs.

**The styler is still not good.** These numbers say replay is a much better
exchange rate than no replay. They do not say the 26.3M model performs the task. Generations still
garble the opening content word - `Sestavljanje pohištva` (assembling furniture)
comes back as `Sestanek` (a meeting) - and drift from the source. What replay
fixed is the grammar and the voice around that drift, which is what BPB
measures. Whether the mapping itself is capacity-bound remains open; it is now
an open question measured with a sound method rather than a settled one measured
with a broken one.

## Where the ceiling is

The benefit had not saturated at 0.65, the top of the range tested. It could not
be pushed further: 0.8 needs 8,141 windows and Cankar's 2.77M non-held-out
tokens supply 5,395, so the pool-size guard refused the run. **The constraint is
corpus size, not the method.** The default is 0.5 rather than the
best-measured 0.65 for that reason - 0.65 consumes 3,780 of 5,395 windows (70%),
leaving no headroom before a `seq_len` or pair-set change trips the guard.

## Epochs

Sweep 1 fixed 2 epochs because more was strictly worse for BPB at every learning
rate. Re-run at the calibrated point (lr 0.3, replay 0.5), that still holds for
BPB - and pair loss turns out to have a minimum there too:

| epochs | BPB | vs base | pair |
|---:|---:|---:|---:|
| 1 | 1.5125 | +0.0617 | 1.225 |
| 2 | 1.5450 | +0.0942 | **1.187** |
| 3 | 1.5700 | +0.1192 | 1.207 |
| 4 | 1.5964 | +0.1456 | 1.248 |
| 6 | 1.6500 | +0.1992 | 1.353 |

Past 2 epochs both metrics degrade together, so it is overfitting rather than a
trade. `configs/train/styler-v1.toml` had shipped `epochs = 3.0`, which was never
measured; it is now 2.0.

## Reproducing

The grid is a one-off methodology experiment, not a standing gate, so it lives
in the session scratchpad rather than as a `cankar` subcommand (rule of two - if
a second sweep needs it, it earns a home). The mechanism it validated is in
`cankar/train/sft.py` and gated by `tests/train/test_sft.py` plus eight entries
in `ops/mutations.toml`.

**Protocol.** Every cell: `init_from` `cankar-v1` (step 500), tokenizer `v8192`,
`seq_len` 512, `batch_size` 16, `seed` 20260728, warmup 0.03, min_lr_frac 0.1,
weight_decay 0, grad_clip 1.0, on one CUDA device. `pair` is
`sft_loop.evaluate` over `data/pairs/holdout-pairs.jsonl` (285 pairs, pairs
only). `BPB` is `cankar evals bpb --checkpoint <ckpt>`, the frozen 50-work
held-out set, base `cankar-v1` = 1.4508. `copy` is the longest common substring
with the prompt over output length, on 8 register-arm drafts at temperature 0.8,
top_k 50, seed 7.
