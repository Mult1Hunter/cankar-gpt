"""Style-transfer SFT data (Phase 6): plain Slovene -> Cankar's voice.

Fine-tunes a pretrained checkpoint on the Phase 5 pairs. Three decisions carry
this module, all measured on the real 9,950-pair set rather than assumed.

**The chat specials already exist.** The frozen v8192 tokenizer inherited
nanochat's `<|user_start|>` / `<|user_end|>` / `<|assistant_start|>` /
`<|assistant_end|>` (ids 8184-8187), so the training format uses them directly.
The ROADMAP sketched `<plain> ... <cankar> ...`, which would have been worse:
invented markers are not in the vocabulary, so they fragment into several
ordinary tokens the model must learn to recognise as a boundary - spending
capacity to rebuild something the tokenizer already provides for free.

**Loss is masked to the target span.** The prompt is 49% of all tokens. Training
on it teaches a 26M-parameter model to generate PLAIN Slovene, which is not the
task and is capacity it cannot spare. `IGNORE_INDEX` matches the BPB harness so
both use one masking convention.

**Replay is a first-class data source, not a flag.** The set the loop trains on
is pairs PLUS unconditional Cankar windows, mixed by scored-token share
(`with_rehearsal`). The held-out eval deliberately does not get the mix - it
scores pairs only, or it would reward voice retention as if it were mapping
progress.

**Over-length pairs are dropped, never truncated.** Measured token lengths:
p50 194, p95 413, p99 469, max 562. At `seq_len` 512 that is 99.9% coverage, so
truncation would affect ~10 pairs - and a truncated target teaches the model to
stop mid-sentence, which is the fluent-but-wrong failure this pipeline exists to
avoid. Dropped pairs are counted, never silent.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tiktoken
import torch
from pydantic import BaseModel, ConfigDict

from cankar.core.encoding import bos_id
from cankar.core.errors import CankarError

log = logging.getLogger("cankar.train")

# Same sentinel as the BPB harness: y < 0 is masked out of the loss.
IGNORE_INDEX = -1

USER_START = "<|user_start|>"
USER_END = "<|user_end|>"
ASSISTANT_START = "<|assistant_start|>"
ASSISTANT_END = "<|assistant_end|>"
SPECIALS = (USER_START, USER_END, ASSISTANT_START, ASSISTANT_END)


@dataclass(frozen=True)
class Example:
    """One tokenized training item. `n_target` is what the loss actually sees -
    the rest is context the model reads and is never scored on.

    `is_rehearsal` marks a replay window rather than a pair. It is carried on the
    example, not inferred from a position in the list, because the batcher sorts
    and shuffles: any accounting that assumed "replay is appended last" would
    silently report the wrong mix the moment it ran through `iter_batches`.
    """

    tokens: list[int]
    n_prompt: int  # tokens before the target span, all masked
    n_target: int
    is_rehearsal: bool = False


@dataclass
class SftData:
    examples: list[Example]
    n_dropped_too_long: int
    n_pairs: int
    n_rehearsal: int = 0

    @property
    def n_target_tokens(self) -> int:
        return sum(e.n_target for e in self.examples)

    @property
    def rehearsal_token_frac(self) -> float:
        """Realized replay share of the SCORED tokens - the only honest unit.

        Not the share of examples: a replay window scores `seq_len` tokens while
        the median pair scores 107, so an example-counted "10% mix" would be
        roughly 35% of the actual loss. The requested fraction is checked against
        this, not against a count of rows.
        """
        total = self.n_target_tokens
        if total == 0:
            return 0.0
        return sum(e.n_target for e in self.examples if e.is_rehearsal) / total


def special_ids(enc: tiktoken.Encoding) -> dict[str, int]:
    """Resolve the four chat specials, failing loud if the tokenizer lacks one.

    A missing special would otherwise be encoded as ordinary text and the format
    would silently degrade into unmarked concatenation - the model would have no
    reliable signal for where the prompt ends.
    """
    missing = [s for s in SPECIALS if s not in enc._special_tokens]
    if missing:
        raise CankarError(
            f"tokenizer lacks the chat specials {missing} - Phase 6 needs them to mark "
            "the prompt/target boundary (expected in the frozen v8192 vocabulary)"
        )
    return {s: enc.encode_single_token(s) for s in SPECIALS}


def build_example(plain: str, cankar: str, enc: tiktoken.Encoding, sp: dict[str, int]) -> Example:
    """`<|bos|><|user_start|>plain<|user_end|><|assistant_start|>cankar<|assistant_end|>`

    The target span deliberately INCLUDES the closing `<|assistant_end|>`: the
    model has to learn where to stop, and a target that never contains the stop
    token produces generations that run on past the passage.
    """
    prompt = [bos_id(enc), sp[USER_START], *enc.encode_ordinary(plain), sp[USER_END]]
    prompt.append(sp[ASSISTANT_START])
    target = [*enc.encode_ordinary(cankar), sp[ASSISTANT_END]]
    return Example(tokens=prompt + target, n_prompt=len(prompt), n_target=len(target))


def load_pairs(path: Path, enc: tiktoken.Encoding, seq_len: int) -> SftData:
    """Tokenize a pair shard, dropping what will not fit."""
    if not path.exists():
        raise CankarError(f"pairs not found: {path} (run: cankar pairs destyle)")
    sp = special_ids(enc)
    examples: list[Example] = []
    dropped = 0
    n_pairs = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_pairs += 1
            row = json.loads(line)
            ex = build_example(row["plain"], row["cankar"], enc, sp)
            # +1: the (x, y) shift needs one token beyond the window.
            if len(ex.tokens) > seq_len + 1:
                dropped += 1
                continue
            examples.append(ex)
    log.info(
        "sft data: %d pairs -> %d examples (%d dropped over seq_len %d), %d target tokens",
        n_pairs,
        len(examples),
        dropped,
        seq_len,
        sum(e.n_target for e in examples),
    )
    return SftData(examples=examples, n_dropped_too_long=dropped, n_pairs=n_pairs)


def n_rehearsal_windows(pair_target_tokens: int, seq_len: int, frac: float) -> int:
    """How many replay windows make `frac` of the scored tokens.

    Solved in tokens, not rows: each window scores exactly `seq_len` of them, so
    n = frac/(1-frac) * pair_tokens / seq_len.
    """
    if not 0.0 <= frac < 1.0:
        raise CankarError(f"rehearsal_frac must be in [0, 1), got {frac}")
    if frac == 0.0:
        return 0
    return max(1, round(frac / (1.0 - frac) * pair_target_tokens / seq_len))


def rehearsal_examples(
    texts: list[str], enc: tiktoken.Encoding, seq_len: int, n_windows: int, seed: int
) -> list[Example]:
    """Unconditional Cankar LM windows, shaped exactly like a pretraining batch.

    Built the way `train/data.py` builds pretraining batches - docs BOS-prefixed,
    concatenated into one permuted stream, sliced at a fixed width - because the
    point of replay is to rehearse the distribution the base checkpoint actually
    learned. Windows that all began at a clean document boundary would be an
    easier distribution than the model ever saw, and would defend the wrong thing.

    `n_prompt=1` marks the whole window as scored: there is no prompt to mask, so
    every position contributes, which is the objective the BPB eval measures.
    Getting this wrong is the module's quietest failure - a fully-masked window
    contributes no gradient while the composition log still reports a healthy
    mix, so it reads as "replay does not help" rather than "replay never ran".
    `SftData.rehearsal_token_frac` exists to make that visible.
    """
    if n_windows <= 0:
        return []
    rng = np.random.default_rng(seed)
    bos = bos_id(enc)
    docs = [[bos, *enc.encode_ordinary(t)] for t in texts]
    stream: list[int] = []
    for i in rng.permutation(len(docs)):
        stream.extend(docs[i])
    width = seq_len + 1  # +1: the (x, y) shift needs one token beyond the window
    n_avail = len(stream) // width
    if n_avail < n_windows:
        raise CankarError(
            f"need {n_windows} replay windows of {width} tokens but the corpus holds "
            f"{n_avail} ({len(stream)} tokens) - lower rehearsal_frac or seq_len"
        )
    return [
        Example(
            tokens=stream[p * width : (p + 1) * width],
            n_prompt=1,
            n_target=seq_len,
            is_rehearsal=True,
        )
        for p in rng.choice(n_avail, size=n_windows, replace=False)
    ]


def with_rehearsal(
    data: SftData, texts: list[str], enc: tiktoken.Encoding, config: SftConfig
) -> SftData:
    """Mix replay windows into a pair set.

    Returns a new SftData rather than mutating: the caller's pair-only set stays
    usable for the held-out eval, which must never score replay windows - that
    would measure voice retention and call it style-transfer progress.
    """
    windows = rehearsal_examples(
        texts,
        enc,
        config.seq_len,
        n_rehearsal_windows(data.n_target_tokens, config.seq_len, config.rehearsal_frac),
        config.seed,
    )
    mixed = SftData(
        examples=[*data.examples, *windows],
        n_dropped_too_long=data.n_dropped_too_long,
        n_pairs=data.n_pairs,
        n_rehearsal=len(windows),
    )
    log.info(
        "rehearsal: %d windows over %d pair examples | %d scored tokens, %.1f%% replay "
        "(requested %.1f%%)",
        mixed.n_rehearsal,
        len(data.examples),
        mixed.n_target_tokens,
        100 * mixed.rehearsal_token_frac,
        100 * config.rehearsal_frac,
    )
    return mixed


def collate(batch: list[Example], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(x, y) padded to the longest example IN THIS BATCH.

    Per-batch rather than to `seq_len`: median length is 194 against a 512
    window, so padding globally would spend roughly 3x the compute on padding.
    Every position that is not a target token - prompt AND padding - is
    IGNORE_INDEX in y, so the loss sees only what the model must generate.
    """
    width = max(len(e.tokens) for e in batch) - 1
    xs, ys = [], []
    for e in batch:
        x = e.tokens[:-1]
        y = [IGNORE_INDEX] * (e.n_prompt - 1) + e.tokens[e.n_prompt :]
        pad = width - len(x)
        xs.append(x + [pad_id] * pad)
        ys.append(y + [IGNORE_INDEX] * pad)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def iter_batches(
    data: SftData, batch_size: int, seed: int, epoch: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic shuffled batches - a pure function of (seed, epoch), so a
    resumed run replays the same order (matching `train/data.py`).

    Sorted into length buckets before batching so a batch pads to something near
    its own median rather than to the longest example in the whole set; the
    shuffle is over BATCHES, which keeps order random without re-mixing lengths.
    """
    pad_id = 0
    g = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(data.examples), generator=g).tolist()
    by_len = sorted(order, key=lambda i: len(data.examples[i].tokens))
    groups = [by_len[i : i + batch_size] for i in range(0, len(by_len), batch_size)]
    for gi in torch.randperm(len(groups), generator=g).tolist():
        yield collate([data.examples[i] for i in groups[gi]], pad_id)


class SftConfig(BaseModel):
    """One SFT run, fully determined.

    Deliberately carries NO model-shape fields. SFT always fine-tunes an
    existing checkpoint, so the shape comes from that checkpoint's own
    `gptconfig` (the ADR 0017 self-describing contract, as `evals/bpb.py`
    reads it). Duplicating n_layer/n_embd here would let a config disagree
    with the weights it loads - a mismatch worth making impossible rather
    than merely detected.

    `extra="forbid"` because pydantic's default is to IGNORE unknown keys, which
    makes a stale or misspelled field in a TOML read as "set" while the run
    quietly uses the default. A test carried `matrix_lr=1e-3` for exactly this
    reason after that field was renamed, and went on asserting things about a
    learning rate it was not setting.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "styler-v1"
    init_from: str = "cankar-v1"  # checkpoints/<name>.pt - the voice to build on
    tokenizer: str = "v8192"
    seed: int = 20260728

    seq_len: int = 512  # p99 of the pair set is 469 tokens; 512 covers 99.9%
    batch_size: int = 16
    epochs: float = 3.0

    # Fine-tuning, not pretraining. `setup_optimizer` builds SIX parameter
    # groups with independently tuned rates (lm_head 4e-3, embedding 0.2,
    # value_embeds 0.1, x0 0.5, smear 0.2, matrix 2e-3), so setting matrix_lr
    # alone leaves the embedding group running ~1000x higher than intended -
    # which is exactly the catastrophic forgetting this comment claimed to
    # prevent, and did not (caught on the first GPU run, 2026-07-28).
    #
    # lr_scale multiplies EVERY group, preserving nanochat's tuned ratios
    # between them while lowering the whole schedule. 0.1 is a tenth of the
    # pretraining rates.
    # Replay against catastrophic forgetting: this share of the SCORED tokens is
    # unconditional Cankar prose rather than a pair. Fine-tuning without it is
    # the highest-forgetting option available, and the first Phase 6 sweep paid
    # for that - a perfectly monotonic frontier where every configuration that
    # improved the mapping degraded held-out BPB, because BPB was measured after
    # the fact and nothing in the objective defended it. Replay puts it back in
    # the loss: the optimizer can no longer buy pair loss with voice, because
    # voice is now part of what it is scoring.
    #
    # 5-20% is the published band. Default 0.0 keeps the frozen no-rehearsal
    # sweep reproducible; the calibrated value is set from the comparison grid.
    rehearsal_frac: float = 0.0

    lr_scale: float = 0.1
    warmup_frac: float = 0.03
    min_lr_frac: float = 0.1
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    log_every: int = 25
    eval_every: int = 100  # held-out pair loss, the only honest progress signal
    checkpoint_every: int = 250


def steps_per_epoch(data: SftData, batch_size: int) -> int:
    return max(1, (len(data.examples) + batch_size - 1) // batch_size)


def lr_multiplier(step: int, total: int, config: SftConfig) -> float:
    """Linear warmup then cosine decay to `min_lr_frac`, matching train/loop."""
    warmup = max(1, int(total * config.warmup_frac))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.min_lr_frac + (1.0 - config.min_lr_frac) * cosine
