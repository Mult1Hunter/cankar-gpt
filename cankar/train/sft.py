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

import tiktoken
import torch
from pydantic import BaseModel

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
    """One tokenized pair. `n_target` is what the loss actually sees - the rest
    is context the model reads and is never scored on."""

    tokens: list[int]
    n_prompt: int  # tokens before the target span, all masked
    n_target: int


@dataclass
class SftData:
    examples: list[Example]
    n_dropped_too_long: int
    n_pairs: int

    @property
    def n_target_tokens(self) -> int:
        return sum(e.n_target for e in self.examples)


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
    """

    name: str = "styler-v1"
    init_from: str = "cankar-v1"  # checkpoints/<name>.pt - the voice to build on
    tokenizer: str = "v8192"
    seed: int = 20260728

    seq_len: int = 512  # p99 of the pair set is 469 tokens; 512 covers 99.9%
    batch_size: int = 16
    epochs: float = 3.0

    # Fine-tuning, not pretraining: a tenth of the base run's 2e-3, because the
    # checkpoint already holds the Cankar voice and the job is to attach it to a
    # prompt, not to relearn it. Too high here is catastrophic forgetting.
    matrix_lr: float = 2e-4
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
