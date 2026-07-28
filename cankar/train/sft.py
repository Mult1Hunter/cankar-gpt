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
from pydantic import BaseModel, ConfigDict, Field

from cankar.core.encoding import bos_id
from cankar.core.errors import CankarError
from cankar.train import data

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
    def n_pair_examples(self) -> int:
        return len(self.examples) - self.n_rehearsal

    @property
    def rehearsal_token_frac(self) -> float:
        """Replay share of the SCORED TOKENS - a data-composition figure.

        This is what `rehearsal_frac` sets, and it is NOT the share of the
        gradient. `GPT.forward` reduces with `mean`, so every batch moves the
        weights equally regardless of how many tokens it scored, and
        `iter_batches` length-sorts - replay windows are all exactly seq_len+1
        while pairs sit at p99 469, so the two kinds land in separate batches
        almost perfectly (measured: 609 pure-pair, 127 pure-replay, 1 mixed).
        The gradient share is therefore the EXAMPLE share. See
        `rehearsal_step_frac`, which is the number to reason about.
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


def build_prompt(plain: str, enc: tiktoken.Encoding, sp: dict[str, int]) -> list[int]:
    """The prompt side of the training format, up to and including
    `<|assistant_start|>`.

    Shared with inference (`train/sample.py::style_transfer`) rather than
    reconstructed there. A second copy would let the training format gain a field
    and inference silently keep the old one - a train/inference mismatch, which is
    the failure design invariant #1 exists to prevent, in a place the invariant
    itself does not reach.
    """
    return [
        bos_id(enc),
        sp[USER_START],
        *enc.encode_ordinary(plain),
        sp[USER_END],
        sp[ASSISTANT_START],
    ]


def build_example(plain: str, cankar: str, enc: tiktoken.Encoding, sp: dict[str, int]) -> Example:
    """`<|bos|><|user_start|>plain<|user_end|><|assistant_start|>cankar<|assistant_end|>`

    The target span deliberately INCLUDES the closing `<|assistant_end|>`: the
    model has to learn where to stop, and a target that never contains the stop
    token produces generations that run on past the passage.
    """
    prompt = build_prompt(plain, enc, sp)
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
    # The same packer pretraining uses, not a lookalike - see permuted_stream.
    stream = data.permuted_stream(data.TokenizedCorpus.build(texts, enc), seed)
    rng = np.random.default_rng(seed)
    width = seq_len + 1  # +1: the (x, y) shift needs one token beyond the window
    n_avail = len(stream) // width
    if n_avail < n_windows:
        raise CankarError(
            f"need {n_windows} replay windows of {width} tokens but the corpus holds "
            f"{n_avail} ({len(stream)} tokens) - lower rehearsal_frac or seq_len"
        )
    return [
        Example(
            tokens=[int(t) for t in stream[p * width : (p + 1) * width]],
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
        "rehearsal: %d windows over %d pair examples | tokens %.1f%% replay "
        "(requested %.1f%%) | optimizer steps %.1f%% replay <- the effective weight",
        mixed.n_rehearsal,
        mixed.n_pair_examples,
        100 * mixed.rehearsal_token_frac,
        100 * config.rehearsal_frac,
        100 * rehearsal_step_frac(mixed, config.batch_size, config.seed),
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


def length_buckets(data: SftData, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """The grouping `iter_batches` emits, as indices.

    Extracted because `rehearsal_step_frac` has to measure the SAME grouping.
    Re-deriving it there broke on tie-breaking - this sorts a PERMUTED order, so
    equal-length examples group by shuffle position, not by index - and a replay
    share computed off a lookalike grouping is exactly the kind of number that
    reads as authoritative while being wrong.
    """
    g = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(data.examples), generator=g).tolist()
    by_len = sorted(order, key=lambda i: len(data.examples[i].tokens))
    return [by_len[i : i + batch_size] for i in range(0, len(by_len), batch_size)]


def rehearsal_step_frac(data: SftData, batch_size: int, seed: int, epoch: int = 0) -> float:
    """Replay share of the OPTIMIZER STEPS - the effective mixing weight.

    `GPT.forward` reduces with `mean`, so every batch moves the weights equally
    regardless of how many tokens it scored, and length bucketing puts the
    fixed-width replay windows in their own batches. The gradient share is
    therefore the batch share, NOT the token share `rehearsal_frac` sets. At the
    shipped settings a `rehearsal_frac` of 0.5 realizes about 0.17 here - which
    is what puts this run inside the 5-20% band the literature quotes rather
    than above it, as the token figure alone suggests.
    """
    groups = length_buckets(data, batch_size, seed, epoch)
    if not groups:
        return 0.0
    return sum(any(data.examples[i].is_rehearsal for i in g) for g in groups) / len(groups)


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
    groups = length_buckets(data, batch_size, seed, epoch)
    g = torch.Generator().manual_seed(seed + epoch)
    torch.randperm(len(data.examples), generator=g)  # keep the group-shuffle stream aligned
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

    seq_len: int = Field(default=512, gt=0)  # p99 of the pairs is 469; 512 covers 99.9%
    batch_size: int = Field(default=16, gt=0)

    # 2, not 3. Held-out pair loss BOTTOMS at 2 epochs and rises after
    # (1.187 -> 1.207 -> 1.248 -> 1.353 at 2/3/4/6) while BPB degrades
    # monotonically throughout, so past 2 epochs both metrics get worse together
    # - overfitting, not a trade. The 3.0 this shipped with was never measured.
    epochs: float = Field(default=2.0, gt=0)

    # Replay against catastrophic forgetting: this share of the SCORED TOKENS is
    # unconditional Cankar prose rather than a pair. Fine-tuning without it is
    # the highest-forgetting option available, and the first Phase 6 sweep paid
    # for that - a monotonic frontier where every configuration that improved the
    # mapping degraded held-out BPB, because BPB was measured after the fact and
    # nothing in the objective defended it. Replay puts it back in the loss.
    #
    # Read `rehearsal_token_frac` before tuning this: the effective weight on the
    # gradient is the STEP share, ~0.17 at a setting of 0.5.
    #
    # Calibrated on a 16-cell grid (lr_scale x this) plus saturation and epoch
    # extensions, against sweep 1's frozen no-rehearsal numbers; protocol and
    # full table in docs/style-transfer-rehearsal.md. Replay traded steeply in
    # its favour at every learning rate, and the benefit grew with lr because a
    # higher rate forgets more: at lr_scale 0.3 it recovered 84% of the BPB
    # damage (+0.454 -> +0.071) for a 1.4% rise in pair loss (1.175 -> 1.192).
    #
    # 0.5 rather than the best-measured 0.65 because the benefit had not
    # saturated at the top of the range - the ceiling here is corpus size, not
    # the method. 0.65 consumes 70% of the 5,395 windows Cankar's 2.77M
    # non-held-out tokens supply, leaving no headroom for a seq_len or pair-set
    # change before `rehearsal_examples` refuses the run.
    rehearsal_frac: float = Field(default=0.5, ge=0.0, lt=1.0)

    # Fine-tuning, not pretraining. `setup_optimizer` builds SIX parameter groups
    # with independently tuned rates (lm_head 4e-3, embedding 0.2, value_embeds
    # 0.1, x0 0.5, smear 0.2, matrix 2e-3), so setting matrix_lr alone left the
    # embedding group running ~1000x higher than intended - exactly the
    # catastrophic forgetting it claimed to prevent (caught on the first GPU run,
    # 2026-07-28). lr_scale multiplies EVERY group, preserving nanochat's tuned
    # ratios while lowering the whole schedule.
    #
    # 0.3, not the 0.1 this shipped with: with replay defending the voice, the
    # higher rate is better on BOTH axes (pair 1.187 vs 1.293, BPB +0.094 vs
    # +0.100 at rehearsal_frac 0.5 / 0.2 respectively). Without replay 0.1 was
    # the sane ceiling; the calibration moved it.
    lr_scale: float = Field(default=0.3, gt=0)
    warmup_frac: float = Field(default=0.03, ge=0.0, lt=1.0)
    min_lr_frac: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0)

    log_every: int = Field(default=25, gt=0)
    eval_every: int = Field(default=100, gt=0)  # held-out pair loss, the honest signal
    checkpoint_every: int = Field(default=250, gt=0)

    @property
    def uses_rehearsal(self) -> bool:
        """One place decides this - it was being re-derived in cli.py and the
        loop, which is how the two would drift into disagreeing about whether a
        run is mixing replay."""
        return self.rehearsal_frac > 0.0


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
