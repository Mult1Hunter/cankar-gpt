"""Cankar-only training data (ADR 0016).

Reads the chunked corpus (ADR 0012), keeps Cankar's chunks, DROPS the held-out
works (or the BPB eval is contaminated - invariant #2), tokenizes with the frozen
tokenizer, and streams low-waste (x, y) batches. Each doc is BOS-prepended and
docs are concatenated into one stream sliced into seq_len windows - windows cross
doc boundaries (BOS marks the resets), so only the final partial window per epoch
is dropped, vs nanochat's BOS-bestfit ~35% crop (bpb.py) which a 2.77M-token
corpus cannot afford. Deterministic and resumable: the batch order is a pure
function of (seed, epoch), so resume replays it and skips to the saved step.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tiktoken
import torch

from cankar.core.encoding import bos_id
from cankar.core.errors import CankarError
from cankar.core.holdout import CANKAR_AUTHOR, holdout_excludes, load_holdout

log = logging.getLogger("cankar.train")


def cankar_chunk_texts(chunks_path: Path, holdout_path: Path) -> list[str]:
    """Cankar chunk texts, held-out works excluded (both closure directions)."""
    excludes = holdout_excludes(load_holdout(holdout_path))
    texts: list[str] = []
    dropped = 0
    if not chunks_path.exists():
        raise CankarError(f"no chunks at {chunks_path} (run: cankar tokenizer chunk)")
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("author") != CANKAR_AUTHOR:
                continue
            if d["url"] in excludes:
                dropped += 1
                continue
            texts.append(d["text"])
    if not texts:
        raise CankarError("no Cankar training chunks (run: cankar tokenizer chunk)")
    log.info("cankar training chunks: %d (%d held-out chunks dropped)", len(texts), dropped)
    return texts


@dataclass
class TokenizedCorpus:
    """Each doc = [BOS] + chunk tokens, cached once (tokenizing 2.77M tokens is
    seconds); the batch stream reshuffles + concatenates these per epoch."""

    docs: list[np.ndarray]
    n_tokens: int
    bos: int

    @classmethod
    def build(cls, texts: list[str], enc: tiktoken.Encoding) -> TokenizedCorpus:
        bos = bos_id(enc)
        docs = [np.array([bos, *enc.encode_ordinary(t)], dtype=np.int64) for t in texts]
        return cls(docs=docs, n_tokens=sum(len(d) for d in docs), bos=bos)


def steps_per_epoch(corpus: TokenizedCorpus, batch_size: int, seq_len: int) -> int:
    n_windows = (corpus.n_tokens - 1) // seq_len
    return n_windows // batch_size


def iter_batches(
    corpus: TokenizedCorpus,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: str,
    start_step: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Infinite, deterministic (x, y) batches. Resume by passing start_step: the
    stream replays the same (seed, epoch) order and skips the first start_step
    batches, so a checkpoint resumes the exact data position."""
    produced = 0
    epoch = 0
    while True:
        order = np.random.default_rng(seed + epoch).permutation(len(corpus.docs))
        stream = np.concatenate([corpus.docs[i] for i in order])
        n = (len(stream) - 1) // seq_len
        if n == 0:
            raise CankarError(f"corpus ({corpus.n_tokens} tok) smaller than seq_len {seq_len}")
        xs = torch.from_numpy(np.ascontiguousarray(stream[: n * seq_len].reshape(n, seq_len)))
        ys = torch.from_numpy(np.ascontiguousarray(stream[1 : n * seq_len + 1].reshape(n, seq_len)))
        for b in range(n // batch_size):
            if produced < start_step:
                produced += 1
                continue
            sl = slice(b * batch_size, (b + 1) * batch_size)
            yield xs[sl].to(device), ys[sl].to(device)
            produced += 1
        epoch += 1
