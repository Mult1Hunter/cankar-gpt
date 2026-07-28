"""Training data (ADR 0016).

Reads the chunked corpus (ADR 0012), selects the scope (Cankar-only or all
sources), DROPS the held-out
works (or the BPB eval is contaminated - invariant #2), tokenizes with the frozen
tokenizer, and streams low-waste (x, y) batches. Each doc is BOS-prepended and
docs are concatenated into one stream sliced into seq_len windows - windows cross
doc boundaries (BOS marks the resets), so only the epoch's final partial batch is
dropped, vs nanochat's BOS-bestfit ~35% crop (bpb.py) which a 2.77M-token
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


def _chunk_texts(
    chunks_path: Path,
    holdout_path: Path,
    chunks_manifest_path: Path,
    *,
    cankar_only: bool,
) -> list[str]:
    """Chunk texts with the held-out works excluded (both closure directions).
    `cankar_only` keeps just Cankar's chunks (Phase 2.5 / Phase 4 specialization);
    otherwise every source is kept (Phase 3 base pretrain). The frozen-holdout
    exclusion applies EITHER WAY, so held-out Cankar works never leak into
    training and the BPB eval stays honest (invariant #2).

    Provenance guard (design-review 2026-07): the exclusion urls are only valid
    against the corpus the holdout was frozen on. If the chunks were built on a
    DIFFERENT corpus revision, excluding those urls can silently retain excerpts
    of held-out works. Refuse to train on skewed artifacts."""
    manifest = load_holdout(holdout_path)
    excludes = holdout_excludes(manifest)
    chunks_sha = json.loads(chunks_manifest_path.read_text(encoding="utf-8"))["corpus_sha256"]
    if chunks_sha != manifest.corpus_sha256:
        raise CankarError(
            f"corpus revision skew: chunks {chunks_sha[:12]} != holdout "
            f"{manifest.corpus_sha256[:12]}. Re-run: cankar tokenizer chunk"
        )
    if not chunks_path.exists():
        raise CankarError(f"no chunks at {chunks_path} (run: cankar tokenizer chunk)")
    texts: list[str] = []
    dropped = 0
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if cankar_only and d.get("author") != CANKAR_AUTHOR:
                continue
            if d["url"] in excludes:  # held-out works, dropped in BOTH scopes
                dropped += 1
                continue
            texts.append(d["text"])
    scope = "cankar" if cankar_only else "all-source"
    if not texts:
        raise CankarError(f"no {scope} training chunks (run: cankar tokenizer chunk)")
    log.info("%s training chunks: %d (%d held-out chunks dropped)", scope, len(texts), dropped)
    return texts


def cankar_chunk_texts(
    chunks_path: Path, holdout_path: Path, chunks_manifest_path: Path
) -> list[str]:
    """Cankar-only chunk texts (Phase 2.5 TinyCankar / Phase 4 specialization)."""
    return _chunk_texts(chunks_path, holdout_path, chunks_manifest_path, cankar_only=True)


def all_chunk_texts(chunks_path: Path, holdout_path: Path, chunks_manifest_path: Path) -> list[str]:
    """Full-corpus chunk texts for the Phase 3 base pretrain - every source,
    held-out Cankar works still excluded."""
    return _chunk_texts(chunks_path, holdout_path, chunks_manifest_path, cankar_only=False)


@dataclass
class TokenizedCorpus:
    """Each doc = [BOS] + chunk tokens, cached once. Stored int32, not int64:
    the token ids fit (vocab ~8k), and at the full-corpus scale (142.78M tokens)
    int64 would hold ~4x the necessary footprint across docs+stream+xs+ys. The
    embedding needs long, so batches cast to long at the device boundary."""

    docs: list[np.ndarray]
    n_tokens: int
    bos: int

    @classmethod
    def build(cls, texts: list[str], enc: tiktoken.Encoding) -> TokenizedCorpus:
        bos = bos_id(enc)
        docs = [np.array([bos, *enc.encode_ordinary(t)], dtype=np.int32) for t in texts]
        return cls(docs=docs, n_tokens=sum(len(d) for d in docs), bos=bos)


def permuted_stream(corpus: TokenizedCorpus, seed: int) -> np.ndarray:
    """BOS-prefixed docs concatenated in a seeded permutation - the one token
    stream that pretraining windows are cut from.

    Extracted so Phase 6 replay (`train/sft.py`) cuts its windows from the SAME
    construction rather than a lookalike. The correctness argument for replay is
    that it rehearses the distribution the base checkpoint actually saw, and two
    copies of this packing would let that quietly stop being true - a change to
    BOS handling or ordering here would leave replay rehearsing something the
    model never learned, which is the exact failure replay exists to avoid.
    """
    order = np.random.default_rng(seed).permutation(len(corpus.docs))
    return np.concatenate([corpus.docs[i] for i in order])


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
        stream = permuted_stream(corpus, seed + epoch)
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
            yield (
                xs[sl].to(device=device, dtype=torch.long),
                ys[sl].to(device=device, dtype=torch.long),
            )
            produced += 1
        epoch += 1
