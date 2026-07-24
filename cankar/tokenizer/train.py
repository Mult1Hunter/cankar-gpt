"""Train Slovene BPE candidates and emit nanochat's two-file artifact.

Replicates the artifact recipe from nanochat's scripts/tok_train.py (see
vendored.py provenance): tokenizer.pkl is the pickled tiktoken Encoding
(loadable by RustBPETokenizer.from_directory), token_bytes.pt is the
int32 tensor base_train.py asserts on at startup - length n_vocab, zeros
at special-token ids, byte length elsewhere. Both are required; shipping
only the pickle crashes Phase 3 (architect critique MF-1).

Deliberate deviation from tok_train.py, recorded in the manifest: no
--doc-cap. nanochat caps documents at 10k chars, which would shrink the
literary share ~5x (55 volumes exceed 200k chars); training on full docs
is a mix decision in favor of the literary slice (critique A-2).
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections.abc import Iterator
from pathlib import Path

import rustbpe
import tiktoken
import torch
from pydantic import BaseModel

from cankar.core.errors import CankarError
from cankar.core.jsonl import iter_jsonl_docs
from cankar.tokenizer.vendored import SPECIAL_TOKENS, SPLIT_PATTERN

log = logging.getLogger("cankar.tokenizer")


class TokenizerManifest(BaseModel):
    """Committed provenance for one trained candidate (ADR 0003)."""

    schema_version: int = 1
    name: str
    vocab_size: int  # total, including special tokens
    n_mergeable_ranks: int  # vocab_size - len(SPECIAL_TOKENS)
    special_tokens: list[str]
    split_pattern_sha256: str
    corpus_sha256: str
    n_docs: int
    n_chars: int
    doc_cap: None  # explicit: full documents, no nanochat --doc-cap (A-2)
    rustbpe_version: str
    tiktoken_version: str
    torch_version: str
    nanochat_commit: str
    git_sha: str
    trained_at: str
    tokenizer_pkl_sha256: str
    token_bytes_pt_sha256: str
    determinism_verified: bool  # train-twice hash comparison (critique A-1)


def iter_corpus_docs(corpus_path: Path) -> Iterator[dict]:
    """Merged-corpus stream via the core reader (promoted at third consumer -
    the chunking work, honoring the design-review deferral)."""
    return iter_jsonl_docs(corpus_path, missing_hint="run: cankar corpus merge")


def iter_corpus_texts(corpus_path: Path) -> Iterator[str]:
    """Document texts in file order - the deterministic training stream."""
    for doc in iter_corpus_docs(corpus_path):
        yield doc["text"]


def train_encoding(corpus_path: Path, vocab_size: int) -> tiktoken.Encoding:
    """rustbpe training + tiktoken Encoding construction, per nanochat's
    RustBPETokenizer.train_from_iterator (vendored recipe)."""
    n_ranks = vocab_size - len(SPECIAL_TOKENS)
    if n_ranks < 256:
        raise CankarError(f"vocab_size {vocab_size} leaves {n_ranks} ranks; need >= 256")
    tok = rustbpe.Tokenizer()
    tok.train_from_iterator(iter_corpus_texts(corpus_path), n_ranks, pattern=SPLIT_PATTERN)
    mergeable_ranks = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    offset = len(mergeable_ranks)
    special_tokens = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    return tiktoken.Encoding(
        name="rustbpe",
        pat_str=tok.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


def token_bytes_tensor(enc: tiktoken.Encoding) -> torch.Tensor:
    """nanochat's token_bytes.pt contract: len == n_vocab, 0 at special ids,
    raw byte length elsewhere (tok_train.py lines 72-91)."""
    special_ids = {enc.encode_single_token(s) for s in enc.special_tokens_set}
    counts = [
        0 if tid in special_ids else len(enc.decode_single_token_bytes(tid))
        for tid in range(enc.n_vocab)
    ]
    return torch.tensor(counts, dtype=torch.int32, device="cpu")


def save_artifacts(enc: tiktoken.Encoding, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / "tokenizer.pkl"
    with pkl.open("wb") as f:
        pickle.dump(enc, f)
    tb = out_dir / "token_bytes.pt"
    with tb.open("wb") as f:
        torch.save(token_bytes_tensor(enc), f)
    return pkl, tb


def encoding_fingerprint(enc: tiktoken.Encoding) -> str:
    """Stable digest of the learned vocab - the train-twice comparison key.
    Public API only (design-review 2026-07): in this construction rank == id,
    so iterating ids below the specials walks tokens in rank order."""
    h = hashlib.sha256()
    for tid in range(enc.n_vocab - len(enc.special_tokens_set)):
        h.update(tid.to_bytes(4, "big"))
        h.update(enc.decode_single_token_bytes(tid))
    return h.hexdigest()


def verify_determinism(corpus_path: Path, vocab_size: int, first: tiktoken.Encoding) -> bool:
    """Retrain and compare vocab fingerprints (critique A-1: expected to pass
    with rustbpe==0.1.0; guards version-bump regressions)."""
    second = train_encoding(corpus_path, vocab_size)
    match = encoding_fingerprint(first) == encoding_fingerprint(second)
    if not match:
        log.error("determinism check FAILED: retrain produced a different vocab")
    return match
