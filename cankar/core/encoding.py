"""Load a trained tiktoken Encoding from its artifact dir.

Shared by the tokenizer stage that writes it and every stage that scores with
it (evals now; model at Phase 3) - promoted to core at the second consumer
(design-review 2026-07), same rationale as core.textsim.
"""

from __future__ import annotations

import pickle

import tiktoken

from cankar.core.errors import CankarError
from cankar.core.paths import tokenizer_dir

BOS = "<|bos|>"  # the tokenizer's document-boundary special token (tokenizer.vendored)


def load_encoding(name: str) -> tiktoken.Encoding:
    pkl = tokenizer_dir(name) / "tokenizer.pkl"
    if not pkl.exists():
        raise CankarError(f"no tokenizer '{name}' at {pkl} (run: cankar tokenizer train)")
    with pkl.open("rb") as f:
        enc: tiktoken.Encoding = pickle.load(f)
    return enc


def bos_id(enc: tiktoken.Encoding) -> int:
    """The <|bos|> token id. Resolved here (not by importing the tokenizer stage)
    so evals/train stay independent siblings (bpb.py did this inline)."""
    return enc.encode_single_token(BOS)
