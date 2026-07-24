"""Compute-environment shim for the vendored GPT (ADR 0016).

A minimal stand-in for the three `nanochat.common` symbols that `gpt.py` and
`flash_attention.py` need - `COMPUTE_DTYPE` (bf16 on Ampere+ CUDA incl. the
4070 Ti Super at SM 8.9, fp32 on CPU / pre-Ampere) and `print0` (routed through
logging, since the structure law bans `print()` outside cli.py). This avoids
dragging in nanochat's DDP/wandb/logging harness. Named `compute.py`, not
`common.py` - that basename is banned (tests/structure/test_layout.py).
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("cankar.model")

_DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _detect_compute_dtype() -> torch.dtype:
    """bf16 on Ampere+ CUDA, else fp32 (fp16 training needs a GradScaler we do
    not ship). Override with CANKAR_DTYPE={float32,bfloat16,float16}. Matches
    nanochat's _detect_compute_dtype so the drift test sees identical dtype."""
    env = os.environ.get("CANKAR_DTYPE")
    if env is not None:
        if env not in _DTYPE_MAP:
            raise ValueError(f"CANKAR_DTYPE must be one of {sorted(_DTYPE_MAP)}, got {env!r}")
        return _DTYPE_MAP[env]
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
        return torch.bfloat16
    return torch.float32


COMPUTE_DTYPE = _detect_compute_dtype()


def print0(s: str = "", **kwargs: object) -> None:
    """nanochat's rank-0 print, routed through logging (structure law: no
    print() outside cli.py). kwargs (end/flush) are ignored - the GPT call sites
    pass plain strings."""
    log.info("%s", s)
