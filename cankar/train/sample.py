"""Generate samples from a trained checkpoint (ADR 0016).

Rebuilds the model from the config embedded in the checkpoint (no TOML needed),
loads the weights, and autoregressively generates. Deterministic given
(checkpoint, seed) - the "before" samples archive as (checkpoint + this command).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from cankar.core.encoding import bos_id, load_encoding
from cankar.core.errors import CankarError
from cankar.train.checkpoint import load_checkpoint
from cankar.train.config import TrainConfig
from cankar.train.loop import build_model

log = logging.getLogger("cankar.train")


def sample_from_checkpoint(
    ckpt_path: Path,
    device: str,
    prompt: str = "",
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    n_samples: int = 3,
) -> list[str]:
    state = load_checkpoint(ckpt_path, device)
    config = TrainConfig.model_validate(state["config"])
    enc = load_encoding(config.tokenizer)
    model = build_model(config, enc, device)
    model.load_state_dict(state["model"])
    model.eval()

    prompt_ids = enc.encode_ordinary(prompt)
    context = [bos_id(enc), *prompt_ids]  # generate a fresh doc after BOS
    if len(context) < 2:
        raise CankarError("sample needs a non-empty prompt (the naive generate requires T>1)")
    out: list[str] = []
    for i in range(n_samples):
        with torch.inference_mode():
            new = list(
                model.generate(
                    context, max_tokens=max_tokens, temperature=temperature, top_k=top_k, seed=i
                )
            )
        out.append(enc.decode(prompt_ids + new))  # drop the BOS marker from the shown text
    return out
