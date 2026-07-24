"""Generate samples from a trained checkpoint (ADR 0016).

Rebuilds the model from the config embedded in the checkpoint (no TOML needed),
loads the weights, and generates via the shared generate_text (loop.py).
Deterministic given (checkpoint, seed) - the "before" samples archive as
(checkpoint + this command).
"""

from __future__ import annotations

import logging
from pathlib import Path

from cankar.core.encoding import load_encoding
from cankar.train.checkpoint import load_checkpoint
from cankar.train.config import TrainConfig
from cankar.train.loop import build_model, generate_text

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
    return [
        generate_text(model, enc, prompt, max_tokens, temperature, top_k, seed=i)
        for i in range(n_samples)
    ]
