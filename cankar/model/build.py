"""Construct a GPT (ADR 0017). Shared by the train loop and the BPB-on-checkpoint
eval so both build the model identically - a bottom-layer helper both stages import.
"""

from __future__ import annotations

from cankar.model.gpt import GPT, GPTConfig


def build_gpt(config: GPTConfig, device: str) -> GPT:
    """GPT -> init_weights (non-optional: the module holds fake-init tensors until
    then, ADR 0016) -> move to device. The single construction path."""
    model = GPT(config)
    model.init_weights()
    return model.to(device)
