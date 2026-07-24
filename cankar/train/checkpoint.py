"""Checkpoint save/load (ADR 0016) - reboot-safe, exact resume.

A checkpoint carries everything needed to continue a run: model + optimizer
state, the step counter (which also fixes the data position - iter_batches is a
pure function of (seed, epoch), so start_step=step replays it exactly), the RNG
state, and the config (so sampling can rebuild the model without the TOML). The
.pt lives under checkpoints/ (gitignored heavy artifact).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypedDict

import torch

from cankar.core.errors import CankarError
from cankar.model.gpt import GPT
from cankar.train.config import TrainConfig


class CheckpointState(TypedDict):
    """The saved run state - a wrong key here silently breaks resume (ADR 0008).
    `gptconfig` makes the checkpoint self-describing: any consumer (evals BPB)
    rebuilds the exact model from it without importing the train stage (ADR 0017)."""

    step: int
    config: dict[str, Any]  # the TrainConfig
    gptconfig: dict[str, Any]  # the resolved GPTConfig (model shape incl. vocab_size)
    model: dict[str, Any]
    optimizer: dict[str, Any]
    torch_rng: torch.Tensor


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainConfig,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")  # write-then-rename: a crash never leaves a torn file
    torch.save(
        {
            "step": step,
            "config": config.model_dump(),
            "gptconfig": dataclasses.asdict(model.config),  # self-describing (ADR 0017)
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
        },
        tmp,
    )
    tmp.replace(path)
    return path


def load_checkpoint(path: Path, device: str) -> CheckpointState:
    if not path.exists():
        raise CankarError(f"no checkpoint at {path} (run: cankar train run)")
    state: CheckpointState = torch.load(path, map_location=device, weights_only=False)
    return state
