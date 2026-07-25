"""Training config (ADR 0016) - a validated pydantic model loaded from TOML.

One config fully determines a run (model shape, data, optimizer, schedule, seed)
so a checkpoint + config + seed reproduce the metrics. Committed presets live in
configs/train/.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from cankar.core.errors import CankarError


class TrainConfig(BaseModel):
    """Everything a run needs. Field names map onto GPTConfig / AdamW / the loop."""

    name: str = "tinycankar"
    tokenizer: str = "v8192"  # data/tokenizer/<name>/ (frozen; ADR 0011)
    seed: int = 20260724

    # model shape (-> cankar.model.gpt.GPTConfig). window_pattern "L" = full
    # context; sliding windows are a long-context optimization we do not need
    # at these sizes (architect critique).
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    n_kv_head: int = 4
    window_pattern: str = "L"

    # data / batching
    seq_len: int = 512  # training window; must be <= the chunker's 2048 (ADR 0012)
    batch_size: int = 16

    # optimizer + schedule (LRs are scaled by 1/sqrt(dmodel) inside setup_optimizer)
    max_steps: int = 3000
    warmup_steps: int = 100
    min_lr_frac: float = 0.1  # cosine decays to this fraction of the peak LR
    weight_decay: float = 0.0
    matrix_lr: float = 0.002  # AdamW rate for the transformer matrices (was Muon 0.02)
    grad_clip: float = 1.0

    # checkpointing + logging + sampling
    log_every: int = 50
    checkpoint_every: int = 500  # steps between checkpoints (reboot-safe resume)
    sample_every: int = 500
    sample_max_tokens: int = 120
    # non-empty: the naive generate needs T>1 (smear op); a typo'd empty prompt
    # should fail at config load, not at step 0 (design-review)
    sample_prompt: str = Field("Bilo je", min_length=1)

    # cost discipline (B3): wall-clock budget in hours. On exceeding it the loop
    # checkpoints and stops gracefully (resume with --resume) - the guard that
    # keeps a rented cloud pod from over-running. None = no cap.
    max_hours: float | None = Field(default=None, gt=0)


def load_train_config(path: Path) -> TrainConfig:
    if not path.exists():
        raise CankarError(f"no train config at {path}")
    return TrainConfig.model_validate(tomllib.loads(path.read_text()))
