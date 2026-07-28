"""Generate samples from a trained checkpoint (ADR 0016).

Rebuilds the model from the config embedded in the checkpoint (no TOML needed),
loads the weights, and generates via the shared generate_text (loop.py).
Deterministic given (checkpoint, seed) - the "before" samples archive as
(checkpoint + this command).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from cankar.core.encoding import load_encoding
from cankar.core.errors import CankarError
from cankar.model.build import build_gpt
from cankar.model.gpt import GPTConfig
from cankar.train.checkpoint import load_checkpoint
from cankar.train.config import TrainConfig
from cankar.train.loop import generate_text

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
    if (
        "gptconfig" not in state
    ):  # layering-forced dup of the bpb.py guard (model can't import core)
        raise CankarError(f"{ckpt_path} predates the self-describing checkpoint (ADR 0017)")
    config = TrainConfig.model_validate(state["config"])
    enc = load_encoding(config.tokenizer)
    model = build_gpt(GPTConfig(**state["gptconfig"]), device)  # self-describing (ADR 0017)
    model.load_state_dict(state["model"])
    model.eval()
    return [
        generate_text(model, enc, prompt, max_tokens, temperature, top_k, seed=i)
        for i in range(n_samples)
    ]


@dataclass(frozen=True)
class StyledOutput:
    """One generation. `stopped_at_end` distinguishes a finished passage from one
    cut off at max_tokens - indistinguishable in the text alone, and the first
    thing a reader asks when the fluency score is at the floor."""

    text: str
    stopped_at_end: bool


def style_transfer(
    ckpt_path: Path,
    sources: list[str],
    device: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 20260728,
) -> list[StyledOutput]:
    """Run a styler checkpoint over plain-Slovene sources, one output each.

    Separate from `sample_from_checkpoint` rather than a flag on it: that one
    continues a free-text prompt with the base LM, while this wraps each source
    in the chat specials the SFT format trained on and stops at
    `<|assistant_end|>`. Sharing an entry point would mean a bare prompt string
    silently producing a continuation where a rewrite was wanted.

    Validates its config with SftConfig, not TrainConfig - a styler checkpoint
    carries SFT fields (rehearsal_frac, lr_scale) that TrainConfig would reject.
    """
    from cankar.train.sft import ASSISTANT_END, SftConfig, build_prompt, special_ids

    state = load_checkpoint(ckpt_path, device)
    if "gptconfig" not in state:
        raise CankarError(f"{ckpt_path} predates the self-describing checkpoint (ADR 0017)")
    config = SftConfig.model_validate(state["config"])
    enc = load_encoding(config.tokenizer)
    sp = special_ids(enc)
    model = build_gpt(GPTConfig(**state["gptconfig"]), device)
    model.load_state_dict(state["model"])
    model.eval()

    outputs: list[StyledOutput] = []
    for i, source in enumerate(sources):
        emitted: list[int] = []
        stopped = False
        for tok in model.generate(
            build_prompt(source, enc, sp),
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed + i,
        ):
            if tok == sp[ASSISTANT_END]:
                stopped = True
                break
            emitted.append(tok)
        outputs.append(StyledOutput(enc.decode(emitted).strip(), stopped))
    return outputs
