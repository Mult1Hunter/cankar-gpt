"""Style-transfer SFT loop (Phase 6): fine-tune the Cankar voice onto a prompt.

Separate from `train/loop.py` rather than a flag on it. The two runs differ in
almost everything that matters - data unit (a pair, not a token stream), loss
(masked to the target span, not every token), initialization (always from an
existing checkpoint), and the progress signal (held-out pair loss, not training
CE). What they share is the optimizer/schedule shape, which is small.

The load-bearing difference is the eval. Training CE falls whether or not the
model is learning the task, because the target side is Cankar's own prose and a
checkpoint that already speaks Cankar can score well by ignoring the prompt
entirely. Held-out pair loss on works the base model never saw is the only
number here that can distinguish "learned the mapping" from "recited the voice".
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from cankar.core.encoding import load_encoding
from cankar.core.errors import CankarError
from cankar.model.build import build_gpt
from cankar.model.gpt import GPT, GPTConfig
from cankar.train import sft

log = logging.getLogger("cankar.train")


def load_base(init_from: Path, device: str) -> tuple[GPT, dict]:
    """Rebuild a checkpoint's model from its OWN gptconfig (ADR 0017).

    Never from a config file: the shape must match the weights, and reading it
    from the checkpoint makes disagreement impossible rather than detected.
    """
    if not init_from.exists():
        raise CankarError(f"no checkpoint at {init_from} - SFT fine-tunes an existing model")
    state: dict = torch.load(init_from, map_location="cpu", weights_only=False)
    if "gptconfig" not in state:
        raise CankarError(f"{init_from} predates the self-describing checkpoint (ADR 0017)")
    model = build_gpt(GPTConfig(**state["gptconfig"]), device)
    model.load_state_dict(state["model"])
    return model, state


@torch.no_grad()
def evaluate(model: GPT, data: sft.SftData, batch_size: int, device: str) -> float:
    """Mean masked loss over a held-out pair set.

    Summed over target TOKENS, not averaged over batches: batches hold different
    numbers of scored tokens once the prompt is masked out, so a per-batch mean
    would silently weight short targets more heavily.
    """
    model.eval()
    total, n = 0.0, 0
    for x, y in sft.iter_batches(data, batch_size, seed=0, epoch=0):
        x, y = x.to(device), y.to(device)
        scored = int((y != sft.IGNORE_INDEX).sum())
        if scored == 0:
            continue
        total += model(x, targets=y).item() * scored
        n += scored
    model.train()
    return total / max(1, n)


def train_styler(
    config: sft.SftConfig,
    train_pairs: Path,
    holdout_pairs: Path,
    checkpoints_dir: Path,
    device: str,
    rehearsal_texts: list[str] | None = None,
) -> Path:
    """`rehearsal_texts` are resolved by the caller, not read from a path here.

    The other data this loop takes are pair shards it opens itself; replay comes
    in as text because selecting it (Cankar-only, held-out works dropped, corpus
    revision checked) is `train/data.py`'s job and duplicating that resolution
    here is how the two would drift apart.
    """
    torch.manual_seed(config.seed)
    enc = load_encoding(config.tokenizer)
    model, base = load_base(checkpoints_dir / f"{config.init_from}.pt", device)

    data = sft.load_pairs(train_pairs, enc, config.seq_len)
    # Pairs only, deliberately: scoring replay windows here would let voice
    # retention masquerade as style-transfer progress.
    holdout = sft.load_pairs(holdout_pairs, enc, config.seq_len)
    if not data.examples:
        raise CankarError(f"no usable pairs in {train_pairs}")
    if config.rehearsal_frac > 0:
        if not rehearsal_texts:
            raise CankarError(
                f"rehearsal_frac is {config.rehearsal_frac} but no replay texts were passed - "
                "a silently pair-only run would look like replay that did not help"
            )
        data = sft.with_rehearsal(data, rehearsal_texts, enc, config)

    spe = sft.steps_per_epoch(data, config.batch_size)
    total_steps = max(1, int(spe * config.epochs))
    optimizer = model.setup_optimizer(weight_decay=config.weight_decay)
    # Scale EVERY group, not just the matrices. nanochat tunes six groups
    # independently and their ratios are load-bearing; uniform scaling lowers
    # the schedule without disturbing them.
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"] * config.lr_scale
        group["lr"] = group["initial_lr"]
    log.info(
        "%s: from %s (step %d) | %d pairs -> %d examples (+%d replay) | %d target tokens "
        "(%.1f%% replay) | %d steps/epoch | %d steps (%.1f epochs) | %d held-out | "
        "lr_scale %.2f",
        config.name,
        config.init_from,
        base.get("step", -1),
        data.n_pairs,
        len(data.examples),
        data.n_rehearsal,
        data.n_target_tokens,
        100 * data.rehearsal_token_frac,
        spe,
        total_steps,
        config.epochs,
        len(holdout.examples),
        config.lr_scale,
    )
    log.info(
        "held-out loss before training: %.4f", evaluate(model, holdout, config.batch_size, device)
    )

    out = checkpoints_dir / f"{config.name}.pt"
    model.train()
    step = 0
    t0 = time.monotonic()
    for epoch in range(int(config.epochs) + 1):
        for x, y in sft.iter_batches(data, config.batch_size, config.seed, epoch):
            if step >= total_steps:
                break
            x, y = x.to(device), y.to(device)
            mult = sft.lr_multiplier(step, total_steps, config)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * mult
            loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % config.log_every == 0:
                log.info(
                    "step %d/%d | loss %.4f | lr %.2e | %.0f tok/s",
                    step,
                    total_steps,
                    loss.item(),
                    max(g["lr"] for g in optimizer.param_groups),
                    (step + 1) * config.batch_size * config.seq_len / (time.monotonic() - t0),
                )
            if step and step % config.eval_every == 0:
                log.info(
                    "step %d | HELD-OUT loss %.4f",
                    step,
                    evaluate(model, holdout, config.batch_size, device),
                )
            if step and step % config.checkpoint_every == 0:
                save_styler(out, model, base, config, step)
            step += 1
        if step >= total_steps:
            break

    final = evaluate(model, holdout, config.batch_size, device)
    log.info("held-out loss after training: %.4f", final)
    save_styler(out, model, base, config, step)
    return out


def save_styler(out: Path, model: GPT, base: dict, config: sft.SftConfig, step: int) -> Path:
    """Self-describing like every other checkpoint here (ADR 0017): the gptconfig
    is carried through from the base so anything loading this can rebuild it
    without a config file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "gptconfig": base["gptconfig"],
            "config": config.model_dump(),
            "step": step,
            "init_from": config.init_from,
            "base_step": base.get("step", -1),
        },
        out,
    )
    return out
