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
from typing import Any

import torch

from cankar.core.encoding import load_encoding
from cankar.core.errors import CankarError
from cankar.model.build import build_gpt
from cankar.model.gpt import GPT, GPTConfig
from cankar.train import sft
from cankar.train.checkpoint import load_checkpoint

log = logging.getLogger("cankar.train")


def load_base(init_from: Path, device: str, seq_len: int | None = None) -> tuple[GPT, dict]:
    """Rebuild a checkpoint's model from its OWN gptconfig (ADR 0017).

    Never from a config file: the shape must match the weights, and reading it
    from the checkpoint makes disagreement impossible rather than detected.

    The `gptconfig` guard is a third copy of the one in `train/sample.py` and
    `evals/bpb.py`. That is layering-forced, not accidental: `cankar.core` and
    `cankar.model` sit in ONE import-linter layer, so a helper needing both
    `CankarError` and `GPTConfig` has no legal home today (`sample.py` carries
    the same note).
    """
    state: dict[str, Any] = dict(load_checkpoint(init_from, "cpu"))
    if "gptconfig" not in state:
        raise CankarError(f"{init_from} predates the self-describing checkpoint (ADR 0017)")
    trained_len = state["gptconfig"].get("sequence_len")
    if seq_len is not None and trained_len is not None and seq_len > trained_len:
        # Otherwise this dies mid-run on a bare `assert T <= self.cos.size(1)`
        # inside GPT.forward - on a rented pod, after the data is already loaded.
        raise CankarError(
            f"seq_len {seq_len} exceeds {init_from.name}'s trained sequence_len {trained_len} - "
            "the rotary tables are not built that long"
        )
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
    model, base = load_base(checkpoints_dir / f"{config.init_from}.pt", device, config.seq_len)

    data = sft.load_pairs(train_pairs, enc, config.seq_len)
    # Pairs only, deliberately: scoring replay windows here would let voice
    # retention masquerade as style-transfer progress.
    holdout = sft.load_pairs(holdout_pairs, enc, config.seq_len)
    if not data.examples:
        raise CankarError(f"no usable pairs in {train_pairs}")
    if config.uses_rehearsal:
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
        "%s: from %s (step %d) | %d pairs -> %d pair examples + %d replay = %d | "
        "%d target tokens (%.1f%% replay by token, %.1f%% by optimizer step) | "
        "%d steps/epoch | %d steps (%.1f epochs) | %d held-out | lr_scale %.2f",
        config.name,
        config.init_from,
        base.get("step", -1),
        data.n_pairs,
        data.n_pair_examples,
        data.n_rehearsal,
        len(data.examples),
        data.n_target_tokens,
        100 * data.rehearsal_token_frac,
        100 * sft.rehearsal_step_frac(data, config.batch_size, config.seed),
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
    seen_tokens = 0  # real, not batch_size*seq_len: collate pads to the batch's
    t0 = time.monotonic()  # own longest (median 194 vs 512), so the nominal
    #                        figure overstates throughput ~2.6x - and this is
    #                        the number GPU spend gets sized from.
    for epoch in range(int(config.epochs) + 1):
        for x, y in sft.iter_batches(data, config.batch_size, config.seed, epoch):
            if step >= total_steps:
                break
            x, y = x.to(device), y.to(device)
            seen_tokens += x.numel()
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
                    seen_tokens / (time.monotonic() - t0),
                )
            if step and step % config.eval_every == 0:
                log.info(
                    "step %d | HELD-OUT loss %.4f",
                    step,
                    evaluate(model, holdout, config.batch_size, device),
                )
            if step and step % config.checkpoint_every == 0:
                save_styler(out, model, base, config, step, data)
            step += 1
        if step >= total_steps:
            break

    final = evaluate(model, holdout, config.batch_size, device)
    log.info("held-out loss after training: %.4f", final)
    save_styler(out, model, base, config, step, data)
    return out


def save_styler(
    out: Path,
    model: GPT,
    base: dict,
    config: sft.SftConfig,
    step: int,
    data: sft.SftData | None = None,
) -> Path:
    """Self-describing like every other checkpoint here (ADR 0017): the gptconfig
    is carried through from the base so anything loading this can rebuild it
    without a config file.

    Records the REALIZED composition, not just the requested `rehearsal_frac`.
    The whole argument of this stage is that requested and realized are different
    numbers - the token share is what the config asks for, the step share is what
    reaches the gradient - so a checkpoint carrying only the request would leave
    the actual mix as a line of stdout on a machine that no longer exists.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    state: dict = {
        "model": model.state_dict(),
        "gptconfig": base["gptconfig"],
        "config": config.model_dump(),
        "step": step,
        "init_from": config.init_from,
        "base_step": base.get("step", -1),
    }
    if data is not None:
        state["composition"] = {
            "n_pairs": data.n_pairs,
            "n_pair_examples": data.n_pair_examples,
            "n_rehearsal": data.n_rehearsal,
            "n_dropped_too_long": data.n_dropped_too_long,
            "n_target_tokens": data.n_target_tokens,
            "rehearsal_token_frac": data.rehearsal_token_frac,
            "rehearsal_step_frac": sft.rehearsal_step_frac(data, config.batch_size, config.seed),
        }
    tmp = out.with_suffix(".pt.tmp")  # write-then-rename: a crash never leaves a torn file
    torch.save(state, tmp)
    tmp.replace(out)
    return out
