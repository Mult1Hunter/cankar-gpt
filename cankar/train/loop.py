"""Training loop (ADR 0016) - lean, single-device, AdamW.

Builds the vendored GPT from the config, trains on the Cankar chunk stream, logs
CE loss + tokens/sec (the calibration number the ROADMAP wants before the long
run), checkpoints periodically for reboot-safe resume, and emits sample text so
you can watch the "before" take shape. No DDP/fp8/torch.compile - those are
Phase-3 concerns and irrelevant at 10-30M params on one device.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import tiktoken
import torch

from cankar.core.encoding import bos_id, load_encoding
from cankar.core.errors import CankarError
from cankar.core.paths import chunks_manifest, chunks_shard, holdout_manifest
from cankar.model.build import build_gpt
from cankar.model.gpt import GPT, GPTConfig
from cankar.train.checkpoint import load_checkpoint, save_checkpoint
from cankar.train.config import CorpusScope, TrainConfig
from cankar.train.data import (
    TokenizedCorpus,
    all_chunk_texts,
    cankar_chunk_texts,
    iter_batches,
    steps_per_epoch,
)

log = logging.getLogger("cankar.train")


def build_model(config: TrainConfig, enc: tiktoken.Encoding, device: str) -> GPT:
    gcfg = GPTConfig(
        sequence_len=config.seq_len,
        vocab_size=enc.n_vocab,  # GPT pads to a multiple of 64 internally
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        n_embd=config.n_embd,
        window_pattern=config.window_pattern,
    )
    return build_gpt(gcfg, device)


def lr_multiplier(step: int, config: TrainConfig) -> float:
    """Linear warmup then cosine decay to min_lr_frac of the peak."""
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return config.min_lr_frac + (1 - config.min_lr_frac) * cosine


def generate_text(
    model: GPT,
    enc: tiktoken.Encoding,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int = 0,
) -> str:
    """BOS-seed the prompt, generate, render. The single generation path shared
    by the loop's periodic sampling and `cankar train sample` (design-review)."""
    context = [bos_id(enc), *enc.encode_ordinary(prompt)]
    if len(context) < 2:
        raise CankarError("sample needs a non-empty prompt (the naive generate needs T>1)")
    new = list(
        model.generate(
            context, max_tokens=max_tokens, temperature=temperature, top_k=top_k, seed=seed
        )
    )
    return prompt + enc.decode(new)


def _sample(model: GPT, enc: tiktoken.Encoding, config: TrainConfig) -> str:
    model.eval()
    text = generate_text(model, enc, config.sample_prompt, config.sample_max_tokens)
    model.train()
    return text


def train(config: TrainConfig, out_dir: Path, device: str, resume: bool = False) -> Path:
    torch.manual_seed(config.seed)
    enc = load_encoding(config.tokenizer)
    model = build_model(config, enc, device)
    optimizer = model.setup_optimizer(matrix_lr=config.matrix_lr, weight_decay=config.weight_decay)
    loader = all_chunk_texts if config.corpus is CorpusScope.ALL else cankar_chunk_texts
    texts = loader(chunks_shard(), holdout_manifest(), chunks_manifest())
    corpus = TokenizedCorpus.build(texts, enc)
    spe = steps_per_epoch(corpus, config.batch_size, config.seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "%s: %.1fM params | %d train tokens | %d steps/epoch | %d steps (~%.1f epochs)",
        config.name,
        n_params / 1e6,
        corpus.n_tokens,
        spe,
        config.max_steps,
        config.max_steps / max(spe, 1),
    )

    ckpt = out_dir / f"{config.name}.pt"
    start_step = 0
    if resume and ckpt.exists():
        state = load_checkpoint(ckpt, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].to("cpu"))
        start_step = state["step"]
        log.info("resumed from %s at step %d", ckpt, start_step)

    batches = iter_batches(
        corpus, config.batch_size, config.seq_len, config.seed, device, start_step=start_step
    )
    tokens_per_step = config.batch_size * config.seq_len
    model.train()
    t0 = time.monotonic()
    for step in range(start_step, config.max_steps):
        x, y = next(batches)
        mult = lr_multiplier(step, config)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * mult
        loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed = step + 1

        if step % config.log_every == 0 or completed == config.max_steps:
            dt = time.monotonic() - t0
            tok_s = tokens_per_step * (completed - start_step) / dt if dt > 0 else 0.0
            log.info(
                "step %d/%d | loss %.4f | lr %.2e | %.0f tok/s | epoch %.2f",
                completed,
                config.max_steps,
                loss.item(),
                optimizer.param_groups[-1]["lr"],
                tok_s,
                completed / max(spe, 1),
            )
        if config.sample_every and step % config.sample_every == 0:
            log.info("  sample: %s", _sample(model, enc, config))
        if completed % config.checkpoint_every == 0 and completed < config.max_steps:
            save_checkpoint(ckpt, model, optimizer, completed, config)
            log.info("  checkpoint -> %s (step %d)", ckpt, completed)
        if config.max_hours and (time.monotonic() - t0) / 3600 >= config.max_hours:
            save_checkpoint(ckpt, model, optimizer, completed, config)
            log.info(
                "  max_hours %.3f reached at step %d -> checkpointed, stop (resume with --resume)",
                config.max_hours,
                completed,
            )
            return ckpt

    save_checkpoint(ckpt, model, optimizer, config.max_steps, config)
    log.info("done: %d steps -> %s", config.max_steps, ckpt)
    return ckpt
