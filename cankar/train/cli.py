"""Train-stage CLI (ADR 0007): the only argparse holder for this stage.

cankar train run   [--config configs/train/tinycankar.toml] [--resume]
cankar train sample --checkpoint checkpoints/tinycankar.pt [--prompt ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from cankar.core.paths import checkpoints_dir, train_config
from cankar.train.config import load_train_config
from cankar.train.loop import train
from cankar.train.sample import sample_from_checkpoint

log = logging.getLogger("cankar.train")


def _device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run(args: argparse.Namespace) -> int:
    config = load_train_config(args.config)
    device = _device(args.device)
    log.info("training on %s", device)
    train(config, checkpoints_dir(), device, resume=args.resume)
    return 0


def _sample(args: argparse.Namespace) -> int:
    samples = sample_from_checkpoint(
        args.checkpoint,
        _device(args.device),
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        n_samples=args.n,
    )
    for i, text in enumerate(samples, 1):
        print(f"--- sample {i} ---\n{text}\n")
    return 0


def register(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="train (or resume) a model from a config (ADR 0016)")
    r.add_argument("--config", type=Path, default=train_config("tinycankar"))
    r.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    r.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    r.set_defaults(func=_run)

    s = sub.add_parser("sample", help="generate text from a trained checkpoint")
    s.add_argument("--checkpoint", type=Path, default=checkpoints_dir() / "tinycankar.pt")
    s.add_argument("--prompt", default="Bilo je", help="seed prompt (needs >=1 word)")
    s.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--top-k", type=int, default=50, dest="top_k")
    s.add_argument("-n", type=int, default=3, help="number of samples")
    s.add_argument("--device", default=None)
    s.set_defaults(func=_sample)
