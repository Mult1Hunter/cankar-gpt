"""Train-stage CLI (ADR 0007): the only argparse holder for this stage.

cankar train run   [--config configs/train/tinycankar.toml] [--resume]
cankar train sft   [--config configs/train/styler-v1.toml]   # Phase 6
cankar train sample --checkpoint checkpoints/tinycankar.pt [--prompt ...]
cankar train style  --checkpoint checkpoints/styler-v1.pt   # -> JSONL for evals
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from cankar.core.paths import (
    PairSet,
    checkpoints_dir,
    chunks_manifest,
    chunks_shard,
    drafts_shard,
    holdout_manifest,
    pairs_shard,
    styled_outputs,
    train_config,
)
from cankar.train.config import load_sft_config, load_train_config
from cankar.train.data import cankar_chunk_texts
from cankar.train.loop import train
from cankar.train.sample import sample_from_checkpoint, style_transfer
from cankar.train.sft_loop import train_styler

log = logging.getLogger("cankar.train")


def _device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run(args: argparse.Namespace) -> int:
    config = load_train_config(args.config)
    if args.max_hours is not None:
        config = config.model_copy(update={"max_hours": args.max_hours})
    device = _device(args.device)
    log.info("training on %s", device)
    train(config, checkpoints_dir(), device, resume=args.resume, init_from=args.init_from)
    return 0


def _sft(args: argparse.Namespace) -> int:
    """Phase 6: fine-tune the Cankar voice onto a plain-Slovene prompt."""
    config = load_sft_config(args.config)
    device = _device(args.device)
    log.info("style-transfer SFT on %s", device)
    # Resolved here rather than inside the loop: this is the layer that owns
    # artifact paths, and cankar_chunk_texts carries the holdout exclusion and
    # the corpus-revision check with it.
    replay = (
        cankar_chunk_texts(chunks_shard(), holdout_manifest(), chunks_manifest())
        if config.uses_rehearsal
        else None
    )
    out = train_styler(
        config,
        pairs_shard(PairSet.TRAIN),
        pairs_shard(PairSet.HOLDOUT),
        checkpoints_dir(),
        device,
        rehearsal_texts=replay,
    )
    log.info("styler -> %s", out)
    return 0


def _style(args: argparse.Namespace) -> int:
    """Generate styler output for the Phase 6 eval sources and write the shard
    `cankar evals judge` reads.

    A separate command rather than a step inside the judge because `evals` may
    not import `train` (ADR 0007) - and because generation and judging have very
    different costs, so coupling them would mean every re-judge re-generates and
    every re-generate re-buys a batch."""
    import json

    device = _device(args.device)
    pairs = [
        json.loads(x) for x in pairs_shard(PairSet.HOLDOUT).open(encoding="utf-8") if x.strip()
    ][: args.limit]
    drafts = [
        json.loads(x)
        for x in drafts_shard().open(encoding="utf-8")
        if x.strip() and json.loads(x)["arm"] == "register"
    ]
    log.info("styling %d held-out sources + %d drafts on %s", len(pairs), len(drafts), device)

    rows = []
    for series, sources in (
        ("holdout", [p["plain"] for p in pairs]),
        ("draft", [d["text"] for d in drafts]),
    ):
        results = style_transfer(args.checkpoint, sources, device)
        rows += [
            {
                "item_id": f"{series}-{i}",
                "series": series,
                "source": src,
                "candidate": r.text,
                "stopped_at_end": r.stopped_at_end,
            }
            for i, (src, r) in enumerate(zip(sources, results, strict=True))
        ]

    out = styled_outputs(args.checkpoint.stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    truncated = sum(not r["stopped_at_end"] for r in rows)
    log.info(
        "wrote %d styled outputs (%d truncated at max_tokens) -> %s", len(rows), truncated, out
    )
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
    r.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="wall-clock budget: checkpoint+stop when reached",
    )
    r.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="seed model weights from a checkpoint (specialization; fresh optimizer)",
    )
    r.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    r.set_defaults(func=_run)

    f = sub.add_parser("sft", help="Phase 6: style-transfer fine-tune on the pairs")
    # Defaults to the committed preset, matching `train run`. One file = one
    # reproducible run (configs/README.md, ADR 0003) - a bare `cankar train sft`
    # must not quietly diverge from the record of the shipped run.
    f.add_argument("--config", type=Path, default=train_config("styler-v1"))
    f.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    f.set_defaults(func=_sft)

    y = sub.add_parser("style", help="styler output for the eval sources -> JSONL (Phase 6)")
    y.add_argument("--checkpoint", type=Path, default=checkpoints_dir() / "styler-v1.pt")
    y.add_argument("--limit", type=int, default=285, help="held-out pairs to style")
    y.add_argument("--device", default=None)
    y.set_defaults(func=_style)

    s = sub.add_parser("sample", help="generate text from a trained checkpoint")
    s.add_argument("--checkpoint", type=Path, default=checkpoints_dir() / "tinycankar.pt")
    s.add_argument("--prompt", default="Bilo je", help="seed prompt (needs >=1 word)")
    s.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--top-k", type=int, default=50, dest="top_k")
    s.add_argument("-n", type=int, default=3, help="number of samples")
    s.add_argument("--device", default=None)
    s.set_defaults(func=_sample)
