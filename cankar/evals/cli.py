"""Evals-stage CLI subcommands - the ONLY argparse holder for this stage.

Registered under the single `cankar` console entry (ADR 0007):
    cankar evals holdout-freeze --name v8192      # freeze the held-out set (ADR 0013)
    cankar evals style-train                       # train the style classifier (ADR 0015)
    cankar evals bpb --checkpoint <path>           # held-out BPB for a checkpoint (ADR 0017)
    cankar evals bpb-freeze                        # score the canonical set, commit provenance
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import torch

from cankar.core.encoding import load_encoding
from cankar.core.holdout import holdout_excludes, load_holdout
from cankar.core.manifest import (
    git_sha,
    library_versions,
    sha256_of,
    utc_now_iso,
    write_manifest,
)
from cankar.core.paths import (
    bpb_manifest,
    bpb_report,
    checkpoints_dir,
    holdout_manifest,
    holdout_report,
    merged_shard,
    style_manifest,
    style_model,
    style_report,
    tokenizer_base_dir,
)
from cankar.evals import bpb, holdout, style

log = logging.getLogger("cankar.evals")


def _holdout_freeze(args: argparse.Namespace) -> int:
    corpus = merged_shard()
    enc = load_encoding(args.name)
    docs = holdout.cankar_docs(corpus)
    log.info("selecting holdout from %d Cankar docs", len(docs))
    params = holdout.HoldoutParams()
    result = holdout.select_holdout(docs, enc, params)
    holdout_tokens = sum(w.n_tokens for w in result.works)
    manifest = holdout.HoldoutManifest(
        corpus_sha256=sha256_of(corpus),
        tokenizer_name=args.name,
        params=params,
        cankar_total_tokens=result.cankar_total_tokens,
        holdout_tokens=holdout_tokens,
        holdout_fraction=round(holdout_tokens / result.cankar_total_tokens, 4),
        git_sha=git_sha(),
        created_at=utc_now_iso(),
        works=result.works,
        also_exclude_urls=result.also_exclude_urls,
    )
    out = write_manifest(manifest, holdout_manifest())
    report = holdout.write_holdout_report(holdout_report(), manifest, result.rejected)
    log.info(
        "froze %d works / %d tokens (%.2f%%), +%d reverse-contained urls -> %s + %s",
        len(result.works),
        holdout_tokens,
        100 * manifest.holdout_fraction,
        len(result.also_exclude_urls),
        out,
        report,
    )
    return 0


def _style_train(args: argparse.Namespace) -> int:
    corpus = merged_shard()
    params = style.StyleParams()
    excludes = holdout_excludes(load_holdout(holdout_manifest()))
    data = style.load_labeled_chunks(corpus, excludes, params)
    log.info(
        "style data: %d chunks (%d Cankar / %d peer), %d groups, %d verse docs dropped",
        len(data.texts),
        int(data.labels.sum()),
        int((1 - data.labels).sum()),
        len(set(data.groups.tolist())),
        data.n_verse_docs_dropped,
    )
    ev = style.evaluate(data, params)
    model = style.fit_shipped(data, params)
    out_model = style_model(args.name)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)

    manifest = style.StyleManifest(
        corpus_sha256=sha256_of(corpus),
        git_sha=git_sha(),
        created_at=utc_now_iso(),
        lib_versions=library_versions("scikit-learn", "numpy", "scipy", "joblib"),
        params=params,
        n_chunks=len(data.texts),
        n_cankar=int(data.labels.sum()),
        n_other=int((1 - data.labels).sum()),
        pos_rate=round(float(data.labels.mean()), 4),
        n_groups=len(set(data.groups.tolist())),
        n_verse_docs_dropped=data.n_verse_docs_dropped,
        n_docs=data.n_docs,
        metrics=ev.fold,
        ablation=ev.ablation,
        per_author_meanp=ev.per_author_meanp,
        artifact_sha256=sha256_of(out_model),
    )
    out = write_manifest(manifest, style_manifest())
    report = style.write_style_report(style_report(), manifest, ev)
    log.info(
        "froze style classifier: ROC-AUC %.3f+/-%.3f -> %s + %s + %s",
        manifest.metrics.roc_auc_mean,
        manifest.metrics.roc_auc_std,
        out_model,
        out,
        report,
    )
    return 0


def _bpb(args: argparse.Namespace) -> int:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    result = bpb.bpb_on_checkpoint(
        args.checkpoint, merged_shard(), holdout_manifest(), tokenizer_base_dir(), device
    )
    log.info(
        "held-out BPB %.4f | %d works | checkpoint step %d (lower is better)",
        result.bpb,
        result.n_works,
        result.step,
    )
    return 0


def _bpb_freeze(args: argparse.Namespace) -> int:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    corpus = merged_shard()
    rows = bpb.score_canonical(
        checkpoints_dir(), corpus, holdout_manifest(), tokenizer_base_dir(), device
    )
    manifest = bpb.BpbManifest(
        corpus_sha256=sha256_of(corpus),
        git_sha=git_sha(),
        created_at=utc_now_iso(),
        device=device,
        lib_versions=library_versions("torch", "tiktoken"),
        checkpoints=rows,
    )
    out = write_manifest(manifest, bpb_manifest())
    report = bpb.write_bpb_report(bpb_report(), manifest)
    log.info(
        "froze held-out BPB: %s -> %s + %s",
        ", ".join(f"{c.name} {c.bpb:.4f}" for c in rows),
        out,
        report,
    )
    return 0


def register(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "holdout-freeze", help="freeze the held-out Cankar set before Phase 3 (ADR 0013)"
    )
    p.add_argument("--name", required=True, help="tokenizer candidate (the selected one)")
    p.set_defaults(func=_holdout_freeze)

    s = sub.add_parser("style-train", help="train + freeze the style classifier (ADR 0015)")
    s.add_argument("--name", default="v1", help="artifact name suffix (checkpoints/style-<name>)")
    s.set_defaults(func=_style_train)

    b = sub.add_parser("bpb", help="held-out bits-per-byte for a trained checkpoint (ADR 0017)")
    b.add_argument("--checkpoint", type=Path, required=True, help="a cankar train checkpoint (.pt)")
    b.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    b.set_defaults(func=_bpb)

    f = sub.add_parser(
        "bpb-freeze",
        help="score the canonical checkpoints and commit the provenance (ADR 0017)",
    )
    f.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    f.set_defaults(func=_bpb_freeze)
