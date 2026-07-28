"""Evals-stage CLI subcommands - the ONLY argparse holder for this stage.

Registered under the single `cankar` console entry (ADR 0007):
    cankar evals holdout-freeze --name v8192      # freeze the held-out set (ADR 0013)
    cankar evals style-train                       # train the style classifier (ADR 0015)
    cankar evals bpb --checkpoint <path>           # held-out BPB for a checkpoint (ADR 0017)
    cankar evals bpb-freeze                        # score the canonical set, commit provenance
    cankar evals deploy-check                      # is the style scorer fit for deploy? (MF-3)
    cankar evals judge --checkpoint <path>         # LLM meaning-judge (eval pillar #3)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import torch

from cankar.core.encoding import load_encoding
from cankar.core.errors import CankarError
from cankar.core.holdout import holdout_excludes, load_holdout
from cankar.core.manifest import (
    git_sha,
    library_versions,
    sha256_of,
    utc_now_iso,
    write_manifest,
)
from cankar.core.paths import (
    PairSet,
    bpb_manifest,
    bpb_report,
    checkpoints_dir,
    drafts_shard,
    holdout_manifest,
    holdout_report,
    judge_raw,
    judge_report,
    merged_shard,
    pairs_shard,
    style_deploy_manifest,
    style_deploy_report,
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
    # The deploy verdict is measured separately and belongs to a specific
    # artifact. Carry it forward ONLY if these are the same weights; anything
    # else resets to PENDING, which blocks claims rather than publishing a
    # verdict about a model nobody measured.
    deploy_status = style.deploy_status_for(style_deploy_manifest(), sha256_of(out_model))

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
        deploy_validated=deploy_status,
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
        holdout_sha256=sha256_of(holdout_manifest()),
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


def _deploy_check(args: argparse.Namespace) -> int:
    """MF-3: the classifier was trained on 1900s prose vs 1900s prose, and is
    deployed against modern plain Slovene. Measure it there before any claim
    rides on it."""
    from cankar.evals.style import deploy_check, load_style_manifest, write_deploy_report

    manifest = load_style_manifest(style_manifest())
    artifact = style_model(args.name)
    model = joblib.load(artifact)

    pairs = [
        json.loads(x) for x in pairs_shard(PairSet.HOLDOUT).open(encoding="utf-8") if x.strip()
    ]
    drafts = [
        json.loads(x)
        for x in drafts_shard().open(encoding="utf-8")
        if x.strip() and json.loads(x)["arm"] == "register"
    ]
    check = deploy_check(
        model,
        cankar=[r["cankar"] for r in pairs],
        destyled=[r["plain"] for r in pairs],
        modern=[d["text"] for d in drafts],
    )
    record = style.DeployRecord(
        artifact_sha256=sha256_of(artifact),
        corpus_sha256=manifest.corpus_sha256,
        git_sha=git_sha(),
        created_at=utc_now_iso(),
        check=check,
    )
    write_manifest(record, style_deploy_manifest())
    write_deploy_report(
        style_deploy_report(), check, manifest.metrics.roc_auc_mean, manifest.corpus_sha256
    )
    log.info(
        "deploy check: AUC %.3f (train task %.3f) | style effect %+.3f vs topic effect %+.3f -> %s",
        check.deploy_auc,
        manifest.metrics.roc_auc_mean,
        check.style_effect,
        check.topic_effect,
        check.verdict.value,
    )
    log.info("-> %s", style_deploy_report())
    return 0


def _judge(args: argparse.Namespace) -> int:
    """Eval pillar #3. Generates styler output for held-out pairs and fresh
    drafts, judges every item, and validates the JUDGE against blind controls
    before reporting a single score.

    The control check gates the report deliberately: an unvalidated instrument
    producing plausible numbers is how the style classifier came to be trusted
    for a task it fails at, and that mistake is cheap to repeat."""
    import os
    import time

    from anthropic import Anthropic

    from cankar.evals import judge

    pairs = [
        json.loads(x) for x in pairs_shard(PairSet.HOLDOUT).open(encoding="utf-8") if x.strip()
    ][: args.limit]
    items = judge.build_controls(pairs, n=args.controls)

    # styler output for the same held-out sources, so the scored items and the
    # REAL_CANKAR control share a source and differ only in the candidate
    from cankar.evals.judge import ControlKind, JudgeItem
    from cankar.train import sample as train_sample

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outputs = train_sample.style_transfer(args.checkpoint, [p["plain"] for p in pairs], device)
    items += [
        JudgeItem(f"holdout-{i}", p["plain"], out, ControlKind.NONE)
        for i, (p, out) in enumerate(zip(pairs, outputs, strict=True))
    ]

    # Fresh drafts too: the GAP between these and the held-out set is the honest
    # headline (ROADMAP Phase 6). Held-out sources are de-styled Cankar, so they
    # still carry his subject matter and rhythm; the drafts are modern topics in
    # the same plain register by design invariant #1, which makes topic the only
    # variable that moves.
    drafts = [
        json.loads(x)
        for x in drafts_shard().open(encoding="utf-8")
        if x.strip() and json.loads(x)["arm"] == "register"
    ]
    draft_out = train_sample.style_transfer(args.checkpoint, [d["text"] for d in drafts], device)
    items += [
        JudgeItem(f"draft-{i}", d["text"], out, ControlKind.NONE)
        for i, (d, out) in enumerate(zip(drafts, draft_out, strict=True))
    ]

    est = judge.estimate_cost(items)
    log.info(
        "judging %d items (%d controls + %d scored): ~%d in / %d out tokens, ~$%.2f batch",
        est.n_requests,
        est.n_requests - len(outputs),
        len(outputs),
        est.input_tokens,
        est.output_tokens,
        est.usd_batch,
    )
    if args.dry_run:
        log.info("dry run - nothing submitted")
        return 0

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    requests = [judge.build_request(i, args.model) for i in items]
    batch = client.with_options(max_retries=0).messages.batches.create(
        requests=requests  # type: ignore[arg-type]
    )
    log.info("submitted judge batch %s (%d requests)", batch.id, len(requests))

    delay = 30
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        log.info("judge batch: %s, waiting %ds", b.processing_status, delay)
        time.sleep(delay)
        delay = min(delay * 2, 300)

    # raw FIRST, parse second - see judge.append_raw
    records = [json.loads(r.model_dump_json()) for r in client.messages.batches.results(batch.id)]
    judge.append_raw(judge_raw(), records)
    log.info("saved %d raw responses -> %s", len(records), judge_raw())
    verdicts = judge.parse_raw(judge_raw())

    outcome = judge.check_controls(items, verdicts)

    def series(prefix: str) -> list[judge.Verdict]:
        return [v for k, v in verdicts.items() if k.startswith(prefix)]

    def means_of(rows: list[judge.Verdict]) -> dict[str, float]:
        return {
            axis.value: sum(getattr(v, axis.value) for v in rows) / len(rows) for axis in judge.Axis
        }

    holdout_rows, draft_rows = series("holdout-"), series("draft-")
    scored = holdout_rows + draft_rows
    if not scored:
        raise CankarError(
            f"no scored verdicts among {len(verdicts)} parsed - the item ids the batch "
            "was built with do not match the ones being read back"
        )
    means = means_of(scored)
    judge.write_judge_report(
        judge_report(),
        outcome,
        means,
        len(scored),
        holdout=means_of(holdout_rows) if holdout_rows else None,
        drafts=means_of(draft_rows) if draft_rows else None,
        n_holdout=len(holdout_rows),
        n_drafts=len(draft_rows),
        corpus_sha=sha256_of(merged_shard()),
    )

    for f in outcome.failures:
        log.error("CONTROL FAILED - %s", f)
    log.info(
        "judge %s | styler-v1 meaning %.2f voice %.2f fluency %.2f (n=%d) -> %s",
        "USABLE" if outcome.usable else "NOT USABLE",
        means["meaning"],
        means["voice"],
        means["fluency"],
        len(scored),
        judge_report(),
    )
    # A failed control run is not a bad score, it is an unusable instrument -
    # exiting 0 would let it be quoted anyway.
    return 0 if outcome.usable else 1


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

    d = sub.add_parser(
        "deploy-check",
        help="measure the style classifier on its DEPLOY task, not its training task (MF-3)",
    )
    d.add_argument("--name", default="v1", help="classifier artifact suffix")
    d.set_defaults(func=_deploy_check)

    j = sub.add_parser("judge", help="LLM meaning-judge over styler output (eval pillar #3)")
    j.add_argument("--checkpoint", type=Path, default=checkpoints_dir() / "styler-v1.pt")
    j.add_argument("--limit", type=int, default=100, help="held-out pairs to score")
    j.add_argument("--controls", type=int, default=12, help="pairs sampled for blind controls")
    j.add_argument("--model", default="claude-sonnet-5")
    j.add_argument("--device", default=None)
    j.add_argument("--dry-run", action="store_true", help="estimate cost, submit nothing")
    j.set_defaults(func=_judge)

    f = sub.add_parser(
        "bpb-freeze",
        help="score the canonical checkpoints and commit the provenance (ADR 0016)",
    )
    f.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    f.set_defaults(func=_bpb_freeze)
