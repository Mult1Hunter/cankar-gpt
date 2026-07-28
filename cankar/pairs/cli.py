"""Pairs-stage CLI subcommands - the ONLY argparse holder for this stage.

Registered under the single `cankar` console entry (ADR 0007):
    cankar pairs segment                  # cut de-styling passages + provenance
    cankar pairs destyle --limit 10000    # Batch API de-styling (spends money)
    cankar pairs destyle --dry-run        # price it first, submit nothing
    cankar pairs destyle --parse-only     # re-parse the raw log, bill nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

from cankar.core.errors import CankarError
from cankar.core.holdout import holdout_excludes, load_holdout
from cankar.core.manifest import (
    git_sha,
    library_versions,
    load_frozen,
    sha256_of,
    utc_now_iso,
    write_manifest,
)
from cankar.core.paths import (
    batch_receipts,
    destyle_raw,
    holdout_manifest,
    merged_shard,
    pairs_manifest,
    pairs_report,
    pairs_samples,
    pairs_shard,
    passages_manifest,
    passages_report,
    passages_shard,
    works_registry,
)
from cankar.core.register import REGISTER_SHA256
from cankar.core.works import load_work_genres
from cankar.pairs import destyle, publish, segment

log = logging.getLogger("cankar.pairs")

# Batch polling: most batches finish in minutes, the ceiling is 24h. Backs off
# to one minute so a long batch costs a handful of API calls, not thousands.
POLL_START_SECONDS = 5
POLL_MAX_SECONDS = 60


def _segment(args: argparse.Namespace) -> int:
    corpus = merged_shard()
    ledger = works_registry(segment.WORKS_LEDGER)
    excludes = holdout_excludes(load_holdout(holdout_manifest()))
    genres = load_work_genres(ledger)
    params = segment.SegmentParams()
    result = segment.segment_corpus(corpus, excludes, genres, params)

    out_shard = segment.write_passages(passages_shard(), result.passages)
    manifest = segment.PassagesManifest(
        segmenter_version=segment.SEGMENTER_VERSION,
        corpus_sha256=sha256_of(corpus),
        holdout_sha256=sha256_of(holdout_manifest()),
        works_sha256=sha256_of(ledger),
        passages_sha256=sha256_of(out_shard),
        git_sha=git_sha(),
        created_at=utc_now_iso(),
        source=segment.SOURCE_WIKIVIR,
        prose_genres=sorted(segment.PROSE_GENRES),
        params=params,
        n_docs=result.n_docs,
        n_passages=len(result.passages),
        n_chars=sum(p.n_chars for p in result.passages),
        reject_counts=result.reject_counts,
        doc_skips=result.doc_skips,
    )
    out_manifest = write_manifest(manifest, passages_manifest())
    report = segment.write_passages_report(passages_report(), manifest)
    log.info(
        "segmented %d docs -> %d passages (%d chars); skipped docs: %s -> %s + %s + %s",
        result.n_docs,
        len(result.passages),
        manifest.n_chars,
        ", ".join(f"{k} {v}" for k, v in sorted(result.doc_skips.items())) or "none",
        out_shard,
        out_manifest,
        report,
    )
    return 0


def _client():  # type: ignore[no-untyped-def]
    """Anthropic client, or a loud failure. Imported lazily so `cankar pairs
    segment` works without the key or the SDK."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise CankarError("ANTHROPIC_API_KEY not set (see .env.example)")
    import anthropic

    return anthropic.Anthropic()


def _drain(client, receipt: destyle.BatchReceipt) -> int:  # type: ignore[no-untyped-def]
    """Poll one submitted batch to completion and append its raw responses.

    Idempotent by construction: `already_done` reads the raw log, so a second
    drain of the same batch adds duplicate lines but never a duplicate charge.
    """
    delay = POLL_START_SECONDS
    while True:
        batch = client.messages.batches.retrieve(receipt.batch_id)
        if batch.processing_status == "ended":
            break
        log.info("batch %s: %s, waiting %ds", receipt.batch_id, batch.processing_status, delay)
        time.sleep(delay)
        delay = min(delay * 2, POLL_MAX_SECONDS)

    results = client.messages.batches.results(receipt.batch_id)
    records = [json.loads(r.model_dump_json()) for r in results]
    destyle.append_raw(destyle_raw(), records)
    log.info("batch %s: %d responses -> %s", receipt.batch_id, len(records), destyle_raw())
    return len(records)


def _destyle(args: argparse.Namespace) -> int:
    passages = destyle.load_passages(passages_shard())
    receipts = destyle.load_receipts(batch_receipts())

    if not args.parse_only:
        client = _client()
        # Drain first: a run interrupted between submit and download must resume
        # the paid batch, never submit a second one for the same passages.
        for receipt in destyle.pending_receipts(receipts):
            log.info("resuming un-downloaded batch %s", receipt.batch_id)
            _drain(client, receipt)
            receipt.downloaded = True
            destyle.write_receipts(batch_receipts(), receipts)

        done = destyle.already_done(destyle_raw())
        if args.retry_rejected:
            # The one path that knowingly pays twice - only worth it after
            # fixing a cause on our side.
            retry = destyle.retryable(destyle_raw(), passages)
            log.info("retrying %d previously rejected passages (re-billed)", len(retry))
            done = done - retry
        todo = destyle.select_passages(passages, args.limit, done)
        destyle.require_batch_size(len(todo))
        log.info(
            "%d passages, %d already paid for, %d to submit", len(passages), len(done), len(todo)
        )

        if args.dry_run:
            est = destyle.estimate_cost(todo)
            log.info(
                "DRY RUN, nothing submitted: %d requests, ~%d input tokens "
                "(%d passage + %d system). The system prompt is %d tokens and repeats on "
                "every request - below the caching floor, so it is %.0f%% of the input bill. "
                "Batch pricing halves it; apply the current price list.",
                est.n_requests,
                est.input_tokens,
                est.passage_tokens,
                est.system_tokens,
                destyle.SYSTEM_PROMPT_TOKENS,
                100 * est.system_tokens / est.input_tokens,
            )
            return 0

        if todo:
            requests = [destyle.build_request(p, args.model) for p in todo]
            batch = client.messages.batches.create(requests=requests)
            receipt = destyle.BatchReceipt(
                batch_id=batch.id,
                created_at=utc_now_iso(),
                model=args.model,
                n_requests=len(requests),
            )
            # Committed BEFORE the wait: this id is the only route back to
            # results that already exist server-side.
            receipts.append(receipt)
            destyle.write_receipts(batch_receipts(), receipts)
            log.info("submitted batch %s (%d requests) - receipt written", batch.id, len(requests))
            _drain(client, receipt)
            receipt.downloaded = True
            destyle.write_receipts(batch_receipts(), receipts)

    pairs, rejected = destyle.build_pairs(passages, destyle_raw(), args.model)
    out_shard = destyle.write_pairs(pairs_shard(), pairs)
    reject_counts: dict[str, int] = {}
    for r in rejected:
        reject_counts[str(r.reason)] = reject_counts.get(str(r.reason), 0) + 1

    manifest = destyle.PairsManifest(
        destyler_version=destyle.DESTYLER_VERSION,
        model=args.model,
        corpus_sha256=sha256_of(merged_shard()),
        passages_sha256=sha256_of(passages_shard()),
        register_sha256=REGISTER_SHA256,
        pairs_sha256=sha256_of(out_shard),
        batch_ids=[r.batch_id for r in receipts],
        git_sha=git_sha(),
        created_at=utc_now_iso(),
        lib_versions=library_versions("anthropic"),
        n_requested=sum(r.n_requests for r in receipts),
        n_pairs=len(pairs),
        n_rejected=len(rejected),
        reject_counts=reject_counts,
        in_tokens=sum(p.in_tokens for p in pairs),
        out_tokens=sum(p.out_tokens for p in pairs),
    )
    out_manifest = write_manifest(manifest, pairs_manifest())
    destyle.write_pairs(pairs_samples(), publish.select_samples(pairs))
    report = destyle.write_pairs_report(pairs_report(), manifest, pairs[:3])
    log.info(
        "%d pairs, %d rejected (%s) -> %s + %s + %s",
        len(pairs),
        len(rejected),
        ", ".join(f"{k} {v}" for k, v in sorted(reject_counts.items())) or "none",
        out_shard,
        out_manifest,
        report,
    )
    return 0


def _publish(args: argparse.Namespace) -> int:
    if not os.environ.get("HF_TOKEN"):
        raise CankarError("HF_TOKEN not set (see .env.example; needs write on the dataset repo)")
    from huggingface_hub import HfApi

    pairs_mf = load_frozen(pairs_manifest(), destyle.PairsManifest, "cankar pairs destyle")
    passages_mf = load_frozen(passages_manifest(), segment.PassagesManifest, "cankar pairs segment")

    card_path = pairs_shard().parent / "DATASET_CARD.md"
    card_path.write_text(publish.dataset_card(pairs_mf, passages_mf, args.repo), encoding="utf-8")
    uploads = publish.build_uploads(pairs_shard(), destyle_raw(), pairs_manifest(), card_path)
    publish.check_uploads(uploads)

    if args.dry_run:
        for u in uploads:
            size = u.local.stat().st_size if u.local.exists() else 0
            log.info(
                "DRY RUN: %s -> %s:%s (%.1f MB)", u.local.name, args.repo, u.remote, size / 1e6
            )
        return 0

    api = HfApi()
    for u in uploads:
        if not u.local.exists():
            log.info("skipping absent %s", u.local.name)
            continue
        api.upload_file(
            path_or_fileobj=str(u.local),
            path_in_repo=u.remote,
            repo_id=args.repo,
            repo_type=publish.REPO_TYPE,
            commit_message=f"{pairs_mf.n_pairs} pairs from git {pairs_mf.git_sha}",
        )
        log.info("uploaded %s -> %s", u.local.name, u.remote)
    log.info(
        "published %d pairs to https://huggingface.co/datasets/%s (private)",
        pairs_mf.n_pairs,
        args.repo,
    )
    return 0


def register(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("segment", help="cut Cankar prose into de-styling passages (Phase 5)")
    s.set_defaults(func=_segment)

    d = sub.add_parser("destyle", help="Batch API de-styling into plain Slovene (SPENDS MONEY)")
    d.add_argument("--limit", type=int, default=10_000, help="max passages to submit this run")
    d.add_argument("--model", default=destyle.DEFAULT_MODEL, help="model id")
    d.add_argument("--dry-run", action="store_true", help="report size and submit nothing")
    d.add_argument(
        "--parse-only", action="store_true", help="re-parse the raw log without calling the API"
    )
    d.add_argument(
        "--retry-rejected",
        action="store_true",
        help="re-send passages whose paid response was rejected (RE-BILLS them)",
    )
    d.set_defaults(func=_destyle)

    p = sub.add_parser("publish", help="upload the pair dataset to the HF Hub")
    p.add_argument("--repo", default=publish.DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument("--dry-run", action="store_true", help="list what would be uploaded")
    p.set_defaults(func=_publish)
