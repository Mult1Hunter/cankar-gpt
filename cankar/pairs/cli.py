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
    PairSet,
    batch_receipts,
    dataset_card,
    destyle_raw,
    draft_topics,
    drafts_manifest,
    drafts_shard,
    holdout_manifest,
    merged_shard,
    pairs_manifest,
    pairs_report,
    pairs_samples,
    pairs_shard,
    passages_manifest,
    passages_report,
    passages_shard,
    rejected_pairs,
    works_registry,
)
from cankar.core.register import REGISTER_SHA256
from cankar.core.works import load_work_genres
from cankar.pairs import destyle, drafts, publish, segment

log = logging.getLogger("cankar.pairs")

# Batch polling: most batches finish in minutes, the ceiling is 24h. Backs off
# to one minute so a long batch costs a handful of API calls, not thousands.
POLL_START_SECONDS = 5
POLL_MAX_SECONDS = 60


def _segment(args: argparse.Namespace) -> int:
    pair_set = PairSet(args.set)
    corpus = merged_shard()
    ledger = works_registry(segment.WORKS_LEDGER)
    excludes = holdout_excludes(load_holdout(holdout_manifest()))
    genres = load_work_genres(ledger)
    params = segment.SegmentParams()
    result = segment.segment_corpus(corpus, excludes, genres, params, pair_set)

    out_shard = segment.write_passages(passages_shard(pair_set), result.passages)
    manifest = segment.PassagesManifest(
        segmenter_version=segment.SEGMENTER_VERSION,
        corpus_sha256=sha256_of(corpus),
        holdout_sha256=sha256_of(holdout_manifest()),
        works_sha256=sha256_of(ledger),
        passages_sha256=sha256_of(out_shard),
        git_sha=git_sha(),
        created_at=utc_now_iso(),
        source=segment.SOURCE_WIKIVIR,
        pair_set=pair_set.value,
        prose_genres=sorted(segment.PROSE_GENRES),
        params=params,
        n_docs=result.n_docs,
        n_passages=len(result.passages),
        n_chars=sum(p.n_chars for p in result.passages),
        reject_counts=result.reject_counts,
        doc_skips=result.doc_skips,
    )
    out_manifest = write_manifest(manifest, passages_manifest(pair_set))
    report = segment.write_passages_report(passages_report(pair_set), manifest)
    log.info(
        "[%s] segmented %d docs -> %d passages (%d chars); skipped docs: %s -> %s + %s + %s",
        pair_set.value,
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


def _drain(client, receipt: destyle.BatchReceipt, pair_set: PairSet) -> int:  # type: ignore[no-untyped-def]
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
    destyle.append_raw(destyle_raw(pair_set), records)
    log.info("batch %s: %d responses -> %s", receipt.batch_id, len(records), destyle_raw(pair_set))
    return len(records)


def _destyle(args: argparse.Namespace) -> int:
    pair_set = PairSet(args.set)
    passages = destyle.load_passages(passages_shard(pair_set))
    receipts = destyle.load_receipts(batch_receipts())

    # Costing must not require a credential: --dry-run submits nothing.
    if args.dry_run:
        done = destyle.already_done(destyle_raw(pair_set))
        todo = destyle.select_passages(passages, args.limit, done)
        est = destyle.estimate_cost(todo)
        if not todo:
            log.info("DRY RUN: nothing left to submit (%d already paid for)", len(done))
            return 0
        log.info(
            "DRY RUN, nothing submitted: %d requests, ~%d input + ~%d output tokens. "
            "Fixed input is %d tokens/request (%.0f%% of input) - below the caching floor, "
            "so it is paid every time. Output usually dominates the bill at a 5x "
            "multiplier. Batch pricing halves both; apply the current price list.",
            est.n_requests,
            est.input_tokens,
            est.output_tokens,
            destyle.FIXED_INPUT_TOKENS,
            100 * est.fixed_tokens / est.input_tokens,
        )
        return 0

    if not args.parse_only:
        client = _client()
        # Reconcile against the API BEFORE anything else. The SDK retries a
        # failed create() with no idempotency key, so a lost response can leave
        # a billed batch this repo has no receipt for - and whose responses
        # `already_done` therefore cannot subtract. Refuse rather than risk it.
        api_ids = [b.id for b in client.messages.batches.list(limit=100)]
        orphans = destyle.unrecorded_batches(api_ids, receipts)
        if orphans:
            raise CankarError(
                f"batches exist with no receipt: {orphans}. They were paid for. Add them to "
                f"{batch_receipts()} (id, model, n_requests) and re-run, so their responses "
                "are drained instead of bought a second time."
            )

        # Drain next: a run interrupted between submit and download must resume
        # the paid batch, never submit a second one for the same passages.
        for receipt in destyle.pending_receipts(receipts):
            log.info("resuming un-downloaded batch %s", receipt.batch_id)
            _drain(client, receipt, receipt.pair_set)
            destyle.append_receipt(
                batch_receipts(), receipt.model_copy(update={"downloaded": True})
            )

        done = destyle.already_done(destyle_raw(pair_set))
        if args.retry_rejected:
            # The one path that knowingly pays twice - only worth it after
            # fixing a cause on our side.
            retry = destyle.retryable(destyle_raw(pair_set), passages)
            log.info("retrying %d previously rejected passages (re-billed)", len(retry))
            done = done - retry
        todo = destyle.select_passages(passages, args.limit, done)
        destyle.require_batch_size(len(todo))
        log.info(
            "%d passages, %d already paid for, %d to submit", len(passages), len(done), len(todo)
        )

        if todo:
            requests = [destyle.build_request(p, args.model) for p in todo]
            # max_retries=0: create() is NOT idempotent (the SDK sends no
            # idempotency key), so an automatic retry of a request the server
            # already accepted bills the whole batch twice.
            batch = client.with_options(max_retries=0).messages.batches.create(requests=requests)
            receipt = destyle.BatchReceipt(
                batch_id=batch.id,
                created_at=utc_now_iso(),
                model=args.model,
                n_requests=len(requests),
                pair_set=pair_set,
            )
            # Appended BEFORE the wait: this id is the only route back to
            # results that already exist server-side.
            destyle.append_receipt(batch_receipts(), receipt)
            receipts = destyle.load_receipts(batch_receipts())
            log.info("submitted batch %s (%d requests) - receipt written", batch.id, len(requests))
            _drain(client, receipt, receipt.pair_set)
            destyle.append_receipt(
                batch_receipts(), receipt.model_copy(update={"downloaded": True})
            )

    receipts = destyle.load_receipts(batch_receipts())
    pairs, rejected = destyle.build_pairs(passages, destyle_raw(pair_set), args.model)
    out_shard = destyle.write_pairs(pairs_shard(pair_set), pairs)
    destyle.write_rejected(rejected_pairs(pair_set), rejected)
    reject_counts: dict[str, int] = {}
    for r in rejected:
        reject_counts[str(r.reason)] = reject_counts.get(str(r.reason), 0) + 1

    manifest = destyle.PairsManifest(
        destyler_version=destyle.DESTYLER_VERSION,
        model=args.model,
        pair_set=pair_set.value,
        corpus_sha256=sha256_of(merged_shard()),
        passages_sha256=sha256_of(passages_shard(pair_set)),
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
    # A manifest claiming fewer requests than it has results describes a run
    # whose receipts are incomplete - i.e. paid batches this repo cannot reach.
    # It shipped once (n_requested=50 against n_pairs=10043) because nothing
    # checked (design-review 2026-07-28).
    if manifest.n_requested < manifest.n_pairs + manifest.n_rejected:
        raise CankarError(
            f"receipts account for {manifest.n_requested} requests but "
            f"{manifest.n_pairs + manifest.n_rejected} responses are on disk - a batch "
            f"receipt is missing from {batch_receipts()}. Recover the id from "
            "`client.messages.batches.list()` before freezing."
        )
    out_manifest = write_manifest(manifest, pairs_manifest(pair_set))
    destyle.write_pairs(pairs_samples(pair_set), publish.select_samples(pairs))
    report = destyle.write_pairs_report(
        pairs_report(pair_set), manifest, publish.select_samples(pairs, 3)
    )
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

    # The published dataset is the TRAINING corpus. The holdout set is the
    # measuring stick and stays local: publishing it invites someone to train on
    # it and then report a number it can no longer support.
    published = PairSet.TRAIN
    pairs_mf = load_frozen(pairs_manifest(published), destyle.PairsManifest, "cankar pairs destyle")
    passages_mf = load_frozen(
        passages_manifest(published), segment.PassagesManifest, "cankar pairs segment"
    )

    card = publish.dataset_card(pairs_mf, passages_mf, args.repo)
    publish.require_licensed(card)
    card_path = dataset_card()
    card_path.write_text(card, encoding="utf-8")
    uploads = publish.build_uploads(
        pairs_shard(published), destyle_raw(published), pairs_manifest(published), card_path
    )
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
        "published %d pairs (CC BY-SA 4.0) to https://huggingface.co/datasets/%s",
        pairs_mf.n_pairs,
        args.repo,
    )
    return 0


def _drafts(args: argparse.Namespace) -> int:
    topics = drafts.load_topics(draft_topics())
    wild = drafts.sample_wild(merged_shard())

    client = _client()
    requests = [drafts.build_request(t, i, args.model) for i, t in enumerate(topics)]
    batch = client.with_options(max_retries=0).messages.batches.create(requests=requests)
    log.info("submitted draft batch %s (%d topics)", batch.id, len(requests))
    delay = POLL_START_SECONDS
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        log.info("draft batch: %s, waiting %ds", b.processing_status, delay)
        time.sleep(delay)
        delay = min(delay * 2, POLL_MAX_SECONDS)
    records = [json.loads(r.model_dump_json()) for r in client.messages.batches.results(batch.id)]

    register_drafts = drafts.parse_register_drafts(records, topics)
    out = drafts.write_drafts(drafts_shard(), register_drafts + wild)
    manifest = drafts.DraftsManifest(
        drafts_version=drafts.DRAFTS_VERSION,
        model=args.model,
        corpus_sha256=sha256_of(merged_shard()),
        topics_sha256=sha256_of(draft_topics()),
        register_sha256=REGISTER_SHA256,
        drafts_sha256=sha256_of(out),
        batch_ids=[batch.id],
        git_sha=git_sha(),
        created_at=utc_now_iso(),
        n_register=len(register_drafts),
        n_wild=len(wild),
    )
    write_manifest(manifest, drafts_manifest())
    log.info("%d register + %d wild drafts -> %s", len(register_drafts), len(wild), out)
    return 0


def register(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("segment", help="cut Cankar prose into de-styling passages (Phase 5)")
    s.add_argument(
        "--set",
        default=PairSet.TRAIN.value,
        choices=[p.value for p in PairSet],
        help="train excludes held-out works; holdout is those works only",
    )
    s.set_defaults(func=_segment)

    d = sub.add_parser("destyle", help="Batch API de-styling into plain Slovene (SPENDS MONEY)")
    d.add_argument(
        "--set",
        default=PairSet.TRAIN.value,
        choices=[p.value for p in PairSet],
        help="which passage set to de-style",
    )
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

    dr = sub.add_parser("drafts", help="generate the Phase 6 fresh-draft eval set")
    dr.add_argument("--model", default=destyle.DEFAULT_MODEL, help="model id")
    dr.set_defaults(func=_drafts)

    p = sub.add_parser("publish", help="upload the pair dataset to the HF Hub")
    p.add_argument("--repo", default=publish.DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument("--dry-run", action="store_true", help="list what would be uploaded")
    p.set_defaults(func=_publish)
