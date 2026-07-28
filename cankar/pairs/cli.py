"""Pairs-stage CLI subcommands - the ONLY argparse holder for this stage.

Registered under the single `cankar` console entry (ADR 0007):
    cankar pairs segment          # cut de-styling passages + commit provenance
"""

from __future__ import annotations

import argparse
import logging

from cankar.core.holdout import holdout_excludes, load_holdout
from cankar.core.manifest import git_sha, sha256_of, utc_now_iso, write_manifest
from cankar.core.paths import (
    holdout_manifest,
    merged_shard,
    passages_manifest,
    passages_report,
    passages_shard,
    works_registry,
)
from cankar.core.works import load_work_genres
from cankar.pairs import segment

log = logging.getLogger("cankar.pairs")


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


def register(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("segment", help="cut Cankar prose into de-styling passages (Phase 5)")
    s.set_defaults(func=_segment)
