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
)
from cankar.core.register import REGISTER_SHA256
from cankar.pairs import segment

log = logging.getLogger("cankar.pairs")


def _segment(args: argparse.Namespace) -> int:
    corpus = merged_shard()
    excludes = holdout_excludes(load_holdout(holdout_manifest()))
    params = segment.SegmentParams()
    result = segment.segment_corpus(corpus, excludes, params)

    out_shard = segment.write_passages(passages_shard(), result.passages)
    manifest = segment.PassagesManifest(
        segmenter_version=segment.SEGMENTER_VERSION,
        corpus_sha256=sha256_of(corpus),
        holdout_sha256=sha256_of(holdout_manifest()),
        register_sha256=REGISTER_SHA256,
        git_sha=git_sha(),
        created_at=utc_now_iso(),
        source=segment.SOURCE_WIKIVIR,
        params=params,
        n_docs=result.n_docs,
        n_passages=len(result.passages),
        n_chars=sum(p.n_chars for p in result.passages),
        reject_counts=result.reject_counts,
    )
    out_manifest = write_manifest(manifest, passages_manifest())
    report = segment.write_passages_report(passages_report(), manifest)
    log.info(
        "segmented %d docs -> %d passages (%d chars), excluded %d held-out urls -> %s + %s + %s",
        result.n_docs,
        len(result.passages),
        manifest.n_chars,
        len(excludes),
        out_shard,
        out_manifest,
        report,
    )
    return 0


def register(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("segment", help="cut Cankar prose into de-styling passages (Phase 5)")
    s.set_defaults(func=_segment)
