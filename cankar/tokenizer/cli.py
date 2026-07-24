"""Tokenizer-stage CLI subcommands - the ONLY argparse holder for this stage.

Registered under the single `cankar` console entry (ADR 0007):
    cankar tokenizer train --vocab-size 8192 [--skip-determinism-check]
    cankar tokenizer eval [--select v8192 --reason "..."]
    cankar tokenizer chunk --name v8192 [--budget 2048]
    cankar tokenizer stats --name v8192
    cankar tokenizer install --name v8192
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
from pathlib import Path

from cankar.core.encoding import load_encoding
from cankar.core.errors import CankarError
from cankar.core.manifest import (
    git_sha,
    library_versions,
    sha256_of,
    utc_now_iso,
    write_manifest,
)
from cankar.core.paths import (
    chunks_report,
    chunks_shard,
    dataset_manifest,
    merged_shard,
    token_stats_report,
    tokenizer_base_dir,
    tokenizer_dir,
    tokenizer_eval_report,
    tokenizer_probes_config,
)
from cankar.tokenizer import chunk, evaluate, stats, train
from cankar.tokenizer.vendored import NANOCHAT_COMMIT, SPECIAL_TOKENS, SPLIT_PATTERN

log = logging.getLogger("cankar.tokenizer")


def _trained_names() -> list[str]:
    base = tokenizer_base_dir()
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "tokenizer.pkl").exists())


def _train(args: argparse.Namespace) -> int:
    corpus = merged_shard()
    name = f"v{args.vocab_size}"
    log.info("training %s (vocab %d) on %s", name, args.vocab_size, corpus)
    enc = train.train_encoding(corpus, args.vocab_size)
    deterministic = False
    if not args.skip_determinism_check:
        log.info("determinism check: retraining %s", name)
        deterministic = train.verify_determinism(corpus, args.vocab_size, enc)
        if not deterministic:
            return 1
    pkl, tb = train.save_artifacts(enc, tokenizer_dir(name))
    n_docs = 0
    n_chars = 0
    for doc in train.iter_corpus_docs(corpus):
        n_docs += 1
        n_chars += doc["n_chars"]
    manifest = train.TokenizerManifest(
        name=name,
        vocab_size=enc.n_vocab,
        n_mergeable_ranks=enc.n_vocab - len(SPECIAL_TOKENS),
        special_tokens=SPECIAL_TOKENS,
        split_pattern_sha256=hashlib.sha256(SPLIT_PATTERN.encode()).hexdigest(),
        corpus_sha256=sha256_of(corpus),
        n_docs=n_docs,
        n_chars=n_chars,
        doc_cap=None,
        nanochat_commit=NANOCHAT_COMMIT,
        git_sha=git_sha(),
        trained_at=utc_now_iso(),
        tokenizer_pkl_sha256=sha256_of(pkl),
        token_bytes_pt_sha256=sha256_of(tb),
        determinism_verified=deterministic,
        **library_versions("rustbpe", "tiktoken", "torch"),
    )
    out = write_manifest(manifest, dataset_manifest("tokenizer", name))
    log.info("saved %s + %s; manifest %s", pkl, tb, out)
    return 0


def _eval(args: argparse.Namespace) -> int:
    names = args.names or _trained_names()
    if not names:
        raise CankarError("no trained candidates found (run: cankar tokenizer train)")
    if args.select and args.select not in names:
        raise CankarError(f"--select {args.select} is not a trained candidate {names}")
    encodings = {name: load_encoding(name) for name in names}
    evals, notes = evaluate.evaluate_candidates(merged_shard(), encodings)
    probes = evaluate.load_probes(tokenizer_probes_config())
    out = evaluate.write_eval_report(
        tokenizer_eval_report(),
        sha256_of(merged_shard()),
        evals,
        notes,
        probes,
        encodings,
        args.select,
        args.reason,
    )
    log.info("wrote %s", out)
    return 0


def _chunk(args: argparse.Namespace) -> int:
    enc = load_encoding(args.name)
    corpus = merged_shard()
    log.info("chunking %s with %s at budget %d", corpus, args.name, args.budget)
    run = chunk.run_chunking(corpus, enc, args.budget, chunks_shard())
    log.info("simulating bestfit crop over %d chunks", run.n_chunks)
    crop = chunk.simulate_bestfit_crop(run.chunk_token_lengths, args.budget)
    manifest = chunk.ChunksManifest(
        budget=args.budget,
        ladder_version=chunk.LADDER_VERSION,
        tokenizer_name=args.name,
        tokenizer_pkl_sha256=sha256_of(tokenizer_dir(args.name) / "tokenizer.pkl"),
        corpus_sha256=sha256_of(corpus),
        n_docs=run.n_docs,
        n_docs_split=run.n_docs_split,
        n_chunks=run.n_chunks,
        n_tokens_total=run.n_tokens_total,
        seam_hard_splits=run.seam_hard_splits,
        ladder_by_source=run.ladder_by_source,
        chunks_sha256=sha256_of(chunks_shard()),
        git_sha=git_sha(),
        created_at=utc_now_iso(),
    )
    write_manifest(manifest, dataset_manifest("tokenizer", "chunks"))
    out = chunk.write_chunks_report(
        chunks_report(), run, args.budget, crop, args.name, manifest.corpus_sha256
    )
    log.info("wrote %s + manifest + %s", chunks_shard(), out)
    return 0


def _stats(args: argparse.Namespace) -> int:
    enc = load_encoding(args.name)
    corpus = merged_shard()
    groups = stats.collect(corpus, enc)
    out = stats.write_stats_report(token_stats_report(), groups, args.name, sha256_of(corpus))
    log.info("wrote %s", out)
    return 0


def _install(args: argparse.Namespace) -> int:
    """Copy the selected candidate into nanochat's expected location (critique
    MF-5): base_train resolves $NANOCHAT_BASE_DIR/tokenizer/ and nothing else."""
    src = tokenizer_dir(args.name)
    base = os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))
    dst = Path(base) / "tokenizer"
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("tokenizer.pkl", "token_bytes.pt"):
        if not (src / fname).exists():
            raise CankarError(f"candidate '{args.name}' is missing {fname} at {src}")
        shutil.copy2(src / fname, dst / fname)
    log.info("installed %s -> %s (tokenizer.pkl + token_bytes.pt)", args.name, dst)
    return 0


def register(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train one BPE candidate on the merged corpus")
    p.add_argument("--vocab-size", type=int, required=True, help="total vocab incl. specials")
    p.add_argument(
        "--skip-determinism-check",
        action="store_true",
        help="skip the train-twice fingerprint comparison (recorded in the manifest)",
    )
    p.set_defaults(func=_train)

    p = sub.add_parser("eval", help="fertility report over trained candidates")
    p.add_argument("--names", nargs="*", help="candidates to compare (default: all trained)")
    p.add_argument("--select", help="record the winning candidate in the report")
    p.add_argument("--reason", help="one-line selection rationale for the report")
    p.set_defaults(func=_eval)

    p = sub.add_parser("chunk", help="chunk merged corpus to <= budget tokens (ADR 0012)")
    p.add_argument("--name", required=True, help="tokenizer candidate (the selected one)")
    p.add_argument(
        "--budget",
        type=int,
        default=chunk.DEFAULT_BUDGET,
        help="max tokens per chunk == Phase 3 max_seq_len (re-chunk if that changes)",
    )
    p.set_defaults(func=_chunk)

    p = sub.add_parser("stats", help="token stats per source x author + sizing inputs")
    p.add_argument("--name", required=True, help="tokenizer candidate (the selected one)")
    p.set_defaults(func=_stats)

    p = sub.add_parser("install", help="copy a candidate to $NANOCHAT_BASE_DIR/tokenizer/")
    p.add_argument("--name", required=True)
    p.set_defaults(func=_install)
