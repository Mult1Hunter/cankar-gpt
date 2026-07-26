"""Corpus-stamp parsing (ADR 0021).

The header samples below are REAL - copied from the four writers that stamp a
corpus sha, each of which phrases it differently. Per ADR 0006 the enumerated
classes become regression fixtures: a fifth phrasing added later must either
match this parser or fail loudly here, never be silently read as UNSTAMPED.
"""

from __future__ import annotations

import pytest

from cankar.core.reports import ReportFreshness, generated_marker, read_stamp

LIVE = "d9b05bf04db96db6d733a08540bda86f5458ed91b8182a61f5262b6d0dd22a6b"
OLD = "cdfea262ce5898e5b35001afb2fb8899ccb531ea1c6962c39a7f1ed917e418a2"


def _header(command: str, body: str) -> str:
    """Real marker + the writer's own body, so fixtures track the contract."""
    return generated_marker(command, snapshot=True) + "\n" + body


# name -> (generating command, body exactly as that writer emits it):
# cankar/tokenizer/stats.py, tokenizer/evaluate.py, tokenizer/chunk.py,
# evals/holdout.py.
PHRASINGS = {
    "token-stats": (
        "cankar tokenizer stats",
        "\n# Token stats (Phase 2)\n\nTokenizer: `v8192`. Corpus sha256 `{sha}`.\n",
    ),
    "tokenizer-eval": (
        "cankar tokenizer eval",
        "# Tokenizer evaluation (Phase 2)\nCorpus: `data/merged/corpus.jsonl` sha256 `{sha}`\n",
    ),
    # chunk.py breaks the line between the label and the sha - the parser must
    # span that newline, which a per-line regex would not.
    "chunks": (
        "cankar tokenizer chunk",
        "\n# Chunks (Phase 2 - ADR 0012)\n\nTokenizer `v8192`, budget 2048 tokens\n"
        "(== Phase 3 max_seq_len; re-chunk if that changes). Corpus sha256\n`{sha}`.\n",
    ),
    "eval-holdout": (
        "cankar evals holdout-freeze",
        "# Held-out set\nCorpus sha256 `{sha}`, tokenizer `v8192`.\n",
    ),
}


@pytest.mark.parametrize("name", sorted(PHRASINGS))
def test_every_writer_phrasing_is_parsed(name: str, tmp_path) -> None:
    command, body = PHRASINGS[name]
    report = tmp_path / f"{name}.md"
    report.write_text(_header(command, body.format(sha=LIVE)), encoding="utf-8")

    stamp = read_stamp(report, LIVE)

    assert stamp.freshness is ReportFreshness.FRESH
    assert stamp.stamped_sha == LIVE
    assert stamp.command == command, "regeneration command must survive parsing"


@pytest.mark.parametrize("name", sorted(PHRASINGS))
def test_every_writer_phrasing_detects_drift(name: str, tmp_path) -> None:
    command, body = PHRASINGS[name]
    report = tmp_path / f"{name}.md"
    report.write_text(_header(command, body.format(sha=OLD)), encoding="utf-8")

    assert read_stamp(report, LIVE).freshness is ReportFreshness.STALE


def test_report_without_a_corpus_sha_is_unstamped(tmp_path) -> None:
    report = tmp_path / "style.md"
    report.write_text(
        _header("cankar evals style-train", "# Style classifier\nROC-AUC 0.993.\n"),
        encoding="utf-8",
    )

    assert read_stamp(report, LIVE).freshness is ReportFreshness.UNSTAMPED


def test_a_truncated_sha_is_never_read_as_fresh(tmp_path) -> None:
    """A malformed stamp must not pass - UNSTAMPED is tracked, FRESH is not."""
    report = tmp_path / "token-stats.md"
    report.write_text(
        _header("cankar tokenizer stats", f"Corpus sha256 `{LIVE[:20]}`.\n"),
        encoding="utf-8",
    )

    assert read_stamp(report, LIVE).freshness is not ReportFreshness.FRESH


def test_a_non_corpus_sha_is_not_mistaken_for_the_corpus(tmp_path) -> None:
    """Anchoring on "corpus" keeps another artifact's hash from being compared."""
    report = tmp_path / "style.md"
    report.write_text(
        _header("cankar evals style-train", f"# Style classifier\nModel sha256 `{OLD}`.\n"),
        encoding="utf-8",
    )

    assert read_stamp(report, LIVE).freshness is ReportFreshness.UNSTAMPED


def test_a_corpus_command_in_the_marker_does_not_fake_a_stamp(tmp_path) -> None:
    """`cankar corpus merge` puts "corpus" in the header; that is not a stamp.

    The real merge.md and corpus-quality.md hit this, which is why the parser
    requires "corpus" and "sha256" on the SAME line.
    """
    report = tmp_path / "merge.md"
    report.write_text(
        _header("cankar corpus merge", f"# Merge report\nShard sha256 `{OLD}`.\n"),
        encoding="utf-8",
    )

    assert read_stamp(report, LIVE).freshness is ReportFreshness.UNSTAMPED
