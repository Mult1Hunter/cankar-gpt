"""Corpus-stamp parsing (ADR 0021).

The positive cases parametrize over the REAL committed reports, not transcribed
copies: a hand-maintained "exactly as the writer emits it" fixture drifts, and a
drifted fixture cannot detect the drift it exists to catch (design-review
2026-07-26). Parsing needs no corpus, so these run in CI.

Synthetic bodies are kept only for the negatives and the boundary cases, which
have no real example by definition.
"""

from __future__ import annotations

import pytest

from cankar.core.paths import repo_root
from cankar.core.reports import ReportFreshness, generated_marker, parse_stamp

LIVE = "d9b05bf04db96db6d733a08540bda86f5458ed91b8182a61f5262b6d0dd22a6b"
OLD = "cdfea262ce5898e5b35001afb2fb8899ccb531ea1c6962c39a7f1ed917e418a2"

# The committed reports that stamp a corpus sha, one per distinct writer phrasing:
# tokenizer/stats.py, tokenizer/evaluate.py, tokenizer/chunk.py (which breaks the
# line between label and sha), evals/holdout.py.
STAMPED_REPORTS = ("token-stats.md", "tokenizer-eval.md", "chunks.md", "eval-holdout.md")


def _header(command: str, body: str) -> str:
    return generated_marker(command, snapshot=True) + "\n" + body


@pytest.mark.parametrize("name", STAMPED_REPORTS)
def test_real_reports_parse(name: str) -> None:
    """Every committed writer phrasing must yield a sha and a regeneration command."""
    stamp = parse_stamp(repo_root() / "registry" / "reports" / name)

    assert stamp.corpus_sha is not None, f"{name}: real header no longer parses"
    assert len(stamp.corpus_sha) == 64
    assert stamp.command, f"{name}: regeneration command must survive parsing"
    assert stamp.snapshot is True


@pytest.mark.parametrize("name", STAMPED_REPORTS)
def test_real_reports_detect_drift(name: str) -> None:
    stamp = parse_stamp(repo_root() / "registry" / "reports" / name)

    assert stamp.freshness(OLD) is ReportFreshness.STALE
    assert stamp.freshness(stamp.corpus_sha or "") is ReportFreshness.FRESH


def test_an_uppercase_stamp_is_fresh_not_stale(tmp_path) -> None:
    """IGNORECASE covers the label; the hex is normalised, so case is not drift."""
    report = tmp_path / "token-stats.md"
    report.write_text(
        _header("cankar tokenizer stats", f"Corpus sha256 `{LIVE.upper()}`.\n"), encoding="utf-8"
    )

    assert parse_stamp(report).freshness(LIVE) is ReportFreshness.FRESH


def test_report_without_a_corpus_sha_is_unstamped(tmp_path) -> None:
    report = tmp_path / "style.md"
    report.write_text(
        _header("cankar evals style-train", "# Style classifier\nROC-AUC 0.993.\n"),
        encoding="utf-8",
    )

    assert parse_stamp(report).freshness(LIVE) is ReportFreshness.UNSTAMPED


def test_a_truncated_sha_is_never_read_as_fresh(tmp_path) -> None:
    report = tmp_path / "token-stats.md"
    report.write_text(
        _header("cankar tokenizer stats", f"Corpus sha256 `{LIVE[:20]}`.\n"), encoding="utf-8"
    )

    assert parse_stamp(report).freshness(LIVE) is not ReportFreshness.FRESH


def test_a_non_corpus_sha_is_not_mistaken_for_the_corpus(tmp_path) -> None:
    report = tmp_path / "style.md"
    report.write_text(
        _header("cankar evals style-train", f"# Style classifier\nModel sha256 `{OLD}`.\n"),
        encoding="utf-8",
    )

    assert parse_stamp(report).freshness(LIVE) is ReportFreshness.UNSTAMPED


def test_a_corpus_command_in_the_marker_does_not_fake_a_stamp(tmp_path) -> None:
    """`cankar corpus merge` puts "corpus" in the header; that is not a stamp.

    The real merge.md and corpus-quality.md hit this, which is why the parser
    requires "corpus" and "sha256" on the SAME line.
    """
    report = tmp_path / "merge.md"
    report.write_text(
        _header("cankar corpus merge", f"# Merge report\nShard sha256 `{OLD}`.\n"), encoding="utf-8"
    )

    assert parse_stamp(report).freshness(LIVE) is ReportFreshness.UNSTAMPED


# --- boundary cases: the two thresholds no real report exercises ---------------


def test_a_gap_wider_than_the_window_degrades_to_unstamped(tmp_path) -> None:
    """Pins the 80-char label window, so widening it is a deliberate edit."""
    report = tmp_path / "token-stats.md"
    filler = "x" * 81
    report.write_text(
        _header("cankar tokenizer stats", f"Corpus {filler} sha256 `{LIVE}`.\n"), encoding="utf-8"
    )

    assert parse_stamp(report).corpus_sha is None


def test_a_stamp_below_the_header_window_degrades_to_unstamped(tmp_path) -> None:
    """Pins the 20-line header window for the same reason."""
    report = tmp_path / "token-stats.md"
    padding = "\n".join(f"filler {i}" for i in range(25))
    report.write_text(
        _header("cankar tokenizer stats", f"{padding}\nCorpus sha256 `{LIVE}`.\n"), encoding="utf-8"
    )

    assert parse_stamp(report).corpus_sha is None


def test_the_marker_tail_classifies_committed_input_reports(tmp_path) -> None:
    """snapshot=False reports are CI's job, not the freshness gate's."""
    report = tmp_path / "collisions.md"
    report.write_text(
        generated_marker("cankar corpus report", snapshot=False) + "\n# Collisions\n",
        encoding="utf-8",
    )

    assert parse_stamp(report).snapshot is False
