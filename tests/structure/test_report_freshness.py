"""Snapshot-report freshness (ADR 0021).

`registry/reports/` holds two classes of generated file, and the marker line
encodes which (`cankar/core/reports.py`):

- committed-input (`snapshot=False`) - CI already rebuilds these byte-identically
  (`cankar corpus report --all && git diff --exit-code`). Not this module's job.
- snapshot (`snapshot=True`) - computed from gitignored `data/`, which CI never
  has. Unguarded, they drifted twice after the ADR 0014 re-merge: `token-stats.md`
  and `tokenizer-eval.md` both kept a dead corpus sha in a public repo.

Class membership is read from the marker, not from filenames, so a new
`report --all` output is classified correctly without editing this file.

Calibrated on a real positive (ADR 0006): this module was written against a tree
where `tokenizer-eval.md` was genuinely stale, and was required to FAIL before the
report was regenerated. Cost is one sha256 over ~503MB, ~0.3s.

Only the sha COMPARISON needs the corpus, so only that test skips locally-gated.
The pinning test below runs everywhere, including CI - which is the only place a
contributor's PR is checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cankar.core.manifest import sha256_of
from cankar.core.paths import merged_shard, repo_root
from cankar.core.reports import ReportFreshness, parse_stamp

# Snapshot reports built from the merged corpus that do NOT stamp a sha yet.
# Pinned so a new unstamped report cannot be added quietly; stamping them means
# touching three writers, deferred under rule of two (ADR 0021).
UNSTAMPED_ALLOWLIST = frozenset(
    {
        "corpus-quality.md",
        "merge.md",
        "near-duplicates.md",
    }
)

# Snapshot reports not built from the merged corpus at all, so a corpus-sha stamp
# would be meaningless. Permanently out of scope, not a TODO.
NOT_CORPUS_DERIVED = frozenset(
    {
        "dlib-reconcile.md",  # dLib metadata reconciliation
        "style.md",  # the trained style classifier
    }
)


def _snapshot_reports() -> list[Path]:
    """Reports whose own marker declares them snapshots over gitignored data."""
    reports_dir = repo_root() / "registry" / "reports"
    return sorted(
        p
        for p in reports_dir.glob("*.md")
        if parse_stamp(p).snapshot and p.name not in NOT_CORPUS_DERIVED
    )


@pytest.fixture(scope="module")
def current_corpus_sha() -> str:
    corpus = merged_shard()
    if not corpus.exists():
        pytest.skip("merged corpus absent (CI / fresh clone) - comparison is a local gate")
    return sha256_of(corpus)


def test_stamped_reports_match_the_corpus_on_disk(current_corpus_sha: str) -> None:
    """No snapshot report may claim a corpus sha other than the live one."""
    stale = [
        stamp
        for report in _snapshot_reports()
        if (stamp := parse_stamp(report)).freshness(current_corpus_sha) is ReportFreshness.STALE
    ]

    assert not stale, "reports generated from a stale corpus:\n" + "\n".join(
        s.describe(current_corpus_sha) for s in stale
    )


def test_unstamped_reports_are_the_known_set() -> None:
    """A new corpus-derived report must stamp its sha, or be allowlisted on purpose.

    No corpus needed: whether a report carries a stamp is a property of the file,
    so this guard runs in CI - the only place a contributor's PR is checked.
    """
    unstamped = {
        report.name for report in _snapshot_reports() if parse_stamp(report).corpus_sha is None
    }

    assert unstamped == UNSTAMPED_ALLOWLIST, (
        "unstamped corpus-derived reports changed.\n"
        f"  now:      {sorted(unstamped)}\n"
        f"  expected: {sorted(UNSTAMPED_ALLOWLIST)}\n"
        "A new report here silently escapes the freshness gate - stamp the corpus "
        "sha in its writer, or widen UNSTAMPED_ALLOWLIST deliberately."
    )
