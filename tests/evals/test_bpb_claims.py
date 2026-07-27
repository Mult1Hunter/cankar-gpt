"""Published BPB figures must match the frozen measurement (ADR 0017).

Invariant #2 says quality claims carry numbers. It did not say where the numbers
come from, and for three phases they came from a log line: `cankar evals bpb`
computed the BPB, logged it, and wrote nothing. The README badge, the docs table
and the live landing page all cited a figure that nothing in the repo could
reproduce or contradict - on a public repo.

`registry/evals/bpb.json` is now that source, and this closes the loop in both
directions: a figure claimed but not measured is drift (a badge edited without a
run, or a rerun whose result never reached the docs), and a figure measured but
claimed nowhere means the canonical set has grown a checkpoint no public claim
stands behind - which contradicts what `CanonicalCheckpoint` says it is.

Two blind spots, stated because the gate's docstring is the record (ADR 0023):

- It compares a SET of values, so it cannot see mis-assignment. Swap `base` and
  `cankar-v1` on the landing page and this stays green while the repo publishes
  a false progression. Catching that needs per-site name-to-figure association;
  not built - the human at review is the check.
- It matches `\\d.\\d{4}` exactly. `1.45`, `1.451`, `1.45080` and `10.4508` all
  match nothing at all, so a figure rewritten to a different precision leaves
  the gate silent rather than failing.
"""

from __future__ import annotations

import re

from cankar.core.manifest import load_frozen
from cankar.core.paths import bpb_manifest, repo_root
from cankar.evals.bpb import BpbManifest, CanonicalCheckpoint

# Every tracked file that states a held-out BPB to the public.
CLAIM_SITES = (
    "README.md",  # the shields.io badge and the intro line
    "ROADMAP.md",  # phase lines; checkboxes are canonical status (CLAUDE.md)
    "docs/cankar-v1.md",  # the three-model progression table
    "docs/tinycankar-samples.md",
    "apps/landing-page/index.html",  # cankar-gpt.nextgen-solutions.xyz
    # The report is snapshot-class, so a hand edit fails no CI drift check. It is
    # listed here so the human-readable table is checked against its own manifest
    # rather than being the one BPB surface with no gate (design-review).
    "registry/reports/bpb.md",
)

# 4-decimal figures in those files that are NOT held-out BPB. Kept explicit
# rather than pattern-narrowed: proximity matching against "BPB" is fragile
# across a markdown table and an HTML stat tile, and an allowlist makes a new
# non-BPB figure a conscious edit instead of a silent gate widening.
NOT_BPB = frozenset(
    {
        "0.0000",  # max containment of held-out works in training (docs/cankar-v1.md)
    }
)

_FIGURE_RE = re.compile(r"\b\d\.\d{4}\b")


def _claimed_figures() -> set[str]:
    found: set[str] = set()
    for site in CLAIM_SITES:
        found |= set(_FIGURE_RE.findall((repo_root() / site).read_text(encoding="utf-8")))

    # Subtracting a stale entry is silent, so a dead allowlist widens the gate
    # forever. Same reason test_report_freshness pins UNSTAMPED_ALLOWLIST by
    # equality: an exemption that no longer exempts anything must fail.
    dead = NOT_BPB - found
    assert not dead, f"NOT_BPB exempts figures no claim site contains: {sorted(dead)}"
    return found - NOT_BPB


def test_published_bpb_figures_match_the_frozen_measurement() -> None:
    """Set equality, so drift in either direction fails."""
    manifest = load_frozen(bpb_manifest(), BpbManifest, "cankar evals bpb-freeze")
    measured = {f"{c.bpb:.4f}" for c in manifest.checkpoints}
    claimed = _claimed_figures()

    assert claimed == measured, (
        "published BPB figures disagree with registry/evals/bpb.json:\n"
        f"  claimed, never measured: {sorted(claimed - measured)}\n"
        f"  measured, never claimed: {sorted(measured - claimed)}\n"
        "Re-run `uv run cankar evals bpb-freeze` and update the claim sites, or "
        "add a non-BPB figure to NOT_BPB deliberately."
    )


def test_claim_sites_exist() -> None:
    """A renamed claim site would empty the probe set and let the gate above pass
    while checking nothing - the vacuous-gate failure this repo keeps hitting."""
    missing = [s for s in CLAIM_SITES if not (repo_root() / s).is_file()]

    assert not missing, f"claim sites moved or were renamed: {missing}"


def test_every_canonical_checkpoint_is_measured() -> None:
    """The manifest must cover the whole enum, not whatever happened to be on
    disk. `score_canonical` raises on a missing checkpoint, but a hand-edited or
    part-written manifest would not."""
    manifest = load_frozen(bpb_manifest(), BpbManifest, "cankar evals bpb-freeze")
    measured = {c.name for c in manifest.checkpoints}
    expected = {c.value for c in CanonicalCheckpoint}

    assert measured == expected, f"manifest covers {sorted(measured)}, expected {sorted(expected)}"
