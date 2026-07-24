"""Frozen held-out-set contract (ADR 0013), promoted to core so every stage can
read the exclusion set without importing the evals stage (import-linter, ADR
0007). The SELECTION logic (which works to hold out, and the report/audit) stays
in `cankar.evals.holdout`; this module is only the frozen data model plus the
read/exclude helpers that the training stage and the Phase-3 conversion filter
also need. Deliberately tiktoken-free - core must stay light.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from cankar.core.manifest import load_frozen

CANKAR_AUTHOR = "Ivan Cankar"
# transcription, not dLib OCR (critique A-2). Matches corpus Source.WIKIVIR.
HOLDOUT_SOURCE = "wikivir"

# Medium prose band, calibrated on the real Cankar slice (median doc 8,211
# chars): the lower bound drops tiny verse, the upper bound drops novels and
# collected volumes so no single work dominates and containers are not held out
# (critique A-3/A-5). Audited against the generated set (ADR 0006).
MIN_CHARS = 3_000
MAX_CHARS = 45_000
# A candidate this-fraction contained in any other kept Cankar doc leaks and is
# rejected. 0.5 matches the merge's registry-confirm threshold (merge.py).
CONTAINMENT_REJECT = 0.5
# ~5% of the 2.92M-token Cankar bottleneck: enough for a stable summed-bytes
# BPB, small enough that Phase 4 barely feels it (critique A-3).
TARGET_TOKEN_FRACTION = 0.05
MIN_WORKS = 8  # BPB must average over work-level idiosyncrasy, not one novel


class HoldoutParams(BaseModel):
    """Selection thresholds (defaults are the calibrated production values;
    tests pass smaller ones). Frozen into the manifest for reproducibility."""

    min_chars: int = MIN_CHARS
    max_chars: int = MAX_CHARS
    containment_reject: float = CONTAINMENT_REJECT
    target_token_fraction: float = TARGET_TOKEN_FRACTION
    min_works: int = MIN_WORKS


class HoldoutWork(BaseModel):
    url: str
    title: str
    n_chars: int
    n_tokens: int  # whole-doc encode_ordinary, BOS excluded (matches token-stats.md)
    content_sha256: str  # detects text drift independent of the corpus-wide hash
    max_containment_elsewhere: float  # audit trail: how clean the closure left it


class HoldoutManifest(BaseModel):
    """Frozen, provenance-stamped, load-bearing (ADR 0013). Modeled on
    registry/datasets/ - generated once, committed, never hand-edited."""

    schema_version: int = 1
    corpus_sha256: str  # per-work content shas are only valid against this text
    tokenizer_name: str
    author: str = CANKAR_AUTHOR
    source: str = HOLDOUT_SOURCE
    params: HoldoutParams
    cankar_total_tokens: int
    holdout_tokens: int
    holdout_fraction: float
    git_sha: str
    created_at: str
    works: list[HoldoutWork]
    # reverse-containment: training docs >= reject contained in a held-out work;
    # the Phase 3 filter must drop these too or they leak held-out text
    also_exclude_urls: list[str] = []


def holdout_excludes(manifest: HoldoutManifest) -> frozenset[str]:
    """The Phase 3 / training conversion filter: urls to drop from training data.
    Held-out works PLUS reverse-contained training docs (excerpts of held-out
    works) - both directions of the closure (see cankar.evals.holdout docstring)."""
    return frozenset(w.url for w in manifest.works) | frozenset(manifest.also_exclude_urls)


def load_holdout(path: Path) -> HoldoutManifest:
    return load_frozen(path, HoldoutManifest, "cankar evals holdout-freeze")
