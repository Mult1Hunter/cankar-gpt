"""Repo-anchored path policy - the only place artifact locations are defined.

Hardcoded relative f-string paths break the moment CWD changes (RunPod pods,
Phase 3); every module computes locations through these helpers instead
(ADR 0007). Assumes an editable install (uv sync), which this repo always uses.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class PairSet(StrEnum):
    """Which side of the contamination boundary an artifact belongs to.

    TRAIN excludes held-out works; HOLDOUT is those works and ONLY those. They
    are disjoint by construction and must never share a file: a random split of
    TRAIN cannot evaluate the styler honestly, because every passage in it comes
    from a work the base model already saw in pretraining.

    An explicit enum rather than a boolean, and required at every path call, so
    the set a caller means is never implied (`.claude/rules/code-standards.md`).
    """

    TRAIN = "train"
    HOLDOUT = "holdout"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def works_registry(slug: str) -> Path:
    return repo_root() / "registry" / "works" / f"{slug}.jsonl"


def works_registries() -> list[Path]:
    return sorted((repo_root() / "registry" / "works").glob("*.jsonl"))


def corpus_shard(slug: str) -> Path:
    return repo_root() / "data" / "corpus" / f"{slug}.jsonl"


def dataset_manifest(stage: str, name: str) -> Path:
    """Committed provenance ledger (ADR 0007): manifests live in git, not in
    gitignored data/ - otherwise 'regenerate and diff' is unimplementable."""
    return repo_root() / "registry" / "datasets" / stage / f"{name}.manifest.json"


def corpus_dir() -> Path:
    return repo_root() / "data" / "corpus"


def coverage_report(slug: str) -> Path:
    return repo_root() / "registry" / "reports" / f"coverage-{slug}.md"


def collisions_report() -> Path:
    return repo_root() / "registry" / "reports" / "collisions.md"


def quality_report() -> Path:
    """Snapshot report (computed from gitignored data/) - see reports README."""
    return repo_root() / "registry" / "reports" / "corpus-quality.md"


def near_duplicates_report() -> Path:
    """Snapshot report (computed from gitignored data/) - see reports README."""
    return repo_root() / "registry" / "reports" / "near-duplicates.md"


def merged_shard() -> Path:
    """The merged corpus - outside data/corpus/ so the stats glob never double-counts."""
    return repo_root() / "data" / "merged" / "corpus.jsonl"


def merge_report() -> Path:
    """Snapshot report (computed from gitignored data/) - see reports README."""
    return repo_root() / "registry" / "reports" / "merge.md"


def collision_resolution() -> Path:
    """Human-curated cross-author collision decisions the merge consumes."""
    return repo_root() / "registry" / "works" / "collision_resolution.toml"


def dlib_reconcile_report() -> Path:
    """Snapshot report (live dLib state at run time) - see reports README."""
    return repo_root() / "registry" / "reports" / "dlib-reconcile.md"


def authors_config() -> Path:
    return repo_root() / "configs" / "corpus" / "authors.toml"


def tokenizer_base_dir() -> Path:
    return repo_root() / "data" / "tokenizer"


def tokenizer_dir(name: str) -> Path:
    """One trained candidate: tokenizer.pkl + token_bytes.pt (both required -
    nanochat's base_train asserts on token_bytes.pt at startup)."""
    return tokenizer_base_dir() / name


def tokenizer_eval_report() -> Path:
    """Snapshot report (computed from gitignored data/) - see reports README."""
    return repo_root() / "registry" / "reports" / "tokenizer-eval.md"


def tokenizer_probes_config() -> Path:
    return repo_root() / "configs" / "tokenizer" / "probes.toml"


def chunks_shard() -> Path:
    """Training chunks (ADR 0012) - own dir so corpus globs never see them."""
    return repo_root() / "data" / "chunks" / "chunks.jsonl"


def chunks_manifest() -> Path:
    """Committed provenance for the chunks (its corpus_sha256 must match the
    holdout's before training - ADR 0016)."""
    return dataset_manifest("tokenizer", "chunks")


def chunks_report() -> Path:
    """Snapshot report (computed from gitignored data/) - see reports README."""
    return repo_root() / "registry" / "reports" / "chunks.md"


def token_stats_report() -> Path:
    """Snapshot report (computed from gitignored data/) - see reports README."""
    return repo_root() / "registry" / "reports" / "token-stats.md"


def passages_shard(pair_set: PairSet) -> Path:
    """Phase 5 de-styling passages - own dir so corpus/chunks globs never see them."""
    return repo_root() / "data" / "pairs" / f"{pair_set.value}-passages.jsonl"


def passages_manifest(pair_set: PairSet) -> Path:
    """Committed provenance for the passage set (ADR 0003)."""
    return dataset_manifest("pairs", f"{pair_set.value}-passages")


def passages_report(pair_set: PairSet) -> Path:
    """Snapshot report (computed from gitignored data/) - see reports README."""
    return repo_root() / "registry" / "reports" / f"{pair_set.value}-passages.md"


def destyle_raw(pair_set: PairSet) -> Path:
    """Raw Batch API responses, append-only. Written BEFORE parsing: these bytes
    are what the money bought, so a parser bug must cost a re-parse and never a
    re-purchase. Also the ledger a resumed run subtracts against."""
    return repo_root() / "data" / "pairs" / f"{pair_set.value}-destyle-raw.jsonl"


def batch_receipts() -> Path:
    """Committed batch-id receipts, appended at SUBMIT time. Results live
    server-side for weeks; without the id that copy is unreachable, so this is
    the difference between a re-download and paying twice."""
    return repo_root() / "registry" / "datasets" / "pairs" / "batches.jsonl"


def pairs_shard(pair_set: PairSet) -> Path:
    """The generated (plain -> cankar) pairs."""
    return repo_root() / "data" / "pairs" / f"{pair_set.value}-pairs.jsonl"


def pairs_manifest(pair_set: PairSet) -> Path:
    return dataset_manifest("pairs", f"{pair_set.value}-pairs")


def pairs_samples(pair_set: PairSet) -> Path:
    """Committed before/after excerpts quoted by docs/. Small and in git on
    purpose: the pair shards are gitignored, so without this a published sample
    could not be checked against the real data.

    Set-parameterised like every other artifact here: unparameterised, a holdout
    run silently overwrote the training excerpts docs/ is gated against - the
    exact shared-file hazard PairSet exists to prevent, caught by that gate."""
    return repo_root() / "registry" / "datasets" / "pairs" / f"{pair_set.value}-samples.jsonl"


def dataset_card() -> Path:
    """The card published to the Hub. In registry/, not gitignored data/: what
    gets published must be reviewable in git like any other committed claim."""
    return repo_root() / "registry" / "datasets" / "pairs" / "DATASET_CARD.md"


def rejected_pairs(pair_set: PairSet) -> Path:
    """Responses paid for but unusable. Kept as evidence about the model and the
    prompt - and never re-sent, since `already_done` reads the raw log."""
    return repo_root() / "data" / "pairs" / f"{pair_set.value}-rejected.jsonl"


def pairs_report(pair_set: PairSet) -> Path:
    """Snapshot report (computed from gitignored data/) - see reports README."""
    return repo_root() / "registry" / "reports" / f"{pair_set.value}-pairs.md"


def holdout_manifest() -> Path:
    """Frozen held-out eval set (ADR 0013): generated once, committed,
    provenance-stamped like registry/datasets/, load-bearing - never
    hand-edited. JSON, not TOML: it is generated provenance, not curated input."""
    return repo_root() / "registry" / "evals" / "holdout.json"


def holdout_report() -> Path:
    """Snapshot report (computed from gitignored data/) - see reports README."""
    return repo_root() / "registry" / "reports" / "eval-holdout.md"


def bpb_manifest() -> Path:
    """Frozen held-out BPB for the canonical checkpoints (ADR 0016). The public
    quality claims (README badge, docs/cankar-v1.md, the landing page) cite these
    numbers; this is the artifact they are auditable against."""
    return repo_root() / "registry" / "evals" / "bpb.json"


def bpb_report() -> Path:
    """Snapshot report (computed from gitignored data/ + checkpoints/)."""
    return repo_root() / "registry" / "reports" / "bpb.md"


def style_manifest() -> Path:
    """Frozen style-classifier provenance (ADR 0015): versions, config, seed,
    metrics, artifact sha256, deploy-validation status. Committed, load-bearing."""
    return repo_root() / "registry" / "evals" / "style.json"


def style_report() -> Path:
    """Human-audited confound report (ADR 0015: top features, ablation, per-author
    confusion). Committed - no quality claim rides on the classifier until read."""
    return repo_root() / "registry" / "reports" / "style.md"


def style_model(name: str) -> Path:
    """Trained classifier artifact (joblib). Heavy binary -> checkpoints/ is
    gitignored; the manifest pins its sha256 + a reproducibility contract."""
    return repo_root() / "checkpoints" / f"style-{name}.joblib"


def train_config(name: str) -> Path:
    """A committed training preset (configs/train/<name>.toml, ADR 0016)."""
    return repo_root() / "configs" / "train" / f"{name}.toml"


def checkpoints_dir() -> Path:
    """Trained model checkpoints (gitignored heavy artifacts)."""
    return repo_root() / "checkpoints"
