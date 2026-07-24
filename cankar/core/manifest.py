"""Shard provenance manifests (ADR 0003): regenerate and diff to verify.

Every corpus shard `<name>.jsonl` gets a sibling `<name>.manifest.json` that is
the authority on its expected schema, counts, and sanity band - the corpus-qa
agent reads it before auditing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from cankar.core.errors import CankarError

_M = TypeVar("_M", bound=BaseModel)


class ShardManifest(BaseModel):
    schema_version: int = 1
    source: str  # e.g. "wikivir"
    script: str  # repo-relative generating script
    git_sha: str
    retrieved_at: str  # ISO 8601 UTC
    args: dict[str, object]  # CLI args used, for reproduction
    n_docs: int
    n_chars: int
    n_words: int
    sha256: str  # content hash of the shard file
    expected_band_words: tuple[int, int] | None = None  # sanity band (corpus-qa)
    # per-reason drop counts for sources that filter by count, not per-item triage
    # (ADR 0004 amendment: this IS the committed "never silently dropped" record)
    skip_counts: dict[str, int] | None = None


def git_sha() -> str:
    """Short HEAD sha, '-dirty'-suffixed when the tree has uncommitted changes -
    a clean sha in a manifest must mean 'regenerate and diff' is followable
    (ADR 0003; design-review 2026-07)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = out.stdout.strip()
        # untracked-files=no: git-describe --dirty semantics. Untracked files
        # cannot corrupt regeneration of tracked state - and artifacts awaiting
        # their first commit must not dirty their own provenance stamp.
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        )
        return sha if not status.stdout.strip() else f"{sha}-dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def library_versions(*dists: str) -> dict[str, str]:
    """{'<dist>_version': installed version} for provenance manifests (ADR 0003).
    The single home for dependency-version capture across stages - the suffixed
    key matches the tokenizer-manifest convention (design-review 2026-07)."""
    return {f"{dist}_version": _pkg_version(dist) for dist in dists}


def load_frozen(path: Path, model_cls: type[_M], run_hint: str) -> _M:
    """Load a frozen provenance manifest, failing loud if it was never generated -
    the shared shape behind the eval-stage `load_*` readers (UTF-8: non-ASCII
    author names)."""
    if not path.exists():
        raise CankarError(f"{model_cls.__name__} not frozen: {path} (run: {run_hint})")
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def write_manifest(manifest: BaseModel, out: Path) -> Path:
    """Write to the committed ledger path (cankar.core.paths.dataset_manifest) -
    manifests are provenance and live in git, never only beside gitignored data."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n")
    return out
