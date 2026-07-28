"""Works-registry data model (ADR 0004), promoted to core so every stage can
read work metadata without importing the corpus stage (import-linter, ADR 0007).

Same split as `cankar.core.holdout`: the CURATION logic - indexing, upsert,
coverage, reconciliation - stays in `cankar.corpus.registry`, which re-exports
these names so its callers are unchanged. This module is only the frozen data
model, the title-matching keys, and the read helper that other stages need.

Promoted at the second consumer: `cankar.pairs` must filter Cankar's plays out
of prose segmentation, and the genre that says so is already committed data.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from cankar.core.errors import CankarError


class Source(StrEnum):
    """Where corpus text comes from (`.claude/rules/code-standards.md`)."""

    WIKIVIR = "wikivir"
    DLIB = "dlib"
    WIKIPEDIA = "wikipedia"


class SourceStatus(StrEnum):
    """Per-source ingestion state of a work."""

    INGESTED = "ingested"  # text is in a corpus shard
    CANDIDATE = "candidate"  # exists at the source, text not fetched
    SKIPPED_QUALITY = "skipped-quality"  # fetched but failed the OCR quality gate
    SKIPPED_MANUSCRIPT = "skipped-manuscript"  # handwriting scan, OCR unusable by policy
    SKIPPED_RIGHTS = "skipped-rights"  # not public domain at this source
    MISSING = "missing"  # known work, no usable source found yet


class WorkFlag(StrEnum):
    """Closed set of work-level flags."""

    PREVOD = "prevod"  # the author's translation of someone else's work
    DLIB_DISCOVERED = "dlib-discovered"  # found via dLib, absent from Wikivir catalogs
    NOT_BY_AUTHOR = "not-by-author"  # crawled into this author's shard but written by
    # someone else (about-subject memoir/essay, misfiled bibliography) - excluded from the
    # merged corpus by merge.py; distinct from cross-author works kept under their true
    # author via collision_resolution.toml. See ADR 0014.


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Matching key: NFC, casefold, punctuation to spaces, diacritics KEPT."""
    t = unicodedata.normalize("NFC", title).casefold()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def normalize_for_author(title: str, author: str) -> str:
    """Also strip a trailing disambiguator naming the author: "Ada (Ivan Cankar)" -> "ada"."""
    surname = author.split()[-1].casefold()
    t = unicodedata.normalize("NFC", title)
    t = re.sub(
        r"\s*\(([^)]*)\)\s*$",
        lambda m: "" if surname in m.group(1).casefold() else m.group(0),
        t,
    )
    return normalize_title(t)


def slugify(title: str) -> str:
    t = normalize_title(title)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return _WS_RE.sub("-", t).strip("-")


class SourceRef(BaseModel):
    source: Source
    id: str  # wikivir page title or dLib URN
    status: SourceStatus
    year: int | None = None  # publication year of this edition, if known
    note: str = ""


class WorkRecord(BaseModel):
    work_id: str
    title: str  # canonical display title
    author: str
    year: int | None = None  # first known publication year
    genre: str | None = None
    flags: list[WorkFlag] = []
    aliases: list[str] = []  # alternate titles ("gl." cross-references)
    sources: list[SourceRef] = []
    notes: str = ""  # human notes - tooling must never clobber this


def load_work_genres(path: Path) -> dict[str, str | None]:
    """`normalized title -> genre` for one author's committed ledger.

    Aliases are indexed too (setdefault, so the canonical title wins), matching
    `Registry._index`. Genre is often None in the ledger; callers decide policy
    for that - this reader reports what is written, it does not interpret."""
    if not path.exists():
        raise CankarError(f"works registry not found: {path}")
    genres: dict[str, str | None] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            work = WorkRecord.model_validate_json(line)
            genres[normalize_title(work.title)] = work.genre
            for alias in work.aliases:
                genres.setdefault(normalize_title(alias), work.genre)
    return genres


def genre_of(genres: dict[str, str | None], title: str, author: str) -> str | None:
    """Genre for a corpus doc title, or None when the ledger has no genre AND
    when the title matches no row - two different facts that callers of a
    default-deny policy treat the same way. `known_work` separates them."""
    return genres.get(normalize_for_author(title, author))


def known_work(genres: dict[str, str | None], title: str, author: str) -> bool:
    """Whether the title resolves to a ledger row at all (ADR 0004: an
    authored-literary doc that matches nothing is a triage case, not a silent
    drop) - so a caller can tell 'no genre recorded' from 'no such work'."""
    return normalize_for_author(title, author) in genres
