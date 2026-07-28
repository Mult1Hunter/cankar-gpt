"""Works registry - the source of truth for known works and where we got them.

One JSONL file per author under registry/ (committed, diffable). Every document
that enters the corpus must map to a registry entry; every known-but-unusable
item (manuscripts, in-copyright editions) is recorded, never silently dropped.
See ADR 0004.

The data model and title-matching keys live in `cankar.core.works` so sibling
stages can read work metadata without importing this stage (same split as
`cankar.core.holdout`). They are re-exported here: this module remains the
import site for everything in the corpus stage, and owns the CURATION logic -
indexing, upsert, coverage, reconciliation.
"""

from __future__ import annotations

from pathlib import Path

from cankar.core.works import (
    Source,
    SourceRef,
    SourceStatus,
    WorkFlag,
    WorkRecord,
    normalize_for_author,
    normalize_title,
    slugify,
)

__all__ = [
    "Registry",
    "Source",
    "SourceRef",
    "SourceStatus",
    "WorkFlag",
    "WorkRecord",
    "normalize_for_author",
    "normalize_title",
    "slugify",
]


class Registry:
    """In-memory registry for one author, keyed by normalized title."""

    def __init__(self, author: str, works: list[WorkRecord] | None = None):
        self.author = author
        self.works: dict[str, WorkRecord] = {}
        self._by_norm: dict[str, str] = {}  # normalized title/alias -> work_id
        for w in works or []:
            self._index(w)

    def _index(self, work: WorkRecord) -> None:
        self.works[work.work_id] = work
        self._by_norm[normalize_title(work.title)] = work.work_id
        for a in work.aliases:
            self._by_norm.setdefault(normalize_title(a), work.work_id)

    def find(self, title: str) -> WorkRecord | None:
        norm = normalize_for_author(title, self.author)
        wid = self._by_norm.get(norm)
        return self.works.get(wid) if wid else None

    def upsert(
        self,
        title: str,
        year: int | None = None,
        genre: str | None = None,
        flags: list[WorkFlag] | None = None,
    ) -> WorkRecord:
        existing = self.find(title)
        if existing:
            if year and not existing.year:
                existing.year = year
            if genre and not existing.genre:
                existing.genre = genre
            for f in flags or []:
                if f not in existing.flags:
                    existing.flags.append(f)
            return existing
        work = WorkRecord(
            work_id=slugify(normalize_for_author(title, self.author)),
            title=title,
            author=self.author,
            year=year,
            genre=genre,
            flags=flags or [],
        )
        self._index(work)
        return work

    def add_alias(self, work: WorkRecord, alias: str) -> None:
        if alias not in work.aliases:
            work.aliases.append(alias)
        self._by_norm.setdefault(normalize_title(alias), work.work_id)

    def add_source(self, work: WorkRecord, ref: SourceRef) -> None:
        """Idempotent by (source, id); an upgrade to `ingested` always wins."""
        for existing in work.sources:
            if existing.source == ref.source and existing.id == ref.id:
                if (
                    ref.status is SourceStatus.INGESTED
                    or existing.status is not SourceStatus.INGESTED
                ):
                    existing.status = ref.status
                if ref.year:
                    existing.year = ref.year
                if ref.note:
                    existing.note = ref.note
                return
        work.sources.append(ref)

    # --- persistence (sorted -> stable diffs) ---

    @classmethod
    def load(cls, path: Path, author: str) -> Registry:
        works = [
            WorkRecord.model_validate_json(line) for line in path.read_text().splitlines() if line
        ]
        return cls(author, works)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            self.works[wid].model_dump_json(exclude_defaults=False) for wid in sorted(self.works)
        ]
        path.write_text("\n".join(lines) + "\n")

    # --- validation ---

    def validate(self, min_year: int | None = None, max_year: int | None = None) -> list[str]:
        problems: list[str] = []
        seen_norm: dict[str, str] = {}
        for wid, w in self.works.items():
            if wid != w.work_id:
                problems.append(f"{wid}: key/work_id mismatch")
            norm = normalize_title(w.title)
            if norm in seen_norm and seen_norm[norm] != wid:
                problems.append(f"duplicate normalized title {norm!r}: {wid} vs {seen_norm[norm]}")
            seen_norm[norm] = wid
            for s in w.sources:
                if s.year and min_year and max_year and not (min_year <= s.year <= max_year):
                    problems.append(
                        f"{wid}: source {s.source}:{s.id} year {s.year} outside "
                        f"plausible range {min_year}-{max_year}"
                    )
        return problems
