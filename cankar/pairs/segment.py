"""Segment Cankar prose into de-styling passages - the SOURCE side of Phase 5.

A passage is 2-6 sentences of coherent prose, small enough to de-style in one
API call and self-contained enough that the rewrite has no missing antecedent.

Three decisions worth the paragraph, all measured on the real 246-doc Cankar
slice rather than assumed:

1. **This stage does NOT reuse `tokenizer/chunk.py`'s sentence splitter.** That
   one is `[.!?…]\\s+`, which is correct where it lives: a token-budget ladder
   where an over-split still concatenates byte-exact, so its errors cost
   nothing. Here a boundary is semantic. Measured on the corpus, that regex cuts
   at 4,273 mid-sentence ellipses - 28.5% of all 15,004 `...` occurrences - and
   the ellipsis is Cankar's signature device. A truncated source de-styles into
   fluent, plausible, wrong prose, which is invisible in spot-check and teaches
   Phase 6 to hallucinate at boundaries (the ADR 0006 failure shape).

2. **Passages never cross a paragraph break.** The median Cankar paragraph is
   already 2 sentences (p90 = 5), so paragraphs ARE the 2-6 sentence unit; a
   window that spans them would join text across a scene break for no gain.

3. **Wikivir only.** The 41 dLib docs carry a median of ZERO blank-line
   paragraphs at 78-char hard-wrapped lines - the paragraph structure this
   segmenter reads simply is not present, and recovering it means guessing where
   breaks were. With 17.5k passages available from wikivir alone against a
   5-15k target, that guess buys nothing. Recorded as a ROADMAP deferral.

The surplus is the design: everything ambiguous is REJECTED rather than parsed,
because we have roughly 2x the material we need and correctness is cheaper than
coverage here.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from cankar.core.jsonl import iter_jsonl_docs
from cankar.core.reports import generated_marker, write_report

log = logging.getLogger("cankar.pairs")

SEGMENTER_VERSION = 1

# The one source whose paragraph structure survives transcription (see docstring).
SOURCE_WIKIVIR = "wikivir"

# Sentence-terminal punctuation, plus any run of closing quotes. The corpus mixes
# three conventions - `»` (9,170 hits / 175 docs), `„` (12,454 / 72) and `"`
# (4,225 / 46) - because wikivir and dLib transcribe differently, so all three
# close a sentence here.
_TERMINAL = re.compile(r"[.!?…]+[»«”\"']*\s+")

# A real sentence start: a capital, or a quote/dash opening a line of dialogue.
_OPENERS = "»„“\"'-—–("

# Tokens that take a period without ending a sentence. Deliberately short: only
# 67 abbreviation hits exist across all 246 docs (37 of them), so this list is a
# rounding error, not the main defence - the capital-letter rule below is.
_ABBREVIATIONS = frozenset(
    {"dr", "g", "ga", "sv", "št", "str", "prim", "tj", "npr", "itd", "oz", "sl", "op"}
)

_WORD_BEFORE = re.compile(r"(\w+)\s*$")

# Roman-numeral section marks (534 across 69 docs) and ALLCAPS titles (92 / 15).
# Not prose: a passage made of one would de-style into nonsense.
_HEADING = re.compile(r"^(?:[IVXL]{1,6}\.?|[A-ZČŠŽ][A-ZČŠŽ \-]{3,})$")


class RejectReason(StrEnum):
    """Why a candidate did not become a passage. Closed set, so a StrEnum
    (`.claude/rules/code-standards.md`); the counts land in the manifest so a
    rejection class can never grow silently (ADR 0004's never-silently-dropped
    rule applied to a non-authored split)."""

    HEADING = "heading"
    TOO_FEW_SENTENCES = "too_few_sentences"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"


class SegmentParams(BaseModel):
    """Thresholds, each carrying its calibration (ADR 0006 companion rule).

    Measured over 101,509 sentences in 36,830 paragraphs from the 155 wikivir
    Cankar docs left after holdout exclusion.
    """

    # ROADMAP Phase 5 defines the passage as 2-6 sentences.
    min_sentences: int = 2
    max_sentences: int = 6
    # Median sentence is 65 chars, so a typical 2-sentence paragraph is ~130.
    # 120 admits that while rejecting the p5 (53-char) trivia, where de-styling
    # is close to an identity transform and the pair teaches nothing.
    min_chars: int = 120
    # p95 of kept passages is 545. 800 bounds the per-request size while
    # rejecting only 604 of 18,166 candidates (3.3%).
    max_chars: int = 800


class Passage(BaseModel):
    """One de-styling unit. `passage_id` is content-addressed so the Batch API
    step can be resumed: re-running after a crash re-derives the same ids, and
    everything already on disk is subtracted instead of re-purchased."""

    passage_id: str
    url: str
    title: str
    text: str
    n_sentences: int
    n_chars: int


class SegmentResult(BaseModel):
    passages: list[Passage]
    reject_counts: dict[str, int]
    n_docs: int


def passage_id(text: str) -> str:
    """Content address over NFC bytes - the same normalization the corpus is
    ingested under (CLAUDE.md), so an NFD-vs-NFC slip cannot mint a second id
    for identical text."""
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()[:16]


def split_sentences(text: str) -> list[str]:
    """Split one paragraph into sentences.

    The load-bearing rule is that a boundary must be FOLLOWED by a capital or an
    opening quote. Slovene sentences start capitalized, so this single test
    rejects every mid-sentence ellipsis (`... in nato`) without needing to know
    that an ellipsis is what it is looking at.
    """
    out: list[str] = []
    start = 0
    for m in _TERMINAL.finditer(text):
        nxt = text[m.end() : m.end() + 1]
        if not nxt:
            continue
        word = _WORD_BEFORE.search(text[start : m.start()])
        token = word.group(1).lower() if word else ""
        if token in _ABBREVIATIONS or token.isdigit():
            continue
        if not (nxt.isupper() or nxt in _OPENERS):
            continue
        out.append(text[start : m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def iter_paragraphs(text: str) -> Iterator[str]:
    """Blank-line-separated paragraphs, internal whitespace flattened. Flattening
    is safe here and NOT in the chunker: a passage is sent to an API as prose,
    it is never concatenated back into the training stream."""
    for para in re.split(r"\n\s*\n", text):
        flat = " ".join(para.split())
        if flat:
            yield flat


def segment_doc(doc: dict, params: SegmentParams) -> tuple[list[Passage], Counter[str]]:
    """Passages for one corpus doc, plus its per-reason rejection counts."""
    kept: list[Passage] = []
    rejects: Counter[str] = Counter()
    for para in iter_paragraphs(doc["text"]):
        if _HEADING.match(para):
            rejects[RejectReason.HEADING] += 1
            continue
        sentences = split_sentences(para)
        for i in range(0, len(sentences), params.max_sentences):
            group = sentences[i : i + params.max_sentences]
            text = " ".join(group)
            if len(group) < params.min_sentences:
                rejects[RejectReason.TOO_FEW_SENTENCES] += 1
                continue
            if len(text) < params.min_chars:
                rejects[RejectReason.TOO_SHORT] += 1
                continue
            if len(text) > params.max_chars:
                rejects[RejectReason.TOO_LONG] += 1
                continue
            kept.append(
                Passage(
                    passage_id=passage_id(text),
                    url=doc["url"],
                    title=doc["title"],
                    text=text,
                    n_sentences=len(group),
                    n_chars=len(text),
                )
            )
    return kept, rejects


def is_source_doc(doc: dict, excludes: frozenset[str]) -> bool:
    """Cankar prose eligible for pair generation: right author, right source, and
    NOT held out. The holdout filter is the load-bearing one - a pair built from
    a held-out work contaminates the Phase 6 evaluation of the styler, and that
    contamination is one-way and undetectable afterwards."""
    author = doc.get("author") or ""
    return "Cankar" in author and doc.get("source") == SOURCE_WIKIVIR and doc["url"] not in excludes


def segment_corpus(
    corpus_path: Path, excludes: frozenset[str], params: SegmentParams
) -> SegmentResult:
    """Segment every eligible doc, deduplicating by passage_id.

    Dedup matters because the corpus keeps excerpt-vs-volume overlaps that the
    holdout's containment closure only resolves for held-out works; two printings
    of the same crtica would otherwise both bill.
    """
    seen: set[str] = set()
    passages: list[Passage] = []
    rejects: Counter[str] = Counter()
    n_docs = 0
    for doc in iter_jsonl_docs(corpus_path, "run: cankar corpus merge"):
        if not is_source_doc(doc, excludes):
            continue
        n_docs += 1
        kept, doc_rejects = segment_doc(doc, params)
        rejects.update(doc_rejects)
        for p in kept:
            if p.passage_id in seen:
                continue
            seen.add(p.passage_id)
            passages.append(p)
    return SegmentResult(
        passages=passages,
        reject_counts={str(k): v for k, v in sorted(rejects.items())},
        n_docs=n_docs,
    )


def write_passages(out: Path, passages: list[Passage]) -> Path:
    """One JSON object per line, in segmentation order (ADR 0003: JSONL is the
    interchange format between stages). ensure_ascii=False keeps the carons
    readable in the file the de-styler reads back."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(p.model_dump_json() + "\n")
    return out


class PassagesManifest(BaseModel):
    """Committed provenance for the passage set (ADR 0003 shape).

    `register_sha256` is the first stamp of design invariant #1 anywhere in the
    pipeline: it records WHICH register text these passages were cut for, so a
    later edit to `PLAIN_REGISTER` marks the generated pairs stale instead of
    silently mixing two distributions.
    """

    schema_version: int = 1
    segmenter_version: int
    corpus_sha256: str
    holdout_sha256: str
    register_sha256: str
    git_sha: str
    created_at: str
    source: str
    params: SegmentParams
    n_docs: int
    n_passages: int
    n_chars: int
    reject_counts: dict[str, int]


def write_passages_report(out: Path, manifest: PassagesManifest) -> Path:
    """Human-readable face of the passages manifest."""
    m = manifest
    total = m.n_passages + sum(m.reject_counts.values())
    lines: list[str] = [
        generated_marker("cankar pairs segment", snapshot=True),
        "",
        "# De-styling passages - Phase 5 source side",
        "",
        f"Corpus sha256 `{m.corpus_sha256}`.",
        f"Held-out set sha256 `{m.holdout_sha256}` (registry/evals/holdout.json).",
        f"Register sha256 `{m.register_sha256}` (cankar/core/register.py).",
        f"Segmenter v{m.segmenter_version}, cut at {m.created_at} (git `{m.git_sha}`).",
        "",
        f"**{m.n_passages:,} passages** ({m.n_chars:,} chars) from {m.n_docs} `{m.source}`",
        "Cankar docs, held-out works excluded. Passages are paragraph-bounded and",
        "content-addressed by `passage_id`, so de-styling is resumable without",
        "re-billing work already done.",
        "",
        "| threshold | value |",
        "|---|---:|",
        f"| sentences per passage | {m.params.min_sentences}-{m.params.max_sentences} |",
        f"| chars per passage | {m.params.min_chars}-{m.params.max_chars} |",
        "",
        "## Rejections",
        "",
        "Every candidate is accounted for - a rejection class cannot grow silently.",
        "",
        "| reason | count | share of candidates |",
        "|---|---:|---:|",
    ]
    for reason, count in sorted(m.reject_counts.items()):
        lines.append(f"| `{reason}` | {count:,} | {count / total:.1%} |")
    lines += [
        f"| **kept** | **{m.n_passages:,}** | **{m.n_passages / total:.1%}** |",
        "",
        "## Reproducing",
        "",
        "```",
        "uv run cankar pairs segment",
        "```",
    ]
    write_report(out, lines)
    return out
