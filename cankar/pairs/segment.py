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
   already 2 sentences (p90 = 5), so paragraphs mostly ARE the 2-6 sentence
   unit; a window that spans them would join text across a scene break for no
   gain.

   The tail is the exception, and the docstring used to overstate this: a
   paragraph longer than `max_sentences` is windowed, so **19.1% of passages
   start mid-paragraph** and can open on a dangling demonstrative ("Ta ni
   zmerjal ...") or a conjunction. `mid_paragraph` counts them in the manifest
   rather than leaving the class invisible (design-review 2026-07-28).

   Kept rather than dropped, on a specific argument: BOTH sides of such a pair
   share the same missing antecedent, so the pair is internally consistent and
   teaches fragment-to-fragment rewriting, which is fine. The real risk is
   narrower - a de-styler that INVENTS a referent to smooth the opening - and
   that is a `SOURCE_FIDELITY` violation no current filter can see. Dropping
   non-initial windows is the conservative alternative and costs ~2,670
   passages out of a surplus that can afford it.

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
from cankar.core.works import genre_of

log = logging.getLogger("cankar.pairs")

SEGMENTER_VERSION = 2  # v2: `“` closes a sentence; genre filter; dedup counted

# The author ledger the genre filter reads (registry/works/<slug>.jsonl).
WORKS_LEDGER = "cankar"

# The one source whose paragraph structure survives transcription (see docstring).
SOURCE_WIKIVIR = "wikivir"

# Genres eligible for de-styling, read from `registry/works/cankar.jsonl`'s
# committed `genre` field (ADR 0004). ALLOWLIST, not a denylist, and the rule is
# "only genres that positively assert prose".
#
# This exists because segmenting Cankar's six plays as prose produced 1,223
# passages (7.1%) with speaker labels glued into the source - "Kantor Ne bodi
# neusmiljena, jaz sem tvoj oce" - plus stage directions and mid-turn
# parentheticals (design-review 2026-07-28). In drama a paragraph is a speaker
# turn, so the paragraph unit this module relies on means something else there.
# Those pairs would have taught the Phase 6 styler to emit speaker labels.
#
# Everything else is denied, INCLUDING works whose ledger row records no genre:
# a venue label ("Dela, objavljena v Slovenskem narodu") asserts nothing about
# form and may contain drama, and an absent genre asserts nothing at all. That
# costs ~3.5k passages against a ~10k need - the surplus is what pays for
# default-deny, exactly as with dLib.
PROSE_GENRES = frozenset({"Proza", "Pripovedni spisi", "Esejistika", "Mladinska dela"})

# Sentence-terminal punctuation, plus any run of closing quotes. The corpus mixes
# three conventions - `»` (9,170 hits / 175 docs), `„` (12,454 / 72) and `"`
# (4,225 / 46) - because wikivir and dLib transcribe differently, so all three
# close a sentence here.
#
# `“` (U+201C) MUST be in this class: Slovene low-high quoting closes `„` with
# `“`, and it does so 9,154 times against `"`'s 2,009. Omitting it swallowed
# 2,429 boundaries in the source slice and understated n_sentences on 951 of
# 17,304 passages, which silently broke the 2-6 sentence contract
# (design-review 2026-07-28). It is deliberately ALSO in _OPENERS: the same
# glyph opens English-style quotes, and position disambiguates.
_TERMINAL = re.compile(r"[.!?…]+[»«”“\"']*\s+")

# A real sentence start: a capital, or a quote/dash opening a line of dialogue.
_OPENERS = "»„“\"'-—–("

# Tokens that take a period without ending a sentence. Deliberately short: the
# splitter inspects 158 abbreviation sites across 62 docs, and this rule changes
# the outcome at 126 of them (51 docs) - a rounding error against 101,509
# sentences, not the main defence. The capital-letter rule below is.
_ABBREVIATIONS = frozenset(
    {"dr", "g", "ga", "sv", "št", "str", "prim", "tj", "npr", "itd", "oz", "sl", "op"}
)

_WORD_BEFORE = re.compile(r"(\w+)\s*$")

# Roman-numeral section marks (534 across 69 docs) and ALLCAPS titles (95 / 15).
# Not prose: a passage made of one would de-style into nonsense.
_HEADING = re.compile(r"^(?:[IVXL]{1,6}\.?|[A-ZČŠŽ][A-ZČŠŽ \-]{3,})$")

# The same numeral where the source has no blank line after it, so the section
# mark and the first sentence flatten into one paragraph: "I. Solnce se je bilo
# skrilo ...". 22 passages carried one and counted it as a sentence
# (design-review 2026-07-28). Roman numerals ONLY - a leading arabic "1." is far
# more often an ordinal date ("1. maja") than a section mark.
#
# The undotted form ("VII Ko sem prvikrat videl smrt ...") needs TWO+ letters:
# bare "V" is the Slovene preposition and "V Ljubljani" is ordinary prose, so
# stripping a one-letter undotted numeral would eat real text. No Slovene word
# is spelled from IVXL alone at length >= 2.
_LEADING_HEADING = re.compile(r"^(?:[IVXL]{1,6}\.|[IVXL]{2,6})\s+(?=[A-ZČŠŽ»„“])")


# Not a rejection: a COUNTER, carried in the same dict so the manifest surfaces
# it. A passage windowed out of a long paragraph is kept (see module docstring),
# but the class must be visible rather than implied.
_MID_PARAGRAPH = "mid_paragraph"


class RejectReason(StrEnum):
    """Why a candidate did not become a passage. Closed set, so a StrEnum
    (`.claude/rules/code-standards.md`); the counts land in the manifest so a
    rejection class can never grow silently (ADR 0004's never-silently-dropped
    rule applied to a non-authored split)."""

    HEADING = "heading"
    TOO_FEW_SENTENCES = "too_few_sentences"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    # A window that passed every gate but repeats one already emitted. Counted,
    # not silent: 258 passages were being dropped here into no bucket at all,
    # while this docstring claimed nothing is dropped silently and the report
    # claimed every candidate is accounted for (design-review 2026-07-28).
    DUPLICATE = "duplicate"


class SegmentParams(BaseModel):
    """Thresholds, each carrying its calibration (ADR 0006 companion rule).

    Re-measured at segmenter v2 over the 32,009 candidate windows from the 82
    wikivir prose docs that survive the holdout and genre filters. The first
    version quoted v1 numbers (155 docs, 41,641 candidates) that the genre
    filter had already invalidated - provenance measured on a population that no
    longer exists is worse than none, because it looks checked
    (design-review 2026-07-28).
    """

    # ROADMAP Phase 5 defines the passage as 2-6 sentences.
    min_sentences: int = 2
    max_sentences: int = 6
    # Median sentence is 65 chars. 120 sits just under the median 2-sentence
    # paragraph, so this is a deliberate floor, not a permissive one: it rejects
    # 3,456 windows, 10.8% of all candidates. That is intended - below ~120
    # chars de-styling is near an identity transform and the pair teaches
    # nothing - but the cost is real and paid out of the surplus, not free.
    min_chars: int = 120
    # Kept-passage lengths at v2: p50 304, p90 594, p95 673, p99 764. 800 sits
    # above p95 and bounds per-request cost, rejecting 496 windows (1.5% of the
    # 32,009 candidates). Two earlier figures here did not reproduce - p95 = 545
    # was measured on greedy 4-sentence groups, and 604/41,641 predates the
    # genre filter. Both are corrected rather than the conclusion retrofitted.
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
    doc_skips: dict[str, int]
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
        flat = _LEADING_HEADING.sub("", " ".join(para.split()))
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
            mid_paragraph = i > 0
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
            if mid_paragraph:
                rejects[_MID_PARAGRAPH] += 1
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


class DocSkip(StrEnum):
    """Why an eligible-looking doc contributed nothing. Doc-level, so separate
    from RejectReason, which counts candidate windows inside a kept doc."""

    NOT_CANKAR_PROSE_SOURCE = "not_cankar_prose_source"
    HELD_OUT = "held_out"
    GENRE_NOT_PROSE = "genre_not_prose"


def classify_doc(
    doc: dict, excludes: frozenset[str], genres: dict[str, str | None]
) -> DocSkip | None:
    """None when the doc is eligible for segmentation, else why it was skipped.

    The holdout filter is load-bearing: a pair built from a held-out work
    contaminates the Phase 6 evaluation of the styler, one-way and undetectably.
    The genre filter keeps drama out of a prose segmenter.
    """
    author = doc.get("author") or ""
    if "Cankar" not in author or doc.get("source") != SOURCE_WIKIVIR:
        return DocSkip.NOT_CANKAR_PROSE_SOURCE
    if doc["url"] in excludes:
        return DocSkip.HELD_OUT
    if genre_of(genres, doc["title"], author) not in PROSE_GENRES:
        return DocSkip.GENRE_NOT_PROSE
    return None


def segment_corpus(
    corpus_path: Path,
    excludes: frozenset[str],
    genres: dict[str, str | None],
    params: SegmentParams,
) -> SegmentResult:
    """Segment every eligible doc, deduplicating by passage_id.

    Dedup matters because the corpus keeps excerpt-vs-volume overlaps that the
    holdout's containment closure only resolves for held-out works; two printings
    of the same crtica would otherwise both bill.
    """
    seen: set[str] = set()
    passages: list[Passage] = []
    rejects: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    n_docs = 0
    for doc in iter_jsonl_docs(corpus_path, "run: cankar corpus merge"):
        skip = classify_doc(doc, excludes, genres)
        if skip is not None:
            # not_cankar_prose_source is the whole non-Cankar corpus and would
            # swamp the ledger; the two decisions worth auditing are recorded.
            if skip is not DocSkip.NOT_CANKAR_PROSE_SOURCE:
                skips[skip] += 1
            continue
        n_docs += 1
        kept, doc_rejects = segment_doc(doc, params)
        rejects.update(doc_rejects)
        for p in kept:
            if p.passage_id in seen:
                rejects[RejectReason.DUPLICATE] += 1
                continue
            seen.add(p.passage_id)
            passages.append(p)
    return SegmentResult(
        passages=passages,
        reject_counts={str(k): v for k, v in sorted(rejects.items())},
        doc_skips={str(k): v for k, v in sorted(skips.items())},
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

    Deliberately does NOT stamp `REGISTER_SHA256`. Segmentation never reads the
    register - passages.jsonl is byte-identical for any register text - so the
    stamp recorded an input this artifact does not have, and would have marked
    the passages stale on an edit that cannot affect them while protecting
    nothing. It belongs on the de-styled PAIRS manifest, where the register is a
    real input (design-review 2026-07-28).
    """

    schema_version: int = 1
    segmenter_version: int
    corpus_sha256: str
    holdout_sha256: str
    works_sha256: str  # the ledger the genre filter read
    # ADR 0003's contract is "regenerate and diff"; without the artifact's own
    # hash you can only compare counts, and the next stage spends real money
    # against whatever passages.jsonl happens to be on disk. Three sibling
    # manifests already hash their artifact - this was an omission, not a choice.
    passages_sha256: str
    git_sha: str
    created_at: str
    source: str
    prose_genres: list[str]
    params: SegmentParams
    n_docs: int
    n_passages: int
    n_chars: int
    reject_counts: dict[str, int]
    doc_skips: dict[str, int]


def write_passages_report(out: Path, manifest: PassagesManifest) -> Path:
    """Human-readable face of the passages manifest."""
    m = manifest
    # Numerator and denominator must come from the SAME side of dedup. `duplicate`
    # is a reject reason, so kept + all rejects is the true candidate count; the
    # earlier version added post-dedup kept to pre-dedup rejects and divided by a
    # number that described neither (design-review 2026-07-28).
    total = m.n_passages + sum(v for k, v in m.reject_counts.items() if k != _MID_PARAGRAPH)
    lines: list[str] = [
        generated_marker("cankar pairs segment", snapshot=True),
        "",
        "# De-styling passages - Phase 5 source side",
        "",
        f"Corpus sha256 `{m.corpus_sha256}`.",
        f"Held-out set sha256 `{m.holdout_sha256}` (registry/evals/holdout.json).",
        f"Works ledger sha256 `{m.works_sha256}` (registry/works/cankar.jsonl).",
        f"Passages sha256 `{m.passages_sha256}` (data/pairs/passages.jsonl).",
        f"Segmenter v{m.segmenter_version}, cut at {m.created_at} (git `{m.git_sha}`).",
        "",
        f"**{m.n_passages:,} passages** ({m.n_chars:,} chars) from {m.n_docs} `{m.source}`",
        "Cankar docs. Passages are paragraph-bounded and content-addressed by",
        "`passage_id`, so de-styling is resumable without re-billing work already",
        "done.",
        "",
        "| threshold | value |",
        "|---|---:|",
        f"| sentences per passage | {m.params.min_sentences}-{m.params.max_sentences} |",
        f"| chars per passage | {m.params.min_chars}-{m.params.max_chars} |",
        f"| eligible genres | {', '.join(sorted(m.prose_genres))} |",
        "",
        "## Documents skipped",
        "",
        "Held-out works are excluded so Phase 6's evaluation stays clean. Non-prose",
        "genres are excluded because a paragraph means something else in drama - a",
        "speaker turn - and segmenting plays as prose glues speaker labels into the",
        "source text.",
        "",
        "| reason | docs |",
        "|---|---:|",
    ]
    for reason, count in sorted(m.doc_skips.items()):
        lines.append(f"| `{reason}` | {count:,} |")
    lines += [
        "",
        "## Rejections",
        "",
        f"Every one of the {total:,} candidate windows is accounted for - a rejection",
        "class cannot grow silently.",
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
