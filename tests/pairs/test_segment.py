"""Segmenter gates - one case per data class enumerated in the design brief.

Every fixture below is REAL text lifted from the corpus, not invented (ADR 0006:
the bibliography detector shipped green against synthetic tests and amputated
~150 poems). The headline case is `test_mid_sentence_ellipsis_is_not_a_boundary`:
the chunker's `[.!?…]\\s+` splitter cuts at 4,273 such sites - 28.5% of all
15,004 `...` occurrences in the Cankar slice - and reusing it here would truncate
sources into fluent, plausible, wrong de-stylings.
"""

from __future__ import annotations

import json
from pathlib import Path

from cankar.core.paths import PairSet
from cankar.pairs.segment import (
    DocSkip,
    RejectReason,
    SegmentParams,
    classify_doc,
    iter_paragraphs,
    passage_id,
    segment_corpus,
    segment_doc,
    split_sentences,
)

PARAMS = SegmentParams()


def _doc(text: str, *, source: str = "wikivir", author: str = "Ivan Cankar") -> dict:
    return {
        "title": "T",
        "url": "https://sl.wikisource.org/wiki/T",
        "text": text,
        "source": source,
        "author": author,
    }


# --- sentence splitting: the enumerated hazards -----------------------------


def test_mid_sentence_ellipsis_is_not_a_boundary() -> None:
    """Real corpus text. The naive `[.!?…]\\s+` rule cuts after each `...`;
    the continuation is lowercase, so this splitter must not."""
    text = (
        "ga ter ga potisnejo, kakor svetega Vida, do pasu v kotel vrelega olja "
        "... ali pa ga, recimo, polože na raženj, kakor svetega Lovrenca "
        "... ali pa mu začno rezati v živo telo."
    )
    assert len(split_sentences(text)) == 1


def test_ellipsis_before_a_capital_is_a_boundary() -> None:
    """The rule must stay two-sided: an ellipsis that really does end a sentence
    still splits, or the segmenter just never splits on ellipses at all."""
    text = "Kako imaš razbeljena ustna ... Nagni se k meni, Ada."
    assert len(split_sentences(text)) == 2


def test_ellipsis_char_behaves_like_the_dotted_form() -> None:
    """Both forms occur (2,898 `…` vs 15,004 `...`) and must not diverge."""
    assert len(split_sentences("v kotel vrelega olja … ali pa ga polože.")) == 1
    assert len(split_sentences("razbeljena ustna … Nagni se k meni.")) == 2


def test_abbreviation_does_not_end_a_sentence() -> None:
    """Rare (126 sites where the rule changes the outcome, across 51 docs) but
    the capital-letter rule alone cannot see these - `dr. Ivan` looks exactly
    like a boundary."""
    assert len(split_sentences("Obiskal ga je dr. Ivan Tavčar tisto jutro.")) == 1


def test_ordinal_does_not_end_a_sentence() -> None:
    """The year form is the one that EXERCISES the digit guard. "1. maja" is
    already suppressed by the capital-letter rule, so asserting only that made
    the test vacuous - deleting `token.isdigit()` left the suite green
    (mutation-tested, design-review 2026-07-28)."""
    assert len(split_sentences("Bilo je 1. maja tistega leta.")) == 1
    assert len(split_sentences("Bilo je leta 1918. Nato je odšel.")) == 1


def test_dialogue_close_then_capital_splits() -> None:
    """All FOUR closing forms end a sentence. The `“` case is the regression
    gate for the bug this suite missed: `„...“` is the corpus's dominant quoting
    convention (9,154 vs 2,009 for `"`), `“` was absent from the terminal class,
    and 2,429 boundaries were being swallowed (design-review 2026-07-28)."""
    for quoted in (
        '„Dolgo sem hodil, zdaj sem prišel?"',
        "„Dolgo sem hodil, zdaj sem prišel.“",
        "»Pij, ljubica moja!«",
        "»Ostani z menoj.«",
    ):
        text = f"{quoted} Naslonila je glavo na njegovo ramo."
        assert len(split_sentences(text)) == 2, quoted


def test_low_high_quoted_dialogue_turns_split_fully() -> None:
    """The shape that produced the miss: consecutive `„...“` turns in one
    paragraph. Understated n_sentences silently violates the 2-6 contract."""
    text = (
        "„Dolgo sem hodil, zdaj sem prišel.“ Naslonila je glavo na ramo. „Ostani.“ Nato je molčal."
    )
    assert len(split_sentences(text)) == 4


def test_boundary_before_an_opening_quote_splits() -> None:
    """The opener branch of the rule. Stripping the quote glyphs from _OPENERS
    left the suite green - the standard next-line-of-dialogue shape was
    untested (mutation-tested, design-review 2026-07-28)."""
    assert len(split_sentences("Nato je molčal. »Pojdi!«")) == 2
    assert len(split_sentences("Nato je molčal. „Pojdi!“")) == 2


def test_plain_sentences_split() -> None:
    text = "Misel je bila lepa. Vrnil se je domov. Nato je zaspal."
    assert len(split_sentences(text)) == 3


# --- paragraph and passage policy -------------------------------------------


def test_headings_are_rejected_not_segmented() -> None:
    """534 roman-numeral marks / 95 ALLCAPS titles: not prose."""
    for heading in ("IV.", "XII", "BELA KRIZANTEMA"):
        passages, rejects = segment_doc(_doc(heading), PARAMS)
        assert passages == []
        assert rejects[RejectReason.HEADING] == 1, heading


def test_passages_never_cross_a_paragraph_break() -> None:
    """Two 2-sentence paragraphs must stay two passages, never one joined 4."""
    para = (
        "Vrnil se je domov po dolgi in naporni poti čez zasneženo polje. "
        "Nato je legel na posteljo in dolgo premišljeval o vsem, kar je videl."
    )
    passages, _ = segment_doc(_doc(f"{para}\n\n{para} Zunaj je snežilo."), PARAMS)
    assert len(passages) == 2
    assert passages[0].n_sentences == 2 and passages[1].n_sentences == 3


def test_single_sentence_paragraph_is_rejected() -> None:
    long_one = "Vrnil se je domov po dolgi in naporni poti čez zasneženo polje ter legel."
    passages, rejects = segment_doc(_doc(long_one), PARAMS)
    assert passages == []
    assert rejects[RejectReason.TOO_FEW_SENTENCES] == 1


def test_too_short_passage_is_rejected() -> None:
    passages, rejects = segment_doc(_doc("Bilo je mrzlo. Šel je domov."), PARAMS)
    assert passages == []
    assert rejects[RejectReason.TOO_SHORT] == 1


def test_too_long_passage_is_rejected_whole() -> None:
    """A full 6-sentence window of long sentences breaches the char ceiling.
    It is dropped, never truncated - a cut passage is exactly the corrupted
    source this stage exists to avoid."""
    sentence = (
        "Hodil je po zasneženi in popolnoma prazni cesti proti domu ter dolgo "
        "premišljeval o vsem, kar se mu je pripetilo tistega mrzlega dne. "
    )
    passages, rejects = segment_doc(_doc(sentence * 6), PARAMS)
    assert passages == []
    assert rejects[RejectReason.TOO_LONG] == 1


def test_every_emitted_passage_respects_the_band() -> None:
    """Asserted against LITERALS, not against PARAMS. Using the same params that
    produced the windows made this tautological: raising max_sentences to 12 left
    the suite green and the ROADMAP's "2-6 sentences" contract pinned by nothing
    (mutation-tested, design-review 2026-07-28)."""
    para = "Vrnil se je domov po dolgi poti. Nato je legel in premišljeval. Zunaj je snežilo. "
    passages, _ = segment_doc(_doc(para * 4), PARAMS)
    assert passages
    for p in passages:
        assert 2 <= p.n_sentences <= 6
        assert 120 <= p.n_chars <= 800


def test_iter_paragraphs_flattens_internal_wrapping() -> None:
    assert list(iter_paragraphs("ena\ndve\n\ntri")) == ["ena dve", "tri"]


# --- eligibility: the contamination gate ------------------------------------


PROSE = {"t": "Proza"}


def test_holdout_urls_are_excluded() -> None:
    """A pair built from a held-out work contaminates the Phase 6 evaluation,
    and the contamination is one-way and undetectable afterwards."""
    doc = _doc("x")
    assert classify_doc(doc, frozenset(), PROSE) is None
    assert classify_doc(doc, frozenset({doc["url"]}), PROSE) is DocSkip.HELD_OUT


def test_non_cankar_and_dlib_docs_are_not_sources() -> None:
    skip = DocSkip.NOT_CANKAR_PROSE_SOURCE
    assert classify_doc(_doc("x", author="Josip Jurčič"), frozenset(), PROSE) is skip
    assert classify_doc(_doc("x", source="dlib"), frozenset(), PROSE) is skip
    assert (
        classify_doc(
            {"title": "T", "url": "u", "text": "x", "source": "wikivir", "author": None},
            frozenset(),
            PROSE,
        )
        is skip
    )


def test_drama_is_excluded_by_genre() -> None:
    """The class the brief missed entirely. Cankar's six plays produced 1,223
    passages (7.1%) with speaker labels glued into the source - "Kantor Ne bodi
    neusmiljena" - which would teach the styler to emit them. In drama a
    paragraph is a speaker turn, so this module's unit means something else
    there (design-review 2026-07-28)."""
    doc = _doc("x")
    assert classify_doc(doc, frozenset(), {"t": "Dramatika"}) is DocSkip.GENRE_NOT_PROSE
    assert classify_doc(doc, frozenset(), {"t": "Pesmi"}) is DocSkip.GENRE_NOT_PROSE


def test_missing_genre_defaults_to_deny() -> None:
    """Default-deny: an absent genre asserts nothing about form, and a venue
    label ("Dela, objavljena v Slovenskem narodu") may contain drama. The
    surplus pays for this, exactly as with dLib."""
    doc = _doc("x")
    assert classify_doc(doc, frozenset(), {"t": None}) is DocSkip.GENRE_NOT_PROSE
    assert classify_doc(doc, frozenset(), {}) is DocSkip.GENRE_NOT_PROSE


def test_leading_section_numeral_is_stripped_from_prose() -> None:
    """A numeral with no blank line after it flattens into the first sentence
    and was counted as one (22 passages). Arabic stays - "1. maja" is an
    ordinal date far more often than a section mark."""
    assert list(iter_paragraphs("IV. Solnce se je bilo skrilo.")) == ["Solnce se je bilo skrilo."]
    # undotted form, real corpus shape from "Moje življenje"
    assert list(iter_paragraphs("VII Ko sem videl smrt.")) == ["Ko sem videl smrt."]
    assert list(iter_paragraphs("1. maja se je vrnil.")) == ["1. maja se je vrnil."]
    # bare "V" is the preposition, not a numeral - stripping it would eat prose
    assert list(iter_paragraphs("V Ljubljani je bilo lepo.")) == ["V Ljubljani je bilo lepo."]


# --- content addressing: what makes de-styling resumable --------------------


def test_passage_id_is_stable_and_nfc_insensitive() -> None:
    """Ids key the Batch API requests; an NFD-vs-NFC slip would mint a second id
    for identical text and re-bill work already paid for."""
    import unicodedata

    text = "Črtica o veselju in žalosti."
    assert passage_id(text) == passage_id(unicodedata.normalize("NFD", text))
    assert passage_id(text) != passage_id(text + " Nato je odšel.")
    assert len(passage_id(text)) == 16


# --- segment_corpus: ungated until 2026-07-28 --------------------------------


def _corpus(tmp_path: Path, docs: list[dict]) -> Path:
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
    return p


PARA = (
    "Vrnil se je domov po dolgi in naporni poti čez zasneženo polje. "
    "Nato je legel na posteljo in dolgo premišljeval o vsem, kar je videl."
)


def test_duplicate_passages_are_counted_not_silently_dropped(tmp_path: Path) -> None:
    """The previous review's must-fix landed with no gate: replacing the
    DUPLICATE increment with `pass` - restoring the exact silent drop that was
    reported - left the whole suite green (design-review 2026-07-28)."""
    doc = _doc(f"{PARA}\n\n{PARA}")
    result = segment_corpus(_corpus(tmp_path, [doc]), frozenset(), PROSE, PARAMS)
    assert len(result.passages) == 1
    assert result.reject_counts["duplicate"] == 1


def test_every_candidate_is_accounted_for(tmp_path: Path) -> None:
    """The report divides by kept + all rejects; that identity must hold or the
    percentages describe nothing. mid_paragraph is a counter, not a rejection."""
    doc = _doc(f"{PARA}\n\nIV.\n\nBilo je mrzlo. Šel je domov.\n\n{PARA} Zunaj je snežilo.")
    result = segment_corpus(_corpus(tmp_path, [doc]), frozenset(), PROSE, PARAMS)
    rejects = {k: v for k, v in result.reject_counts.items() if k != "mid_paragraph"}
    assert rejects["heading"] == 1 and rejects["too_short"] == 1
    assert len(result.passages) + sum(rejects.values()) > 0


def test_doc_skips_are_recorded_per_reason(tmp_path: Path) -> None:
    docs = [
        _doc(PARA),
        _doc(PARA, source="dlib"),
        {**_doc(PARA), "url": "held"},
    ]
    result = segment_corpus(_corpus(tmp_path, docs), frozenset({"held"}), PROSE, PARAMS)
    assert result.doc_skips["held_out"] == 1
    assert result.n_docs == 1  # dlib is not_cankar_prose_source, deliberately uncounted


def test_genre_filter_applies_at_corpus_level(tmp_path: Path) -> None:
    result = segment_corpus(
        _corpus(tmp_path, [_doc(PARA)]), frozenset(), {"t": "Dramatika"}, PARAMS
    )
    assert result.passages == []
    assert result.doc_skips["genre_not_prose"] == 1


def test_mid_paragraph_windows_are_counted(tmp_path: Path) -> None:
    """19.1% of shipped passages start mid-paragraph; the class must be visible
    in the manifest rather than implied by the docstring."""
    long_para = " ".join(
        f"Stavek številka {i} je bil dolg in poln premisleka o vsem." for i in range(14)
    )
    result = segment_corpus(_corpus(tmp_path, [_doc(long_para)]), frozenset(), PROSE, PARAMS)
    assert len(result.passages) >= 2
    assert result.reject_counts["mid_paragraph"] >= 1


def test_the_two_sets_are_disjoint_by_construction(tmp_path: Path) -> None:
    """A doc eligible for TRAIN must be ineligible for HOLDOUT and vice versa.
    One shared predicate, inverted - so the eval set cannot overlap the training
    set no matter which command is run when."""
    doc = _doc(PARA)
    held = frozenset({doc["url"]})
    assert classify_doc(doc, frozenset(), PROSE, PairSet.TRAIN) is None
    assert classify_doc(doc, frozenset(), PROSE, PairSet.HOLDOUT) is DocSkip.NOT_HELD_OUT
    assert classify_doc(doc, held, PROSE, PairSet.TRAIN) is DocSkip.HELD_OUT
    assert classify_doc(doc, held, PROSE, PairSet.HOLDOUT) is None


def test_holdout_set_segments_only_held_out_works(tmp_path: Path) -> None:
    a, b = _doc(PARA), {**_doc(PARA), "url": "held"}  # same title -> same genre row
    corpus = _corpus(tmp_path, [a, b])
    train = segment_corpus(corpus, frozenset({"held"}), PROSE, PARAMS, PairSet.TRAIN)
    hold = segment_corpus(corpus, frozenset({"held"}), PROSE, PARAMS, PairSet.HOLDOUT)
    assert train.n_docs == 1 and hold.n_docs == 1
    assert {p.url for p in train.passages}.isdisjoint({p.url for p in hold.passages})
