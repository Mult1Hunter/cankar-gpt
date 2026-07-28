"""Segmenter gates - one case per data class enumerated in the design brief.

Every fixture below is REAL text lifted from the corpus, not invented (ADR 0006:
the bibliography detector shipped green against synthetic tests and amputated
~150 poems). The headline case is `test_mid_sentence_ellipsis_is_not_a_boundary`:
the chunker's `[.!?…]\\s+` splitter cuts at 4,273 such sites - 28.5% of all
15,004 `...` occurrences in the Cankar slice - and reusing it here would truncate
sources into fluent, plausible, wrong de-stylings.
"""

from __future__ import annotations

from cankar.pairs.segment import (
    RejectReason,
    SegmentParams,
    is_source_doc,
    iter_paragraphs,
    passage_id,
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
    """Rare (67 hits / 246 docs) but the capital-letter rule alone cannot see
    these - `dr. Ivan` looks exactly like a boundary."""
    assert len(split_sentences("Obiskal ga je dr. Ivan Tavčar tisto jutro.")) == 1


def test_ordinal_does_not_end_a_sentence() -> None:
    assert len(split_sentences("Bilo je 1. maja tistega leta.")) == 1


def test_dialogue_close_then_capital_splits() -> None:
    """All three transcription conventions close a sentence."""
    for quoted in ('„Dolgo sem hodil, zdaj sem prišel?"', "»Pij, ljubica moja!«"):
        text = f"{quoted} Naslonila je glavo na njegovo ramo."
        assert len(split_sentences(text)) == 2, quoted


def test_plain_sentences_split() -> None:
    text = "Misel je bila lepa. Vrnil se je domov. Nato je zaspal."
    assert len(split_sentences(text)) == 3


# --- paragraph and passage policy -------------------------------------------


def test_headings_are_rejected_not_segmented() -> None:
    """534 roman-numeral marks / 92 ALLCAPS titles: not prose."""
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
    para = "Vrnil se je domov po dolgi poti. Nato je legel in premišljeval. Zunaj je snežilo. "
    passages, _ = segment_doc(_doc(para * 4), PARAMS)
    assert passages
    for p in passages:
        assert PARAMS.min_sentences <= p.n_sentences <= PARAMS.max_sentences
        assert PARAMS.min_chars <= p.n_chars <= PARAMS.max_chars


def test_iter_paragraphs_flattens_internal_wrapping() -> None:
    assert list(iter_paragraphs("ena\ndve\n\ntri")) == ["ena dve", "tri"]


# --- eligibility: the contamination gate ------------------------------------


def test_holdout_urls_are_excluded() -> None:
    """A pair built from a held-out work contaminates the Phase 6 evaluation,
    and the contamination is one-way and undetectable afterwards."""
    doc = _doc("x")
    assert is_source_doc(doc, frozenset())
    assert not is_source_doc(doc, frozenset({doc["url"]}))


def test_non_cankar_and_dlib_docs_are_not_sources() -> None:
    assert not is_source_doc(_doc("x", author="Josip Jurčič"), frozenset())
    assert not is_source_doc(_doc("x", source="dlib"), frozenset())
    assert not is_source_doc(
        {"title": "T", "url": "u", "text": "x", "source": "wikivir", "author": None}, frozenset()
    )


# --- content addressing: what makes de-styling resumable --------------------


def test_passage_id_is_stable_and_nfc_insensitive() -> None:
    """Ids key the Batch API requests; an NFD-vs-NFC slip would mint a second id
    for identical text and re-bill work already paid for."""
    import unicodedata

    text = "Črtica o veselju in žalosti."
    assert passage_id(text) == passage_id(unicodedata.normalize("NFD", text))
    assert passage_id(text) != passage_id(text + " Nato je odšel.")
    assert len(passage_id(text)) == 16
