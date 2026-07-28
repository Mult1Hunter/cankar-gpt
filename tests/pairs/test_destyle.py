"""De-styler gates - the money step, so the property under test is "no failure
after payment costs money again".

Response fixtures use REAL shapes observed in the 2026-07-28 pilot (100 live
requests), not invented ones - notably the leading `thinking` block, which
appeared on 4 of 50 Sonnet responses and crashed the naive `content[0]["text"]`
extraction that this module now forbids.
"""

from __future__ import annotations

from pathlib import Path

from cankar.pairs.destyle import (
    MAX_TOKENS,
    MIN_CARONS_FOR_RATIO,
    MIN_SOURCE_SIMILARITY,
    SYSTEM_PROMPT,
    Anomaly,
    BatchReceipt,
    already_done,
    append_raw,
    append_receipt,
    build_pairs,
    build_request,
    caron_retention,
    classify_response,
    extract_text,
    foreign_letters,
    load_receipts,
    misassimilated,
    pending_receipts,
    retryable,
    select_passages,
    source_similarity,
    unrecorded_batches,
)
from cankar.pairs.segment import Passage

SRC = (
    "Šimen se je trudoma vzdignil; tako slab je bil, kakor da je bil prespal deset "
    "dolgih let. Stopiti je hotel na noge, noge pa ga niso več nosile."
)
GOOD = (
    "Šimen se je s težavo vzdignil; bil je tako slaboten, kot da bi spal deset "
    "dolgih let. Hotel je stopiti na noge, a te ga niso več držale."
)


def _passage(pid: str = "aaaa000000000000", text: str = SRC) -> Passage:
    return Passage(passage_id=pid, url="u", title="T", text=text, n_sentences=2, n_chars=len(text))


def _response(pid: str, blocks: list[dict], stop: str = "end_turn") -> dict:
    return {
        "custom_id": pid,
        "result": {
            "type": "succeeded",
            "message": {
                "content": blocks,
                "stop_reason": stop,
                "usage": {"input_tokens": 452, "output_tokens": 120},
            },
        },
    }


def _ok(pid: str = "aaaa000000000000", text: str = GOOD) -> dict:
    return _response(pid, [{"type": "text", "text": text}])


# --- the pilot's lesson: never index content[0] ------------------------------


def test_extract_text_skips_a_leading_thinking_block() -> None:
    """Observed on 4/50 Sonnet responses. `content[0]["text"]` raises KeyError
    there - in a batch run that is a crash or silent loss of paid output."""
    msg = {
        "content": [
            {"type": "thinking", "thinking": "The passage uses archaic forms ..."},
            {"type": "text", "text": GOOD},
        ]
    }
    assert extract_text(msg) == GOOD


def test_extract_text_joins_multiple_text_blocks() -> None:
    msg = {"content": [{"type": "text", "text": "Prvi."}, {"type": "text", "text": "Drugi."}]}
    assert extract_text(msg) == "Prvi.\nDrugi."


def test_a_thinking_only_response_is_not_a_pair() -> None:
    r = _response("x", [{"type": "thinking", "thinking": "..."}])
    assert classify_response(_passage(), r)[1] is Anomaly.NO_TEXT_BLOCK


# --- anomaly classes, one fixture each ---------------------------------------


def test_good_response_yields_no_anomaly() -> None:
    assert classify_response(_passage(), _ok())[1] is None


def test_truncation_is_rejected_never_salvaged() -> None:
    """max_tokens output is fluent and plausible and stops mid-sentence - the
    exact corrupted pair this stage exists to avoid."""
    r = _response("x", [{"type": "text", "text": GOOD}], stop="max_tokens")
    assert classify_response(_passage(), r)[1] is Anomaly.TRUNCATED


def test_per_request_api_error_inside_a_batch_is_rejected() -> None:
    r = {"custom_id": "x", "result": {"type": "errored", "error": {"type": "overloaded_error"}}}
    assert classify_response(_passage(), r)[1] is Anomaly.API_ERROR


def test_empty_output_is_rejected() -> None:
    r = _response("x", [{"type": "text", "text": "   "}])
    assert classify_response(_passage(), r)[1] is Anomaly.EMPTY_OUTPUT


def test_commentary_instead_of_a_rewrite_is_rejected() -> None:
    """Real shape from a throwaway probe: the model explained the passage rather
    than rewriting it, opening with a markdown heading."""
    r = _response("x", [{"type": "text", "text": "# Bilo je sonce.\n\nTo je preprosta izjava."}])
    assert classify_response(_passage(), r)[1] is Anomaly.NOT_A_REWRITE
    eng = _response("x", [{"type": "text", "text": "Here is the plain rendering: Vstal je."}])
    assert classify_response(_passage(), eng)[1] is Anomaly.NOT_A_REWRITE


def test_slovene_prose_beginning_like_a_lead_in_is_not_a_preamble() -> None:
    """ "Tukaj je" is ordinary Slovene and rejected a correct rewrite of
    "Tukaj je napisano, trdno prisito ..." on the real run. Only markup and
    ENGLISH openers mark commentary - the output side is always Slovene
    (2026-07-28)."""
    src = "»Tukaj je napisano, trdno prišito, ni je več moči na svetu, ki bi izbrisala to sramoto!«"
    out = "Tukaj je napisano, trdno prišito, ni več moči na svetu, ki bi izbrisala to sramoto!"
    r = _response("x", [{"type": "text", "text": out}])
    assert classify_response(_passage(text=src), r)[1] is None


def test_length_collapse_and_runaway_are_rejected() -> None:
    short = _response("x", [{"type": "text", "text": "Vstal je."}])
    assert classify_response(_passage(), short)[1] is Anomaly.LENGTH_OUTLIER
    long = _response("x", [{"type": "text", "text": GOOD * 3}])
    assert classify_response(_passage(), long)[1] is Anomaly.LENGTH_OUTLIER


def test_dropped_carons_are_rejected() -> None:
    """Sliding out of Slovene orthography produces fluent text that is wrong in
    a way a length check cannot see. Source must be caron-dense enough for the
    ratio to mean anything - see the vocabulary-substitution test below."""
    caron_rich = (
        "Šel je čez cesto, žalosten in čemeren, ker mu je žena rekla, da še ni čas. "
        "Čez čas je čutil, da mu je težko, in šepetal je žalostno pesem."
    )
    assert sum(c in "čšžČŠŽ" for c in caron_rich) >= MIN_CARONS_FOR_RATIO
    stripped = caron_rich.replace("š", "s").replace("ž", "z").replace("č", "c")
    stripped = stripped.replace("Š", "S").replace("Ž", "Z").replace("Č", "C")
    p = _passage(text=caron_rich)
    r = _response("x", [{"type": "text", "text": stripped}])
    assert classify_response(p, r)[1] is Anomaly.LOST_DIACRITICS


def test_caron_ratio_is_not_applied_to_caron_sparse_sources() -> None:
    """The gate as first written rejected 87 flawless pairs on the real run.
    De-styling legitimately swaps caron-bearing archaisms for caron-free modern
    words - "razkrecil" -> "razprl" leaves a 1-caron source at 0.00 retention -
    so below MIN_CARONS_FOR_RATIO the ratio is noise, not signal (2026-07-28)."""
    src = "Iztegnil je roke proti njej, razkrečil dolge prste."
    out = "Iztegnil je roke proti njej, razprl svoje dolge prste."
    assert sum(c in "čšžČŠŽ" for c in src) < MIN_CARONS_FOR_RATIO
    assert caron_retention(src, out) == 0.0
    assert (
        classify_response(_passage(text=src), _response("x", [{"type": "text", "text": out}]))[1]
        is None
    )


def test_caron_retention_is_one_when_source_has_none() -> None:
    assert caron_retention("Bilo je lepo.", "Bilo je lepo.") == 1.0


# --- resumability: the property the whole design exists for -------------------


def test_custom_id_is_the_content_address(tmp_path: Path) -> None:
    p = _passage()
    req = build_request(p, "claude-sonnet-5")
    assert req["custom_id"] == p.passage_id
    assert req["params"]["max_tokens"] == MAX_TOKENS
    assert req["params"]["system"] == SYSTEM_PROMPT
    assert req["params"]["messages"][0]["content"] == p.text


def test_thinking_is_disabled() -> None:
    """Not a preference. A 20-request validation batch lost one response
    entirely (5%) to a model that spent the whole max_tokens budget on a
    thinking block and emitted no text - paid for, nothing returned. Thinking
    counts against max_tokens (2026-07-28)."""
    req = build_request(_passage(), "claude-sonnet-5")
    assert req["params"]["thinking"] == {"type": "disabled"}


def test_thinking_only_max_tokens_response_is_the_observed_failure() -> None:
    """The exact shape that cost a request: succeeded, stop_reason max_tokens,
    a lone thinking block, no text."""
    r = _response("x", [{"type": "thinking", "thinking": "x" * 200}], stop="max_tokens")
    assert classify_response(_passage(), r)[1] is Anomaly.NO_TEXT_BLOCK


def test_retryable_finds_only_rejected_responses(tmp_path: Path) -> None:
    """--retry-rejected is the one path that knowingly pays twice, so it must
    select exactly the rejected ids and nothing else."""
    raw = tmp_path / "raw.jsonl"
    good, bad = _passage("aaaa000000000000"), _passage("bbbb000000000000")
    append_raw(raw, [_ok(good.passage_id), _response(bad.passage_id, [], stop="max_tokens")])
    assert retryable(raw, [good, bad]) == frozenset({bad.passage_id})


def test_already_paid_passages_are_never_resubmitted(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    passages = [_passage(f"{i:016x}") for i in range(5)]
    append_raw(raw, [_ok(p.passage_id) for p in passages[:3]])

    todo = select_passages(passages, limit=10, done=already_done(raw))
    assert [p.passage_id for p in todo] == [p.passage_id for p in passages[3:]]


def test_rejected_responses_are_not_rebilled(tmp_path: Path) -> None:
    """The subtle one: `already_done` reads the RAW log, not the parsed pairs.
    A response the quality filter threw away was still paid for, and keying off
    pairs.jsonl would buy it a second time."""
    raw = tmp_path / "raw.jsonl"
    p = _passage("bbbb000000000000")
    append_raw(raw, [_response(p.passage_id, [{"type": "text", "text": GOOD}], stop="max_tokens")])

    pairs, rejected = build_pairs([p], raw, "claude-sonnet-5")
    assert pairs == [] and rejected[0].reason is Anomaly.TRUNCATED
    assert select_passages([p], limit=10, done=already_done(raw)) == []


def test_selection_is_deterministic_and_extendable(tmp_path: Path) -> None:
    """Raising --limit later must ADD work, never re-bill: the sorted prefix has
    to be stable, so the first N of a bigger run are the same N."""
    passages = [_passage(f"{i:016x}") for i in range(20)]
    first = select_passages(passages, limit=5, done=frozenset())
    bigger = select_passages(passages, limit=12, done=frozenset())
    assert [p.passage_id for p in bigger[:5]] == [p.passage_id for p in first]
    assert select_passages(list(reversed(passages)), 5, frozenset()) == first


def test_results_are_matched_by_id_not_position(tmp_path: Path) -> None:
    """Batch results come back in ARBITRARY order. Matching by position would
    pair every plain rendering with the wrong Cankar source - and the result
    would look entirely well-formed."""
    raw = tmp_path / "raw.jsonl"
    a, b = _passage("aaaa000000000000", SRC), _passage("bbbb000000000000", SRC.upper())
    append_raw(raw, [_ok(b.passage_id, GOOD.upper()), _ok(a.passage_id, GOOD)])

    pairs, _ = build_pairs([a, b], raw, "claude-sonnet-5")
    by_id = {p.passage_id: p for p in pairs}
    assert by_id[a.passage_id].cankar == a.text
    assert by_id[b.passage_id].cankar == b.text


def test_a_response_for_an_unknown_passage_is_quarantined(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    append_raw(raw, [_ok("ffff000000000000")])
    pairs, rejected = build_pairs([_passage()], raw, "claude-sonnet-5")
    assert pairs == [] and len(rejected) == 1


# --- receipts: the route back to results already paid for --------------------


def test_undownloaded_receipts_are_what_a_resumed_run_drains(tmp_path: Path) -> None:
    path = tmp_path / "batches.jsonl"
    receipts = [
        BatchReceipt(batch_id="a", created_at="t", model="m", n_requests=1, downloaded=True),
        BatchReceipt(batch_id="b", created_at="t", model="m", n_requests=1),
    ]
    for r in receipts:
        append_receipt(path, r)
    assert [r.batch_id for r in pending_receipts(load_receipts(path))] == ["b"]


def test_receipts_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "batches.jsonl"
    r = BatchReceipt(batch_id="msgbatch_1", created_at="t", model="m", n_requests=42)
    append_receipt(path, r)
    assert load_receipts(path) == [r]


def test_receipt_log_is_append_only_and_folds_last_write_wins(tmp_path: Path) -> None:
    """The ledger lost a 10,000-request receipt when an unrelated
    `git checkout -- registry/` reverted this tracked file mid-run, because
    write_receipts truncated and rewrote from an in-memory list. Status changes
    are now appended as events (design-review 2026-07-28)."""
    path = tmp_path / "batches.jsonl"
    r = BatchReceipt(batch_id="b1", created_at="t", model="m", n_requests=9)
    append_receipt(path, r)
    append_receipt(path, r.model_copy(update={"downloaded": True}))
    assert len(path.read_text().splitlines()) == 2  # nothing was overwritten
    folded = load_receipts(path)
    assert len(folded) == 1 and folded[0].downloaded is True


def test_unrecorded_api_batches_are_detected(tmp_path: Path) -> None:
    """A billed batch with no receipt cannot be subtracted by already_done, so
    submitting alongside it buys the same passages twice."""
    known = BatchReceipt(batch_id="b1", created_at="t", model="m", n_requests=1)
    assert unrecorded_batches(["b1"], [known]) == []
    assert unrecorded_batches(["b1", "b2"], [known]) == ["b2"]


def test_system_prompt_carries_both_register_halves() -> None:
    """Design invariant #1: the de-styler is consumer (a) of the shared register,
    and it appends the fidelity constraints because a rewrite has a source."""
    from cankar.core.register import PLAIN_REGISTER, SOURCE_FIDELITY

    assert PLAIN_REGISTER in SYSTEM_PROMPT
    assert SOURCE_FIDELITY in SYSTEM_PROMPT


def test_a_passage_retried_successfully_is_not_counted_as_a_loss(tmp_path: Path) -> None:
    """The raw log legitimately holds several records per id: --retry-rejected
    adds one, and re-draining a batch appends the same responses again.
    Resolving per RECORD emitted a duplicate pair for one passage and reported a
    passage that succeeded on retry as rejected (observed 2026-07-28)."""
    raw = tmp_path / "raw.jsonl"
    p = _passage("cccc000000000000")
    append_raw(raw, [_response(p.passage_id, [], stop="max_tokens"), _ok(p.passage_id)])

    pairs, rejected = build_pairs([p], raw, "claude-sonnet-5")
    assert [x.passage_id for x in pairs] == [p.passage_id]
    assert rejected == []


def test_a_redrained_batch_does_not_duplicate_pairs(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    p = _passage("dddd000000000000")
    append_raw(raw, [_ok(p.passage_id), _ok(p.passage_id)])
    pairs, _ = build_pairs([p], raw, "claude-sonnet-5")
    assert len(pairs) == 1


def test_a_passage_that_never_succeeded_is_still_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    p = _passage("eeee000000000000")
    append_raw(raw, [_response(p.passage_id, [], stop="max_tokens")] * 2)
    pairs, rejected = build_pairs([p], raw, "claude-sonnet-5")
    assert pairs == [] and len(rejected) == 1


def test_foreign_script_in_the_output_is_rejected() -> None:
    """84 real responses carried Cyrillic homoglyphs inside Slovene words -
    `vesело`, `neumnе` - visually identical, invisible to every length and
    diacritic check, and a token the Slovene tokenizer has never seen. This
    gate had no test at all until the mutation registry exposed it."""
    cyrillic = "Vstal je in odsel v vesело mesto."  # 'e','о' are Cyrillic here
    assert foreign_letters(cyrillic)
    r = _response("x", [{"type": "text", "text": cyrillic}])
    assert classify_response(_passage(text=SRC), r)[1] is Anomaly.FOREIGN_SCRIPT


def test_foreign_letters_present_in_the_source_are_not_corruption() -> None:
    """Diffed against the source: a Greek letter Cankar himself wrote is not
    the model corrupting anything."""
    src = "Pisal je o Ω in o svetu."
    out = "Pisal je o Ω in o svetu, preprosto."
    assert foreign_letters(out) - foreign_letters(src) == set()
    r = _response("x", [{"type": "text", "text": out}])
    assert classify_response(_passage(text=src), r)[1] is None


def test_the_raw_log_is_append_only(tmp_path: Path) -> None:
    """The raw log is the artifact the money bought. If it truncated, a resumed
    run would re-bill everything already paid for - `already_done` reads it.
    Untested until the mutation registry caught an ambiguous pattern hitting
    this function and surviving (2026-07-28)."""
    raw = tmp_path / "raw.jsonl"
    append_raw(raw, [_ok("aaaa000000000000")])
    append_raw(raw, [_ok("bbbb000000000000")])
    assert len(raw.read_text().splitlines()) == 2
    assert already_done(raw) == frozenset({"aaaa000000000000", "bbbb000000000000"})


def test_sz_assimilation_error_is_rejected() -> None:
    """Slovene assimilates the preposition: `z` before a vowel or voiced sound,
    `s` before a voiceless one. The generated side broke this in 190 of 9,950
    passages against 4 in the untouched public-domain side - a 48x enrichment,
    so it is a generator artifact, not a feature of the language."""
    src = "Gledal ga je z veseljem in dolgo molčal, ne da bi izrekel eno samo besedo."
    bad = "Gledal ga je s občudovanjem in dolgo molčal, ne da bi rekel besedo."
    assert misassimilated(bad) == ["s o"] and misassimilated(src) == []
    r = _response("x", [{"type": "text", "text": bad}])
    assert classify_response(_passage(text=src), r)[1] is Anomaly.MISASSIMILATED


def test_sz_usage_cankar_himself_wrote_is_not_the_models_error() -> None:
    """Diffed against the source, like every other content gate here."""
    src = "Gledal ga je s občudovanjem in dolgo molčal, brez ene same besede."
    out = "Gledal ga je s občudovanjem in molčal, ne da bi rekel eno besedo."
    assert (
        classify_response(_passage(text=src), _response("x", [{"type": "text", "text": out}]))[1]
        is None
    )


def test_correct_assimilation_is_not_flagged() -> None:
    assert misassimilated("Šel je z avtom s prijateljem in z ženo.") == []


def test_a_continuation_instead_of_a_rewrite_is_rejected() -> None:
    """The worst pair in the corpus: given a pure-dialogue passage the model
    narrated what happened NEXT instead of restyling it. Fluent, plausible, and
    it teaches the styler to invent continuations. Similarity 0.316 - the
    minimum across all 9,950 - against a median of 0.85."""
    src = (
        "»Kaj se ti sanja, Aleš?« se je zasmejal Stržinar, krčmar, "
        "in je postavil polno steklenico predenj."
    )
    cont = (
        "Aleš se ni takoj odzval, ampak je nekaj časa strmel predenj, "
        "kot da bi razmišljal o čem drugem."
    )
    assert source_similarity(src, cont) < MIN_SOURCE_SIMILARITY
    r = _response("x", [{"type": "text", "text": cont}])
    assert classify_response(_passage(text=src), r)[1] is Anomaly.DIVERGENT


def test_an_ordinary_heavy_rewrite_is_not_divergent() -> None:
    """The floor must not police legitimate restyling - p1 of the real
    distribution is 0.65, well clear of it."""
    assert source_similarity(SRC, GOOD) >= MIN_SOURCE_SIMILARITY
    assert classify_response(_passage(text=SRC), _ok(text=GOOD))[1] is None


def test_similarity_disables_difflib_autojunk() -> None:
    """autojunk discards any character in over 1% of a sequence - on prose that
    is every vowel and the space, collapsing every ratio toward zero. It ranked
    perfect de-stylings as the most divergent pairs in the corpus."""
    import difflib

    a, b = SRC * 4, GOOD * 4
    assert source_similarity(a, b) > difflib.SequenceMatcher(None, a, b).ratio() + 0.3
