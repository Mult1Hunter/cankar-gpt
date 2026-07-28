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
    SYSTEM_PROMPT,
    Anomaly,
    BatchReceipt,
    already_done,
    append_raw,
    build_pairs,
    build_request,
    caron_retention,
    classify_response,
    extract_text,
    load_receipts,
    pending_receipts,
    retryable,
    select_passages,
    write_receipts,
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


def test_length_collapse_and_runaway_are_rejected() -> None:
    short = _response("x", [{"type": "text", "text": "Vstal je."}])
    assert classify_response(_passage(), short)[1] is Anomaly.LENGTH_OUTLIER
    long = _response("x", [{"type": "text", "text": GOOD * 3}])
    assert classify_response(_passage(), long)[1] is Anomaly.LENGTH_OUTLIER


def test_dropped_carons_are_rejected() -> None:
    """Sliding out of Slovene orthography produces fluent text that is wrong in
    a way a length check cannot see."""
    stripped = GOOD.replace("š", "s").replace("ž", "z").replace("Š", "S").replace("č", "c")
    r = _response("x", [{"type": "text", "text": stripped}])
    assert classify_response(_passage(), r)[1] is Anomaly.LOST_DIACRITICS


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
    write_receipts(path, receipts)
    assert [r.batch_id for r in pending_receipts(load_receipts(path))] == ["b"]


def test_receipts_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "batches.jsonl"
    r = BatchReceipt(batch_id="msgbatch_1", created_at="t", model="m", n_requests=42)
    write_receipts(path, [r])
    assert load_receipts(path) == [r]


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
