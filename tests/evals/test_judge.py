"""Judge gates. The controls are the load-bearing part: they are how this
module knows its own instrument works, and a control that cannot fail is worth
nothing - so every one of them is tested against a judge that should fail it.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cankar.core.errors import CankarError
from cankar.evals.judge import (
    ControlKind,
    JudgeItem,
    Verdict,
    build_controls,
    build_request,
    check_controls,
    estimate_cost,
    extract_text,
    parse_verdict,
)

PAIRS = [
    {
        "plain": f"Vrnil se je domov in legel, ker je bil truden {i}.",
        "cankar": f"Vrnil se je na dom svoj, trudoma, in legel {i}.",
    }
    for i in range(8)
]


def _items() -> list[JudgeItem]:
    return build_controls(PAIRS, n=4, seed=1)


def _verdicts(items, *, real, echo, mismatch) -> dict[str, Verdict]:
    """Score every control by its kind, so a test states only the behaviour it
    is exercising. `real`/`echo`/`mismatch` are (meaning, voice) tuples."""
    table = {
        ControlKind.REAL_CANKAR: real,
        ControlKind.ECHO: echo,
        ControlKind.MISMATCH: mismatch,
    }
    out = {}
    for it in items:
        if it.control is ControlKind.NONE:
            continue
        m, v = table[it.control]
        out[it.item_id] = Verdict(item_id=it.item_id, meaning=m, voice=v, fluency=5)
    return out


def test_controls_cover_all_three_kinds() -> None:
    items = _items()
    kinds = {i.control for i in items}
    assert kinds == {ControlKind.REAL_CANKAR, ControlKind.ECHO, ControlKind.MISMATCH}
    assert len(items) == 12  # 4 sampled pairs x 3 kinds


def test_a_mismatch_control_never_matches_its_own_source() -> None:
    """A mismatch that accidentally selects its own pair is a control that
    asserts nothing while still reporting a pass."""
    items = build_controls(PAIRS, n=len(PAIRS), seed=3)
    by_id = {i.item_id: i for i in items}
    for item in items:
        if item.control is ControlKind.MISMATCH:
            idx = item.item_id.rsplit("-", 1)[1]
            real = by_id[f"ctl-real-{idx}"]
            assert item.candidate != real.candidate, "mismatch got its own pair"


def test_an_echo_control_is_the_source_verbatim() -> None:
    for item in _items():
        if item.control is ControlKind.ECHO:
            assert item.candidate == item.source


def test_a_working_judge_is_usable() -> None:
    """Real Cankar scores high on both; echo keeps meaning but loses voice;
    mismatch keeps voice but loses meaning."""
    items = _items()
    outcome = check_controls(items, _verdicts(items, real=(5, 5), echo=(5, 1), mismatch=(1, 5)))
    assert outcome.usable
    assert outcome.voice_margin == pytest.approx(4.0)
    assert outcome.meaning_margin == pytest.approx(4.0)
    assert outcome.axes_independent


def test_a_judge_that_scores_everything_high_is_rejected() -> None:
    """The exact failure the style classifier had, one layer up: plausible
    numbers, no discrimination. Nothing in the scores alone reveals it."""
    items = _items()
    outcome = check_controls(items, _verdicts(items, real=(5, 5), echo=(5, 5), mismatch=(5, 5)))
    assert not outcome.usable
    assert any("cannot see styling" in f for f in outcome.failures)
    assert any("cannot see content" in f for f in outcome.failures)


def test_a_judge_blind_to_styling_is_rejected() -> None:
    """Scores the source echoed back as well-styled - it is reading 'is this
    literary Slovene', which the source already is."""
    items = _items()
    outcome = check_controls(items, _verdicts(items, real=(5, 5), echo=(5, 5), mismatch=(1, 5)))
    assert not outcome.usable
    assert any("cannot see styling" in f for f in outcome.failures)


def test_a_judge_that_mistakes_period_vocabulary_for_meaning_is_rejected() -> None:
    """Unrelated content in real Cankar scores high on MEANING. This is the
    classifier's failure repeated: period vocabulary read as substance."""
    items = _items()
    outcome = check_controls(items, _verdicts(items, real=(5, 5), echo=(5, 1), mismatch=(5, 5)))
    assert not outcome.usable
    assert any("cannot see content" in f for f in outcome.failures)


def test_axes_that_move_together_are_rejected() -> None:
    """A judge with one internal quality dimension wearing three labels: the
    mismatched passage loses voice along with meaning, though it IS real Cankar.
    Both margins still pass, so only the independence check catches this."""
    items = _items()
    outcome = check_controls(items, _verdicts(items, real=(5, 5), echo=(5, 1), mismatch=(1, 1)))
    assert outcome.voice_margin >= 1.0 and outcome.meaning_margin >= 1.0
    assert not outcome.axes_independent
    assert not outcome.usable
    assert any("not independent" in f for f in outcome.failures)


def test_a_missing_control_verdict_raises() -> None:
    """A dropped control silently shrinks the evidence the run is judged on."""
    items = _items()
    verdicts = _verdicts(items, real=(5, 5), echo=(5, 1), mismatch=(1, 5))
    verdicts.pop(next(i.item_id for i in items if i.control is ControlKind.ECHO))
    with pytest.raises(CankarError, match="no verdict"):
        check_controls(items, verdicts)


def test_a_batch_without_controls_raises() -> None:
    plain = [JudgeItem("x1", "a", "b")]
    with pytest.raises(CankarError, match="controls in this run"):
        check_controls(plain, {})


def test_the_control_kind_never_reaches_the_model() -> None:
    """A control the judge can recognise is not a control.

    Stated as an invariance rather than a keyword scan: two items with the same
    text must produce byte-identical model-visible content whatever their
    control kind. The earlier keyword version matched the rubric's own sentence
    about echoing and failed on its own instructions.

    `custom_id` is excluded deliberately - it is Batch API correlation metadata
    and is never part of the prompt, which the message assertion below pins.
    """
    text = ("izvorno besedilo", "izvorno besedilo")
    seen = set()
    for kind in ControlKind:
        req = build_request(JudgeItem(f"id-{kind.value}", *text, kind))
        seen.add(json.dumps([req["params"]["system"], req["params"]["messages"]], sort_keys=True))
    assert len(seen) == 1, "the control kind changes what the model is shown"

    req = build_request(JudgeItem("ctl-echo-3", *text, ControlKind.ECHO))
    prompt = req["params"]["system"] + json.dumps(req["params"]["messages"])
    assert "ctl-echo-3" not in prompt


def test_thinking_is_disabled_so_it_does_not_eat_max_tokens() -> None:
    req = build_request(JudgeItem("a", "s", "c"))
    assert req["params"]["thinking"] == {"type": "disabled"}


def test_verdicts_are_read_from_every_text_block() -> None:
    """Never content[0]: with thinking on, block 0 is a thinking block."""
    msg = {
        "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": '{"meaning": 4, "voice": 3, "fluency": 5}'},
        ]
    }
    v = parse_verdict("a", extract_text(msg))
    assert (v.meaning, v.voice, v.fluency) == (4, 3, 5)


def test_a_fenced_verdict_still_parses() -> None:
    raw = '```json\n{"meaning": 2, "voice": 5, "fluency": 4, "notes": "ok"}\n```'
    assert parse_verdict("a", raw).voice == 5


@pytest.mark.parametrize("raw", ["not json at all", "", "{oops", '{"meaning": 9}'])
def test_an_unparseable_or_out_of_range_verdict_raises(raw: str) -> None:
    """Never defaults to a score: a silent 0 moves every aggregate it lands in,
    and a vanished control is the one failure this module cannot detect."""
    with pytest.raises((CankarError, ValidationError)):
        parse_verdict("a", raw)


def test_cost_scales_with_the_text_and_halves_for_batch() -> None:
    items = [JudgeItem(f"i{i}", "a" * 300, "b" * 300) for i in range(100)]
    est = estimate_cost(items)
    assert est.n_requests == 100
    full = est.input_tokens / 1e6 * 3.0 + est.output_tokens / 1e6 * 15.0
    assert est.usd_batch == pytest.approx(full / 2, abs=0.01)


def test_the_control_margin_is_the_line_it_claims_to_be() -> None:
    """ADR 0006: every other fixture here sits at margin 0 or 4, so
    MIN_CONTROL_MARGIN could be 0.1 or 3.9 with a green suite. These bracket it."""
    from cankar.evals.judge import MIN_CONTROL_MARGIN, check_controls

    assert MIN_CONTROL_MARGIN == 1.0
    items = _items()
    # voice margin exactly 1.0 (5 -> 4) passes; 0 fails. Meaning margin held wide.
    assert check_controls(items, _verdicts(items, real=(5, 5), echo=(5, 4), mismatch=(1, 5))).usable
    assert not check_controls(
        items, _verdicts(items, real=(5, 5), echo=(5, 5), mismatch=(1, 5))
    ).usable


def test_an_echo_scored_low_on_meaning_is_rejected() -> None:
    """ControlKind declares four MUSTs and only two were enforced: a judge that
    scores the source-verbatim ECHO as poor on MEANING is broken - the echo
    preserves meaning perfectly by construction - and used to pass."""
    from cankar.evals.judge import check_controls

    items = _items()
    outcome = check_controls(items, _verdicts(items, real=(5, 5), echo=(2, 1), mismatch=(1, 5)))
    assert not outcome.usable
    assert any("ECHO meaning" in f for f in outcome.failures)


def test_a_halo_judge_is_rejected_on_the_same_axis() -> None:
    """MISMATCH candidates ARE real Cankar, so their voice must stay near
    REAL_CANKAR's. If it collapses with meaning, one quality dimension is wearing
    three labels. Compared same-axis: the earlier version subtracted a meaning
    score from a voice score, which carries a constant axis offset."""
    from cankar.evals.judge import check_controls

    items = _items()
    outcome = check_controls(items, _verdicts(items, real=(5, 5), echo=(5, 1), mismatch=(1, 2)))
    assert not outcome.axes_independent
    assert any("halo" in f for f in outcome.failures)


def test_scores_come_from_this_batch_not_the_whole_ledger() -> None:
    """The raw ledger is append-only, so a prefix match over it mixes a previous
    larger run's verdicts into a later smaller one - silently averaging over
    generations from a different checkpoint."""
    from cankar.evals.judge import ControlKind, JudgeItem, score_series

    items = _items() + [JudgeItem("holdout-0", "s", "c", ControlKind.NONE)]
    verdicts = _verdicts(items, real=(5, 5), echo=(5, 1), mismatch=(1, 5))
    verdicts["holdout-0"] = Verdict(item_id="holdout-0", meaning=2, voice=2, fluency=2)
    # a stale row from an earlier, larger run - not in `items`
    verdicts["holdout-99"] = Verdict(item_id="holdout-99", meaning=5, voice=5, fluency=5)

    scores = score_series(items, verdicts)
    assert scores.overall.n == 1, "only this batch's scored items count"
    assert scores.holdout is not None and scores.holdout.meaning == 2.0
