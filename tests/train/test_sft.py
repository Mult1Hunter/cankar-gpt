"""SFT data gates - the loss mask is the load-bearing property.

If the mask is wrong the run still trains, the loss still falls, and the model
learns the wrong task: continuing plain Slovene instead of restyling it. Nothing
downstream would surface that except a human reading samples, so it gets tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cankar.core.errors import CankarError
from cankar.tokenizer import train as tok_train
from cankar.train.sft import (
    ASSISTANT_END,
    ASSISTANT_START,
    IGNORE_INDEX,
    USER_END,
    USER_START,
    build_example,
    collate,
    iter_batches,
    load_pairs,
    special_ids,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "tokenizer" / "mini-corpus.jsonl"


@pytest.fixture(scope="module")
def enc():
    """A tiny REAL encoding trained from the committed fixture, carrying the
    same chat specials as v8192.

    Deliberately not `load_encoding("v8192")`: that artifact lives in gitignored
    data/, so a module-level load aborted collection on CI and took the whole
    336-test suite down with it - a test that cannot run in CI verifies nothing
    (matches tests/train/test_training.py)."""
    return tok_train.train_encoding(FIXTURE, 300)


@pytest.fixture(scope="module")
def sp(enc):
    return special_ids(enc)


PLAIN = "Vrnil se je domov in legel."
CANKAR = "Vrnil se je na dom svoj in je legel, trudoma."


def _pairs_file(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


def test_the_tokenizer_carries_every_chat_special(enc, sp) -> None:
    """The format depends on all four. A missing one would encode as ordinary
    text and the prompt/target boundary would silently vanish.

    Asserted structurally, not against v8192's literal ids (8184-8187): the ids
    are vocabulary-size dependent, and pinning them here would make this test
    pass only on the one artifact CI does not have."""
    assert set(sp) == {USER_START, USER_END, ASSISTANT_START, ASSISTANT_END}
    assert len(set(sp.values())) == 4  # distinct
    assert all(i >= enc.n_vocab - len(enc._special_tokens) for i in sp.values())


def test_example_layout_is_bos_prompt_then_target(enc, sp) -> None:
    ex = build_example(PLAIN, CANKAR, enc, sp)
    assert ex.tokens[1] == sp[USER_START]
    assert ex.tokens[ex.n_prompt - 2] == sp[USER_END]
    assert ex.tokens[ex.n_prompt - 1] == sp[ASSISTANT_START]
    assert ex.tokens[-1] == sp[ASSISTANT_END]
    assert ex.n_prompt + ex.n_target == len(ex.tokens)


def test_target_span_includes_the_stop_token(enc, sp) -> None:
    """A target that never contains <|assistant_end|> teaches the model to run
    on past the passage - it would have no example of stopping."""
    ex = build_example(PLAIN, CANKAR, enc, sp)
    x, y = collate([ex], pad_id=0)
    scored = [t for t in y[0].tolist() if t != IGNORE_INDEX]
    assert scored[-1] == sp[ASSISTANT_END]


def test_loss_is_masked_over_the_entire_prompt(enc, sp) -> None:
    """The prompt is 49% of all tokens. Scoring it trains a 26M model to
    generate plain Slovene, which is not the task."""
    ex = build_example(PLAIN, CANKAR, enc, sp)
    _, y = collate([ex], pad_id=0)
    ys = y[0].tolist()
    assert all(t == IGNORE_INDEX for t in ys[: ex.n_prompt - 1])
    assert all(t != IGNORE_INDEX for t in ys[ex.n_prompt - 1 :])
    assert sum(t != IGNORE_INDEX for t in ys) == ex.n_target


def test_scored_targets_are_the_cankar_side_shifted_by_one(enc, sp) -> None:
    """Guards the off-by-one in the (x, y) shift: y[i] must be the token the
    model should emit AFTER seeing x[i]."""
    ex = build_example(PLAIN, CANKAR, enc, sp)
    x, y = collate([ex], pad_id=0)
    for i, t in enumerate(y[0].tolist()):
        if t != IGNORE_INDEX:
            assert t == ex.tokens[i + 1]


def test_padding_is_never_scored(enc, sp) -> None:
    short = build_example("Kratko.", "Kratko in staro.", enc, sp)
    long = build_example(PLAIN * 4, CANKAR * 4, enc, sp)
    x, y = collate([short, long], pad_id=0)
    assert x.shape == y.shape
    assert sum(t != IGNORE_INDEX for t in y[0].tolist()) == short.n_target
    assert sum(t != IGNORE_INDEX for t in y[1].tolist()) == long.n_target


def test_batch_pads_to_its_own_longest_not_to_seq_len(enc, sp) -> None:
    """Median length is 194 against a 512 window; padding globally would spend
    ~3x the compute on padding."""
    a = build_example("A.", "A staro.", enc, sp)
    b = build_example("B.", "B staro.", enc, sp)
    x, _ = collate([a, b], pad_id=0)
    assert x.shape[1] == max(len(a.tokens), len(b.tokens)) - 1


def test_over_length_pairs_are_dropped_and_counted(tmp_path: Path, enc, sp) -> None:
    """Never truncated: a cut target teaches the model to stop mid-sentence."""
    rows = [{"plain": PLAIN, "cankar": CANKAR}, {"plain": PLAIN * 50, "cankar": CANKAR * 50}]
    data = load_pairs(_pairs_file(tmp_path, rows), enc, seq_len=128)
    assert data.n_pairs == 2
    assert len(data.examples) == 1
    assert data.n_dropped_too_long == 1


def test_every_kept_example_fits_the_window(tmp_path: Path, enc, sp) -> None:
    rows = [{"plain": PLAIN * i, "cankar": CANKAR * i} for i in range(1, 8)]
    data = load_pairs(_pairs_file(tmp_path, rows), enc, seq_len=256)
    assert data.examples
    assert all(len(e.tokens) <= 257 for e in data.examples)


def test_batching_is_deterministic_and_covers_every_example(tmp_path: Path, enc, sp) -> None:
    """Order is a pure function of (seed, epoch) so a resumed run replays it."""
    rows = [{"plain": f"{PLAIN} {i}", "cankar": f"{CANKAR} {i}"} for i in range(20)]
    data = load_pairs(_pairs_file(tmp_path, rows), enc, seq_len=256)
    first = [x.shape for x, _ in iter_batches(data, 4, seed=7, epoch=0)]
    again = [x.shape for x, _ in iter_batches(data, 4, seed=7, epoch=0)]
    other = [x.shape for x, _ in iter_batches(data, 4, seed=7, epoch=1)]
    assert first == again
    assert sum(s[0] for s in first) == len(data.examples)
    assert other != [] and sum(s[0] for s in other) == len(data.examples)


def test_a_tokenizer_without_chat_specials_fails_loud(enc, sp) -> None:
    class Bare:
        _special_tokens: dict[str, int] = {}

    with pytest.raises(CankarError, match="chat specials"):
        special_ids(Bare())  # type: ignore[arg-type]
