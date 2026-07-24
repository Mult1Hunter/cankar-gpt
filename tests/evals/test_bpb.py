"""BPB harness: deterministic batcher + vendored metric (ADR 0013, MF-5/MF-6)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from cankar.evals import bpb
from cankar.evals.bpb import IGNORE_INDEX
from cankar.tokenizer import train
from cankar.tokenizer.vendored import BOS_TOKEN

TOK = Path(__file__).parent.parent / "fixtures" / "tokenizer" / "mini-corpus.jsonl"


@pytest.fixture(scope="module")
def enc():
    return train.train_encoding(TOK, 300)


@pytest.fixture(scope="module")
def bos(enc):
    return enc.encode_single_token(BOS_TOKEN)


class StubModel:
    """Duck-typed BpbModel returning a constant per-position loss."""

    def __init__(self, loss_value: float = 2.0):
        self.loss_value = loss_value

    def __call__(self, x: torch.Tensor, y: torch.Tensor, loss_reduction: str) -> torch.Tensor:
        return torch.full(x.shape, self.loss_value, dtype=torch.float32)

    def get_device(self) -> str:
        return "cpu"


def test_batcher_scores_each_real_token_once(enc, bos) -> None:
    """MF-6: every ordinary token is a target exactly once; BOS is input-only,
    never a scored target; padding targets are IGNORE_INDEX."""
    text = "Solnce je sijalo nad klancem in mati je gledala v dolino."
    ordinary = enc.encode_ordinary(text)
    batches = bpb.build_eval_batches([text], enc, seq_len=8, bos_id=bos)
    ys = torch.cat([y.view(-1) for _, y in batches])
    # the scored targets are EXACTLY the ordinary tokens, in order, once each
    assert ys[ys >= 0].tolist() == ordinary
    assert bos not in ys[ys >= 0].tolist()  # BOS is never a target


def test_batcher_windows_cover_without_overlap(enc, bos) -> None:
    """Reconstruct the target stream from the y-windows: it must equal the
    BOS-prepended token stream minus its first token, once, in order."""
    text = "".join(f"beseda{i} " for i in range(40))
    batches = bpb.build_eval_batches([text], enc, seq_len=6, bos_id=bos)
    toks = [bos, *enc.encode_ordinary(text)]
    recovered = []
    for _, y in batches:
        recovered.extend(t for t in y.view(-1).tolist() if t != IGNORE_INDEX)
    assert recovered == toks[1:]  # covers every target once, no overlap, no gap


def test_bpb_math_matches_hand_computation(enc, bos) -> None:
    """Vendored evaluate_bpb: constant loss L over the valid targets divided by
    ln2 * summed target-token bytes."""
    text = "Tiha noč, sveta noč, vse že spi."
    token_bytes = train.token_bytes_tensor(enc)
    batches = bpb.build_eval_batches([text], enc, seq_len=16, bos_id=bos)
    model = StubModel(loss_value=1.5)

    ys = torch.cat([y.view(-1) for _, y in batches])
    valid = ys[ys >= 0]
    exp_bytes = int(token_bytes[valid].sum())
    exp_nats = 1.5 * len(valid)
    expected = exp_nats / (math.log(2) * exp_bytes)

    got = bpb.holdout_bpb(model, [text], enc, token_bytes, seq_len=16, bos_id=bos)
    assert got == pytest.approx(expected, rel=1e-6)


def test_empty_holdout_is_inf(enc, bos) -> None:
    token_bytes = train.token_bytes_tensor(enc)
    assert bpb.holdout_bpb(StubModel(), [], enc, token_bytes, seq_len=8, bos_id=bos) == float("inf")


def test_vendored_bpb_golden() -> None:
    """CI-pinned golden (no sibling needed): a hand-built batch with an ignored
    target and a special (0-byte) target, exercising both mask paths. Guards
    the vendored metric against drift in CI where the sibling test is skipped."""
    from cankar.evals.vendored_bpb import evaluate_bpb

    token_bytes = torch.tensor([0, 1, 2, 3], dtype=torch.int32)  # id 0 is special
    x = torch.tensor([[1, 2, 3, 1]])
    y = torch.tensor([[2, 3, IGNORE_INDEX, 0]])  # one ignored, one special target

    class ConstModel:
        def __call__(self, x, y, loss_reduction):
            return torch.full(x.shape, 2.0, dtype=torch.float32)

        def get_device(self):
            return "cpu"

    # valid, byte-carrying targets: id2 (2 bytes) and id3 (3 bytes); id0 special
    # (0 bytes) and -1 ignored both drop out. nats = 2.0*2, bytes = 2+3 = 5.
    expected = (2.0 * 2) / (math.log(2) * 5)
    assert evaluate_bpb(ConstModel(), [(x, y)], 1, token_bytes) == pytest.approx(expected)


def test_bpb_on_checkpoint(enc, tmp_path, monkeypatch) -> None:
    """Integration (ADR 0017): a self-describing checkpoint -> held-out BPB, with
    no import of the train stage. Builds a tiny GPT, a one-work holdout keyed to a
    tiny corpus (content-sha verified), and scores it."""
    import dataclasses
    import hashlib
    import json

    from cankar.core.holdout import HoldoutManifest, HoldoutParams, HoldoutWork
    from cankar.model.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    gcfg = GPTConfig(
        sequence_len=32,
        vocab_size=enc.n_vocab,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    model = GPT(gcfg)
    model.init_weights()
    ckpt = tmp_path / "m.pt"
    torch.save(
        {
            "step": 5,
            "config": {"tokenizer": "vtest"},
            "gptconfig": dataclasses.asdict(gcfg),
            "model": model.state_dict(),
            "optimizer": {},
            "torch_rng": torch.get_rng_state(),
        },
        ckpt,
    )

    text = "Solnce je sijalo nad klancem in mati je gledala v dolino, tiho in otožno."
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "title": "T",
                "url": "u/1",
                "text": text,
                "n_chars": len(text),
                "source": "wikivir",
                "author": "Ivan Cankar",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = HoldoutManifest(
        corpus_sha256="x",
        tokenizer_name="vtest",
        params=HoldoutParams(),
        cankar_total_tokens=100,
        holdout_tokens=10,
        holdout_fraction=0.1,
        git_sha="x",
        created_at="t",
        works=[
            HoldoutWork(
                url="u/1",
                title="T",
                n_chars=len(text),
                n_tokens=len(enc.encode_ordinary(text)),
                content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                max_containment_elsewhere=0.0,
            )
        ],
    )
    hpath = tmp_path / "holdout.json"
    hpath.write_text(manifest.model_dump_json(), encoding="utf-8")

    monkeypatch.setattr("cankar.evals.bpb.load_encoding", lambda name: enc)
    monkeypatch.setattr(
        "cankar.evals.bpb.load_token_bytes", lambda p: train.token_bytes_tensor(enc)
    )

    result = bpb.bpb_on_checkpoint(ckpt, corpus, hpath, tmp_path, "cpu")
    assert result.n_works == 1 and result.step == 5
    assert math.isfinite(result.bpb) and result.bpb > 0


def test_bpb_on_checkpoint_rejects_old_format(tmp_path) -> None:
    from cankar.core.errors import CankarError

    ckpt = tmp_path / "old.pt"
    torch.save({"step": 1, "model": {}}, ckpt)  # no gptconfig
    with pytest.raises(CankarError, match="self-describing"):
        bpb.bpb_on_checkpoint(ckpt, tmp_path / "c.jsonl", tmp_path / "h.json", tmp_path, "cpu")
