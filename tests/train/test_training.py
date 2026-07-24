"""Train-stage smoke + invariants (ADR 0016). Tiny synthetic corpus - the real
2.77M-token run is a GPU job; here we prove the mechanics on CPU."""

from __future__ import annotations

import json

import pytest
import torch

from cankar.core.encoding import load_encoding
from cankar.core.errors import CankarError
from cankar.core.holdout import HoldoutManifest, HoldoutParams
from cankar.train.checkpoint import load_checkpoint
from cankar.train.config import TrainConfig
from cankar.train.data import TokenizedCorpus, cankar_chunk_texts, iter_batches
from cankar.train.loop import build_model, lr_multiplier, train
from cankar.train.sample import sample_from_checkpoint

TOK = "v8192"


def _write_holdout(path, corpus_sha: str) -> None:
    m = HoldoutManifest(
        corpus_sha256=corpus_sha,
        tokenizer_name=TOK,
        params=HoldoutParams(),
        cankar_total_tokens=1,
        holdout_tokens=0,
        holdout_fraction=0.0,
        git_sha="x",
        created_at="t",
        works=[],
    )
    path.write_text(m.model_dump_json(), encoding="utf-8")


def test_cankar_chunk_texts_rejects_corpus_skew(tmp_path) -> None:
    """Invariant #2 (design-review): training must refuse chunks built on a
    different corpus revision than the holdout - that can silently retain
    excerpts of held-out works."""
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps({"author": "Ivan Cankar", "url": "u/1", "text": "besedilo", "n_tokens": 1})
        + "\n"
    )
    holdout = tmp_path / "holdout.json"
    _write_holdout(holdout, "AAA")
    manifest = tmp_path / "chunks.manifest.json"
    manifest.write_text(json.dumps({"corpus_sha256": "BBB"}))  # skew
    with pytest.raises(CankarError, match="corpus revision skew"):
        cankar_chunk_texts(chunks, holdout, manifest)
    manifest.write_text(json.dumps({"corpus_sha256": "AAA"}))  # matching -> works
    assert cankar_chunk_texts(chunks, holdout, manifest) == ["besedilo"]


def _tiny_cfg(**over: object) -> TrainConfig:
    base = dict(
        name="smoke",
        tokenizer=TOK,
        n_layer=2,
        n_embd=64,
        n_head=2,
        n_kv_head=2,
        seq_len=16,
        batch_size=2,
        max_steps=4,
        warmup_steps=1,
        log_every=1,
        checkpoint_every=1000,
        sample_every=0,
        sample_max_tokens=4,
    )
    base.update(over)
    return TrainConfig(**base)


def test_lr_multiplier_warmup_then_cosine() -> None:
    cfg = TrainConfig(warmup_steps=10, max_steps=100, min_lr_frac=0.1)
    assert lr_multiplier(0, cfg) < lr_multiplier(5, cfg)  # warming up
    assert lr_multiplier(9, cfg) == 1.0  # peak at end of warmup
    assert lr_multiplier(99, cfg) < lr_multiplier(50, cfg)  # decaying
    assert lr_multiplier(99, cfg) >= cfg.min_lr_frac - 1e-9  # floors at min_lr_frac


def test_iter_batches_deterministic_and_resumable() -> None:
    enc = load_encoding(TOK)
    corpus = TokenizedCorpus.build([f"beseda{i} in nekaj drugega tukaj" for i in range(30)], enc)
    kw = dict(batch_size=2, seq_len=8, seed=1, device="cpu")
    a = [next(iter_batches(corpus, **kw)) for _ in range(1)][0]
    b = next(iter_batches(corpus, **kw))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])  # deterministic
    # resume: skipping 3 batches equals taking the 4th from the start
    full = iter_batches(corpus, **kw)
    for _ in range(3):
        next(full)
    fourth = next(full)
    skipped = next(iter_batches(corpus, **kw, start_step=3))
    assert torch.equal(fourth[0], skipped[0]) and torch.equal(fourth[1], skipped[1])


def test_model_learns_reduces_loss_on_cpu() -> None:
    """Forward+backward+AdamW must drive loss down - a tiny model memorizes a
    fixed batch. Proves the training mechanics, independent of the loop plumbing."""
    from cankar.model.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    m = GPT(
        GPTConfig(
            sequence_len=16,
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            window_pattern="L",
        )
    )
    m.init_weights()
    m.train()
    opt = m.setup_optimizer(matrix_lr=0.003)
    x = torch.randint(0, 100, (4, 16))
    y = x.roll(-1, dims=1)
    first = m(x, targets=y).item()
    loss = torch.tensor(first)
    for _ in range(30):
        loss = m(x, targets=y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert loss.item() < first * 0.5


def test_train_smoke_checkpoint_resume_sample(tmp_path, monkeypatch) -> None:
    texts = [f"To je stavek {i} o življenju in mestu ob tihi reki spomladi." for i in range(40)]
    monkeypatch.setattr("cankar.train.loop.cankar_chunk_texts", lambda *a: texts)

    ck = train(_tiny_cfg(max_steps=4), tmp_path, "cpu")
    assert ck.exists()
    assert load_checkpoint(ck, "cpu")["step"] == 4

    # resume: two more steps, exact continuation (same model shape)
    train(_tiny_cfg(max_steps=6), tmp_path, "cpu", resume=True)
    assert load_checkpoint(ck, "cpu")["step"] == 6

    samples = sample_from_checkpoint(ck, "cpu", prompt="Bilo", max_tokens=5, n_samples=2)
    assert len(samples) == 2 and all(isinstance(s, str) for s in samples)


def test_build_model_wires_tokenizer_vocab() -> None:
    enc = load_encoding(TOK)
    m = build_model(_tiny_cfg(), enc, "cpu")
    assert m.config.vocab_size == enc.n_vocab  # vocab from the frozen tokenizer, not hardcoded
