"""Deterministic held-out BPB harness (ADR 0013, 0017).

`bpb_on_checkpoint` scores a trained checkpoint against the frozen held-out set;
the load-bearing piece is the eval BATCHER, which must score every held-out token
exactly once. nanochat's training dataloader (BOS-bestfit) crops ~11-35% of tokens
and packs across document boundaries - non-deterministic and lossy, wrong for
a held-out measurement (architect critique MF-6). This batcher instead scores
every held-out token exactly once: each doc is BOS-prepended and tiled into
non-overlapping T-windows, tail padded with ignore_index (-1) targets.

The model is a duck type (BpbModel); a trained GPT satisfies it directly
(forward(idx, targets, ..., loss_reduction) matches). bpb_on_checkpoint loads a
train checkpoint via its self-describing gptconfig (ADR 0017) and the shared
cankar.model builder - reading the .pt file, never importing the train stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import tiktoken
import torch
from pydantic import BaseModel

from cankar.core.encoding import bos_id as resolve_bos_id  # aliased: `bos_id` is a param name below
from cankar.core.encoding import load_encoding
from cankar.core.errors import CankarError
from cankar.core.holdout import load_holdout
from cankar.core.manifest import sha256_of
from cankar.core.reports import generated_marker, write_report
from cankar.evals.holdout import iter_holdout_texts
from cankar.evals.vendored_bpb import BpbModel, evaluate_bpb
from cankar.model.build import build_gpt
from cankar.model.gpt import GPTConfig

log = logging.getLogger("cankar.evals")

IGNORE_INDEX = -1  # nanochat masks y < 0 out of the metric (loss_eval.py)

EvalBatch = tuple[torch.Tensor, torch.Tensor]  # (x, y), each (1, T)


def build_eval_batches(
    texts: list[str], enc: tiktoken.Encoding, seq_len: int, bos_id: int
) -> list[EvalBatch]:
    """One BOS-prepended stream per doc, tiled into non-overlapping (1, T)
    windows so every target token is scored exactly once (MF-6). The final
    window of each doc is padded: x with token 0, y with IGNORE_INDEX.

    bos_id is passed in (not imported from the tokenizer stage) so evals stays
    an independent sibling: the Phase 3 caller that owns the tokenizer resolves
    it via enc.encode_single_token('<|bos|>')."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    bos = bos_id
    batches: list[EvalBatch] = []
    for text in texts:
        toks = [bos, *enc.encode_ordinary(text)]
        for i in range(0, len(toks) - 1, seq_len):
            xw = toks[i : i + seq_len]
            yw = toks[i + 1 : i + 1 + seq_len]
            pad = seq_len - len(xw)
            xw = xw + [0] * pad
            yw = yw + [IGNORE_INDEX] * (seq_len - len(yw))
            batches.append(
                (
                    torch.tensor(xw, dtype=torch.long).unsqueeze(0),
                    torch.tensor(yw, dtype=torch.long).unsqueeze(0),
                )
            )
    return batches


def load_token_bytes(path: Path) -> torch.Tensor:
    """The int32 byte-length-per-token tensor the selected tokenizer emitted
    (cankar tokenizer train). BPB indexes target tokens into it."""
    if not path.exists():
        raise CankarError(f"token_bytes.pt missing: {path} (run: cankar tokenizer train)")
    return torch.load(path, map_location="cpu")


def holdout_bpb(
    model: BpbModel,
    texts: list[str],
    enc: tiktoken.Encoding,
    token_bytes: torch.Tensor,
    seq_len: int,
    bos_id: int,
) -> float:
    """Held-out BPB for a model over the frozen held-out texts. Batches and
    token_bytes are moved to the model's device (a no-op on CPU)."""
    device = model.get_device()
    batches = build_eval_batches(texts, enc, seq_len, bos_id)
    if not batches:
        return float("inf")
    batches = [(x.to(device), y.to(device)) for x, y in batches]
    return evaluate_bpb(model, batches, len(batches), token_bytes.to(device))


@dataclass
class BpbResult:
    """Held-out BPB for one checkpoint (ADR 0017)."""

    bpb: float
    n_works: int
    step: int  # the checkpoint's training step, for tracking progress
    n_params: int  # also a public claim ("26.3M") - measured here, not asserted
    tokenizer: str  # per checkpoint, not per run: a future checkpoint may differ


def bpb_on_checkpoint(
    ckpt_path: Path,
    corpus_path: Path,
    holdout_path: Path,
    tokenizer_base_dir: Path,
    device: str,
) -> BpbResult:
    """Load a trained GPT checkpoint and score the frozen held-out set (invariant
    #2). The checkpoint is read as a file and rebuilt from its self-describing
    gptconfig via the shared cankar.model builder - evals never imports the train
    stage (ADR 0017). iter_holdout_texts re-verifies each work's content sha, so
    a drifted corpus fails loud rather than scoring the wrong bytes."""
    # map to CPU: only model/gptconfig/config/step are read; build_gpt + load_state_dict
    # place the model on `device` (avoids pulling the fp32 optimizer state onto the GPU).
    state: dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "gptconfig" not in state:
        raise CankarError(f"{ckpt_path} predates the self-describing checkpoint (ADR 0017)")
    model = build_gpt(GPTConfig(**state["gptconfig"]), device)
    model.load_state_dict(state["model"])
    model.eval()

    tokenizer_name = state["config"]["tokenizer"]
    enc = load_encoding(tokenizer_name)
    token_bytes = load_token_bytes(tokenizer_base_dir / tokenizer_name / "token_bytes.pt")
    manifest = load_holdout(holdout_path)
    texts = [text for _title, text in iter_holdout_texts(corpus_path, manifest)]
    bpb = holdout_bpb(
        model, texts, enc, token_bytes, state["gptconfig"]["sequence_len"], resolve_bos_id(enc)
    )
    return BpbResult(
        bpb=bpb,
        n_works=len(texts),
        step=int(state["step"]),
        n_params=sum(p.numel() for p in model.parameters()),
        tokenizer=tokenizer_name,
    )


class CanonicalCheckpoint(StrEnum):
    """The checkpoints public quality claims are made about (ADR 0008: closed
    sets are enums). Values are the `checkpoints/<value>.pt` stems.

    Deliberately NOT a glob over `checkpoints/`: that directory also holds
    experiment artifacts nothing claims - `nanocankar.pt` is one today, tracked
    in no doc, config or registry entry. Globbing would publish a BPB for a model
    with no story attached to it.
    """

    TINYCANKAR = "tinycankar"
    BASE = "base"
    CANKAR_V1 = "cankar-v1"


class CheckpointBpb(BaseModel):
    """One scored checkpoint. `sha256` is what makes the row auditable - the .pt
    is gitignored, so the hash is the only durable statement of WHICH weights
    produced this number."""

    name: str
    sha256: str
    step: int
    n_params: int
    tokenizer: str
    bpb: float
    n_works: int


class BpbManifest(BaseModel):
    """Frozen held-out BPB for the canonical checkpoints (ADR 0017).

    Modeled on `style.json`: generated once, committed, never hand-edited. Before
    this existed the headline number lived only in a log line that scrolled away -
    the README badge, docs/cankar-v1.md and the landing page all cited a figure
    nothing in the repo could reproduce or contradict.
    """

    schema_version: int = 1
    corpus_sha256: str  # holdout texts are read from this corpus; BPB is only valid against it
    git_sha: str
    created_at: str
    device: str  # cuda and cpu differ in the last float places
    lib_versions: dict[str, str]
    checkpoints: list[CheckpointBpb]


def score_canonical(
    checkpoints_dir: Path,
    corpus_path: Path,
    holdout_path: Path,
    tokenizer_base_dir: Path,
    device: str,
) -> list[CheckpointBpb]:
    """Score every canonical checkpoint, in enum order (the progression order the
    report and the docs table both present).

    A missing checkpoint raises rather than being skipped: a silently short table
    still reads as "the three-model progression" and would understate the claim
    it exists to support.
    """
    rows: list[CheckpointBpb] = []
    for ckpt in CanonicalCheckpoint:
        path = checkpoints_dir / f"{ckpt.value}.pt"
        if not path.exists():
            raise CankarError(
                f"canonical checkpoint missing: {path}. All of "
                f"{[c.value for c in CanonicalCheckpoint]} must be present - a partial "
                "table would misreport the progression (pull from HF/R2, or retrain)."
            )
        log.info("scoring %s ...", path.name)
        r = bpb_on_checkpoint(path, corpus_path, holdout_path, tokenizer_base_dir, device)
        rows.append(
            CheckpointBpb(
                name=ckpt.value,
                sha256=sha256_of(path),
                step=r.step,
                n_params=r.n_params,
                tokenizer=r.tokenizer,
                bpb=round(r.bpb, 4),
                n_works=r.n_works,
            )
        )
    return rows


def write_bpb_report(out: Path, manifest: BpbManifest) -> Path:
    """Human-readable face of bpb.json - the progression table the docs cite."""
    m = manifest
    L: list[str] = [
        generated_marker("cankar evals bpb-freeze", snapshot=True),
        "",
        "# Held-out BPB - canonical checkpoints (ADR 0017)",
        "",
        f"Corpus sha256 `{m.corpus_sha256}`.",
        f"Scored on `{m.device}` at {m.created_at} (git `{m.git_sha}`).",
        "",
        "Bits per byte over the frozen held-out Cankar set (ADR 0013), every held-out",
        "token scored exactly once. Lower is better. **These are the numbers the README",
        "badge, `docs/cankar-v1.md` and the landing page cite** -",
        "`tests/evals/test_bpb_claims.py` fails if any of them drifts from this file.",
        "",
        "| checkpoint | params | tokenizer | step | held-out BPB |",
        "|---|---:|---|---:|---:|",
    ]
    for c in m.checkpoints:
        L.append(
            f"| `{c.name}` | {c.n_params / 1e6:.1f}M | `{c.tokenizer}` | "
            f"{c.step:,} | **{c.bpb:.4f}** |"
        )
    # every row reads the same frozen holdout, so a split here means one row was
    # scored against a different set and the comparison is meaningless - say so.
    works = sorted({c.n_works for c in m.checkpoints})
    scope = f"{works[0]} held-out works" if len(works) == 1 else f"DIFFERING work counts {works}"
    L += [
        "",
        f"All rows scored over the same {scope}.",
        "",
        "## Reproducing",
        "",
        "```",
        "uv run cankar evals bpb-freeze",
        "```",
        "",
        "Then `git diff` this file and `registry/evals/bpb.json`. The checkpoints are",
        "gitignored, so the per-row `sha256` in the manifest is the durable statement of",
        "which weights produced each number.",
        "",
        "## Library versions",
        "",
        *[f"- `{k}`: {v}" for k, v in sorted(m.lib_versions.items())],
    ]
    write_report(out, L)
    return out
