"""Publish the pair dataset to the Hugging Face Hub.

Two jobs at once, and it is worth being explicit that they are different:

1. **Durability.** The pairs cost real money and, until this runs, exist on one
   local disk inside a gitignored directory. The committed manifest proves WHICH
   bytes were generated but cannot reconstruct them. This is the offsite copy.
2. **Publication.** ROADMAP Phase 5's last deliverable - the parallel corpus is
   a standalone contribution, useful outside this project. (No claim is made
   about it being a first; nobody has surveyed what else exists for Slovene.)

The repo stays PRIVATE by default and the card says so, because job 1 needs no
audience and job 2 has an unresolved licensing question: the Cankar side is
public domain (died 1918), but the wikivir transcriptions may carry their own
terms, and that is a decision for the maintainer, not a default. Flipping
private -> public is easy; the reverse is not.

The raw API responses ship alongside the pairs. They are the artifact the money
actually bought - the pairs are a re-derivable parse of them - so a backup that
omitted them would protect the cheap half.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from cankar.core.errors import CankarError
from cankar.pairs.destyle import Pair, PairsManifest
from cankar.pairs.segment import PassagesManifest

log = logging.getLogger("cankar.pairs")

DEFAULT_REPO = "matic-korosec/cankar-parallel"
REPO_TYPE = "dataset"

# Files uploaded, in dependency order. Keys are local paths resolved by the CLI;
# values are the path inside the repo.
CARD_NAME = "README.md"


class Upload(BaseModel):
    """One file to publish. `required` distinguishes the dataset itself from
    provenance that may legitimately be absent on a partial run."""

    local: Path
    remote: str
    required: bool = True


def dataset_card(pairs: PairsManifest, passages: PassagesManifest, repo_id: str) -> str:
    """The card. Written from the manifests so it cannot drift from the data -
    every number here is read, not typed."""
    return f"""---
language:
- sl
license: other
license_name: mixed-see-card
pretty_name: CankarParallel
size_categories:
- 10K<n<100K
task_categories:
- text2text-generation
tags:
- slovene
- style-transfer
- ivan-cankar
- public-domain
---

# CankarParallel - plain Slovene to Ivan Cankar's prose voice

{pairs.n_pairs:,} parallel passages for Slovene literary style transfer. Each row
pairs a passage of Ivan Cankar's prose with a plain modern Slovene rendering of
the same content.

Built for [CankarGPT](https://github.com/Mult1Hunter/cankar-gpt), a from-scratch
Slovene micro-LLM. The pairs train the direction `plain -> cankar`.

## Fields

- `passage_id` - sha256 of the Cankar text, first 16 hex chars
- `cankar` - the original passage, **unmodified public-domain text**
- `plain` - a plain modern Slovene rendering, **generated**
- `title`, `url` - the source work
- `model`, `in_tokens`, `out_tokens` - generation provenance

## How it was made

The Cankar side comes from Wikivir (Slovene Wikisource), segmented into 2-6
sentence paragraph-bounded passages. Only prose genres were used - Cankar's
plays are excluded, because a paragraph in drama is a speaker turn and
segmenting them as prose glues speaker labels into the text.

The plain side was generated with `{pairs.model}` via the Batch API, using one
fixed "plain Slovene register" prompt (sha256 `{pairs.register_sha256[:16]}`)
shared with the inference-time drafting stage, so the styler never meets a
distribution at inference it was not trained on.

Model choice was measured, not assumed. A 50-passage pilot against a cheaper
model found broken Slovene grammar - wrong dual verb forms, gender
disagreement, a reflexive-only verb used transitively - which a char-ngram
style classifier scored identically to the stronger model's output. On a
low-resource language the cheap option corrupts the data invisibly.

{pairs.n_pairs:,} of {pairs.n_pairs + pairs.n_rejected:,} responses passed the
quality filters. Rejections are quarantined, never silently dropped.

## Held-out works are excluded

Passages from the frozen held-out evaluation set (holdout sha256
`{passages.holdout_sha256[:16]}`) were excluded **before** generation. Pairs
built from held-out works would contaminate any evaluation of a model trained
on them, one-way and undetectably.

## Licensing - read before redistributing

Ivan Cankar died in 1918, so **his text is public domain**. The `plain` side is
machine-generated output.

The open question is whether the Wikivir *transcriptions* carry terms of their
own. That is unresolved, which is why this dataset is not yet released for
redistribution. Treat it as reference material until the card says otherwise.

## Reproducing

```
uv run cankar pairs segment
uv run cankar pairs destyle --limit 10000
uv run cankar pairs publish
```

Provenance: corpus sha256 `{pairs.corpus_sha256[:16]}`, passages sha256
`{pairs.passages_sha256[:16]}`, pairs sha256 `{pairs.pairs_sha256[:16]}`,
generated at {pairs.created_at} from git `{pairs.git_sha}`.
"""


# Archaisms the register names explicitly, plus Cankar's own period markers.
# Used only to RANK samples for publication - never to filter the dataset.
_ARCHAISMS = (
    "solnce",
    "vender",
    "duri",
    "život",
    "izpre",
    "zakaj ",
    "dacar",
    "trudoma",
    "hipoma",
    "vekomaj",
    "romal",
    "tolikanj",
    "malodušen",
)

# Enough to show the pattern without turning the doc into a data dump.
N_SAMPLES = 12


def modernization_score(pair: Pair) -> int:
    """How many named archaisms the rewrite removed. Ranking only - a low score
    means an undramatic sample, never a bad pair."""
    old, new = pair.cankar.lower(), pair.plain.lower()
    return sum(1 for a in _ARCHAISMS if a in old and a not in new)


def select_samples(pairs: list[Pair], n: int = N_SAMPLES) -> list[Pair]:
    """The most legible before/after examples, deterministically.

    Committed to `registry/` precisely because `pairs.jsonl` is gitignored: a
    published sample nobody can check against the real data is an invitation to
    quietly improve one. `test_sample_claims.py` gates every excerpt quoted in
    docs/ against this file, so a prettified example fails CI.

    Ties break on passage_id, so re-running picks the same twelve.
    """
    return sorted(pairs, key=lambda p: (-modernization_score(p), p.passage_id))[:n]


def build_uploads(
    pairs_path: Path, raw_path: Path, manifest_path: Path, card: Path
) -> list[Upload]:
    return [
        Upload(local=card, remote=CARD_NAME),
        Upload(local=pairs_path, remote="pairs.jsonl"),
        Upload(local=manifest_path, remote="provenance/pairs.manifest.json"),
        # The artifact the money bought; the pairs are a re-derivable parse of it.
        Upload(local=raw_path, remote="provenance/destyle-raw.jsonl", required=False),
    ]


def check_uploads(uploads: list[Upload]) -> None:
    missing = [u.local for u in uploads if u.required and not u.local.exists()]
    if missing:
        raise CankarError(
            f"cannot publish, missing: {missing} "
            "(run: cankar pairs segment, then cankar pairs destyle)"
        )
