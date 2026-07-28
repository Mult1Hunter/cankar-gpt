"""Publish the pair dataset to the Hugging Face Hub.

Two jobs at once, and it is worth being explicit that they are different:

1. **Durability.** The pairs cost real money and, until this runs, exist on one
   local disk inside a gitignored directory. The committed manifest proves WHICH
   bytes were generated but cannot reconstruct them. This is the offsite copy.
2. **Publication.** ROADMAP Phase 5's last deliverable - the parallel corpus is
   a standalone contribution, useful outside this project. (No claim is made
   about it being a first; nobody has surveyed what else exists for Slovene.)

Licensing is settled and the reasoning is worth keeping, because the answer is
not the obvious one. Cankar died in 1918, so the WORK is public domain. But the
text was not taken from 1900s printings - it was transcribed by volunteers on
Wikivir, and Wikimedia applies CC BY-SA to contributions. Whether a faithful
transcription attracts rights of its own is genuinely unsettled: EU copyright
needs "the author's own intellectual creation", which retyping is not, while the
EU sui generis DATABASE right protects substantial investment regardless of
originality - and a decade of volunteer transcription is exactly that.

Resolved as CC BY-SA 4.0 with attribution, on asymmetry rather than certainty:
if the transcriptions do carry rights we have complied, and if they do not we
gave away nothing that matters. Asserting public domain unilaterally would
require being RIGHT about the unsettled question. The project also already
carries CC BY-SA exposure - the base model was pretrained on 65.3M words of
Slovenian Wikipedia - so this adds no new category of obligation.

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
license: cc-by-sa-4.0
pretty_name: CankarParallel
size_categories:
- 10K<n<100K
task_categories:
- text-generation
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

## Licensing and attribution

**CC BY-SA 4.0.** Attribute **[Slovene Wikisource (Wikivir)
contributors](https://sl.wikisource.org)**, whose volunteer transcriptions the
`cankar` side reproduces, and this dataset.

Ivan Cankar died in 1918, so the underlying work is public domain. The
transcriptions are a separate layer: whether a faithful transcription of a
public-domain text attracts rights of its own is unsettled in EU law - it lacks
the originality copyright requires, but the sui generis database right protects
substantial investment regardless. CC BY-SA is applied deliberately rather than
because the question was answered: it is what Wikivir asks for, it costs nothing
here, and the alternative required being right about an open question.

Share-alike applies to redistribution of the dataset. Whether it reaches model
weights trained on it is unsettled everywhere and is not a position this card
takes.

## Reproducing

```
uv run cankar pairs segment
uv run cankar pairs destyle --parse-only   # re-derives pairs from the raw log
uv run cankar pairs publish
```

The middle step re-parses committed raw responses and costs nothing. Generating
them from scratch is `destyle --limit N`, which submits paid batches.

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


# What every published copy must state. Not a style preference: redistributing
# volunteer transcriptions without naming the licence or the contributors is the
# actual hazard now that the dataset is public, and publication is one-way in
# practice - forks and caches survive un-publishing.
REQUIRED_CARD_TERMS = ("license: cc-by-sa-4.0", "Wikisource", "CC BY-SA 4.0")


def require_licensed(card: str) -> None:
    """Refuse to publish a card that does not state its terms.

    Replaces an earlier `require_private`, which enforced the licensing question
    being OPEN. That check would now block the deliberate answer while leaving
    the real risk - publishing without stating terms - ungated. When a gate's
    premise is resolved, retarget it rather than delete it.
    """
    missing = [t for t in REQUIRED_CARD_TERMS if t not in card]
    if missing:
        raise CankarError(
            f"dataset card is missing required licensing terms: {missing}. "
            "Redistributing Wikivir transcriptions requires naming the licence "
            "and the contributors."
        )


def check_uploads(uploads: list[Upload]) -> None:
    missing = [u.local for u in uploads if u.required and not u.local.exists()]
    if missing:
        raise CankarError(
            f"cannot publish, missing: {missing} "
            "(run: cankar pairs segment, then cankar pairs destyle)"
        )
