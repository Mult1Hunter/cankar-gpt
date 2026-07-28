"""Fresh plain-Slovene drafts - the other half of the Phase 6 headline number.

Held-out pairs measure how well the styler restyles Cankar's OWN de-styled text.
That is the easier question. The one that decides whether this ships is what
happens to prose the styler has never seen the subject of, arriving the way
inference will actually deliver it.

Two arms, and the gap between them is itself the result:

- **REGISTER** - drafted under `PLAIN_REGISTER`, the exact inference path
  (invariant #1: the same register defines de-styling and drafting). Only the
  TOPIC differs from training, deliberately: every de-styled passage is about
  poverty, Ljubljana, longing or 1900s village life, so a styler that works only
  on those would score perfectly and be useless in production.
- **WILD** - contemporary Slovene from the Wikipedia slice, written under no
  register of ours. A stress test of how much the styler DEPENDS on the register
  rather than merely preferring it. Caveat stated rather than hidden: Wikipedia
  was in the base model's pretraining, so this is out-of-register, not unseen.

If REGISTER scores well and WILD collapses, that is the honest finding - it
works when the drafter cooperates - and it belongs in the write-up, not in a
footnote.
"""

from __future__ import annotations

import logging
import random
import tomllib
import unicodedata
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cankar.core.errors import CankarError
from cankar.core.jsonl import iter_jsonl_docs
from cankar.core.register import PLAIN_REGISTER

log = logging.getLogger("cankar.pairs")

DRAFTS_VERSION = 1
WIKIPEDIA_SOURCE = "wikipedia"

# Matches the passage band the styler was trained on (SegmentParams), so length
# is not a confound when comparing arms.
MIN_CHARS = 120
MAX_CHARS = 800

# 15 wild against 35 register: enough to detect a collapse, small enough that
# the primary arm stays the headline. Seeded so the sample is reproducible.
N_WILD = 15
WILD_SEED = 20260728


class DraftArm(StrEnum):
    """Which distribution a draft came from. Closed set, so a StrEnum."""

    REGISTER = "register"  # drafted under PLAIN_REGISTER - the inference path
    WILD = "wild"  # contemporary Slovene written under no register of ours


# The drafting prompt. Carries PLAIN_REGISTER verbatim - this is consumer (b) of
# invariant #1, the direction that did not exist until now. It deliberately does
# NOT carry SOURCE_FIDELITY: there is no source to be faithful to when writing
# from a topic, which is exactly why that block is a separate constant.
DRAFT_SYSTEM = unicodedata.normalize(
    "NFC",
    f"""You write short passages of plain modern Slovene.

{PLAIN_REGISTER}

Write 3-5 sentences on the topic the user gives you. Ordinary, factual, everyday
prose - the kind a competent modern writer produces without reaching for effect.

Output ONLY the Slovene passage. No preamble, no title, no commentary.""",
)


class Draft(BaseModel):
    """One eval input. `arm` is what makes the comparison meaningful."""

    draft_id: str
    arm: DraftArm
    topic: str  # the prompt topic, or the source title for WILD
    text: str
    n_chars: int


def load_topics(path: Path) -> list[str]:
    if not path.exists():
        raise CankarError(f"draft topics not found: {path}")
    topics = tomllib.loads(path.read_text(encoding="utf-8")).get("topics", [])
    if not topics:
        raise CankarError(f"{path} declares no topics")
    return [str(t) for t in topics]


def build_request(topic: str, index: int, model: str) -> dict[str, Any]:
    """One Batch API request. custom_id encodes the arm so results cannot be
    mixed up with a de-styling batch sharing the same raw log."""
    return {
        "custom_id": f"{DraftArm.REGISTER.value}-{index:03d}",
        "params": {
            "model": model,
            "max_tokens": 512,
            "thinking": {"type": "disabled"},
            "system": DRAFT_SYSTEM,
            "messages": [{"role": "user", "content": topic}],
        },
    }


def sample_wild(corpus_path: Path, n: int = N_WILD, seed: int = WILD_SEED) -> list[Draft]:
    """Contemporary Slovene paragraphs from the Wikipedia slice.

    Reservoir-free: collects every eligible paragraph then samples with a fixed
    seed, so the set is reproducible from the corpus sha alone. One paragraph per
    article, so 15 drafts are 15 different subjects rather than one article's.
    """
    rng = random.Random(seed)
    pool: list[tuple[str, str]] = []
    for doc in iter_jsonl_docs(corpus_path, "run: cankar corpus merge"):
        if doc.get("source") != WIKIPEDIA_SOURCE:
            continue
        for para in doc["text"].split("\n\n"):
            flat = " ".join(para.split())
            if MIN_CHARS <= len(flat) <= MAX_CHARS:
                pool.append((doc["title"], flat))
                break
    if len(pool) < n:
        raise CankarError(f"only {len(pool)} eligible Wikipedia paragraphs, need {n}")
    return [
        Draft(
            draft_id=f"{DraftArm.WILD.value}-{i:03d}",
            arm=DraftArm.WILD,
            topic=title,
            text=text,
            n_chars=len(text),
        )
        for i, (title, text) in enumerate(rng.sample(pool, n))
    ]


def parse_register_drafts(records: list[dict[str, Any]], topics: list[str]) -> list[Draft]:
    """Turn raw batch responses into drafts, keyed by custom_id - results return
    in arbitrary order, so position must never be trusted."""
    drafts: list[Draft] = []
    for record in records:
        cid = record.get("custom_id", "")
        outcome = record.get("result", {})
        if outcome.get("type") != "succeeded":
            log.warning("draft %s failed: %s", cid, outcome.get("type"))
            continue
        blocks = outcome.get("message", {}).get("content") or []
        text = "\n".join(b["text"] for b in blocks if b.get("type") == "text").strip()
        if not text:
            log.warning("draft %s returned no text", cid)
            continue
        index = int(cid.rsplit("-", 1)[1])
        drafts.append(
            Draft(
                draft_id=cid,
                arm=DraftArm.REGISTER,
                topic=topics[index],
                text=text,
                n_chars=len(text),
            )
        )
    return sorted(drafts, key=lambda d: d.draft_id)


def write_drafts(out: Path, drafts: list[Draft]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for d in drafts:
            f.write(d.model_dump_json() + "\n")
    return out


class DraftsManifest(BaseModel):
    """Committed provenance for the eval inputs. Stamps `register_sha256`
    because the REGISTER arm is generated through it - editing the register
    invalidates this set exactly as it invalidates the pairs."""

    schema_version: int = 1
    drafts_version: int
    model: str
    corpus_sha256: str
    topics_sha256: str
    register_sha256: str
    drafts_sha256: str
    batch_ids: list[str]
    git_sha: str
    created_at: str
    n_register: int
    n_wild: int
