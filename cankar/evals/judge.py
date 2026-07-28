"""LLM meaning-judge (eval pillar #3): does the styled output still say the
same thing, in Cankar's voice, in real Slovene?

Built because the style classifier turned out to be unfit for this. It scores
0.993 on the task it was trained for (Cankar vs 14 public-domain peers) and
0.650 on the task it is deployed on (Cankar vs its own de-styled pair), because
its negative was 14 authors who all write 1900s literary prose - so period
vocabulary carried the signal and nothing in its training could reveal that
(registry/reports/style-deploy.md). It stays a directional signal; it cannot
carry a claim.

**Absolute per-item scoring, not pairwise.** Pairwise judging is the more
sensitive design and it buys a bias this project does not need: LLM judges favour
whichever candidate is shown first, reported as high as a 75% preference. Scoring
one item at a time removes the failure mode instead of correcting for it, and the
controls below recover the discrimination that pairwise would have bought.

**The judge is itself an unvalidated instrument.** That is exactly the mistake
that produced the classifier, so it does not get made twice: every batch carries
blind controls whose correct answer is known in advance, and the run is only
usable if they come out right. See `ControlKind`.

**Self-preference is not fixable here, only bounded.** The plain side of every
pair was written by Claude, so a Claude judge is scoring text descended from its
own output, and judges are documented to favour their own style. Using a bigger
model from the same family does not address it. The controls bound it (a
self-preferring judge still has to rank ECHO below a real rewrite) and a human
spot-check by a Slovene speaker is the only real calibration.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from cankar.core.batch import CHARS_PER_TOKEN, DEFAULT_MODEL, extract_text
from cankar.core.errors import CankarError

log = logging.getLogger("cankar.evals")

JUDGE_VERSION = 1
MAX_TOKENS = 512

FIXED_INPUT_TOKENS = 620  # the rubric below, measured once and pinned


class Axis(StrEnum):
    """What each item is scored on. Separate axes rather than one quality score:
    a single number lets a fluent rewrite that drops the content score well, and
    the whole point here is to catch exactly that."""

    MEANING = "meaning"
    VOICE = "voice"
    FLUENCY = "fluency"


class ControlKind(StrEnum):
    """Blind items with a known correct answer, mixed into every batch.

    A judge that scores everything highly is useless in the same way the style
    classifier was, and nothing in a set of plausible-looking scores reveals it.
    These do, because each one is wrong in a DIFFERENT direction:

    - REAL_CANKAR: the true passage, presented as if the model wrote it. Ceiling.
      Should score high on every axis.
    - ECHO: the plain input handed back unchanged. Meaning is perfectly preserved
      and no styling happened, so it MUST score high on meaning and low on voice.
      Catches a judge that is really scoring "is this Slovene prose".
    - MISMATCH: a real Cankar passage from a DIFFERENT pair. Perfect voice,
      unrelated content, so it MUST score low on meaning and high on voice.
      Catches a judge that reads period vocabulary and calls it meaning - which
      is precisely how the style classifier failed.

    ECHO and MISMATCH are the load-bearing pair: together they prove the meaning
    and voice axes move independently. A judge that passes REAL_CANKAR alone has
    demonstrated nothing.
    """

    NONE = "none"
    REAL_CANKAR = "real_cankar"
    ECHO = "echo"
    MISMATCH = "mismatch"


@dataclass(frozen=True)
class JudgeItem:
    """One thing to score. `control` is never sent to the model."""

    item_id: str
    source: str  # the plain-Slovene input the styler was given
    candidate: str  # what is being judged
    control: ControlKind = ControlKind.NONE


class Verdict(BaseModel):
    """One judged item. Scores are 1-5 integers; `notes` is kept because a
    rationale is the only way a human spot-check can tell a wrong score from a
    right one for the wrong reason."""

    item_id: str
    meaning: int = Field(ge=1, le=5)
    voice: int = Field(ge=1, le=5)
    fluency: int = Field(ge=1, le=5)
    notes: str = ""

    @field_validator("notes", mode="before")
    @classmethod
    def _flatten_notes(cls, v: Any) -> str:
        """The rubric asks for "one short clause per axis", which the model
        reasonably reads as an object keyed by axis. Both shapes are correct
        answers to the instruction, so both are accepted rather than making a
        prose field the reason a paid batch is unusable."""
        if isinstance(v, dict):
            return "; ".join(f"{k}: {val}" for k, val in v.items())
        return "" if v is None else str(v)


RUBRIC = """You are evaluating Slovene style transfer. A small language model was
given a passage of plain modern Slovene and asked to rewrite it in the prose voice
of Ivan Cankar (1876-1918) WITHOUT changing what the passage says.

Score the candidate on three INDEPENDENT axes, 1-5 integers.

MEANING - does the candidate say the same thing as the source?
  5 every fact, actor and event preserved; nothing added or lost
  4 preserved, with a small embellishment or omission that changes nothing
  3 the gist survives but a concrete detail is wrong, dropped or invented
  2 substantially different content; only the topic survives
  1 unrelated content
  Judge ONLY correspondence to the source. Archaic wording is not a meaning
  change. A passage that is beautiful and unrelated scores 1.

VOICE - does it read like Cankar's prose?
  5 indistinguishable from Cankar: his rhythm, subordination, repetition,
    emotional register, concrete-image-then-turn movement
  4 clearly styled, a few flat or modern stretches
  3 some archaic vocabulary applied to otherwise modern sentences
  2 barely touched
  1 plain modern Slovene, unchanged
  Vocabulary alone is NOT voice. Sprinkling old words on a modern sentence is a 3
  at most. If the candidate is word-for-word the source, VOICE is 1 however
  literary the source already sounds.

FLUENCY - is it well-formed Slovene?
  5 grammatical and natural throughout
  4 minor awkwardness
  3 noticeable errors: wrong case, broken agreement, an invented word
  2 several errors, hard to read
  1 word salad, repetition loops, non-Slovene fragments
  Judge form only. Archaic-but-correct Slovene is 5. An invented word that
  looks Slovene but is not (e.g. a plausible-looking non-word) caps this at 3.

The axes are independent. Unrelated content in perfect Cankar is MEANING 1,
VOICE 5. The source echoed back is MEANING 5, VOICE 1. Do not let one axis pull
another.

Reply with ONLY a JSON object, no markdown fence:
{"meaning": <1-5>, "voice": <1-5>, "fluency": <1-5>, "notes": "<one short clause per axis>"}"""

SYSTEM_PROMPT = unicodedata.normalize("NFC", RUBRIC)


class CostEstimate(BaseModel):
    n_requests: int
    input_tokens: int
    output_tokens: int
    usd_batch: float


def estimate_cost(
    items: list[JudgeItem], usd_in: float = 3.0, usd_out: float = 15.0
) -> CostEstimate:
    """Batch API halves both rates. Defaults are Sonnet's published per-MTok
    prices, passed in so a model change cannot silently keep the old numbers."""
    text = sum(len(i.source) + len(i.candidate) for i in items)
    inp = int(text / CHARS_PER_TOKEN) + FIXED_INPUT_TOKENS * len(items)
    out = 200 * len(items)  # structured verdict + a short rationale
    return CostEstimate(
        n_requests=len(items),
        input_tokens=inp,
        output_tokens=out,
        usd_batch=round(inp / 1e6 * usd_in / 2 + out / 1e6 * usd_out / 2, 2),
    )


def build_request(item: JudgeItem, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Batch request for one item.

    The control kind is deliberately absent from everything the model sees - a
    control the judge can recognise is not a control. Thinking is disabled: it
    counts against max_tokens and the verdict is a rubric application, not a
    reasoning problem.
    """
    return {
        "custom_id": item.item_id,
        "params": {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "thinking": {"type": "disabled"},
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"SOURCE (plain Slovene):\n{item.source}\n\n"
                        f"CANDIDATE (to score):\n{item.candidate}"
                    ),
                }
            ],
        },
    }


def append_raw(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write what the money bought, BEFORE parsing it.

    The Phase 5 rule, learned there and skipped here on the first run: a parser
    bug must cost a re-parse, not a re-purchase. The first judge batch was paid
    for and thrown away because a `notes` field came back as an object instead of
    a string and nothing had persisted the response.

    Append-only, so a resumed or re-parsed run never truncates the ledger.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def parse_raw(path: Path) -> dict[str, Verdict]:
    """Re-parse a saved batch. The whole point of `append_raw` - re-running the
    judge on stored bytes costs nothing."""
    verdicts: dict[str, Verdict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            msg = (rec.get("result") or {}).get("message") or {}
            if not msg:
                log.warning("no message for %s", rec.get("custom_id"))
                continue
            v = parse_verdict(rec["custom_id"], extract_text(msg))
            verdicts[v.item_id] = v
    return verdicts


def parse_verdict(item_id: str, raw: str) -> Verdict:
    """Parse one JSON verdict, tolerating a stray markdown fence.

    A malformed verdict raises rather than defaulting: a silent 0 or a dropped
    row would move every aggregate it lands in, and the controls are how this
    module knows anything - one of them quietly vanishing is the failure it is
    least able to detect.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise CankarError(f"{item_id}: no JSON object in judge reply: {raw[:120]!r}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise CankarError(f"{item_id}: unparseable judge reply: {raw[:120]!r}") from exc
    return Verdict(item_id=item_id, **payload)


# ----------------------------------------------------------------------------
# the controls - "is this judge usable at all"
# ----------------------------------------------------------------------------

MIN_CONTROL_MARGIN = 1.0
"""How far apart two control groups must sit, on a 1-5 scale, to count as
discriminated.

One full scale point: the axes are integers, so anything smaller can be produced
by rounding noise alone and would let a judge that scores everything 4 pass by
drifting to 4.3 vs 3.8. Not a tuned value - it is the smallest gap that is
meaningful on this scale, and the check below reports the measured margins so a
marginal pass is visible rather than swallowed.
"""


class ControlOutcome(BaseModel):
    """Whether the judge did the one thing it must do to be believed."""

    n_real: int
    n_echo: int
    n_mismatch: int
    real_voice: float
    echo_voice: float
    mismatch_voice: float
    real_meaning: float
    echo_meaning: float
    mismatch_meaning: float
    # The ceiling for every reported axis. Real Cankar scores ~3.1 on VOICE, not
    # 5: the rubric's "indistinguishable from Cankar" is not awarded even to
    # Cankar on a short excerpt. A styler score read against 5 would therefore
    # understate it, and one read against this row is honest. Measured, not
    # assumed - which is the entire reason REAL_CANKAR is in every batch.
    real_fluency: float
    voice_margin: float  # real vs echo - can it see styling at all?
    meaning_margin: float  # real vs mismatch - can it see content at all?
    axes_independent: bool
    usable: bool
    failures: list[str] = []


def check_controls(items: list[JudgeItem], verdicts: dict[str, Verdict]) -> ControlOutcome:
    """Two questions, both of which must come out yes.

    1. Can it see styling? ECHO is the source verbatim, so its VOICE must sit a
       full point below REAL_CANKAR's. A judge failing this is scoring "is this
       literary Slovene", which the source already is.
    2. Can it see content? MISMATCH is real Cankar about something else, so its
       MEANING must sit a full point below REAL_CANKAR's. A judge failing this
       is scoring period vocabulary and calling it meaning - the classifier's
       exact failure, one layer up.

    Independence is the conjunction: MISMATCH must keep a HIGH voice score while
    its meaning collapses. If both axes fall together the judge has one internal
    quality dimension wearing three labels.
    """
    by_kind: dict[ControlKind, list[Verdict]] = {k: [] for k in ControlKind}
    for item in items:
        if item.control is not ControlKind.NONE:
            if item.item_id not in verdicts:
                raise CankarError(
                    f"control {item.item_id} has no verdict - cannot validate the run"
                )
            by_kind[item.control].append(verdicts[item.item_id])

    def mean(kind: ControlKind, axis: Axis) -> float:
        rows = by_kind[kind]
        if not rows:
            raise CankarError(f"no {kind.value} controls in this run - the batch was built wrong")
        return sum(getattr(v, axis.value) for v in rows) / len(rows)

    rv, ev, mv = (
        mean(k, Axis.VOICE)
        for k in (ControlKind.REAL_CANKAR, ControlKind.ECHO, ControlKind.MISMATCH)
    )
    rf = mean(ControlKind.REAL_CANKAR, Axis.FLUENCY)
    rm, em, mm = (
        mean(k, Axis.MEANING)
        for k in (ControlKind.REAL_CANKAR, ControlKind.ECHO, ControlKind.MISMATCH)
    )

    voice_margin = rv - ev
    meaning_margin = rm - mm
    # a mismatched passage is still real Cankar: its voice must stay up while
    # its meaning collapses, or the two axes are one axis
    # SAME-AXIS. The earlier version compared MISMATCH's voice against its own
    # meaning, subtracting one axis from another - the two have different
    # empirical ceilings (real Cankar: 4.60 meaning vs 3.65 voice), so that
    # difference carries a constant axis offset unrelated to independence.
    # MISMATCH candidates ARE real Cankar, so the well-founded test is that their
    # voice stays NEAR REAL_CANKAR's: if it collapses alongside meaning, the
    # meaning failure is bleeding into voice, which is the halo being ruled out.
    independent = abs(mv - rv) < MIN_CONTROL_MARGIN

    failures = []
    if voice_margin < MIN_CONTROL_MARGIN:
        failures.append(
            f"cannot see styling: REAL voice {rv:.2f} vs ECHO voice {ev:.2f} "
            f"(margin {voice_margin:.2f} < {MIN_CONTROL_MARGIN})"
        )
    if meaning_margin < MIN_CONTROL_MARGIN:
        failures.append(
            f"cannot see content: REAL meaning {rm:.2f} vs MISMATCH meaning {mm:.2f} "
            f"(margin {meaning_margin:.2f} < {MIN_CONTROL_MARGIN})"
        )
    if not independent:
        failures.append(
            f"axes are not independent: MISMATCH voice {mv:.2f} vs REAL voice {rv:.2f} "
            "- both are real Cankar, so a voice gap here means the meaning failure "
            "is bleeding into the voice score (halo)"
        )
    # The ControlKind docstring declares four MUSTs; enforcing two left a judge
    # that scores the source-verbatim ECHO as meaning 2 passing silently.
    if em < rm - MIN_CONTROL_MARGIN:
        failures.append(
            f"ECHO meaning {em:.2f} is below REAL meaning {rm:.2f} - the echoed source "
            "preserves meaning perfectly by construction, so this judge is not "
            "reading meaning"
        )

    return ControlOutcome(
        n_real=len(by_kind[ControlKind.REAL_CANKAR]),
        n_echo=len(by_kind[ControlKind.ECHO]),
        n_mismatch=len(by_kind[ControlKind.MISMATCH]),
        real_voice=rv,
        echo_voice=ev,
        mismatch_voice=mv,
        real_meaning=rm,
        echo_meaning=em,
        mismatch_meaning=mm,
        real_fluency=rf,
        voice_margin=voice_margin,
        meaning_margin=meaning_margin,
        axes_independent=independent,
        usable=not failures,
        failures=failures,
    )


def build_controls(pairs: list[dict[str, str]], n: int, seed: int = 20260728) -> list[JudgeItem]:
    """One of each control kind per sampled pair, so all three share a source
    and differ only in what is presented as the candidate.

    MISMATCH pairs a source with the Cankar side of a DIFFERENT passage, offset
    rather than randomly drawn so it can never accidentally select its own pair -
    a mismatch that matches is a control that silently asserts nothing.
    """
    import random

    if len(pairs) < 2:
        raise CankarError("need at least 2 pairs to build a mismatch control")
    rng = random.Random(seed)
    picked = rng.sample(range(len(pairs)), min(n, len(pairs)))
    items: list[JudgeItem] = []
    for i in picked:
        src = pairs[i]["plain"]
        other = pairs[(i + len(pairs) // 2) % len(pairs)]
        items += [
            JudgeItem(f"ctl-real-{i}", src, pairs[i]["cankar"], ControlKind.REAL_CANKAR),
            JudgeItem(f"ctl-echo-{i}", src, src, ControlKind.ECHO),
            JudgeItem(f"ctl-mismatch-{i}", src, other["cankar"], ControlKind.MISMATCH),
        ]
    return items


class SeriesScores(BaseModel):
    n: int
    meaning: float
    voice: float
    fluency: float


class JudgeScores(BaseModel):
    """Scores per series. Derived from the ITEMS of this batch, never by prefix
    match over the append-only raw ledger - that let a previous larger run's
    verdicts into the headline of a later smaller one, silently averaging over
    generations from a different checkpoint."""

    holdout: SeriesScores | None = None
    drafts: SeriesScores | None = None
    overall: SeriesScores

    def describe(self) -> str:
        d = self.drafts
        tail = f" | drafts m{d.meaning:.2f} v{d.voice:.2f} f{d.fluency:.2f}" if d else ""
        return (
            f"overall m{self.overall.meaning:.2f} v{self.overall.voice:.2f} "
            f"f{self.overall.fluency:.2f} (n={self.overall.n}){tail}"
        )


def score_series(items: list[JudgeItem], verdicts: dict[str, Verdict]) -> JudgeScores:
    """Aggregate the scored (non-control) items of THIS batch."""

    def agg(rows: list[Verdict]) -> SeriesScores | None:
        if not rows:
            return None
        return SeriesScores(
            n=len(rows),
            **{a.value: sum(getattr(r, a.value) for r in rows) / len(rows) for a in Axis},
        )

    scored = [
        (i, verdicts[i.item_id])
        for i in items
        if i.control is ControlKind.NONE and i.item_id in verdicts
    ]
    if not scored:
        raise CankarError("no scored verdicts for this batch's items")
    overall = agg([v for _, v in scored])
    assert overall is not None
    return JudgeScores(
        holdout=agg([v for i, v in scored if i.item_id.startswith("holdout-")]),
        drafts=agg([v for i, v in scored if i.item_id.startswith("draft-")]),
        overall=overall,
    )


class JudgeReceipt(BaseModel):
    """Committed at SUBMIT time. Results live server-side for weeks, so a crash
    before download is a re-download - but only if the id survived."""

    batch_id: str
    created_at: str
    model: str
    n_requests: int
    outputs: str


def append_receipt(path: Path, receipt: JudgeReceipt) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(receipt.model_dump_json() + "\n")
    return path


def write_judge_report(
    out: Path,
    outcome: ControlOutcome,
    scores: JudgeScores,
    corpus_sha: str = "",
    checkpoint: str = "",
    checkpoint_sha: str = "",
    n_truncated: int = 0,
) -> None:
    from cankar.core.reports import generated_marker, write_report

    verdict = "USABLE" if outcome.usable else "NOT USABLE"

    def row(label: str, s: SeriesScores) -> list[str]:
        return [f"| {label} | {s.n} | {s.meaning:.2f} | {s.voice:.2f} | {s.fluency:.2f} |"]

    L = [
        generated_marker("cankar evals judge", snapshot=True),
        "",
        "# LLM meaning-judge - controls and scores",
        "",
        f"Corpus sha256 `{corpus_sha}`.",
        f"Checkpoint `{checkpoint}` sha256 `{checkpoint_sha or 'UNRECORDED'}`.",
        "",
        f"**Judge status: {verdict}.**",
        "",
    ]

    # The finding first: the pooled number is ~89% held-out and buries it.
    if scores.drafts:
        d = scores.drafts
        L += [
            "## The finding",
            "",
            f"On **fresh drafts** (n={d.n}, modern topics the model never saw) styler-v1",
            f"scores meaning **{d.meaning:.2f}**, voice **{d.voice:.2f}**, fluency"
            f" **{d.fluency:.2f}** on a 1-5 scale whose floor is 1. Fluency at the floor",
            "means the output is not well-formed Slovene, which no amount of style",
            "tuning fixes.",
            "",
        ]

    L += [
        "## Scores",
        "",
        "Read against the ceiling row, not against 5: real Cankar scores about",
        f"{outcome.real_voice:.1f} on voice, so 5 is not a reachable target. Ceilings are",
        "measured in the same batch.",
        "",
        "| series | n | meaning | voice | fluency |",
        "|---|--:|--:|--:|--:|",
    ]
    if scores.holdout:
        L += row("held-out pairs", scores.holdout)
    if scores.drafts:
        L += row("fresh drafts", scores.drafts)
    L += row("**all scored**", scores.overall)
    L += [
        f"| _ceiling (real Cankar)_ | {outcome.n_real} | {outcome.real_meaning:.2f} | "
        f"{outcome.real_voice:.2f} | {outcome.real_fluency:.2f} |",
        "",
    ]
    if scores.holdout and scores.drafts:
        h, d = scores.holdout, scores.drafts
        L += [
            "The gap between the two rows is the honest headline. Held-out sources are",
            "de-styled Cankar, so they still carry his subject matter; the drafts are",
            "modern topics in the same plain register (design invariant #1), so topic is",
            "the only variable that moves.",
            "",
            f"- meaning {h.meaning - d.meaning:+.2f}, voice {h.voice - d.voice:+.2f}, "
            f"fluency {h.fluency - d.fluency:+.2f}",
            "",
        ]
    if n_truncated:
        L += [
            f"{n_truncated} of the scored generations hit `max_tokens` without emitting",
            "a stop token, so they were cut mid-passage - a fluency penalty the model",
            "earned in a way the rubric cannot distinguish from ordinary incoherence.",
            "",
        ]

    L += [
        "## Controls",
        "",
        "An unvalidated instrument producing plausible numbers is how the style",
        "classifier came to be trusted for a task it fails. Every batch carries items",
        "with a known answer.",
        "",
        "| control | n | meaning | voice |",
        "|---|--:|--:|--:|",
        f"| REAL_CANKAR (the true passage) | {outcome.n_real} | "
        f"{outcome.real_meaning:.2f} | {outcome.real_voice:.2f} |",
        f"| ECHO (source verbatim) | {outcome.n_echo} | "
        f"{outcome.echo_meaning:.2f} | {outcome.echo_voice:.2f} |",
        f"| MISMATCH (Cankar, wrong content) | {outcome.n_mismatch} | "
        f"{outcome.mismatch_meaning:.2f} | {outcome.mismatch_voice:.2f} |",
        "",
        f"- can it see styling? REAL vs ECHO voice **{outcome.voice_margin:+.2f}** "
        f"(needs >= {MIN_CONTROL_MARGIN})",
        f"- can it see content? REAL vs MISMATCH meaning **{outcome.meaning_margin:+.2f}** "
        f"(needs >= {MIN_CONTROL_MARGIN})",
        f"- no halo? MISMATCH voice within {MIN_CONTROL_MARGIN} of REAL voice: "
        f"**{outcome.axes_independent}**",
        "",
    ]
    if outcome.failures:
        L += ["### Why it is not usable", ""] + [f"- {f}" for f in outcome.failures] + [""]
    L += [
        "### What a pass does NOT establish",
        "",
        "The rubric states both extremes explicitly (an echoed source is VOICE 1,",
        "unrelated content is MEANING 1), so ECHO and MISMATCH scoring perfectly - with",
        "zero variance across every item - shows the judge APPLIES the rubric at its",
        "stated ends. It does not establish discrimination in the 2-3 band where",
        "styler-v1 actually sits. That is the same class of gap this harness diagnosed",
        "in the style classifier, and the honest bound on what these numbers support.",
        "",
        "Self-preference is bounded, not removed: the plain side of every pair was",
        "written by Claude, so a Claude judge scores text descended from its own",
        "output. A human spot-check by a Slovene speaker is the only real calibration.",
    ]
    write_report(out, L)
