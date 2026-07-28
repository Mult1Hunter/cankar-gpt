"""De-style Cankar passages into plain Slovene - the money step of Phase 5.

Produces the SOURCE side of the `(plain -> Cankar)` training pairs by rewriting
each passage through `PLAIN_REGISTER`. The Cankar side is the passage itself,
untouched public-domain text.

The whole design goal is that **no failure after the money is spent costs money
again**, because the thing being bought is tokens, not files:

1. `custom_id` is the content-addressed `passage_id`, and a run subtracts
   everything already on disk before submitting. A crash, a Ctrl-C or a dead
   laptop therefore costs nothing - re-running is always safe and always cheap.
2. Raw responses are persisted verbatim and append-only BEFORE any parsing. A
   bug in the parser or the quality filter is then a re-parse, not a
   re-purchase. This is the mistake that actually costs people money.
3. The `batch_id` is committed at SUBMIT time, not on completion. Results live
   server-side for weeks, so a total local loss inside that window is a
   re-download - but only if the receipt survives.

Model choice is measured, not assumed. A 50-passage pilot across Haiku 4.5 and
Sonnet 5 (2026-07-28) found Haiku producing grammatical errors in Slovene -
broken dual verb forms ("sta se zbudi"), gender disagreement ("da je otroka
presenetila"), a reflexive-only verb used transitively ("Pocutila je mraz") and
anachronism ("biric" -> "varnostnik") - while the frozen style classifier scored
the two models identically (0.693 vs 0.690). The metric could not see the only
thing that mattered. On a low-resource language, the cheap model corrupts the
training data invisibly; that is why this module pins a strong one.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cankar.core.errors import CankarError
from cankar.core.jsonl import iter_jsonl_docs
from cankar.core.paths import PairSet
from cankar.core.register import PLAIN_REGISTER, SOURCE_FIDELITY
from cankar.core.reports import generated_marker, write_report
from cankar.pairs.segment import Passage

log = logging.getLogger("cankar.pairs")

DESTYLER_VERSION = 1

# Pinned by the pilot above. A model change invalidates the pairs: it is stamped
# into the manifest so a later swap marks earlier output stale rather than
# silently mixing two rewriting styles into one training set.
DEFAULT_MODEL = "claude-sonnet-5"

# The passage p99 is 763 chars ~ 280 tokens; de-styled output runs at or below
# source length. 1024 leaves ~3.5x headroom, so `max_tokens` truncation means
# something went wrong (a runaway or a commentary preamble) rather than a long
# passage - which is why TRUNCATED is treated as a defect, not as a resize hint.
MAX_TOKENS = 1024

# Anthropic's per-batch request ceiling.
MAX_BATCH_REQUESTS = 100_000

# Both constants are REGRESSED ON THE REAL BILL, not estimated:
#   in_tokens = 0.4796 * source_chars + 482.5   (10,043 billed requests, 2026-07-28)
#
# The first version measured `count_tokens` against Haiku and applied the result
# to a Sonnet run, underestimating the input bill by 31%. The evidence was
# already in hand and went unused: the pilot recorded Sonnet consuming 628 input
# tokens where Haiku consumed 446 for identical text. Tokenizer density is
# per-model - never carry a measurement across model families.
CHARS_PER_TOKEN = 2.09

# Fixed input per request: the system prompt plus message framing. Sits below
# every prompt-caching floor (1024 for the Sonnet/Haiku class, 512 for Opus 5),
# so caching never engages and this is paid in full on EVERY request - about 75%
# of the input bill at these passage sizes. Padding the prompt to clear the
# floor was considered and rejected: inflating a load-bearing design invariant
# with filler to game a threshold is indefensible.
FIXED_INPUT_TOKENS = 483

# Output tracks source length closely (out_tokens = 0.4807 * plain_chars + 6.3,
# and a de-styled passage runs at ~0.99x the source). Estimated because output
# is the MAJORITY of the bill at Sonnet's 5x output price - a dry run that
# reports only input cannot answer the question it exists to answer.
OUTPUT_CHARS_PER_TOKEN = 2.08


class CostEstimate(BaseModel):
    """What a run would cost, in TOKENS. Prices change and are deliberately not
    hardcoded - this reports volume, the operator applies the current price
    list. Output is included because at a 5x output multiplier it dominates."""

    n_requests: int
    passage_chars: int
    passage_tokens: int
    fixed_tokens: int
    input_tokens: int
    output_tokens: int


def estimate_cost(passages: list[Passage]) -> CostEstimate:
    chars = sum(len(p.text) for p in passages)
    passage_tokens = int(chars / CHARS_PER_TOKEN)
    fixed_tokens = FIXED_INPUT_TOKENS * len(passages)
    return CostEstimate(
        n_requests=len(passages),
        passage_chars=chars,
        passage_tokens=passage_tokens,
        fixed_tokens=fixed_tokens,
        input_tokens=passage_tokens + fixed_tokens,
        output_tokens=int(chars / OUTPUT_CHARS_PER_TOKEN),
    )


SYSTEM_PROMPT = unicodedata.normalize(
    "NFC",
    f"""You rewrite Slovene literary prose into a plain modern register.

{PLAIN_REGISTER}

{SOURCE_FIDELITY}

Output ONLY the rewritten Slovene passage. No preamble, no explanation, no
commentary, no title, no markdown formatting.""",
)


class Anomaly(StrEnum):
    """Why a returned response did not become a pair. Closed set, so a StrEnum
    (`.claude/rules/code-standards.md`). Every one of these was observed or is
    directly reachable - none is speculative.

    The two dangerous ones are TRUNCATED and NOT_A_REWRITE: both yield fluent,
    plausible text that reads fine in a spot-check and teaches the Phase 6
    styler something false.
    """

    API_ERROR = "api_error"  # per-request failure inside a succeeded batch
    NO_TEXT_BLOCK = "no_text_block"  # content empty, or only non-text blocks
    TRUNCATED = "truncated"  # stop_reason == max_tokens
    EMPTY_OUTPUT = "empty_output"
    NOT_A_REWRITE = "not_a_rewrite"  # commentary/preamble instead of the passage
    LENGTH_OUTLIER = "length_outlier"  # far shorter/longer than the source
    LOST_DIACRITICS = "lost_diacritics"  # Slovene carons dropped wholesale
    FOREIGN_SCRIPT = "foreign_script"  # non-Latin letter inside Slovene words
    IDENTITY = "identity"  # output byte-identical to the source


# Calibrated on the 100-response pilot: every good rewrite landed at a length
# ratio of 0.99 +/- small, so these bounds sit far outside the observed good
# band and catch collapse (a one-line summary) or runaway (commentary appended),
# not ordinary variation.
MIN_LENGTH_RATIO = 0.5
MAX_LENGTH_RATIO = 1.8

# A rewrite that opens with markdown or an English lead-in is not a rewrite.
# Observed on a throwaway probe: "# Bilo je sonce.\n\nTo je preprosta izjava ..."
#
# Markup and ENGLISH only. Slovene lead-ins were tried and removed: "Tukaj je"
# rejected a correct rewrite of "Tukaj je napisano, trdno prisito ..." - the
# phrase is ordinary Slovene, and the output side of these pairs is always
# Slovene, so any English opener is unambiguously the model talking to us
# (2026-07-28).
_PREAMBLE_MARKERS = ("#", "```", "Here is", "Rewritten", "Translation:", "The passage")

# Cankar's prose is caron-dense; a rewrite that drops them wholesale has slid
# into a neighbouring orthography. Ratio, not count, so long passages are fair.
_CARONS = frozenset("čšžČŠŽ")

# The ratio only carries signal when the source has enough carons to divide by.
# Measured over the full 10,049-response run: p1 retention is 0.00 for sources
# with 0-2 carons but 0.62 for sources with 11+, because de-styling legitimately
# swaps caron-bearing archaisms for caron-free modern words - "razkrecil" ->
# "razprl" leaves a 1-caron source at 0.00 retention with flawless output.
# Applying the ratio below this floor rejected 87 perfectly good pairs
# (2026-07-28); every one was a false positive.
MIN_CARONS_FOR_RATIO = 8

# Targets COLLAPSE, not variation. Retention of 0.24-0.33 turned out to be
# ordinary vocabulary substitution even above the floor - one archaic word can
# carry four carons ("cicikovscina" -> "prevara") - so a threshold that treats
# a quarter-retention as corruption rejects good pairs. Real orthographic
# collapse lands near zero. Measured: no output in the 10,049-response run
# dropped its carons wholesale, so this is a guard against a model change (the
# Haiku pilot is the reason to believe a weaker one would trip it), not a live
# filter. It is proven to fire by test, not by production traffic.
MIN_CARON_RETENTION = 0.1


class Pair(BaseModel):
    """One training pair. `cankar` is the untouched PD source; `plain` is the
    generated side. Named for the direction Phase 6 trains: plain -> cankar."""

    passage_id: str
    url: str
    title: str
    cankar: str
    plain: str
    model: str
    in_tokens: int
    out_tokens: int


class Rejected(BaseModel):
    """A response that was paid for but is not usable. Kept, not discarded: it
    is evidence about the model and the prompt, and re-running must not re-bill
    it (ADR 0004's never-silently-dropped rule applied to generated output)."""

    passage_id: str
    reason: Anomaly
    detail: str = ""
    output: str = ""


def build_request(passage: Passage, model: str) -> dict[str, Any]:
    """One Batch API request. `custom_id` is the passage_id - results come back
    in ARBITRARY order, so nothing may ever be matched by position.

    Thinking is explicitly DISABLED. Left at the default it is not merely wasted
    output tokens on a rewrite task: thinking counts against `max_tokens`, and a
    20-request validation batch lost one response entirely to a model that spent
    the whole 1024-token budget thinking and emitted no text block at all
    (2026-07-28). That is a 5% loss rate, paid for, with nothing to show.
    """
    return {
        "custom_id": passage.passage_id,
        "params": {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "thinking": {"type": "disabled"},
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": passage.text}],
        },
    }


def foreign_letters(text: str) -> set[str]:
    """Alphabetic characters outside the Latin script.

    Catches homoglyph corruption: the model occasionally emits a Cyrillic `о`,
    `е` or `а` inside a Slovene word - `vesело`, `zavesо`, `neumnе` - which is
    visually identical, survives every length and diacritic check, and lands in
    training data as a token the Slovene tokenizer has never seen. 83 accepted
    pairs carried one before this existed, plus a Tamil codepoint in a rejected
    response that only the length ratio caught, by luck (design-review
    2026-07-28).
    """
    return {ch for ch in text if ch.isalpha() and "LATIN" not in unicodedata.name(ch, "")}


def caron_retention(source: str, output: str) -> float:
    """Fraction of the source's caron count surviving in the output. 1.0 when
    the source has none, so caron-free passages are never penalised."""
    src = sum(ch in _CARONS for ch in source)
    if src == 0:
        return 1.0
    return sum(ch in _CARONS for ch in output) / src


def extract_text(message: dict[str, Any]) -> str:
    """Join every text block.

    NEVER `content[0]["text"]`: the pilot returned a `thinking` block ahead of
    the text on 4 of 50 Sonnet responses, which crashes that indexing and would
    silently lose paid output in a batch run.
    """
    blocks = message.get("content") or []
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text").strip()


def classify_response(passage: Passage, result: dict[str, Any]) -> tuple[str, Anomaly | None]:
    """(text, anomaly). Anomaly is None only when the response is usable."""
    outcome = result.get("result", {})
    if outcome.get("type") != "succeeded":
        return "", Anomaly.API_ERROR
    message = outcome.get("message", {})
    blocks = message.get("content") or []
    if not any(b.get("type") == "text" for b in blocks):
        return "", Anomaly.NO_TEXT_BLOCK
    text = extract_text(message)
    if not text:
        return text, Anomaly.EMPTY_OUTPUT
    if message.get("stop_reason") == "max_tokens":
        # Truncated mid-sentence. Fluent and plausible, and exactly the corrupted
        # pair this stage exists to avoid - never salvage it.
        return text, Anomaly.TRUNCATED
    if text.startswith(_PREAMBLE_MARKERS):
        return text, Anomaly.NOT_A_REWRITE
    if foreign_letters(text) - foreign_letters(passage.text):
        # Diffed against the source: a Greek letter Cankar himself wrote is not
        # corruption, one the rewrite introduced is.
        return text, Anomaly.FOREIGN_SCRIPT
    if text == passage.text:
        # Full-length identity. `min_chars` exists because a near-identity pair
        # teaches nothing; this is that, at any length.
        return text, Anomaly.IDENTITY
    ratio = len(text) / len(passage.text)
    if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
        return text, Anomaly.LENGTH_OUTLIER
    source_carons = sum(ch in _CARONS for ch in passage.text)
    if (
        source_carons >= MIN_CARONS_FOR_RATIO
        and caron_retention(passage.text, text) < MIN_CARON_RETENTION
    ):
        return text, Anomaly.LOST_DIACRITICS
    return text, None


def select_passages(passages: list[Passage], limit: int, done: frozenset[str]) -> list[Passage]:
    """The next `limit` passages still needing de-styling.

    Sorted by passage_id so the subset is deterministic and, crucially,
    EXTENDABLE: raising the limit later adds work without re-billing anything
    already bought, because the prefix is stable.
    """
    pending = [p for p in sorted(passages, key=lambda p: p.passage_id) if p.passage_id not in done]
    return pending[:limit]


def load_passages(path: Path) -> list[Passage]:
    return [Passage.model_validate(d) for d in iter_jsonl_docs(path, "run: cankar pairs segment")]


def append_raw(path: Path, records: list[dict[str, Any]]) -> Path:
    """Persist raw responses verbatim, append-only, BEFORE parsing them. The
    money bought these bytes; parsing is a separate, repeatable concern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def iter_raw(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())
    return iter_jsonl_docs(path)


def already_done(raw_path: Path) -> frozenset[str]:
    """Passage ids with a response already on disk - paid for, never re-sent.
    Read from the RAW log, not from the parsed pairs: a response rejected by the
    quality filter was still billed, and re-sending it would buy it twice."""
    return frozenset(r["custom_id"] for r in iter_raw(raw_path) if "custom_id" in r)


def retryable(raw_path: Path, passages: list[Passage]) -> frozenset[str]:
    """Ids whose paid response was rejected, so a re-send would buy a SECOND
    response for the same passage.

    Deliberately opt-in (`--retry-rejected`), never automatic: it is the one
    operation in this module that knowingly pays twice. It exists because some
    rejections have a fixable cause on our side - the thinking-budget
    exhaustion above was one - and after such a fix the passage is worth
    re-buying. A rejection caused by the passage itself never will be.
    """
    by_id = {p.passage_id: p for p in passages}
    bad: set[str] = set()
    for record in iter_raw(raw_path):
        pid = record.get("custom_id")
        passage = by_id.get(pid) if pid else None
        if passage is None:
            continue
        if classify_response(passage, record)[1] is not None:
            bad.add(passage.passage_id)
    return frozenset(bad)


def build_pairs(
    passages: list[Passage], raw_path: Path, model: str
) -> tuple[list[Pair], list[Rejected]]:
    """Parse the raw log into usable pairs plus a quarantine, keyed by custom_id
    (never by position - batch results return in arbitrary order)."""
    by_id = {p.passage_id: p for p in passages}
    pairs: dict[str, Pair] = {}
    failures: dict[str, Rejected] = {}
    unknown: list[Rejected] = []

    for record in iter_raw(raw_path):
        pid = record.get("custom_id")
        passage = by_id.get(pid) if pid else None
        if passage is None:
            # A response whose passage is no longer in the frozen set - a
            # re-segmentation happened between purchase and parse.
            unknown.append(
                Rejected(passage_id=str(pid), reason=Anomaly.API_ERROR, detail="unknown passage_id")
            )
            continue
        text, anomaly = classify_response(passage, record)
        if anomaly is not None:
            failures.setdefault(
                passage.passage_id,
                Rejected(passage_id=passage.passage_id, reason=anomaly, output=text),
            )
            continue
        usage = record["result"]["message"].get("usage", {})
        pairs[passage.passage_id] = Pair(
            passage_id=passage.passage_id,
            url=passage.url,
            title=passage.title,
            cankar=passage.text,
            plain=text,
            model=model,
            in_tokens=int(usage.get("input_tokens", 0)),
            out_tokens=int(usage.get("output_tokens", 0)),
        )

    # Resolved PER PASSAGE, not per record. The raw log can legitimately hold
    # several records for one id - `--retry-rejected` adds one, and re-draining
    # a batch appends the same responses again - so a passage is a pair if ANY
    # record succeeded, and rejected only if none did. Keying off records
    # instead would emit duplicate pairs for one passage and count a passage
    # that succeeded on retry as a loss (observed 2026-07-28).
    rejected = [r for pid, r in failures.items() if pid not in pairs]
    return list(pairs.values()), rejected + unknown


def write_pairs(out: Path, pairs: list[Pair]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p.model_dump_json() + "\n")
    return out


def write_rejected(out: Path, rejected: list[Rejected]) -> Path:
    """Persist the quarantine. `Rejected` claimed to be "kept, not discarded"
    while the CLI reduced it to counts and threw the text away, so the evidence
    that would explain a filter change did not survive the run that produced it
    (design-review 2026-07-28)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(r.model_dump_json() + "\n")
    return out


class PairsManifest(BaseModel):
    """Committed provenance for the generated pairs (ADR 0003 shape).

    This IS the artifact that depends on the register, so `register_sha256`
    belongs here - editing `PLAIN_REGISTER` marks these pairs stale, which is
    the train/inference distribution guard of design invariant #1. (The passages
    manifest deliberately does not stamp it: segmentation never reads it.)

    `batch_ids` are the receipts. Results live server-side for weeks, so they
    are the difference between a re-download and a re-purchase.
    """

    schema_version: int = 1
    destyler_version: int
    model: str
    pair_set: str
    corpus_sha256: str
    passages_sha256: str
    register_sha256: str
    pairs_sha256: str
    batch_ids: list[str]
    git_sha: str
    created_at: str
    lib_versions: dict[str, str]
    n_requested: int
    n_pairs: int
    n_rejected: int
    reject_counts: dict[str, int]
    in_tokens: int
    out_tokens: int


def write_pairs_report(out: Path, manifest: PairsManifest, samples: list[Pair]) -> Path:
    m = manifest
    total = m.n_pairs + m.n_rejected
    lines: list[str] = [
        generated_marker("cankar pairs destyle", snapshot=True),
        "",
        "# De-styled pairs - Phase 5",
        "",
        f"Corpus sha256 `{m.corpus_sha256}`.",
        f"Passages sha256 `{m.passages_sha256}` (data/pairs/passages.jsonl).",
        f"Register sha256 `{m.register_sha256}` (cankar/core/register.py).",
        f"Pairs sha256 `{m.pairs_sha256}` (data/pairs/pairs.jsonl).",
        f"Model `{m.model}`, de-styler v{m.destyler_version}, "
        f"at {m.created_at} (git `{m.git_sha}`).",
        "",
        f"**{m.n_pairs:,} pairs** from {total:,} responses "
        f"({m.in_tokens:,} in / {m.out_tokens:,} out tokens).",
        "",
        "The Cankar side is untouched public-domain text; the plain side is",
        "generated. Pairs train `plain -> cankar`.",
        "",
        "## Rejected responses",
        "",
        "Paid for, kept, never silently dropped - and never re-sent, because the",
        "raw log is what a resumed run subtracts against.",
        "",
        "| reason | count |",
        "|---|---:|",
    ]
    for reason, count in sorted(m.reject_counts.items()):
        lines.append(f"| `{reason}` | {count:,} |")
    lines += ["", "## Samples", ""]
    for p in samples:
        lines += [
            f"**{p.title}**",
            "",
            f"- cankar: {p.cankar}",
            f"- plain: {p.plain}",
            "",
        ]
    lines += ["## Reproducing", "", "```", "uv run cankar pairs destyle", "```"]
    write_report(out, lines)
    return out


class BatchReceipt(BaseModel):
    """A submitted batch. Written to a COMMITTED file the moment the API
    returns an id, before anything can be lost, and marked `downloaded` only
    once the responses are safely on disk.

    The un-downloaded case is the one that matters: a run interrupted between
    submit and download must resume the existing batch, never submit a second
    one for the same passages. That is the difference between resuming and
    paying twice.
    """

    batch_id: str
    created_at: str
    model: str
    n_requests: int
    downloaded: bool = False
    # Which set this batch belongs to, so a resumed drain appends to the right
    # raw log. Defaults to TRAIN so receipts written before the split still load.
    pair_set: PairSet = PairSet.TRAIN


def load_receipts(path: Path) -> list[BatchReceipt]:
    """Fold the append-only event log into one record per batch, last write wins.

    Order is preserved by first appearance so the ledger reads chronologically.
    """
    if not path.exists():
        return []
    folded: dict[str, BatchReceipt] = {}
    for row in iter_jsonl_docs(path):
        r = BatchReceipt.model_validate(row)
        folded[r.batch_id] = r
    return list(folded.values())


def append_receipt(path: Path, receipt: BatchReceipt) -> Path:
    """APPEND one event. Never rewrites.

    The previous version truncated and rewrote from an in-memory list, which lost
    the receipt for a 10,000-request batch when an unrelated `git checkout --
    registry/` reverted this tracked file to its committed state mid-run
    (2026-07-28). The raw-response log next door was append-only and survived;
    the ledger that is the ONLY route back to paid results was not. One module,
    two ledgers, and the weaker discipline was on the file that mattered more.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(receipt.model_dump_json() + "\n")
    return path


def unrecorded_batches(api_batch_ids: list[str], receipts: list[BatchReceipt]) -> list[str]:
    """Batch ids the API knows about and this repo does not.

    Submitting while one exists risks paying twice for the same passages: the
    SDK retries a failed POST with no idempotency key, so a create() whose
    response was lost can leave a billed batch nobody has a receipt for, whose
    responses `already_done` therefore cannot subtract.
    """
    known = {r.batch_id for r in receipts}
    return [b for b in api_batch_ids if b not in known]


def pending_receipts(receipts: list[BatchReceipt]) -> list[BatchReceipt]:
    """Batches paid for but not yet on disk - drain these before submitting."""
    return [r for r in receipts if not r.downloaded]


def require_batch_size(n: int) -> None:
    if n > MAX_BATCH_REQUESTS:
        raise CankarError(
            f"{n:,} requests exceeds the {MAX_BATCH_REQUESTS:,} per-batch ceiling - lower --limit"
        )
