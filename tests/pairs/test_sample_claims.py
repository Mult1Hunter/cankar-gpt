"""Every Cankar/plain excerpt published in docs/ must exist in the frozen data.

The blog-content failure mode is not fabrication, it is *improvement*: quietly
tidying an example so the before/after reads better than the pipeline actually
performs. That is unfalsifiable once `data/pairs/pairs.jsonl` is gitignored, so
`registry/datasets/pairs/samples.jsonl` commits the excerpts and this gate
compares against it.

Modelled on `tests/evals/test_bpb_claims.py`, which does the same job for
published numbers. Same known blind spot, stated plainly: this proves each
quoted string exists in a real pair, NOT that the two sides of a quoted pair
belong to each other, and NOT that a claimed vocabulary swap runs in the
direction the prose claims. A reversed row (`strmel` -> `zrl`, published as
modernization when it is the reverse) passed this suite and was caught by a
human reading it. A sample doc that paired one work's Cankar text with
another's plain rendering would read green here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SAMPLES = REPO / "registry" / "datasets" / "pairs" / "train-samples.jsonl"
DOC = REPO / "docs" / "pairs-samples.md"

# Blockquote lines of the form "> **Cankar:** ..." / "> **plain:** ..."
_QUOTE = re.compile(r"^>\s*\*\*(Cankar|plain):\*\*\s*(.+?)\s*$", re.MULTILINE)

# Vocabulary-table rows: | `duri` | `vrata` | door |
_VOCAB = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|", re.MULTILINE)


def _samples() -> list[dict]:
    return [json.loads(line) for line in SAMPLES.read_text(encoding="utf-8").splitlines() if line]


def test_samples_file_is_populated() -> None:
    """Guards the whole suite against passing vacuously on an empty file."""
    rows = _samples()
    assert len(rows) >= 10, f"only {len(rows)} samples committed"
    assert all(r["cankar"] and r["plain"] for r in rows)


def test_doc_quotes_real_passages() -> None:
    """Each quoted excerpt must appear verbatim inside a committed pair."""
    rows = _samples()
    quotes = _QUOTE.findall(DOC.read_text(encoding="utf-8"))
    assert quotes, "no before/after quotes found - did the doc format change?"

    haystack = {"Cankar": [r["cankar"] for r in rows], "plain": [r["plain"] for r in rows]}
    missing = [
        (side, text) for side, text in quotes if not any(text in hay for hay in haystack[side])
    ]
    assert not missing, (
        f"excerpts not found in {SAMPLES.name}: {missing}. Re-run "
        "`cankar pairs destyle --parse-only` or fix the quote - do not edit it to taste."
    )


def test_vocabulary_table_pairs_are_attested() -> None:
    """Every claimed word swap must actually occur: the old form in some Cankar
    side, the new form in the plain side of the SAME pair. A table is the
    easiest place to assert a modernization the model never made."""
    rows = _samples()
    claims = _VOCAB.findall(DOC.read_text(encoding="utf-8"))
    assert len(claims) >= 5, f"vocabulary table has only {len(claims)} rows"

    unattested = [
        (old, new)
        for old, new in claims
        if not any(
            old.lower() in r["cankar"].lower() and new.lower() in r["plain"].lower() for r in rows
        )
    ]
    assert not unattested, f"vocabulary swaps not attested in any committed pair: {unattested}"
