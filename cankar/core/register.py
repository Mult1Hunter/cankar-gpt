"""The shared "plain Slovene register" definition - design invariant #1.

ONE definition of the register, used verbatim in BOTH directions of the
style-transfer pipeline:

- Phase 5 de-styling: Cankar prose -> plain Slovene, producing the SOURCE side of
  the training pairs.
- Phase 8 inference: the knowledge model writes a plain draft that the styler
  then Cankar-izes.

If those two ever diverge, the styler meets a distribution at inference it was
never trained on. That failure is silent - it does not raise, it just makes the
styler quietly mediocre, three phases downstream of the edit that caused it.

Deliberately a module constant, not a config file: `configs/` holds what varies
per run, and NOT varying is the whole property being bought here. (Contrast
`cankar/train/config.py`'s `sample_prompt`, which is correctly a config field
because it genuinely is per-run.)

Instructions are in English - clearer to author and audit; the OUTPUT is always
Slovene. `REGISTER_SHA256` is the register's identity, for stamping into the
manifest of anything generated with it, so that a later edit marks earlier
artifacts stale instead of silently mixing two distributions. Nothing stamps it
yet; Phase 5 is its first consumer.
"""

from __future__ import annotations

import hashlib
import unicodedata

# NFC at definition time: the block carries carons, and this repo has been bitten
# by NFD-vs-NFC before (CLAUDE.md). Without this an editor that saves NFD would
# change REGISTER_SHA256 without changing a single visible glyph.
PLAIN_REGISTER = unicodedata.normalize(
    "NFC",
    """\
**Plain Slovene register.** Contemporary standard Slovene (knjižni jezik), the
kind found in clear modern expository prose or quality journalism. Specifically:

- **Neutral and unadorned.** No literary flourish, no poetic diction, no
  rhetorical repetition, no exclamatory or archaic tone.
- **Modern orthography and grammar.** Standard present-day spelling (e.g.
  "sonce", never "solnce"; "veselje", not dated variants). No archaisms.
- **Plain vocabulary.** Everyday modern words; replace archaic or literary terms
  with their ordinary equivalents.
- **Clear syntax.** Straightforward word order, moderate sentence length; avoid
  inverted or heavily subordinated literary constructions.
- **Prose only.** No line breaks as verse, no headings, no quotation marks
  wrapping the whole output, no commentary or meta text.""",
)

# NOT part of the register, and deliberately a separate constant: these are
# fidelity constraints on a REWRITE, so they only mean anything when a source
# passage exists. Phase 5 (de-styling) appends them; Phase 8 (drafting from a
# request, with no source to be faithful to) does not.
#
# Folding them into PLAIN_REGISTER would look like a stronger shared block while
# actually weakening it: bytes identical in both directions, a third of them
# inert in one - and the pressure that creates is a call site that slices or
# `.replace()`s the block, which no gate here can see (design-review 2026-07-27).
SOURCE_FIDELITY = unicodedata.normalize(
    "NFC",
    """\
- **Meaning preserved exactly.** Keep every event, fact, relationship, and
  emotional beat. Do not add, remove, reinterpret, or invent anything.
- **Entities verbatim.** Preserve all proper nouns - people, places - exactly as
  written.""",
)

REGISTER_SHA256 = hashlib.sha256(PLAIN_REGISTER.encode("utf-8")).hexdigest()
