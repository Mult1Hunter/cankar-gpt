"""Drift and encoding checks on the shared register (invariant #1).

The repo-wide one-home gate is `tests/structure/test_register_home.py`; these
are the checks that only concern the module's own bytes.
"""

from __future__ import annotations

import hashlib
import unicodedata

from cankar.core.register import PLAIN_REGISTER, REGISTER_SHA256, SOURCE_FIDELITY
from tests.conftest import REPO

SOURCE = REPO / "cankar" / "core" / "register.py"

# Pinned 2026-07-27, the register's first commit. NOT a lock - a session editing
# the block can edit these lines too. It is a tripwire, and its value is the
# failure message: everything generated before the change used a DIFFERENT
# source-side distribution. Same device as UNSTAMPED_ALLOWLIST in
# tests/structure/test_report_freshness.py - two places must change consciously.
PINNED_REGISTER_SHA256 = "cec36eb07e77c6d99076ad98ce176c0c097827f6ecdc70ee274b4547be6d985d"
PINNED_FIDELITY_SHA256 = "2fa1a7f8f750763cb33a4bf727458f29f9167fa9e50b8abdec772a7b0554595d"


def test_register_text_has_not_drifted() -> None:
    """Tripwire, not a lock - see the pins above for why it is only that."""
    assert REGISTER_SHA256 == PINNED_REGISTER_SHA256, (
        "the plain Slovene register changed.\n"
        "Every training pair generated before this commit used the OLD register on "
        "its source side. Mixing them trains the styler on two distributions.\n"
        "If this change is intended: regenerate the pairs, or version the register "
        "and keep the old pairs separate. Then update PINNED_REGISTER_SHA256."
    )


def test_source_fidelity_text_has_not_drifted() -> None:
    """Pinned for the same reason: Phase 5 sends these bullets alongside the
    register, so they are equally part of the de-styling distribution."""
    actual = hashlib.sha256(SOURCE_FIDELITY.encode("utf-8")).hexdigest()

    assert actual == PINNED_FIDELITY_SHA256, (
        "the source-fidelity constraints changed - see PINNED_REGISTER_SHA256's "
        "note; the same staleness applies to any pairs already generated."
    )


def test_register_module_source_is_nfc() -> None:
    """Checks the FILE, not the constant.

    Asserting `PLAIN_REGISTER == normalize("NFC", PLAIN_REGISTER)` would be a
    tautology - the constant is already that function's output, and NFC is
    idempotent, so it passes even for a fully NFD source file (design-review
    2026-07-27; this repo's third vacuous gate).

    It matters because the runtime `normalize()` keeps the SHA stable but not the
    file: the one-home probe is derived from the constant (NFC) and searched for
    in file bytes (as saved). Today the longest lines are pure ASCII so an NFD
    save is harmless, but one edit that makes a caron-carrying line long enough
    and the gate stops matching its own home - reporting `found: []`.
    """
    source = SOURCE.read_text(encoding="utf-8")

    assert source == unicodedata.normalize("NFC", source), (
        f"{SOURCE.name} is not NFC-normalized on disk - re-save it as NFC "
        "(c/s/z-caron NFD is a known failure mode in this repo, CLAUDE.md)"
    )


def test_fidelity_bullets_are_not_in_the_register() -> None:
    """The split is the point: PLAIN_REGISTER describes a register and is valid
    in both directions; SOURCE_FIDELITY constrains a rewrite and only means
    anything when a source passage exists (Phase 5). Folding them back together
    would make the shared block look stronger while a third of it is inert in
    direction (b) - bytes identical, distributions not.
    """
    assert "Meaning preserved exactly" not in PLAIN_REGISTER
    assert "Entities verbatim" not in PLAIN_REGISTER
    assert "Meaning preserved exactly" in SOURCE_FIDELITY
