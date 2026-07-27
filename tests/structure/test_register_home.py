"""Design invariant #1: the plain Slovene register has exactly one home.

The invariant is stated in CLAUDE.md and ROADMAP Phase 5 as "non-negotiable",
and until `cankar/core/register.py` existed nothing enforced it - the definition
lived only in a private draft. The failure it guards is a FORK: a second copy,
edited independently, so the styler trained against copy A meets copy B at
inference. That failure never raises. It surfaces phases later as "the styler is
mediocre", with nothing pointing at the edit that caused it.

Repo-wide entropy, so it lives here rather than in tests/core/ (tests/README.md).
"""

from __future__ import annotations

from cankar.core.register import PLAIN_REGISTER, SOURCE_FIDELITY
from tests.conftest import REPO, tracked_files

HOME = "cankar/core/register.py"

# Long enough that a probe cannot collide with ordinary prose. Measured against
# every tracked file at introduction: 14 qualifying lines, zero false positives.
MIN_PROBE_CHARS = 40


def _flat(text: str) -> str:
    """Collapse all whitespace runs to single spaces.

    Both sides get flattened, so a copy that REFLOWS the block to different line
    widths still matches. Comparing raw lines would miss that, and reflowing is
    what a human pasting into a doc does by default.
    """
    return " ".join(text.split())


def _probes() -> list[str]:
    """Every substantial line of both constants, flattened.

    All of them, not the single longest: picking one by `max(len)` means an edit
    that shortens that line silently relocates what the gate checks while the
    gate stays green. Checking every long line also catches a partial quote.
    """
    lines = PLAIN_REGISTER.splitlines() + SOURCE_FIDELITY.splitlines()
    return [_flat(line) for line in lines if len(line) >= MIN_PROBE_CHARS]


def test_register_has_exactly_one_home() -> None:
    """A copy of the register anywhere else is a fork waiting to happen.

    This includes docs and blog drafts: prose that QUOTES the block is
    indistinguishable from the block when someone greps for it, and the copy is
    what gets edited. Docs point at the module; they never reproduce it.
    """
    probes = _probes()
    homes = sorted(
        name
        for name in tracked_files()
        if (path := REPO / name).is_file()
        and any(p in _flat(path.read_text(encoding="utf-8", errors="ignore")) for p in probes)
    )

    assert homes == [HOME], (
        f"the plain Slovene register must live in exactly one file, found: {homes}\n"
        f"Invariant #1 (CLAUDE.md): ONE definition, used verbatim for de-styling "
        f"AND for inference drafting. Import it from {HOME} - do not copy it."
    )


def test_the_home_is_tracked() -> None:
    """HOME is compared against git paths, so an untracked or renamed module
    would fail the gate above with a bare `found: []` - true, but it would send
    the reader hunting for a phantom second copy instead of a moved file."""
    assert (REPO / HOME).is_file()
    assert HOME in tracked_files()


def test_probes_are_distinctive() -> None:
    """The gate degrades quietly if the register is ever trimmed to short lines:
    probes get shorter, collide with ordinary prose, and it fails for the wrong
    reason. This repo has shipped three vacuous gates already - guard the guard."""
    probes = _probes()

    assert len(probes) >= 5, f"too few probe lines ({len(probes)}) for a reliable one-home gate"
    assert all(len(p) >= MIN_PROBE_CHARS for p in probes)


def test_probe_paths_are_not_silently_dropped() -> None:
    """`tracked_files()` must hand back paths that resolve, or the gate scans a
    shorter list than it thinks it does - the C-quoting bypass in its docstring."""
    unresolvable = [name for name in tracked_files() if not (REPO / name).exists()]

    assert not unresolvable, f"tracked paths that do not resolve: {unresolvable}"
