"""ADR index completeness and template shape (ADR 0023).

Deliberately NOT a line-count gate. The `adr` skill suggests a 25-40 line shape,
but the overflow in the long records is the case-study material - "a bibliography
detector shipped against synthetic tests and amputated ~150 poems" (ADR 0006) is
why the calibration rule gets obeyed. A budget test would force exactly the
trimming that destroys the value, and is gamed by reflowing prose into fewer,
longer lines. So this checks what is cheap and non-gameable instead.

Both assertions below are anchored parses, not substring searches. The first draft
used `f.name[:4] in index_text` and `section in text`, and a design-review pass
found both vacuous: five rows were self-protecting because their numbers appeared
in OTHER rows' Status cells, and an ADR satisfied the section check through prose
that merely mentioned the heading names.
"""

from __future__ import annotations

import re
from pathlib import Path

from cankar.core.paths import repo_root

DECISIONS = repo_root() / "docs" / "decisions"
INDEX = DECISIONS / "README.md"

# Sections the template makes load-bearing. Rationale is optional - several early
# records fold it into Decision, and splitting them retroactively would be churn.
REQUIRED_SECTIONS = ("## Context", "## Decision", "## Consequences")

# `->` is THE sentinel for "this record's content lives somewhere else", in both
# the ADR's own Status line and its index row. It is deliberately one token: the
# previous word-list ("withdrawn", "merged into") did not recognise the `adr`
# skill's own prescribed phrasing, so a record written to spec read as live.
DEAD = "->"

_ROW_RE = re.compile(r"^\|\s*(\d{4})\s*\|", re.MULTILINE)
_DESTINATION_RE = re.compile(r"->\s*(\S)")


def _adr_files() -> list[Path]:
    # explicit 4-digit glob: `0*.md` goes blind at ADR 1000 and would silently
    # narrow every gate in this module.
    return sorted(DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md"))


def _indexed_numbers() -> set[str]:
    """Numbers in the leading cell of a table row - not anywhere in the file."""
    return set(_ROW_RE.findall(INDEX.read_text(encoding="utf-8")))


def _index_rows() -> dict[str, str]:
    """number -> Status cell. Split on `|` so a Decision cell containing a pipe
    cannot be mistaken for the Status cell, and short malformed rows are ignored."""
    rows: dict[str, str] = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= 3 and re.fullmatch(r"\d{4}", cells[0]):
            rows[cells[0]] = cells[-1]
    return rows


def _status_line(path: Path) -> str | None:
    m = re.search(r"^\*\*Status:\*\*(.*)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1) if m else None


def test_index_lists_exactly_the_adrs_that_exist() -> None:
    """Set equality, so a deleted row and a ghost row both fail.

    A substring check would let any number mentioned in another row's Status cell
    protect itself - which is most of them, and gets worse as the table gets more
    useful.
    """
    indexed = _indexed_numbers()
    actual = {f.name[:4] for f in _adr_files()}

    assert indexed == actual, (
        "docs/decisions/README.md is out of sync with docs/decisions/:\n"
        f"  missing rows:  {sorted(actual - indexed)}\n"
        f"  ghost rows:    {sorted(indexed - actual)}\n"
        "Every ADR needs a row; the row is the entry point."
    )


def test_adr_numbers_are_unique_and_contiguous() -> None:
    """Numbers are permanent: hundreds of `ADR NNNN` citations point at them.

    A withdrawn record keeps its number and becomes a tombstone (ADR 0023) - never
    a hole, and never a renumber, which would break every inbound citation.
    """
    numbers = [int(f.name[:4]) for f in _adr_files()]

    assert len(numbers) == len(set(numbers)), f"duplicate ADR number: {numbers}"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"ADR numbering has a gap: {numbers}. Do NOT renumber to close it - "
        "tombstone the missing number instead."
    )


def test_every_adr_has_a_status_and_the_template_sections() -> None:
    """Anchored to line starts, so prose naming a heading does not satisfy it."""
    broken: list[str] = []
    for f in _adr_files():
        text = f.read_text(encoding="utf-8")
        if not re.search(r"^\*\*Status:\*\*", text, re.MULTILINE):
            broken.append(f"{f.name}: no Status line")
        for section in REQUIRED_SECTIONS:
            if not re.search(rf"^{re.escape(section)}", text, re.MULTILINE):
                broken.append(f"{f.name}: missing {section}")

    assert not broken, "ADRs deviating from the template:\n" + "\n".join(broken)


def test_tombstoned_adrs_say_so_in_the_index() -> None:
    """Supersession symmetry: the file and the index must agree.

    The index Status cell is where a reader scans for what is still live; the ADR's
    own Status line is what a reader sees when they land cold via grep or a code
    citation. Drift between them makes one of the two lie.

    Note `->` is required, not merely "a word meaning dead". A partly-superseded
    record that is still authoritative keeps its body and does NOT carry `->` -
    that is the distinction between amendment and tombstone (ADR 0023).
    """
    rows = _index_rows()

    mismatched: list[str] = []
    for f in _adr_files():
        status = _status_line(f)
        if status is None:
            continue  # already reported by the template test
        file_is_dead = DEAD in status
        index_is_dead = DEAD in rows.get(f.name[:4], "")
        if file_is_dead != index_is_dead:
            mismatched.append(
                f"{f.name}: file says {'dead' if file_is_dead else 'live'}, "
                f"index says {'dead' if index_is_dead else 'live'}"
            )

    assert not mismatched, "ADR status disagrees with its index row:\n" + "\n".join(mismatched)


def test_every_tombstone_names_where_its_content_went() -> None:
    """Content loss is the whole risk of demoting a record, so gate the pointer.

    `**Status:** withdrawn` with nothing after it passes every other check in this
    module while telling a reader nothing. A destination - a path or an `ADR NNNN` -
    is the minimum that makes a tombstone useful.
    """
    undirected = [
        f.name
        for f in _adr_files()
        if (status := _status_line(f)) and DEAD in status and not _DESTINATION_RE.search(status)
    ]

    assert not undirected, (
        "tombstones with no destination after `->`:\n"
        + "\n".join(undirected)
        + "\nSay where the content went, or the record is a dead end."
    )
