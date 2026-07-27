"""Shared test helpers.

`tracked_files()` lives here because two independent repo-wide gates need it and
both need it to be correct about the same thing - see its docstring.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def tracked_files() -> list[str]:
    """Every git-tracked path, unquoted.

    `-z` is load-bearing, not style. Plain `git ls-files` C-quotes any path with
    a non-ASCII byte: `docs/zapiski-ž.md` comes back as the 22-char literal
    `"docs/zapiski-\\305\\276.md"`, quote characters included. A gate that then
    does `(REPO / name).is_file()` silently drops it, so anything hidden behind a
    caron is invisible to every check built on this - in a Slovene-literature
    repo. `-z` also removes the newline-in-filename case for free.

    This exact class was a live bypass twice before (.claude/rules/gates.md), and
    a third time in the first draft of the register gate.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return [name for name in out.stdout.split("\0") if name]
