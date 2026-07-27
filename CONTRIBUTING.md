# Contributing

Solo project, but issues and PRs are welcome. Setup, conventions, and guardrails:

## Setup

```bash
uv sync                    # Python >=3.11; installs runtime + dev deps
uv run pre-commit install  # installs pre-commit AND commit-msg hooks
```

## Ground rules

- **Python via uv** - never pip-install globally; run things as `uv run <script>`.
- **Nothing heavy in git** - datasets, checkpoints, and weights live on HF Hub / R2.
  `data/` and `checkpoints/` are gitignored; a pre-commit hook blocks files >1 MB.
- **No secrets** - config via `.env` only (see `.env.example`). gitleaks scans every
  commit locally and every PR in CI.
- **Slovene text is NFC-normalized** at every ingestion point - decomposed č/š/ž
  from past migrations silently corrupt tokenization; use
  `unicodedata.normalize("NFC", text)`.

## Commits

Enforced by the commit-msg hook (`ops/check-commit-msg.sh`):

```
TYPE: imperative summary
```

Allowed types (**uppercase, always**): `FEAT`, `FIX`, `DATA`, `TRAIN`, `DOCS`, `CHORE`.
Examples: `DATA: add wikipedia dump ingestion`, `TRAIN: resume from checkpoint on OOM`.
Optional lowercase scope: `FEAT(corpus): ...`. Keep the summary <=72 chars; add a body
only when the diff doesn't explain itself.

## Style: plain ASCII

Repo prose (docs, commit messages, comments) uses plain ASCII punctuation:
`-`, `->`, `...`, straight quotes. No typographic dashes, arrows, middots, or
emoji. The commit-msg hook enforces this for commit messages. Literary corpus
text (`tests/fixtures/`, `data/`) keeps its authentic typography - the rule
covers our writing, not the literature's.

## Pull requests

After the initial commit, `main` only moves via PR. CI must be green:
lint + format (ruff), config checks, and a full-history secrets scan.

Commits are **thematic**: one topic per commit, conventional format, dependency-
ordered - they land on `main` unchanged via **merge commits** (multi-commit PRs)
or **squash** (single-commit PRs). Your branch commits are public history: no
"wip" - rewrite the series locally first. PR titles follow the commit
convention. Update a branch by rebasing onto `main`; strict status checks require
the branch to be current before merge. Read `main` linearly with
`git log --first-parent`.

Merge over squash for multi-commit PRs is not only about readable history: a merge
commit preserves the SSH-signed branch commits, so their **Verified** badge
survives on `main`. Squashing re-authors the content and drops it.

## Where facts live (canonical homes)

Every fact has exactly one canonical home; everything else is a pointer to it:
**mechanical config** (hooks, CI, `.gitignore`) wherever enforceable -> **this file**
for human workflow -> **`docs/decisions/` ADRs** for decisions -> **ROADMAP.md** for
plan and status (its checkboxes are the canonical phase state). If you find the same
rule stated twice in prose, one copy is wrong - collapse it into a pointer.

## License

MIT - contributions are accepted under the same license.
