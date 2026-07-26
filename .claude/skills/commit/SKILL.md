---
name: commit
description: Commit workflow for this repo - staged-content safety checks, enforced message format, milestone tagging, branch discipline. Use whenever committing changes.
---

# Commit workflow

This repo is **public** - every commit is publicly visible once pushed. Slow down.

## 1. Review what's staged

- `git status` + `git diff --staged`. Stage files deliberately; never `git add -A`
  without reading the resulting list.
- **Hard blocks** - unstage immediately if present:
  - anything under `data/` or `checkpoints/`, or matching
    `*.pt *.bin *.safetensors *.gguf *.onnx`
  - `.env` or any credential-bearing file (only `.env.example` belongs in git)
  - personal notes / session scratch (those live in `../cankar-gpt-meta`)

  (This list deliberately mirrors `.gitignore` - defense-in-depth against
  force-adds. When one changes, sync the other.)

## 2. Message format (mechanically enforced by the commit-msg hook)

`TYPE: imperative summary` - types **always UPPERCASE**:
`FEAT` `FIX` `DATA` `TRAIN` `DOCS` `CHORE` (optional lowercase scope: `FEAT(corpus):`).

- Summary <=72 chars, imperative mood ("add", not "added").
- Body only when the diff doesn't explain itself; reference ADRs
  (`docs/decisions/NNNN`) for design choices.
- **Never use `--no-verify`.** If a hook fails, fix the cause, not the messenger.

## 3. Branch discipline

After the initial commit, `main` moves only via PR (CI: lint + secrets scan).
Branch names: `<type>/<slug>`, e.g. `data/wikipedia-ingest` - branch names stay
lowercase (git/URL convention); only commit types are uppercase.

**Committing vs opening a PR (user rule, 2026-07):** commit freely on branches
without asking. But **opening a PR and/or enabling auto-merge requires explicit
user confirmation every time** - never `gh pr create` or `gh pr merge` on your
own initiative. Do the work, commit it, run the design-review pass, then ASK
before the PR.

**Thematic commit series (ADR 0009):** one topic per commit, dependency-ordered,
each coherent on its own - branch commits LAND ON MAIN via merge commits, so no
"wip" commits; rewrite the series locally before opening the PR. Multi-commit
PRs merge with `gh pr merge --merge`; single-commit PRs may `--squash`. PR
titles still follow `TYPE: summary` (they title merge/squash commits). PR bodies:
plain ASCII, no session links, no AI attribution, ever. Update branches by
rebasing onto `main` (strict status checks require the branch to be current).

## 4. Milestone tags

At phase completion (see ROADMAP.md) tag the merge commit: `v0.1-tinycankar`,
`v0.2-base`, ... Tags anchor blog posts. Push tags only after the
`public-hygiene` skill passes.

## 5. ROADMAP tracking (mandatory)

ROADMAP checkboxes are the canonical status. Two triggers:

- **Deliverable completed** (any PR): tick its `- [ ]` in ROADMAP.md IN THE SAME
  PR - reference the PR number on the line. Work added in-flight (not on the
  roadmap) gets a new checked line marked *(added in-flight - ADR NNNN)*.
  The PR template carries this as a mandatory checklist item.
- **Phase completed**: additionally sync the README status line in the same
  commit. CLAUDE.md no longer mirrors phase status - that block was deleted for
  rotting against the checkboxes it declared canonical; ROADMAP is the only
  place status lives.
