# ADR 0022 - design-review is mandatory on machinery PRs

**Status:** accepted, 2026-07-26

## Context

ADR 0003's per-PR ritual puts a `design-review` agent pass on the diff, and the
PR template carries the checkbox. It is honoured when someone remembers. The two
PRs opened on 2026-07-26 both shipped with the box unticked and a note explaining
why - honest, but the honesty was voluntary, and a PR that simply deleted the
line would have merged clean. The ritual most likely to be skipped is the one
that costs a step before merge, which is exactly the one worth keeping.

CI cannot observe whether an agent ran: no artifact proves it, and any artifact
we invented would be forgeable by the same person choosing to skip the pass.

## Decision

A gate (`ops/check-design-review-pr.sh`, run by `design-review-gate.yml` on
`opened|edited|reopened|synchronize`, and listed in `ruleset-main.json` as a
REQUIRED status check) fails a PR whose diff touches reviewable paths unless the
PR-template attestation line is present and ticked.

Membership is **default-deny**: every changed path counts unless it is exempt.
Exempt are `docs/`, `registry/`, `apps/landing-page/`, `.claude/`, and loose
root-level `*.md`.

## Rationale

- **Attestation, not proof.** The gate enforces that a code PR cannot stay
  *silent* about the pass; the claim is the reviewer's. Same contract as the
  roadmap gate (ADR 0010), and the same honest limit.
- **Default-deny, not an allowlist.** The first draft allowlisted `cankar/`,
  `tests/`, `ops/`, `.github/workflows/` and its own design-review pass killed it:
  that exempts `pyproject.toml`, which holds the import-linter contracts and
  mypy/ruff strictness, plus `.pre-commit-config.yaml` and `ruleset-main.json` -
  the gate machinery itself. Denying by default means a new root entry needs
  review until someone consciously exempts it. `apps/landing-page/` is named
  rather than `apps/`, because ADR 0002 reserves `apps/web` and `apps/api` for
  real code that must not inherit a prose exemption.
- **Required status check, not just a workflow.** A `*-gate.yml` that is not in
  the ruleset goes red while the PR merges anyway. That shipped in the first
  draft, so `tests/structure/test_gates_are_required.py` now asserts every gate
  job id appears in `required_status_checks` - the ratchet applied to the
  ratchets.
- **`core.quotePath=false` and `--no-renames`.** Git C-quotes non-ASCII paths, so
  a Slovene filename arrived as `"cankar/\304\215ebela.py"` and matched no prefix;
  and with rename detection `git mv cankar/x.py docs/x.py` printed only the
  destination, exempting the PR. Both were live bypasses, both are now fixtures
  of the verification below.
- **Its own workflow,** mirroring ADR 0010: it must re-run on `edited` when the
  box is ticked, without dragging the test suite along.
- **Security posture copied verbatim** from the roadmap gate: `PR_BODY` arrives
  as an env var (never inline `${{ }}`, this is a public repo), the base is
  `HEAD^1` on the merge commit, and the diff is captured rather than
  process-substituted so a bad ref cannot escape `set -e`.

## Consequences

- Code PRs now cost one agent pass before merge. That is the point.
- Deleting the checkbox no longer helps - a missing line fails the same as an
  unticked one.
- The ruleset change must be APPLIED, not just committed: `ops/github-repo-setup.sh`
  skips an existing ruleset, so it needs
  `gh api -X PUT repos/$REPO/rulesets/<id> --input .github/ruleset-main.json`.
- The gate cannot tell a real pass from a ticked box. Recorded, not solved:
  the alternative is unforgeable evidence that does not exist.
- Deferred, noted so the third case triggers it: `ops/check-roadmap-pr.sh` and
  this script now duplicate the attestation check and the `summary` helper almost
  character-for-character. At two, copying is defensible; a third gate extracts
  `ops/lib/attest.sh`.
- Lesson (the ratchet): a ritual documented in an ADR and a PR template is still
  a wish until something red-flags its absence - and a gate that is not a
  required check is itself only a wish. This ADR's own first draft proved both
  halves, and its own design-review pass is what caught them.
