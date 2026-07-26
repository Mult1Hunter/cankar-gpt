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
`opened|edited|reopened|synchronize`) fails a PR whose diff touches machinery
unless the PR-template attestation line is present and ticked. Machinery is
`cankar/`, `tests/`, `ops/`, `.github/workflows/`. Content-only PRs - docs,
`registry/`, `apps/`, `.claude/`, loose `*.md` - pass without it.

## Rationale

- **Attestation, not proof.** The gate enforces that a code PR cannot stay
  *silent* about the pass; the claim is the reviewer's. Same contract as the
  roadmap gate (ADR 0010), and the same honest limit.
- **Scoped to machinery** because the ritual is for design decisions, not prose.
  A landing-page copy edit does not need an architect. Widening `CODE_PREFIXES`
  is a conscious act.
- **Its own workflow,** mirroring ADR 0010: it must re-run on `edited` when the
  box is ticked, without dragging the test suite along.
- **Security posture copied verbatim** from the roadmap gate: `PR_BODY` arrives
  as an env var (never inline `${{ }}`, this is a public repo), the base is
  `HEAD^1` on the merge commit, and the diff is captured rather than
  process-substituted so a bad ref cannot escape `set -e`.

## Consequences

- Machinery PRs now cost one agent pass before merge. That is the point.
- Deleting the checkbox no longer helps - a missing line fails the same as an
  unticked one.
- The gate cannot tell a real pass from a ticked box. Recorded, not solved:
  the alternative is unforgeable evidence that does not exist.
- Lesson (the ratchet): a ritual documented in an ADR and a PR template is still
  a wish until something red-flags its absence.
