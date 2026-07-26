#!/usr/bin/env bash
# Design-review gate (ADR 0022): the `design-review` agent pass is mandatory on
# PRs that change machinery, and merely optional on content-only PRs.
#
# CI cannot observe whether an agent actually ran - no artifact proves it. What
# CI CAN do is refuse to let a code PR stay silent about it: if the diff touches
# machinery, the PR-template attestation line must be present and ticked. The
# claim is the reviewer's, the same contract as the roadmap gate (ADR 0010).
#
# Machinery = the paths where a wrong abstraction is expensive to undo. Content
# paths (docs/, registry/, apps/, *.md, .claude/) are deliberately exempt: the
# ritual is for design decisions, not for prose. Widening this list is a
# conscious act - edit CODE_PREFIXES and say so in the PR.
#
# Usage:
#   ops/check-design-review-pr.sh origin/main              # local, before opening a PR
#   ops/check-design-review-pr.sh <base> <head>            # replay a historical PR
#   PR_BODY="..." ops/check-design-review-pr.sh <base-sha> # CI (design-review-gate.yml)
set -euo pipefail

base="${1:?usage: check-design-review-pr.sh <base-ref> [head-ref]}"
head="${2:-HEAD}"

CODE_PREFIXES=('cankar/' 'tests/' 'ops/' '.github/workflows/')

summary() { # human output, teed into the Actions job summary when present
    printf '%s\n' "$1"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        printf '%s\n' "$1" >>"$GITHUB_STEP_SUMMARY"
    fi
}

# capture, not process-substitute: a bad ref inside <(...) escapes set -e and
# would silently pass the gate (same trap the roadmap gate documents).
changed=$(git diff --name-only "$base...$head")

touched=()
while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    for prefix in "${CODE_PREFIXES[@]}"; do
        if [[ "$file" == "$prefix"* ]]; then
            touched+=("$file")
            break
        fi
    done
done <<<"$changed"

summary "## Design-review gate"
summary ""

if ((${#touched[@]} == 0)); then
    summary "No machinery paths touched - design-review is optional on this PR."
    exit 0
fi

summary "Machinery paths touched (${#touched[@]} file(s)):"
for f in "${touched[@]:0:10}"; do summary "- $f"; done
((${#touched[@]} > 10)) && summary "- ... and $((${#touched[@]} - 10)) more"
summary ""

# Attestation is checked in CI mode only; locally this script is a preview.
if [[ -z "${PR_BODY+set}" ]]; then
    summary "PR_BODY unset - local preview, attestation not checked."
    exit 0
fi

attest=$(tr -d '\r' <<<"$PR_BODY" |
    grep -E '^[[:space:]]*[-*+][[:space:]]+\[[xX ]\].*`design-review` agent pass' |
    head -n1 || true)

if [[ -z "$attest" ]]; then
    summary "FAIL: PR body is missing the attestation line - restore the PR-template"
    summary "      item '\`design-review\` agent pass run on the diff ...'"
    exit 1
fi

# anchored on the checkbox itself, so "[x]" later in the line's text cannot pass
if [[ ! "$attest" =~ ^[[:space:]]*[-*+][[:space:]]+\[[xX]\] ]]; then
    summary "FAIL: this PR changes machinery, so the design-review pass is required."
    summary "      Run the \`design-review\` agent on the diff, address must-fixes,"
    summary "      then tick '[x] \`design-review\` agent pass run on the diff'."
    exit 1
fi

summary "Attestation present and ticked."
