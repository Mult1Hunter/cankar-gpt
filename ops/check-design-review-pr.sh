#!/usr/bin/env bash
# Design-review gate (ADR 0022): the `design-review` agent pass is mandatory on
# PRs that change machinery, and merely optional on content-only PRs.
#
# CI cannot observe whether an agent actually ran - no artifact proves it. What
# CI CAN do is refuse to let a code PR stay silent about it: if the diff touches
# machinery, the PR-template attestation line must be present and ticked. The
# claim is the reviewer's, the same contract as the roadmap gate (ADR 0010).
#
# DEFAULT-DENY: every changed path needs a review pass unless it is on the content
# exemption list below. An allowlist of "code" prefixes was the first design and it
# was wrong (design-review 2026-07-26): it exempted pyproject.toml, which holds the
# import-linter contracts and mypy/ruff strictness, plus .pre-commit-config.yaml and
# .github/ruleset-main.json - the gate machinery itself. Denying by default means a
# NEW root entry needs review until someone consciously exempts it, which is the
# safe direction; tests/structure/test_layout.py freezes the root allowlist anyway.
#
# Content = prose and generated ledgers, where the ritual buys nothing.
#
# Usage:
#   ops/check-design-review-pr.sh origin/main              # local, before opening a PR
#   ops/check-design-review-pr.sh <base> <head>            # replay a historical PR
#   PR_BODY="..." ops/check-design-review-pr.sh <base-sha> # CI (design-review-gate.yml)
set -euo pipefail

base="${1:?usage: check-design-review-pr.sh <base-ref> [head-ref]}"
head="${2:-HEAD}"

# apps/landing-page is named explicitly, not apps/: ADR 0002 reserves apps/web and
# apps/api for real code, which must NOT inherit a prose exemption when they land.
CONTENT_PREFIXES=('docs/' 'registry/' 'apps/landing-page/' '.claude/')

summary() { # human output, teed into the Actions job summary when present
    printf '%s\n' "$1"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        printf '%s\n' "$1" >>"$GITHUB_STEP_SUMMARY"
    fi
}

# capture, not process-substitute: a bad ref inside <(...) escapes set -e and
# would silently pass the gate (same trap the roadmap gate documents).
#
# core.quotePath=false: git C-quotes non-ASCII paths by default, so a Slovene
# filename arrives as "cankar/\304\215ebela.py" - leading double-quote included -
# and never matches a prefix test. In this repo that is a live foot-gun, not a
# hypothesis.
# --no-renames: with rename detection, `git mv cankar/x.py docs/x.py` prints only
# the DESTINATION, so moving code out of a reviewed path would exempt the PR.
changed=$(git -c core.quotePath=false diff --no-renames --name-only "$base...$head")

touched=()
while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    exempt=0
    for prefix in "${CONTENT_PREFIXES[@]}"; do
        if [[ "$file" == "$prefix"* ]]; then
            exempt=1
            break
        fi
    done
    # loose root-level *.md (README, ROADMAP, CLAUDE.md, ...) is prose too
    if [[ "$exempt" -eq 0 && "$file" != */* && "$file" == *.md ]]; then
        exempt=1
    fi
    [[ "$exempt" -eq 1 ]] || touched+=("$file")
done <<<"$changed"

summary "## Design-review gate"
summary ""

if ((${#touched[@]} == 0)); then
    summary "Content-only PR - design-review is optional here."
    exit 0
fi

summary "Reviewable paths touched (${#touched[@]} file(s)):"
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
    summary "FAIL: this PR changes reviewable paths, so the design-review pass is required."
    summary "      Run the \`design-review\` agent on the diff, address must-fixes,"
    summary "      then tick '[x] \`design-review\` agent pass run on the diff'."
    exit 1
fi

summary "Attestation present and ticked."
