---
paths:
  - "ops/**"
  - ".github/workflows/**"
  - ".github/ruleset-main.json"
  - ".pre-commit-config.yaml"
  - "tests/structure/test_gates_are_required.py"
  - "tests/structure/test_report_freshness.py"
---

# CI gates (ADR 0022)

You are editing gate machinery. Read `docs/decisions/0022-ci-gates.md` first.

- **A gate that is not in `required_status_checks` is decorative** - it goes red
  and the PR merges anyway. `tests/structure/test_gates_are_required.py` enforces
  that every `*-gate.yml` job id appears in `.github/ruleset-main.json`. Committing
  the ruleset is not applying it; that needs
  `gh api -X PUT repos/$REPO/rulesets/<id> --input .github/ruleset-main.json`.
- **Attestation, not proof.** CI cannot judge whether a deliverable was completed
  or a design pass happened. Gates force the explicit claim; the claim is the
  reviewer's.
- **`PR_BODY` arrives as an env var, never inline `${{ }}`** - this is a public
  repo and the body is attacker-controlled text.
- Diff scope is **default-deny**: everything needs review unless explicitly exempt.
  Use `git -c core.quotePath=false ... --no-renames`; both were live bypasses.
- A repo-local `pre-push` hook is dead on this machine - a global `core.hooksPath`
  shadows repo hooks and chains only `pre-commit` and `commit-msg`.
