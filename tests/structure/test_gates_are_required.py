"""Every gate workflow must be a required status check (ADR 0022).

A `*-gate.yml` workflow that is not in the branch ruleset's
`required_status_checks` is decorative: it goes red and the PR merges anyway.
That is exactly what shipped in the first draft of the design-review gate - the
workflow existed, the ruleset did not know about it, and the ADR's claim that it
"fails a PR" was false (design-review 2026-07-26).

This is the repo's own ratchet applied to the ratchets: a gate without
enforcement is a wish.
"""

from __future__ import annotations

import json

import yaml

from cankar.core.paths import repo_root

RULESET = repo_root() / ".github" / "ruleset-main.json"
WORKFLOWS = repo_root() / ".github" / "workflows"


def _required_contexts() -> set[str]:
    ruleset = json.loads(RULESET.read_text(encoding="utf-8"))
    for rule in ruleset.get("rules", []):
        checks = rule.get("parameters", {}).get("required_status_checks")
        if checks is not None:
            return {c["context"] for c in checks}
    raise AssertionError("ruleset-main.json has no required_status_checks rule")


def _gate_job_ids() -> dict[str, list[str]]:
    """job ids per *-gate.yml - the job id is the status-check context name."""
    return {
        path.name: sorted(yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs", {}))
        for path in sorted(WORKFLOWS.glob("*-gate.yml"))
    }


def test_every_gate_workflow_job_is_a_required_check() -> None:
    required = _required_contexts()
    missing = [
        f"{workflow}: job '{job}'"
        for workflow, jobs in _gate_job_ids().items()
        for job in jobs
        if job not in required
    ]

    assert not missing, (
        "gate jobs absent from ruleset-main.json required_status_checks:\n"
        + "\n".join(missing)
        + "\n\nA red gate that does not block a merge is decorative. Add the context, "
        "then apply the ruleset:\n"
        "  gh api -X PUT repos/$REPO/rulesets/<id> --input .github/ruleset-main.json"
    )


def test_the_ruleset_has_at_least_one_gate() -> None:
    """Guards the test above from passing vacuously if the glob stops matching."""
    assert _gate_job_ids(), "no *-gate.yml workflows found - did the naming change?"
