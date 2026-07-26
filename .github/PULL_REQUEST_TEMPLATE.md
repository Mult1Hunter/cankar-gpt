## What

<!-- One paragraph. Link ADRs (docs/decisions/NNNN) for design choices. -->

## Validation ladder (ADR 0003 - tick what applies)

- [ ] Design brief posted before implementation (non-trivial code - ADR 0006)
- [ ] `design-review` agent pass run on the diff; must-fixes addressed
      **(mandatory unless the PR is pure content - ADR 0022)**
- [ ] L0 - hooks/CI green
- [ ] L1 - tests added/updated for changed behavior
- [ ] L2 - data contracts / `MANIFEST.json` updated and diffed (data PRs)
- [ ] L3 - eval harness numbers attached (any `TRAIN:` PR claiming quality)
- [ ] **ROADMAP.md checkboxes ticked for every deliverable this PR completes (mandatory)**
- [ ] `public-hygiene` pass run before push

## Phase

<!-- Which ROADMAP phase this serves. If it completes a phase: CLAUDE.md
     "Current phase" + README status updated in this PR (commit skill section 5). -->
