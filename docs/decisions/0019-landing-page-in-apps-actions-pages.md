# ADR 0019 - public landing page in apps/landing-page, deployed via Actions Pages

**Status:** accepted, 2026-07-26

## Context

The Phase 4 MVP homepage was first placed in `docs/` and served with GitHub Pages
"deploy from a branch" (`main` / `/docs`). That mode only publishes the repo root
or `/docs`, so `/docs` was the path of least resistance. It was the wrong call:
`docs/` is documentation (ADRs now, `cards/` and `blog/` later - its README
contract and ADR 0007), not a place for a public marketing site. The structure
was chosen for deploy convenience instead of correctness, and the placement was
made without surfacing it as a decision.

## Decision

1. **The public static site lives in `apps/landing-page/`.** This sits under the
   `apps/` deployable-frontends root anticipated in ADR 0002, and is deliberately
   distinct from `apps/web` (the future Astro app + browser ONNX, Phase 7.5) and
   `apps/orchestrator` (Laravel, Phase 8). It is plain framework-free static HTML
   (`index.html`, `how-it-works.html`, `style.css`, `CNAME`); no build step.

2. **Served by a GitHub Actions Pages deploy** (`.github/workflows/pages.yml`,
   `upload-pages-artifact` + `deploy-pages`), which can publish any folder - so
   the site is no longer constrained to `/` or `/docs`. Pages `build_type` is
   switched from `legacy` (branch/folder) to `workflow`. The deploy triggers on
   changes under `apps/landing-page/**`.

3. **Custom domain** `cankar-gpt.nextgen-solutions.xyz` stays configured in Pages
   settings; the `CNAME` file travels in `apps/landing-page/` and is served in the
   artifact. `.nojekyll` is dropped - the Actions artifact is served as-is, Jekyll
   never runs.

4. **`apps` is added to the structure-law `ROOT_ALLOWLIST`** (tests/structure/
   test_layout.py) in this PR, per ADR 0007 (a new root entry is a conscious act
   that cites an ADR).

## Consequences

- `docs/` returns to documentation-only; ADR 0002's `apps/` table gains the
  `apps/landing-page` row.
- A brief content lag on `main` is possible right after merge: the old
  branch/`/docs` source no longer finds the site, so `build_type` must flip to
  `workflow` and the Pages workflow must run to restore serving.
- Future web work (`apps/web` Astro, the parked Astro-vs-Laravel decision) is
  unaffected; the landing page is a separate, minimal deliverable.
- Lesson recorded (the ratchet): a placement that touches the structure law is a
  decision to surface, not a convenience to take quietly.
