# apps/landing-page/

The public static homepage for CankarGPT, served by GitHub Pages at
cankar-gpt.nextgen-solutions.xyz (custom domain; `CNAME` lives here).

- `index.html` - the samples page (the "Mati je" fork, the model arc, eval
  numbers, versions, roadmap, glossary, sources).
- `how-it-works.html` - a plain-language explainer for non-technical readers.
- `style.css` - the shared stylesheet for both pages.

Framework-free static HTML, no build step. Deployed by
`.github/workflows/pages.yml` (GitHub Actions -> Pages), which publishes this
folder directly - folder-based Pages only allows `/` or `/docs`, which is why the
site is not in `docs/` (that is documentation). Distinct from `apps/web` (the
future Astro app, ADR 0002). See ADR 0019.
