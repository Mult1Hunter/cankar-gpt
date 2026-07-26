# docs/

`decisions/` - ADRs, the only place layout/architecture law changes.
`index.html` + `how-it-works.html` - the public homepage and its plain-language
explainer; `style.css` is their shared stylesheet. GitHub Pages serves this dir
at cankar-gpt.nextgen-solutions.xyz. `CNAME` + `.nojekyll` are its Pages config.
Later: `cards/` (generated HF model/dataset cards, Ph2.5+), `blog/` (captured
artifacts, Ph4+). Session scratch never lands here (goes to the private meta
repo).
