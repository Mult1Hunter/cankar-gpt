# docs/

`decisions/` - ADRs, the only place layout/architecture law changes.
`pairs-samples.md` / `cankar-v1.md` / `tinycankar-samples.md` - published
samples. `pairs-samples.md` has its excerpts CI-gated; the other two have their
BPB figures gated (`test_bpb_claims.py`) but their prose samples are ungated.
`style-transfer-rehearsal.md` - the Phase 6 calibration record: why the first
sweep's "capacity wall" was a method error, and the grid behind every constant
in `configs/train/styler-v1.toml`. Numbers are a snapshot, not CI-gated.
Later: `cards/` (generated HF model/dataset cards, Ph2.5+), `blog/` (captured
artifacts, Ph4+). Session scratch never lands here (goes to the private meta
repo). The public website is NOT here - it lives in `apps/landing-page/`
(ADR 0019).
