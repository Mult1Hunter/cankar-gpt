# docs/decisions - architecture decision records

Rows are the QUESTION each record answers - scan here, open one. Records are
topic-based and amended in place (ADR 0023); numbers are never reused, so a
demoted record becomes a tombstone rather than a deletion. `graphify query` finds
live records by title.

**`->` in the Status column means the content now lives elsewhere** - the same
sentinel the record's own Status line carries, and the one
`tests/structure/test_adr_index.py` checks for agreement. A live record that is
only *partly* overtaken keeps its body and stays `live`.

| # | Answers | Status |
|---|---|---|
| 0001 | why nanochat and not nanoGPT or HF Trainer | -> 0016 |
| 0002 | why is a half-finished project public, and what that forbids | live |
| 0003 | why does every change carry ceremony; how provenance is proven | live |
| 0004 | why must every literary doc match a hand-curated ledger, and how misattribution is excluded | live |
| 0005 | why subpackages instead of a uv workspace | -> 0007 |
| 0006 | why a design brief before code; why heuristics need real calibration data | live |
| 0007 | why is there no scripts/ directory and why is the repo root frozen | live |
| 0008 | why StrEnums, typed results, no print in library code | -> `.claude/rules/code-standards.md` |
| 0009 | why merge commits instead of squash | -> CONTRIBUTING.md |
| 0010 | why CI asks for an attestation it cannot verify | -> 0022 |
| 0011 | why vendor the tokenizer seam instead of depending on nanochat; why vocab 8192 | live |
| 0012 | why T=2048 and why chunk_budget must equal max_seq_len | live |
| 0013 | why the held-out set is whole-work and containment-closed, not url-keyed | live |
| 0014 | how works by other authors got out of Cankar's training slice | -> 0004 |
| 0015 | why char n-grams over word features for the style classifier | live |
| 0016 | why AdamW not Muon; why vendor-as-port; why the checkpoint is self-describing | live |
| 0017 | why evals never imports train | -> 0016 |
| 0018 | what the Phase 3 base pretrain actually cost | -> ROADMAP.md, docs/cankar-v1.md |
| 0019 | why the landing page sits in apps/ | -> `.github/workflows/pages.yml`, `test_layout.py` |
| 0020 | why the graphify index rebuilds at session start | -> `.claude/settings.json`, CLAUDE.md |
| 0021 | why the report freshness gate is local and not CI | -> 0022 |
| 0022 | why CI gates ask for attestations; why default-deny scope; why a gate must be a required check | live |
| 0023 | when something deserves an ADR at all | live |
