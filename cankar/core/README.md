# cankar/core/ - cross-stage contracts

Schemas (CorpusDoc), provenance manifests, path policy (`paths.py` - the only
place artifact locations are defined), shared prompts (`register.py` - design
invariant #1, the one plain-Slovene definition both pipeline directions use),
and frozen ledgers promoted here so sibling stages can read them without
importing each other (`holdout.py`, `works.py` - data model and read helpers
only; the curation logic stays in the owning stage).
Imports NO other cankar module (import-linter enforced). If it knows about a
specific stage, it does not belong here.
