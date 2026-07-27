# cankar/core/ - cross-stage contracts

Schemas (CorpusDoc), provenance manifests, path policy (`paths.py` - the only
place artifact locations are defined), shared prompts (`register.py` - design
invariant #1, the one plain-Slovene definition both pipeline directions use).
Imports NO other cankar module (import-linter enforced). If it knows about a
specific stage, it does not belong here.
