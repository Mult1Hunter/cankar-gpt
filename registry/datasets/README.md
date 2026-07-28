# registry/datasets/ - committed provenance manifests

One manifest per generated shard/dataset (per stage subdir), written by the
generating command: git SHA, args, counts, content sha256, sanity band. This is
what makes "regenerate and diff" (ADR 0003) real - manifests in gitignored
data/ would prove nothing.

`pairs/batches.jsonl` is not a manifest but a RECEIPT log: one line per Batch
API submission, written the moment an id is returned. Batch results live
server-side for weeks, so the id is the only route back to output already paid
for - it is committed before the wait, not after it.
