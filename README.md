# CankarGPT

[![CI](https://github.com/Mult1Hunter/cankar-gpt/actions/workflows/ci.yml/badge.svg)](https://github.com/Mult1Hunter/cankar-gpt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> *Mati, ali je model že konvergiral?*

A Slovene micro language model (~40M parameters) trained **from scratch** - custom BPE
tokenizer, pretraining on public-domain Slovenian literature and Wikipedia, then
specialized in the prose voice of **Ivan Cankar** (1876-1918), with a plain->Cankar
style-transfer stage trained on synthetic parallel data.

**Status:** Phase 2.25 (evaluation harness) complete - held-out BPB set + a
confound-audited style classifier (ROC-AUC 0.993) both frozen; corpus merged,
tokenized (v8192, 142.77M tokens) and chunked. Next: base pretrain. See
[ROADMAP.md](ROADMAP.md) for the full plan, risk register, and budget (spoiler:
the whole thing costs about one dinner in Ljubljana).

## Why from scratch, when GaMS exists?

Deliberately both, in order: the from-scratch model demonstrates the full stack
(tokenizer -> pretraining -> SFT -> serving) on consumer/rented hardware; a later
GaMS fine-tune on the same dataset demonstrates practical delivery with modern
open models. One project, two claims. Details in the roadmap.

## Reproducing

Corpus-building scripts are published here (the merged corpus itself is not
redistributed - licensing note in ROADMAP Phase 1):

```bash
uv sync
uv run cankar corpus crawl-wikivir --category "Kategorija:Ivan Cankar" \
    --out data/corpus/cankar.jsonl
```

## Contributing & security

See [CONTRIBUTING.md](CONTRIBUTING.md) (setup, commit conventions) and
[SECURITY.md](SECURITY.md) (private vulnerability reporting).

## Data sources and attribution

Training text comes from public sources; per-document provenance (source URL,
identifiers, retrieval metadata) is recorded in the works registry (`registry/`)
and shard manifests:

- **Wikivir / Slovene Wikisource** (sl.wikisource.org) - hand-transcribed
  public-domain literature, courtesy of the Wikisource contributor community.
- **Digitalna knjiznica Slovenije - dLib.si** (National and University Library
  of Slovenia) - digitized public-domain editions. Only works carrying dLib's
  public-domain rights mark are ingested, and dLib.si is cited here as their
  source, per the dLib terms of use.
- **Wikipedia (Slovenian)** (sl.wikipedia.org) - general-language text from the
  official dump; CC BY-SA 4.0, attributed to Wikipedia contributors (dump
  filename, sha256, and license are recorded in the shard manifest).

The merged training corpus is **not redistributed** (CC BY-SA share-alike vs.
public-domain mixing - see ROADMAP Phase 1). Any published artifact built from
this data (HF datasets, model cards, demo pages) repeats these credits.

## License

Code is [MIT](LICENSE). The merged training corpus is not redistributed; model
weights will be licensed at Hugging Face release time.

---

Built by [Matic Korošec](https://nextgen-solutions.xyz) - training logs and
write-ups land on the site as the project progresses.

*AI pastiche for educational purposes. Not affiliated with any Cankar institution,
memorial, or the Vrhnika tourist board. Cankar's works are in the public domain.*
