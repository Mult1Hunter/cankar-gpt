---
language:
- sl
license: other
license_name: mixed-see-card
pretty_name: CankarParallel
size_categories:
- 10K<n<100K
task_categories:
- text2text-generation
tags:
- slovene
- style-transfer
- ivan-cankar
- public-domain
---

# CankarParallel - plain Slovene to Ivan Cankar's prose voice

9,950 parallel passages for Slovene literary style transfer. Each row
pairs a passage of Ivan Cankar's prose with a plain modern Slovene rendering of
the same content.

Built for [CankarGPT](https://github.com/Mult1Hunter/cankar-gpt), a from-scratch
Slovene micro-LLM. The pairs train the direction `plain -> cankar`.

## Fields

- `passage_id` - sha256 of the Cankar text, first 16 hex chars
- `cankar` - the original passage, **unmodified public-domain text**
- `plain` - a plain modern Slovene rendering, **generated**
- `title`, `url` - the source work
- `model`, `in_tokens`, `out_tokens` - generation provenance

## How it was made

The Cankar side comes from Wikivir (Slovene Wikisource), segmented into 2-6
sentence paragraph-bounded passages. Only prose genres were used - Cankar's
plays are excluded, because a paragraph in drama is a speaker turn and
segmenting them as prose glues speaker labels into the text.

The plain side was generated with `claude-sonnet-5` via the Batch API, using one
fixed "plain Slovene register" prompt (sha256 `cec36eb07e77c6d9`)
shared with the inference-time drafting stage, so the styler never meets a
distribution at inference it was not trained on.

Model choice was measured, not assumed. A 50-passage pilot against a cheaper
model found broken Slovene grammar - wrong dual verb forms, gender
disagreement, a reflexive-only verb used transitively - which a char-ngram
style classifier scored identically to the stronger model's output. On a
low-resource language the cheap option corrupts the data invisibly.

9,950 of 10,049 responses passed the
quality filters. Rejections are quarantined, never silently dropped.

## Held-out works are excluded

Passages from the frozen held-out evaluation set (holdout sha256
`07ccca96a4fe339d`) were excluded **before** generation. Pairs
built from held-out works would contaminate any evaluation of a model trained
on them, one-way and undetectably.

## Licensing - read before redistributing

Ivan Cankar died in 1918, so **his text is public domain**. The `plain` side is
machine-generated output.

The open question is whether the Wikivir *transcriptions* carry terms of their
own. That is unresolved, which is why this dataset is not yet released for
redistribution. Treat it as reference material until the card says otherwise.

## Reproducing

```
uv run cankar pairs segment
uv run cankar pairs destyle --parse-only   # re-derives pairs from the raw log
uv run cankar pairs publish
```

The middle step re-parses committed raw responses and costs nothing. Generating
them from scratch is `destyle --limit N`, which submits paid batches.

Provenance: corpus sha256 `d9b05bf04db96db6`, passages sha256
`e1cad3a2643782f9`, pairs sha256 `c387b33c7d036b95`,
generated at 2026-07-28T12:39:40+00:00 from git `acf0d2c`.
