# CankarGPT v1 (Phase 4)

CankarGPT v1 is the Phase-3 base model specialized on Cankar. It is
`checkpoints/base.pt` (26.3M params, general Slovene) continued for ~3 gentle
epochs on the Cankar-only scope via `--init-from` - fluent Slovene from the
base, Cankar's voice from the specialization. The `matrix_lr` is 4x below the
base run so the model adapts its register without forgetting general Slovene.

    cankar train run --config configs/train/cankar-v1.toml --init-from checkpoints/base.pt

## Held-out BPB - the three-model arc

Each model is scored on the same 50 held-out Cankar works. "Never in any training
set" is measured, not assumed: containment of all 50 against the full 126,302-doc
training corpus, **both directions, = max 0.0000 with zero hits >= 0.80**. Lower
BPB is better.

| model | params | training | held-out BPB |
|---|--:|---|--:|
| TinyCankar | 15M | Cankar-only, from scratch (Phase 2.5) | 2.2227 |
| base | 26M | full corpus (Phase 3) | 1.5056 |
| **CankarGPT v1** | 26M | base + Cankar specialization (Phase 4) | **1.4508** |

More data cut BPB 32% (TinyCankar -> base); specialization cut another 3.6%
(base -> v1) WITHOUT overfitting the held-out works - the specialization sharpens
Cankar's distribution rather than memorizing it.

## base vs v1 - the same prompt, the register shift

Prompt `Na cesti je` ("On the road there is"), temperature 0.9:

**base** (general Slovene) - coherent but mundane, encyclopedic drift:

> Na cesti je bil tudi trgovec s krvjo in se je še bolj zmenil za trg pa je bil
> jako dobro in je šel čez travnike ter so ga imeli ljudje iz hiše.

**CankarGPT v1** (Cankar-specialized) - narrative, a named character, intimate
physical detail and cadence:

> Na cesti je sedela Marica, prijazno začudena, tako majhna in prijazna, da je
> pokleknila pred njim; iztegnila ga je po pesti za obedve roki, ki je stala za
> njima.

The base reaches for facts; v1 reaches for a person. That register shift is the
whole point of the specialization stage.

## What this is (and is not)

- CankarGPT v1 is the model the MVP samples page and blog will demo. It writes
  fluent, Cankar-flavoured Slovene continuations from a prompt.
- It is a continuation model, not yet the plain->Cankar STYLER (Phase 5-6): it
  does not take a modern-Slovene draft and restyle it - that needs the synthetic
  parallel pairs. v1 is the voice; the transfer of meaning into it comes next.
- Numbers come from the eval harness (invariant #2), not vibes: BPB 1.4508.
