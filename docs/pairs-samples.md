# What "Cankar style" actually is

CankarGPT's job is to take plain modern Slovene and write it in Ivan Cankar's
voice. To train that, we first had to go the other way: take 10,043 passages of
real Cankar and strip the style out, leaving the meaning intact.

That inverse direction turns out to be the clearest picture of what the style
*consists of*. Below are real pairs from the frozen dataset - not retyped, not
tidied. `tests/pairs/test_sample_claims.py` fails if any excerpt on this page
stops matching `registry/datasets/pairs/samples.jsonl`.

## The vocabulary layer

The most visible thing Cankar's prose does is reach for words that had already
aged by 1900, and which no Slovene speaker writes today:

| Cankar (1900s) | plain modern | in English |
|---|---|---|
| `duri` | `vrata` | door |
| `dacar` | `davkar` | tax collector |
| `romal` | `hodil` | walked, journeyed |
| `na vekomaj` | `za vedno` | forever |
| `izpregovoril` | `izrekel` | uttered |
| `zakaj` (conjunction) | `namreč`, `kajti` | for, because |
| `tedaj` | `takrat` | then |
| `strmel` | `zrl` | stared |

`zakaj` is the interesting one. In modern Slovene it means only "why". Cankar
uses it constantly as "for/because" - a sentence-opening conjunction that
carries much of his cadence, and that a modern reader now trips over.

## Three real pairs

**Za križem** - the archaism `na vekomaj` collapses to plain `za vedno`, and a
piled-up clause is unwound:

> **Cankar:** Zdaj pa, ko bom že skoro sit na vekomaj, zdaj mi kažejo rumen kolač.
>
> **plain:** Zdaj pa, ko bom kmalu sit za vedno, mi zdaj kažejo rumen kolač.

**Zgodbe iz doline šentflorjanske** - `dacar` modernises to `davkar`, and the
archaic sentence-initial `Zakaj` becomes an ordinary `namreč` moved inside the
clause, which is where modern Slovene puts it:

> **Cankar:** Zvečer so sedeli možje v krčmi in so ugibali; celo dacar je rajši molčal, nego da bi izpregovoril prvo besedo.
>
> **plain:** Zvečer so moški sedeli v krčmi in ugibali; tudi davkar je raje molčal, kot da bi izrekel prvo besedo.

**Mimo življenja** - `duri` becomes `vrata`, and the inverted `Vse temno je
bilo` is straightened to `Vse je bilo temno`:

> **Cankar:** Odprla je duri, šla je skozi temno kuhinjo in vežo, in po stopnicah navzdol. Vse temno je bilo;
>
> **plain:** Odprla je vrata, šla je skozi temno kuhinjo in vežo, potem po stopnicah navzdol. Vse je bilo temno;

## What this tells us about the hard part

Notice how much survives. Every event, every name, every emotional beat is
still there - that was a hard constraint on the rewrite, because a pair whose
two sides mean different things teaches the styler to hallucinate.

So the style Phase 6 has to learn is not the content and not the plot. It is
the vocabulary layer above, the inverted word order, the sentence-initial
conjunctions, and the rhythm of clauses piling up before a short one lands.
That is a narrow target, which is the reason a 26M-parameter model has any
chance at it.

## Why the plain side is machine-generated

Nobody has written modern-Slovene translations of Cankar; the corpus had to be
built. The plain side was produced by `claude-sonnet-5` under a fixed register
prompt - the *same* prompt the inference-time drafting stage will use, so the
styler never meets a distribution at inference it was not trained on.

A cheaper model was tried first and rejected on evidence, not preference: it
produced broken Slovene grammar - wrong dual verb forms, gender disagreement, a
reflexive-only verb used transitively - which our own style classifier scored
identically to the good output. On a low-resource language, the cheap option
corrupts the training data invisibly.

Full dataset: `registry/datasets/pairs/pairs.manifest.json`.
