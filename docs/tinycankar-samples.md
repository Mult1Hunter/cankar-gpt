# TinyCankar - the "before" (Phase 2.5, v0.1)

TinyCankar is a ~15M-parameter GPT (n_layer 6, n_embd 256, tokenizer v8192),
trained Cankar-only for ~12 epochs (4000 steps, ~2.77M tokens, held-out works
excluded) on a single RTX 4070 Ti Super - a coffee-break run and a deliberate
over-fit. It is not meant to be good. It is the charmingly broken "before" whose
samples are unrecoverable once the model improves.

**Held-out quality:** BPB 2.2227 bits/byte on 50 unseen works (checkpoint step
4000). A number to beat, not to boast about.

All samples below are unconditional continuations of the prompt `Bilo je`
("It was"), captured live during the run. Nothing is cherry-picked across seeds -
this is one run, the sampler firing every 500 steps.

## The arc: word-salad -> Cankar's cadence

**step 1** (loss 9.01) - pure token soup, Slovene-shaped but meaningless:

> Bilo je naredil tabori visoke vzrokstne-C prizna naše brdelov te oži okrožja
> Vrhvino zasedtlijala menja števila vrsto zmaga obto ... puunt kontromu minut
> odreravaglas črno HC odprta univerze profe bataljongene HCna Pavla naravni

**step 1001** (loss ~2.5) - syntax and a dreamlike register appear:

> Bilo je, da bi se smejala. Da bi se tako zgodilo tako, kakor mehko nositi, da
> bi ga kdo ne videl prej ... Tam so vsi angelski, je bilo celo iz zemlje; iz
> črnih sten so se gonili in zmerom bolj se je videlo nenadno, kakor da je nebo
> v daljavi.

**step 2501** (loss ~1.8) - coherent prose, and it spontaneously slips into
Cankar's *drama* form, with dramatis-personae cues:

> Bilo je, da bi napravilo kaj posebnega. Oči so ji bile vsaj pogledale v licih
> — kdo pa je pravičen? ... POLJANEC: Kje je Damjan? MRVA: O Bog, odrekel se boš

**step 4000** (loss ~1.2) - the "after" of this micro-milestone: coherent
Slovene with dialogue, interiority, and Cankar's emotional cadence:

> Bilo je takrat drugače. In zdaj sem si zaželel še malo ljudi in se ljubezen
> odkriva do srca. Jaz pa nisem vedel, kdaj sem bil sklenil, odgrnil veliko
> zagrinjalo v skrito rame ter skočil z zofe ter si oprijemal dolge poti. Zakaj
> me nisi udaril z vso srčno ljubeznijo: ,Mini, pameten fant me je klical' — si
> je mislil. ,ja bo zmerom!' sem si upal v mraku.

## What this shows (and does not)

- The corpus pipeline, tokenizer, chunking, and training loop work end-to-end:
  a from-scratch model learned to produce fluent, Cankar-flavoured Slovene from
  ~2.77M tokens. That is the milestone.
- It is over-fit by design (12 epochs on a tiny slice) - it echoes and recombines
  Cankar rather than reasoning. BPB 2.22 is a weak absolute number; the point is
  the trajectory from step 1 to 4000, and having a measured floor.
- The real model (Phase 3, ~15-30M params, full pretrain) starts from here.
