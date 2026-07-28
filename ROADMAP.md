# CankarGPT - Roadmap

> A from-scratch Slovene micro-LLM (26.3M params) trained on public-domain literature,
> specialized in Ivan Cankar's voice, extended with plain->Cankar style transfer via
> synthetic parallel data, orchestrated with a large knowledge model, and served for ~€0.
>
> **Hardware:** local RTX 4070 Ti Super 16GB (dev, tokenization, TinyCankar) + rented cloud GPU (long runs, ~$10-40 total)
> **Stack:** Python/uv + PyTorch + nanochat (scaled down), Claude Batch API, FastAPI, Laravel orchestrator, Astro demo page
> **Repo policy:** single public monorepo from commit #1 (github.com/Mult1Hunter/cankar-gpt - ADR 0002). Public-repo
> hygiene (.gitignore, .env.example, gitleaks pre-commit + CI, weights/data on HF Hub or R2 - never in git, milestone tags).

---

## MVP gate

**MVP = Phases 0-4 + eval numbers + static samples page + blog posts 1-2.**
Everything after Phase 4 gets an explicit go/no-go decision based on MVP quality.
Do not build serving or the Laravel orchestrator before the styler exists.

---

## Phase 0 - Environment (half a day)

- [x] uv project init
- [x] PyTorch + CUDA; `torch.cuda.is_available()` verified - torch 2.13.0+cu130
      on the RTX 4070 Ti Super, driver 580 (roadmap said CUDA 12.x; current
      wheels ship 13.0)
- [x] Clone nanochat (sibling checkout); speedrun read end-to-end. Adaptation
      seams mapped: tok_train takes a plain text iterator (narrow - Phase 2),
      pretrain reads FineWeb-parquet via nanochat.dataset (the real B1 work -
      Phase 3); runs/miniseries.sh + runcpu.sh are TinyCankar material
- [x] Watch Karpathy "Let's build GPT" (nanoGPT stays *reading material*, nanochat is the codebase)
- [x] RunPod account + **spending limit set**; the 15-min throwaway pod folds
      into Phase 2.5's $1 cloud rehearsal (same workflow validation, one spend)
- [x] Repo init (public from commit #1), .gitignore, .env.example, gitleaks pre-commit hook + CI
- [x] Roadmap gate: required CI check on every PR - body attestation, newly
      ticked deliverables surfaced in the job summary, flip-down protection
      *(added in-flight - ADR 0010)*

## Phase 1 - Corpus (1-2 sessions)

- [x] Wikivir/Wikisource crawler via MediaWiki API: Cankar + 14 PD authors, attribution/catalog guards (PR #9, #10)
- [x] Works registry as source of truth + dLib.si gap-fill with OCR quality gates *(added in-flight - ADR 0004)*
- [x] Slovenian Wikipedia dump ingestion: 125,670 articles / 65.3M words, streaming
      (ShardWriter already extracted the one shared seam - the write side; the three
      acquisition sides, API/EDM/dump-stream, are genuinely divergent, so no further
      SourceCrawler protocol - that would be one-shape-fits-none)
- [x] Clean (mwparserfromhell), NFC-normalize -> JSONL shards with manifests
- [x] dLib coverage reconciliation (`reconcile-dlib`): 27 recoverable PD works
      against live dLib, 25 pulled through OCR gates (incl. the play Hlapci -
      never on Wikivir). Two review layers corrected the pull before landing:
      corpus-qa caught 3 non-Cankar docs (fixed: creator-based authorship
      check), design-review caught 19 re-editions pulled via discovery
      loop-back (fixed: segment matching + 27 parallel identities merged).
      MinHash marks 2 spelling/disambiguator variants for the merge stage,
      so genuinely new: 23 works / 60k words, author corpus 1.70M -> ~1.76M
      post-merge; registry is ledger-not-gate (ADR 0004 amendment 2);
      committed audit report in registry/reports/
- [x] Misattribution exclusion (`WorkFlag.NOT_BY_AUTHOR`, ADR 0014): texts
      crawled into an author's shard but written by someone else are flagged in
      the registry and dropped at merge (distinct from cross-author works kept
      under their true author via collision_resolution.toml). 5 about-Cankar
      records flagged (2 Vera Albreht memoirs - in copyright, 1 critic's essay,
      2 mis-crawled bibliography pages); 3 were in the merged Cankar slice, gone
      after re-merge (249 -> 246 Cankar docs). Coverage stops counting them;
      evals keeps an independent last-line assertion. Found by the 2.25 holdout
      audit *(added in-flight - ADR 0014)*
- [x] Merge stage (`cankar corpus merge`): quality-gated, deduplicated,
      deterministic merged corpus. Four dedup signals (exact hash, registry
      work-identity, MinHash near-dup, containment); keep-preference by shard
      tier (Wikivir > dLib > gapfill > Wikipedia); cross-author attribution
      via committed collision_resolution.toml (never silent). Built to the
      architect critique (M1-M4); real-pair fixtures pin every signal.
      -> 126,352 docs / 72.33M words (dropped: 9 gate, 7 registry-identity,
      6 exact, 296 near-dup, 48 containment, 5 not-by-author). Registry-identity is content-
      confirmed (max bidirectional containment >= 0.5): a bare work_id match
      was collapsing distinct-year collections ('Črtice 1914' vs '1907-09',
      0.00 overlap) - caught by enumerating the drops, ~54k words recovered,
      mismatches surfaced for registry cleanup. Committed merge report + manifest.
      - [x] Containment drops (user policy 2026-07: keep collected volumes,
            drop contained works). 48 works >0.95 contained in a kept volume
            dropped (text survives in the volume, <=5% residual; ~10.6k words
            of duplication removed); 58 partial (0.80-0.95) kept and reported;
            volumes never dropped. 0.95 boundary pinned on real works (Zapuščeni
            0.956 drop vs Aleš 0.945 keep). Aškerc ballads in 'Balade in
            romance', Kette poems in 'Poezije 1907', etc.
- ⚠️ **Yield correction (measured 2026-07, incl. dLib gap-fill):** Cankar ~1.76M
      words post-merge (1.65M Wikivir + 0.11M dLib after variant drops);
      14 PD authors 6.0M; Wikipedia 65.3M -> corpus ~73.0M words ≈
      **~110-140M tokens**. The original
      ~200-300M general-Slovene target is **unreachable from Wikipedia alone** (~half
      the low end). Options at Phase 3 sizing: accept ~120M tokens (data-constrained;
      a ~15-30M model is the honest compute-optimal-with-repetition size, not 40M -
      see Phase 3 model-size line), or add a general-Slovene source (KAS/Gigafida/CC).
      The Cankar slice (1.65M words) is the true bottleneck - Phase 5 synthetic pairs
      exist to multiply it.
- **Licensing (B5):** publish reproducible corpus-*building scripts*, not the merged corpus
  (Wikipedia is CC BY-SA; Cankar is PD; the merged blob inherits share-alike obligations)
- **Attribution:** README "Data sources" is the canonical credit statement (dLib.si
  citation required by its terms; Wikipedia CC BY-SA; Wikivir contributors) - every
  published artifact (HF datasets/models, demo pages) repeats it

## Phase 2 - Tokenizer (1 session)

- [x] Slovene BPE via nanochat's tokenizer stage (vendored seam - ADR 0011):
      candidates {4096, 8192, 16384} trained deterministically on the full
      merge; **v8192 selected** (26.35M total at depth-6 reference, fertility
      1.77 tok/word on Cankar, Cankar/Wikipedia ratio 0.886 -> weighted-mix
      trigger NOT fired). Numbers: registry/reports/tokenizer-eval.md
- [x] Inspect segmentation of Slovene morphology + Cankar's archaic orthography
      -> README material: probe tables in tokenizer-eval.md (case paradigms
      consolidate at 8k: misli|jo, ža|lost; archaic solnce stays so|l|nce at
      every size - the model sees morpheme-ish units, not memorized words)
- [x] Chunk merged corpus (ADR 0012): 126,355 docs -> 160,065 chunks <= 2048
      tokens (== Phase 3 max_seq_len - re-chunk if that changes). Recursive
      ladder paragraph -> line -> sentence -> hard (hard never fired on real
      data); byte-exact reconstruction + char spans for the 2.25 holdout key;
      simulated BOS-bestfit crop 11.52% vs ~35% FineWeb reference
      (registry/reports/chunks.md)
- [x] Stats report: tokens per source/author -> registry/reports/token-stats.md.
      **142.77M total tokens** (the real Phase 3 budget), Cankar slice 2.92M
      tokens; steps/epoch reference math + parquet-ordering warning recorded

## Phase 2.25 - Evaluation harness (1 session) *(added per Agent A4)*

- [x] Held-out perplexity set (ADR 0013): whole-work, not chapters - chapter
      holdout leaks (model trains on a work's other chapters). Dominant leak on
      THIS corpus is collected-volume containment, so selection is containment-
      closed (a candidate <0.5 contained in every kept doc). Frozen, provenance-
      stamped registry/evals/holdout.json: **50 Cankar prose works, 146,268
      tokens (5.02%)** (same 50 works after the ADR 0014 re-merge; the fraction
      rose from 5.01% as 3 misattributions left the Cankar denominator). BPB
      harness (nanochat metric vendored + drift-tested, own deterministic
      batcher) ships tested against a stub; real numbers at Phase 3.
      registry/reports/eval-holdout.md
- [x] BPB-on-checkpoint wiring (ADR 0017): `cankar evals bpb --checkpoint` loads
      a trained checkpoint via its self-describing gptconfig (no train import) and
      scores the frozen held-out set - so every model gets a real, comparable
      number (invariant #2; the mechanism Phase 4's "record numbers" needs).
      Verified: a 3-step nano scores BPB 3.57 over the 50 works *(added in-flight
      - ADR 0017)*
- [x] Style classifier (ADR 0015): char n-gram TF-IDF + balanced logistic
      regression, Cankar prose vs 14 PD peers (prose-vs-prose, NOT vs Wikipedia
      - critique A-1), wikivir-only, chunk-level, group-split by within-author
      near-duplicate cluster, holdout-excluded. **ROC-AUC 0.993 +/- 0.004**
      (grouped 5-fold). Confound audit (registry/reports/style.md, required
      human gate): top features are punctuation rhythm + function words (VOICE),
      not topic/source/form; funcwords-only floor 0.874 and an orthography
      spot-check (~no AUC change) show the signal is not topic or edition
      spelling. Frozen manifest registry/evals/style.json (versions, seed,
      config, artifact sha - a scorer is not regenerable data). Deploy negative
      (modern de-styled Slovene) is unseen -> `deploy_validated: PENDING Phase 6`
      (MF-3). SloBERTa still deferred - now with a measured AUC to beat.
- [x] **MF-3 RESOLVED, negatively (Phase 6):** `cankar evals deploy-check` measures
      the classifier on the task it is DEPLOYED on rather than the one it was
      trained on. Train-task AUC 0.993; **deploy AUC 0.650** - ranking a real
      Cankar passage above its OWN de-styled pair, 285 held-out pairs, pairwise so
      passage difficulty cancels. Decomposed via design invariant #1, which holds
      register constant between de-styled Cankar and the modern drafts: the topic
      effect (+0.536) is **4.2x** the style effect (+0.128). The classifier is
      substantially a period/topic detector. The training confound audit could not
      have caught this - every peer in its negative is also 1900s prose, so period
      was held constant by construction and period features carried signal without
      ever reading as topic. **No headline style claim may ride on it**;
      directional signal only. registry/reports/style-deploy.md.
- [ ] LLM-judge template for meaning preservation *(deferred to Phase 6 - needs Phase-5 pairs; building now is speculative)*
- [ ] Dev set design: 200 held-out pairs **+ 50 fresh drafts**. Held-out half DONE:
      `cankar pairs segment/destyle --set holdout` -> **289 pairs** from the 12
      held-out prose works, zero passage- AND work-level overlap with the 9,950
      training pairs (`PairSet` inverts one shared predicate, so a doc cannot be
      eligible for both). Fresh drafts remain - they are the other half of the
      honest headline number, and the only way to measure the train/inference gap
      design invariant #1 exists to close
- Rule: every quality claim in README/blog gets a number from this harness
- [x] **Corpus follow-up RESOLVED (ADR 0014):** the 3 about-Cankar
      misattributions the audit flagged (2 Vera Albreht memoirs, 1 critic's
      essay) were the eval-side stopgap; the root fix flags them (plus 2
      mis-crawled bibliography pages) `NOT_BY_AUTHOR` in the registry and drops
      them at merge, so they no longer pollute the Cankar training slice. See
      the Phase 1 misattribution-exclusion line; broadened sweep also caught a
      Murn letter already correctly re-attributed by the collision table.

## Phase 2.5 - TinyCankar micro-win (few hours) *(added per Agent C2)*

- [x] Training stage built (ADR 0016): `cankar/model` (nanochat GPT vendored-as-
      port, behavioral drift-test, AdamW not Muon) + `cankar/train` (Cankar-only
      holdout-excluded loader, lean AdamW loop with warmup+cosine, tokens/sec
      logging, reboot-safe checkpoint/resume, sampler). Frozen holdout contract
      promoted to `core`. Verified end-to-end on the real slice (1520 chunks, 89
      held-out dropped; loss down; ~10k tok/s CPU; sample; checkpoint; resume).
      TinyCankar is now `cankar train run --config configs/train/tinycankar.toml`
      *(added in-flight - ADR 0016; the ~10M GPU run + items below are next)*
- [x] ~10M model (15.2M), Cankar-only: 4000 steps / ~12 epochs on a 4070 Ti
      Super (~2 min), held-out BPB 2.2227 on 50 works. checkpoints/tinycankar.pt
- [x] Save the charmingly broken samples: docs/tinycankar-samples.md
      (step 1 word-salad -> step 4000 coherent Cankar cadence, the "before")
- [ ] **Publish TinyCankar samples** (LinkedIn / blog teaser) - public commitment = project survival
      (repo is public from commit #1; this milestone *promotes* it)
- [x] Tag `v0.1-tinycankar` (git tag exists; `v0.2-cankar-v1` also tagged at Phase 4)

## Phase 3 - Base pretrain (ran LOCAL, $0 - ADR 0018)

- [x] Full-corpus training data (B1, ADR 0018): `CorpusScope` config selects
      cankar|all; the `all` scope keeps every source with the frozen-holdout
      exclusion still applied (tested). 142.78M train tokens / 159,973 chunks.
- [x] **Calibration (B3):** 264k tok/s measured on the 4070 Ti Super;
      checkpoint-resume tested (test_max_hours + the run itself)
- [ ] `setup.sh`: pod -> clone -> deps -> pull data (HF/R2) -> tmux train -> checkpoint-sync loop. Target: <5 min to training, unattended
      *(RESERVED - ADR 0018: Phase 3 ran local at $0; the cloud bootstrap earns its
      keep only for a future run that outgrows the 4070 Ti Super's 16GB)*
- [x] Cost discipline: `--max-hours` self-terminating flag built + validated
      (`cankar/train/loop.py:167`, config field, CLI override). Cloud-ops parts
      (terminate-not-stop, delete-volumes-after-sync) are RESERVED for a future
      cloud run - moot locally (ADR 0018)
- [x] **Model size confirmed (2026-07):** 26.3M params (n_layer 6, n_embd 384),
      in the ~15-30M honest range for a 142.78M-token budget (Chinchilla ~4
      epochs ≈ 24M). Ran 3 epochs, loss 9.0 -> 2.8. **Base pretrain DONE:**
      held-out BPB **1.5056** (vs TinyCankar 2.2227, -32%). checkpoints/base.pt.
- [x] bf16 mixed-precision (`cankar/model/compute.py` auto-detect -> bf16 on the
      4070 Ti Super) + flash attention (`cankar/model/flash_attention.py`, FA3/SDPA
      switch, vendored) shipped and used in the local run. W&B live loss curves
      dropped for local console logging - the blog loss-curve artifact moves to
      Phase 10 capture
- [ ] Model config: RunPod 4090 for experiments, A100 for the final run
      *(RESERVED - ADR 0018: ran local on the 4070 Ti Super; cloud unused)*
- Reference: total compute ~ 50-100x *less* than nanochat's $100 speedrun; budget anxiety = zero

## Phase 4 - Cankar specialization (hours)

- [x] Continued pretraining -> CankarGPT v1 (continuation model): base.pt
      specialized on Cankar via `--init-from` (fresh optimizer, 4x-lower LR,
      ~3 epochs). checkpoints/cankar-v1.pt. docs/cankar-v1.md
- [x] Checkpoint-progression samples ("gibberish becomes Cankar"): the
      three-model arc (TinyCankar word-salad -> base general Slovene -> v1
      Cankar voice) in docs/cankar-v1.md + docs/tinycankar-samples.md
- [x] Eval harness numbers: held-out BPB 2.2227 (TinyCankar) -> 1.5056 (base)
      -> **1.4508 (CankarGPT v1)**. Specialization improved held-out Cankar
      modeling without overfitting.
- [ ] **-> MVP SHIP: static samples page + blog posts 1-2. Go/no-go for everything below.**

## Phase 5 - Synthetic style pairs (1-2 sessions, ~$5-15 API)

- [x] Chunk Cankar into 5-15k passages (2-6 sentences): `cankar pairs segment`
      cuts **13,872** paragraph-bounded passages (4.73M chars) from the 82 wikivir
      prose docs left after excluding 50 held-out works and 73 non-prose ones.
      Own sentence splitter - the chunker's cuts at 28.5% of the corpus's
      ellipses, harmless in a token-budget ladder and corrupting here. Genre is
      read from the committed works ledger: segmenting Cankar's six plays as
      prose produced 1,223 passages with speaker labels glued into the source
      (design-review 2026-07-28). Surplus is what lets the segmenter reject every
      ambiguous candidate - including works whose ledger row records no genre -
      instead of parsing it
- [x] Claude Batch API de-styling -> plain modern Slovene; pair `(plain -> original Cankar)`:
      `cankar pairs destyle` -> **10,043 pairs** from 10,049 responses on
      `claude-sonnet-5` (6.4M in / 1.7M out tokens, batch-priced). Model chosen by
      a 50-passage pilot, not preference: Haiku 4.5 produced broken Slovene
      grammar that the frozen style classifier scored identically (0.693 vs
      0.690) - the metric could not see the only thing that mattered. Resumable
      by construction: content-addressed `custom_id`, raw responses persisted
      before parsing, batch receipts committed at submit time
- [ ] **Distribution-shift fix (A1):** ONE shared "plain Slovene register" prompt, reused verbatim for
  (a) de-styling in training data generation and (b) draft-writing at inference. Non-negotiable design invariant.
  *(the definition + its one-home gate landed early: `cankar/core/register.py`,
  `tests/structure/test_register_home.py`. Unticked on purpose - neither consumer
  exists yet, so "reused verbatim" is not true until (a) and (b) both import
  `PLAIN_REGISTER`)*
- [ ] QA: spot-check 50-100 pairs; auto-filter bottom 5-10% (length-ratio + LLM meaning score)
- [x] Publish dataset to HF Hub (`cankar-parallel`) - target side PD, source side own
      output; standalone contribution. `cankar pairs publish` -> pairs + raw responses +
      manifest + a card generated from the manifests so it cannot drift. **PRIVATE**
      until the Wikivir-transcription licensing question is answered - that is a
      maintainer decision, and private->public is the easy direction
- [x] Content capture (added in-flight): `docs/pairs-samples.md` shows what "Cankar
      style" consists of via real before/after pairs, gated by
      `tests/pairs/test_sample_claims.py` against committed excerpts in
      `registry/datasets/pairs/samples.jsonl`. Blog examples cannot be quietly
      prettified - the gate caught its own author doing exactly that on first run

## Phase 6 - Style-transfer SFT (hours)

- [x] Adapt nanochat's SFT stage to `<plain> ... <cankar> ...` format - built as
      `cankar/train/sft.py` + `sft_loop.py`, using the tokenizer's EXISTING chat
      specials rather than the invented markers sketched here (`<plain>` is not
      in the v8192 vocabulary, so it would fragment into ordinary tokens the
      model must spend capacity learning to read as a boundary)
- [x] **Rehearsal against catastrophic forgetting** *(added in-flight)* - the
      first lr sweep produced a monotonic Pareto frontier and it was read as a
      capacity wall; it was a method error. Replay mixed into the objective
      recovers up to 84% of the BPB damage while held-out pair loss moves ~1%,
      and the best cell strictly dominates the old best on both axes at once.
      Numbers and the two things it does not show: `docs/style-transfer-rehearsal.md`
- [ ] Evaluate on held-out pairs AND fresh drafts (the gap between the two is the honest headline number)
- [ ] Product framing (A3): this is a **prose-poem / črtica styler**, not a poem generator - brand it honestly

## Phase 7 - Serving v1 (1 session)

- [ ] FastAPI sidecar, `/generate`, loads PyTorch checkpoint on **CPU** - deploy on existing VPS (marginal cost €0)
  (~80MB fp16 model, ~200-500MB RAM, 50-200+ tok/s on CPU - no GPU hosting needed)
- [ ] HF Space (free CPU, Gradio) as public mirror + fallback link
- [ ] GGUF/Ollama export: **stretch goal only** (B2 - custom arch/tokenizer may not convert; never promise it)

## Phase 7.5 - MILESTONE: In-browser model (v1.5 flex)

- [ ] ONNX export of the trained model (real work - same risk category as GGUF; timebox it)
- [ ] transformers.js / ONNX Runtime Web integration on the demo site
      (site framework: Phase 8 open decision)
- [ ] ~40MB quantized download, cached; generation fully client-side
- **Payoff:** €0 serving, infinite scale, HN-front-page-proof, and the demo line
  "this Slovene LLM is running in your browser right now"
- Fallback if export fights back: browser demo calls the VPS FastAPI endpoint; ONNX ships later

## Phase 8 - Orchestration + two-tier demo (2-3 sessions, Laravel home turf)

> **Open decision (parked 2026-07, due at this phase's go/no-go): web stack.**
> Challenged: Laravel + Astro + FastAPI = three deployables for a solo project, and
> FastAPI (Phase 7, mandatory) could absorb the orchestration. Options: keep both
> (portfolio claim + static demo), Laravel-only web, cut Laravel (Astro + FastAPI).
> Decide via ADR at gate time; Laravel/Astro lines below and the `apps/web` /
> `apps/orchestrator` rows in ADR 0002 are **provisional** until then.

- [ ] Tier 1 "Piši kot Cankar" - free/unlimited, local model only (browser or VPS), zero marginal cost
- [ ] Tier 2 "Vprašaj Cankarja" - knowledge model (behind an interface - provider-agnostic, per C5) ->
  plain draft in the shared register -> styler -> cleanup pass (restores mangled named entities, per A2)
- [ ] Abuse controls: Cloudflare Turnstile, per-IP rate limits, **daily global spend cap** -> degrades to Tier 1
- [ ] Cost reality: ~$0.001-0.005/generation; 1k uses ~ few $; viral day ~ $20-50 capped
- [ ] Demo UX (C1): 10-second graspability - text box -> output -> one-line "trained from scratch on rented GPUs
  for $15" caption; architecture one click below. Centerpiece of the Astro site relaunch
  at nextgen-solutions.xyz (+ unblocks the LinkedIn post)

## Phase 9 - GaMS v2 (later; separate go/no-go)

- [ ] Same pair dataset. Prototype: LoRA on GaMS-2B (fast loop). Quality: QLoRA on GaMS-9B (fits 16GB w/ Unsloth)
- [ ] Enters HF Transformers + PEFT ecosystem (deliberately second, after from-scratch understanding)
- [ ] **Release as downloadable weights on HF ("run it in Ollama"), not hosted** - live demo stays on the free 40M
  model; v2 is proof of range, not infrastructure ($10-50/mo GPU serverless not worth it for a demo)
- [ ] README answers "why not just GaMS from the start?" up front (C3): from-scratch = full-stack understanding;
  GaMS fine-tune = practical delivery. One project, both claims.

## Phase 10 - Content (runs *during*, not after - C5)

- [ ] Blog 1: the launch post - why build a Slovene LLM from scratch, the three-model arc,
      the honest limits *(drafted; the "rented GPUs for $15" framing is dead - ADR 0018,
      it ran local at EUR 0, which is the better story)*
- [ ] Blog 2: building the corpus - provenance, dedup, and the ADR 0014 misattribution
      incident *(drafted; absorbed the original Blog-1 corpus slot)*
- [ ] Blog 3: two-model orchestration (knowledge model + own styler + Laravel)
- [ ] README: architecture diagram, eval numbers, "why from scratch" section, reproducibility ($10 RunPod path
  AND 16GB-consumer-card path documented - environment-agnostic training script)
- [ ] Disclaimer (C4): AI pastiche, educational, unaffiliated with Cankar institutions - one sentence, mandatory
- [ ] HF uploads: model weights, pairs dataset, (v2 weights later)

---

## Engineering debt (deferrals recorded when their ADR was consolidated - ADR 0023)

- [ ] Fold `token-stats.md` and `tokenizer-eval.md` into the typed `corpus_sha256`
      manifest channel (already used by `chunks.manifest.json` and `holdout.json`),
      retiring the header-regex freshness path for those two
- [ ] `corpus_stamp(sha)` helper in `cankar/core/reports.py`, adopted by the five
      writers (evals/holdout, evals/bpb, tokenizer/chunk, tokenizer/evaluate,
      tokenizer/stats, pairs/segment, pairs/destyle), so the stamp is an exact-line
      match not an 80-char window. Seven writers now, still five phrasings -
      `passages.md` and `pairs.md` reuse `bpb.md`'s exact line - all parse, one by
      4 chars.
      **Blocked on a decision, not on effort:** adopting a canonical stamp means
      regenerating each report, and `eval-holdout.md`'s writer is
      `cankar evals holdout-freeze` - re-running it re-selects the held-out works
      that every BPB number is measured against, which `registry/evals/README.md`
      permits only on a deliberate corpus re-merge. Do this AT the next re-merge,
      or accept a legacy phrasing for that one report
- [ ] Segment the 41 dLib Cankar docs. They carry a median of ZERO blank-line
      paragraphs at 78-char hard-wrapped lines, so `pairs/segment.py`'s paragraph
      unit is absent and recovering it means guessing where breaks were. Skipped
      because wikivir prose alone yields 13,872 passages against a ~10k need - revisit
      only if Phase 6 turns out to be data-starved, never for coverage's sake
- [ ] `ProvenanceStamped` base for the FIVE manifests that all declare
      `schema_version / corpus_sha256 / git_sha / created_at` (core/holdout,
      evals/style, evals/bpb, pairs/segment, pairs/destyle). The drift risk is the
      caller side - `cli.py` now types those fields out five times and forgetting
      one is silent. Phase 5 added two at once, so the trigger has now fired.
      Is-a, not inheritance-for-reuse, and field order keeps the committed JSON
      byte-compatible. Deferred: it touches three frozen artifacts at once
- [x] Committed BPB report/manifest for canonical checkpoints: `cankar evals
      bpb-freeze` -> `registry/evals/bpb.json` + `registry/reports/bpb.md`, with
      per-checkpoint sha256 (the .pt files are gitignored). The published figures
      are gated against it (`tests/evals/test_bpb_claims.py`) - PR #45
- [ ] `ops/lib/attest.sh` - extract when a THIRD gate needs the attestation check

## Risk register (from agent review)

*Codes (A1 ... C5) index the pre-kickoff multi-agent plan review; kept for traceability.*

| # | Risk | Mitigation | Phase |
|---|------|-----------|-------|
| A1 | Train/inference distribution mismatch in style pairs | Shared register prompt; fresh-draft dev set | 5, 6 |
| A2 | Modern vocab/named entities mangled by tiny model | Wikipedia in pretrain mix; cleanup pass restores entities | 1, 8 |
| A3 | "Poem" overpromise (Cankar = prose) | Brand as prose-poem/črtica styler | 6, 8 |
| A4 | No measurable quality claims | Eval harness before long training | 2.25 |
| B1 | nanochat pipeline assumes FineWeb layout | Data-adaptation is its own budgeted deliverable | 3 |
| B2 | GGUF/ONNX export of custom arch may fail | Stretch goals, timeboxed; FastAPI is the primary path | 7, 7.5 |
| B3 | Interrupted runs / unmeasured estimates | Calibration run; tested resume; cloud offloads workstation | 3 |
| B4 | Scope death (solo, evenings) | Hard MVP gate; Laravel last; per-phase go/no-go | gate |
| B5 | Corpus redistribution licensing | Ship scripts, not merged corpus | 1 |
| C2 | Mid-project motivation collapse | TinyCankar micro-win + samples post at 2.5 | 2.5 |
| - | Secrets in public history | Public from commit #1: gitleaks hook + CI, `public-hygiene` skill pre-push | 0 |
| - | API cost abuse on public demo | Turnstile, rate limits, hard daily cap -> Tier 1 degrade | 8 |

## Budget summary

| Item | Cost |
|---|---|
| Pretraining | **EUR 0 actual** - ran local on a 4070 Ti Super (ADR 0018; budgeted $15-40 cloud, never spent) |
| Claude Batch API (pair generation) | $5-15 one-time |
| Serving (VPS already owned + HF free tier + browser) | ~€0/mo |
| Tier-2 demo API usage | capped, ~€0-5/mo |
| **Total to full v1** | **~ one dinner in Ljubljana** |

## Timeline

6-10 weeks part-time. MVP shippable ~week 3-4. Next concrete action: **Phase 1 Wikivir crawler** (cloud-independent, everything downstream feeds on it).

---

## Appendix - repo policy pointers

Layout, visibility, naming: **ADR 0002**, engineering/validation system: **ADR 0003** ,
human workflow (setup, commit types, PR rules): **CONTRIBUTING.md**, pre-push
procedure: `public-hygiene` skill, staged-content rules: `commit` skill.

Private material (notes, plans, deploy inventory, progress files) lives in the sibling
private repo `../cankar-gpt-meta` - never here. Optional symlink
`notes -> ../cankar-gpt-meta/notes` (gitignored); personal ignores that shouldn't
pollute the shared `.gitignore` go in `.git/info/exclude`.
