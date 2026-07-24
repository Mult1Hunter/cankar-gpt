"""Freeze the held-out Cankar set before Phase 3 (ADR 0013).

Held-out BPB is only honest if the eval text is genuinely unseen. On THIS
corpus the dominant leak is NOT chapter-siblings but collected-volume
containment: the merge kept both individual crtice AND the volumes that
contain them (registry/reports/merge.md), and one Cankar work even appears
under both a Wikivir and a dLib url. So whole-work holdout by url is not
enough - a candidate whose text also lives inside a kept volume (or a
cross-source twin) would be scored over text the model trained on
(architect critique MF-1/MF-2).

Selection therefore runs a containment-closure in BOTH directions:
- FORWARD: a candidate is CLEAN only if it is < CONTAINMENT_REJECT contained
  in every other kept doc (its text is not a chapter of a kept volume);
- REVERSE (design-review 2026-07): a held-out work may itself CONTAIN a
  separately-published excerpt that stays in training - dropping only the
  held-out url would leave that excerpt reproducing the held-out text. So the
  freeze also records `also_exclude_urls`: every other doc >= CONTAINMENT_REJECT
  contained in a held-out work. The Phase 3 filter drops both sets.

The forward claim is precisely "no SINGLE other doc contains >=0.5 of a
held-out work" - max-over-single-doc, not union (union over-rejects on shared
Cankar idiom); the audit report lists containers so a human can confirm.

Candidacy (architect critique A-2/A-5): Wikivir prose only (dLib is OCR),
medium length band (excludes tiny verse and giant volumes). The committed
manifest is human-auditable - the last line of defence against a stray
play or poem is a person reading registry/evals/holdout.json.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from pydantic import BaseModel

from cankar.core.errors import CankarError
from cankar.core.jsonl import iter_jsonl_docs
from cankar.core.manifest import load_frozen
from cankar.core.reports import generated_marker, write_report
from cankar.core.textsim import containment, shingles

log = logging.getLogger("cankar.evals")

CANKAR_AUTHOR = "Ivan Cankar"
# transcription, not dLib OCR (critique A-2). Value matches corpus Source.WIKIVIR
# (cankar/corpus/registry.py) - that StrEnum is a promote-to-core candidate now
# that evals is a second cross-boundary consumer (deferred: 11-file refactor).
HOLDOUT_SOURCE = "wikivir"
# Corpus MISATTRIBUTIONS surfaced by the holdout audit (2026-07): Wikivir texts
# ABOUT Cankar by others (two Vera Albreht memoirs, a critic's essay, two
# mis-crawled bibliography pages) that carried author="Ivan Cankar". The ROOT
# fix lives in the corpus stage - they are WorkFlag.NOT_BY_AUTHOR in
# registry/works/cankar.jsonl and merge.py excludes them (ADR 0014). This set is
# the eval stage's independent LAST line of defence (the module ethos): evals
# reads the merged corpus, never the corpus registry, so it cannot see the flag
# - if a merge bug re-admits one, cankar_docs() fails loud instead of silently
# scoring seen text. It mirrors every Wikivir NOT_BY_AUTHOR url; the mirror is
# not left to memory - test_eval_set_mirrors_registry_flags fails if the two
# drift (mechanize over remember). Defensive post-condition, not a candidacy
# filter, so the two gate-dropped bibliography urls are harmless padding here.
MISATTRIBUTED_URLS = frozenset(
    {
        "https://sl.wikisource.org/wiki/Kulturni_pomen_Ivana_Cankarja",
        "https://sl.wikisource.org/wiki/Nekaj_mladostnih_spominov_na_Ivana_Cankarja",
        "https://sl.wikisource.org/wiki/Iz_prvih_spominov_na_Ivana_Cankarja",
        "https://sl.wikisource.org/wiki/Vera_Albreht",
        "https://sl.wikisource.org/wiki/Izidor_Cankar",
    }
)
# Medium prose band, calibrated on the real Cankar slice (median doc 8,211
# chars): the lower bound drops tiny verse, the upper bound drops novels and
# collected volumes so no single work dominates and containers are not held
# out (critique A-3/A-5). Audited against the generated set (ADR 0006).
MIN_CHARS = 3_000
MAX_CHARS = 45_000
# A candidate this-fraction contained in any other kept Cankar doc leaks and
# is rejected. 0.5 matches the merge's registry-confirm threshold (merge.py).
CONTAINMENT_REJECT = 0.5
# ~5% of the 2.92M-token Cankar bottleneck: enough for a stable summed-bytes
# BPB, small enough that Phase 4 barely feels it (critique A-3).
TARGET_TOKEN_FRACTION = 0.05
MIN_WORKS = 8  # BPB must average over work-level idiosyncrasy, not one novel


class HoldoutParams(BaseModel):
    """Selection thresholds (defaults are the calibrated production values;
    tests pass smaller ones). Frozen into the manifest for reproducibility."""

    min_chars: int = MIN_CHARS
    max_chars: int = MAX_CHARS
    containment_reject: float = CONTAINMENT_REJECT
    target_token_fraction: float = TARGET_TOKEN_FRACTION
    min_works: int = MIN_WORKS


class HoldoutWork(BaseModel):
    url: str
    title: str
    n_chars: int
    n_tokens: int  # whole-doc encode_ordinary, BOS excluded (matches token-stats.md)
    content_sha256: str  # detects text drift independent of the corpus-wide hash
    max_containment_elsewhere: float  # audit trail: how clean the closure left it


class HoldoutManifest(BaseModel):
    """Frozen, provenance-stamped, load-bearing (ADR 0013). Modeled on
    registry/datasets/ - generated once, committed, never hand-edited."""

    schema_version: int = 1
    corpus_sha256: str  # per-work content shas are only valid against this text
    tokenizer_name: str
    author: str = CANKAR_AUTHOR
    source: str = HOLDOUT_SOURCE
    params: HoldoutParams
    cankar_total_tokens: int
    holdout_tokens: int
    holdout_fraction: float
    git_sha: str
    created_at: str
    works: list[HoldoutWork]
    # reverse-containment: training docs >= reject contained in a held-out work;
    # the Phase 3 filter must drop these too or they leak held-out text
    also_exclude_urls: list[str] = []


@dataclass
class SelectionResult:
    works: list[HoldoutWork]
    rejected: list[tuple[str, float, str]]  # (title, max_containment, container title)
    also_exclude_urls: list[str]
    cankar_total_tokens: int


def cankar_docs(corpus_path: Path) -> list[dict]:
    """Every Cankar doc in the MERGED corpus (critique MF-3: the merged corpus
    is the training universe; the registry's 'ingested' flags are pre-merge
    and include works whose text survives only inside a volume).

    Defensive post-condition (ADR 0014): the corpus stage excludes the audited
    misattributions, but evals is the independent last line of defence - if one
    ever re-appears in the Cankar slice, fail loud rather than score seen text."""
    docs = [
        d
        for d in iter_jsonl_docs(corpus_path, missing_hint="run: cankar corpus merge")
        if d.get("author") == CANKAR_AUTHOR
    ]
    leaked = sorted(d["url"] for d in docs if d["url"] in MISATTRIBUTED_URLS)
    if leaked:
        raise CankarError(
            f"misattributed about-Cankar text in the merged Cankar slice: {leaked}. "
            "The corpus-stage NOT_BY_AUTHOR exclusion (ADR 0014) regressed - re-merge."
        )
    return docs


def _candidate(doc: dict, params: HoldoutParams) -> bool:
    return (
        doc["source"] == HOLDOUT_SOURCE and params.min_chars <= doc["n_chars"] <= params.max_chars
    )


def select_holdout(
    docs: list[dict], enc: tiktoken.Encoding, params: HoldoutParams
) -> SelectionResult:
    """Bidirectionally containment-closed, deterministic, budget-filled."""
    all_shingles = {d["url"]: shingles(d["text"]) for d in docs}
    by_url = {d["url"]: d for d in docs}
    cankar_total = sum(len(enc.encode_ordinary(d["text"])) for d in docs)
    budget = int(params.target_token_fraction * cankar_total)

    # FORWARD closure: reject a candidate contained in any single other doc
    clean: list[tuple[dict, float]] = []
    rejected: list[tuple[str, float, str]] = []
    for d in docs:
        if not _candidate(d, params):
            continue
        sub = all_shingles[d["url"]]
        best_url, best_cont = "", 0.0
        for o in docs:
            if o["url"] == d["url"]:
                continue
            c = containment(sub, all_shingles[o["url"]])
            if c > best_cont:
                best_url, best_cont = o["url"], c
        if best_cont >= params.containment_reject:
            rejected.append((d["title"], round(best_cont, 4), by_url[best_url]["title"]))
        else:
            clean.append((d, best_cont))

    # deterministic, append-only order: sha256(url) so adding future works
    # never reshuffles an already-frozen pick (critique A-6)
    clean.sort(key=lambda dc: hashlib.sha256(dc[0]["url"].encode()).hexdigest())

    selected: list[HoldoutWork] = []
    total = 0
    for d, max_cont in clean:
        if total >= budget and len(selected) >= params.min_works:
            break
        n_tokens = len(enc.encode_ordinary(d["text"]))
        selected.append(
            HoldoutWork(
                url=d["url"],
                title=d["title"],
                n_chars=d["n_chars"],
                n_tokens=n_tokens,
                content_sha256=hashlib.sha256(d["text"].encode()).hexdigest(),
                max_containment_elsewhere=round(max_cont, 4),
            )
        )
        total += n_tokens

    if len(selected) < params.min_works:
        raise CankarError(
            f"only {len(selected)} clean holdout works (need >= {params.min_works}); "
            "widen the length band or lower the token target"
        )

    # REVERSE closure: any OTHER doc mostly contained in a held-out work would
    # reproduce held-out text if left in training (design-review 2026-07)
    held_urls = {w.url for w in selected}
    held_shingles = [all_shingles[u] for u in held_urls]
    also_exclude = sorted(
        d["url"]
        for d in docs
        if d["url"] not in held_urls
        and any(
            containment(all_shingles[d["url"]], hs) >= params.containment_reject
            for hs in held_shingles
        )
    )
    return SelectionResult(selected, rejected, also_exclude, cankar_total)


def holdout_excludes(manifest: HoldoutManifest) -> frozenset[str]:
    """The Phase 3 conversion filter: urls to drop from training. Held-out
    works PLUS reverse-contained training docs (excerpts of held-out works) -
    both directions of the closure (see module docstring)."""
    return frozenset(w.url for w in manifest.works) | frozenset(manifest.also_exclude_urls)


def write_holdout_report(
    out: Path, manifest: HoldoutManifest, rejected: list[tuple[str, float, str]]
) -> Path:
    """Human-auditable snapshot: the selected works (read these to catch a
    stray play/poem the length band missed) and the containment rejections."""
    L: list[str] = []
    L.append(generated_marker("cankar evals holdout-freeze", snapshot=True))
    L.append("")
    L.append("# Held-out Cankar set (Phase 2.25 - ADR 0013)")
    L.append("")
    L.append(f"Corpus sha256 `{manifest.corpus_sha256}`, tokenizer `{manifest.tokenizer_name}`.")
    p = manifest.params
    L.append(
        f"Whole-work holdout, Wikivir prose, {p.min_chars:,}-{p.max_chars:,} chars, "
        f"containment-closed at {p.containment_reject} (critique MF-1)."
    )
    L.append(
        f"**{len(manifest.works)} works, {manifest.holdout_tokens:,} tokens "
        f"({100 * manifest.holdout_fraction:.2f}% of the {manifest.cankar_total_tokens:,}-token "
        "Cankar slice).**"
    )
    L.append("")
    L.append("## Held-out works (audit these - the last check against a stray play/poem)")
    L.append("")
    L.append("| title | chars | tokens | max containment elsewhere |")
    L.append("|---|--:|--:|--:|")
    for w in manifest.works:
        L.append(f"| {w.title} | {w.n_chars:,} | {w.n_tokens:,} | {w.max_containment_elsewhere} |")
    L.append("")
    L.append("## Rejected: candidate text lives inside a kept work (forward closure)")
    L.append("")
    L.append("Audit these: no rejected work should be a piece of a HELD-OUT work.")
    if rejected:
        for title, cont, container in sorted(rejected, key=lambda x: -x[1]):
            L.append(f"- {title} ({cont:.3f} contained in '{container}')")
    else:
        L.append("- none in the candidate band")
    L.append("")
    L.append("## Also excluded from training: excerpts of held-out works (reverse closure)")
    L.append("")
    L.append(
        f"{len(manifest.also_exclude_urls)} training docs are >= "
        f"{manifest.params.containment_reject} contained in a held-out work; the Phase 3"
    )
    L.append("filter drops them too, else they reproduce held-out text in training.")
    for url in manifest.also_exclude_urls:
        L.append(f"- {url}")
    write_report(out, L)
    return out


def load_holdout(path: Path) -> HoldoutManifest:
    return load_frozen(path, HoldoutManifest, "cankar evals holdout-freeze")


def iter_holdout_texts(corpus_path: Path, manifest: HoldoutManifest) -> Iterable[tuple[str, str]]:
    """(title, text) for each held-out work, re-read from the corpus and
    content-verified against the frozen sha (guards silent corpus drift)."""
    by_url = {w.url: w for w in manifest.works}
    seen = set()
    for d in iter_jsonl_docs(corpus_path, missing_hint="run: cankar corpus merge"):
        w = by_url.get(d["url"])
        if w is None:
            continue
        if hashlib.sha256(d["text"].encode()).hexdigest() != w.content_sha256:
            raise CankarError(f"held-out work '{w.title}' text drifted from frozen sha - re-freeze")
        seen.add(d["url"])
        yield d["title"], d["text"]
    missing = set(by_url) - seen
    if missing:
        raise CankarError(f"held-out urls absent from corpus (re-merge drift?): {sorted(missing)}")
