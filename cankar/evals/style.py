"""Style classifier - eval pillar #2 (ADR 0015).

Scores whether a prose passage is in Cankar's voice, for judging Phase-5/6
plain->Cankar style-transfer outputs. Trained prose-vs-prose against 14 other
public-domain same-era authors - NOT vs Wikipedia (the era/encyclopedia
confound, ROADMAP critique A-1). Built to the architect critique:

- FEATURES char n-grams (voice: punctuation rhythm + morphology) over word
  n-grams (topic: Ljubljana vs villages). funcwords-only is reported as the
  topic- AND orthography-robust floor (evidence the signal is voice, not
  spelling - so no speculative historical normalizer, ADR 0006/0015).
- UNIT passage-sized chunks, not whole works - matches the Phase-6 deployment
  distribution (the invariant-#1 spirit for the classifier).
- VERSE stripped by a content short-line detector calibrated on labeled
  fixtures (genre labels are 74% empty; clean.py records the ~150-poem
  amputation this form-confound already cost once).
- LEAK-FREE group-split by within-author near-duplicate CLUSTER (containment,
  content-based) - never bare url (that reproduces the volume-containment leak
  ADR 0013 calls dominant). Fully content-based -> no corpus-registry import,
  so stage independence is real, not claimed.
- The frozen model is a load-bearing SCORER: the manifest pins lib versions,
  seed, config, and the artifact sha256; retrain reproduces metrics within
  tolerance (test-guarded). Deploy negative at Phase 6 is modern de-styled
  Slovene, unseen here -> manifest stamps deploy_validated PENDING (MF-3).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline, make_pipeline

from cankar.core.errors import CankarError
from cankar.core.holdout import CANKAR_AUTHOR
from cankar.core.jsonl import iter_jsonl_docs
from cankar.core.manifest import load_frozen
from cankar.core.reports import generated_marker, write_report
from cankar.core.textsim import containment, shingles

log = logging.getLogger("cankar.evals")

STYLE_SOURCE = "wikivir"  # hand-transcribed prose; dLib OCR is Cankar-only -> source confound
SEED = 20260724
CV_FOLDS = 5

# Passage window: whole sentences accumulated to ~this many words, then flushed.
# Sized to the Phase-5 transfer-unit (a few sentences), not the ~8k-char work.
CHUNK_TARGET_WORDS = 90
CHUNK_MIN_WORDS = 30  # a trailing fragment shorter than this is dropped, not kept
CHUNK_MAX_WORDS = 200  # a single unpunctuated sentence over this is hard word-split

# Verse detector: fraction of non-blank lines with <= this many words. Calibrated
# on committed fixtures (tests/fixtures/evals/style-verse.txt vs style-prose.txt);
# labeled verse sits ~0.95, prose ~0.38 - 0.55 is the empty band (ADR 0006).
VERSE_SHORT_LINE_MAX_WORDS = 8
VERSE_LINE_FRACTION = 0.55

# Near-duplicate clustering: two same-author docs join a group when either is at
# least this contained in the other (volume/part, transcription twins). Matches
# the merge's registry-confirm threshold (merge.py); reuses core.textsim.
CLUSTER_CONTAINMENT = 0.5

# Slovene closed-class function words (topic- and largely orthography-independent
# voice signal). Proper diacritics; single-char prepositions kept via a 1+ char
# token pattern. Standard stylometry feature set, not a tuned list.
FUNCTION_WORDS = sorted(
    set(
        """in pa ter ali a ampak toda ker da ki ko če kakor kot saj sicer
        je so sem si smo ste bil bila bilo bili bom boš bo bomo boste bodo bi biti
        ne ni nič se me te ga jo jih nas vas jim mu ji mi ti on ona ono oni vi
        moj tvoj njegov njen naš vaš svoj ta to tega tem taka tisti tale
        v na z s k h o po za do od pri pred med nad pod ob čez skozi brez proti iz
        zaradi kljub kdo kaj kje kam kdaj kako zakaj kateri kdor
        tudi še že le samo celo prav zelo bolj najbolj tako potem zdaj takrat tam
        tukaj tja vsak ves vsi nekaj nekdo nihče noben""".split()
    )
)

# Orthography spot-check (ADR 0015): the clearest era-marker, l-insertion in the
# sonce family (Cankar's "solnce" -> modern "sonce"). Deliberately minimal - the
# ablation shows normalizing it barely moves AUC, which together with the
# funcwords floor is the evidence orthography is not the load-bearing signal. A
# comprehensive historical normalizer is NOT built (ADR 0006: the data says the
# confound is not there).
_ORTHO_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"([sS])oln([cčCČ])"), r"\1on\2"),  # case-preserving (drop the archaic l)
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZČŠŽ])")


class Analyzer(StrEnum):
    """Feature families compared in the ablation; CHAR_WB is shipped."""

    WORD = "word"  # 1-2 word n-grams: separates well but on TOPIC
    CHAR_WB = "char_wb"  # 3-5 char n-grams: voice (punctuation rhythm, morphology)
    FUNCWORDS = "funcwords"  # function-word frequencies: the topic/orthography-robust floor
    CHAR_WB_ORTHONORM = "char_wb_orthonorm"  # char_wb after the sonce spot-check


class DeployStatus(StrEnum):
    """Whether the classifier is validated against its DEPLOY negative (modern
    de-styled Slovene). Train negative is 19th-c peer prose; the deploy negative
    does not exist until Phase 5 pairs (MF-3 - the invariant-#1 analogue)."""

    PENDING_PHASE6 = "PENDING Phase 6"
    VALIDATED = "validated"


class StyleParams(BaseModel):
    """Frozen selection/feature config (reproducibility - ADR 0015)."""

    source: str = STYLE_SOURCE
    seed: int = SEED
    cv_folds: int = CV_FOLDS
    chunk_target_words: int = CHUNK_TARGET_WORDS
    chunk_min_words: int = CHUNK_MIN_WORDS
    chunk_max_words: int = CHUNK_MAX_WORDS
    verse_short_line_max_words: int = VERSE_SHORT_LINE_MAX_WORDS
    verse_line_fraction: float = VERSE_LINE_FRACTION
    cluster_containment: float = CLUSTER_CONTAINMENT
    analyzer: Analyzer = Analyzer.CHAR_WB  # the shipped feature family
    char_ngram: tuple[int, int] = (3, 5)
    word_ngram: tuple[int, int] = (1, 2)
    min_df: int = 5
    C: float = 1.0
    max_iter: int = 2000


# ----------------------------------------------------------------------------
# text utilities (content-only - no registry, MF-2)
# ----------------------------------------------------------------------------


def short_line_fraction(text: str, max_words: int = VERSE_SHORT_LINE_MAX_WORDS) -> float:
    """Fraction of non-blank lines with <= max_words words. Verse ~0.95, prose ~0.4."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    return sum(len(ln.split()) <= max_words for ln in lines) / len(lines)


def is_verse(text: str, params: StyleParams) -> bool:
    return short_line_fraction(text, params.verse_short_line_max_words) > params.verse_line_fraction


def normalize_orthography(text: str) -> str:
    """The sonce-family spot-check normalization (see _ORTHO_RULES)."""
    for pat, repl in _ORTHO_RULES:
        text = pat.sub(repl, text)
    return text


def _cap_length(chunk: str, params: StyleParams) -> list[str]:
    """A single unpunctuated (or unsplittable) sentence can exceed the window;
    hard word-split it into target-sized pieces so no chunk drifts far from the
    passage-sized deploy unit. No-op on normal prose (design-review 2026-07)."""
    words = chunk.split()
    if len(words) <= params.chunk_max_words:
        return [chunk]
    t = params.chunk_target_words
    return [" ".join(words[i : i + t]) for i in range(0, len(words), t)]


def passage_chunks(text: str, params: StyleParams) -> list[str]:
    """Whole-sentence windows accumulated to ~chunk_target_words; a trailing
    fragment below chunk_min_words is dropped (never a half-sentence sample); a
    lone over-long sentence is capped to the passage unit (_cap_length)."""
    sentences = _SENTENCE_SPLIT.split(text.replace("\n", " "))
    out: list[str] = []
    buf: list[str] = []
    n = 0
    for s in sentences:
        w = len(s.split())
        if not w:
            continue
        buf.append(s)
        n += w
        if n >= params.chunk_target_words:
            out.extend(_cap_length(" ".join(buf), params))
            buf, n = [], 0
    if n >= params.chunk_min_words:
        out.extend(_cap_length(" ".join(buf), params))
    return out


def _cluster_ids(docs: list[dict], params: StyleParams) -> dict[str, str]:
    """url -> group id, clustering near-duplicate docs of the SAME author
    (volume/part, transcription twins) via containment union-find. Content-based
    (reuses core.textsim) so no registry work_id is needed - the group key never
    lets a work's text straddle train and test (MF-1)."""
    by_author: dict[str, list[dict]] = {}
    for d in docs:
        by_author.setdefault(d["author"], []).append(d)

    parent: dict[str, str] = {d["url"]: d["url"] for d in docs}

    def find(u: str) -> str:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for group in by_author.values():
        sh = {d["url"]: shingles(d["text"]) for d in group}
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                sa, sb = sh[a["url"]], sh[b["url"]]
                if max(containment(sa, sb), containment(sb, sa)) >= params.cluster_containment:
                    union(a["url"], b["url"])
    return {u: find(u) for u in parent}


@dataclass
class LabeledChunks:
    """Passage chunks with aligned label / group / author arrays (ADR 0008)."""

    texts: list[str]
    labels: np.ndarray  # 1 = Cankar, 0 = other
    groups: np.ndarray  # near-dup cluster id (group-split key)
    authors: np.ndarray
    n_verse_docs_dropped: int
    n_docs: int


def load_labeled_chunks(
    corpus_path: Path, exclude_urls: frozenset[str], params: StyleParams
) -> LabeledChunks:
    """Wikivir prose chunks for Cankar (positive) vs every other authored peer
    (negative), holdout-excluded, verse-filtered, near-dup clustered. The peer
    set is every non-null `author` in the merged corpus (Wikipedia carries
    author=None and is excluded) - derived from content, no registry/config
    read, so stage independence is real (MF-2)."""
    docs = [
        d
        for d in iter_jsonl_docs(corpus_path, missing_hint="run: cankar corpus merge")
        if d.get("author") is not None
        and d.get("source") == params.source
        and d["url"] not in exclude_urls
    ]
    kept: list[dict] = []
    n_verse = 0
    for d in docs:
        if is_verse(d["text"], params):
            n_verse += 1
            continue
        kept.append(d)

    clusters = _cluster_ids(kept, params)
    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    authors: list[str] = []
    contributing = 0  # docs that yielded >=1 chunk (a short doc yields none)
    for d in kept:
        doc_chunks = passage_chunks(d["text"], params)
        contributing += bool(doc_chunks)
        for ch in doc_chunks:
            texts.append(ch)
            labels.append(1 if d["author"] == CANKAR_AUTHOR else 0)
            groups.append(clusters[d["url"]])
            authors.append(d["author"])
    if not texts or CANKAR_AUTHOR not in set(authors):
        raise CankarError("no labeled chunks (need Cankar + peer prose; run: cankar corpus merge)")
    return LabeledChunks(
        texts=texts,
        labels=np.array(labels),
        groups=np.array(groups),
        authors=np.array(authors),
        n_verse_docs_dropped=n_verse,
        n_docs=contributing,
    )


# ----------------------------------------------------------------------------
# features + model
# ----------------------------------------------------------------------------


def _vectorizer(analyzer: Analyzer, params: StyleParams) -> TfidfVectorizer:
    if analyzer is Analyzer.WORD:
        return TfidfVectorizer(
            analyzer="word", ngram_range=params.word_ngram, min_df=params.min_df, sublinear_tf=True
        )
    if analyzer is Analyzer.FUNCWORDS:
        return TfidfVectorizer(
            vocabulary=FUNCTION_WORDS, token_pattern=r"(?u)\b\w+\b", sublinear_tf=True
        )
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=params.char_ngram, min_df=params.min_df, sublinear_tf=True
    )


def build_pipeline(analyzer: Analyzer, params: StyleParams) -> Pipeline:
    """A TF-IDF -> balanced logistic-regression pipeline for one feature family."""
    return make_pipeline(
        _vectorizer(analyzer, params),
        LogisticRegression(
            class_weight="balanced",
            C=params.C,
            max_iter=params.max_iter,
            solver="liblinear",
            random_state=params.seed,
        ),
    )


def _texts_for(analyzer: Analyzer, texts: list[str]) -> list[str]:
    if analyzer is Analyzer.CHAR_WB_ORTHONORM:
        return [normalize_orthography(t) for t in texts]
    return texts


def _pipeline_for(analyzer: Analyzer, params: StyleParams) -> Pipeline:
    base = Analyzer.CHAR_WB if analyzer is Analyzer.CHAR_WB_ORTHONORM else analyzer
    return build_pipeline(base, params)


# ----------------------------------------------------------------------------
# evaluation (typed results)
# ----------------------------------------------------------------------------


class FoldMetrics(BaseModel):
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    per_fold_pos_rate: list[float]


@dataclass
class Evaluation:
    fold: FoldMetrics
    ablation: dict[str, float]  # analyzer value -> roc_auc mean
    per_author_meanp: dict[str, float]  # author -> mean out-of-fold P(Cankar)
    top_pos: list[str]  # shipped-model top +Cankar features (human audit)
    top_neg: list[str]
    ablation_top: dict[str, list[str]] = field(default_factory=dict)  # analyzer -> top +features


def _cv(params: StyleParams) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=params.cv_folds, shuffle=True, random_state=params.seed)


def _fold_positive_rates(data: LabeledChunks, params: StyleParams) -> list[float]:
    rates: list[float] = []
    for _, test in _cv(params).split(data.texts, data.labels, groups=data.groups):
        rates.append(round(float(data.labels[test].mean()), 4))
    return rates


def _auc(analyzer: Analyzer, data: LabeledChunks, params: StyleParams) -> np.ndarray:
    return cross_val_score(
        _pipeline_for(analyzer, params),
        _texts_for(analyzer, data.texts),
        data.labels,
        groups=data.groups,
        cv=_cv(params),
        scoring="roc_auc",
    )


def _top_features(
    analyzer: Analyzer, data: LabeledChunks, params: StyleParams, k: int = 25
) -> tuple[list[str], list[str]]:
    pipe = _pipeline_for(analyzer, params).fit(_texts_for(analyzer, data.texts), data.labels)
    vec = pipe[0]
    coef = pipe[-1].coef_[0]
    names = np.array(vec.get_feature_names_out())
    pos = names[np.argsort(coef)[-k:]][::-1].tolist()
    neg = names[np.argsort(coef)[:k]].tolist()
    return pos, neg


def evaluate(data: LabeledChunks, params: StyleParams) -> Evaluation:
    """Group-split CV for the shipped model + the confound-audit evidence: the
    ablation across feature families, per-author confusion, and top features."""
    shipped = params.analyzer
    # shipped char_wb in ONE CV pass (roc + PR). This is the headline AND the
    # char_wb ablation entry - a single source of truth, so the table cannot drift
    # from the headline (design-review 2026-07).
    cv_res = cross_validate(
        _pipeline_for(shipped, params),
        _texts_for(shipped, data.texts),
        data.labels,
        groups=data.groups,
        cv=_cv(params),
        scoring=["roc_auc", "average_precision"],
    )
    roc = cv_res["test_roc_auc"]
    fold = FoldMetrics(
        roc_auc_mean=round(float(roc.mean()), 4),
        roc_auc_std=round(float(roc.std()), 4),
        pr_auc_mean=round(float(cv_res["test_average_precision"].mean()), 4),
        per_fold_pos_rate=_fold_positive_rates(data, params),
    )
    top_pos, top_neg = _top_features(shipped, data, params)  # shipped audit features

    ablation: dict[str, float] = {}
    ablation_top: dict[str, list[str]] = {}
    for a in (Analyzer.WORD, Analyzer.CHAR_WB, Analyzer.FUNCWORDS, Analyzer.CHAR_WB_ORTHONORM):
        if a is shipped:  # reuse the headline pass, do not refit the shipped model
            ablation[a.value] = fold.roc_auc_mean
            ablation_top[a.value] = top_pos[:12]
        else:
            ablation[a.value] = round(float(_auc(a, data, params).mean()), 4)
            ablation_top[a.value] = _top_features(a, data, params, k=12)[0]

    proba = cross_val_predict(
        _pipeline_for(shipped, params),
        _texts_for(shipped, data.texts),
        data.labels,
        groups=data.groups,
        cv=_cv(params),
        method="predict_proba",
    )[:, 1]
    per_author = {
        a: round(float(proba[data.authors == a].mean()), 4) for a in sorted(set(data.authors))
    }
    return Evaluation(
        fold=fold,
        ablation=ablation,
        per_author_meanp=dict(sorted(per_author.items(), key=lambda kv: -kv[1])),
        top_pos=top_pos,
        top_neg=top_neg,
        ablation_top=ablation_top,
    )


def fit_shipped(data: LabeledChunks, params: StyleParams) -> Pipeline:
    """The single seeded fit on ALL data - the artifact Phase 6 loads."""
    return _pipeline_for(params.analyzer, params).fit(
        _texts_for(params.analyzer, data.texts), data.labels
    )


# ----------------------------------------------------------------------------
# frozen manifest
# ----------------------------------------------------------------------------


class StyleManifest(BaseModel):
    """Frozen, provenance-stamped scorer contract (ADR 0015). A trained model is
    not regenerable data: pins lib versions + seed + config + artifact sha so
    retrain reproduces metrics within tolerance (test-guarded)."""

    schema_version: int = 1
    corpus_sha256: str
    git_sha: str
    created_at: str
    tokenizer_independent: bool = True  # features are text-level, no tokenizer
    lib_versions: dict[str, str]
    params: StyleParams
    n_chunks: int
    n_cankar: int
    n_other: int
    pos_rate: float
    n_groups: int
    n_verse_docs_dropped: int
    n_docs: int
    metrics: FoldMetrics
    ablation: dict[str, float]
    per_author_meanp: dict[str, float]
    artifact_sha256: str
    deploy_validated: DeployStatus = DeployStatus.PENDING_PHASE6


def load_style_manifest(path: Path) -> StyleManifest:
    return load_frozen(path, StyleManifest, "cankar evals style-train")


# ----------------------------------------------------------------------------
# confound-audit report (MF-5): read by a human before any quality claim rides
# ----------------------------------------------------------------------------


def write_style_report(out: Path, manifest: StyleManifest, ev: Evaluation) -> Path:
    m = manifest
    lines = [
        generated_marker("cankar evals style-train", snapshot=True),
        "# Style classifier - confound audit",
        "",
        "Eval pillar #2 (ADR 0015): Cankar prose voice vs 14 public-domain peers,",
        "char n-gram TF-IDF + balanced logistic regression, group-split by",
        "within-author near-duplicate cluster. **This report is the required human",
        "audit** - no quality claim rides on the classifier until the top features",
        "read as VOICE (not topic/source/form) and the ablation confirms it.",
        "",
        "## Headline (grouped 5-fold)",
        "",
        f"- **ROC-AUC {m.metrics.roc_auc_mean:.3f} +/- {m.metrics.roc_auc_std:.3f}**, "
        f"PR-AUC {m.metrics.pr_auc_mean:.3f}",
        f"- {m.n_chunks:,} chunks ({m.n_cankar:,} Cankar / {m.n_other:,} peer, "
        f"pos-rate {m.pos_rate:.3f}), {m.n_groups} groups, {m.n_docs} docs, "
        f"{m.n_verse_docs_dropped} verse docs dropped",
        f"- per-fold positive rate: {m.metrics.per_fold_pos_rate}",
        f"- deploy status: **{m.deploy_validated.value}** (train negative is 19th-c",
        "  peer prose; the Phase-6 negative is modern de-styled Slovene - unseen here)",
        "",
        "## Ablation - which feature family carries the signal (MF-5b)",
        "",
        "| feature family | ROC-AUC | reads as |",
        "|---|--:|---|",
        f"| word 1-2gram | {ev.ablation.get('word', 0):.3f} | TOPIC (vocabulary/subject) |",
        f"| **char_wb 3-5 (shipped)** | {ev.ablation.get('char_wb', 0):.3f} | "
        "VOICE (punctuation rhythm, morphology) |",
        f"| funcwords only | {ev.ablation.get('funcwords', 0):.3f} | "
        "topic- AND orthography-robust FLOOR |",
        f"| char_wb + orthonorm | {ev.ablation.get('char_wb_orthonorm', 0):.3f} | "
        "sonce-family spot-check (~no change) |",
        "",
        "The funcwords floor separates Cankar on closed-class words alone (no topic,",
        "stable spelling), and the orthonorm spot-check barely moves AUC: the voice",
        "signal is not a topic or edition-spelling artifact (ADR 0015 deviation note).",
        "",
        "## Top features - the human audit (MF-5a)",
        "",
        "Shipped char_wb model. Reject if +Cankar features are TOPIC (place/character",
        "names), SOURCE (OCR/edition tokens), or verse FORM. Expected: punctuation",
        "rhythm + function words + morphology.",
        "",
        f"- **+Cankar**: {', '.join(repr(x) for x in ev.top_pos)}",
        f"- **-peers**: {', '.join(repr(x) for x in ev.top_neg)}",
        "",
        "## Per-author confusion (MF-5c)",
        "",
        "Mean out-of-fold P(Cankar) per author. A voice classifier confuses Cankar",
        "with the stylistically closest peers (modernists), not with whoever writes",
        "verse - the latter would mean form is leaking.",
        "",
        "| author | mean P(Cankar) |",
        "|---|--:|",
        *[
            f"| {a}{' (positive)' if a == CANKAR_AUTHOR else ''} | {p:.3f} |"
            for a, p in ev.per_author_meanp.items()
        ],
        "",
        "Regenerate: `cankar evals style-train`. Provenance: registry/evals/style.json.",
    ]
    write_report(out, lines)
    return out
