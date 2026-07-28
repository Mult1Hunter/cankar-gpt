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

import itertools
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


DEPLOY_AUC_FLOOR = 0.85
"""Minimum deploy-task ROC-AUC for the classifier to carry a quality claim.

Conventional "good discrimination" for a diagnostic test is 0.8-0.9; 0.85 sits
in that band and far below the 0.993 the training task reaches, so it is a
lenient bar rather than a demanding one.

The measured deploy AUC is 0.650 (Phase 6, all 285 held-out pairs), so the
verdict does not hinge on where in that band the line falls - anything from 0.70
to 0.98 returns the same answer, as does the second condition (style effect must
exceed topic effect: measured +0.128 vs +0.536). A threshold whose exact value
would change the conclusion would need a real calibration; this one does not.
"""


class DeployStatus(StrEnum):
    """Whether the classifier is validated against its DEPLOY negative (modern
    de-styled Slovene). Train negative is 19th-c peer prose; the deploy negative
    does not exist until Phase 5 pairs (MF-3 - the invariant-#1 analogue)."""

    PENDING_PHASE6 = "PENDING Phase 6"
    VALIDATED = "validated"
    # Measured in Phase 6 and found unfit for the deploy task. A third state,
    # not a variant of PENDING: pending means unknown and blocks claims by
    # default; this means known-bad, which blocks them permanently and points at
    # a different instrument. Collapsing the two would let a later "just run the
    # check" read as though the answer might still come out fine.
    MEASURED_INADEQUATE = "measured - inadequate for deploy"


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
        f"- deploy status: **{m.deploy_validated.value}** - train negative is 19th-c",
        "  peer prose; the deploy negative is modern plain Slovene, which this",
        "  training run never saw. Measured separately: `style-deploy.md`.",
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


# ----------------------------------------------------------------------------
# MF-3: is the scorer fit for the task it is DEPLOYED on? (Phase 6)
# ----------------------------------------------------------------------------


class DeployRecord(BaseModel):
    """A deploy verdict, bound to the exact classifier it was measured on.

    `artifact_sha256` is the whole point. The verdict is a property of a
    particular trained scorer, not of the project - carrying it forward onto a
    retrained artifact without checking would let a stale "inadequate" (or a
    stale "validated") attach itself to weights nobody measured.
    """

    schema_version: int = 1
    artifact_sha256: str
    corpus_sha256: str
    git_sha: str
    created_at: str
    check: DeployCheck


def load_deploy_record(path: Path) -> DeployRecord:
    return load_frozen(path, DeployRecord, "cankar evals deploy-check")


def deploy_status_for(record_path: Path, artifact_sha: str) -> DeployStatus:
    """The recorded verdict IF it was measured on this exact artifact.

    Anything else is PENDING, including a missing record and a sha mismatch. A
    retrain that changes the weights invalidates the measurement, and defaulting
    to "unknown" is the only safe direction: it blocks claims until someone
    re-measures, rather than publishing a verdict about different weights.
    """
    if not record_path.exists():
        return DeployStatus.PENDING_PHASE6
    record = load_deploy_record(record_path)
    if record.artifact_sha256 != artifact_sha:
        log.warning(
            "deploy record is for artifact %s but this one is %s - status resets to %s",
            record.artifact_sha256[:12],
            artifact_sha[:12],
            DeployStatus.PENDING_PHASE6.value,
        )
        return DeployStatus.PENDING_PHASE6
    return record.check.verdict


class DeployCheck(BaseModel):
    """The classifier measured on its deploy task, not its training task.

    Trained to separate Cankar from 14 public-domain peers - all 1900s literary
    prose. Deployment asks something else: is THIS passage Cankar's voice, where
    the negative is plain modern Slovene. A train-task ROC-AUC says nothing about
    that, and the confound audit that cleared the training signal as VOICE held
    period constant by construction (every peer is also 1900s prose), so it could
    not have detected period features either.
    """

    n_cankar: int
    n_destyled: int
    n_modern: int
    mean_cankar: float
    mean_destyled: float
    mean_modern: float
    style_effect: float  # real vs de-styled Cankar - topic held constant
    topic_effect: float  # de-styled Cankar vs modern plain - register held constant
    # TWO AUCs, because they answer different questions and only one matches how
    # the scorer is used. Reporting a single "deploy AUC" was wrong: the code
    # computed the unpaired one while the docstring, the report and the ROADMAP
    # all described the paired one, and they differ by 0.22.
    deploy_auc: float  # unpaired, all cross comparisons - the deployment-shaped one
    paired_auc: float  # each Cankar passage against its OWN de-styled pair
    verdict: DeployStatus


def deploy_check(
    model: Pipeline, cankar: list[str], destyled: list[str], modern: list[str]
) -> DeployCheck:
    """Decompose the score into a style effect and a topic effect.

    The decomposition is only available because of design invariant #1: the SAME
    plain-register prompt produced both the de-styled Cankar passages and the
    fresh modern drafts, so register is held constant between them and what is
    left is topic and period. Without that shared prompt the two would differ in
    two ways at once and neither effect could be attributed.

    Both AUCs are reported because they answer different questions:

    - `paired_auc` ranks each Cankar passage against its OWN de-styled version.
      Content is held constant, so this asks "can it see styling at all?"
    - `deploy_auc` is unpaired, every Cankar passage against every de-styled one.
      Passage difficulty does NOT cancel, which is the point: deployment compares
      scores across DIFFERENT passages (styler output on one topic against plain
      text on another), so this is the shape the scorer is actually used in.

    Fitness is decided on `deploy_auc` and the effect decomposition. A high
    paired and low unpaired AUC - which is what this classifier shows - means it
    can detect styling on matched content but cannot produce scores comparable
    between passages, and comparability is what a quality claim needs.
    """

    if not (cankar and destyled and modern):
        raise CankarError("deploy check needs all three series - a missing one is not a zero")
    if len(cankar) != len(destyled):
        raise CankarError(
            f"paired AUC needs aligned series, got {len(cankar)} Cankar and "
            f"{len(destyled)} de-styled - index i of each must be the same passage"
        )

    def p(texts: list[str]) -> list[float]:
        return [float(x) for x in model.predict_proba(texts)[:, 1]]

    pc, pd, pm = p(cankar), p(destyled), p(modern)
    mc, md, mm = (sum(v) / len(v) for v in (pc, pd, pm))

    wins = sum(a > b for a, b in itertools.product(pc, pd))
    ties = sum(a == b for a, b in itertools.product(pc, pd))
    auc = (wins + 0.5 * ties) / (len(pc) * len(pd))

    pw = sum(a > b for a, b in zip(pc, pd, strict=True))
    pt = sum(a == b for a, b in zip(pc, pd, strict=True))
    paired = (pw + 0.5 * pt) / len(pc)

    return DeployCheck(
        n_cankar=len(pc),
        n_destyled=len(pd),
        n_modern=len(pm),
        mean_cankar=mc,
        mean_destyled=md,
        mean_modern=mm,
        style_effect=mc - md,
        topic_effect=md - mm,
        deploy_auc=auc,
        paired_auc=paired,
        verdict=(
            DeployStatus.VALIDATED
            if auc >= DEPLOY_AUC_FLOOR and (mc - md) > (md - mm)
            else DeployStatus.MEASURED_INADEQUATE
        ),
    )


def write_deploy_report(out: Path, check: DeployCheck, train_auc: float, corpus_sha: str) -> None:
    """The MF-3 answer, kept next to the audit that could not have reached it."""
    c = check
    ratio = c.topic_effect / c.style_effect if c.style_effect else float("inf")
    L = [
        generated_marker("cankar evals deploy-check", snapshot=True),
        "",
        "# Style classifier - deploy fitness (MF-3)",
        "",
        # Stamped so the freshness gate can catch this going stale: the verdict
        # is only valid for the pair set and classifier this corpus produced.
        f"Corpus sha256 `{corpus_sha}`.",
        "",
        f"**Verdict: {c.verdict.value}.**",
        "",
        f"Train-task ROC-AUC is {train_auc:.3f} - Cankar against 14 public-domain",
        "peers, all 1900s literary prose. That is not the task it is deployed on.",
        "Deployment asks whether a passage is Cankar's VOICE, with plain modern",
        "Slovene as the negative.",
        "",
        f"**Deploy ROC-AUC = {c.deploy_auc:.3f}** (unpaired, {c.n_cankar} held-out pairs):",
        "every real Cankar passage against every de-styled one. Passage difficulty",
        "does NOT cancel, and that is the point - deployment compares scores across",
        "DIFFERENT passages, so this is the shape the scorer is actually used in.",
        "",
        f"**Paired ROC-AUC = {c.paired_auc:.3f}**: each Cankar passage against its OWN",
        "de-styled version, content held constant. The scorer CAN see styling when",
        "the content is fixed; what it cannot do is produce scores comparable between",
        "passages, and comparability is what a quality claim needs.",
        "",
        "## Where the score actually comes from",
        "",
        "Design invariant #1 makes this decomposable: the same plain-register",
        "prompt produced both the de-styled Cankar passages and the fresh modern",
        "drafts, so register is held constant between them and what remains is",
        "topic and period.",
        "",
        "| series | n | mean P(Cankar) |",
        "|---|--:|--:|",
        f"| real Cankar | {c.n_cankar} | {c.mean_cankar:.3f} |",
        f"| de-styled Cankar (Cankar topic, plain register) | {c.n_destyled} | "
        f"{c.mean_destyled:.3f} |",
        f"| modern drafts (modern topic, plain register) | {c.n_modern} | {c.mean_modern:.3f} |",
        "",
        f"- style effect (topic held constant): **{c.style_effect:+.3f}**",
        f"- topic effect (register held constant): **{c.topic_effect:+.3f}**",
        f"- ratio: **{ratio:.1f}x** in favour of topic",
        "",
        "## Why the training audit did not catch this",
        "",
        "`style.md` cleared the training signal as VOICE on a feature ablation:",
        "char n-grams beat word n-grams, and a function-words-only floor held at",
        "0.874. That audit was sound for the task it examined and could not have",
        "found this - every peer in the training negative is also 1900s literary",
        "prose, so PERIOD was held constant by construction and period features",
        "were free to carry signal without ever showing up as topic.",
        "",
        "## What this blocks, and what it does not",
        "",
        "The classifier cannot carry a headline style-transfer claim on modern",
        "topics: a large part of any score it gives is the passage's period",
        "vocabulary, not its voice. It remains usable as a DIRECTIONAL signal -",
        "the sign of a change is informative, the magnitude is not.",
        "",
        "This is a known failure class in the style-transfer literature, where",
        "classifier-based style strength is standardly paired with a content",
        "preservation metric and a fluency metric rather than used alone. The",
        "ROADMAP's LLM meaning-judge is the intended replacement instrument.",
    ]
    write_report(out, L)
