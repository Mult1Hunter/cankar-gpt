"""Style-classifier invariants (ADR 0015, architect critique MF-1..MF-6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cankar.core.errors import CankarError
from cankar.evals import style
from cankar.evals.style import Analyzer, StyleParams

FIX = Path(__file__).parent.parent / "fixtures" / "evals"
PARAMS = StyleParams()


def _body(name: str) -> str:
    txt = (FIX / name).read_text(encoding="utf-8")
    return "\n".join(ln for ln in txt.splitlines() if not ln.startswith("#"))


def test_verse_detector_calibrated_on_fixtures() -> None:
    """MF-6: the short-line threshold separates REAL labeled verse from prose,
    not a guessed number - clean.py records the ~150-poem cost of guessing."""
    verse, prose = _body("style-verse.txt"), _body("style-prose.txt")
    assert style.short_line_fraction(verse) > PARAMS.verse_line_fraction
    assert style.short_line_fraction(prose) < PARAMS.verse_line_fraction
    assert style.is_verse(verse, PARAMS)
    assert not style.is_verse(prose, PARAMS)


def test_passage_chunks_window_and_min() -> None:
    text = " ".join(f"To je stavek številka {i} v tem odstavku." for i in range(60))
    chunks = style.passage_chunks(text, PARAMS)
    assert chunks and all(len(c.split()) >= PARAMS.chunk_min_words for c in chunks)
    # a text below the min-words floor yields no chunk (never a half-sentence sample)
    assert style.passage_chunks("Kratek stavek.", PARAMS) == []


def test_normalize_orthography_sonce_family() -> None:
    assert style.normalize_orthography("v solncu in solnce") == "v soncu in sonce"
    assert style.normalize_orthography("Solnce sije") == "Sonce sije"


def _doc(url: str, author: str, text: str) -> dict:
    return {"url": url, "author": author, "text": text, "source": "wikivir", "n_chars": len(text)}


def test_cluster_groups_near_duplicates(tmp_path: Path) -> None:
    """MF-1: a volume and the part it contains (two urls, same text) must land in
    ONE group, so a work's text cannot straddle train and test."""
    part = " ".join(f"beseda{i}" for i in range(200))
    volume = part + " " + " ".join(f"drugo{i}" for i in range(60))  # contains the part
    distinct = " ".join(f"cisto{i}" for i in range(200))
    docs = [
        _doc("u/part", "Ivan Cankar", part),
        _doc("u/volume", "Ivan Cankar", volume),
        _doc("u/distinct", "Ivan Cankar", distinct),
    ]
    clusters = style._cluster_ids(docs, PARAMS)
    assert clusters["u/part"] == clusters["u/volume"]  # near-dups grouped
    assert clusters["u/distinct"] != clusters["u/part"]  # distinct work separate
    # never cross authors even if text were similar
    docs2 = docs + [_doc("u/other", "Josip Jurčič", part)]
    c2 = style._cluster_ids(docs2, PARAMS)
    assert c2["u/other"] != c2["u/part"]


def test_load_excludes_wikipedia_holdout_and_verse(tmp_path: Path) -> None:
    corpus = tmp_path / "merged.jsonl"
    prose = " ".join(f"To je dolg prozni stavek o življenju številka {i}." for i in range(40))
    verse = "\n".join(f"kratka vrstica {i}" for i in range(30))  # short lines -> verse
    rows = [
        _doc("c/1", "Ivan Cankar", prose),
        _doc("j/1", "Josip Jurčič", prose.replace("življenju", "vasi")),
        _doc("c/verse", "Ivan Cankar", verse),  # dropped: verse
        _doc("c/hold", "Ivan Cankar", prose),  # dropped: in holdout
        {"url": "w/1", "author": None, "text": prose, "source": "wikipedia", "n_chars": len(prose)},
    ]
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    data = style.load_labeled_chunks(corpus, frozenset({"c/hold"}), PARAMS)
    authors = set(data.authors.tolist())
    assert authors == {"Ivan Cankar", "Josip Jurčič"}  # no None/wikipedia
    assert data.n_verse_docs_dropped == 1
    assert 1 in data.labels and 0 in data.labels


def test_load_raises_without_cankar(tmp_path: Path) -> None:
    corpus = tmp_path / "m.jsonl"
    prose = " ".join(f"Prozni stavek {i} o vasi in polju in gozdu." for i in range(40))
    corpus.write_text(json.dumps(_doc("j/1", "Josip Jurčič", prose), ensure_ascii=False) + "\n")
    with pytest.raises(CankarError, match="no labeled chunks"):
        style.load_labeled_chunks(corpus, frozenset(), PARAMS)


def _toy_data(n_groups: int = 40) -> style.LabeledChunks:
    """Separable toy chunks with one group per doc (both classes present)."""
    texts, labels, groups, authors = [], [], [], []
    for g in range(n_groups):
        cankar = g % 2 == 0
        for j in range(6):
            if cankar:
                texts.append(f"Megla se je vlekla nad reko; molče, otožno... vzdih{j} tega{g}.")
                authors.append("Ivan Cankar")
            else:
                texts.append(f"Zopet je potem mož odšel na polje in delal ves dan{j} tam{g}.")
                authors.append("Josip Jurčič")
            labels.append(1 if cankar else 0)
            groups.append(f"g{g}")
    return style.LabeledChunks(
        texts=texts,
        labels=np.array(labels),
        groups=np.array(groups),
        authors=np.array(authors),
        n_verse_docs_dropped=0,
        n_docs=n_groups,
    )


def test_fit_shipped_is_seed_deterministic() -> None:
    """MF-4 reproducibility contract in miniature: same seed -> identical scores."""
    data = _toy_data()
    p1 = style.fit_shipped(data, PARAMS).predict_proba(data.texts)[:, 1]
    p2 = style.fit_shipped(data, PARAMS).predict_proba(data.texts)[:, 1]
    assert np.allclose(p1, p2)


def test_evaluate_produces_ablation_and_audit() -> None:
    data = _toy_data()
    ev = style.evaluate(data, PARAMS)
    # all four ablation arms reported; the separable toy is learnable
    assert set(ev.ablation) == {a.value for a in Analyzer}
    assert ev.fold.roc_auc_mean > 0.9
    assert "Ivan Cankar" in ev.per_author_meanp
    assert ev.top_pos and ev.top_neg  # audit features present
    # SF-3: the char_wb ablation entry IS the headline - one source, no drift
    assert ev.ablation[Analyzer.CHAR_WB.value] == ev.fold.roc_auc_mean


def test_manifest_round_trip(tmp_path: Path) -> None:
    """MF-4 read path (Phase 6 loads this): StyleParams.char_ngram must reload as
    a tuple (not list) and DeployStatus must round-trip through JSON."""
    from cankar.core.manifest import library_versions
    from cankar.evals.style import DeployStatus, FoldMetrics, StyleManifest

    m = StyleManifest(
        corpus_sha256="x",
        git_sha="y",
        created_at="2026-07-24T00:00:00+00:00",
        lib_versions=library_versions("scikit-learn"),
        params=StyleParams(),
        n_chunks=10,
        n_cankar=3,
        n_other=7,
        pos_rate=0.3,
        n_groups=5,
        n_verse_docs_dropped=1,
        n_docs=4,
        metrics=FoldMetrics(
            roc_auc_mean=0.9, roc_auc_std=0.01, pr_auc_mean=0.8, per_fold_pos_rate=[0.3]
        ),
        ablation={"char_wb": 0.9},
        per_author_meanp={"Ivan Cankar": 0.8},
        artifact_sha256="z",
    )
    p = tmp_path / "style.json"
    p.write_text(m.model_dump_json(indent=2), encoding="utf-8")
    loaded = style.load_style_manifest(p)
    assert isinstance(loaded.params.char_ngram, tuple) and loaded.params.char_ngram == (3, 5)
    assert loaded.deploy_validated is DeployStatus.PENDING_PHASE6
    assert "scikit-learn_version" in loaded.lib_versions  # suffixed-key convention


def test_load_style_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(CankarError, match="not frozen"):
        style.load_style_manifest(tmp_path / "nope.json")


# --- MF-3: the scorer measured on its DEPLOY task ----------------------------


class _FakeModel:
    """Returns a fixed P(Cankar) per text, so the decomposition arithmetic is
    checkable without a trained artifact (the real one is gitignored)."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def predict_proba(self, texts):
        import numpy as np

        p = np.array([self.scores[t] for t in texts])
        return np.column_stack([1 - p, p])


def test_a_topic_driven_scorer_is_called_inadequate() -> None:
    """The real failure: the classifier separates 1900s prose from modern prose
    far better than it separates Cankar from de-styled Cankar. A scorer whose
    topic effect dwarfs its style effect cannot carry a style claim, however
    good its training AUC was."""
    from cankar.evals.style import DeployStatus, deploy_check

    scores = {"c1": 0.80, "c2": 0.72, "d1": 0.66, "d2": 0.58, "m1": 0.09, "m2": 0.11}
    check = deploy_check(_FakeModel(scores), ["c1", "c2"], ["d1", "d2"], ["m1", "m2"])

    assert check.style_effect == pytest.approx(0.14, abs=1e-6)
    assert check.topic_effect == pytest.approx(0.52, abs=1e-6)
    assert check.topic_effect > check.style_effect
    assert check.verdict is DeployStatus.MEASURED_INADEQUATE


def test_a_genuinely_style_driven_scorer_validates() -> None:
    """The gate must be able to return VALIDATED, or it is a constant dressed as
    a measurement. Same shape, but the style effect now dominates and every real
    Cankar passage outranks every de-styled one."""
    from cankar.evals.style import DeployStatus, deploy_check

    scores = {"c1": 0.95, "c2": 0.92, "d1": 0.20, "d2": 0.15, "m1": 0.10, "m2": 0.08}
    check = deploy_check(_FakeModel(scores), ["c1", "c2"], ["d1", "d2"], ["m1", "m2"])

    assert check.deploy_auc == 1.0
    assert check.style_effect > check.topic_effect
    assert check.verdict is DeployStatus.VALIDATED


def test_deploy_auc_is_pairwise_not_thresholded() -> None:
    """Ranking a Cankar passage above its own de-styled counterpart is the
    deploy question; a 0.5 cutoff would instead measure calibration, which is
    not what is being asked and moves with passage difficulty."""
    from cankar.evals.style import deploy_check

    # every score sits above 0.5, so any thresholded accuracy would read 100%
    scores = {"c1": 0.60, "c2": 0.90, "d1": 0.70, "d2": 0.80, "m1": 0.55, "m2": 0.55}
    check = deploy_check(_FakeModel(scores), ["c1", "c2"], ["d1", "d2"], ["m1", "m2"])
    assert check.deploy_auc == pytest.approx(0.5, abs=1e-6)


def test_a_missing_series_is_not_treated_as_zero() -> None:
    from cankar.evals.style import deploy_check

    with pytest.raises(CankarError, match="all three series"):
        deploy_check(_FakeModel({"c1": 0.9}), ["c1"], [], ["c1"])


def test_a_deploy_verdict_does_not_follow_retrained_weights(tmp_path) -> None:
    """The verdict belongs to a specific artifact, not to the project. Carrying
    it onto a retrained classifier would publish a claim about weights nobody
    measured - in either direction, a stale VALIDATED being the dangerous one."""
    import json

    from cankar.evals.style import DeployStatus, deploy_status_for

    rec = tmp_path / "style-deploy.json"
    rec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_sha256": "a" * 64,
                "corpus_sha256": "b" * 64,
                "git_sha": "deadbee",
                "created_at": "2026-07-28T00:00:00+00:00",
                "check": {
                    "n_cankar": 2,
                    "n_destyled": 2,
                    "n_modern": 2,
                    "mean_cankar": 0.9,
                    "mean_destyled": 0.2,
                    "mean_modern": 0.1,
                    "style_effect": 0.7,
                    "topic_effect": 0.1,
                    "deploy_auc": 1.0,
                    "paired_auc": 1.0,
                    "verdict": DeployStatus.VALIDATED.value,
                },
            }
        )
    )
    assert deploy_status_for(rec, "a" * 64) is DeployStatus.VALIDATED
    assert deploy_status_for(rec, "c" * 64) is DeployStatus.PENDING_PHASE6


def test_a_missing_deploy_record_reads_as_pending(tmp_path) -> None:
    """Absence must mean unknown, never validated - the safe direction blocks
    claims until someone measures."""
    from cankar.evals.style import DeployStatus, deploy_status_for

    assert deploy_status_for(tmp_path / "nope.json", "a" * 64) is DeployStatus.PENDING_PHASE6


def test_unaligned_series_are_rejected() -> None:
    """The paired AUC assumes index i of each list is the SAME passage. Without
    this the report can claim "its own de-styled pair" over series that were
    never paired - which is how the unpaired statistic got published under the
    paired description."""
    from cankar.evals.style import deploy_check

    with pytest.raises(CankarError, match="aligned series"):
        deploy_check(_FakeModel({"a": 0.9, "b": 0.2, "m": 0.1}), ["a", "b"], ["b"], ["m"])


def test_paired_and_unpaired_auc_are_different_statistics() -> None:
    """They answer different questions and the real classifier splits them 0.87
    vs 0.65. A fixture where each Cankar passage beats its OWN pair but loses to
    others must show paired 1.0 and unpaired below it."""
    from cankar.evals.style import deploy_check

    scores = {"c1": 0.40, "c2": 0.90, "d1": 0.30, "d2": 0.80, "m1": 0.05, "m2": 0.05}
    check = deploy_check(_FakeModel(scores), ["c1", "c2"], ["d1", "d2"], ["m1", "m2"])
    assert check.paired_auc == 1.0, "each beats its own pair"
    assert check.deploy_auc < check.paired_auc, "c1 loses to d2 across passages"


def test_the_deploy_auc_floor_is_the_line_it_claims_to_be() -> None:
    """ADR 0006: a threshold no test sits either side of can be set to anything
    and the suite stays green. Every other fixture here reaches its verdict via
    the style-vs-topic condition, leaving DEPLOY_AUC_FLOOR unexercised.

    Both cases hold `style effect > topic effect` true, so the AUC condition is
    the ONLY thing deciding them.
    """
    from cankar.evals.style import DEPLOY_AUC_FLOOR, DeployStatus, deploy_check

    assert DEPLOY_AUC_FLOOR == 0.85

    # 4x4 = 16 comparisons. Spread the de-styled scores so each Cankar passage
    # wins a countable number of them.
    destyled = {"d0": 0.30, "d1": 0.40, "d2": 0.60, "d3": 0.70}

    # 14/16 = 0.875, just above the floor
    above = {"c0": 0.65, "c1": 0.65, "c2": 0.75, "c3": 0.75} | destyled
    above |= {f"m{i}": 0.35 for i in range(4)}  # style +0.20 > topic +0.15
    got = deploy_check(
        _FakeModel(above), ["c0", "c1", "c2", "c3"], list(destyled), [f"m{i}" for i in range(4)]
    )
    assert got.deploy_auc == pytest.approx(0.875)
    assert got.style_effect > got.topic_effect
    assert got.verdict is DeployStatus.VALIDATED

    # 12/16 = 0.75, just below it - same shape, one notch down
    below = {"c0": 0.65, "c1": 0.65, "c2": 0.65, "c3": 0.65} | destyled
    below |= {f"m{i}": 0.52 for i in range(4)}  # style +0.15 > topic +0.02
    got = deploy_check(
        _FakeModel(below), ["c0", "c1", "c2", "c3"], list(destyled), [f"m{i}" for i in range(4)]
    )
    assert got.deploy_auc == pytest.approx(0.75)
    assert got.style_effect > got.topic_effect, "the AUC must be what decides this"
    assert got.verdict is DeployStatus.MEASURED_INADEQUATE
