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
