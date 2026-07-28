"""Publication gates - what leaves this repo must state its terms.

Redistributing volunteer transcriptions without naming the licence or the
contributors is the hazard that survives the licensing decision, and
publication is one-way in practice: forks and caches outlive un-publishing.
"""

from __future__ import annotations

import pytest

from cankar.core.errors import CankarError
from cankar.pairs.publish import REQUIRED_CARD_TERMS, require_licensed


def test_a_fully_licensed_card_passes() -> None:
    require_licensed("license: cc-by-sa-4.0\nAttribute Wikisource contributors. CC BY-SA 4.0.")


@pytest.mark.parametrize("term", REQUIRED_CARD_TERMS)
def test_each_required_term_is_individually_enforced(term: str) -> None:
    """Parametrised so no single term can silently stop being checked - the
    whole-card assertion would still pass with one of them dropped."""
    card = "license: cc-by-sa-4.0\nAttribute Wikisource contributors. CC BY-SA 4.0."
    with pytest.raises(CankarError, match="licensing terms"):
        require_licensed(card.replace(term, ""))


def test_the_generated_card_satisfies_its_own_gate() -> None:
    """The card writer and the gate must not drift apart."""
    from cankar.core.manifest import load_frozen
    from cankar.core.paths import pairs_manifest, passages_manifest
    from cankar.pairs.destyle import PairsManifest
    from cankar.pairs.publish import dataset_card
    from cankar.pairs.segment import PassagesManifest

    pairs = load_frozen(pairs_manifest(), PairsManifest, "cankar pairs destyle")
    passages = load_frozen(passages_manifest(), PassagesManifest, "cankar pairs segment")
    require_licensed(dataset_card(pairs, passages, "x/y"))
