"""Unit tests for the text-folding and scoring primitives in consensus.py.

Every fold and threshold in consensus.py was added for a measured failure
(see README / project notes); these tests pin the behaviours so a refactor
or "cleanup" cannot silently lose one.
"""
import pytest

from consensus import (_match_key, _normalise, phonetic_fold, scan_fold,
                       similar_spoken, token_containment, _cluster)


# --- scan_fold: the OG scan's confusable letter groups ----------------------

@pytest.mark.parametrize("scanned, engraved", [
    ("iown", "town"),        # T -> I
    ("idves", "loves"),      # L -> I, O -> D
    ("idvfs", "loves"),      # ... and E -> F
    ("ihe", "the"),          # T -> I ("'IHE" after normalise)
    ("aiaska", "alaska"),    # L -> I
    ("jil", "jll"),          # J -> I
    ("qtto", "otto"),        # Q -> O
    ("5tan", "stan"),        # 5 -> S
    ("8ob", "bob"),          # 8 -> B
    ("d0nna", "donna"),      # 0 -> O
])
def test_scan_fold_confusables_compare_equal(scanned, engraved):
    assert scan_fold(scanned) == scan_fold(engraved)


def test_scan_fold_does_not_merge_different_names():
    assert scan_fold("miller") != scan_fold("wilson")


# --- phonetic_fold: voice-transcribed spelling variants ----------------------

@pytest.mark.parametrize("a, b", [
    ("brian", "bryan"),
    ("cathy", "kathy"),
    ("britany", "brittany"),   # doubled-letter collapse
    ("philip", "filip"),       # ph -> f
    ("jack", "jak"),           # ck -> k
    ("zusan", "susan"),        # z -> s
])
def test_phonetic_fold_variants_compare_equal(a, b):
    assert phonetic_fold(a) == phonetic_fold(b)
    assert similar_spoken(a, b) == 1.0


def test_phonetic_fold_keeps_different_names_apart():
    assert phonetic_fold("brian") != phonetic_fold("karen")
    assert similar_spoken("brian", "karen") < 0.8


# --- _normalise / _match_key -------------------------------------------------

def test_normalise_strips_punctuation_and_case():
    assert _normalise("THE LEON H. LAVIGNE / FAMILY EST. 1969") == \
        "the leon h lavigne family est 1969"


def test_match_key_drops_dedication_boilerplate():
    # "IN MEMORY OF" appears on many bricks; only distinctive words remain.
    assert _match_key(_normalise("IN LOVING MEMORY OF WANDA")) == "wanda"


def test_match_key_never_returns_empty_for_boilerplate_only():
    norm = _normalise("IN MEMORY OF")
    assert _match_key(norm) == norm   # falls back to the full text


# --- token_containment: worn-brick partial reads ------------------------------

REF = _match_key(_normalise("HAROLD G BEATY 1938-1991 MY 'BUDDY'"))


def test_containment_scores_distinctive_subset_high():
    read = _match_key(_normalise("BEATY BUDDY 1938"))
    assert token_containment(read, REF) > 0.95


def test_containment_rejects_single_word_reads():
    # One word, however exact, is not identification.
    assert token_containment("beaty", REF) == 0.0


def test_containment_ignores_short_read_words():
    # "g" / "my" are under 3 chars; only 1 usable word remains -> 0.0.
    assert token_containment("g my beaty", REF) == 0.0


def test_containment_empty_reference():
    assert token_containment("beaty buddy", "") == 0.0


# --- _cluster: reads of one brick group together ------------------------------

def _item(text):
    norm = _normalise(text)
    return {"text": text, "norm": norm, "key": _match_key(norm), "method": "m"}


def test_cluster_groups_same_brick_and_splits_different():
    items = [_item("IN MEMORY OF WANDA SMITH"),
             _item("IN MEMORY OF WANDA SMTH"),    # noisy read, same brick
             _item("IN MEMORY OF SID JOHNSON")]   # same boilerplate, other brick
    clusters = _cluster(items)
    assert sorted(len(c) for c in clusters) == [1, 2]
