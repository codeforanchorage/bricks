"""Tests for merge_lists.py: the E/F/G number-range rule, the text join,
NO BRICK handling, identical-copy spreading, and the unclaimed report.

The end-to-end test runs merge_lists.main() on tiny synthetic lists so the
whole merge path runs exactly as the real build does.
"""
import csv

import pytest

import merge_lists


# --- the key rule: unmoved-area section from the original number -------------

@pytest.mark.parametrize("orig_id, section", [
    (1, "F"), (3377, "F"),          # F range boundaries
    (3378, None), (8278, None),     # the moved gap between F and G
    (8279, "G"), (9126, "G"),       # G range boundaries
    (9127, "E"), (10070, "E"),      # E range boundaries
    (10071, None), (13344, None),   # above E: moved
])
def test_unmoved_section_ranges(orig_id, section):
    assert merge_lists._unmoved_section(orig_id) == section


# --- surname corroboration for the rescue pass --------------------------------

def test_surname_corroborates_fuzzy_hit():
    candidate = {"_key": merge_lists._key("DAVID FISHER FAMILY 1990"),
                 "key_word": "FISHER"}
    assert merge_lists._surname_corroborates("FISaIER DAVID", candidate)


def test_surname_rejects_unrelated_name():
    candidate = {"_key": merge_lists._key("DAVID MITCHELL FAMILY 1990"),
                 "key_word": "MITCHELL"}
    assert not merge_lists._surname_corroborates("FISHER DAVID", candidate)


# --- end-to-end merge on synthetic lists --------------------------------------

OG_COLS = ["section", "assigned_id", "pos1", "pos2", "full_name", "key_word",
           "alt_name", "verified", "flag"]
NEW_COLS = ["section", "assigned_id", "pos1", "pos2", "full_name", "key_word"]


def _og(assigned_id, full_name, key_word="", alt_name="", verified="agreed",
        flag=""):
    return {"section": "", "assigned_id": assigned_id, "pos1": "", "pos2": "",
            "full_name": full_name, "key_word": key_word,
            "alt_name": alt_name, "verified": verified, "flag": flag}


def _new(section, assigned_id, full_name, key_word=""):
    return {"section": section, "assigned_id": assigned_id, "pos1": "",
            "pos2": "", "full_name": full_name, "key_word": key_word}


def _run_merge(tmp_path, og_rows, new_rows):
    """Run merge_lists.main() on synthetic lists -> (master, rows, unclaimed)."""
    og_path, new_path = tmp_path / "og.csv", tmp_path / "new.csv"
    for path, cols, rows in ((og_path, OG_COLS, og_rows),
                             (new_path, NEW_COLS, new_rows)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    out = tmp_path / "master.csv"
    merge_lists.main(["--og", str(og_path), "--new", str(new_path),
                      "--output", str(out)])
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    master = {r["orig_id"]: r for r in rows}   # dup ids: last row wins here
    with open(tmp_path / "master_unclaimed.csv", newline="",
              encoding="utf-8") as f:
        unclaimed = list(csv.DictReader(f))
    return master, rows, unclaimed


@pytest.fixture()
def merged(tmp_path):
    og_rows = [
        _og("100", "JOHN DOE FAMILY", "DOE"),                # unmoved -> F
        _og("9500", "ALPHA BETA HOMESTEAD", "ALPHA"),        # unmoved -> E
        _og("5000", "SUNSHINE BAKERY EST 1990", "SUNSHINE",  # moved, clean join;
            verified="strip", flag="number?:4999"),          #   v2 markers carry
        _og("5001", "NO BRICK NO INSCRIPTION", "SMITH"),     # no-brick sale
        _og("5002", "ZZQQ XXYY", "NOBODY"),                  # moved, no match
        _og("5003", "MAGIC FM RADIO", "KMAG"),               # identical copies:
        _og("5004", "MAGIC FM RADIO", "KMAG"),               #   2 new-list rows,
        _og("5005", "MAGIC FM RADIO", "KMAG"),               #   3 claimants
        _og("150", "DUP COPY ONE", "AAA"),                   # scan id collision:
        _og("150", "DUP COPY TWO", "BBB"),                   #   both in F range
        _og("26955", "WAY OUT OF RANGE", "OOR"),             # impossible orig id
    ]
    new_rows = [
        _new("A", "77", "SUNSHINE BAKERY EST 1990", "SUNSHINE"),
        _new("B", "200", "MAGIC FM RADIO", "KMAG"),
        _new("B", "201", "MAGIC FM RADIO", "KMAG"),
        _new("C", "300", "UNCLAIMED BRICK TEXT", "NOONE"),
    ]
    return _run_merge(tmp_path, og_rows, new_rows)


def test_unmoved_rows_get_section_from_number(merged):
    master, _, _ = merged
    assert master["100"]["section"] == "F"
    assert master["100"]["new_id"] == "100"     # number still valid
    assert master["100"]["moved"] == "no"
    assert master["9500"]["section"] == "E"


def test_moved_row_joined_by_text(merged):
    master, _, _ = merged
    row = master["5000"]
    assert row["moved"] == "yes"
    assert row["new_id"] == "77"
    assert row["section"] == "A"
    assert row["status"] == "ok"


def test_no_brick_sale_flagged(merged):
    master, _, _ = merged
    assert master["5001"]["status"] == "no_brick"
    assert master["5001"]["new_id"] == ""


def test_unjoinable_row_lands_in_review(merged):
    master, _, _ = merged
    assert master["5002"]["status"] == "unjoined"


def test_identical_copies_spread_one_to_one(merged):
    master, _, _ = merged
    ids = {master["5003"]["new_id"], master["5004"]["new_id"]}
    assert ids == {"200", "201"}                # spread, not shared
    # The third claimant has no distinct counterpart -> review, not silently
    # sharing a copy.
    assert master["5005"]["status"] == "unjoined"
    assert master["5005"]["new_id"] == ""


def test_unclaimed_new_rows_reported(merged):
    _, _, unclaimed = merged
    assert [r["new_id"] for r in unclaimed] == ["300"]


# --- v2 trust markers and merge-level validation flags ------------------------

def test_v2_verified_and_flag_carry_through(merged):
    master, _, _ = merged
    row = master["5000"]
    assert row["og_verified"] == "strip"
    assert "number?:4999" in row["flag"].split(";")
    # Clean rows carry their tier too, with an empty flag.
    assert master["100"]["og_verified"] == "agreed"
    assert master["100"]["flag"] == ""


def test_duplicate_orig_ids_flagged_on_every_copy(merged):
    _, rows, _ = merged
    dups = [r for r in rows if r["orig_id"] == "150"]
    assert len(dups) == 2                        # both copies kept...
    for r in dups:
        assert "dup_orig" in r["flag"].split(";")   # ...and both flagged
        # Best-effort assignment is kept -- the flag routes to review,
        # it does not withhold the E/F/G number-rule result.
        assert r["section"] == "F"
        assert r["new_id"] == "150"


def test_impossible_orig_id_flagged(merged):
    master, _, _ = merged
    row = master["26955"]
    assert "orig_range" in row["flag"].split(";")
    assert row["status"] == "unjoined"           # its text found no match


# --- the rescue pass and identical copies --------------------------------------
#
# 'HANSEN FMLY BKRY EST 195' scores 0.78 against the clean inscription --
# under the 0.80 join bar, inside the [0.70, 0.80) rescue window -- with a
# 0.16 margin over the best DIFFERENT inscription (SUNSHINE BAKERY, 0.63).
# The brick was sold in two identical copies: before the same-key runner-up
# fix, the copies tied with each other, zeroed the margin, and the rescue
# could never fire for any batch brick.

RESCUE_NEW = [
    _new("D", "400", "THE HANSEN FAMILY BAKERY EST 1985", "HANSEN"),
    _new("D", "401", "THE HANSEN FAMILY BAKERY EST 1985", "HANSEN"),
    _new("A", "77", "SUNSHINE BAKERY EST 1990", "SUNSHINE"),
]


def test_rescue_fires_for_identical_copy_batches(tmp_path):
    master, _, _ = _run_merge(
        tmp_path,
        [_og("5100", "HANSEN FMLY BKRY EST 195", "HANSEN GRETA")],
        RESCUE_NEW)
    row = master["5100"]
    assert row["status"] == "ok"                 # rescued, not unjoined
    assert row["new_id"] in {"400", "401"}
    assert row["section"] == "D"
    assert 0.70 <= float(row["join_score"]) < 0.80   # via the rescue window


def test_rescue_still_requires_surname_corroboration(tmp_path):
    # Same noisy text and margin, but the buyer's surname does not appear in
    # the candidate -- the rescue must NOT fire (this is the gate that
    # rejects lookalike pairs).
    master, _, _ = _run_merge(
        tmp_path,
        [_og("5100", "HANSEN FMLY BKRY EST 195", "ZZZZZ AAA")],
        RESCUE_NEW)
    assert master["5100"]["status"] == "unjoined"


def test_copy_spread_awards_by_score_not_id(tmp_path):
    # The real MALONEY case: two 0.9 claims own the two "KARRI MALONEY"
    # copies; a 0.71 same-surname rescue of "BRIDGET A MALONEY" (whose brick
    # has NO new-list row) lands on the same brick. Score-ordered spread
    # keeps both rightful owners and overflows the weak claim to review --
    # id-ordered spread gave Bridget a Karri brick and evicted an owner.
    master, _, _ = _run_merge(
        tmp_path,
        [_og("7000", "Bl KARRI MAIDNEY", "LENHART BILL"),
         _og("7001", "Bl KARRI MAIDNEY", "LENHART VELMA"),
         _og("7002", "BRIDGET A MAIONEY", "MAIONEY DEBBIE")],
        [_new("H", "400", "KARRI MALONEY", "MALONEY"),
         _new("B", "401", "KARRI MALONEY", "MALONEY"),
         _new("A", "77", "SUNSHINE BAKERY EST 1990", "SUNSHINE")])
    assert {master["7000"]["new_id"], master["7001"]["new_id"]} == \
        {"400", "401"}                            # rightful owners keep both
    assert master["7002"]["status"] == "unjoined"  # weak rescue -> review
