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


def _og(assigned_id, full_name, key_word="", alt_name=""):
    return {"section": "", "assigned_id": assigned_id, "pos1": "", "pos2": "",
            "full_name": full_name, "key_word": key_word,
            "alt_name": alt_name, "verified": "agreed", "flag": ""}


def _new(section, assigned_id, full_name, key_word=""):
    return {"section": section, "assigned_id": assigned_id, "pos1": "",
            "pos2": "", "full_name": full_name, "key_word": key_word}


@pytest.fixture()
def merged(tmp_path):
    og_rows = [
        _og("100", "JOHN DOE FAMILY", "DOE"),                # unmoved -> F
        _og("9500", "ALPHA BETA HOMESTEAD", "ALPHA"),        # unmoved -> E
        _og("5000", "SUNSHINE BAKERY EST 1990", "SUNSHINE"), # moved, clean join
        _og("5001", "NO BRICK NO INSCRIPTION", "SMITH"),     # no-brick sale
        _og("5002", "ZZQQ XXYY", "NOBODY"),                  # moved, no match
        _og("5003", "MAGIC FM RADIO", "KMAG"),               # identical copies:
        _og("5004", "MAGIC FM RADIO", "KMAG"),               #   2 new-list rows,
        _og("5005", "MAGIC FM RADIO", "KMAG"),               #   3 claimants
    ]
    new_rows = [
        _new("A", "77", "SUNSHINE BAKERY EST 1990", "SUNSHINE"),
        _new("B", "200", "MAGIC FM RADIO", "KMAG"),
        _new("B", "201", "MAGIC FM RADIO", "KMAG"),
        _new("C", "300", "UNCLAIMED BRICK TEXT", "NOONE"),
    ]
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
        master = {r["orig_id"]: r for r in csv.DictReader(f)}
    with open(tmp_path / "master_unclaimed.csv", newline="",
              encoding="utf-8") as f:
        unclaimed = list(csv.DictReader(f))
    return master, unclaimed


def test_unmoved_rows_get_section_from_number(merged):
    master, _ = merged
    assert master["100"]["section"] == "F"
    assert master["100"]["new_id"] == "100"     # number still valid
    assert master["100"]["moved"] == "no"
    assert master["9500"]["section"] == "E"


def test_moved_row_joined_by_text(merged):
    master, _ = merged
    row = master["5000"]
    assert row["moved"] == "yes"
    assert row["new_id"] == "77"
    assert row["section"] == "A"
    assert row["status"] == "ok"


def test_no_brick_sale_flagged(merged):
    master, _ = merged
    assert master["5001"]["status"] == "no_brick"
    assert master["5001"]["new_id"] == ""


def test_unjoinable_row_lands_in_review(merged):
    master, _ = merged
    assert master["5002"]["status"] == "unjoined"


def test_identical_copies_spread_one_to_one(merged):
    master, _ = merged
    ids = {master["5003"]["new_id"], master["5004"]["new_id"]}
    assert ids == {"200", "201"}                # spread, not shared
    # The third claimant has no distinct counterpart -> review, not silently
    # sharing a copy.
    assert master["5005"]["status"] == "unjoined"
    assert master["5005"]["new_id"] == ""


def test_unclaimed_new_rows_reported(merged):
    _, unclaimed = merged
    assert [r["new_id"] for r in unclaimed] == ["300"]
