"""Tests for match.py: reference loading, scoring, and the acceptance rules.

The synthetic end-to-end tests run match.main() on tiny CSVs so the whole
path (loading, blocking, scoring, acceptance, output) is exercised exactly as
production runs it.
"""
import csv

import pytest

import match
from consensus import _match_key, _normalise


# --- _best_match: containment margin and duplicate-copy handling -------------

def _ref(key, key2=""):
    return {"_key": key, "_key2": key2, "assigned_id": "1", "section": "A",
            "full_name": key, "key_word": ""}


def test_generic_token_set_ties_and_earns_no_margin():
    # "smith alaska" is contained in BOTH inscriptions -> the two candidates
    # tie, so the margin is ~0 and main() must reject a tokens-basis match.
    refs = [_ref(_match_key(_normalise("JOHN SMITH FAMILY OF ALASKA 1959"))),
            _ref(_match_key(_normalise("MARY SMITH FAMILY ALASKA HOMESTEAD")))]
    score, basis, margin, ref, _read = match._best_match(
        ["SMITH FAMILY ALASKA"], refs)
    assert basis == "tokens"
    assert margin < match.CONTAIN_MARGIN


def test_identical_copies_do_not_veto_their_own_match():
    # A brick sold in identical copies: the duplicate must NOT count as the
    # runner-up, or every batch brick would self-veto its containment margin.
    copy = _match_key(_normalise("MAGIC FM RADIO ANCHORAGE"))
    other = _match_key(_normalise("COMPLETELY DIFFERENT WORDS HERE"))
    refs = [_ref(copy), _ref(copy), _ref(other)]
    score, basis, margin, ref, _read = match._best_match(["MAGIC FM RADIO"],
                                                         refs)
    assert ref["_key"] == copy
    assert basis == "tokens"
    assert margin >= match.CONTAIN_MARGIN


def test_distinctive_subset_wins_with_margin():
    refs = [_ref(_match_key(_normalise("HAROLD G BEATY 1938-1991 MY BUDDY"))),
            _ref(_match_key(_normalise("THE JOHNSON FAMILY OF ANCHORAGE")))]
    score, basis, margin, ref, _read = match._best_match(["BEATY BUDDY 1938"],
                                                         refs)
    assert basis == "tokens"
    assert score >= match.DEFAULT_MIN_SCORE - match.CONTAIN_DISCOUNT
    assert margin >= match.CONTAIN_MARGIN
    assert "beaty" in ref["_key"]


# --- _load_reference: master-list adaptation ---------------------------------

def _write_master(path, rows):
    cols = ["orig_id", "new_id", "section", "moved", "status", "buyer",
            "og_inscription", "og_alt", "new_inscription", "join_score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_load_reference_prefers_new_inscription_and_keeps_alt(tmp_path):
    p = tmp_path / "master.csv"
    _write_master(p, [{
        "orig_id": "42", "new_id": "7", "section": "A", "moved": "yes",
        "status": "ok", "buyer": "DOE",
        "og_inscription": "OOE FAMIIY HOMESIEAD 1959",  # noisy scan text
        "og_alt": "DOE FAMILY HOMESTEAD",               # vision re-read
        "new_inscription": "Doe Family Homestead 1959", # clean xls text
        "join_score": "0.92"}])
    refs = match._load_reference(p, section="", scan_ocr=True)
    assert len(refs) == 1
    assert refs[0]["assigned_id"] == "7"             # new id preferred
    assert refs[0]["_key"]                            # from new_inscription
    # og_alt kept as a distinct second key (folded differently than _key)
    assert refs[0]["_key2"] and refs[0]["_key2"] != refs[0]["_key"]


def test_load_reference_drops_wrong_row_alt(tmp_path):
    # When the two transcriptions barely resemble each other, one read the
    # wrong row of the scan -- the alt must be ignored, not matched against.
    p = tmp_path / "master.csv"
    _write_master(p, [{
        "orig_id": "42", "new_id": "", "section": "F", "moved": "no",
        "status": "ok", "buyer": "DOE",
        "og_inscription": "THE DOE FAMILY HOMESTEAD",
        "og_alt": "GENERAL ELECTRIC SUPPLY CO",       # a neighbour's row
        "new_inscription": "", "join_score": ""}])
    refs = match._load_reference(p, section="", scan_ocr=True)
    assert refs[0]["_key2"] == ""


# --- end-to-end on synthetic CSVs ---------------------------------------------

def _run_match(tmp_path, catalog_rows, ref_rows, extra_args=()):
    catalog = tmp_path / "catalog.csv"
    with open(catalog, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "brick_id", "x", "y", "w",
                                          "h", "gemini-flash", "status"])
        w.writeheader()
        w.writerows(catalog_rows)
    reference = tmp_path / "reference.csv"
    with open(reference, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "assigned_id", "pos1",
                                          "pos2", "full_name", "key_word"])
        w.writeheader()
        w.writerows(ref_rows)
    output = tmp_path / "matched.csv"
    match.main(["--catalog", str(catalog), "--reference", str(reference),
                "--output", str(output), *extra_args])
    with open(output, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_scan_ocr_flag_bridges_scan_noise(tmp_path):
    rows = _run_match(
        tmp_path,
        [{"image": "a.jpg", "brick_id": 1, "gemini-flash": "LOVES ALASKA"}],
        [{"section": "F", "assigned_id": "10", "full_name": "IDVFS AIASKA",
          "key_word": "X"}],
        extra_args=("--scan-ocr",))
    assert rows[0]["match_status"] == "matched"
    assert rows[0]["official_id"] == "10"


def test_error_reads_are_ignored(tmp_path):
    rows = _run_match(
        tmp_path,
        [{"image": "a.jpg", "brick_id": 1,
          "gemini-flash": "ERROR: 429 rate limited"}],
        [{"section": "F", "assigned_id": "10",
          "full_name": "ERROR 429 RATE LIMITED", "key_word": "X"}])
    assert rows[0]["match_status"] == "unmatched"


def test_section_filter_writes_missing_report(tmp_path):
    rows = _run_match(
        tmp_path,
        [{"image": "a.jpg", "brick_id": 1, "gemini-flash": "ALPHA BRAVO"}],
        [{"section": "A", "assigned_id": "1", "full_name": "ALPHA BRAVO",
          "key_word": "X"},
         {"section": "A", "assigned_id": "2", "full_name": "CHARLIE DELTA",
          "key_word": "Y"},
         {"section": "B", "assigned_id": "3", "full_name": "ECHO FOXTROT",
          "key_word": "Z"}],
        extra_args=("--section", "A"))
    assert rows[0]["match_status"] == "matched"
    missing = tmp_path / "missing_A.csv"
    assert missing.is_file()
    with open(missing, newline="", encoding="utf-8") as f:
        ids = [r["assigned_id"] for r in csv.DictReader(f)]
    assert ids == ["2"]   # only the unphotographed section-A brick
