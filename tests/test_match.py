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


def test_rejected_tokens_winner_falls_back_to_valid_text_match():
    # Two long inscriptions both fully contain the read's words (tokens 1.0,
    # tied -> margin 0, rejected), while the TRUE brick is a hair lower on
    # whole-string similarity. The rejected containment winner must not drag
    # the photo to review: the text match clears the bar on its own.
    read = "ANNA BELL"
    true_key = _match_key(_normalise("ANNA BELLE"))
    refs = [_ref(_match_key(_normalise("ANNA AND BELL SMITH FAMILY 1959"))),
            _ref(_match_key(_normalise("ANNA BELL JOHNSON MEMORIAL"))),
            _ref(true_key)]
    score, basis, margin, ref, _read = match._best_match([read], refs)
    assert basis == "text"
    assert ref["_key"] == true_key
    assert score >= match.DEFAULT_MIN_SCORE


def test_no_text_fallback_when_nothing_clears_the_bar():
    # Same tie, but no valid text candidate exists: the tied tokens result
    # comes back (and main() rejects it on margin) -- the fallback must not
    # invent a match from a sub-threshold text score.
    refs = [_ref(_match_key(_normalise("ANNA AND BELL SMITH FAMILY 1959"))),
            _ref(_match_key(_normalise("ANNA BELL JOHNSON MEMORIAL")))]
    score, basis, margin, ref, _read = match._best_match(["ANNA BELL"], refs)
    assert basis == "tokens"
    assert margin < match.CONTAIN_MARGIN


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


def test_tree_sponsor_header_is_stripped_from_matching(tmp_path):
    # The engraved "TREE SPONSOR" header is boilerplate: without stripping
    # it, the read fails whole-string similarity against the right row
    # (which lacks the words) and review fills with OTHER tree sponsors.
    rows = _run_match(
        tmp_path,
        [{"image": "p.jpg", "brick_id": 1,
          "gemini-flash": "TREE SPONSOR / JOHN C. STEPP SR"}],
        [{"section": "F", "assigned_id": "1634",
          "full_name": "JOHN C STEPP SR", "key_word": "STEPP"},
         {"section": "F", "assigned_id": "9", "full_name":
          "TREE SPONSOR ALYESKA TITLE CO", "key_word": "ALYESKA"}])
    assert rows[0]["match_status"] == "matched"
    assert rows[0]["official_id"] == "1634"

    # A read that is ONLY the header must not match anything.
    rows = _run_match(
        tmp_path,
        [{"image": "p2.jpg", "brick_id": 1, "gemini-flash": "TREE SPONSOR"}],
        [{"section": "F", "assigned_id": "9",
          "full_name": "TREE SPONSOR ALYESKA TITLE CO",
          "key_word": "ALYESKA"}])
    assert rows[0]["match_status"] == "unmatched"


def test_master_display_prefers_clean_alt_for_humans(tmp_path):
    # og-only row (unmoved section F): matching keys use both texts, but the
    # human-facing official_name shows the clean re-read, not the parse.
    p = tmp_path / "master.csv"
    _write_master(p, [{
        "orig_id": "1375", "new_id": "", "section": "F", "moved": "no",
        "status": "ok", "buyer": "CONDY SYLVIA",
        "og_inscription": "THE WOOFIER FAMIIX",
        "og_alt": "THE WOOFTER FAMILY",
        "new_inscription": "", "join_score": ""}])
    refs = match._load_reference(p, section="", scan_ocr=True)
    assert match._display(refs[0]) == "THE WOOFTER FAMILY"
    # A wrong-row alt is not shown.
    _write_master(p, [{
        "orig_id": "2", "new_id": "", "section": "F", "moved": "no",
        "status": "ok", "buyer": "X",
        "og_inscription": "THE SMITH FAMILY",
        "og_alt": "GENERAL ELECTRIC SUPPLY CO",
        "new_inscription": "", "join_score": ""}])
    refs = match._load_reference(p, section="", scan_ocr=True)
    assert match._display(refs[0]) == "THE SMITH FAMILY"


# --- end-to-end on synthetic CSVs ---------------------------------------------

def _run_match(tmp_path, catalog_rows, ref_rows, extra_args=()):
    catalog = tmp_path / "catalog.csv"
    with open(catalog, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "brick_id", "section",
                                          "pallet", "x", "y", "w",
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


def test_text_fallback_end_to_end(tmp_path):
    # The JOLY COY case from the Pallet K test run, synthesized: the photo
    # must auto-match the text candidate, not land in review.
    rows = _run_match(
        tmp_path,
        [{"image": "p.jpg", "brick_id": 1, "gemini-flash": "ANNA BELL"}],
        [{"section": "K", "assigned_id": "1",
          "full_name": "ANNA AND BELL SMITH FAMILY 1959", "key_word": "X"},
         {"section": "K", "assigned_id": "2",
          "full_name": "ANNA BELL JOHNSON MEMORIAL", "key_word": "Y"},
         {"section": "K", "assigned_id": "3", "full_name": "ANNA BELLE",
          "key_word": "Z"}])
    assert rows[0]["match_status"] == "matched"
    assert rows[0]["official_id"] == "3"
    assert rows[0]["match_basis"] == "text"


# --- per-row section scoping (the catalogue's section/pallet columns) ---------

def test_section_tag_picks_the_copy_in_its_own_section(tmp_path):
    # The same inscription exists in A and B; the photo's pallet folder says
    # B, so the B copy must win (untagged matching would take the first).
    rows = _run_match(
        tmp_path,
        [{"image": "B/pallet-1/p.jpg", "brick_id": 1, "section": "B",
          "pallet": "pallet-1", "gemini-flash": "MAGIC RADIO"}],
        [{"section": "A", "assigned_id": "1", "full_name": "MAGIC RADIO",
          "key_word": "KMAG"},
         {"section": "B", "assigned_id": "2", "full_name": "MAGIC RADIO",
          "key_word": "KMAG"}])
    assert rows[0]["match_status"] == "matched"
    assert rows[0]["official_id"] == "2"
    assert rows[0]["official_section"] == "B"
    assert rows[0]["section_check"] == "ok"
    assert rows[0]["pallet"] == "pallet-1"


def test_off_section_match_is_kept_and_flagged(tmp_path):
    # The brick's inscription exists only in section B, but the photo is
    # tagged A (mis-sorted brick or mis-tagged folder): the global fallback
    # must still find it, flagged off-section.
    rows = _run_match(
        tmp_path,
        [{"image": "A/pallet-2/p.jpg", "brick_id": 1, "section": "A",
          "pallet": "pallet-2", "gemini-flash": "ECHO FOXTROT"}],
        [{"section": "B", "assigned_id": "3", "full_name": "ECHO FOXTROT",
          "key_word": "Z"}])
    assert rows[0]["match_status"] == "matched"
    assert rows[0]["official_id"] == "3"
    assert rows[0]["section_check"] == "off-section"


def test_sectionless_reference_rows_are_in_every_scope(tmp_path):
    # An unjoined master row (no section assigned) could be on any pallet,
    # so a tagged photo must still be able to match it -- without the
    # off-section flag, since there is no contradiction to report.
    rows = _run_match(
        tmp_path,
        [{"image": "A/pallet-1/p.jpg", "brick_id": 1, "section": "A",
          "pallet": "pallet-1", "gemini-flash": "GOLF HOTEL"}],
        [{"section": "", "assigned_id": "9", "full_name": "GOLF HOTEL",
          "key_word": "G"},
         {"section": "B", "assigned_id": "3", "full_name": "ECHO FOXTROT",
          "key_word": "Z"}])
    assert rows[0]["match_status"] == "matched"
    assert rows[0]["official_id"] == "9"
    assert rows[0]["section_check"] == "unassigned"


def test_review_candidates_come_from_the_tagged_section(tmp_path):
    # A worn, unmatchable read on an A pallet: the review queue must offer
    # section-A bricks, not a lookalike from section B (the global retry
    # already had its chance to accept a cross-section match).
    _run_match(
        tmp_path,
        [{"image": "A/pallet-1/worn.jpg", "brick_id": 1, "section": "A",
          "pallet": "pallet-1", "gemini-flash": "MAGC RASIO ZZZZZ QQQQ"}],
        [{"section": "A", "assigned_id": "1", "full_name": "ALPHA BRAVO",
          "key_word": "AB"},
         {"section": "A", "assigned_id": "2", "full_name": "CHARLIE DELTA",
          "key_word": "CD"},
         {"section": "B", "assigned_id": "3", "full_name": "MAGIC RADIO",
          "key_word": "KMAG"}])
    with open(tmp_path / "review_matched.csv", newline="",
              encoding="utf-8") as f:
        cand = list(csv.DictReader(f))
    assert cand
    assert all(c["official_section"] == "A" for c in cand)
    assert all(c["section"] == "A" and c["pallet"] == "pallet-1"
               for c in cand)


def test_global_section_flag_overrides_row_tags(tmp_path):
    # --section A loads only section A; a row mis-tagged B must still match
    # within it, with no per-row scoping applied (section_check stays "").
    rows = _run_match(
        tmp_path,
        [{"image": "p.jpg", "brick_id": 1, "section": "B",
          "gemini-flash": "ALPHA BRAVO"}],
        [{"section": "A", "assigned_id": "1", "full_name": "ALPHA BRAVO",
          "key_word": "AB"},
         {"section": "B", "assigned_id": "3", "full_name": "ECHO FOXTROT",
          "key_word": "Z"}],
        extra_args=("--section", "A"))
    assert rows[0]["match_status"] == "matched"
    assert rows[0]["official_id"] == "1"
    assert rows[0]["section_check"] == ""


# --- per-section missing reports from catalogue tags ---------------------------

def test_tagged_catalog_writes_missing_report_per_photographed_section(tmp_path):
    # No --section flag: every section with tagged photos gets a
    # missing_<S>.csv; a section nobody photographed gets none.
    _run_match(
        tmp_path,
        [{"image": "A/pallet-1/p1.jpg", "brick_id": 1, "section": "A",
          "pallet": "pallet-1", "gemini-flash": "ALPHA BRAVO"},
         {"image": "H/pallet-9/p2.jpg", "brick_id": 1, "section": "H",
          "pallet": "pallet-9", "gemini-flash": "GOLF HOTEL"}],
        [{"section": "A", "assigned_id": "1", "full_name": "ALPHA BRAVO",
          "key_word": "AB"},
         {"section": "A", "assigned_id": "2", "full_name": "CHARLIE DELTA",
          "key_word": "CD"},
         {"section": "H", "assigned_id": "7", "full_name": "GOLF HOTEL",
          "key_word": "GH"},
         {"section": "B", "assigned_id": "3", "full_name": "ECHO FOXTROT",
          "key_word": "Z"}])
    with open(tmp_path / "missing_A.csv", newline="", encoding="utf-8") as f:
        ids = [r["assigned_id"] for r in csv.DictReader(f)]
    assert ids == ["2"]                     # the unphotographed A brick
    with open(tmp_path / "missing_H.csv", newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == []   # H fully accounted for
    assert not (tmp_path / "missing_B.csv").exists()


def test_missing_sections_flag_forces_reports_without_tags(tmp_path):
    # Pallet-only layouts carry no section tags; --missing-sections forces
    # the report for sections believed fully photographed.
    _run_match(
        tmp_path,
        [{"image": "pallets/F1/p1.jpg", "brick_id": 1, "pallet": "F1",
          "gemini-flash": "ALPHA BRAVO"}],
        [{"section": "F", "assigned_id": "1", "full_name": "ALPHA BRAVO",
          "key_word": "AB"},
         {"section": "F", "assigned_id": "2", "full_name": "CHARLIE DELTA",
          "key_word": "CD"}],
        extra_args=("--missing-sections", "f"))
    with open(tmp_path / "missing_F.csv", newline="", encoding="utf-8") as f:
        ids = [r["assigned_id"] for r in csv.DictReader(f)]
    assert ids == ["2"]


def test_untagged_catalog_writes_no_missing_reports(tmp_path):
    _run_match(
        tmp_path,
        [{"image": "p1.jpg", "brick_id": 1, "gemini-flash": "ALPHA BRAVO"}],
        [{"section": "A", "assigned_id": "1", "full_name": "ALPHA BRAVO",
          "key_word": "AB"}])
    assert not list(tmp_path.glob("missing_*.csv"))


def test_missing_report_keys_on_section_and_id(tmp_path):
    # The same numeric id exists in A and B (cross-era collision). Matching
    # A#5 must NOT mark B#5 as photographed.
    _run_match(
        tmp_path,
        [{"image": "A/pallet-1/p1.jpg", "brick_id": 1, "section": "A",
          "pallet": "pallet-1", "gemini-flash": "ALPHA BRAVO"},
         {"image": "B/pallet-2/p2.jpg", "brick_id": 1, "section": "B",
          "pallet": "pallet-2", "gemini-flash": "ZZZZ QQQQ XXXX"}],
        [{"section": "A", "assigned_id": "5", "full_name": "ALPHA BRAVO",
          "key_word": "AB"},
         {"section": "B", "assigned_id": "5", "full_name": "ECHO FOXTROT",
          "key_word": "Z"}])
    with open(tmp_path / "missing_B.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(r["section"], r["assigned_id"]) for r in rows] == [("B", "5")]
    with open(tmp_path / "missing_A.csv", newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == []


def test_off_section_match_counts_for_the_bricks_real_section(tmp_path):
    # A photo tagged A matches a brick that actually lives in B (off-section).
    # B's missing report -- triggered by another B photo -- must not list
    # that brick: it has been photographed, wherever it was found.
    _run_match(
        tmp_path,
        [{"image": "A/pallet-1/p1.jpg", "brick_id": 1, "section": "A",
          "pallet": "pallet-1", "gemini-flash": "ECHO FOXTROT"},
         {"image": "B/pallet-2/p2.jpg", "brick_id": 1, "section": "B",
          "pallet": "pallet-2", "gemini-flash": "MAGIC RADIO"}],
        [{"section": "B", "assigned_id": "3", "full_name": "ECHO FOXTROT",
          "key_word": "Z"},
         {"section": "B", "assigned_id": "4", "full_name": "MAGIC RADIO",
          "key_word": "KMAG"}])
    with open(tmp_path / "missing_B.csv", newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == []


# --- the review queue: top-N candidates for unmatched bricks -------------------

REVIEW_REFS = [
    {"section": "A", "assigned_id": "1", "full_name": "ALPHA BRAVO",
     "key_word": "AB"},
    {"section": "A", "assigned_id": "2", "full_name": "CHARLIE DELTA",
     "key_word": "CD"},
    {"section": "B", "assigned_id": "3", "full_name": "MAGIC RADIO",
     "key_word": "KMAG"},                       # identical copies: must
    {"section": "B", "assigned_id": "4", "full_name": "MAGIC RADIO",
     "key_word": "KMAG"},                       #   fill ONE review slot
]


def test_unmatched_brick_gets_ranked_candidates(tmp_path):
    rows = _run_match(
        tmp_path,
        [{"image": "good.jpg", "brick_id": 1, "gemini-flash": "ALPHA BRAVO"},
         {"image": "worn.jpg", "brick_id": 1, "gemini-flash": "MAGC RASIO ZZZZZ QQQQ"}],
        REVIEW_REFS)
    by_image = {r["image"]: r for r in rows}
    assert by_image["good.jpg"]["match_status"] == "matched"
    assert by_image["worn.jpg"]["match_status"] == "unmatched"

    review = tmp_path / "review_matched.csv"
    assert review.is_file()
    with open(review, newline="", encoding="utf-8") as f:
        cand = list(csv.DictReader(f))
    # Only the unmatched brick appears.
    assert {c["image"] for c in cand} == {"worn.jpg"}
    # 3 distinct inscriptions exist; the identical copies collapsed to one.
    assert len(cand) == 3
    assert sum(1 for c in cand if c["official_name"] == "MAGIC RADIO") == 1
    # Ranked 1..n, best first, and the near-miss is rank 1.
    assert [c["rank"] for c in cand] == ["1", "2", "3"]
    scores = [float(c["score"]) for c in cand]
    assert scores == sorted(scores, reverse=True)
    assert cand[0]["official_name"] == "MAGIC RADIO"


def test_top_limits_candidate_count(tmp_path):
    _run_match(
        tmp_path,
        [{"image": "worn.jpg", "brick_id": 1, "gemini-flash": "MAGC RASIO ZZZZZ QQQQ"}],
        REVIEW_REFS, extra_args=("--top", "2"))
    with open(tmp_path / "review_matched.csv", newline="",
              encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_top_zero_disables_review_file(tmp_path):
    _run_match(
        tmp_path,
        [{"image": "worn.jpg", "brick_id": 1, "gemini-flash": "MAGC RASIO ZZZZZ QQQQ"}],
        REVIEW_REFS, extra_args=("--top", "0"))
    assert not (tmp_path / "review_matched.csv").exists()


# --- the duplicate-claim QA report ---------------------------------------------

def test_two_photos_on_one_brick_reported(tmp_path):
    _run_match(
        tmp_path,
        [{"image": "p1.jpg", "brick_id": 1, "gemini-flash": "ALPHA BRAVO"},
         {"image": "p2.jpg", "brick_id": 1, "gemini-flash": "ALPHA BRAVO"},
         {"image": "p3.jpg", "brick_id": 1, "gemini-flash": "CHARLIE DELTA"}],
        REVIEW_REFS)
    dup = tmp_path / "duplicates_matched.csv"
    assert dup.is_file()
    with open(dup, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Only the doubly-claimed brick appears; the singleton (p3) does not.
    assert {r["image"] for r in rows} == {"p1.jpg", "p2.jpg"}
    assert all(r["official_id"] == "1" for r in rows)
    assert all(r["n_claims"] == "2" for r in rows)
    # A single-copy inscription with 2 claims: at least one claim is wrong.
    assert all(r["copies"] == "1" for r in rows)


def test_copy_count_recorded_for_batch_bricks(tmp_path):
    # Two photos of a 2-copy inscription both land on the same reference row
    # (deterministic argmax) -- reported, but copies=2 tells the reviewer
    # this is expected, not a false positive.
    _run_match(
        tmp_path,
        [{"image": "p1.jpg", "brick_id": 1, "gemini-flash": "MAGIC RADIO"},
         {"image": "p2.jpg", "brick_id": 1, "gemini-flash": "MAGIC RADIO"}],
        REVIEW_REFS)
    with open(tmp_path / "duplicates_matched.csv", newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(r["copies"] == "2" for r in rows)
    assert all(r["n_claims"] == "2" for r in rows)


def test_no_duplicates_no_file(tmp_path):
    _run_match(
        tmp_path,
        [{"image": "p1.jpg", "brick_id": 1, "gemini-flash": "ALPHA BRAVO"}],
        REVIEW_REFS)
    assert not (tmp_path / "duplicates_matched.csv").exists()
