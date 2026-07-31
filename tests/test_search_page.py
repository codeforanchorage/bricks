"""Tests for make_search_page.py: the pickup-counter search page build.

The search itself runs in the browser; these tests pin what the generator
bakes in -- the data rows, the photo/pallet join, and the fold tables
emitted from consensus.py (single source of truth, so the JS matching
cannot drift from the Python matcher's).
"""
import csv
import json
import re

import make_search_page
from consensus import _CONFUSABLE, _STOPWORDS

MASTER_COLS = ["orig_id", "new_id", "section", "moved", "status", "buyer",
               "og_inscription", "og_alt", "new_inscription", "join_score",
               "og_verified", "flag"]
MATCH_COLS = ["image", "brick_id", "section", "pallet", "match_status",
              "match_basis", "score", "official_id", "official_section",
              "section_check", "official_name", "official_keyword",
              "matched_read"]


def _write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _embedded(page, name):
    """Parse the JSON baked into `const <name> = ...;`."""
    match = re.search(rf"const {name} = (.*?);\s*//", page)
    assert match, f"{name} not found in page"
    payload = match.group(1)
    wrapper = re.fullmatch(r"new Set\((.*)\)", payload)
    return json.loads(wrapper.group(1) if wrapper else payload)


def _build(tmp_path, master_rows, matched_rows=None):
    master = tmp_path / "master.csv"
    _write(master, MASTER_COLS, master_rows)
    argv = ["--master", str(master),
            "--output", str(tmp_path / "search.html")]
    if matched_rows is not None:
        matched = tmp_path / "matched.csv"
        _write(matched, MATCH_COLS, matched_rows)
        argv += ["--matched", str(matched)]
    make_search_page.main(argv)
    return (tmp_path / "search.html").read_text(encoding="utf-8")


def test_page_bakes_rows_and_fold_tables(tmp_path):
    page = _build(tmp_path, [
        {"orig_id": "3770", "new_id": "7305", "section": "I", "moved": "yes",
         "status": "ok", "buyer": "WILLIAMS",
         "og_inscription": "TRAVIS E WILLIAMS",
         "new_inscription": "Travis E Williams"},
        {"orig_id": "2000", "new_id": "", "section": "F", "moved": "no",
         "status": "ok", "buyer": "COY", "og_inscription": "JOEY COY"},
    ])
    data = _embedded(page, "DATA")
    assert len(data) == 2
    assert data[0][:3] == ["3770", "7305", "I"]

    # Fold tables come straight from consensus.py -- byte-identical content.
    scan = _embedded(page, "SCAN")
    assert scan == {chr(k): v for k, v in _CONFUSABLE.items()}
    stop = _embedded(page, "STOP")
    assert set(stop) == set(_STOPWORDS)

    # The front-desk instructions and offline search shell are present.
    assert "Front desk" in page
    assert "attestation" in page
    assert "phoneticFold" in page


def test_display_text_prefers_clean_transcription(tmp_path):
    # The vision re-read (og_alt) displays when it resembles the parse;
    # a wrong-row alt is neither displayed nor indexed.
    page = _build(tmp_path, [
        {"orig_id": "1", "new_id": "", "section": "F", "moved": "no",
         "status": "ok", "buyer": "DOE",
         "og_inscription": "OOE FAMIIY HOMESIEAD",     # noisy parse
         "og_alt": "DOE FAMILY HOMESTEAD"},            # clean re-read
        {"orig_id": "2", "new_id": "", "section": "F", "moved": "no",
         "status": "ok", "buyer": "X",
         "og_inscription": "THE SMITH FAMILY",
         "og_alt": "GENERAL ELECTRIC SUPPLY"},         # wrong-row slip
    ])
    data = _embedded(page, "DATA")
    og_display = [r[6] for r in data]
    og_extra = [r[9] for r in data]
    assert og_display[0] == "DOE FAMILY HOMESTEAD"     # alt shown
    assert og_extra[0] == "OOE FAMIIY HOMESIEAD"       # parse still indexed
    assert og_display[1] == "THE SMITH FAMILY"         # parse shown
    assert og_extra[1] == ""                           # bad alt dropped


def test_photo_map_carries_the_ocr_read_for_display(tmp_path):
    page = _build(
        tmp_path,
        [{"orig_id": "5", "new_id": "", "section": "K", "moved": "no",
          "status": "ok", "buyer": "COY", "og_inscription": "IOFY C0Y"}],
        matched_rows=[
            {"image": "pallets/K/a.jpg", "brick_id": "1", "pallet": "K",
             "match_status": "matched", "official_id": "5",
             "official_section": "K", "matched_read": "JOEY COY"}])
    photos = _embedded(page, "PHOTOS")
    assert photos["K|5"][3] == "JOEY COY"
    # Verification panel wiring: relative photo base, toggle, strip row.
    assert "togglePanel" in page
    assert '/thumbs/' in page and '/strips/' in page
    # Image path stored thumb-ready: .JPG source -> .jpg derivative name.
    assert photos["K|5"][1].endswith(".jpg")


def test_photo_join_keys_on_section_and_id(tmp_path):
    page = _build(
        tmp_path,
        [{"orig_id": "5", "new_id": "", "section": "K", "moved": "no",
          "status": "ok", "buyer": "COY", "og_inscription": "JOEY COY"},
         {"orig_id": "9", "new_id": "5", "section": "A", "moved": "yes",
          "status": "ok", "buyer": "DOE", "og_inscription": "JANE DOE",
          "new_inscription": "Jane Doe"}],
        matched_rows=[
            {"image": "pallets/K/a.jpg", "brick_id": "1", "pallet": "K",
             "match_status": "matched", "official_id": "5",
             "official_section": "K"},
            {"image": "pallets/K/b.jpg", "brick_id": "1", "pallet": "K",
             "match_status": "unmatched", "official_id": "5",
             "official_section": "A"}])
    photos = _embedded(page, "PHOTOS")
    # Only the MATCHED row lands, keyed section|id so A#5 stays unphotographed.
    assert list(photos) == ["K|5"]
    assert photos["K|5"][0] == "K"
    assert "at pickup site" in page


def test_notes_and_unofficial_bricks_are_baked_in(tmp_path):
    # Applied catalogue in: review notes ride the photo map (searchable),
    # and human-confirmed no_match photos become unofficial entries.
    final_cols = MATCH_COLS + ["reviewer", "review_note"]
    master = tmp_path / "master.csv"
    _write(master, MASTER_COLS, [
        {"orig_id": "5", "new_id": "", "section": "K", "moved": "no",
         "status": "ok", "buyer": "COY", "og_inscription": "JOEY COY"}])
    matched = tmp_path / "final.csv"
    _write(matched, final_cols, [
        {"image": "pallets/K/a.jpg", "brick_id": "1", "pallet": "K",
         "match_status": "matched", "official_id": "5",
         "official_section": "K", "matched_read": "JOEY COY",
         "reviewer": "bb", "review_note": "chipped corner"},
        {"image": "pallets/F2/serff.jpg", "brick_id": "1", "pallet": "F2",
         "match_status": "no_match", "matched_read": "THE SERFF FAMILY",
         "reviewer": "bb", "review_note": "not in list, keep for pickup"}])
    make_search_page.main(["--master", str(master), "--matched",
                           str(matched),
                           "--output", str(tmp_path / "search.html")])
    page = (tmp_path / "search.html").read_text(encoding="utf-8")
    photos = _embedded(page, "PHOTOS")
    assert photos["K|5"][4] == "chipped corner"        # note rides along
    unofficial = _embedded(page, "UNOFFICIAL")
    assert unofficial == [["pallets/F2/serff.jpg", "F2",
                           "THE SERFF FAMILY", "not in list, keep for "
                           "pickup"]]
    assert "unofficial" in page                        # card + help text


def test_duplicate_photos_of_one_brick_are_counted_not_duplicated(tmp_path):
    page = _build(
        tmp_path,
        [{"orig_id": "5", "new_id": "", "section": "K", "moved": "no",
          "status": "ok", "buyer": "COY", "og_inscription": "JOEY COY"}],
        matched_rows=[
            {"image": "pallets/K/a.jpg", "brick_id": "1", "pallet": "K",
             "match_status": "matched", "official_id": "5",
             "official_section": "K"},
            {"image": "pallets/K/b.jpg", "brick_id": "1", "pallet": "K",
             "match_status": "matched", "official_id": "5",
             "official_section": "K"}])
    photos = _embedded(page, "PHOTOS")
    assert photos["K|5"][2] == 1     # one extra claim recorded, one entry
