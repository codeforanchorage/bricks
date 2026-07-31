"""Tests for the review handoff: page generation, decision application,
and the Parks & Rec Excel report.
"""
import csv

import pytest

import apply_decisions
import make_report
import make_review_page

from match import MATCH_COLUMNS as MATCH_COLS
from match import REVIEW_COLUMNS as REVIEW_COLS


def _write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _photo(path):
    from PIL import Image
    Image.new("RGB", (200, 150), (120, 100, 90)).save(path, "JPEG")


# --- make_review_page ----------------------------------------------------------

def test_review_page_is_self_contained(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    _photo(photos / "worn.jpg")
    review = tmp_path / "review.csv"
    _write(review, REVIEW_COLS, [
        {"image": "worn.jpg", "brick_id": "1", "rank": "1", "score": "0.71",
         "basis": "tokens", "official_id": "11331", "official_section": "B",
         "official_name": 'KARRI "K" <MALONEY>', "official_keyword": "MALONEY",
         "matched_read": "KARRI MALONE"},
        {"image": "worn.jpg", "brick_id": "1", "rank": "2", "score": "0.60",
         "basis": "text", "official_id": "5407", "official_section": "H",
         "official_name": "Karri Maloney", "official_keyword": "Maloney",
         "matched_read": "KARRI MALONE"},
    ])
    out = tmp_path / "review.html"
    make_review_page.main(["--review", str(review), "--photos", str(photos),
                           "--output", str(out)])
    page = out.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in page      # photo embedded, no refs
    assert "#11331" in page and "#5407" in page   # both candidates offered
    assert "None of these" in page
    assert "&lt;MALONEY&gt;" in page              # HTML-escaped inscription
    assert "decisions.csv" in page                # export wiring present
    assert 'data-decision="match"' in page
    assert "clearItem" in page                    # per-photo undo control


def test_review_page_receiver_wiring(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    _photo(photos / "worn.jpg")
    review = tmp_path / "review.csv"
    _write(review, REVIEW_COLS, [
        {"image": "worn.jpg", "brick_id": "1", "rank": "1", "score": "0.7",
         "basis": "text", "official_id": "1", "official_section": "A",
         "official_name": "X", "official_keyword": "X", "matched_read": "X"}])
    out = tmp_path / "review.html"

    # Without the flags: no server config, download-only page.
    make_review_page.main(["--review", str(review), "--photos", str(photos),
                           "--output", str(out)])
    page = out.read_text(encoding="utf-8")
    assert "__RECEIVER__" not in page
    assert "const RECEIVER = null;" in page
    # The CSV line endings must be JS escape sequences, not raw newlines.
    assert "join('\\r\\n')" in page

    # With the flags: config embedded, autosave + final-send wiring present.
    make_review_page.main(["--review", str(review), "--photos", str(photos),
                           "--output", str(out),
                           "--receiver-url", "https://x.test/receiver.php",
                           "--receiver-token", "sekrit"])
    page = out.read_text(encoding="utf-8")
    assert '"url": "https://x.test/receiver.php"' in page
    assert '"token": "sekrit"' in page
    assert "scheduleUpload" in page

    # One flag without the other is a setup mistake, not a silent fallback.
    with pytest.raises(SystemExit):
        make_review_page.main(["--review", str(review), "--photos",
                               str(photos), "--output", str(out),
                               "--receiver-url", "https://x.test/r.php"])


def test_review_page_hosted_photo_mode(tmp_path):
    # --photo-base-url: no embedding, no local photos needed; lazy thumbs
    # link to the zoom derivative, and pallet-folder spaces are URL-quoted.
    review = tmp_path / "review.csv"
    _write(review, REVIEW_COLS, [
        {"image": "pallets/Pallet F1/GP010481.JPG", "brick_id": "1",
         "rank": "1", "score": "0.7", "basis": "text", "official_id": "1",
         "official_section": "A", "official_name": "X",
         "official_keyword": "X", "matched_read": "X"}])
    out = tmp_path / "review.html"
    make_review_page.main(["--review", str(review), "--output", str(out),
                           "--photo-base-url",
                           "https://bricks.example.com/photos/"])
    page = out.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," not in page
    assert ('src="https://bricks.example.com/photos/thumbs/'
            'pallets/Pallet%20F1/GP010481.jpg"') in page
    assert ('href="https://bricks.example.com/photos/zoom/'
            'pallets/Pallet%20F1/GP010481.jpg"') in page
    assert 'loading="lazy"' in page

    # Neither photo source is a setup mistake, not a blank page.
    with pytest.raises(SystemExit):
        make_review_page.main(["--review", str(review),
                               "--output", str(out)])


def test_review_page_shows_scanned_row_strips(tmp_path):
    # --master + hosted mode: each candidate shows the image of its row in
    # the scanned list, keyed by ORIGINAL number (new_id -> orig_id map).
    review = tmp_path / "review.csv"
    _write(review, REVIEW_COLS, [
        {"image": "pallets/K/p.jpg", "brick_id": "1", "rank": "1",
         "score": "0.7", "basis": "text", "official_id": "7305",
         "official_section": "I", "official_name": "Travis E Williams",
         "official_keyword": "WILLIAMS", "matched_read": "TRAVIS"}])
    master = tmp_path / "master.csv"
    _write(master, ["orig_id", "new_id", "section", "moved", "status",
                    "buyer", "og_inscription", "og_alt", "new_inscription",
                    "join_score", "og_verified", "flag"],
           [{"orig_id": "3770", "new_id": "7305", "section": "I",
             "moved": "yes", "status": "ok", "buyer": "WILLIAMS",
             "og_inscription": "TRAVIS E WILLIAMS"}])
    out = tmp_path / "review.html"
    make_review_page.main(["--review", str(review), "--output", str(out),
                           "--photo-base-url", "https://x.test/photos",
                           "--master", str(master)])
    page = out.read_text(encoding="utf-8")
    # Strip named by ORIGINAL number, not the candidate's assigned id.
    assert 'src="https://x.test/photos/strips/3770.jpg"' in page
    assert "onerror" in page          # missing strips hide themselves

    # Without --master: no strip references at all.
    make_review_page.main(["--review", str(review), "--output", str(out),
                           "--photo-base-url", "https://x.test/photos"])
    assert "/strips/" not in out.read_text(encoding="utf-8")


def test_stack_photos_sort_last_with_one_click_decision(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    _photo(photos / "brick.jpg")
    _photo(photos / "pallet.jpg")
    _photo(photos / "blank.jpg")
    review = tmp_path / "review.csv"
    _write(review, REVIEW_COLS, [
        {"image": "pallet.jpg", "brick_id": "1", "rank": "1", "score": "0.5",
         "basis": "text", "official_id": "1", "official_section": "A",
         "official_name": "GARBLE", "official_keyword": "G",
         "matched_read": "EDGE EDGE"},
        {"image": "brick.jpg", "brick_id": "1", "rank": "1", "score": "0.7",
         "basis": "text", "official_id": "2", "official_section": "A",
         "official_name": "REAL BRICK", "official_keyword": "R",
         "matched_read": "REAL"},
    ])
    # blank.jpg: unmatched with NO candidates -- only reachable via --matched
    matched = tmp_path / "matched.csv"
    _write(matched, MATCH_COLS, [
        {"image": "blank.jpg", "brick_id": "1", "match_status": "unmatched"},
        {"image": "done.jpg", "brick_id": "1", "match_status": "matched"}])
    types = tmp_path / "types.csv"
    _write(types, ["image", "label"], [
        {"image": "pallet.jpg", "label": "stack"},
        {"image": "blank.jpg", "label": "other"},
        {"image": "brick.jpg", "label": "single"}])
    out = tmp_path / "review.html"
    make_review_page.main(["--review", str(review), "--photos", str(photos),
                           "--matched", str(matched),
                           "--photo-types", str(types),
                           "--output", str(out)])
    page = out.read_text(encoding="utf-8")
    # Stack section exists and comes AFTER the normal items.
    assert "Pallet &amp; stack photos" in page
    assert page.index("brick.jpg") < page.index("Pallet &amp; stack photos")
    assert page.index("Pallet &amp; stack photos") < page.index("pallet.jpg")
    # Stack item: one-click decision FIRST, but candidates KEPT -- the
    # classifier sometimes mislabels a centred brick with stacks behind
    # it, and that brick still needs its match options.
    assert 'data-decision="stack"' in page
    assert "GARBLE" in page
    assert page.index('data-decision="stack"') < page.index("GARBLE")
    # No-candidate unmatched photo included; matched photo is not.
    assert "blank.jpg" in page
    assert "done.jpg" not in page


def test_apply_decisions_stack_kind(tmp_path):
    matched = tmp_path / "matched.csv"
    _write(matched, MATCH_COLS, [
        {"image": "pallet.jpg", "brick_id": "1",
         "match_status": "unmatched"}])
    decisions = tmp_path / "decisions.csv"
    _write(decisions, ["reviewer", "image", "brick_id", "decision",
                       "official_id", "official_section", "official_name",
                       "official_keyword", "note"],
           [{"reviewer": "test", "image": "pallet.jpg", "brick_id": "1",
             "decision": "stack", "note": "whole pallet H3"}])
    out = tmp_path / "final.csv"
    apply_decisions.main(["--matched", str(matched), "--decisions",
                          str(decisions), "--output", str(out)])
    with open(out, newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    assert row["match_status"] == "stack_photo"
    assert row["review_note"] == "whole pallet H3"
    assert row["official_id"] == ""


def test_review_page_warns_on_missing_photo(tmp_path, capsys):
    photos = tmp_path / "photos"
    photos.mkdir()
    review = tmp_path / "review.csv"
    _write(review, REVIEW_COLS, [
        {"image": "gone.jpg", "brick_id": "1", "rank": "1", "score": "0.7",
         "basis": "text", "official_id": "1", "official_section": "A",
         "official_name": "X", "official_keyword": "X",
         "matched_read": "X"}])
    out = tmp_path / "review.html"
    make_review_page.main(["--review", str(review), "--photos", str(photos),
                           "--output", str(out)])
    assert "not found" in capsys.readouterr().out
    assert out.is_file()                          # page still generated


# --- apply_decisions -------------------------------------------------------------

def _matched_rows():
    return [
        {"image": "a.jpg", "brick_id": "1", "match_status": "matched",
         "match_basis": "text", "score": "1.00", "official_id": "10",
         "official_section": "F", "official_name": "ALPHA BRAVO",
         "official_keyword": "AB", "matched_read": "ALPHA BRAVO"},
        {"image": "b.jpg", "brick_id": "1", "match_status": "unmatched",
         "match_basis": "tokens", "score": "0.71", "official_id": "11",
         "official_section": "B", "official_name": "NEAR MISS",
         "official_keyword": "NM", "matched_read": "NEAR MSS"},
        {"image": "c.jpg", "brick_id": "1", "match_status": "unmatched",
         "match_basis": "text", "score": "0.40", "official_id": "",
         "official_section": "", "official_name": "", "official_keyword": "",
         "matched_read": "???"},
    ]


def _decisions_rows():
    return [
        {"reviewer": "Pat", "image": "b.jpg", "brick_id": "1",
         "decision": "match", "official_id": "12", "official_section": "H",
         "official_name": "THE RIGHT BRICK", "official_keyword": "RIGHT",
         "note": "clear on the photo"},
        {"reviewer": "Pat", "image": "c.jpg", "brick_id": "1",
         "decision": "illegible", "official_id": "", "official_section": "",
         "official_name": "", "official_keyword": "", "note": ""},
        {"reviewer": "Pat", "image": "ghost.jpg", "brick_id": "1",
         "decision": "none", "official_id": "", "official_section": "",
         "official_name": "", "official_keyword": "", "note": ""},
    ]


def test_decisions_are_applied(tmp_path, capsys):
    matched = tmp_path / "matched.csv"
    decisions = tmp_path / "decisions.csv"
    out = tmp_path / "final.csv"
    _write(matched, MATCH_COLS, _matched_rows())
    _write(decisions,
           ["reviewer", "image", "brick_id", "decision", "official_id",
            "official_section", "official_name", "official_keyword", "note"],
           _decisions_rows())
    apply_decisions.main(["--matched", str(matched),
                          "--decisions", str(decisions),
                          "--output", str(out)])
    with open(out, newline="", encoding="utf-8") as f:
        rows = {r["image"]: r for r in csv.DictReader(f)}

    # Machine match untouched.
    assert rows["a.jpg"]["match_status"] == "matched"
    assert rows["a.jpg"]["match_basis"] == "text"
    assert rows["a.jpg"]["reviewer"] == ""
    # Human match: status, basis, and the CHOSEN candidate (not the near-miss).
    b = rows["b.jpg"]
    assert (b["match_status"], b["match_basis"]) == ("matched", "human")
    assert b["official_id"] == "12" and b["official_name"] == "THE RIGHT BRICK"
    assert b["reviewer"] == "Pat" and b["review_note"] == "clear on the photo"
    # Illegible: dead end, machine candidates cleared.
    c = rows["c.jpg"]
    assert c["match_status"] == "illegible"
    assert c["official_id"] == ""
    # Orphan decision reported.
    assert "matched no catalogue row" in capsys.readouterr().out


# --- make_report -----------------------------------------------------------------

MASTER_COLS = ["orig_id", "new_id", "section", "moved", "status", "buyer",
               "og_inscription", "og_alt", "new_inscription", "join_score",
               "og_verified", "flag"]


def test_report_workbook(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    master = tmp_path / "master.csv"
    _write(master, MASTER_COLS, [
        {"orig_id": "100", "new_id": "100", "section": "F", "moved": "no",
         "status": "ok", "buyer": "DOE", "og_inscription": "JOHN DOE FAMILY",
         "og_alt": "", "new_inscription": "", "join_score": "",
         "og_verified": "agreed", "flag": ""},
        {"orig_id": "5000", "new_id": "77", "section": "A", "moved": "yes",
         "status": "ok", "buyer": "SUN", "og_inscription": "SUNSHINE BAKERY",
         "og_alt": "", "new_inscription": "Sunshine Bakery",
         "join_score": "0.95", "og_verified": "agreed", "flag": ""},
        {"orig_id": "5002", "new_id": "", "section": "", "moved": "yes",
         "status": "unjoined", "buyer": "??", "og_inscription": "ZZQQ",
         "og_alt": "", "new_inscription": "", "join_score": "",
         "og_verified": "strip", "flag": "number?:4999"},
    ])
    matched = tmp_path / "final.csv"
    _write(matched, MATCH_COLS + ["reviewer", "review_note"], [
        {"image": "p1.jpg", "brick_id": "1", "match_status": "matched",
         "match_basis": "human", "score": "1.00", "official_id": "100",
         "official_section": "F", "official_name": "JOHN DOE FAMILY",
         "official_keyword": "DOE", "matched_read": "JOHN DOE",
         "reviewer": "Pat", "review_note": ""},
        {"image": "p2.jpg", "brick_id": "1", "match_status": "unmatched",
         "match_basis": "text", "score": "0.5", "official_id": "",
         "official_section": "", "official_name": "", "official_keyword": "",
         "matched_read": "???", "reviewer": "", "review_note": ""},
        {"image": "p3.jpg", "brick_id": "1", "match_status": "no_match",
         "pallet": "F2", "matched_read": "THE SERFF FAMILY",
         "reviewer": "Pat", "review_note": "not in list"},
        {"image": "p4.jpg", "brick_id": "1", "match_status": "stack_photo",
         "reviewer": "Pat", "review_note": ""},
    ])
    out = tmp_path / "report.xlsx"
    make_report.main(["--master", str(master), "--matched", str(matched),
                      "--output", str(out)])

    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames[:2] == ["Summary", "All bricks"]
    assert "Section F" in wb.sheetnames and "Unassigned" in wb.sheetnames

    all_rows = {r[1]: r for r in
                wb["All bricks"].iter_rows(min_row=2, values_only=True)}
    assert all_rows["100"][7] == "Present"        # photographed brick
    assert all_rows["100"][9] == "Pat"            # reviewer carried through
    assert all_rows["5000"][7] in ("", None)      # not photographed
    assert all_rows["5002"][6] == "number?:4999"  # flag visible to staff

    summary = list(wb["Summary"].iter_rows(values_only=True))
    totals = next(r for r in summary if r[0] == "TOTAL")
    assert totals[1] == 3 and totals[2] == 1      # 3 bricks, 1 present
    # The still-in-review photo is called out.
    assert any(r[0] and "in review" in str(r[0]) for r in summary)
    # Stack shots and unofficial bricks are called out too.
    assert any(r[0] and "stack" in str(r[0]) for r in summary)
    assert any(r[0] and "NOT in the official lists" in str(r[0])
               for r in summary)

    # Unofficial bricks get their own sheet: photo text, pallet, note.
    un = list(wb["Unofficial bricks"].iter_rows(min_row=2, values_only=True))
    assert un == [("p3.jpg", "F2", "THE SERFF FAMILY", "Pat", "not in list")]


def test_report_note_and_clean_inscription(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    master = tmp_path / "master.csv"
    _write(master, MASTER_COLS, [
        {"orig_id": "1375", "new_id": "", "section": "F", "moved": "no",
         "status": "ok", "buyer": "CONDY",
         "og_inscription": "THE WOOFIER FAMIIX",
         "og_alt": "THE WOOFTER FAMILY", "new_inscription": "",
         "join_score": "", "og_verified": "strip", "flag": ""}])
    matched = tmp_path / "final.csv"
    _write(matched, MATCH_COLS + ["reviewer", "review_note"], [
        {"image": "p1.jpg", "brick_id": "1", "match_status": "matched",
         "score": "0.93", "official_id": "1375", "official_section": "F",
         "reviewer": "Pat", "review_note": "small chip lower left"}])
    out = tmp_path / "report.xlsx"
    make_report.main(["--master", str(master), "--matched", str(matched),
                      "--output", str(out)])
    wb = openpyxl.load_workbook(out)
    row = next(wb["All bricks"].iter_rows(min_row=2, values_only=True))
    assert row[4] == "THE WOOFTER FAMILY"          # clean re-read displayed
    assert row[10] == "small chip lower left"      # note reaches the book
