"""Tests for the review handoff: page generation, decision application,
and the Parks & Rec Excel report.
"""
import csv

import pytest

import apply_decisions
import make_report
import make_review_page

REVIEW_COLS = ["image", "brick_id", "rank", "score", "basis", "official_id",
               "official_section", "official_name", "official_keyword",
               "matched_read"]
MATCH_COLS = ["image", "brick_id", "match_status", "match_basis", "score",
              "official_id", "official_section", "official_name",
              "official_keyword", "matched_read"]


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
