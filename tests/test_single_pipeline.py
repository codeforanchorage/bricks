"""Tests for the production-runner behaviours of single_pipeline.py.

The resume logic is pure CSV bookkeeping and the section/pallet folder
parsing is pure path arithmetic, so both are tested directly with no API
calls or images.
"""
import csv
from pathlib import Path

from single_pipeline import _keep_previous_rows, _photo_meta

COLUMNS = ["image", "brick_id", "section", "pallet", "x", "y", "w", "h",
           "gemini-flash", "status"]


def _write(path, rows, columns=COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def _row(image, read, status="single", section="", pallet=""):
    return {"image": image, "brick_id": 1, "section": section,
            "pallet": pallet, "x": "", "y": "", "w": "", "h": "",
            "gemini-flash": read, "status": status}


def test_resume_keeps_good_rows_and_redoes_errors(tmp_path):
    p = tmp_path / "out.csv"
    _write(p, [
        _row("a.jpg", "SMITH FAMILY"),                     # good -> keep
        _row("b.jpg", "", status="none"),                  # empty = real answer -> keep
        _row("c.jpg", "ERROR: 429 rate limited", "none"),  # transient -> redo
    ])
    keep, seen = _keep_previous_rows(p, ["gemini-flash"], COLUMNS)
    assert seen == {"a.jpg", "b.jpg"}
    assert [r["image"] for r in keep] == ["a.jpg", "b.jpg"]
    assert keep[0]["gemini-flash"] == "SMITH FAMILY"


def test_resume_redoes_images_missing_a_requested_method(tmp_path):
    # The old run used flash only; resuming with sonnet+flash must redo
    # every image (the sonnet column does not exist yet).
    p = tmp_path / "out.csv"
    _write(p, [_row("a.jpg", "SMITH FAMILY")])
    keep, seen = _keep_previous_rows(p, ["sonnet", "gemini-flash"],
                                     COLUMNS + ["claude-sonnet"])
    assert seen == set()
    assert keep == []


def test_resume_deduplicates_repeated_images(tmp_path):
    # An interrupted-then-resumed old run can hold the same image twice;
    # only the first good row survives.
    p = tmp_path / "out.csv"
    _write(p, [_row("a.jpg", "FIRST READ"), _row("a.jpg", "SECOND READ")])
    keep, seen = _keep_previous_rows(p, ["gemini-flash"], COLUMNS)
    assert len(keep) == 1
    assert keep[0]["gemini-flash"] == "FIRST READ"


def test_resume_tolerates_truncated_last_line(tmp_path):
    # A crash mid-write leaves a torn final line; it must not crash resume
    # and must not be kept as a good row for a full method set.
    p = tmp_path / "out.csv"
    _write(p, [_row("a.jpg", "SMITH FAMILY")])
    with open(p, "a", newline="", encoding="utf-8") as f:
        f.write("b.jpg,1,,,")   # torn row: no read column reached the disk
    keep, seen = _keep_previous_rows(p, ["gemini-flash"], COLUMNS)
    assert seen == {"a.jpg"}


def test_resume_distinguishes_same_filename_on_different_pallets(tmp_path):
    # Camera filenames repeat across pallets; the catalogue keys on the
    # relative path, so IMG_0001.jpg on pallet-1 done != pallet-2 done.
    p = tmp_path / "out.csv"
    _write(p, [_row("H/pallet-1/IMG_0001.jpg", "SMITH FAMILY",
                    section="H", pallet="pallet-1")])
    keep, seen = _keep_previous_rows(p, ["gemini-flash"], COLUMNS)
    assert seen == {"H/pallet-1/IMG_0001.jpg"}
    assert "H/pallet-2/IMG_0001.jpg" not in seen
    assert keep[0]["section"] == "H"
    assert keep[0]["pallet"] == "pallet-1"


def test_resume_accepts_pre_convention_catalogue(tmp_path):
    # A catalogue written before the section/pallet columns existed resumes
    # cleanly: its rows are kept with empty section/pallet.
    p = tmp_path / "out.csv"
    old_columns = ["image", "brick_id", "x", "y", "w", "h",
                   "gemini-flash", "status"]
    _write(p, [{"image": "a.jpg", "brick_id": 1, "x": "", "y": "", "w": "",
                "h": "", "gemini-flash": "SMITH FAMILY",
                "status": "single"}], columns=old_columns)
    keep, seen = _keep_previous_rows(p, ["gemini-flash"], COLUMNS)
    assert seen == {"a.jpg"}
    assert keep[0]["section"] == ""
    assert keep[0]["pallet"] == ""


def test_photo_meta_section_pallet_convention():
    root = Path("photos")
    assert (_photo_meta(Path("photos/H/pallet-12/IMG_0421.jpg"), root)
            == ("H/pallet-12/IMG_0421.jpg", "H", "pallet-12"))
    # Lowercase section folder still tags (stored uppercase).
    assert (_photo_meta(Path("photos/k/pallet-3/x.heic"), root)
            == ("k/pallet-3/x.heic", "K", "pallet-3"))
    # Photo directly in a section folder: section, no pallet.
    assert (_photo_meta(Path("photos/E/IMG_9.jpg"), root)
            == ("E/IMG_9.jpg", "E", ""))
    # Flat layout (pre-convention): no tags, name unchanged.
    assert (_photo_meta(Path("photos/IMG_1.jpg"), root)
            == ("IMG_1.jpg", "", ""))
    # A non-section top folder is NOT the convention: no section, and no
    # pallet guessed from its subfolder either.
    assert (_photo_meta(Path("photos/misc/pallet-9/IMG_2.jpg"), root)
            == ("misc/pallet-9/IMG_2.jpg", "", ""))


def test_photo_meta_pallet_only_convention():
    # Warehouse pallet labels are arbitrary (K, H1..H6) and do NOT encode
    # the park section: pallets/<pallet>/ records the pallet alone and the
    # section is left for the match to reveal.
    root = Path("photos")
    assert (_photo_meta(Path("photos/pallets/K/IMG_1.jpg"), root)
            == ("pallets/K/IMG_1.jpg", "", "K"))
    assert (_photo_meta(Path("photos/Pallets/H3/IMG_2.jpg"), root)
            == ("Pallets/H3/IMG_2.jpg", "", "H3"))
    # A photo directly in pallets/ names no pallet: no tag to record.
    assert (_photo_meta(Path("photos/pallets/IMG_3.jpg"), root)
            == ("pallets/IMG_3.jpg", "", ""))
    # A section folder named like a pallet label is still the section form.
    assert (_photo_meta(Path("photos/H/K/IMG_4.jpg"), root)
            == ("H/K/IMG_4.jpg", "H", "K"))
