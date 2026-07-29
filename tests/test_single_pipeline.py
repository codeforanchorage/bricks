"""Tests for the production-runner behaviours of single_pipeline.py.

The resume logic is pure CSV bookkeeping, so it is tested directly with no
API calls or images.
"""
import csv

from single_pipeline import _keep_previous_rows

COLUMNS = ["image", "brick_id", "x", "y", "w", "h", "gemini-flash", "status"]


def _write(path, rows, columns=COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def _row(image, read, status="single"):
    return {"image": image, "brick_id": 1, "x": "", "y": "", "w": "", "h": "",
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
