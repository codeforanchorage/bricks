"""Regression gate: the 26 labeled warehouse photos against the master list.

The fixtures freeze the OCR reads from the validated 2026-07-28 capture test
(5 test_images photos read by 3 methods + 21 test_bricks photos read by
Gemini Flash) and the human-verified outcome for each: 23 matched to a known
brick, 3 correctly left for review, 0 false positives.

The test replays those reads through match.py against the real committed
reference/master_list.csv -- no API calls, pure CSV -- so any change to the
matching layers (scan_fold, phonetic_fold, token containment, margins) or a
master-list rebuild that would break identification fails here first.
"""
import csv
from pathlib import Path

import pytest

import match

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MASTER = ROOT / "reference" / "master_list.csv"


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    if not MASTER.is_file():
        pytest.skip("reference/master_list.csv not built")
    output = tmp_path_factory.mktemp("regression") / "matched.csv"
    match.main(["--catalog", str(FIXTURES / "labeled_reads.csv"),
                "--reference", str(MASTER),
                "--output", str(output), "--scan-ocr"])
    with open(output, newline="", encoding="utf-8") as f:
        return {r["image"]: r for r in csv.DictReader(f)}


def _expected():
    with open(FIXTURES / "expected_matches.csv", newline="",
              encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("exp", _expected(), ids=lambda e: e["image"])
def test_labeled_photo(results, exp):
    got = results[exp["image"]]
    assert got["match_status"] == exp["match_status"], (
        f"{exp['image']}: expected {exp['match_status']}, got "
        f"{got['match_status']} (#{got['official_id']} "
        f"{got['official_name']!r} @ {got['score']})")
    if exp["match_status"] == "matched":
        assert got["official_id"] == exp["official_id"], (
            f"{exp['image']}: matched the WRONG brick -- expected "
            f"#{exp['official_id']}, got #{got['official_id']} "
            f"{got['official_name']!r}")


def test_no_false_positives(results):
    """Every matched photo must match its verified brick -- 0 FP overall."""
    expected = {e["image"]: e for e in _expected()}
    wrong = [img for img, got in results.items()
             if got["match_status"] == "matched"
             and (expected[img]["match_status"] != "matched"
                  or got["official_id"] != expected[img]["official_id"])]
    assert wrong == []


def test_match_rate_floor(results):
    """At least the validated 23/26 must auto-identify."""
    n = sum(1 for r in results.values() if r["match_status"] == "matched")
    assert n >= 23
