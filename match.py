#!/usr/bin/env python3
"""Match the per-brick OCR catalogue against the official Muni brick list.

For each detected brick, fuzzy-matches its candidate reads (one per OCR
method) against the official inscriptions in the parsed reference list, so the
methods' disagreement collapses onto the one real brick -- the closest read
to a known official name wins.

Output: a matched CSV -- each detected brick paired with its best official
match (assigned id, section, official name) and a match score. With --section
it also writes that section's official bricks that no photo matched, i.e.
candidates for lost/broken (a true list only once a section is photographed
in full).

Matching is brute-force (every read vs every official name); fine for a test
batch, would need key-word blocking to scale to the full ~13,000 bricks.

Usage:
    python match.py --catalog output/bricks_test.csv \
        --reference reference/brick_list.csv --output output/matched.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from consensus import _match_key, _normalise, _similar, scan_fold

# Catalogue columns that are not OCR-method reads.
_NON_READ = {"image", "brick_id", "x", "y", "w", "h", "status"}
DEFAULT_MIN_SCORE = 0.80   # genuine matches score >=0.84; false positives <=0.71

MATCH_COLUMNS = ["image", "brick_id", "match_status", "score",
                 "official_id", "official_section", "official_name",
                 "official_keyword", "matched_read"]


def _key_for(text: str, scan_ocr: bool) -> str:
    """Comparison key for one inscription."""
    key = _match_key(_normalise(text))
    return scan_fold(key) if scan_ocr else key


def _load_reference(path: Path, section: str, scan_ocr: bool = False) -> list[dict]:
    """Load the official brick list, optionally one section, with normalised text.

    Accepts either parsed-PDF list (parse_brick_list.py / parse_tsp_list.py)
    or the merged master list (merge_lists.py), which is adapted to the same
    shape -- preferring the cleaner new-list inscription when the row has one.
    """
    refs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "og_inscription" in row:   # master_list.csv
                row = {"section": row["section"],
                       "assigned_id": row["new_id"] or row["orig_id"],
                       "full_name": row["new_inscription"] or row["og_inscription"],
                       "key_word": row["buyer"]}
            if section and row["section"].upper() != section.upper():
                continue
            # Match on the distinctive words only -- boilerplate like
            # "IN MEMORY OF" otherwise inflates similarity between bricks.
            row["_key"] = _key_for(row["full_name"], scan_ocr)
            if row["_key"]:
                refs.append(row)
    return refs


def _best_match(reads: list[str], reference: list[dict], scan_ocr: bool = False):
    """Best (score, ref_row, winning_read) of any read vs any official name."""
    best_score, best_ref, best_read = 0.0, None, ""
    for read in reads:
        key = _key_for(read, scan_ocr)
        if not key:
            continue
        for ref in reference:
            score = _similar(key, ref["_key"])
            if score > best_score:
                best_score, best_ref, best_read = score, ref, read
    return best_score, best_ref, best_read


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", required=True, type=Path,
                        help="Per-brick catalogue from brick_pipeline.py")
    parser.add_argument("--reference", required=True, type=Path,
                        help="Official brick list from parse_brick_list.py")
    parser.add_argument("--output", required=True, type=Path,
                        help="Matched CSV to write")
    parser.add_argument("--section", default="",
                        help="Restrict matching to one section (A-K) and "
                             "report that section's unmatched official bricks")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"Similarity needed to accept a match "
                             f"(default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--scan-ocr", action="store_true",
                        help="Fold OCR-confusable letters (I/L/T, E/F, O/D) on "
                             "both sides. Use with the scanned by-name list "
                             "(tsp_brick_list.csv), whose own OCR is noisy.")
    args = parser.parse_args(argv)

    for path in (args.catalog, args.reference):
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")

    reference = _load_reference(args.reference, args.section, args.scan_ocr)
    if not reference:
        raise SystemExit(f"No reference bricks loaded (section {args.section!r}?)")

    with open(args.catalog, newline="", encoding="utf-8") as f:
        catalog = list(csv.DictReader(f))
    read_cols = [c for c in (catalog[0] if catalog else {})
                 if c not in _NON_READ]

    scope = f"section {args.section.upper()}" if args.section else "all sections"
    print(f"Matching {len(catalog)} detected brick(s) against "
          f"{len(reference)} official brick(s) ({scope}) ...")

    rows = []
    matched_ids = set()
    for brick in catalog:
        reads = [brick[c] for c in read_cols
                 if brick.get(c) and not brick[c].startswith("ERROR:")]
        score, ref, read = _best_match(reads, reference, args.scan_ocr)
        matched = ref is not None and score >= args.min_score
        if matched:
            matched_ids.add(ref["assigned_id"])
        rows.append({
            "image": brick.get("image", ""),
            "brick_id": brick.get("brick_id", ""),
            "match_status": "matched" if matched else "unmatched",
            "score": f"{score:.2f}",
            "official_id": ref["assigned_id"] if ref else "",
            "official_section": ref["section"] if ref else "",
            "official_name": ref["full_name"] if ref else "",
            "official_keyword": ref["key_word"] if ref else "",
            "matched_read": read,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_matched = sum(1 for r in rows if r["match_status"] == "matched")
    print(f"\n  matched   : {n_matched}/{len(rows)} detected bricks")
    print(f"  unmatched : {len(rows) - n_matched}")
    print(f"\nWrote {args.output}")

    if args.section:
        missing = [r for r in reference if r["assigned_id"] not in matched_ids]
        miss_path = args.output.with_name(f"missing_{args.section.upper()}.csv")
        with open(miss_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["section", "assigned_id", "full_name", "key_word"])
            for r in missing:
                writer.writerow([r["section"], r["assigned_id"],
                                 r["full_name"], r["key_word"]])
        print(f"\nSection {args.section.upper()}: {len(missing)} official "
              f"brick(s) not matched by any photo -> {miss_path}")
        print("  (a true lost/broken list only once the section is "
              "photographed in full)")


if __name__ == "__main__":
    main()
