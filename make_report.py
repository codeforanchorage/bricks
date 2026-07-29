#!/usr/bin/env python3
"""Build the Parks & Rec Excel workbook from the master list + photo results.

This is the handoff document: one row per official brick, annotated with
whether a warehouse photo has confirmed it. Staff at the pickup counter
filter/search it to answer "does this person's brick exist, which section,
and have we actually seen it?".

Sheets:
  Summary     per-section totals: bricks, photographed, review, remaining
  All bricks  every master row with its photo status
  A..K        the same rows split per section (what a pallet crew works from)

Photo status values:
  Present     a photo matched this brick (machine >= threshold, or human)
  No match    reviewer confirmed the photo's brick is not in the list (the
              status lands on the PHOTO, so it never marks a brick Present)
  Pending     photographed but the identification is still in review
  (blank)     no photo of this brick yet -- NOT evidence it is missing until
              its section is photographed in full

Accepts one or more matched catalogues (match.py / apply_decisions.py
output); rows whose match_status is 'matched' claim their official brick.

Usage:
    python make_report.py --master reference/master_list.csv \
        --matched output/matched_final.csv --output output/brick_report.xlsx
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HEADERS = ["Section", "Orig #", "New #", "Buyer", "Inscription",
           "List status", "Flag", "Photo status", "Photo", "Reviewer"]
WIDTHS = [9, 9, 9, 24, 46, 11, 16, 13, 22, 14]


def _load_matches(paths: list[Path]) -> tuple[dict, dict]:
    """Photo results keyed by (section, official id).

    Returns (present, pending): present maps the key to the best matched
    photo row; pending counts photos still unresolved (unmatched -- their
    review decision hasn't come back yet). 'no_match'/'illegible' rows are
    photo dead ends, not brick evidence, so they touch neither dict.
    """
    present: dict[tuple[str, str], dict] = {}
    pending_count = 0
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                status = row.get("match_status", "")
                if status == "matched":
                    key = (row.get("official_section", "").upper(),
                           row.get("official_id", ""))
                    best = present.get(key)
                    if best is None or float(row.get("score") or 0) > \
                            float(best.get("score") or 0):
                        present[key] = row
                elif status == "unmatched":
                    pending_count += 1
    return present, {"unresolved_photos": pending_count}


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", required=True, type=Path,
                        help="reference/master_list.csv from merge_lists.py")
    parser.add_argument("--matched", type=Path, nargs="*", default=[],
                        help="Matched catalogue(s) from match.py / "
                             "apply_decisions.py (omit for a list-only book)")
    parser.add_argument("--output", required=True, type=Path,
                        help=".xlsx workbook to write")
    args = parser.parse_args(argv)

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise SystemExit("openpyxl is not installed -- run: "
                         "pip install -r requirements.txt")

    with open(args.master, newline="", encoding="utf-8") as f:
        master = list(csv.DictReader(f))
    present, pending = _load_matches(args.matched)

    def brick_row(row: dict) -> list[str]:
        section = row["section"].upper()
        # A master row's photo key: the id match.py reports for it.
        key = (section, row["new_id"] or row["orig_id"])
        photo = present.get(key)
        return [
            section or "?",
            row["orig_id"],
            row["new_id"],
            row["buyer"],
            row["new_inscription"] or row["og_inscription"],
            row["status"],
            row["flag"],
            "Present" if photo else "",
            photo.get("image", "") if photo else "",
            photo.get("reviewer", "") if photo else "",
        ]

    wb = Workbook()
    bold = Font(bold=True)
    green = PatternFill("solid", start_color="D8EFD0")

    def start_sheet(ws):
        ws.append(HEADERS)
        for cell in ws[1]:
            cell.font = bold
        for i, width in enumerate(WIDTHS, 1):
            ws.column_dimensions[get_column_letter(i)].width = width
        ws.freeze_panes = "A2"

    def finish_sheet(ws):
        ws.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}{ws.max_row}"

    # --- All bricks ------------------------------------------------------
    all_ws = wb.active
    all_ws.title = "All bricks"
    start_sheet(all_ws)
    by_section: dict[str, list[dict]] = defaultdict(list)
    section_stats: dict[str, dict] = defaultdict(lambda: {"total": 0,
                                                          "present": 0})
    for row in master:
        section = row["section"].upper() or "?"
        by_section[section].append(row)
        values = brick_row(row)
        all_ws.append(values)
        section_stats[section]["total"] += 1
        if values[7] == "Present":
            section_stats[section]["present"] += 1
            for cell in all_ws[all_ws.max_row]:
                cell.fill = green
    finish_sheet(all_ws)

    # --- Per-section sheets ----------------------------------------------
    for section in sorted(by_section):
        ws = wb.create_sheet(f"Section {section}" if section != "?"
                             else "Unassigned")
        start_sheet(ws)
        rows = sorted(by_section[section],
                      key=lambda r: int((r["new_id"] or r["orig_id"])
                                        .replace(",", ""))
                      if (r["new_id"] or r["orig_id"]).replace(",", "").isdigit()
                      else 0)
        for row in rows:
            values = brick_row(row)
            ws.append(values)
            if values[7] == "Present":
                for cell in ws[ws.max_row]:
                    cell.fill = green
        finish_sheet(ws)

    # --- Summary (inserted first) ------------------------------------------
    summary = wb.create_sheet("Summary", 0)
    summary.append(["Section", "Bricks", "Photographed (Present)",
                    "Not yet photographed"])
    for cell in summary[1]:
        cell.font = bold
    for i, width in enumerate((10, 10, 24, 22), 1):
        summary.column_dimensions[get_column_letter(i)].width = width
    total = present_total = 0
    for section in sorted(section_stats):
        stats = section_stats[section]
        summary.append([section, stats["total"], stats["present"],
                        stats["total"] - stats["present"]])
        total += stats["total"]
        present_total += stats["present"]
    summary.append(["TOTAL", total, present_total, total - present_total])
    for cell in summary[summary.max_row]:
        cell.font = bold
    if pending["unresolved_photos"]:
        summary.append([])
        summary.append([f"{pending['unresolved_photos']} photo(s) are still "
                        f"in review and not counted as Present."])
    summary.append([])
    summary.append(["A blank Photo status means the brick has not been "
                    "photographed yet -- it is NOT evidence the brick is "
                    "missing until its section is photographed in full."])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.output)
    print(f"Wrote {args.output}")
    print(f"  {total} brick(s); {present_total} confirmed present by photo; "
          f"{pending['unresolved_photos']} photo(s) still in review")


if __name__ == "__main__":
    main()
