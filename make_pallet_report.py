#!/usr/bin/env python3
"""Build the warehouse Excel workbook: one tab per PALLET, not per section.

make_report.py answers "does this official brick exist?" and is organised
by the park section printed in the official list. This workbook answers
the warehouse question -- "what is actually stacked on this pallet?" --
so staff pulling a brick can work from the pallet they are standing at.

Sheets:
  Summary      per-pallet totals and which sections the pallet holds
  Pallet <X>   one row per PHOTO taken on that pallet, with the brick it
               matched (section, number, inscription) or its review state

Pallet labels are warehouse labels, not sections: most pallets are ~90%
one section but every one carries strays (and 'Pallet H2' is mostly
section F), so the Section column on each tab is the retrieval truth.

Built for the printed-page use case (a PDF of these tabs posted at each
pallet): every row carries the Buyer and BOTH brick numbers -- people
arrive with either the original number (old certificates) or the
post-2008 one, and buyers are often not who's engraved.

Usage:
    python make_pallet_report.py --matched output/pallets_final.csv \
        --master reference/master_list.csv \
        --output output/brick_report_by_pallet.xlsx
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

HEADERS = ["Section", "Orig #", "New #", "Buyer", "Inscription (official)",
           "Photo", "Status", "Review note"]
WIDTHS = [9, 9, 9, 24, 46, 20, 12, 28]

STATUS = {"matched": "Present", "unmatched": "in review",
          "no_match": "unofficial", "stack_photo": "stack shot",
          "illegible": "illegible"}


def _sort_key(row: dict) -> tuple:
    sec = row.get("official_section", "") or "~"  # blanks sort last
    ident = (row.get("official_id", "") or "").replace(",", "")
    return (sec, int(ident) if ident.isdigit() else 10**9,
            row.get("image", ""))


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matched", required=True, type=Path,
                        help="Applied catalogue (output/pallets_final.csv)")
    parser.add_argument("--master", required=True, type=Path,
                        help="reference/master_list.csv (buyer + both "
                             "brick numbers)")
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

    # Master rows keyed the way match.py reports a brick: (section, the
    # id it matched under -- new_id when the brick was renumbered).
    master: dict[tuple[str, str], dict] = {}
    with open(args.master, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["section"].upper(), row["new_id"] or row["orig_id"])
            master.setdefault(key, row)  # dup_orig twins: first row speaks

    by_pallet: dict[str, list[dict]] = defaultdict(list)
    with open(args.matched, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_pallet[row.get("pallet", "") or "(no pallet)"].append(row)

    wb = Workbook()
    bold = Font(bold=True)
    green = PatternFill("solid", start_color="D8EFD0")

    summary = wb.active
    summary.title = "Summary"
    summary.append(["Pallet", "Photos", "Present", "Distinct bricks",
                    "In review", "Other", "Sections on this pallet"])
    for cell in summary[1]:
        cell.font = bold
    for i, width in enumerate((12, 8, 9, 14, 10, 7, 50), 1):
        summary.column_dimensions[get_column_letter(i)].width = width

    totals = Counter()
    all_keys: set = set()
    for pallet in sorted(by_pallet):
        rows = sorted(by_pallet[pallet], key=_sort_key)
        ws = wb.create_sheet(pallet[:31])
        ws.append(HEADERS)
        for cell in ws[1]:
            cell.font = bold
        for i, width in enumerate(WIDTHS, 1):
            ws.column_dimensions[get_column_letter(i)].width = width
        ws.freeze_panes = "A2"

        sections = Counter()
        keys = set()
        n_present = n_review = n_other = 0
        for row in rows:
            status = STATUS.get(row.get("match_status", ""),
                                row.get("match_status", ""))
            if status == "Present":
                n_present += 1
                sections[row.get("official_section", "") or "?"] += 1
                keys.add((row.get("official_section", ""),
                          row.get("official_id", "")))
            elif status == "in review":
                n_review += 1
            else:
                n_other += 1
            m = master.get((row.get("official_section", "").upper(),
                            row.get("official_id", "")), {})
            ws.append([
                row.get("official_section", ""),
                m.get("orig_id", ""),
                m.get("new_id", ""),
                m.get("buyer", ""),
                row.get("official_name", ""),
                Path(row.get("image", "")).name,
                status,
                row.get("review_note", ""),
            ])
            if status == "Present":
                for cell in ws[ws.max_row]:
                    cell.fill = green
        ws.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}{ws.max_row}"

        section_note = ", ".join(f"{s}: {n}" for s, n in sections.most_common())
        summary.append([pallet, len(rows), n_present, len(keys),
                        n_review, n_other, section_note])
        totals.update({"photos": len(rows), "present": n_present,
                       "review": n_review, "other": n_other})
        all_keys |= keys

    summary.append(["TOTAL", totals["photos"], totals["present"],
                    len(all_keys), totals["review"], totals["other"], ""])
    for cell in summary[summary.max_row]:
        cell.font = bold
    summary.append([])
    summary.append(["Pallet labels are warehouse labels, not sections -- "
                    "the Section column on each tab says where the brick "
                    "belongs in the official list."])
    summary.append(["People may arrive with EITHER brick number: Orig # is "
                    "from pre-2008 certificates, New # is the current "
                    "official numbering."])
    summary.append(["'Distinct bricks' can be below 'Present' when the same "
                    "brick was photographed twice on one pallet; the TOTAL "
                    "row de-duplicates across pallets."])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.output)
    print(f"Wrote {args.output}")
    print(f"  {len(by_pallet)} pallet tab(s); {totals['photos']} photo(s); "
          f"{totals['present']} Present ({len(all_keys)} distinct "
          f"brick(s)); {totals['review']} in review")


if __name__ == "__main__":
    main()
