#!/usr/bin/env python3
"""Parse the Municipality's source Excel workbook of new brick numbers.

"TSP Bricks All.xls" (authored 2008, the renovation era) is the spreadsheet
the by-area PDF (ABCDHIJK.pdf) was printed from -- one sheet per brick area.
E, F and G are empty placeholders (those areas were not moved or renumbered);
the other eight sheets carry the relocated bricks: assigned (new) number, the
brick's grid position within the area, the inscription, and the key word.

Being the digital source, it has none of the PDF text-extraction losses --
the PDF parse recovered 7,874 rows, the workbook holds ~8,280 -- so its CSV
supersedes parse_brick_list.py's output as the canonical new-numbers list.
Output schema matches exactly, so every downstream consumer (match.py,
merge_lists.py) takes it unchanged.

Column names vary a little between sheets ("Brick ##"/"Assigned #",
"Inscription"/"Full Name", "Column"/"Brick #"); they are mapped by header.

Usage:
    python parse_xls_list.py --xls "TSP Bricks All.xls" \
        --output reference/brick_list_xls.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_COLUMNS = ["section", "assigned_id", "pos1", "pos2", "full_name", "key_word"]

# Header spellings seen across the sheets, lowercased.
_FIELD_ALIASES = {
    "assigned_id": {"assigned #", "brick ##", "brick #"},
    "pos1": {"column", "row"},          # first position column encountered
    "full_name": {"full name", "inscription"},
    "key_word": {"key word", "keyword"},
}


def _map_headers(header: list[str]) -> dict[str, int]:
    """Column name -> index for one sheet, tolerant of the naming variants."""
    mapping: dict[str, int] = {}
    positions: list[int] = []
    for index, raw in enumerate(header):
        name = str(raw).strip().lower()
        if not name:
            continue
        if name in _FIELD_ALIASES["full_name"]:
            mapping["full_name"] = index
        elif name in _FIELD_ALIASES["key_word"]:
            mapping["key_word"] = index
        elif name in _FIELD_ALIASES["pos1"] and "assigned_id" in mapping:
            positions.append(index)
        elif name in _FIELD_ALIASES["assigned_id"] and "assigned_id" not in mapping:
            # "Brick #" doubles as a position column on sheets that also have
            # "Assigned #" -- the first id-like header wins as the id.
            mapping["assigned_id"] = index
    mapping["pos1"] = positions[0] if positions else -1
    mapping["pos2"] = positions[1] if len(positions) > 1 else -1
    return mapping


def _cell(sheet, row: int, col: int) -> str:
    if col < 0 or col >= sheet.ncols:
        return ""
    value = sheet.cell_value(row, col)
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else str(value)
    return str(value).strip()


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xls", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        import xlrd
    except ImportError:
        raise SystemExit("xlrd is not installed -- run: pip install xlrd")

    workbook = xlrd.open_workbook(str(args.xls))
    rows, skipped = [], 0
    for sheet in workbook.sheets():
        section = sheet.name.strip().upper()
        if sheet.nrows < 2:
            print(f"  {section}: empty (area not moved in 2008)")
            continue
        mapping = _map_headers([sheet.cell_value(0, c)
                                for c in range(sheet.ncols)])
        if "assigned_id" not in mapping or "full_name" not in mapping:
            raise SystemExit(f"Sheet {section}: unrecognised header "
                             f"{[sheet.cell_value(0, c) for c in range(sheet.ncols)]}")
        count = 0
        for r in range(1, sheet.nrows):
            assigned = _cell(sheet, r, mapping["assigned_id"])
            name = _cell(sheet, r, mapping["full_name"])
            if not assigned.isdigit() or not name:
                skipped += 1
                continue
            rows.append({
                "section": section,
                "assigned_id": assigned,
                "pos1": _cell(sheet, r, mapping["pos1"]),
                "pos2": _cell(sheet, r, mapping["pos2"]),
                "full_name": " ".join(name.split()),
                "key_word": _cell(sheet, r, mapping["key_word"]),
            })
            count += 1
        print(f"  {section}: {count} bricks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} brick row(s) to {args.output} "
          f"({skipped} blank/invalid rows skipped)")


if __name__ == "__main__":
    main()
