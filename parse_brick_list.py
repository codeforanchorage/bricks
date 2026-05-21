#!/usr/bin/env python3
"""Parse the Municipality's Town Square brick-list PDF into a CSV.

The PDF (reference/ABCDHIJK.pdf) lists every relocated brick in areas A-D and
H-K as: an Assigned #, one or two grid numbers, the full inscription, and a
key word. Sections are delimited by "Brick Area X" lines. Source:
https://www.muni.org/Departments/parks/Pages/TownSquareBrickLocations.aspx

Output CSV columns: section, assigned_id, pos1, pos2, full_name, key_word --
the official reference list to match OCR results against. The `full_name` is
the engraved inscription; `key_word` is the Municipality's lookup keyword.

Usage:
    python parse_brick_list.py --pdf reference/ABCDHIJK.pdf \
        --output reference/brick_list.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

_SECTION = re.compile(r"^Brick Area ([A-Z])\b", re.IGNORECASE)
_HEADER = re.compile(r"^(Assigned|Brick)\s*#", re.IGNORECASE)
_COLUMNS = ["section", "assigned_id", "pos1", "pos2", "full_name", "key_word"]


def _parse_row(line: str) -> dict | None:
    """Parse one brick data line; return None if it is not a data row.

    A row is: <assigned#> <grid number> [<grid number>] <inscription...>
    <key word>. The assigned number may contain commas.
    """
    tokens = line.split()
    if len(tokens) < 4 or not re.fullmatch(r"[\d,]+", tokens[0]):
        return None

    # One or two integer grid numbers follow the assigned number.
    nums, idx = [], 1
    while idx < len(tokens) and len(nums) < 2 and tokens[idx].isdigit():
        nums.append(int(tokens[idx]))
        idx += 1
    if not nums or idx >= len(tokens):
        return None  # need >=1 grid number and a trailing inscription/key word

    trailing = tokens[idx:]
    if len(trailing) == 1:        # one-word entry: name and key word coincide
        full_name = key_word = trailing[0]
    else:
        full_name, key_word = " ".join(trailing[:-1]), trailing[-1]
    return {
        "assigned_id": int(tokens[0].replace(",", "")),
        "pos1": nums[0],
        "pos2": nums[1] if len(nums) > 1 else "",
        # ‐ is a unicode hyphen used throughout the PDF -- normalise it.
        "full_name": full_name.replace("‐", "-"),
        "key_word": key_word,
    }


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, type=Path,
                        help="The Municipality brick-list PDF")
    parser.add_argument("--output", required=True, type=Path,
                        help="CSV file to write")
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise SystemExit("pypdfium2 is not installed -- run: "
                         "pip install -r requirements.txt") from exc

    pdf = pdfium.PdfDocument(str(args.pdf))
    rows: list[dict] = []
    section = ""
    skipped: list[str] = []
    for i in range(len(pdf)):
        page_no = i + 1
        text = pdf[i].get_textpage().get_text_range()
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            sec = _SECTION.match(line)
            if sec:
                section = sec.group(1).upper()
                # A section always starts on a fresh page, but the "Brick Area
                # X" line can extract after that page's rows -- so reassign any
                # rows already collected from this page to the new section.
                for r in rows:
                    if r["page"] == page_no:
                        r["section"] = section
                continue
            row = _parse_row(line)
            if row:
                row["section"] = section
                row["page"] = page_no
                rows.append(row)
            elif not _HEADER.match(line):
                skipped.append(f"p{page_no}: {line}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    by_section = Counter(r["section"] for r in rows)
    print(f"Parsed {len(rows)} bricks from {len(pdf)} pages")
    for sec in sorted(by_section):
        print(f"  Area {sec or '(none)'}: {by_section[sec]}")
    if skipped:
        print(f"\n{len(skipped)} line(s) not parsed:")
        for line in skipped[:15]:
            print(f"  {line}")
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
