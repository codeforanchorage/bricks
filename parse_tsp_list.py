#!/usr/bin/env python3
"""Parse the original "TSP Bricks ALL - OG List by Name" PDF into a CSV.

This is the Municipality's *original sale* list -- every brick ever sold,
ordered by buyer surname, with columns:

    Last Name | First Name | Brick# | Line #1 | Line #2

Unlike the by-area PDF (ABCDHIJK.pdf, see parse_brick_list.py) this list has no
section letters, but it covers ALL bricks -- including areas E, F and G, which
the by-area list omits. It does carry the buyer's name, which the by-area list
does not.

Two wrinkles this parser handles:

  * The PDF is a *scan* run through OCR, so extract_text() returns the page
    column-by-column, not row-by-row. Rows are rebuilt from character
    coordinates: cluster by y for the row, then split on x for the column.
  * Its own OCR is noisy in a systematic way (TOWN->IOWN, LOVES->IDVES,
    THE->'IHE). No attempt is made to correct it here -- the raw text is kept
    and matching folds the confusable letters instead (see consensus.scan_key).

Output columns mirror parse_brick_list.py's so match.py consumes either list
unchanged: section (always empty -- unknown here), assigned_id (the Brick#),
pos1/pos2 (empty), full_name (Line #1 + Line #2), key_word (the buyer surname).

Usage:
    python parse_tsp_list.py --pdf "TSP Bricks ALL - OG List by Name - OCR.pdf" \
        --output reference/tsp_brick_list.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_COLUMNS = ["section", "assigned_id", "pos1", "pos2", "full_name", "key_word"]

# Column boundaries as a fraction of the page's content width -- the pages are
# scans and do not share an origin (the run drifts from x~88 to x~56, and one
# stretch sits at x~117), so a fixed set of edges clips a different column on
# every stretch. Get an edge wrong and the neighbouring column eats the
# character: a brick number loses its leading digit, or "DICK & JUNE" is
# recorded as "ICK & JUNE". Calibrated on a well-aligned page, then snapped
# per page to the actual gutter.
_EDGE_FRACTIONS = (0.2102, 0.3653, 0.4701, 0.7530)  # last|first|brick#|line1|line2
_EDGE_SNAP = 14.0       # search this far either side for the emptiest column

_ROW_TOLERANCE = 3.0    # chars within this many points of y are one row
_WORD_GAP = 2.5         # x gap that separates two words
_MIN_ROW_CHARS = 15     # a full table row; fewer means scan speckle


def _page_edges(rows: list[list[tuple]], page_width: float) -> list[float]:
    """Locate one page's four column gutters from its own character coverage.

    Each edge is predicted from the page's content span, then snapped to the
    emptiest x within _EDGE_SNAP points -- the printed gutter.
    """
    full = [r for r in rows if len(r) >= _MIN_ROW_CHARS]
    if not full:
        return [page_width * f for f in _EDGE_FRACTIONS]

    lefts = sorted(min(c[0] for c in row) for row in full)
    rights = sorted(max(c[1] for c in row) for row in full)
    x0 = lefts[len(lefts) // 2]                 # median row start
    x1 = rights[int(0.9 * (len(rights) - 1))]   # 90th pct row end, ignoring outliers
    span = max(x1 - x0, 1.0)

    coverage = [0] * (int(page_width) + 2)
    for row in rows:
        for left, right, _char in row:
            for x in range(max(0, int(left)), min(int(right) + 1, len(coverage))):
                coverage[x] += 1

    edges = []
    for fraction in _EDGE_FRACTIONS:
        guess = x0 + fraction * span
        low = max(0, int(guess - _EDGE_SNAP))
        high = min(len(coverage) - 1, int(guess + _EDGE_SNAP))
        # Emptiest column in the window; ties go to the one nearest the guess.
        edges.append(float(min(range(low, high + 1),
                               key=lambda x: (coverage[x], abs(x - guess)))))
    return edges


def _page_rows(textpage, page_width: float) -> list[list[str]]:
    """Rebuild one page's table rows from character coordinates.

    Returns a list of 5-element [last, first, brick#, line1, line2] rows.
    """
    chars = []
    for i in range(textpage.count_chars()):
        char = textpage.get_text_range(i, 1)
        if not char.strip():
            continue
        left, _bottom, right, top = textpage.get_charbox(i)
        chars.append((top, left, right, char))
    chars.sort(key=lambda c: (-c[0], c[1]))

    # Cluster into rows by y, comparing against the row's first character so a
    # slowly drifting baseline cannot walk one row into the next.
    rows: list[list[tuple]] = []
    row_top = None
    for top, left, right, char in chars:
        if row_top is None or abs(top - row_top) > _ROW_TOLERANCE:
            rows.append([])
            row_top = top
        rows[-1].append((left, right, char))

    for row in rows:
        row.sort(key=lambda c: c[0])
    col_edges = _page_edges(rows, page_width)

    table = []
    for row in rows:
        columns = ["", "", "", "", ""]
        prev_right = None
        for left, right, char in row:
            index = sum(1 for edge in col_edges if left >= edge)
            if columns[index] and prev_right is not None and left - prev_right > _WORD_GAP:
                columns[index] += " "
            columns[index] += char
            prev_right = right
        table.append([c.strip() for c in columns])
    return table


def _clean_brick_number(raw: str) -> str:
    """The scan often splits a brick number ('1 1476'); keep the digits."""
    return re.sub(r"[^0-9]", "", raw)


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, type=Path, help="The TSP list PDF")
    parser.add_argument("--output", required=True, type=Path, help="CSV to write")
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise SystemExit("pypdfium2 is not installed -- run: "
                         "pip install -r requirements.txt")

    pdf = pdfium.PdfDocument(str(args.pdf))
    rows, skipped = [], 0
    for index in range(len(pdf)):
        page = pdf[index]
        for last, first, brick, line1, line2 in _page_rows(page.get_textpage(),
                                                           page.get_size()[0]):
            number = _clean_brick_number(brick)
            inscription = " ".join(part for part in (line1, line2) if part).strip()
            # A row needs a brick number; scan speckle produces stray rows of
            # punctuation, and the header row has no number either.
            if not number or not (inscription or last):
                skipped += 1
                continue
            rows.append({
                "section": "",
                "assigned_id": number,
                "pos1": "", "pos2": "",
                "full_name": inscription,
                "key_word": " ".join(part for part in (last, first) if part),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    blank = sum(1 for r in rows if not r["full_name"])
    print(f"Parsed {len(pdf)} page(s) -> {len(rows)} brick row(s)")
    print(f"  rows without an inscription : {blank}")
    print(f"  non-row fragments skipped   : {skipped}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
