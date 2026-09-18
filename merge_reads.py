#!/usr/bin/env python3
"""Merge extra OCR-method reads into the photo catalogue (output/pallets.csv).

single_pipeline.py writes one column per method it ran, and match.py /
make_review_page.py treat EVERY non-reserved column as another read of the
same photo. A second-opinion pass with premium models (e.g.
``--methods opus-5,gemini-pro-31`` over the unresolved photos only) therefore
plugs straight into the normal pipeline once its columns are merged in:

    python merge_reads.py --catalog output/pallets.csv \
        --extra output/premium_reads.csv

Rows are keyed on (image, brick_id). Only the extra file's method columns
are copied; its rows for photos absent from the catalogue are ignored.
ERROR: reads are not copied. Re-run after run_pipeline's ``ocr`` step --
single_pipeline --resume rewrites the catalogue with only its own method
columns, which drops merged ones.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from match import _NON_READ


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--extra", required=True, type=Path, nargs="+")
    args = parser.parse_args(argv)

    with open(args.catalog, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    index = {(r["image"], str(r["brick_id"])): r for r in rows}

    copied = 0
    for extra in args.extra:
        with open(extra, newline="", encoding="utf-8") as f:
            ereader = csv.DictReader(f)
            ecols = [c for c in (ereader.fieldnames or []) if c not in _NON_READ]
            for c in ecols:
                if c not in columns:
                    columns.append(c)
            for erow in ereader:
                row = index.get((erow["image"], str(erow["brick_id"])))
                if row is None:
                    continue
                for c in ecols:
                    val = (erow.get(c) or "").strip()
                    if val and not val.startswith("ERROR:"):
                        row[c] = val
                        copied += 1

    shutil.copy2(args.catalog, args.catalog.with_suffix(".csv.prev"))
    with open(args.catalog, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})
    print(f"merged {copied} read(s) into {args.catalog} "
          f"(columns now: {', '.join(c for c in columns if c not in _NON_READ)})")


if __name__ == "__main__":
    main()
