#!/usr/bin/env python3
"""Adopt clean BUYER names from the strip images into the by-name list.

Buyer names have only one source: the text layer of the scanned by-name
list, which is poor OCR ("IXJRST DIANA H" for "DURST DIANA H."). The clean
Municipal Excel file carries no buyer names, only a search keyword that is
often a word from the inscription -- so nothing has ever corrected this
column, while the inscriptions got a second pass.

The strip images are clean typewriter print and hold the buyer name, the
brick number and the inscription on one line. audit_strips.py --text
transcribes them; this script parses each line and replaces the buyer
(v2's key_word) where two guards both pass:

  1. the number printed in the strip equals the strip's filename -- the
     same wrong-row guard rescan_rows.py uses, so a neighbouring row's
     name can't be adopted;
  2. the inscription in the strip matches the row's own inscription, which
     confirms the strip really belongs to this row before its name is
     taken;
  3. with --text2, a SECOND cheap model's transcription of the same strip
     must produce the same buyer name. One read is not enough: a single
     pass turned "SPARKS VELMA" into "SPARKS VEIMA" (an I/L slip of
     exactly the kind these lists are full of). This is the same
     two-model agreement rule rescan_rows.py uses for list text.

    python audit_strips.py --text --output output/strip_full_text.csv
    python adopt_buyers.py --text output/strip_full_text.csv

Writes the updated list in place (backup .bak_buyers) and a report of
every change. Re-run merge_lists.py afterwards to rebuild master_list.csv;
the buyer surname also corroborates the two lists' join, so re-run the
match as well.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import re
import shutil
import sys
from pathlib import Path

from consensus import _normalise, scan_fold

ARTIFACTS = re.compile(r"(IXJ|IlJ|I'I|'IO|IDV|ll[A-Z]|[0-9])")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", (s or "").upper())).strip()


def split_row(line: str) -> tuple[str, str, str]:
    """-> (buyer, number, inscription) from 'SURNAME FIRST 6891 INSCRIPTION'."""
    m = re.search(r"\b(\d{1,5})\b", line or "")
    if not m:
        return "", "", ""
    return line[:m.start()].strip(" .,"), m.group(1), line[m.end():].strip(" .,")


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", type=Path, default=Path("output/strip_full_text.csv"))
    p.add_argument("--text2", type=Path,
                   help="a second model's transcriptions; a buyer is "
                        "adopted only where both models agree")
    p.add_argument("--list", type=Path,
                   default=Path("reference/tsp_brick_list_v2.csv"))
    p.add_argument("--report", type=Path,
                   default=Path("output/buyer_adoptions.csv"))
    p.add_argument("--min-inscription", type=float, default=0.50,
                   help="how well the strip's inscription must match the "
                        "row's before its buyer name is trusted")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    reads = {r["file"]: r["printed"]
             for r in csv.DictReader(open(a.text, newline="", encoding="utf-8"))}
    reads2 = {}
    if a.text2 and a.text2.is_file():
        reads2 = {r["file"]: r["printed"] for r in
                  csv.DictReader(open(a.text2, newline="", encoding="utf-8"))}
    rows = list(csv.DictReader(open(a.list, newline="", encoding="utf-8")))
    cols = list(rows[0].keys())
    by_id: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        n = (r["assigned_id"] or "").strip().lstrip("0")
        if n:
            by_id.setdefault(n, []).append(i)

    adopted, same, gated, unparsed, disagreed = [], 0, 0, 0, 0
    for f, line in reads.items():
        if not line or line.startswith("ERROR") or line == "NONE":
            unparsed += 1
            continue
        buyer, number, inscription = split_row(line)
        stem = f.lstrip("0") or "0"
        if not buyer or number.lstrip("0") != stem:      # guard 1
            gated += 1
            continue
        for i in by_id.get(stem, []):
            row = rows[i]
            own = max(
                difflib.SequenceMatcher(None, norm(inscription),
                                        norm(row[fld])).ratio()
                for fld in ("full_name", "alt_name"))
            if own < a.min_inscription:                  # guard 2
                gated += 1
                continue
            old = row["key_word"]
            if norm(old) == norm(buyer):
                same += 1
                continue
            if reads2:                                   # guard 3
                b2, n2, _ = split_row(reads2.get(f, ""))
                if n2.lstrip("0") != stem or norm(b2) != norm(buyer):
                    disagreed += 1
                    continue
            adopted.append({
                "assigned_id": row["assigned_id"], "strip": f,
                "old_buyer": old, "new_buyer": buyer,
                "old_had_artifacts": bool(ARTIFACTS.search((old or "").upper())),
                "inscription_match": f"{own:.2f}",
                "strip_line": line[:120],
            })
            row["key_word"] = buyer

    print(f"strips read {len(reads)}: buyer adopted {len(adopted)}, "
          f"already identical {same}, failed a guard {gated}, "
          f"models disagreed {disagreed}, unreadable {unparsed}")
    with open(a.report, "w", newline="", encoding="utf-8") as fh:
        if adopted:
            w = csv.DictWriter(fh, fieldnames=list(adopted[0].keys()))
            w.writeheader()
            w.writerows(adopted)
    print(f"wrote {a.report}")
    if a.dry_run:
        print("dry run -- list not written")
        return
    shutil.copy2(a.list, a.list.with_suffix(".csv.bak_buyers"))
    with open(a.list, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"rewrote {a.list} (backup .csv.bak_buyers)")


if __name__ == "__main__":
    main()
