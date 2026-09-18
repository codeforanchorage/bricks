#!/usr/bin/env python3
"""Adopt inscriptions from the strip images into the by-name list.

Same idea as adopt_buyers.py, one step further: the strip image is the
highest-fidelity look at the printed row, so where two cheap models agree
on what it says, that beats the text the weaker earlier passes produced.

Guards (all three must pass):
  1. both models read a brick number equal to the strip's filename -- the
     wrong-row guard, so a neighbour's row cannot be adopted;
  2. both models agree on the inscription, character for character after
     folding punctuation (one read is not enough -- a single pass turned
     "SPARKS VELMA" into "SPARKS VEIMA" during the buyer pass);
  3. the BUYER printed on the strip matches the row's own buyer. The
     inscription is what is being replaced, so it cannot also be the
     row-identity check; the buyer column (cleaned from these same strips)
     is the independent anchor.

    python adopt_inscriptions.py --text output/strip_full_text.csv \
        --text2 output/strip_full_text_b.csv

Writes reference/tsp_brick_list_v2.csv in place (backup .bak_inscr) plus a
full report. Re-run merge_lists.py and the match afterwards -- inscriptions
are what the matcher keys on.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import re
import shutil
import sys
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", (s or "").upper())).strip()


def split_row(line: str) -> tuple[str, str, str]:
    m = re.search(r"\b(\d{1,5})\b", line or "")
    if not m:
        return "", "", ""
    return line[:m.start()].strip(" .,"), m.group(1), line[m.end():].strip(" .,")


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", type=Path, default=Path("output/strip_full_text.csv"))
    p.add_argument("--text2", type=Path,
                   default=Path("output/strip_full_text_b.csv"))
    p.add_argument("--list", type=Path,
                   default=Path("reference/tsp_brick_list_v2.csv"))
    p.add_argument("--report", type=Path,
                   default=Path("output/inscription_adoptions.csv"))
    p.add_argument("--min-buyer", type=float, default=0.60)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    r1 = {r["file"]: r["printed"] for r in
          csv.DictReader(open(a.text, newline="", encoding="utf-8"))}
    r2 = {r["file"]: r["printed"] for r in
          csv.DictReader(open(a.text2, newline="", encoding="utf-8"))}
    rows = list(csv.DictReader(open(a.list, newline="", encoding="utf-8")))
    cols = list(rows[0].keys())
    by_id: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        n = (r["assigned_id"] or "").strip().lstrip("0")
        if n:
            by_id.setdefault(n, []).append(i)

    adopted, same, gate_num, gate_dis, gate_buyer, missing = [], 0, 0, 0, 0, 0
    for f, line in r1.items():
        if f not in r2:
            missing += 1
            continue
        b1, n1, i1 = split_row(line)
        b2, n2, i2 = split_row(r2[f])
        stem = f.lstrip("0") or "0"
        if n1.lstrip("0") != stem or n2.lstrip("0") != stem:
            gate_num += 1
            continue
        if norm(i1) != norm(i2) or not i1:
            gate_dis += 1
            continue
        for i in by_id.get(stem, []):
            row = rows[i]
            buyer_fit = difflib.SequenceMatcher(
                None, norm(b1), norm(row["key_word"])).ratio()
            if buyer_fit < a.min_buyer:
                gate_buyer += 1
                continue
            cur = row["alt_name"] or row["full_name"]
            if norm(cur) == norm(i1):
                same += 1
                continue
            adopted.append({
                "assigned_id": row["assigned_id"], "strip": f,
                "buyer": row["key_word"], "buyer_fit": f"{buyer_fit:.2f}",
                "old_text": cur, "new_text": i1,
                "similarity": f"{difflib.SequenceMatcher(None, norm(cur), norm(i1)).ratio():.2f}",
                "old_source": row["verified"],
            })
            row["alt_name"] = i1
            row["verified"] = "strip"
            row["flag"] = ";".join(x for x in [row["flag"], "strip_v3"] if x)

    print(f"strips with two reads {len(r2)}: inscription adopted "
          f"{len(adopted)}, already identical {same} | gated: "
          f"number {gate_num}, models disagree {gate_dis}, "
          f"buyer mismatch {gate_buyer}, one read only {missing}")
    with open(a.report, "w", newline="", encoding="utf-8") as fh:
        if adopted:
            w = csv.DictWriter(fh, fieldnames=list(adopted[0].keys()))
            w.writeheader()
            w.writerows(adopted)
    print(f"wrote {a.report}")
    if a.dry_run:
        print("dry run -- list not written")
        return
    shutil.copy2(a.list, a.list.with_suffix(".csv.bak_inscr"))
    with open(a.list, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"rewrote {a.list} (backup .csv.bak_inscr)")


if __name__ == "__main__":
    main()
