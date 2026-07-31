#!/usr/bin/env python3
"""Fold a reviewer's decisions.csv back into the matched catalogue.

decisions.csv comes from the review.html page (make_review_page.py): one row
per decided photo with decision 'match' (plus the chosen brick), 'none', or
'illegible'. This script rewrites the matched catalogue with those decisions
applied:

  match      -> match_status 'matched', match_basis 'human', score 1.00,
                official_* filled from the reviewer's chosen candidate
  none       -> match_status 'no_match'   (human confirmed: not in the list)
  illegible  -> match_status 'illegible'  (human confirmed: photo unreadable)
  stack      -> match_status 'stack_photo' (a pallet/stack overview shot,
                not an individual brick -- nothing to match)

Every decided row also records who decided (reviewer) and their note, so the
final catalogue is auditable. Undecided rows pass through unchanged. Multiple
decisions files (several reviewers, or several sittings) can be given at
once; a later file wins on conflict and the clash is printed.

Usage:
    python apply_decisions.py --matched output/matched.csv \
        --decisions decisions.csv --output output/matched_final.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from match import MATCH_COLUMNS

FINAL_COLUMNS = MATCH_COLUMNS + ["reviewer", "review_note"]
_DECISIONS = {"match", "none", "illegible", "stack"}
_STATUS = {"none": "no_match", "illegible": "illegible",
           "stack": "stack_photo"}


def _load_decisions(paths: list[Path]) -> dict[tuple[str, str], dict]:
    """(image, brick_id) -> decision row; later files win on conflict."""
    decisions: dict[tuple[str, str], dict] = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                decision = (row.get("decision") or "").strip().lower()
                if decision not in _DECISIONS:
                    continue
                key = (row.get("image", ""), str(row.get("brick_id", "")))
                if key in decisions and decisions[key] != row:
                    print(f"  note: {key[0]} decided twice -- keeping the "
                          f"one from {path.name}")
                decisions[key] = row
    return decisions


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matched", required=True, type=Path,
                        help="Matched catalogue from match.py")
    parser.add_argument("--decisions", required=True, type=Path, nargs="+",
                        help="decisions.csv file(s) from review.html")
    parser.add_argument("--output", required=True, type=Path,
                        help="Final catalogue CSV to write")
    args = parser.parse_args(argv)

    decisions = _load_decisions(args.decisions)
    print(f"Loaded {len(decisions)} decision(s) from "
          f"{len(args.decisions)} file(s)")

    with open(args.matched, newline="", encoding="utf-8") as f:
        matched = list(csv.DictReader(f))

    counts = {"match": 0, "none": 0, "illegible": 0, "stack": 0}
    applied = set()
    out_rows = []
    for row in matched:
        row = {c: row.get(c, "") for c in FINAL_COLUMNS}
        key = (row["image"], str(row["brick_id"]))
        decision = decisions.get(key)
        if decision:
            applied.add(key)
            kind = decision["decision"].strip().lower()
            counts[kind] += 1
            row["reviewer"] = decision.get("reviewer", "")
            row["review_note"] = decision.get("note", "")
            if kind == "match":
                row.update(match_status="matched", match_basis="human",
                           score="1.00",
                           official_id=decision.get("official_id", ""),
                           official_section=decision.get("official_section", ""),
                           official_name=decision.get("official_name", ""),
                           official_keyword=decision.get("official_keyword", ""))
            else:
                # Human-confirmed dead ends keep their machine candidates
                # blank -- the catalogue must not imply a near-match was real.
                row.update(match_status=_STATUS[kind],
                           match_basis="human", score="",
                           official_id="", official_section="",
                           official_name="", official_keyword="")
        out_rows.append(row)

    orphans = set(decisions) - applied
    if orphans:
        print(f"  WARNING: {len(orphans)} decision(s) matched no catalogue "
              f"row: {sorted(o[0] for o in orphans)[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    total = sum(1 for r in out_rows if r["match_status"] == "matched")
    print(f"\n  human-matched   : {counts['match']}")
    print(f"  confirmed absent: {counts['none']}")
    print(f"  illegible photo : {counts['illegible']}")
    print(f"  stack/overview  : {counts['stack']}")
    print(f"  total matched   : {total}/{len(out_rows)}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
