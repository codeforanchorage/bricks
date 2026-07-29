#!/usr/bin/env python3
"""Merge the two official brick lists into one master lookup table.

The reclaim workflow needs one row per brick that answers a visitor's
questions: does my brick exist, what is its current number, and which section
(-> pallet group) is it in. That takes both lists (see README):

  * tsp_brick_list.csv  -- the ORIGINAL sale list: every brick, original
    (certificate) number, buyer surname. No sections.
  * brick_list.csv      -- the post-2008 "new brick numbers" list: relocated
    bricks only (areas A-D, H-K), new number + section. No buyer names.

The 2009 "How to Find Your Brick" brochure supplies the bridge: bricks in
areas E, F and G were NOT moved in 2008, and their original numbers are the
contiguous ranges below -- so for those, the original number alone gives the
section. Every other brick was renumbered, and its new number/section must be
found by matching inscription text (the numbers are unrelated across lists;
the join is fuzzy because the OG list is a noisy scan).

Output: reference/master_list.csv, one row per original-list brick:

  orig_id, new_id, section, moved, status, buyer, og_inscription,
  new_inscription, join_score

status is 'ok', 'no_brick' (the OG list marks some sales "NO BRICK NO
INSCRIPTION"), or 'unjoined' (a moved brick whose text found no counterpart --
review by hand). The reverse gaps (new-list rows no OG row claimed) are
written alongside as master_unclaimed.csv.

Usage:
    python merge_lists.py --og reference/tsp_brick_list.csv \
        --new reference/brick_list.csv --output reference/master_list.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from consensus import _match_key, _normalise, _similar, scan_fold

# Original-number ranges of the areas that did NOT move in the 2008
# renovation (2009 brochure / muni.org). For these, orig number = new number.
UNMOVED_RANGES = {"F": (1, 3377), "G": (8279, 9126), "E": (9127, 10070)}

JOIN_MIN_SCORE = 0.80   # same acceptance bar as match.py
_MIN_BLOCK_WORD = 3     # blocking words shorter than this are too common

# Second pass over the leftovers: the scan noise on some rows is too heavy for
# the main threshold, so accept a weaker text score -- but only when the best
# candidate BEATS the runner-up clearly AND the buyer's surname (which the new
# list lacks but the candidate inscription usually contains) corroborates it.
# Tested on the leftovers: surname corroboration rejects the lookalike pairs
# the margin rule alone lets through (e.g. "DAVID FISaIER" -> David Mitchell).
RESCUE_MIN_SCORE = 0.70
RESCUE_MARGIN = 0.06
RESCUE_SURNAME_SIM = 0.65

COLUMNS = ["orig_id", "new_id", "section", "moved", "status", "buyer",
           "og_inscription", "og_alt", "new_inscription", "join_score"]


def _key(text: str) -> str:
    """Folded distinctive-word key -- both lists on equal footing."""
    return scan_fold(_match_key(_normalise(text)))


def _unmoved_section(orig_id: int) -> str | None:
    for section, (lo, hi) in UNMOVED_RANGES.items():
        if lo <= orig_id <= hi:
            return section
    return None


def _load_new(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["_key"] = _key(row["full_name"])
            if row["_key"]:
                rows.append(row)
    return rows


def _block_index(new_rows: list[dict]) -> dict[str, list[int]]:
    """word -> row indices, so a fuzzy join only scores plausible candidates."""
    index = defaultdict(list)
    for i, row in enumerate(new_rows):
        for word in set(row["_key"].split()):
            if len(word) >= _MIN_BLOCK_WORD:
                index[word].append(i)
    return index


def _join(og_key: str, new_rows: list[dict],
          index: dict) -> tuple[float, float, dict | None]:
    """Best new-list row for one OG inscription, via word-blocked fuzzy match.

    Returns (best score, runner-up score, best row) -- the runner-up gives the
    rescue pass its uniqueness margin.
    """
    candidates = set()
    for word in set(og_key.split()):
        if len(word) >= _MIN_BLOCK_WORD:
            candidates.update(index.get(word, ()))
    best_score, second, best_row = 0.0, 0.0, None
    for i in candidates:
        score = _similar(og_key, new_rows[i]["_key"])
        if score > best_score:
            best_score, second, best_row = score, best_score, new_rows[i]
        elif score > second:
            second = score
    return best_score, second, best_row


def _surname_corroborates(buyer: str, candidate: dict) -> bool:
    """Does the OG buyer's surname appear (fuzzily) in the candidate row?"""
    surname = scan_fold(_normalise(buyer)).split()
    if not surname:
        return False
    words = set(candidate["_key"].split())
    words.update(scan_fold(_normalise(candidate["key_word"])).split())
    return any(_similar(surname[0], w) >= RESCUE_SURNAME_SIM for w in words)


def _spread_identical_copies(out_rows: list[dict], new_rows: list[dict],
                             claimed: set, counts: dict) -> None:
    """Reassign identical-inscription bricks one-to-one.

    Organisations bought batches of identical bricks (9x ACPA, 17x
    "98.9 MAGIC FM", ...). Independent best-matching sends every OG copy to
    the SAME new-list row; here each group of identical claims is spread
    across that inscription's identical new-list rows instead. Which copy
    pairs with which is arbitrary (the bricks are indistinguishable), so both
    sides are taken in id order for determinism. OG copies beyond the new
    list's copy count revert to 'unjoined' -- they have no distinct
    counterpart and should surface in review rather than silently share one.
    """
    by_new_id = defaultdict(list)
    for row in out_rows:
        if row["moved"] == "yes" and row["new_id"]:
            by_new_id[row["new_id"]].append(row)

    copies_by_key = defaultdict(list)
    for ref in new_rows:
        copies_by_key[ref["_key"]].append(ref)

    for new_id, claimants in by_new_id.items():
        if len(claimants) < 2:
            continue
        target = next(r for r in new_rows if r["assigned_id"] == new_id)
        copies = sorted(copies_by_key[target["_key"]],
                        key=lambda r: int(r["assigned_id"].replace(",", "") or 0))
        claimants.sort(key=lambda r: int(r["orig_id"]) if r["orig_id"].isdigit()
                       else 0)
        for i, row in enumerate(claimants):
            if i < len(copies):
                ref = copies[i]
                row.update(new_id=ref["assigned_id"], section=ref["section"],
                           new_inscription=ref["full_name"])
                claimed.add(id(ref))
            else:
                row.update(new_id="", section="", new_inscription="",
                           join_score="", status="unjoined")
                counts["joined"] -= 1
                counts["unjoined"] += 1
                counts["copy_overflow"] += 1


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--og", required=True, type=Path,
                        help="Original by-name list (parse_tsp_list.py output)")
    parser.add_argument("--new", required=True, type=Path,
                        help="New by-area list (parse_brick_list.py output)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Master lookup CSV to write")
    parser.add_argument("--min-score", type=float, default=JOIN_MIN_SCORE,
                        help=f"Similarity to accept a text join "
                             f"(default: {JOIN_MIN_SCORE})")
    args = parser.parse_args(argv)

    new_rows = _load_new(args.new)
    index = _block_index(new_rows)
    with open(args.og, newline="", encoding="utf-8") as f:
        og_rows = list(csv.DictReader(f))
    print(f"OG list: {len(og_rows)} rows;  new by-area list: {len(new_rows)} rows")

    out_rows, claimed = [], set()
    counts = defaultdict(int)
    for og in og_rows:
        orig_id = og["assigned_id"]
        inscription = og["full_name"]
        alt = og.get("alt_name", "")
        row = {"orig_id": orig_id, "new_id": "", "section": "", "moved": "",
               "status": "ok", "buyer": og["key_word"],
               "og_inscription": inscription, "og_alt": alt,
               "new_inscription": "", "join_score": ""}

        # Fold before testing: the scan renders NO BRICK as e.g. "NO ERICK".
        folded = scan_fold(_normalise(inscription))
        if "no brick" in folded or "no erick" in folded:
            row["status"] = "no_brick"
            counts["no_brick"] += 1
            out_rows.append(row)
            continue

        section = _unmoved_section(int(orig_id)) if orig_id.isdigit() else None
        if section:
            row.update(section=section, new_id=orig_id, moved="no")
            counts["unmoved"] += 1
        else:
            row["moved"] = "yes"
            # The v2 OG list carries two transcriptions of each row (scan
            # parse + vision re-read); join on whichever scores better. The
            # runner-up must be a DIFFERENT new row -- when both texts pick
            # the same brick, the weaker text's score is corroboration, not
            # competition.
            score, second, match = 0.0, 0.0, None
            for text in (inscription, alt):
                og_key = _key(text) if text else ""
                if not og_key:
                    continue
                s, s2, m = _join(og_key, new_rows, index)
                second = max(second, s2)
                if m is None:
                    continue
                if match is None or m["_key"] == match["_key"]:
                    if s > score:
                        score, match = s, m
                elif s > score:
                    second = max(second, score)
                    score, match = s, m
                else:
                    second = max(second, s)
            accepted = match is not None and score >= args.min_score
            rescued = (not accepted and match is not None
                       and score >= RESCUE_MIN_SCORE
                       and score - second >= RESCUE_MARGIN
                       and _surname_corroborates(og["key_word"], match))
            if accepted or rescued:
                row.update(new_id=match["assigned_id"], section=match["section"],
                           new_inscription=match["full_name"],
                           join_score=f"{score:.2f}")
                claimed.add(id(match))
                counts["rescued" if rescued else "joined"] += 1
            else:
                row["status"] = "unjoined"
                counts["unjoined"] += 1
        out_rows.append(row)

    _spread_identical_copies(out_rows, new_rows, claimed, counts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    unclaimed = [r for r in new_rows if id(r) not in claimed]
    unclaimed_path = args.output.with_name("master_unclaimed.csv")
    with open(unclaimed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "new_id", "full_name", "key_word"])
        for r in unclaimed:
            writer.writerow([r["section"], r["assigned_id"],
                             r["full_name"], r["key_word"]])

    print(f"\n  unmoved (E/F/G, section from number) : {counts['unmoved']}")
    print(f"  moved, joined by text                : {counts['joined']}")
    print(f"  moved, rescued (margin + surname)    : {counts['rescued']}")
    print(f"  moved, no text match (review)        : {counts['unjoined']}"
          + (f" (incl. {counts['copy_overflow']} identical-copy overflow)"
             if counts["copy_overflow"] else ""))
    print(f"  'NO BRICK' sales                     : {counts['no_brick']}")
    print(f"\nWrote {args.output}")
    print(f"New-list rows no OG row claimed: {len(unclaimed)} -> {unclaimed_path}")


if __name__ == "__main__":
    main()
