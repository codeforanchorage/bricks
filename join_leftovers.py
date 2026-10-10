#!/usr/bin/env python3
"""Looser join pass between the two lists' leftovers.

merge_lists.py leaves ~400 by-name rows 'unjoined' (moved bricks whose
scanned text found no renovation-workbook counterpart at the 0.80 bar or
the 0.70 surname-corroborated rescue) and ~600 workbook rows that no
by-name row ever claimed. Some of each are the same bricks, joined by
neither text nor number. This pass scores ONLY those two leftover pools
against each other (small enough to score every pair, no blocking) and
brings in a third witness the merge never had: the pallet photo's OCR
read for the ~130 unjoined rows the photo matcher found on a pallet.

Nothing is written into the master list. The output is a candidate file
for a human (or a later adoption step) to act on:

    output/unjoined_join_candidates.csv

one line per unjoined row with its best workbook candidate, scores from
each witness, the runner-up score (uniqueness margin), surname and
section corroboration, and a tier:

    strong  -- text or photo read matches the workbook text >= 0.80, with
               a clear margin over the runner-up and (when the photo read
               carried it) a read that also resembles the row's own text
    rescue  -- >= 0.70 with a clear margin and the buyer surname in the
               candidate (merge_lists' own rescue rule, applied to the
               photo read as well as the scan text)
    weak    -- >= 0.55 with surname AND pallet-implied section agreeing,
               or a photo read >= 0.70 -- needs a human eye
    none    -- nothing plausible

With --write-overrides PATH the adoptable candidates are also written to
the join-overrides file merge_lists.py reads (reference/join_overrides.csv):
clean 'strong' rows, a contested row whose buyer surname is in the workbook
text (SILVER SPONSOR ATU), and -- for a contested group of IDENTICAL
inscriptions (7 IBEW sales for one unclaimed copy) -- the lowest number,
since the sales are indistinguishable and one of them owns the brick.
Existing lines in the file (incl. hand-added ones) are kept; a candidate
is added only when neither its OG row nor its workbook row is already in
the file. Rescue and weak rows are never written: they are for human eyes.

Usage:
    python join_leftovers.py [--master reference/master_list.csv]
        [--xls reference/brick_list_xls.csv]
        [--matched output/pallets_final.csv]
        [--output output/unjoined_join_candidates.csv]
        [--write-overrides reference/join_overrides.csv]
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from datetime import date

from consensus import _normalise, _similar, scan_fold
from merge_lists import (OVERRIDE_COLUMNS, RESCUE_MARGIN, RESCUE_MIN_SCORE,
                         _key, _surname_corroborates)

STRONG = 0.80
WEAK = 0.55
PHOTO_WEAK = 0.70
# A pallet "implies" a section once that section is this share of the
# pallet's matched bricks (pallets are ~90% one section, with strays).
PALLET_SECTION_SHARE = 0.60

COLUMNS = ["tier", "flags", "orig_id", "buyer", "og_text", "photo_reads",
           "pallets", "implied_section", "cand_section", "cand_new_id",
           "cand_inscription", "cand_key_word", "text_score", "photo_score",
           "og_vs_photo", "word_overlap", "best_score", "runner_up", "surname_ok",
           "section_ok", "cand_claimed_by"]
# The photo witness is only as good as the photo->unjoined-row match that
# produced it. A read that does not resemble the row's own scanned text
# is a suspect photo match (a false positive on the photo side), so it
# may not carry a join on its own.
PHOTO_TRUST = 0.50
MARGIN = RESCUE_MARGIN
# Character similarity on short folded keys is loose ("goo biess iawrence
# genevieve schachie" vs "oee iess iawrence" scores well on the shared
# surname alone), so a strong join must also share most of its WORDS.
# Words compare fuzzily (the workbook misspells: Kelly/Kelley,
# Donahue/Donohue) and run-together inscriptions ("DEDICATEDJENNEWAGNER")
# are compared with spaces removed.
WORD_OVERLAP = 0.60
WORD_SIM = 0.75


def _word_overlap(a: str, b: str) -> float:
    """Share of the two keys' words (3+ letters) that fuzzily match across."""
    wa = [w for w in a.split() if len(w) >= 3]
    wb = [w for w in b.split() if len(w) >= 3]
    if len(wa) < 2 or len(wb) < 2:
        # too few words to count: fall back to the space-less strings
        return _similar(a.replace(" ", ""), b.replace(" ", ""))
    hit_a = sum(1 for w in wa if any(_similar(w, v) >= WORD_SIM for v in wb))
    hit_b = sum(1 for v in wb if any(_similar(w, v) >= WORD_SIM for w in wa))
    return (hit_a + hit_b) / (len(wa) + len(wb))


def _load_unjoined(path: Path) -> list[dict]:
    """Unjoined rows, minus scan duplicates of a row that DID join.

    The scanned list repeats some rows (dup_orig): same certificate number,
    same buyer, same text. When one copy joined the workbook the other is
    the same sale, not a second brick -- it must not be given a brick.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    joined = {(r["orig_id"], r["buyer"], _key(r["og_alt"] or r["og_inscription"]))
              for r in rows if r["status"] == "ok"}
    out, dups = [], 0
    for r in rows:
        if r["status"] != "unjoined":
            continue
        if (r["orig_id"], r["buyer"], _key(r["og_alt"] or r["og_inscription"])) in joined:
            dups += 1
            continue
        out.append(r)
    print(f"skipped {dups} unjoined rows that duplicate a joined row")
    return out


def _load_unclaimed(path: Path, master: Path) -> list[dict]:
    with open(master, newline="", encoding="utf-8") as f:
        claimed = {(r["section"].upper(), r["new_id"])
                   for r in csv.DictReader(f) if r["new_id"]}
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["section"].upper(), r["assigned_id"]) in claimed:
                continue
            r["_key"] = _key(r["full_name"])
            r["key_word"] = r.get("key_word", "")
            if r["_key"]:
                out.append(r)
    return out


def _photo_witness(path: Path) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """orig_id -> matched photo rows for unjoined rows; pallet -> section."""
    by_id: dict[str, list[dict]] = defaultdict(list)
    pallet_sections: dict[str, Counter] = defaultdict(Counter)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("match_status") != "matched":
                continue
            if r.get("official_section"):
                pallet_sections[r["pallet"]][r["official_section"].upper()] += 1
            else:
                by_id[r["official_id"]].append(r)
    implied = {}
    for pallet, counts in pallet_sections.items():
        section, n = counts.most_common(1)[0]
        if n >= PALLET_SECTION_SHARE * sum(counts.values()):
            implied[pallet] = section
    return by_id, implied


def _best(keys: list[str], cands: list[dict]) -> tuple[float, float, dict | None]:
    best, second, row = 0.0, 0.0, None
    for c in cands:
        s = max((_similar(k, c["_key"]) for k in keys if k), default=0.0)
        if s > best:
            if row is not None and c["_key"] != row["_key"]:
                second = best
            best, row = s, c
        elif s > second and (row is None or c["_key"] != row["_key"]):
            second = s
    return best, second, row


def _adoptable(results: list[dict]) -> list[dict]:
    """Strong rows a machine may adopt without a human (see module doc)."""
    out = []
    groups = defaultdict(list)
    for x in results:
        if x["tier"] == "strong":
            groups[(x["cand_section"], x["cand_new_id"])].append(x)
    for group in groups.values():
        if len(group) == 1 and not group[0]["flags"]:
            out.append(group[0])
            continue
        flags = {f for x in group for f in x["flags"].split(";") if f}
        if flags - {"contested"}:
            continue                       # low margin etc.: human only
        winners = [x for x in group if x["surname_ok"] == "yes"]
        if len(winners) == 1:
            out.append(winners[0])
        elif len({_key(x["og_text"]) for x in group}) == 1:
            out.append(min(group, key=lambda x: int(x["orig_id"])
                           if x["orig_id"].isdigit() else 10**9))
    return out


def _write_overrides(results: list[dict], path: Path) -> None:
    existing = []
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    taken_og = {(r["orig_id"], r["buyer"]) for r in existing}
    taken_new = {(r["section"].upper(), r["new_id"]) for r in existing}
    added = 0
    for x in _adoptable(results):
        og, new = (x["orig_id"], x["buyer"]), (x["cand_section"].upper(), x["cand_new_id"])
        if og in taken_og or new in taken_new:
            continue
        taken_og.add(og)
        taken_new.add(new)
        existing.append({
            "orig_id": x["orig_id"], "buyer": x["buyer"],
            "section": x["cand_section"], "new_id": x["cand_new_id"],
            "new_inscription": x["cand_inscription"], "score": x["best_score"],
            "witness": "photo" if float(x["photo_score"]) > float(x["text_score"]) else "text",
            "source": "join_leftovers strong" + (" (contested, surname)" if "contested" in x["flags"] and x["surname_ok"] == "yes"
                                                else " (identical copies, lowest #)" if "contested" in x["flags"] else ""),
            "date": date.today().isoformat(),
        })
        added += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OVERRIDE_COLUMNS)
        w.writeheader()
        w.writerows(existing)
    print(f"overrides: +{added} -> {path} ({len(existing)} lines total)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--master", type=Path, default=Path("reference/master_list.csv"))
    ap.add_argument("--xls", type=Path, default=Path("reference/brick_list_xls.csv"))
    ap.add_argument("--matched", type=Path, default=Path("output/pallets_final.csv"))
    ap.add_argument("--output", type=Path,
                    default=Path("output/unjoined_join_candidates.csv"))
    ap.add_argument("--write-overrides", type=Path, default=None,
                    help="also append the adoptable joins to this "
                         "merge_lists override file")
    args = ap.parse_args(argv)

    unjoined = _load_unjoined(args.master)
    cands = _load_unclaimed(args.xls, args.master)
    photos, implied = _photo_witness(args.matched)
    print(f"{len(unjoined)} unjoined by-name rows x {len(cands)} unclaimed "
          f"workbook rows; {sum(1 for r in unjoined if photos.get(r['orig_id']))} "
          "unjoined rows have a pallet photo")

    results = []
    for r in unjoined:
        text_keys = [_key(r["og_inscription"]), _key(r["og_alt"])]
        reads = [p["matched_read"] for p in photos.get(r["orig_id"], [])
                 if p.get("matched_read")]
        pallets = sorted({p["pallet"] for p in photos.get(r["orig_id"], [])})
        photo_keys = [_key(x) for x in reads]
        t_best, t_second, t_row = _best(text_keys, cands)
        p_best, p_second, p_row = _best(photo_keys, cands) if photo_keys else (0.0, 0.0, None)
        # Prefer the witness with the stronger claim; keep its runner-up.
        if p_best > t_best:
            best, second, cand = p_best, p_second, p_row
        else:
            best, second, cand = t_best, t_second, t_row
        if cand is None:
            continue
        text_score = max((_similar(k, cand["_key"]) for k in text_keys if k), default=0.0)
        photo_score = max((_similar(k, cand["_key"]) for k in photo_keys), default=0.0)
        surname_ok = _surname_corroborates(r["buyer"], cand)
        implied_sections = {implied[p] for p in pallets if p in implied}
        section_ok = (cand["section"].upper() in implied_sections
                      if implied_sections else None)
        margin = best - second
        og_vs_photo = max((_similar(t, k) for t in text_keys if t
                           for k in photo_keys), default=0.0)
        flags = []
        # the witness text that produced best_score, for the overlap check
        witness_keys = photo_keys if photo_score > text_score else text_keys
        overlap = max((_word_overlap(k, cand["_key"]) for k in witness_keys if k),
                      default=0.0)
        if overlap < WORD_OVERLAP:
            flags.append("few_shared_words")
        photo_led = photo_score > text_score
        if photo_led and photo_keys and og_vs_photo < PHOTO_TRUST:
            flags.append("photo_disagrees_og")
        # The row's two scanned transcriptions (parse + vision re-read)
        # normally agree; when they don't, one of them read a neighbouring
        # scan row, and a join carried by the stray text is a join to the
        # neighbour's brick.
        if (not photo_led and all(text_keys)
                and _similar(text_keys[0], text_keys[1]) < PHOTO_TRUST):
            flags.append("og_texts_disagree")
        # A text join carried by the parse alone, while the (usually
        # cleaner, fuller) vision re-read scores poorly, rode a truncated
        # or garbled parse: "G9D BIESS IAWRENCE" fits any Lawrence.
        if (not photo_led and text_keys[1]
                and _similar(text_keys[1], cand["_key"]) < RESCUE_MIN_SCORE
                and _similar(text_keys[0], cand["_key"]) >= STRONG):
            flags.append("parse_only")
        if margin < MARGIN:
            flags.append("low_margin")
        if best >= STRONG and not flags:
            tier = "strong"
        elif best >= STRONG:
            tier = "weak"
        elif best >= RESCUE_MIN_SCORE and margin >= RESCUE_MARGIN and surname_ok:
            tier = "rescue"
        elif (best >= WEAK and surname_ok and section_ok) or photo_score >= PHOTO_WEAK:
            tier = "weak"
        else:
            tier = "none"
        results.append({
            "tier": tier, "flags": ";".join(flags),
            "orig_id": r["orig_id"], "buyer": r["buyer"],
            "og_text": r["og_alt"] or r["og_inscription"],
            "photo_reads": " || ".join(reads), "pallets": "; ".join(pallets),
            "implied_section": "".join(sorted(implied_sections)),
            "cand_section": cand["section"], "cand_new_id": cand["assigned_id"],
            "cand_inscription": cand["full_name"], "cand_key_word": cand["key_word"],
            "text_score": f"{text_score:.2f}", "photo_score": f"{photo_score:.2f}",
            "og_vs_photo": f"{og_vs_photo:.2f}",
            "word_overlap": f"{overlap:.2f}",
            "best_score": f"{best:.2f}", "runner_up": f"{second:.2f}",
            "surname_ok": "yes" if surname_ok else "no",
            "section_ok": {True: "yes", False: "NO", None: ""}[section_ok],
            "cand_claimed_by": "",
        })

    # Flag workbook rows that more than one unjoined row wants (one brick
    # cannot be two sales unless it is an identical-copy batch).
    wanted = defaultdict(list)
    for x in results:
        if x["tier"] != "none":
            wanted[(x["cand_section"], x["cand_new_id"])].append(x)
    for group in wanted.values():
        if len(group) < 2:
            continue
        # When exactly one claimant's buyer surname is in the workbook
        # text (SILVER SPONSOR ATU vs the other silver sponsors), it wins
        # and the rest drop to weak. Otherwise (identical-copy batches
        # like 7 IBEW rows for one unclaimed copy) all stay, flagged, for
        # a human to pick.
        winners = [x for x in group if x["surname_ok"] == "yes"]
        for x in group:
            x["cand_claimed_by"] = f"{len(group)} rows"
            lost = len(winners) == 1 and x is not winners[0]
            x["flags"] = ";".join(filter(None, [
                x["flags"], "contested_lost" if lost else "contested"]))
            if lost and x["tier"] in ("strong", "rescue"):
                x["tier"] = "weak"

    order = {"strong": 0, "rescue": 1, "weak": 2, "none": 3}
    results.sort(key=lambda x: (order[x["tier"]], -float(x["best_score"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(results)

    tiers = Counter(x["tier"] for x in results)
    print("tiers:", dict(tiers))
    if args.write_overrides:
        _write_overrides(results, args.write_overrides)
    print("photo witness decided the candidate:",
          sum(1 for x in results if x["tier"] != "none"
              and float(x["photo_score"]) > float(x["text_score"])))
    print("contested workbook rows:",
          sum(1 for x in results if x["cand_claimed_by"]))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
