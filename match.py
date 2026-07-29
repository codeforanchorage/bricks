#!/usr/bin/env python3
"""Match the per-brick OCR catalogue against the official Muni brick list.

For each detected brick, fuzzy-matches its candidate reads (one per OCR
method) against the official inscriptions in the parsed reference list, so the
methods' disagreement collapses onto the one real brick -- the closest read
to a known official name wins.

Output: a matched CSV -- each detected brick paired with its best official
match (assigned id, section, official name) and a match score. With --section
it also writes that section's official bricks that no photo matched, i.e.
candidates for lost/broken (a true list only once a section is photographed
in full).

Bricks that stay unmatched are exactly the human-review queue, so alongside
the matched CSV a review file (review_<output name>) lists each unmatched
brick's --top N best candidates (default 5), ranked, one row per candidate --
duplicate-inscription copies collapsed to one entry. A reviewer (or the
planned review UI) sees the photo plus its five most plausible bricks instead
of re-searching the whole list.

As continuous QA, a duplicates file (duplicates_<output name>) lists every
official brick claimed by MORE than one photo. Two photos on one brick is
either a duplicate photo (harmless) or a false positive (one of the reads
matched the wrong brick) -- and when the claims exceed the number of
identical copies of that inscription in the reference, at least one claim
must be wrong. This audits the matcher's zero-false-positive record for free
on every batch.

Matching is brute-force (every read vs every official name); fine for a test
batch, would need key-word blocking to scale to the full ~13,000 bricks.

Usage:
    python match.py --catalog output/bricks_test.csv \
        --reference reference/brick_list.csv --output output/matched.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from consensus import (_match_key, _normalise, _similar, phonetic_fold,
                       scan_fold, similar_spoken, token_containment)

# Catalogue columns that are not OCR-method reads.
_NON_READ = {"image", "brick_id", "x", "y", "w", "h", "status"}
DEFAULT_MIN_SCORE = 0.80   # genuine matches score >=0.84; false positives <=0.71

# A containment-based match (read words found inside a longer inscription) is
# accepted only when the best candidate beats the best *different* inscription
# by this margin -- a generic word set that fits many bricks ties and is
# rejected, while a distinctive subset ("BEATY" + "BUDDY") stands alone.
# Because of that extra gate, the tokens path is allowed a slightly lower
# score bar than whole-string matching (a hallucinated extra word drags the
# average even when the real words match exactly).
CONTAIN_MARGIN = 0.05
CONTAIN_DISCOUNT = 0.05

MATCH_COLUMNS = ["image", "brick_id", "match_status", "match_basis", "score",
                 "official_id", "official_section", "official_name",
                 "official_keyword", "matched_read"]

DEFAULT_TOP = 5
REVIEW_COLUMNS = ["image", "brick_id", "rank", "score", "basis",
                  "official_id", "official_section", "official_name",
                  "official_keyword", "matched_read"]

DUP_COLUMNS = ["official_id", "official_section", "official_name",
               "official_keyword", "copies", "n_claims",
               "image", "brick_id", "score", "basis", "matched_read"]


def _key_for(text: str, scan_ocr: bool) -> str:
    """Comparison key for one inscription."""
    key = _match_key(_normalise(text))
    return scan_fold(key) if scan_ocr else key


def _load_reference(path: Path, section: str, scan_ocr: bool = False) -> list[dict]:
    """Load the official brick list, optionally one section, with normalised text.

    Accepts either parsed-PDF list (parse_brick_list.py / parse_tsp_list.py)
    or the merged master list (merge_lists.py), which is adapted to the same
    shape -- preferring the cleaner new-list inscription when the row has one.
    """
    refs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alt = ""
            if "og_inscription" in row:   # master_list.csv
                alt = row.get("og_alt", "")
                row = {"section": row["section"],
                       "assigned_id": row["new_id"] or row["orig_id"],
                       "full_name": row["new_inscription"] or row["og_inscription"],
                       "key_word": row["buyer"]}
            if section and row["section"].upper() != section.upper():
                continue
            # Match on the distinctive words only -- boilerplate like
            # "IN MEMORY OF" otherwise inflates similarity between bricks.
            row["_key"] = _key_for(row["full_name"], scan_ocr)
            # Second transcription of the same brick (master og_alt): score
            # against both, keep the better -- the texts err differently.
            # But when the two transcriptions barely resemble each other, one
            # of them read the wrong row of the scan; matching against it
            # would hand its neighbour's inscription to this brick, so the
            # alt is ignored (the row itself stays in the review pile).
            key2 = _key_for(alt, scan_ocr) if alt else ""
            if key2 and _similar(row["_key"], key2) < 0.40:
                key2 = ""
            row["_key2"] = key2 if key2 != row["_key"] else ""
            if row["_key"] or row["_key2"]:
                if not row["_key"]:
                    row["_key"], row["_key2"] = row["_key2"], ""
                refs.append(row)
    return refs


def _block_index(reference: list[dict]) -> dict[str, list[int]]:
    """Folded word -> reference indices, so most reads score only candidates
    sharing at least one distinctive word instead of the whole list."""
    index = defaultdict(list)
    for i, ref in enumerate(reference):
        words = set(ref["_key"].split())
        if ref.get("_key2"):
            words.update(ref["_key2"].split())
        # Index phonetic folds too, or CATHY never blocks against KATHY.
        words.update(phonetic_fold(w) for w in tuple(words))
        for word in words:
            if len(word) >= 3:
                index[word].append(i)
    return index


def _candidates(reads: list[str], reference: list[dict], index: dict,
                scan_ocr: bool) -> list[dict]:
    """Reference rows sharing >=1 folded word with any read."""
    hits = set()
    for read in reads:
        words = set(_key_for(read, scan_ocr).split())
        words.update(phonetic_fold(w) for w in tuple(words))
        for word in words:
            if len(word) >= 3:
                hits.update(index.get(word, ()))
    return [reference[i] for i in hits]


def _best_match(reads: list[str], reference: list[dict], scan_ocr: bool = False,
                min_score: float = DEFAULT_MIN_SCORE):
    """Best official brick for any of a detection's reads.

    Each (read, reference) pair is scored two ways: whole-string similarity,
    and token containment (see consensus.token_containment) for reads that are
    a noisy subset of a longer inscription. The containment path only counts
    when whole-string similarity fell short, and its acceptance needs a
    CONTAIN_MARGIN lead over the best differently-inscribed candidate --
    tracked here as `runner_up` (duplicate inscriptions excluded, so a brick
    sold in identical copies does not veto itself).

    Returns (score, basis, margin, ref_row, winning_read); basis is "text" or
    "tokens".
    """
    best_score, best_basis, best_ref, best_read = 0.0, "text", None, ""
    runner_up = 0.0
    for read in reads:
        key = _key_for(read, scan_ocr)
        if not key:
            continue
        for ref in reference:
            # similar_spoken: parts of the official lists were voice-
            # transcribed, so phonetic spelling variants (BRIAN/BRYAN,
            # CATHY/KATHY) must compare as equal. Score against both of the
            # row's transcriptions when it has two -- they err differently.
            ref_keys = [ref["_key"]]
            if ref.get("_key2"):
                ref_keys.append(ref["_key2"])
            score = max(similar_spoken(key, rk) for rk in ref_keys)
            basis = "text"
            if score < min_score:
                contained = max(token_containment(key, rk) for rk in ref_keys)
                if contained > score:
                    score, basis = contained, "tokens"
            if score > best_score:
                if best_ref is not None and ref["_key"] != best_ref["_key"]:
                    runner_up = best_score
                best_score, best_basis, best_ref, best_read = score, basis, ref, read
            elif score > runner_up and (best_ref is None
                                        or ref["_key"] != best_ref["_key"]):
                runner_up = score
    return best_score, best_basis, best_score - runner_up, best_ref, best_read


def _top_candidates(reads: list[str], reference: list[dict], scan_ocr: bool,
                    min_score: float, n: int) -> list[tuple]:
    """Best `n` distinct-inscription candidates for one brick's reads.

    This is the review-queue view of the same scoring _best_match uses: every
    (read, reference) pair scored by whole-string similarity with a token-
    containment fallback, but keeping the n best DIFFERENT inscriptions
    instead of only the winner -- a brick sold in identical copies fills one
    review slot, not all of them. No acceptance gates apply: these rows are
    for a human, and the near-misses are exactly what the human needs to see.

    Returns [(score, basis, ref_row, winning_read), ...] best-first.
    """
    best_by_key: dict[str, tuple] = {}
    for read in reads:
        key = _key_for(read, scan_ocr)
        if not key:
            continue
        for ref in reference:
            ref_keys = [ref["_key"]]
            if ref.get("_key2"):
                ref_keys.append(ref["_key2"])
            score = max(similar_spoken(key, rk) for rk in ref_keys)
            basis = "text"
            if score < min_score:
                contained = max(token_containment(key, rk) for rk in ref_keys)
                if contained > score:
                    score, basis = contained, "tokens"
            prev = best_by_key.get(ref["_key"])
            if prev is None or score > prev[0]:
                best_by_key[ref["_key"]] = (score, basis, ref, read)
    ranked = sorted(best_by_key.values(), key=lambda item: -item[0])
    return ranked[:n]


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", required=True, type=Path,
                        help="Per-brick catalogue from brick_pipeline.py")
    parser.add_argument("--reference", required=True, type=Path,
                        help="Official brick list from parse_brick_list.py")
    parser.add_argument("--output", required=True, type=Path,
                        help="Matched CSV to write")
    parser.add_argument("--section", default="",
                        help="Restrict matching to one section (A-K) and "
                             "report that section's unmatched official bricks")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"Similarity needed to accept a match "
                             f"(default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--scan-ocr", action="store_true",
                        help="Fold OCR-confusable letters (I/L/T, E/F, O/D) on "
                             "both sides. Use with the scanned by-name list "
                             "(tsp_brick_list.csv), whose own OCR is noisy.")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help=f"For each unmatched brick, list its best N "
                             f"distinct candidates in review_<output name> "
                             f"for the human review queue (default: "
                             f"{DEFAULT_TOP}; 0 disables)")
    args = parser.parse_args(argv)

    for path in (args.catalog, args.reference):
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")

    reference = _load_reference(args.reference, args.section, args.scan_ocr)
    if not reference:
        raise SystemExit(f"No reference bricks loaded (section {args.section!r}?)")

    with open(args.catalog, newline="", encoding="utf-8") as f:
        catalog = list(csv.DictReader(f))
    read_cols = [c for c in (catalog[0] if catalog else {})
                 if c not in _NON_READ]

    scope = f"section {args.section.upper()}" if args.section else "all sections"
    print(f"Matching {len(catalog)} detected brick(s) against "
          f"{len(reference)} official brick(s) ({scope}) ...")

    index = _block_index(reference)
    # How many identical copies of each inscription exist -- a batch brick
    # legitimately absorbs one photo per copy before a claim is suspect.
    copy_counts = defaultdict(int)
    for r in reference:
        copy_counts[r["_key"]] += 1

    rows, review_rows = [], []
    matched_ids = set()
    claims = defaultdict(list)   # (section, id) -> matched photos of that brick
    for brick in catalog:
        reads = [brick[c] for c in read_cols
                 if brick.get(c) and not brick[c].startswith("ERROR:")]
        # Fast path: only score references sharing a folded word with a read.
        # A read whose every word is misspelled beyond folding shares none, so
        # a miss falls back to the full list -- slow only for the failures.
        pool = _candidates(reads, reference, index, args.scan_ocr)
        score, basis, margin, ref, read = _best_match(reads, pool,
                                                      args.scan_ocr,
                                                      args.min_score)
        if ref is None or score < args.min_score - CONTAIN_DISCOUNT:
            if len(pool) < len(reference):
                score, basis, margin, ref, read = _best_match(reads, reference,
                                                              args.scan_ocr,
                                                              args.min_score)
        min_needed = (args.min_score if basis == "text"
                      else args.min_score - CONTAIN_DISCOUNT)
        matched = (ref is not None and score >= min_needed
                   and (basis == "text" or margin >= CONTAIN_MARGIN))
        if matched:
            matched_ids.add(ref["assigned_id"])
            claims[(ref["section"], ref["assigned_id"])].append({
                "official_id": ref["assigned_id"],
                "official_section": ref["section"],
                "official_name": ref["full_name"],
                "official_keyword": ref["key_word"],
                "copies": copy_counts[ref["_key"]],
                "image": brick.get("image", ""),
                "brick_id": brick.get("brick_id", ""),
                "score": f"{score:.2f}",
                "basis": basis,
                "matched_read": read,
            })
        elif args.top > 0:
            # The review queue: this brick needs a human, so hand them the
            # closest distinct candidates from the WHOLE reference (not the
            # blocked pool -- a badly misread brick shares no words with its
            # own inscription, which is often why it is unmatched at all).
            for rank, (c_score, c_basis, c_ref, c_read) in enumerate(
                    _top_candidates(reads, reference, args.scan_ocr,
                                    args.min_score, args.top), 1):
                review_rows.append({
                    "image": brick.get("image", ""),
                    "brick_id": brick.get("brick_id", ""),
                    "rank": rank,
                    "score": f"{c_score:.2f}",
                    "basis": c_basis,
                    "official_id": c_ref["assigned_id"],
                    "official_section": c_ref["section"],
                    "official_name": c_ref["full_name"],
                    "official_keyword": c_ref["key_word"],
                    "matched_read": c_read,
                })
        rows.append({
            "image": brick.get("image", ""),
            "brick_id": brick.get("brick_id", ""),
            "match_status": "matched" if matched else "unmatched",
            "match_basis": basis,
            "score": f"{score:.2f}",
            "official_id": ref["assigned_id"] if ref else "",
            "official_section": ref["section"] if ref else "",
            "official_name": ref["full_name"] if ref else "",
            "official_keyword": ref["key_word"] if ref else "",
            "matched_read": read,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_matched = sum(1 for r in rows if r["match_status"] == "matched")
    n_unmatched = len(rows) - n_matched
    print(f"\n  matched   : {n_matched}/{len(rows)} detected bricks")
    print(f"  unmatched : {n_unmatched}")
    print(f"\nWrote {args.output}")

    if args.top > 0 and n_unmatched:
        review_path = args.output.with_name(f"review_{args.output.name}")
        with open(review_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(review_rows)
        print(f"Review queue: {n_unmatched} unmatched brick(s), top "
              f"{args.top} candidates each -> {review_path}")

    # QA: any official brick claimed by more than one photo is either a
    # duplicate photo or a false positive; claims beyond the inscription's
    # identical-copy count mean at least one claim IS wrong.
    dups = {key: group for key, group in claims.items() if len(group) > 1}
    if dups:
        dup_rows = []
        for key in sorted(dups):
            group = sorted(dups[key], key=lambda c: (c["image"],
                                                     str(c["brick_id"])))
            dup_rows += [{**claim, "n_claims": len(group)} for claim in group]
        dup_path = args.output.with_name(f"duplicates_{args.output.name}")
        with open(dup_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DUP_COLUMNS)
            writer.writeheader()
            writer.writerows(dup_rows)
        over = sum(1 for group in dups.values()
                   if len(group) > group[0]["copies"])
        print(f"Duplicate claims: {len(dups)} brick(s) matched by more than "
              f"one photo -> {dup_path}")
        print(f"  (duplicate photos or false positives; {over} exceed the "
              f"inscription's copy count -- at least one of those claims is "
              f"wrong)")

    if args.section:
        missing = [r for r in reference if r["assigned_id"] not in matched_ids]
        miss_path = args.output.with_name(f"missing_{args.section.upper()}.csv")
        with open(miss_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["section", "assigned_id", "full_name", "key_word"])
            for r in missing:
                writer.writerow([r["section"], r["assigned_id"],
                                 r["full_name"], r["key_word"]])
        print(f"\nSection {args.section.upper()}: {len(missing)} official "
              f"brick(s) not matched by any photo -> {miss_path}")
        print("  (a true lost/broken list only once the section is "
              "photographed in full)")


if __name__ == "__main__":
    main()
