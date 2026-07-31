#!/usr/bin/env python3
"""Re-read the OG list's troublesome rows with two current vision models.

The v2 list's alt_name re-reads were made with an older model generation.
This targeted pass re-reads ONLY the rows where better text can change an
outcome -- the master list's unjoined and flagged rows, plus the rows
offered as candidates for review-queue photos (near-misses where list
noise, not brick wear, blocked the match).

Each target row's strip is transcribed by TWO models independently; the
new text is adopted into alt_name only when the two agree (folded
similarity >= AGREE_SIM), so a single model's hallucination cannot
degrade a row. A wrong-row guard additionally rejects any adoption whose
read brick number differs from the parse row's number -- strip crops can
include a sliver of the neighbouring row, and models sometimes read that
row instead (even in agreement with each other). The coordinate parse (full_name) is NEVER touched -- the
matcher scores both texts and keeps the better, and replacing the parse
wholesale famously lost 566 joins. Brick numbers are never changed here
either. The buyer column is upgraded on the same both-models-agree rule.

Reads are cached to --state (JSONL, one line per strip) as they finish,
so an interrupted run -- credit wall, Ctrl-C, crash -- resumes for free:
re-run the same command and only unread strips hit the API. The cache
records which model pair produced it and is ignored on a model change;
after changing the strip RENDERER, pass --fresh (the cache cannot see
image changes).

Afterwards, rebuild and re-measure:
    python merge_lists.py --og reference/tsp_brick_list_v2.csv \
        --new reference/brick_list_xls.csv --output reference/master_list.csv
    python match.py --catalog output/pallets.csv ...   # re-match
    python -m pytest tests/                            # regression gate

Usage:
    python rescan_rows.py --pdf "TSP Bricks ALL - OG List by Name - OCR.pdf" \
        --review output/review_pallets_matched.csv --limit 40   # sample
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from consensus import _match_key, _normalise, _similar, scan_fold
from parse_tsp_list import _clean_brick_number
from pipeline import _load_dotenv
from resolve_tsp_rows import AGREE_SIM, _StripReader, physical_rows

# Both cheap, non-preview, validated 23/23 on the labeled photos. The
# second model was gemini-3-flash-preview until 2026-07-30: a thinking
# model whose hidden thinking tokens bill as output -- it quietly burned
# ~10x the expected cost on strip reads. Keep preview/thinking models out
# of bulk passes.
MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
WORKERS = 8


def _key(text: str) -> str:
    return scan_fold(_match_key(_normalise(text)))


def _load_cache(state: Path, models: list[str]) -> dict[str, list]:
    """Cached reads from an interrupted run -- same model pair only."""
    cache: dict[str, list] = {}
    if not state.is_file():
        return cache
    with open(state, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn final line from a crash
            if entry.get("models") == models:
                cache[entry["number"]] = entry["reads"]
    return cache


def _target_ids(master: Path, reviews: list[Path]) -> set[str]:
    """orig_ids whose OG-list text is implicated in a bad outcome."""
    targets: set[str] = set()
    by_assigned: dict[tuple[str, str], str] = {}
    with open(master, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            orig = row["orig_id"]
            key = (row["section"].upper(), row["new_id"] or orig)
            by_assigned[key] = orig
            if row["status"] == "unjoined" or row["flag"]:
                targets.add(orig)
    for review in reviews:
        with open(review, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["official_section"].upper(), row["official_id"])
                orig = by_assigned.get(key)
                if orig:
                    targets.add(orig)
    targets.discard("")
    return targets


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv(Path(__file__).with_name(".env"))

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--v2", type=Path,
                        default=Path("reference/tsp_brick_list_v2.csv"))
    parser.add_argument("--master", type=Path,
                        default=Path("reference/master_list.csv"))
    parser.add_argument("--review", type=Path, nargs="*", default=[],
                        help="review_*.csv file(s) from match.py; their "
                             "candidate rows join the target set")
    parser.add_argument("--sections", default="",
                        help="Comma-separated sections whose EVERY row "
                             "joins the target set (e.g. E to pre-clean "
                             "a scan-only section before its pallet is "
                             "photographed)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap strip reads (0 = no cap) -- for a sample")
    parser.add_argument("--state", type=Path,
                        default=Path("output/rescan_state.jsonl"),
                        help="Read-cache file: interrupted runs resume "
                             "from it for free")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore and overwrite the read cache (use "
                             "after changing the strip renderer)")
    args = parser.parse_args(argv)

    targets = _target_ids(args.master, args.review)
    wanted_sections = {s.strip().upper()
                       for s in args.sections.split(",") if s.strip()}
    if wanted_sections:
        with open(args.master, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["section"].upper() in wanted_sections \
                        and row["orig_id"]:
                    targets.add(row["orig_id"])
    print(f"Target rows (unjoined + flagged + review candidates"
          + (f" + sections {','.join(sorted(wanted_sections))}"
             if wanted_sections else "") + f"): {len(targets)}")

    rows_by_number: dict[str, list[dict]] = {}
    for entry in physical_rows(args.pdf):
        if entry.get("flag") == "page0":
            continue           # preamble page: strips crop wrongly
        rows_by_number.setdefault(entry["number"], []).append(entry)

    jobs = []
    for number in targets:
        entries = rows_by_number.get(number, [])
        if len(entries) == 1:  # a colliding number = ambiguous strip; skip
            jobs.append(entries[0])
    # Page order keeps the renderer's small page cache effective (targets
    # arrive scattered -- the list is sorted by NAME, not number).
    jobs.sort(key=lambda e: (e["page"], -e["crop"][0]))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"Strips to re-read (x{len(MODELS)} models): {len(jobs)}")

    # Read cache: strips already read by a previous (interrupted) run with
    # the SAME model pair are served from disk instead of the API.
    cache = {} if args.fresh else _load_cache(args.state, MODELS)
    if cache:
        print(f"read cache: {len(cache)} strip(s) resume for free "
              f"({args.state})")

    readers = [_StripReader(args.pdf, model=m) for m in MODELS]
    # pdfium is not thread-safe across instances either: share one render
    # lock and one page cache so all rendering serializes, and a page
    # rendered for model A is reused for model B.
    for extra in readers[1:]:
        extra._lock = readers[0]._lock
        extra._pages = readers[0]._pages
    done = [0]
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state_lock = threading.Lock()
    state_file = open(args.state, "w" if args.fresh else "a",
                      encoding="utf-8")

    def _work(entry):
        cached = cache.get(entry["number"])
        if cached is not None:
            entry["reads"] = cached
        else:
            reads = []
            for reader in readers:
                try:
                    reads.append(reader.read(entry["page"], *entry["crop"]))
                except RuntimeError as exc:
                    print(f"  {exc}", flush=True)
                    reads.append(None)
            entry["reads"] = reads
            # Failed reads are NOT cached -- a retry should re-attempt them.
            if all(r is not None for r in reads):
                with state_lock:
                    state_file.write(json.dumps(
                        {"number": entry["number"], "models": MODELS,
                         "reads": reads}) + "\n")
                    state_file.flush()
        done[0] += 1
        if done[0] % 100 == 0:
            print(f"  {done[0]}/{len(jobs)} strips", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(_work, jobs))

    # Wrong-row feedback: when BOTH models read the same WRONG number N',
    # the crop meant for N actually presented its neighbour -- so N's row
    # lies on the other side. Retry once with the crop reflected across
    # the expected centre, away from N', and re-guard strictly.
    retries = []
    for entry in jobs:
        reads = entry.get("reads") or []
        if len(reads) < len(MODELS) or any(r is None for r in reads):
            continue
        nums = {_clean_brick_number(str(r.get("brick", ""))) for r in reads}
        if len(nums) != 1:
            continue
        wrong = nums.pop()
        if not wrong or wrong == entry["number"]:
            continue
        wrong_entries = rows_by_number.get(wrong, [])
        if len(wrong_entries) != 1 \
                or wrong_entries[0]["page"] != entry["page"]:
            continue
        w_top, w_bottom = wrong_entries[0]["extent"]
        top, bottom = entry["crop"]
        shift = ((top + bottom) - (w_top + w_bottom)) / 2
        entry["retry_crop"] = (top + shift, bottom + shift)
        retries.append(entry)

    recovered = [0]

    def _retry(entry):
        reads = []
        for reader in readers:
            try:
                reads.append(reader.read(entry["page"],
                                         *entry["retry_crop"]))
            except RuntimeError:
                reads.append(None)
        nums = [_clean_brick_number(str(r.get("brick", "")))
                for r in reads if r]
        if (len(reads) == len(MODELS) and all(r is not None for r in reads)
                and all(n == entry["number"] for n in nums)):
            entry["reads"] = reads       # re-aimed onto the right row
            recovered[0] += 1

    if retries:
        print(f"wrong-row feedback: retrying {len(retries)} re-aimed "
              f"strip(s)", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(_retry, retries))
        print(f"  recovered {recovered[0]} of {len(retries)}", flush=True)

    state_file.close()

    # Adopt only both-model agreements.
    adopted: dict[str, dict] = {}
    disagreed = failed = wrong_row = 0
    for entry in jobs:
        reads = [r for r in entry.get("reads", []) if r]
        if len(reads) < len(MODELS):
            failed += 1
            continue
        # Wrong-row guard: a strip crop can include a sliver of the
        # neighbouring row, and models sometimes read THAT row instead --
        # measured on a 50-strip premium-model test, where two models
        # agreed with each other on a neighbour's text twice. A wrong-row
        # read carries the neighbour's brick number, so any read number
        # that differs from the parse row's number rejects the adoption.
        numbers = [_clean_brick_number(str(r.get("brick", ""))) for r in reads]
        if any(n and n != entry["number"] for n in numbers):
            wrong_row += 1
            continue
        texts = [" ".join(str(r.get(k, "")) for k in ("line1", "line2")).strip()
                 for r in reads]
        buyers = [" ".join(str(r.get(k, "")) for k in ("last", "first")).strip()
                  for r in reads]
        if not texts[0] or _similar(_key(texts[0]), _key(texts[1])) < AGREE_SIM:
            disagreed += 1
            continue
        update = {"alt_name": " ".join(texts[0].split())}
        if buyers[0] and _similar(_key(buyers[0]), _key(buyers[1])) >= AGREE_SIM:
            update["key_word"] = " ".join(buyers[0].split())
        adopted[entry["number"]] = update

    backup = args.v2.with_suffix(".csv.bak")
    shutil.copy2(args.v2, backup)
    with open(args.v2, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        v2_rows = list(reader)
    changed_alt = changed_buyer = 0
    for row in v2_rows:
        update = adopted.get(row["assigned_id"])
        if not update:
            continue
        if update["alt_name"] != row["alt_name"]:
            row["alt_name"] = update["alt_name"]
            changed_alt += 1
        if "key_word" in update and update["key_word"] != row["key_word"]:
            row["key_word"] = update["key_word"]
            changed_buyer += 1
    with open(args.v2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(v2_rows)

    tokens_in = sum(r.in_tokens for r in readers)
    tokens_out = sum(r.out_tokens for r in readers)
    print(f"\nBoth models agreed : {len(adopted)} rows "
          f"({changed_alt} alt_name updated, {changed_buyer} buyer updated)")
    print(f"Models disagreed   : {disagreed} rows (kept as they were)")
    print(f"Wrong-row rejected : {wrong_row} rows (read number != parse "
          f"number -- likely the neighbouring row)")
    print(f"Read failures      : {failed}")
    print(f"Wrote {args.v2}  (backup: {backup})")
    print(f"Tokens: {tokens_in} in / {tokens_out} out")
    print("\nNext: rebuild master (merge_lists.py), re-run match.py, "
          "run pytest.")


if __name__ == "__main__":
    main()
