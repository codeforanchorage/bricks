#!/usr/bin/env python3
"""Classify warehouse photos: single brick close-up, or stack/overview?

The photo batches include shots of whole pallets and tall stacks (taken
to document the pallet, not a brick). They can never match a list row,
so they pollute the review queue as unmatchable entries. This script
asks a cheap vision model to label each photo so the review page can
sort them into their own section at the bottom.

Labels written to the output CSV (image,label):

  single   a close-up of one engraved brick face
  stack    a pallet, stack, or pile of bricks (side/overview shot)
  other    neither (blank paver, label, scenery, unreadable)

Resumable: images already present in the output CSV are skipped, so add
new photos and re-run cheaply (~$0.0002/photo).

Usage:
    python classify_photos.py --matched output/pallets_matched.csv \
        --photos photos/ --output output/photo_types.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline import _load_dotenv
from vision_ocr import load_jpeg_bytes

MODEL = "gemini-3.1-flash-lite"
WORKERS = 8
LABELS = {"single", "stack", "other"}

PROMPT = (
    "Classify this warehouse photo. Reply with exactly one word:\n"
    "SINGLE - ONE engraved commemorative brick face is the clear subject "
    "(centered, in focus, filling much of the frame). Choose SINGLE even "
    "when stacks or pallets of other bricks are visible in the "
    "background or around the edges.\n"
    "STACK - the subject is a pallet, stack, or pile of bricks as a "
    "whole (side or overview shot, brick edges, shrink-wrap) with NO "
    "single readable brick face as the main subject.\n"
    "OTHER - anything else (blank brick, label, scenery, unreadable)"
)


def _classify(client, image: Path) -> str:
    from google.genai import types
    try:
        jpeg = load_jpeg_bytes(image)
    except Exception:      # corrupt/unreadable file -- nothing to classify
        return "other"
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=jpeg,
                                                mime_type="image/jpeg"),
                          PROMPT])
            word = (r.text or "").strip().split()[0].strip(".,").lower()
            if word in LABELS:
                return word
        except Exception:
            time.sleep(4 * (attempt + 1))
    return "other"


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv(Path(__file__).with_name(".env"))

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matched", required=True, type=Path,
                        help="matched CSV; its UNMATCHED photos get "
                             "classified (matched ones are single bricks "
                             "by definition)")
    parser.add_argument("--photos", required=True, type=Path,
                        help="Photo root (the pipeline's --input)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)

    done: dict[str, str] = {}
    if args.output.is_file():
        with open(args.output, newline="", encoding="utf-8") as f:
            done = {r["image"]: r["label"] for r in csv.DictReader(f)}

    todo = []
    with open(args.matched, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image = row.get("image", "")
            if (row.get("match_status") != "matched" and image
                    and image not in done):
                path = args.photos / image
                if path.is_file():
                    todo.append((image, path))
    print(f"{len(todo)} unmatched photo(s) to classify "
          f"({len(done)} already done)")
    if not todo:
        return

    from google import genai
    client = genai.Client()
    results: dict[str, str] = {}

    def work(item):
        image, path = item
        results[image] = _classify(client, path)
        if len(results) % 50 == 0:
            print(f"  {len(results)}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(work, todo))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        for image, label in {**done, **results}.items():
            writer.writerow([image, label])

    counts = {}
    for label in {**done, **results}.values():
        counts[label] = counts.get(label, 0) + 1
    print(f"Wrote {args.output}: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
