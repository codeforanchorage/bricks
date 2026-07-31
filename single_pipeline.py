#!/usr/bin/env python3
"""One-photo-per-brick OCR: OCR each whole image as a single brick.

The warehouse reclaim workflow also produces close-ups of one brick at a time
(held up, or a single paver filling the frame). Those photos have no paver grid
to find, so detect_bricks would only add garbage boxes -- this front end skips
detection and sends the whole image to the OCR method(s) with the single-brick
prompt.

The output is the same catalogue format brick_pipeline.py writes, so match.py
consumes it unchanged:

    python single_pipeline.py --input test_images/ --output output/singles.csv
    python match.py --catalog output/singles.csv \
        --reference reference/brick_list.csv --output output/singles_matched.csv

This is the production runner for the full ~13,000-photo batch, so it is
built to survive one:

  * --workers N (default 8) OCRs images concurrently -- the LLM calls are
    network-bound, so threads scale nearly linearly. Rows land in the CSV in
    completion order (match.py does not care about order).
  * --resume keeps a previous run's good rows and redoes only the images that
    are missing or whose reads all failed (ERROR: rows -- transient API
    failures), so a crash at photo 12,000 does not restart from zero.
  * every row is flushed as it is written; Ctrl-C loses nothing already done.
  * subdirectories are walked, and the warehouse folder conventions
    photos/<SECTION>/<pallet>/IMG.jpg (section known) and
    photos/pallets/<pallet>/IMG.jpg (section unknown -- pallet labels are
    arbitrary, the match reveals the section) are read back into the
    catalogue: the `image` column is the path relative to --input (so
    IMG_0001.jpg on two pallets never collide), and `section` / `pallet`
    columns record where the photo was taken. Photos in a flat folder work
    exactly as before (both columns empty).
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from brick_pipeline import _ocr_crop, _status
from pipeline import METHODS, _load_dotenv, find_images

DEFAULT_WORKERS = 8

# Valid warehouse section letters (Town Square areas A-K).
SECTION_LETTERS = set("ABCDEFGHIJK")


def _photo_meta(image: Path, root: Path) -> tuple[str, str, str]:
    """(catalogue name, section, pallet) for one photo under --input.

    Two folder conventions are read, because the warehouse pallet labels
    are arbitrary (K, H1..H6, ...) and do NOT encode the park section:

      photos/<SECTION>/<pallet>/IMG.jpg   section known at photo time
        e.g. photos/H/K/IMG_0421.jpg -> ("H/K/IMG_0421.jpg", "H", "K")
      photos/pallets/<pallet>/IMG.jpg     section unknown -- the usual case
        e.g. photos/pallets/K/IMG_0421.jpg
          -> ("pallets/K/IMG_0421.jpg", "", "K")

    A pallet-only photo matches against the full master list and the match
    itself reveals the section (every master row carries one) -- the pallet
    tag is what the pickup counter needs to find the physical brick.

    The catalogue name is the path relative to --input (POSIX slashes), so
    identical camera filenames on different pallets stay distinct rows --
    match.py, the review page, and apply_decisions.py all treat it as an
    opaque key, and the review page resolves it under --photos unchanged.
    A photo directly in a section folder gets a section but no pallet; a
    photo at the top level (the flat pre-convention layout) gets neither.
    A first folder that is neither a section letter A-K nor `pallets`
    yields no tags at all -- the folder was not following the convention,
    so guessing a pallet from it would tag the row with junk.
    """
    rel = image.relative_to(root)
    folders = rel.parts[:-1]
    section, pallet = "", ""
    if folders and folders[0].upper() in SECTION_LETTERS:
        section = folders[0].upper()
        pallet = folders[1] if len(folders) >= 2 else ""
    elif len(folders) >= 2 and folders[0].lower() == "pallets":
        pallet = folders[1]
    return rel.as_posix(), section, pallet


def _keep_previous_rows(path: Path, method_keys: list[str],
                        columns: list[str]) -> tuple[list[dict], set[str]]:
    """Rows from an interrupted run worth keeping on --resume.

    A row is kept when every requested method has a column in the old file
    and none of its reads is an ERROR row. An empty read is a real answer
    (the model saw no readable text) and is kept; an ERROR read was a
    transient API failure and the image is redone.
    """
    keep, seen = [], set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("image")
            if not name or name in seen:
                continue
            reads = [row.get(METHODS[m][0]) for m in method_keys]
            if any(r is None for r in reads):        # method not run before
                continue
            if any(r.startswith("ERROR:") for r in reads):
                continue                             # transient failure: redo
            seen.add(name)
            keep.append({c: row.get(c, "") for c in columns})
    return keep, seen


def _ocr_image(job: tuple[Path, str, str, str],
               method_keys: list[str]) -> tuple[Path, dict]:
    """OCR one whole image with every requested method -> its catalogue row."""
    image, rel, section, pallet = job
    reads = {m: _ocr_crop(image, m)[0] for m in method_keys}
    row = {"image": rel, "brick_id": 1,
           "section": section, "pallet": pallet,
           "x": "", "y": "", "w": "", "h": "",
           "status": _status(reads)}
    for m in method_keys:
        row[METHODS[m][0]] = reads[m].replace("\n", " / ")
    return image, row


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    _load_dotenv(Path(__file__).with_name(".env"))

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Directory of single-brick photos "
                             "(.jpg/.jpeg/.png/.heic); subfolders are "
                             "walked, and <SECTION>/<pallet>/ or "
                             "pallets/<pallet>/ folder names become the "
                             "section/pallet columns")
    parser.add_argument("--output", required=True, type=Path,
                        help="Catalogue CSV to write")
    parser.add_argument("--methods", default="gemini-lite-31",
                        help="Comma-separated methods (default: gemini-lite-31 "
                             "-- validated 2026-07-29: 26/26 labeled photos, "
                             "0 wrong IDs; replaces the deprecated "
                             "gemini-flash, which Google shuts down "
                             "2026-10-16)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent OCR workers (default: "
                             f"{DEFAULT_WORKERS}). The LLM calls are "
                             f"network-bound, so threads scale well.")
    parser.add_argument("--resume", action="store_true",
                        help="Keep the good rows already in --output; redo "
                             "only missing images and ERROR rows")
    args = parser.parse_args(argv)

    requested = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in requested if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown)}. "
                         f"Valid: {', '.join(METHODS)}")
    method_keys = [m for m in METHODS if m in requested]
    if not method_keys:
        raise SystemExit("No methods selected.")

    workers = max(1, args.workers)
    if "paddle" in method_keys and workers > 1:
        # The PaddleOCR engine is a shared single instance and is not
        # thread-safe; the LLM methods are. Fall back to serial.
        print("note: paddle selected -- forcing --workers 1 "
              "(the PaddleOCR engine is not thread-safe)")
        workers = 1

    images = find_images(args.input, recursive=True)
    if not images:
        raise SystemExit(f"No images (.jpg/.jpeg/.png/.heic) found in "
                         f"{args.input}")
    jobs = [(image, *_photo_meta(image, args.input)) for image in images]

    # Nested photos whose folders follow neither convention (a section
    # letter A-K, or pallets/<pallet>/) get no tags -- say so up front,
    # while the folders can still be renamed cheaply, instead of after a
    # 13,000-photo run.
    untagged = sorted({job[1].split("/")[0] for job in jobs
                       if "/" in job[1] and not job[2] and not job[3]})
    if untagged:
        print(f"note: subfolder(s) matching neither <SECTION A-K>/ nor "
              f"pallets/<pallet>/, their photos get no section/pallet "
              f"tag: {', '.join(untagged)}")

    # x/y/w/h are kept for format compatibility; the brick is the whole frame.
    columns = (["image", "brick_id", "section", "pallet", "x", "y", "w", "h"]
               + [METHODS[m][0] for m in method_keys] + ["status"])

    kept_rows: list[dict] = []
    if args.resume and args.output.is_file():
        kept_rows, done = _keep_previous_rows(args.output, method_keys, columns)
        jobs = [job for job in jobs if job[1] not in done]
        print(f"Resuming: {len(kept_rows)} good row(s) kept, "
              f"{len(jobs)} image(s) left to OCR")

    print(f"Found {len(jobs)} image(s) to process; methods: "
          f"{', '.join(METHODS[m][1] for m in method_keys)}; "
          f"workers: {workers}")

    first_col = METHODS[method_keys[0]][0]
    done_count, error_count = 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(kept_rows)
        f.flush()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_ocr_image, job, method_keys)
                       for job in jobs]
            try:
                for future in as_completed(futures):
                    image, row = future.result()
                    writer.writerow(row)
                    f.flush()
                    done_count += 1
                    if any(str(row[METHODS[m][0]]).startswith("ERROR:")
                           for m in method_keys):
                        error_count += 1
                    print(f"  [{done_count}/{len(jobs)}] {row['image']}: "
                          f"[{row['status']}] {row[first_col]}", flush=True)
            except KeyboardInterrupt:
                pool.shutdown(wait=False, cancel_futures=True)
                print(f"\nInterrupted -- {done_count + len(kept_rows)} row(s) "
                      f"safe in {args.output}. Re-run with --resume to "
                      f"continue.", flush=True)
                raise SystemExit(130)

    total = done_count + len(kept_rows)
    print(f"\nWrote {total} brick row(s) to {args.output} "
          f"({done_count} new, {len(kept_rows)} kept)")
    if error_count:
        print(f"  {error_count} image(s) still have ERROR reads -- "
              f"re-run with --resume to retry them")


if __name__ == "__main__":
    main()
