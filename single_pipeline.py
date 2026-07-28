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
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from brick_pipeline import _ocr_crop, _status
from pipeline import METHODS, _load_dotenv, find_images


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
                        help="Directory of single-brick photos (.jpg/.jpeg/.png)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Catalogue CSV to write")
    parser.add_argument("--methods", default="gemini-flash",
                        help="Comma-separated methods (default: gemini-flash)")
    args = parser.parse_args(argv)

    requested = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in requested if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown)}. "
                         f"Valid: {', '.join(METHODS)}")
    method_keys = [m for m in METHODS if m in requested]
    if not method_keys:
        raise SystemExit("No methods selected.")

    images = find_images(args.input)
    if not images:
        raise SystemExit(f"No images (.jpg/.jpeg/.png) found in {args.input}")

    # x/y/w/h are kept for format compatibility; the brick is the whole frame.
    columns = (["image", "brick_id", "x", "y", "w", "h"]
               + [METHODS[m][0] for m in method_keys] + ["status"])
    print(f"Found {len(images)} image(s); methods: "
          f"{', '.join(METHODS[m][1] for m in method_keys)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        f.flush()

        for image in images:
            reads = {m: _ocr_crop(image, m)[0] for m in method_keys}
            row = {"image": image.name, "brick_id": 1,
                   "x": "", "y": "", "w": "", "h": "",
                   "status": _status(reads)}
            for m in method_keys:
                row[METHODS[m][0]] = reads[m].replace("\n", " / ")
            writer.writerow(row)
            f.flush()
            first = row[METHODS[method_keys[0]][0]]
            print(f"  {image.name}: [{row['status']}] {first}", flush=True)

    print(f"\nWrote {len(images)} brick row(s) to {args.output}")


if __name__ == "__main__":
    main()
