#!/usr/bin/env python3
"""Brick OCR test pipeline.

Processes commemorative paver-brick photos through four text-extraction
methods and writes a comparison:

  Method 1  PaddleOCR          local OCR engine (GPU if available, CPU fallback)
  Method 2  Claude Sonnet 4.6  vision LLM via the Anthropic API
  Method 3  Gemini 2.5 Pro     vision LLM via the Google Gemini API
  Method 4  Gemini 2.5 Flash   vision LLM via the Google Gemini API

This is the test harness for cataloguing ~13,000 bricks removed from Town
Square Park. It starts from nadir (downward-facing) JPEG crops and produces a
CSV -- the seed of what becomes the full Excel catalog.

Usage:
    python pipeline.py --input test_images/ --output output/results.csv

Options:
    --methods a,b,c,...             methods to run (default: the 3 LLM methods)
    --tiles RxC                     split each image into an overlapping grid
    --tile-overlap FLOAT            tile overlap fraction (default: 0.2)
    --no-gpu                        force PaddleOCR onto the CPU
    --cpu-threads N                 PaddleOCR CPU inference threads (default: 8)
    --no-group                      show raw PaddleOCR lines instead of bricks
    --brick-gap FLOAT               tune how aggressively lines group into bricks
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

import compare
import group_bricks
import ocr_anthropic
import ocr_google
import ocr_paddle
import tiling

# .heic needs pillow-heif (see vision_ocr.load_jpeg_bytes) -- iPhone default.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}
CSV_COLUMNS = ["filename", "method", "brick_inscription", "confidence"]

# Method key -> (CSV name, display label, provider, model id), canonical order.
METHODS = {
    "paddle":       ("paddleocr",     "PaddleOCR (local)",  "paddle", None),
    "sonnet":       ("claude-sonnet", "Claude Sonnet 4.6",
                     "anthropic", "claude-sonnet-4-6"),
    "sonnet-5":     ("claude-sonnet-5", "Claude Sonnet 5",
                     "anthropic", "claude-sonnet-5"),
    "gemini-pro":   ("gemini-pro",    "Gemini 2.5 Pro",
                     "google", "gemini-2.5-pro"),
    "gemini-flash": ("gemini-flash",  "Gemini 2.5 Flash",
                     "google", "gemini-2.5-flash"),
    # gemini-2.5-flash is deprecated by Google (shutdown 2026-10-16); these
    # are the candidate successors, validated against the labeled photo set.
    # (gemini-2.5-flash-lite returns 404 "no longer available to new users".)
    "gemini-flash-3":    ("gemini-flash-3", "Gemini 3 Flash Preview",
                          "google", "gemini-3-flash-preview"),
    "gemini-lite-31":    ("gemini-lite-31", "Gemini 3.1 Flash-Lite",
                          "google", "gemini-3.1-flash-lite"),
    "gemini-lite-35":    ("gemini-lite-35", "Gemini 3.5 Flash-Lite",
                          "google", "gemini-3.5-flash-lite"),
}


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a local .env file into os.environ.

    Existing environment variables are left untouched. This lets the API keys
    live in brick-ocr/.env -- kept out of shell history, out of the terminal,
    and separate from Claude Code's own ANTHROPIC_API_KEY auth.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def find_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Return every supported image file in `input_dir`, sorted by path.

    With recursive=True subdirectories are walked too (the warehouse photo
    convention nests photos in section/pallet folders, e.g. H/pallet-12/).
    """
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    walk = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        (p for p in walk
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.relative_to(input_dir).as_posix(),
    )


def run_method(method_fn, *args) -> tuple[list[dict], str | None]:
    """Run one OCR method, returning (items, error_message).

    A failure in one method (missing key, GPU error, parse error) must not
    abort the run -- the other methods and remaining images still go ahead.
    """
    try:
        return method_fn(*args), None
    except Exception as exc:  # noqa: BLE001 -- surface any failure per method
        msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        return [], (msg[:200] + " ...") if len(msg) > 200 else msg


def rows_for(filename: str, method: str, items: list[dict],
             error: str | None) -> list[list[str]]:
    """Build CSV rows for one image/method, always emitting at least one row."""
    if error:
        return [[filename, method, f"ERROR: {error}", "error"]]
    if not items:
        return [[filename, method, "", "none"]]
    rows = []
    for item in items:
        # Collapse multi-line inscriptions to one CSV cell for a clean
        # CSV -> Excel hand-off later.
        inscription = item["inscription"].replace("\n", " / ").strip()
        rows.append([filename, method, inscription, item["confidence"]])
    return rows


def _parse_tiles(spec: str) -> tuple[int, int]:
    """Parse a --tiles value like '3x3', '2x4', or '3' into (rows, cols)."""
    text = spec.strip().lower()
    try:
        if "x" in text:
            rows_s, _, cols_s = text.partition("x")
            rows, cols = int(rows_s), int(cols_s)
        else:
            rows = cols = int(text)
    except ValueError:
        raise SystemExit(f"--tiles must look like '3x3' or '3', got {spec!r}")
    if rows < 1 or cols < 1:
        raise SystemExit(f"--tiles values must be >= 1, got {spec!r}")
    return rows, cols


def _ocr_one(path, method_keys, use_gpu, cpu_threads) -> dict:
    """Run each requested method on one image file (a whole image or a tile).

    Returns {method_key: (items, error)}. Prints each method's wall-clock time.
    """
    results = {}
    for key in method_keys:
        _csv, label, provider, model = METHODS[key]
        start = time.perf_counter()
        if provider == "paddle":
            results[key] = run_method(ocr_paddle.run, path, use_gpu, cpu_threads)
        elif provider == "anthropic":
            results[key] = run_method(ocr_anthropic.run, path, model)
        elif provider == "google":
            results[key] = run_method(ocr_google.run, path, model)
        print(f"    {label}: {time.perf_counter() - start:.1f}s", flush=True)
    return results


def process_image(image, method_keys, args, tile_rows, tile_cols) -> dict:
    """OCR one image -- whole, or split into an overlapping tile grid.

    Returns {method_key: (items, error)}. PaddleOCR boxes are in full-image
    coordinates; tiling duplicates are removed.
    """
    if tile_rows * tile_cols <= 1:
        return _ocr_one(image, method_keys, not args.no_gpu, args.cpu_threads)

    combined = {m: ([], None) for m in method_keys}
    with tempfile.TemporaryDirectory(prefix="brick_tiles_") as tmp:
        tiles, img_size = tiling.split(image, tile_rows, tile_cols,
                                       args.tile_overlap, Path(tmp))
        for i, (tile_path, x0, y0, x1, y1) in enumerate(tiles, 1):
            print(f"    tile {i}/{len(tiles)} ...", flush=True)
            tile_res = _ocr_one(tile_path, method_keys,
                                not args.no_gpu, args.cpu_threads)
            for m, (items, err) in tile_res.items():
                if m == "paddle":
                    # Drop detections sliced by an interior cut, then shift the
                    # survivors' boxes into full-image coordinates.
                    items = tiling.drop_clipped(items, (x0, y0, x1, y1), img_size)
                    for det in items:
                        box = det.get("box")
                        if box is not None:
                            det["box"] = (box[0] + x0, box[1] + y0,
                                          box[2] + x0, box[3] + y0)
                prev_items, prev_err = combined[m]
                combined[m] = (prev_items + items, prev_err or err)

    # Overlapping tiles detect boundary content twice -- de-duplicate.
    for m, (items, err) in combined.items():
        if m == "paddle":
            items = tiling.dedup_detections(items)
        else:
            items = tiling.dedup_inscriptions(items)
        if items:  # a tile error only matters if nothing came through at all
            err = None
        combined[m] = (items, err)
    return combined


def main(argv=None) -> None:
    # Brick inscriptions can include characters outside Windows' cp1252 console
    # encoding (e.g. a heart symbol); force UTF-8 so printing never crashes.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    _load_dotenv(Path(__file__).with_name(".env"))

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Directory of brick images (.jpg/.jpeg/.png)")
    parser.add_argument("--output", required=True, type=Path,
                        help="CSV file to write comparison results to")
    parser.add_argument(
        "--methods", default="sonnet,gemini-pro,gemini-flash",
        help="Comma-separated methods: paddle, sonnet, gemini-pro, "
             "gemini-flash (default omits paddle -- add it for an OCR "
             "cross-check)")
    parser.add_argument("--tiles", default="1x1",
                        help="Split each image into an RxC grid of overlapping "
                             "tiles before OCR, e.g. 3x3 (default: 1x1, no "
                             "tiling). Raises effective resolution per brick; "
                             "for the LLM methods it means RxC API calls/image.")
    parser.add_argument("--tile-overlap", type=float, default=0.2,
                        help="Tile overlap as a fraction of tile size "
                             "(default: 0.2). Must exceed half a brick's size "
                             "so boundary bricks stay whole within one tile.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force PaddleOCR to run on the CPU")
    parser.add_argument("--cpu-threads", type=int, default=8,
                        help="PaddleOCR CPU inference threads (default: 8). "
                             "For the eventual thousands-of-images batch, "
                             "parallel worker processes scale better than "
                             "piling more threads onto a single image.")
    parser.add_argument("--no-group", action="store_true",
                        help="Do not group PaddleOCR detections into bricks; "
                             "report each raw line/word detection separately")
    parser.add_argument("--brick-gap", type=float,
                        default=group_bricks.BRICK_STACK_GAP,
                        help="Vertical-gap threshold for stacking lines into a "
                             "brick, as a multiple of text height "
                             f"(default: {group_bricks.BRICK_STACK_GAP}). "
                             "Lower it if separate bricks get merged.")
    args = parser.parse_args(argv)

    requested = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in requested if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown)}. "
                         f"Valid: {', '.join(METHODS)}")
    method_keys = [m for m in METHODS if m in requested]
    if not method_keys:
        raise SystemExit(
            "No methods selected. Use --methods sonnet,gemini-pro,gemini-flash")

    tile_rows, tile_cols = _parse_tiles(args.tiles)

    images = find_images(args.input)
    if not images:
        raise SystemExit(
            f"No images (.jpg/.jpeg/.png) found in {args.input}\n"
            f"Drop the test JPEGs into that folder and re-run."
        )

    print(f"Found {len(images)} image(s) in {args.input}")
    print(f"Methods: {', '.join(METHODS[m][1] for m in method_keys)}")
    if tile_rows * tile_cols > 1:
        n_tiles = tile_rows * tile_cols
        n_llm = sum(1 for m in method_keys if m != "paddle")
        note = f" ({n_tiles * n_llm} LLM API calls/image)" if n_llm else ""
        print(f"Tiling: {tile_rows}x{tile_cols} grid -- {n_tiles} tiles/image "
              f"at {args.tile_overlap:.0%} overlap{note}")

    paddle_unit = "line" if args.no_group else "brick"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    # Open the CSV up front and append each image's rows as it finishes, with a
    # flush, so an interrupted run keeps every image completed so far.
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        f.flush()

        for image in images:
            print(f"\n  processing {image.name} ...", flush=True)
            results = process_image(image, method_keys, args,
                                    tile_rows, tile_cols)

            # Group PaddleOCR line detections into per-brick inscriptions so
            # the comparison with the LLM methods is one row per brick.
            if "paddle" in results:
                dets, p_err = results["paddle"]
                if dets and not args.no_group:
                    bricks = group_bricks.group(dets, brick_gap=args.brick_gap)
                    print(f"    grouped {len(dets)} detection(s) into "
                          f"{len(bricks)} brick(s)", flush=True)
                    over = sum(1 for b in bricks if b.get("n_lines", 1) > 3)
                    if over:
                        print(f"    note: {over} brick(s) exceed 3 lines -- "
                              f"possible over-merge; try a lower --brick-gap "
                              f"(current {args.brick_gap})", flush=True)
                    results["paddle"] = (bricks, p_err)

            comparison = []
            for m in method_keys:
                items, err = results[m]
                unit = paddle_unit if m == "paddle" else "brick"
                comparison.append((METHODS[m][1], items, err, unit))
            compare.print_comparison(image.name, comparison)

            image_rows = []
            for m in method_keys:
                items, err = results[m]
                image_rows += rows_for(image.name, METHODS[m][0], items, err)
            writer.writerows(image_rows)
            f.flush()
            total_rows += len(image_rows)
            print(f"    +{len(image_rows)} row(s) saved to CSV "
                  f"({total_rows} total)", flush=True)

    print(f"\nDone. Wrote {total_rows} row(s) to {args.output}")


if __name__ == "__main__":
    main()
