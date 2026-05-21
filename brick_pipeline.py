#!/usr/bin/env python3
"""Per-brick OCR pipeline: detect bricks -> crop each -> OCR individually.

For each input image:
  1. detect_bricks.detect()  -- find the paver rectangles
  2. crop each brick from the full-resolution image (with padding)
  3. OCR every crop with each method -- one brick per API call
  4. write one catalogue row per brick: each method's read plus an
     agreement status

Because every read is tied to a specific brick id, this removes the count
ambiguity, tile-seam duplication and fuzzy cross-method matching of the
whole-image pipeline. Per-brick crops also give each brick maximum resolution.

Crops are saved under <output-dir>/crops/ and a numbered overview under
<output-dir>/annotated/ so a reviewer can locate any brick id.

Usage:
    python brick_pipeline.py --input test_images/ --output output/bricks.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

import detect_bricks
import ocr_anthropic
import ocr_google
import ocr_paddle
import vision_ocr
from consensus import _normalise, _similar
from pipeline import METHODS, _load_dotenv, find_images, run_method

CROP_PAD = 18            # padding (work-image px) around each detected brick
AGREE_SIMILARITY = 0.80  # min text similarity for two methods to "agree"


def _ocr_crop(crop_path: Path, method_key: str) -> tuple[str, str]:
    """OCR one brick crop with one method -> (inscription, confidence)."""
    _csv, _label, provider, model = METHODS[method_key]

    if provider == "paddle":
        items, err = run_method(ocr_paddle.run, crop_path, False, 8)
        if err:
            return f"ERROR: {err}", "error"
        if not items:
            return "", "none"
        # PaddleOCR returns line detections -- join them into one inscription.
        text = " / ".join(d["inscription"] for d in items)
        score = min((d.get("raw_score") or 0.0) for d in items)
        conf = "high" if score >= 0.9 else "medium" if score >= 0.7 else "low"
        return text, conf

    # LLM providers: a single-brick crop yields a 0- or 1-element list.
    fn = ocr_anthropic.run if provider == "anthropic" else ocr_google.run
    items, err = run_method(fn, crop_path, model, vision_ocr.BRICK_PROMPT)
    if err:
        return f"ERROR: {err}", "error"
    if not items:
        return "", "none"
    return items[0]["inscription"], items[0]["confidence"]


def _status(reads: dict[str, str]) -> str:
    """Cross-method agreement for one brick (reads: method key -> inscription)."""
    texts = [t for t in reads.values() if t and not t.startswith("ERROR:")]
    if not texts:
        return "none"
    if len(texts) == 1:
        return "single"
    norms = [_normalise(t) for t in texts]
    pairs = [_similar(a, b)
             for i, a in enumerate(norms) for b in norms[i + 1:]]
    return "agreed" if pairs and max(pairs) >= AGREE_SIMILARITY else "conflict"


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
                        help="Directory of brick images (.jpg/.jpeg/.png)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Catalogue CSV to write")
    parser.add_argument("--methods", default="sonnet,gemini-pro,gemini-flash",
                        help="Comma-separated methods (default: the 3 LLMs)")
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

    crops_dir = args.output.parent / "crops"
    annotated_dir = args.output.parent / "annotated"
    for directory in (args.output.parent, crops_dir, annotated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    columns = (["image", "brick_id", "x", "y", "w", "h"]
               + [METHODS[m][0] for m in method_keys] + ["status"])
    print(f"Found {len(images)} image(s); methods: "
          f"{', '.join(METHODS[m][1] for m in method_keys)}")

    total = 0
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        f.flush()

        for image in images:
            orig, work, _grid, bricks = detect_bricks.detect(image)
            scale = orig.shape[1] / work.shape[1]
            img_h, img_w = orig.shape[:2]
            print(f"\n  {image.name}: detected {len(bricks)} brick(s)",
                  flush=True)

            annotated = work.copy()
            for brick_id, (x, y, bw, bh) in enumerate(bricks, 1):
                # Detector boxes are eroded/inset -- pad before cropping, then
                # scale from work coordinates to the full-resolution image.
                fx0 = max(0, int((x - CROP_PAD) * scale))
                fy0 = max(0, int((y - CROP_PAD) * scale))
                fx1 = min(img_w, int((x + bw + CROP_PAD) * scale))
                fy1 = min(img_h, int((y + bh + CROP_PAD) * scale))
                crop_path = crops_dir / f"{image.stem}_b{brick_id:02d}.jpg"
                cv2.imwrite(str(crop_path), orig[fy0:fy1, fx0:fx1])

                cv2.rectangle(annotated, (x, y), (x + bw, y + bh),
                              (0, 255, 0), 2)
                cv2.putText(annotated, str(brick_id), (x + 5, y + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                reads = {m: _ocr_crop(crop_path, m)[0] for m in method_keys}
                row = {"image": image.name, "brick_id": brick_id,
                       "x": fx0, "y": fy0, "w": fx1 - fx0, "h": fy1 - fy0,
                       "status": _status(reads)}
                for m in method_keys:
                    row[METHODS[m][0]] = reads[m].replace("\n", " / ")
                writer.writerow(row)
                f.flush()
                total += 1
                print(f"    brick {brick_id}/{len(bricks)}: {row['status']}",
                      flush=True)

            cv2.imwrite(str(annotated_dir / f"{image.stem}.jpg"), annotated)

    print(f"\nWrote {total} brick row(s) to {args.output}")
    print(f"Crops: {crops_dir}    Overview: {annotated_dir}")


if __name__ == "__main__":
    main()
