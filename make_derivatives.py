#!/usr/bin/env python3
"""Generate web-sized photo derivatives for Dreamhost hosting.

Walks the photo tree and writes two mirrored JPEG trees:

    <output>/thumbs/<same relative path>.jpg   (~640 px wide, for lists)
    <output>/zoom/<same relative path>.jpg     (~2500 px wide, for squinting
                                                at worn bricks -- OCR works
                                                at 1568 px, so 2500 px reads
                                                engraving comfortably)

Camera originals are 40-100 GB and HEIC does not display in browsers;
the derivative trees are ~8-15 GB and upload in one rsync. Every output
is EXIF-upright .jpg regardless of source format (.heic included), and
the relative paths mirror the originals -- so review pages built with
make_review_page.py --photo-base-url resolve them directly.

The run is resumable: an output that already exists and is newer than
its source is skipped, so re-running after adding a pallet only does
the new photos. Sizing is CPU-bound; --workers threads help because
Pillow's JPEG codec releases the GIL.

Usage:
    python make_derivatives.py --input photos/ --output derivatives/
    rsync -av derivatives/ user@host:bricks.example.com/photos/
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hostpaths import derivative_rel
from pipeline import find_images

THUMB_WIDTH = 640
ZOOM_WIDTH = 2500
THUMB_QUALITY = 78
ZOOM_QUALITY = 84


def _derivative_paths(rel: Path, output: Path) -> tuple[Path, Path]:
    """(thumb path, zoom path) for one source photo's relative path."""
    jpg_rel = derivative_rel(rel.as_posix())
    return output / "thumbs" / jpg_rel, output / "zoom" / jpg_rel


def _make_one(image: Path, rel: Path, output: Path,
              force: bool) -> tuple[str, str | None]:
    """Write both derivatives for one photo. Returns (status, error)."""
    from PIL import Image, ImageOps

    thumb_path, zoom_path = _derivative_paths(rel, output)
    if not force:
        try:
            src_mtime = image.stat().st_mtime
            if (thumb_path.stat().st_mtime >= src_mtime
                    and zoom_path.stat().st_mtime >= src_mtime):
                return "skipped", None
        except OSError:
            pass  # one or both outputs missing -> build them

    try:
        if image.suffix.lower() == ".heic":
            import pillow_heif
            pillow_heif.register_heif_opener()
        with Image.open(image) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            for path, width, quality in (
                    (zoom_path, ZOOM_WIDTH, ZOOM_QUALITY),
                    (thumb_path, THUMB_WIDTH, THUMB_QUALITY)):
                path.parent.mkdir(parents=True, exist_ok=True)
                out = img
                if img.width > width:
                    scale = width / img.width
                    out = img.resize((width, round(img.height * scale)),
                                     Image.LANCZOS)
                out.save(path, format="JPEG", quality=quality,
                         optimize=True)
        return "done", None
    except Exception as exc:  # noqa: BLE001 -- report, keep the batch going
        return "error", f"{rel.as_posix()}: {exc}"


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Photo root (the same one the pipeline uses)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Derivatives root; thumbs/ and zoom/ go inside")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent resize workers (default: 4)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even when outputs are up to date")
    args = parser.parse_args(argv)

    images = find_images(args.input, recursive=True)
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    counts = {"done": 0, "skipped": 0, "error": 0}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(_make_one, image, image.relative_to(args.input),
                        args.output, args.force)
            for image in images
        ]
        for i, future in enumerate(as_completed(futures), 1):
            status, error = future.result()
            counts[status] += 1
            if error:
                errors.append(error)
            if i % 250 == 0 or i == len(images):
                print(f"  [{i}/{len(images)}] "
                      f"{counts['done']} built, {counts['skipped']} current, "
                      f"{counts['error']} errors", flush=True)

    print(f"\n{counts['done']} built, {counts['skipped']} already current, "
          f"{counts['error']} failed -> {args.output}")
    for error in errors[:10]:
        print(f"  ERROR {error}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")
    print(f"Upload:  rsync -av {args.output}/ "
          f"<user>@<host>:bricks.<domain>/photos/")


if __name__ == "__main__":
    main()
