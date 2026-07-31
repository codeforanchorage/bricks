#!/usr/bin/env python3
"""Render each scanned-list row as a small strip image for human review.

The by-name brick list is a scan whose OCR mangles some rows beyond
repair (WOOFTER -> WOOFIER FAMIIX). Humans read degraded print far
better than any model -- so review pages show the reviewer the ACTUAL
printed row next to the brick photo, and the garbled transcription stops
mattering. This script renders every printed row to:

    <output>/<original brick number>.jpg     (~1400px wide, ~20-40 KB)

using the same single-row crop windows the strip re-reader uses
(resolve_tsp_rows.physical_rows), so the images match what the models
saw. Rows on the preamble page and rows whose number collides with
another physical row are skipped -- an ambiguous strip is worse than no
strip, and the review page hides missing images gracefully.

Rendering is local (no API calls) and resumable: existing outputs are
kept unless --force. Upload with the other derivatives:

    python make_strips.py --pdf "TSP Bricks ALL - OG List by Name - OCR.pdf" \
        --output derivatives/strips
    rsync -av derivatives/ <user>@<host>:bricks.<domain>/photos/

make_review_page.py --photo-base-url then references <base>/strips/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from resolve_tsp_rows import RENDER_SCALE, physical_rows
from strip_image import clean_strip

STRIP_WIDTH = 1400
JPEG_QUALITY = 80


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory for <brick number>.jpg strips")
    parser.add_argument("--force", action="store_true",
                        help="Re-render strips that already exist")
    args = parser.parse_args(argv)

    import pypdfium2 as pdfium
    from PIL import Image

    by_number: dict[str, list[dict]] = {}
    page0 = 0
    for entry in physical_rows(args.pdf):
        if entry.get("flag") == "page0":
            page0 += 1
            continue
        if entry["number"]:
            by_number.setdefault(entry["number"], []).append(entry)

    jobs = [(n, es[0]) for n, es in by_number.items() if len(es) == 1]
    collisions = len(by_number) - len(jobs)
    jobs.sort(key=lambda item: (item[1]["page"], -item[1]["crop"][0]))
    print(f"{len(jobs)} unique rows ({collisions} colliding numbers and "
          f"{page0} preamble rows skipped)")

    args.output.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(args.pdf))
    page_cache: tuple[int, object, float] | None = None
    written = kept = 0
    for number, entry in jobs:
        out = args.output / f"{number}.jpg"
        if out.is_file() and not args.force:
            kept += 1
            continue
        if page_cache is None or page_cache[0] != entry["page"]:
            page = pdf[entry["page"]]
            page_cache = (entry["page"],
                          page.render(scale=RENDER_SCALE).to_pil()
                          .convert("RGB"), page.get_size()[1])
        _, image, page_h = page_cache
        top, bottom = entry["crop"]
        y0 = max(0, int((page_h - top) * RENDER_SCALE))
        y1 = min(image.height, int((page_h - bottom) * RENDER_SCALE))
        strip = clean_strip(image, y0, y1)
        if strip.width > STRIP_WIDTH:
            scale = STRIP_WIDTH / strip.width
            strip = strip.resize((STRIP_WIDTH, max(1, round(strip.height * scale))),
                                 Image.LANCZOS)
        strip.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        written += 1
        if written % 1000 == 0:
            print(f"  {written} rendered", flush=True)

    print(f"\n{written} rendered, {kept} already existed -> {args.output}")


if __name__ == "__main__":
    main()
