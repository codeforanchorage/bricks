#!/usr/bin/env python3
"""Recover scanned-list row images for master rows that have none.

make_strips.py names each strip after the brick number the PDF text layer
gives its row, and skips rows it cannot name safely -- so a master row
ends up with no derivatives/strips/<orig_id>.jpg when:

  * two printed rows carry the same parsed number (a row printed twice,
    or a 5-digit number whose leading digits the parse dropped: 13215
    parsed as 325 collides with the real 325);
  * the row was renumbered after the parse (the strip sits under the old
    parsed number, e.g. 032.jpg for brick 11032);
  * the row is on the preamble page, which make_strips skips;
  * the text layer is offset from the print, so the crop showed a
    neighbour and refile_strips.py set the image aside.

For every such row this script gathers CANDIDATE row images and keeps one
only when a model transcription of the image itself proves it is the
right row -- the printed number equals the brick's original number and
the text resembles the row (or, when the master's number is itself in
doubt, the text alone matches closely and the printed number belongs to
no other brick). Candidates, cheapest first:

  1. the text layer's own crop for the row (by number, else by text);
  2. every ink row found in the page image within a few rows of that
     position -- the print decides the row edges, so a layer offset no
     longer matters.

Additive: an existing strip is never overwritten. Resumable: model reads
are cached in the log. Rows whose original number is shared by two
DIFFERENT inscriptions are skipped (one filename cannot serve both).

    python recover_strips.py --pdf "TSP Bricks ALL - OG List by Name - OCR.pdf"
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from consensus import _normalise, _similar
from pipeline import _load_dotenv

MODEL = "gemini-3.1-flash-lite"
STRIP_WIDTH = 1400
JPEG_QUALITY = 80
SEARCH_ROWS = 4          # ink rows searched each side of the expected row
TEXT_WITH_NUMBER = 0.55  # text similarity needed when the number matches
TEXT_ALONE = 0.85        # ... when only the text can vouch for the row


def _jpeg(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def _fit(strip):
    from PIL import Image
    if strip.width > STRIP_WIDTH:
        scale = STRIP_WIDTH / strip.width
        strip = strip.resize((STRIP_WIDTH,
                              max(1, round(strip.height * scale))),
                             Image.LANCZOS)
    return strip


def _ink_rows(band, pitch: int) -> list[tuple[int, int]]:
    """(top, bottom) pixel spans of the printed rows in a deskewed band.

    Ink bands from the horizontal profile; a band much taller than one
    row is several fused rows, split at the weakest profile minima.
    """
    from strip_image import _profile
    prof = _profile(band)
    if not prof.any():
        return []
    smooth = np.convolve(prof, np.ones(3) / 3, mode="same")
    ink = smooth > max(2.0, 0.03 * float(smooth.max()))
    spans, start = [], None
    for i, on in enumerate(ink):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(ink)))
    rows = []
    for top, bottom in spans:
        n = max(1, round((bottom - top) / pitch))
        if n == 1 or bottom - top < 1.55 * pitch:
            rows.append((top, bottom))
            continue
        cuts = [top]
        for k in range(1, n):
            centre = top + k * (bottom - top) / n
            lo, hi = int(centre - 0.3 * pitch), int(centre + 0.3 * pitch)
            cuts.append(lo + int(np.argmin(prof[lo:hi])))
        cuts.append(bottom)
        rows += list(zip(cuts, cuts[1:]))
    return [(t, b) for t, b in rows if b - t >= 0.35 * pitch]


def _digits(text: str) -> list[str]:
    return re.findall(r"\d{1,5}", text)


def _verdict(transcript: str, row: dict, other_ids: set[str]) -> str:
    """'' when the image is not this row, else why it was accepted."""
    if not transcript or transcript.startswith(("ERROR", "NONE")):
        return ""
    target = row["orig_id"]
    number_ok = any(d.isdigit() and int(d) == int(target)
                    for d in _digits(transcript)) if target.isdigit() else False
    words = _normalise(re.sub(r"\d+", " ", transcript))
    best = 0.0
    for text in (row["og_alt"], row["new_inscription"], row["og_inscription"]):
        if text:
            want = _normalise(re.sub(r"\d+", " ", row["buyer"] + " " + text))
            best = max(best, _similar(words, want))
    if number_ok and best >= TEXT_WITH_NUMBER:
        return f"number+text {best:.2f}"
    printed = {str(int(d)) for d in _digits(transcript)[:1]}
    if best >= TEXT_ALONE and not (printed & other_ids):
        return f"text {best:.2f} (printed {'/'.join(sorted(printed)) or '-'})"
    return ""


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv(Path(__file__).with_name(".env"))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", required=True, type=Path)
    p.add_argument("--master", type=Path,
                   default=Path("reference/master_list.csv"))
    p.add_argument("--strips", type=Path, default=Path("derivatives/strips"))
    p.add_argument("--log", type=Path,
                   default=Path("output/strip_recovery.csv"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)

    import pypdfium2 as pdfium
    from audit_strips import read_number
    from resolve_tsp_rows import RENDER_SCALE, physical_rows
    from strip_image import _deskew, clean_strip

    with open(a.master, newline="", encoding="utf-8") as f:
        master = list(csv.DictReader(f))
    by_orig = defaultdict(list)
    for row in master:
        by_orig[row["orig_id"]].append(row)
    all_ids = {str(int(r["orig_id"])) for r in master if r["orig_id"].isdigit()}

    targets, twins = [], 0
    for orig, rows in by_orig.items():
        if not orig or (a.strips / f"{orig}.jpg").is_file():
            continue
        texts = {_normalise(r["og_inscription"]) for r in rows}
        if len(texts) > 1 and min(_similar(x, y) for x in texts
                                  for y in texts) < 0.8:
            twins += 1          # one number, two different bricks
            continue
        targets.append(rows[0])
    if a.limit:
        targets = targets[:a.limit]
    print(f"{len(targets)} master row(s) without a strip "
          f"({twins} skipped: number shared by different inscriptions)")

    physical = physical_rows(a.pdf)
    by_number, by_text = defaultdict(list), defaultdict(list)
    for entry in physical:
        by_number[entry["number"]].append(entry)
        by_text[_normalise(entry["text"])].append(entry)

    def anchors(row: dict) -> list[dict]:
        key = _normalise(row["og_inscription"])
        # Number first (it names the row); text only when it is specific
        # -- a blank or boilerplate text matches rows all over the list.
        found = list(by_number.get(row["orig_id"], []))
        if key and len(by_text.get(key, [])) <= 4:
            found += [e for e in by_text[key] if e not in found]
        for part in row["flag"].split(";"):      # parsed-number hints
            if part.startswith(("number:", "number?:")):
                found += [e for e in by_number.get(part.split(":", 1)[1], [])
                          if e not in found]
        return found[:6]

    cache: dict[str, str] = {}
    if a.log.is_file():
        with open(a.log, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r["transcript"].startswith("ERROR"):
                    cache[r["candidate"]] = r["transcript"]
    a.log.parent.mkdir(parents=True, exist_ok=True)
    new_log = not a.log.is_file()
    log_f = open(a.log, "a", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    if new_log:
        log_w.writerow(["orig_id", "candidate", "transcript", "verdict"])

    pdf = pdfium.PdfDocument(str(a.pdf))
    pages: dict[int, tuple] = {}

    def page_image(index: int):
        if index not in pages:
            if len(pages) > 6:
                pages.pop(next(iter(pages)))
            page = pdf[index]
            pages[index] = (page.render(scale=RENDER_SCALE).to_pil()
                            .convert("RGB"), page.get_size()[1])
        return pages[index]

    tmp = a.strips.parent / "_recover_tmp"
    tmp.mkdir(exist_ok=True)

    def read(candidate: str, image) -> str:
        if candidate not in cache:
            path = tmp / (re.sub(r"[^\w.-]", "_", candidate) + ".jpg")
            path.write_bytes(_jpeg(image))
            cache[candidate] = read_number(path, MODEL, full=True)
            path.unlink(missing_ok=True)
        return cache[candidate]

    def recover(row: dict) -> tuple[str, str]:
        orig = row["orig_id"]
        others = all_ids - ({str(int(orig))} if orig.isdigit() else set())
        found = anchors(row)
        # 1. the text layer's own crop
        for entry in found:
            image, page_h = page_image(entry["page"])
            top, bottom = entry["crop"]
            y0 = max(0, int((page_h - top) * RENDER_SCALE))
            y1 = min(image.height, int((page_h - bottom) * RENDER_SCALE))
            strip = _fit(clean_strip(image, y0, y1))
            name = f"layer:p{entry['page']}:{y0}-{y1}"
            text = read(name, strip)
            why = _verdict(text, row, others)
            log_w.writerow([orig, name, text, why])
            if why:
                return _save(orig, strip), f"layer crop | {why}"
        # 2. the page's own ink rows around that position
        for entry in found:
            image, page_h = page_image(entry["page"])
            top, bottom = entry["crop"]
            pitch = max(8, int((top - bottom) * RENDER_SCALE))
            centre = int((page_h - (top + bottom) / 2) * RENDER_SCALE)
            ey0 = max(0, centre - (SEARCH_ROWS + 1) * pitch)
            ey1 = min(image.height, centre + (SEARCH_ROWS + 1) * pitch)
            band = _deskew(image.crop((0, ey0, image.width, ey1)))
            rows = sorted(_ink_rows(band, pitch),
                          key=lambda s: abs((s[0] + s[1]) / 2 - (centre - ey0)))
            for t, b in rows:
                strip = _fit(band.crop((0, max(0, t - 3), band.width,
                                        min(band.height, b + 3))))
                name = f"ink:p{entry['page']}:{ey0 + t}-{ey0 + b}"
                text = read(name, strip)
                why = _verdict(text, row, others)
                log_w.writerow([orig, name, text, why])
                if why:
                    return _save(orig, strip), f"ink row | {why}"
        return "", "no anchor in the PDF" if not found else "no row verified"

    def _save(orig: str, strip) -> str:
        out = a.strips / f"{orig}.jpg"
        if not out.exists():
            out.write_bytes(_jpeg(strip))
        return out.name

    results = []
    # Pages render serially (pdfium is not thread-safe); the model calls
    # inside read() are what the pool parallelises, one target per worker.
    import threading
    lock = threading.Lock()
    _page_image = page_image

    def page_image(index: int):                       # noqa: F811
        with lock:
            return _page_image(index)

    with ThreadPoolExecutor(a.workers) as pool:
        for i, (row, (name, how)) in enumerate(
                zip(targets, pool.map(recover, targets)), 1):
            results.append((row, name, how))
            log_f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(targets)}", flush=True)
    log_f.close()
    try:
        tmp.rmdir()
    except OSError:
        pass

    done = [r for r in results if r[1]]
    summary = a.log.with_name(a.log.stem + "_summary.csv")
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["orig_id", "section", "buyer", "inscription", "strip",
                    "how"])
        for row, name, how in results:
            w.writerow([row["orig_id"], row["section"], row["buyer"],
                        row["new_inscription"] or row["og_alt"]
                        or row["og_inscription"], name, how])
    print(f"\nrecovered {len(done)}/{len(results)} strip(s) -> {a.strips}")
    reasons = defaultdict(int)
    for _, name, how in results:
        reasons[how.split(" | ")[0] if name else how] += 1
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {reason}")
    print(f"details: {summary}")


if __name__ == "__main__":
    main()
