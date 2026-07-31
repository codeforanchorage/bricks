#!/usr/bin/env python3
"""Resolve the OG list's disputed rows by re-reading each as an isolated strip.

Whole-page re-OCR (reocr_tsp_pdf.py) proved the two transcriptions of the
scanned OG list fail in complementary ways: the embedded-text parse has
correct row structure but noisy glyphs; the vision model reads glyphs cleanly
but occasionally slips the Line-2 column down a row. Rows where the two agree
(>= AGREE_SIM on folded text, same brick number) are settled. This script
settles the rest: every unconfirmed physical row is cropped out of the page
image as a single-row strip and transcribed alone -- one row per API call, so
there is nothing to misalign by construction.

Output is the resolved list, reference/tsp_brick_list_v2.csv, in the standard
schema plus two extra columns:

  verified  'agreed'  both transcriptions matched (re-OCR text adopted)
            'strip'   settled by an isolated-strip read
            'parse'   strip read failed; embedded-text parse retained
  flag      ''        clean
            'number'  strip read a different brick number than the parse
                      (strip wins -- it is the focused read -- parse id kept
                      in the flag for audit: 'number:<old>')

Requires the whole-page comparison to exist first:
    python reocr_tsp_pdf.py ... --output reference/tsp_reocr_full.csv
    python resolve_tsp_rows.py --pdf "TSP Bricks ALL - OG List by Name - OCR.pdf" \
        --reocr reference/tsp_reocr_full.csv --output reference/tsp_brick_list_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from consensus import _match_key, _normalise, _similar, scan_fold
from parse_tsp_list import _clean_brick_number, _page_rows_ex
from pipeline import _load_dotenv

MODEL = "gemini-2.5-flash"
RENDER_SCALE = 3.0
AGREE_SIM = 0.75       # folded-text similarity that settles a row
STRIP_PAD_PT = 5.0     # vertical padding around a row strip, PDF points
WORKERS = 8
# full_name keeps the embedded-text parse (structurally reliable, noisy
# glyphs); alt_name carries the vision re-read (clean glyphs, occasionally the
# wrong row). Neither replaces the other -- consumers match against BOTH and
# keep whichever scores better. Adopting the re-read as the only text was
# tried and lost 566 master-list joins: on the worst rows the model's noise
# is worse than the scan's, and it is not scan_fold-compatible.
COLUMNS = ["section", "assigned_id", "pos1", "pos2", "full_name", "key_word",
           "alt_name", "verified", "flag"]

STRIP_PROMPT = (
    "This is ONE row cropped from a scanned typewritten table of "
    "commemorative bricks. The columns, left to right: Last Name, First "
    "Name, Brick Number (a 1-5 digit integer), Inscription Line 1, "
    "Inscription Line 2. Some cells are empty. The print is degraded -- "
    "transcribe EXACTLY what is printed, including obvious typos. Return "
    'ONLY a JSON object: {"last": "...", "first": "...", "brick": "...", '
    '"line1": "...", "line2": "..."} with "" for empty cells.'
)


def _key(text: str) -> str:
    return scan_fold(_match_key(_normalise(text)))


def _confirmed_numbers(parse_csv: Path, reocr_csv: Path) -> set[str]:
    """Brick numbers whose two whole-document transcriptions agree."""
    with open(parse_csv, newline="", encoding="utf-8") as f:
        parsed = list(csv.DictReader(f))
    by_id: dict[str, list[str]] = {}
    for row in parsed:
        by_id.setdefault(row["assigned_id"], []).append(row["full_name"])
    with open(reocr_csv, newline="", encoding="utf-8") as f:
        reocr = list(csv.DictReader(f))

    confirmed = set()
    for row in reocr:
        number = row["assigned_id"]
        olds = by_id.get(number)
        if olds is None or len(olds) != 1:
            continue          # missing or colliding id -- never confirmed
        if _similar(_key(row["full_name"]), _key(olds[0])) >= AGREE_SIM:
            confirmed.add(number)
    return confirmed


def _reocr_text(reocr_csv: Path) -> dict[str, dict]:
    with open(reocr_csv, newline="", encoding="utf-8") as f:
        return {r["assigned_id"]: r for r in csv.DictReader(f)}


class _StripReader:
    """Thread-safe page renderer + single-strip transcriber."""

    def __init__(self, pdf_path: Path, model: str = MODEL):
        import pypdfium2 as pdfium
        from google import genai
        self._pdfium = pdfium
        self._pdf_path = pdf_path
        self._model = model
        self._client = genai.Client()
        self._pages: dict[int, object] = {}
        self._lock = threading.Lock()
        self.in_tokens = 0
        self.out_tokens = 0

    def _page_image(self, page_index: int):
        with self._lock:      # pdfium is not thread-safe; render serially
            if page_index not in self._pages:
                if len(self._pages) >= 12:   # strips arrive in page order;
                    self._pages.pop(next(iter(self._pages)))  # a small LRU is enough
                pdf = self._pdfium.PdfDocument(str(self._pdf_path))
                self._pages[page_index] = (
                    pdf[page_index].render(scale=RENDER_SCALE).to_pil()
                    .convert("RGB"), pdf[page_index].get_size()[1])
            return self._pages[page_index]

    def read(self, page_index: int, top: float, bottom: float) -> dict:
        from google.genai import types

        from strip_image import clean_strip
        image, page_h = self._page_image(page_index)
        # PDF y is bottom-up; PIL is top-down. The caller's crop window
        # already carries its neighbour-clamped padding; clean_strip then
        # deskews and re-cuts at the ink valleys so the model sees whole
        # characters and no neighbouring row.
        y0 = max(0, int((page_h - top) * RENDER_SCALE))
        y1 = min(image.height, int((page_h - bottom) * RENDER_SCALE))
        with self._lock:      # PIL ops on the shared cached page image
            strip = clean_strip(image, y0, y1)
        buf = io.BytesIO()
        strip.save(buf, "JPEG", quality=90)

        last_error = None
        for attempt in range(3):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[types.Part.from_bytes(data=buf.getvalue(),
                                                    mime_type="image/jpeg"),
                              STRIP_PROMPT])
                usage = response.usage_metadata
                with self._lock:
                    self.in_tokens += getattr(usage, "prompt_token_count", 0) or 0
                    self.out_tokens += (getattr(usage, "candidates_token_count", 0)
                                        or 0)
                text = (response.text or "").strip()
                fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
                if fence:
                    text = fence.group(1).strip()
                start, end = text.find("{"), text.rfind("}")
                return json.loads(text[start:end + 1])
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"strip p{page_index} failed: {last_error}")


def physical_rows(pdf_path: Path) -> list[dict]:
    """Enumerate the scanned list's printed rows with single-row crop windows.

    Each entry: {page, extent, crop, number, text, buyer} plus flag='page0'
    on the preamble page (differently laid out; strips crop wrongly there).
    Shared by the full resolve run and the targeted re-scan
    (rescan_rows.py) so both see identical row geometry.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    physical = []
    for page_index in range(len(pdf)):
        page = pdf[page_index]
        page_entries = []
        for columns, extent in _page_rows_ex(page.get_textpage(),
                                             page.get_size()[0]):
            last, first, brick, line1, line2 = columns
            number = _clean_brick_number(brick)
            text = " ".join(p for p in (line1, line2) if p).strip()
            if not number and len(re.sub(r"[^A-Za-z0-9]", "",
                                         "".join(columns))) < 8:
                continue      # speckle fragment
            folded = _normalise(text + " " + last + " " + first)
            if "line 1 line 2" in folded or "last name" in folded:
                continue      # a table header read as data
            page_entries.append({"page": page_index, "extent": extent,
                                 "number": number, "text": text,
                                 "buyer": " ".join(p for p in (last, first)
                                                   if p)})

        # A strip must contain exactly ONE printed row, or the model reads
        # the neighbour: clamp each crop at the midpoint to the rows
        # above/below (rows are ~11pt apart; a fixed pad spans two rows).
        for i, entry in enumerate(page_entries):
            top, bottom = entry["extent"]
            if i > 0:
                top = min(top + STRIP_PAD_PT,
                          (page_entries[i - 1]["extent"][1] + top) / 2)
            else:
                top += STRIP_PAD_PT
            if i + 1 < len(page_entries):
                bottom = max(bottom - STRIP_PAD_PT,
                             (page_entries[i + 1]["extent"][0] + bottom) / 2)
            else:
                bottom -= STRIP_PAD_PT
            entry["crop"] = (top, bottom)
            if page_index == 0:
                entry["flag"] = "page0"
            physical.append(entry)
    return physical


def _repair_numbers(rows: list[dict]) -> int:
    """Revert strip renumberings that collide with other rows.

    The strip read is the better transcription of *text*, but on worn digits
    it misreads brick numbers more often than the embedded OCR does -- and a
    misread usually lands on a number some other row legitimately owns.
    Accept a strip's renumbering only when the new number is claimed by no
    other row (it fills a gap); otherwise revert to the parse's number and
    downgrade the flag to 'number?:<strip>' for the review pile. Decisions
    use the pre-repair state so they are order-independent. Returns the
    number of rows reverted.
    """
    agreed = {r["assigned_id"] for r in rows if r["verified"] == "agreed"}
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["assigned_id"]] = counts.get(row["assigned_id"], 0) + 1

    reverted = 0
    for row in rows:
        parts = row["flag"].split(";")
        if not parts[0].startswith("number:"):
            continue
        old = parts[0].split(":", 1)[1]
        new = row["assigned_id"]
        plausible = new.isdigit() and 1 <= int(new) <= 13344
        if counts[new] > 1 or new in agreed or not plausible:
            row["assigned_id"] = old
            parts[0] = f"number?:{new}"
            row["flag"] = ";".join(parts)
            reverted += 1
    return reverted


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv(Path(__file__).with_name(".env"))

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--parse", type=Path,
                        default=Path("reference/tsp_brick_list.csv"))
    parser.add_argument("--reocr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap strip reads (0 = no cap) -- for a dry sample")
    args = parser.parse_args(argv)

    confirmed = _confirmed_numbers(args.parse, args.reocr)
    reocr = _reocr_text(args.reocr)
    print(f"Numbers settled by whole-document agreement: {len(confirmed)}")

    # A row is disputed unless its parsed number is confirmed. Rows with no
    # parsable number but real text are disputed too (the parser dropped
    # them; the 248 recovered numbers live here). Page 0 is a differently-
    # laid-out preamble -- neither transcription is reliable there, so it
    # stays with the parse for human review.
    physical = physical_rows(args.pdf)
    disputed = [entry for entry in physical
                if entry.get("flag") != "page0"
                and entry["number"] not in confirmed]

    print(f"Physical rows: {len(physical)}; disputed -> strip reads: "
          f"{len(disputed)}")
    if args.limit:
        disputed = disputed[:args.limit]
        print(f"  (capped at {args.limit})")

    reader = _StripReader(args.pdf)
    done = [0]

    def _work(entry):
        try:
            entry["strip"] = reader.read(entry["page"], *entry["crop"])
        except RuntimeError as exc:
            entry["strip"] = None
            print(f"  {exc}", flush=True)
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"  {done[0]}/{len(disputed)} strips", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(_work, disputed))

    rows, counts = [], {"agreed": 0, "strip": 0, "parse": 0, "renumbered": 0}
    for entry in physical:
        number, text, buyer = entry["number"], entry["text"], entry["buyer"]
        alt = ""
        verified, flag = "parse", entry.get("flag", "")
        if flag == "page0":
            pass                       # human review; keep the parse as-is
        elif number in confirmed:
            alt = reocr[number]["full_name"]
            verified = "agreed"
        elif entry.get("strip"):
            strip = entry["strip"]
            strip_number = _clean_brick_number(str(strip.get("brick", "")))
            strip_text = " ".join(str(strip.get(k, "")) for k in ("line1", "line2")).strip()
            if strip_number:
                if number and strip_number != number:
                    flag = f"number:{number}"
                    counts["renumbered"] += 1
                number = strip_number
            alt, verified = strip_text, "strip"
            if not text:               # parser dropped the row entirely
                text = strip_text
                buyer = " ".join(str(strip.get(k, ""))
                                 for k in ("last", "first")).strip()
        if not number or not (text or buyer):
            continue
        counts[verified] += 1
        rows.append({"section": "", "assigned_id": number, "pos1": "",
                     "pos2": "", "full_name": " ".join(text.split()),
                     "key_word": buyer, "alt_name": " ".join(alt.split()),
                     "verified": verified, "flag": flag})

    reverted = _repair_numbers(rows)
    counts["renumbered"] -= reverted

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    cost = reader.in_tokens / 1e6 * 0.30 + reader.out_tokens / 1e6 * 2.50
    print(f"\nWrote {len(rows)} rows to {args.output}")
    print(f"  agreed (two-source)   : {counts['agreed']}")
    print(f"  strip-resolved        : {counts['strip']}"
          f"  (incl. {counts['renumbered']} renumbered vs old parse)")
    print(f"  parse-only (unresolved): {counts['parse']}")
    print(f"Strip tokens: {reader.in_tokens} in / {reader.out_tokens} out "
          f"~= ${cost:.2f}")


if __name__ == "__main__":
    main()
