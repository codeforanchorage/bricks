#!/usr/bin/env python3
"""Re-OCR pages of the scanned OG brick-list PDF with a vision LLM.

The OG PDF's embedded text layer is 1990s-era OCR of a degraded typewriter
printout -- systematically noisy inscriptions and, worse, misread brick
numbers (~235 id collisions, which corrupt the E/F/G section-by-number rule).
A modern vision model reading the page *images* produces a second, largely
independent transcription. Where the two agree, the data is trustworthy;
where they differ, the row lands in a review file instead of silently
carrying a wrong number.

This script renders pages, sends each to Gemini Flash, and writes one CSV per
run in the same shape as parse_tsp_list.py's output, plus a comparison report
against that parse. Run it on a few --pages first (the default) to judge
quality and cost before committing to all 174.

Usage:
    python reocr_tsp_pdf.py --pdf "TSP Bricks ALL - OG List by Name - OCR.pdf" \
        --pages 27,81,160 --output reference/tsp_reocr_sample.csv \
        --compare reference/tsp_brick_list.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

from consensus import _match_key, _normalise, _similar, scan_fold
from pipeline import _load_dotenv

MODEL = "gemini-2.5-flash"
RENDER_SCALE = 3.0          # 612pt page -> ~1836px wide; table text ~25px tall

PROMPT = (
    "This is a scanned page of a typewritten table listing commemorative "
    "bricks. The columns are: Last Name, First Name, Brick Number, "
    "Inscription Line 1, Inscription Line 2. Some cells are empty; some rows "
    "have only a brick number and inscription. The print is degraded -- "
    "transcribe EXACTLY what is printed, including obvious typos, without "
    "correcting anything. Brick numbers are 1-5 digit integers.\n\n"
    "Return a JSON array with one object per data row, in page order:\n"
    '{"last": "...", "first": "...", "brick": "...", "line1": "...", '
    '"line2": "..."}\n'
    "Use \"\" for empty cells. Skip page headers. Return ONLY the JSON array."
)

COLUMNS = ["section", "assigned_id", "pos1", "pos2", "full_name", "key_word"]


def _render_page(pdf, index: int) -> bytes:
    """Render one PDF page to JPEG bytes at full table-reading resolution."""
    bitmap = pdf[index].render(scale=RENDER_SCALE)
    image = bitmap.to_pil().convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _transcribe(client, types, image_bytes: bytes) -> tuple[list[dict], object]:
    """One page -> parsed row dicts + usage metadata (for cost reporting)."""
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                  PROMPT],
    )
    text = (response.text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise RuntimeError(f"No JSON array in reply: {text[:200]}")
    rows = json.loads(text[start:end + 1])
    return rows, response.usage_metadata


def _compare(reocr: list[dict], parsed_csv: Path, pages: list[int]) -> None:
    """Report agreement between the re-OCR and the embedded-text parse.

    The embedded parse has no page column, so comparison keys on the brick
    number: both transcriptions of a row should agree on it; a number seen by
    only one side is exactly the suspect class we are hunting.
    """
    with open(parsed_csv, newline="", encoding="utf-8") as f:
        parsed = {r["assigned_id"]: r for r in csv.DictReader(f)}

    matched = num_only_reocr = 0
    sims = []
    disagreements = []
    for row in reocr:
        number = re.sub(r"[^0-9]", "", str(row.get("brick", "")))
        if not number:
            continue
        text_re = " ".join(str(row.get(k, "")) for k in ("line1", "line2")).strip()
        old = parsed.get(number)
        if old is None:
            num_only_reocr += 1
            disagreements.append((number, text_re, "(number not in old parse)"))
            continue
        matched += 1
        key_a = scan_fold(_match_key(_normalise(text_re)))
        key_b = scan_fold(_match_key(_normalise(old["full_name"])))
        sim = _similar(key_a, key_b) if key_a and key_b else 0.0
        sims.append(sim)
        if sim < 0.75:
            disagreements.append((number, text_re, old["full_name"]))

    print(f"\n=== comparison vs embedded-text parse (pages {pages}) ===")
    print(f"  re-OCR rows with a number : {matched + num_only_reocr}")
    print(f"  number found in old parse : {matched}")
    print(f"  number NOT in old parse   : {num_only_reocr}  <- suspect ids")
    if sims:
        agree = sum(1 for s in sims if s >= 0.75)
        print(f"  inscription agreement     : {agree}/{len(sims)} "
              f"(mean similarity {sum(sims)/len(sims):.2f})")
    if disagreements:
        print(f"\n  rows to review ({len(disagreements)}):")
        for number, new, old in disagreements[:15]:
            print(f"    #{number:>6} re-OCR: {new[:44]}")
            print(f"    {'':>7} parse : {old[:44]}")


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv(Path(__file__).with_name(".env"))

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--pages", default="27,81,160",
                        help="Comma-separated 0-based page numbers, or 'all'")
    parser.add_argument("--output", required=True, type=Path,
                        help="CSV of re-OCR rows (parse_tsp_list.py shape)")
    parser.add_argument("--compare", type=Path,
                        help="Existing tsp_brick_list.csv to diff against")
    args = parser.parse_args(argv)

    import pypdfium2 as pdfium
    from google import genai
    from google.genai import types

    pdf = pdfium.PdfDocument(str(args.pdf))
    pages = (list(range(len(pdf))) if args.pages.strip().lower() == "all"
             else [int(p) for p in args.pages.split(",")])
    client = genai.Client()

    all_rows, in_tokens, out_tokens = [], 0, 0
    for index in pages:
        rows, usage = _transcribe(client, types, _render_page(pdf, index))
        in_tokens += getattr(usage, "prompt_token_count", 0) or 0
        out_tokens += getattr(usage, "candidates_token_count", 0) or 0
        print(f"  page {index}: {len(rows)} rows")
        all_rows.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in all_rows:
            number = re.sub(r"[^0-9]", "", str(row.get("brick", "")))
            name = " ".join(str(row.get(k, "")) for k in ("line1", "line2")).strip()
            buyer = " ".join(str(row.get(k, "")) for k in ("last", "first")).strip()
            if number and (name or buyer):
                writer.writerow({"section": "", "assigned_id": number,
                                 "pos1": "", "pos2": "",
                                 "full_name": name, "key_word": buyer})

    cost = in_tokens / 1e6 * 0.30 + out_tokens / 1e6 * 2.50
    print(f"\nWrote {args.output}  ({len(all_rows)} rows, {len(pages)} pages)")
    print(f"Tokens: {in_tokens} in / {out_tokens} out  ~= ${cost:.4f} "
          f"({MODEL}); all 174 pages ~= ${cost / len(pages) * 174:.2f}")

    if args.compare:
        _compare(all_rows, args.compare, pages)


if __name__ == "__main__":
    main()
