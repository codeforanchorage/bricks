#!/usr/bin/env python3
"""Audit the strip images: read the printed brick number off each strip and
compare it with the filename.

Strips are cut from the scanned list using the PDF text layer's row
positions (resolve_tsp_rows.physical_rows). Where that layer is offset
from the printed page, every strip on the page shows a NEIGHBOUR's row
under the wrong number (found 2026-09-17: strips/6889.jpg -- Ashley
Durst -- shows the printed 6891 Zjok Durst row). The printed number is in
the strip itself, so a cheap model read of the digits catches every
misfiled strip. Resumable; ~$2 for the whole tree at Flash-Lite prices.

    python audit_strips.py --strips derivatives/strips --output output/strip_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline import _load_dotenv

PROMPT = ("This image is one row cut from a printed list: surname, first "
          "name, a brick number of 1 to 5 digits, then the inscription. "
          "Reply with ONLY the brick number digits, nothing else. If no "
          "number is visible reply NONE.")
DEFAULT_MODEL = "gemini-2.5-flash-lite"   # cheapest vision model; digits only
_client = None
_lock = threading.Lock()


def _get_client():
    global _client
    with _lock:
        if _client is None:
            from google import genai
            _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def read_number(path: Path, model: str = DEFAULT_MODEL) -> str:
    from google.genai import types
    part = types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg")
    for attempt in range(3):
        try:
            r = _get_client().models.generate_content(
                model=model, contents=[part, PROMPT])
            m = re.search(r"\d{1,5}", r.text or "")
            return m.group(0) if m else "NONE"
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return f"ERROR: {exc}"[:120]
            time.sleep(4 * 2 ** attempt)
    return "ERROR"


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv(Path(__file__).with_name(".env"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strips", type=Path, default=Path("derivatives/strips"))
    p.add_argument("--output", type=Path, default=Path("output/strip_audit.csv"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    a = p.parse_args(argv)

    done = {}
    if a.output.is_file():
        for row in csv.DictReader(open(a.output, newline="", encoding="utf-8")):
            if not row["printed"].startswith("ERROR"):
                done[row["file"]] = row
    log = a.output.with_suffix(".log.csv")
    if log.is_file():
        for file, printed, ok in csv.reader(open(log, newline="", encoding="utf-8")):
            if file not in done and not printed.startswith("ERROR"):
                done[file] = {"file": file, "printed": printed, "match": ok}
    strips = sorted(a.strips.glob("*.jpg"), key=lambda q: int(q.stem))
    todo = [s for s in strips if s.stem not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(strips)} strips, {len(done)} already audited, {len(todo)} to read")
    t0 = time.time()
    log = a.output.with_suffix(".log.csv")      # append-only progress log
    with ThreadPoolExecutor(a.workers) as ex,             open(log, "a", newline="", encoding="utf-8") as lf:
        lw = csv.writer(lf)
        futs = {ex.submit(read_number, s, a.model): s for s in todo}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            printed = f.result()
            ok = ("" if printed.startswith("ERROR") or printed == "NONE"
                  else str(int(printed) == int(s.stem)))
            done[s.stem] = {"file": s.stem, "printed": printed, "match": ok}
            lw.writerow([s.stem, printed, ok]); lf.flush()
            if i % 500 == 0:
                print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    rows = [done[k] for k in sorted(done, key=int)]
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "printed", "match"])
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["match"] == "False"]
    unread = [r for r in rows if r["match"] == ""]
    print(f"audited {len(rows)}: filename matches printed number "
          f"{len(rows) - len(bad) - len(unread)}, MISFILED {len(bad)}, "
          f"unreadable/error {len(unread)}")


if __name__ == "__main__":
    main()
