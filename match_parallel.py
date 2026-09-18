#!/usr/bin/env python3
"""Run match.py across CPU cores by splitting the photo catalogue.

Matching is per-photo: each row is scored against the whole reference list
and nothing about one photo affects another (the duplicate-claim tally is
the only cross-row product, and it is rebuilt here from the combined
result). So the catalogue can be cut into chunks, matched in parallel and
concatenated -- on a 24-core laptop that turns a ~50 minute single-core run
into a few minutes, which also means an interrupted run costs minutes
rather than an evening.

    python match_parallel.py --catalog output/pallets.csv \
        --reference reference/master_list.csv \
        --output output/pallets_matched.csv --workers 8

Chunk outputs land in a temp dir and the final files are written only once
every chunk has finished, so an interruption leaves the previous
pallets_matched.csv untouched.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=Path("output/pallets.csv"))
    p.add_argument("--reference", type=Path,
                   default=Path("reference/master_list.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("output/pallets_matched.csv"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--scan-ocr", action="store_true", default=True)
    a = p.parse_args(argv)

    with open(a.catalog, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    n = max(1, len(rows) // a.workers + 1)
    chunks = [rows[i:i + n] for i in range(0, len(rows), n)]
    tmp = Path(tempfile.mkdtemp(prefix="matchpar_"))
    print(f"{len(rows)} photos -> {len(chunks)} chunk(s) x {a.workers} worker(s)",
          flush=True)

    procs, outs = [], []
    for i, chunk in enumerate(chunks):
        cin, cout = tmp / f"in{i}.csv", tmp / f"out{i}.csv"
        with open(cin, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(chunk)
        cmd = [sys.executable, "match.py", "--catalog", str(cin),
               "--reference", str(a.reference), "--output", str(cout)]
        if a.scan_ocr:
            cmd.append("--scan-ocr")
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE))
        outs.append(cout)

    t0 = time.time()
    for i, pr in enumerate(procs):
        err = pr.communicate()[1]
        if pr.returncode != 0:
            raise SystemExit(f"chunk {i} failed:\n{(err or b'').decode()[:2000]}")
        print(f"  chunk {i + 1}/{len(procs)} done ({time.time() - t0:.0f}s)",
              flush=True)

    matched, review = [], []
    mcols = rcols = None
    for cout in outs:
        with open(cout, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            mcols = mcols or list(r.fieldnames or [])
            matched += list(r)
        rv = cout.with_name(cout.stem + "_review" + cout.suffix)
        rv = cout.parent / f"review_{cout.name}"
        if rv.is_file():
            with open(rv, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                rcols = rcols or list(r.fieldnames or [])
                review += list(r)

    order = {r["image"]: i for i, r in enumerate(rows)}
    matched.sort(key=lambda r: order.get(r["image"], 0))
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mcols)
        w.writeheader()
        w.writerows(matched)
    if review:
        rp = a.output.with_name("review_" + a.output.name)
        with open(rp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rcols)
            w.writeheader()
            w.writerows(review)
        print(f"wrote {rp} ({len(review)} candidate row(s))")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {a.output} ({len(matched)} photo(s)) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
