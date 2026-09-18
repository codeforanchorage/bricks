#!/bin/sh
# Resume the photo-side second-opinion re-read of the bricks whose photo text
# differs from the list (output/brick_vs_strip_diffs.csv). Safe to run any
# number of times: --resume keeps good reads and redoes only ERROR rows, so
# an internet drop or a stop costs nothing but the photos not yet read.
cd "$(dirname "$0")" || exit 1
STAGE="${1:-$TMPDIR/photoside}"
# Re-stage the images if the scratchpad copy is gone (they come from
# derivatives/zoom, so this is free).
.venv/Scripts/python.exe - "$STAGE" <<'PY'
import csv, os, shutil, sys
from pathlib import Path
stage = Path(sys.argv[1])
n = 0
for d in csv.DictReader(open("output/brick_vs_strip_diffs.csv", encoding="utf-8")):
    src = Path("derivatives/zoom") / (os.path.splitext(d["image"])[0] + ".jpg")
    dst = stage / d["image"]
    if dst.exists() or not src.exists():
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    n += 1
print(f"staged {n} more image(s) under {stage}")
PY
.venv/Scripts/python.exe single_pipeline.py --input "$STAGE" \
    --output output/photoside_reads.csv \
    --methods gemini-lite-35,gemini-flash-3 --workers 6 --resume
