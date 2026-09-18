#!/usr/bin/env python3
"""Read images with a Gemini model through the BATCH API.

The interactive path (ocr_google.py, driven by single_pipeline.py) sends one
request per image and is rate-limited per minute: measured 2026-09-18, a
single call takes 4-8s, yet 6 workers x 2 models moved only ~7 photos/min,
because throttled calls were silently retried with 4s and 8s backoffs.
Preview models throttle hardest.

Batch mode is the right tool for a bulk pass: one job carries thousands of
requests, Google schedules them, and the price is half the interactive rate.
The trade is latency -- a job returns in minutes to hours, not seconds -- so
use it for whole-tree passes and keep the interactive path for the handful of
photos a review turns up.

    # submit, print the job name, exit
    python ocr_google_batch.py submit --input photos/ \
        --method gemini-lite-35 --state output/batch_pallets.json

    # later: poll, then write the catalogue rows when it finishes
    python ocr_google_batch.py collect --state output/batch_pallets.json \
        --output output/pallets_batch.csv

The output CSV matches single_pipeline's shape (image, brick_id, section,
pallet, x, y, w, h, <method column>, status), so merge_reads.py folds it into
output/pallets.csv like any other read, and match.py then treats it as one
more read of each photo.

Cost: batch is billed at half the interactive rate. At the measured ~1,170
input tokens for a 1568px brick photo, a full 11,510-photo pass with one
model is roughly $2 interactive, about $1 batched.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import vision_ocr
from pipeline import METHODS, _load_dotenv, find_images

# A job that reaches one of these is finished, for better or worse.
_TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
             "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def _client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set (see .env)")
    return genai.Client(api_key=key)


def _tags(rel: str) -> tuple[str, str]:
    """(section, pallet) from a photo's folders, as single_pipeline does."""
    parts = Path(rel).parts[:-1]
    section = pallet = ""
    for i, part in enumerate(parts):
        if len(part) == 1 and part.upper() in "ABCDEFGHIJK":
            section = part.upper()
        elif part.lower() == "pallets" and i + 1 < len(parts):
            pallet = parts[i + 1]
    return section, pallet


def submit(args) -> None:
    from google.genai import types

    method = METHODS[args.method]
    model = method[3]
    images = find_images(args.input, recursive=True)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"no images under {args.input}")
    prompt = vision_ocr.BRICK_PROMPT if args.single_brick else vision_ocr.PROMPT

    requests, index = [], []
    for image in images:
        data = vision_ocr.load_jpeg_bytes(image)
        requests.append(types.InlinedRequest(
            model=model,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=data, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt)])]))
        index.append(image.relative_to(args.input).as_posix())

    print(f"submitting {len(requests)} image(s) to {model} as one batch job...",
          flush=True)
    # Hold the client in a local: a temporary is collected mid-call and its
    # httpx session closes under the request ("Cannot send a request, as the
    # client has been closed").
    client = _client()
    job = client.batches.create(model=model, src=requests)
    state = {"job": job.name, "model": model, "method": args.method,
             "column": method[0], "input_root": str(args.input),
             "images": index, "submitted": time.time()}
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"job {job.name} ({job.state}) -> {args.state}")
    print("collect it later with:  python ocr_google_batch.py collect "
          f"--state {args.state} --output <csv>")


def _state_name(job) -> str:
    """JOB_STATE_* for a job, whether state is an enum or a plain string.

    job.state is a JobState enum whose str() is "JobState.JOB_STATE_SUCCEEDED"
    -- comparing that against the bare names silently never matches, which
    made an early version of this script report SUCCEEDED and then write
    nothing.
    """
    return getattr(job.state, "name", str(job.state))


def collect(args) -> None:
    state = json.loads(args.state.read_text(encoding="utf-8"))
    client = _client()
    job = client.batches.get(name=state["job"])
    while _state_name(job) not in _TERMINAL and not args.no_wait:
        stamp = time.strftime("%H:%M:%S")
        print(f"  {_state_name(job)} ... (checked {stamp})", flush=True)
        time.sleep(args.poll)
        job = client.batches.get(name=state["job"])
    print(f"job {state['job']}: {_state_name(job)}")
    if _state_name(job) not in _TERMINAL:
        return
    if job.error:
        print(f"  error: {job.error}")

    column, images = state["column"], state["images"]
    dest = getattr(job, "dest", None)
    responses = getattr(dest, "inlined_responses", None) or []
    if len(responses) != len(images):
        print(f"  WARNING: {len(responses)} response(s) for {len(images)} "
              "image(s) -- rows are matched by position, check the tail")

    rows, failed = [], 0
    for rel, resp in zip(images, responses):
        error = getattr(resp, "error", None)
        if error:
            read = f"ERROR: {error}"
            failed += 1
        else:
            try:
                reads = vision_ocr.parse_response(resp.response.text or "")
                read = " / ".join(r["inscription"] for r in reads
                                  if r.get("inscription"))
            except Exception as exc:                        # noqa: BLE001
                read = f"ERROR: {exc}"
                failed += 1
        section, pallet = _tags(rel)
        ok = read and not read.startswith("ERROR:")
        rows.append({"image": rel, "brick_id": 1, "section": section,
                     "pallet": pallet, "x": "", "y": "", "w": "", "h": "",
                     column: read.replace("\n", " / "),
                     "status": "single" if ok else "none"})

    cols = ["image", "brick_id", "section", "pallet", "x", "y", "w", "h",
            column, "status"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} row(s), {failed} failed)")
    print("fold it into the catalogue with:  python merge_reads.py "
          f"--catalog output/pallets.csv --extra {args.output}")


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="send a whole photo tree as one job")
    s.add_argument("--input", required=True, type=Path)
    s.add_argument("--method", default="gemini-lite-35",
                   choices=[k for k, v in METHODS.items() if v[2] == "google"],
                   help="any Google method from pipeline.METHODS")
    s.add_argument("--state", type=Path, default=Path("output/batch_job.json"),
                   help="where the job name and image order are recorded")
    s.add_argument("--single-brick", action="store_true",
                   help="use the single-brick crop prompt")
    s.add_argument("--limit", type=int, default=0,
                   help="submit only the first N images (smoke test)")
    s.set_defaults(func=submit)

    c = sub.add_parser("collect", help="poll a submitted job and write rows")
    c.add_argument("--state", type=Path, default=Path("output/batch_job.json"))
    c.add_argument("--output", required=True, type=Path)
    c.add_argument("--poll", type=int, default=60,
                   help="seconds between state checks")
    c.add_argument("--no-wait", action="store_true",
                   help="report the state and exit instead of polling")
    c.set_defaults(func=collect)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
