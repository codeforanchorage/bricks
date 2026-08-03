#!/usr/bin/env python3
"""Run the whole brick pipeline in order, with one command.

The steps (each skippable) mirror the manual workflow:

  1. ocr        single_pipeline.py --resume  (new/erroneous photos only)
  2. rescan     rescan_rows.py     (re-read troublesome scan rows; costs
                a little API money, so OFF unless --rescan)
  3. merge      merge_lists.py     (rebuild reference/master_list.csv)
  4. gate       pytest             (the 26/26 zero-false-positive gate --
                a failure ABORTS before anything ships)
  5. match      match.py           (all photos vs the master list)
  6. classify   classify_photos.py (unmatched photos: single/stack/other)
  7. pull       fetch reviewer decisions from the hosted receiver
                (REVIEW_RECEIVER_URL/REVIEW_TOKEN/STAFF_LOGIN in .env)
  8. apply      apply_decisions.py (local decisions*.csv + pulled files)
  9. pages      review + search pages into output/dreamhost_upload/
                (+ the duplicate-claim FP check page when
                output/fp_candidates.csv exists)
 10. report     the Parks & Rec Excel workbook

Configuration comes from .env (git-ignored): API keys, receiver URL and
token, the staff basic-auth login, and PHOTO_BASE_URL. Nothing secret
lives on a command line.

Typical runs:
    python run_pipeline.py                     # everything but rescan
    python run_pipeline.py --rescan            # include the strip re-read
    python run_pipeline.py --only pages,report # regenerate outputs only
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from base64 import b64encode
from os import environ
from pathlib import Path

from pipeline import _load_dotenv

PDF = "TSP Bricks ALL - OG List by Name - OCR.pdf"
STEPS = ["ocr", "rescan", "merge", "gate", "match", "classify", "pull",
         "apply", "pages", "report"]


def _banner(step: str) -> None:
    print(f"\n=== {step} " + "=" * (66 - len(step)), flush=True)


def _pull_decisions(dest: Path) -> list[Path]:
    """Download all decision files the hosted receiver has stored."""
    url = environ.get("REVIEW_RECEIVER_URL", "")
    token = environ.get("REVIEW_TOKEN", "")
    login = environ.get("STAFF_LOGIN", "")
    if not (url and token and login):
        print("receiver not configured in .env -- skipping pull "
              "(need REVIEW_RECEIVER_URL, REVIEW_TOKEN, STAFF_LOGIN)")
        return []
    auth = "Basic " + b64encode(login.encode()).decode()

    def get(query: str) -> bytes:
        req = urllib.request.Request(f"{url}?token={token}&{query}",
                                     headers={"Authorization": auth})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    try:
        files = json.loads(get("action=list"))
        dest.mkdir(parents=True, exist_ok=True)
        pulled = []
        for entry in files:
            name = entry["name"]
            path = dest / name
            path.write_bytes(get(f"action=fetch&name={name}"))
            pulled.append(path)
        print(f"pulled {len(pulled)} decision file(s) from the receiver")
        return pulled
    except Exception as exc:  # noqa: BLE001 -- offline/old receiver: warn
        print(f"pull failed ({exc}) -- continuing with local decision "
              f"files only. (Is the updated receiver.php uploaded?)")
        return []


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_dotenv(Path(__file__).with_name(".env"))

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, default=Path("photos"),
                        help="Photo root (default: photos/)")
    parser.add_argument("--rescan", action="store_true",
                        help="Include the strip re-read step (API cost)")
    parser.add_argument("--only", default="",
                        help="Comma-separated subset of steps to run: "
                             + ",".join(STEPS))
    parser.add_argument("--skip", default="",
                        help="Comma-separated steps to skip")
    args = parser.parse_args(argv)

    run = set(args.only.split(",")) if args.only else set(STEPS)
    run -= set(s for s in args.skip.split(",") if s)
    if not args.rescan and not args.only:
        run.discard("rescan")
    unknown = run - set(STEPS)
    if unknown:
        raise SystemExit(f"Unknown step(s): {', '.join(sorted(unknown))}")

    out = Path("output")
    kit = out / "dreamhost_upload"
    photo_base = environ.get("PHOTO_BASE_URL", "")
    receiver_url = environ.get("REVIEW_RECEIVER_URL", "")
    review_token = environ.get("REVIEW_TOKEN", "")

    if "ocr" in run:
        _banner("ocr (resume: new photos and ERROR rows only)")
        import single_pipeline
        single_pipeline.main(["--input", str(args.photos),
                              "--output", str(out / "pallets.csv"),
                              "--resume"])

    if "rescan" in run:
        _banner("rescan troublesome scan rows")
        import rescan_rows
        rescan_rows.main(["--pdf", PDF,
                          "--review", str(out / "review_pallets_matched.csv")])

    if "merge" in run:
        _banner("merge: rebuild the master list")
        import shutil

        import merge_lists
        master = Path("reference/master_list.csv")
        if master.is_file():
            shutil.copy2(master, master.with_suffix(".csv.prev"))
        merge_lists.main(["--og", "reference/tsp_brick_list_v2.csv",
                          "--new", "reference/brick_list_xls.csv",
                          "--output", str(master)])

    if "gate" in run:
        _banner("regression gate (26/26, zero false positives)")
        import pytest
        if pytest.main(["-q", "tests/"]) != 0:
            raise SystemExit(
                "GATE FAILED -- the rebuilt data broke validated matching. "
                "Nothing shipped. reference/master_list.csv.prev holds the "
                "previous master; investigate before re-running.")

    if "match" in run:
        _banner("match all photos against the master list")
        import match
        match.main(["--catalog", str(out / "pallets.csv"),
                    "--reference", "reference/master_list.csv",
                    "--output", str(out / "pallets_matched.csv"),
                    "--scan-ocr"])

    if "classify" in run:
        _banner("classify unmatched photos (single/stack/other)")
        import classify_photos
        classify_photos.main(["--matched", str(out / "pallets_matched.csv"),
                              "--photos", str(args.photos),
                              "--output", str(out / "photo_types.csv")])

    pulled: list[Path] = []
    if "pull" in run:
        _banner("pull reviewer decisions from the receiver")
        pulled = _pull_decisions(out / "decisions")

    if "apply" in run:
        _banner("apply reviewer decisions")
        import apply_decisions
        local = sorted(Path(".").glob("decisions*.csv"))
        pulled = pulled or sorted((out / "decisions").glob("*.csv"))
        # Later files win on conflict, so order deliberately: local file,
        # then autosaves oldest-first, then FINAL submits oldest-first --
        # a reviewer's explicit export always beats a rolling autosave.
        pulled.sort(key=lambda p: (not p.name.startswith("autosave_"),
                                   p.stat().st_mtime))
        decision_files = [str(p) for p in local + pulled]
        if decision_files:
            apply_decisions.main(["--matched",
                                  str(out / "pallets_matched.csv"),
                                  "--decisions", *decision_files,
                                  "--output", str(out / "pallets_final.csv")])
        else:
            import shutil
            shutil.copy2(out / "pallets_matched.csv",
                         out / "pallets_final.csv")
            print("no decision files found -- pallets_final.csv is the "
                  "machine catalogue unchanged")

    if "pages" in run:
        _banner("regenerate the hosted pages")
        import make_review_page
        import make_search_page
        # Stable filenames: the pages' nav bar links them to each other,
        # and .htaccess serves HTML no-cache, so the old reason to
        # date-stamp (stale browser caches) is gone. The built date shows
        # in each page's header instead.
        review_args = ["--review", str(out / "review_pallets_matched.csv"),
                       "--catalog", str(out / "pallets.csv"),
                       "--matched", str(out / "pallets_matched.csv"),
                       "--photo-types", str(out / "photo_types.csv"),
                       "--master", "reference/master_list.csv",
                       "--output", str(kit / "review.html")]
        if photo_base:
            review_args += ["--photo-base-url", photo_base]
        if receiver_url and review_token:
            review_args += ["--receiver-url", receiver_url,
                            "--receiver-token", review_token]
        make_review_page.main(review_args)
        make_search_page.main(["--master", "reference/master_list.csv",
                               "--matched", str(out / "pallets_final.csv"),
                               "--output", str(kit / "search.html")])
        fp = out / "fp_candidates.csv"
        if fp.is_file() and photo_base:
            import make_fp_page
            fp_args = ["--fp", str(fp),
                       "--matched", str(out / "pallets_final.csv"),
                       "--master", "reference/master_list.csv",
                       "--photo-base-url", photo_base,
                       "--output", str(kit / "fp_review.html")]
            if receiver_url and review_token:
                fp_args += ["--receiver-url", receiver_url,
                            "--receiver-token", review_token]
            make_fp_page.main(fp_args)

    if "report" in run:
        _banner("Parks & Rec Excel workbook")
        import make_report
        make_report.main(["--master", "reference/master_list.csv",
                          "--matched", str(out / "pallets_final.csv"),
                          "--output", str(out / "brick_report.xlsx")])

    print("\nDone. Upload output/dreamhost_upload/ changes to the server; "
          "output/brick_report.xlsx is the P&R deliverable.")


if __name__ == "__main__":
    main()
