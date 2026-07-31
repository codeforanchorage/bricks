#!/usr/bin/env python3
"""Build the duplicate-claim (false-positive) check page.

Duplicate forensics over the matched catalogue found groups of photos that
all matched the SAME official brick even though their OCR reads DISAGREE --
for a brick that sold fewer copies than it has photos, at least one of
those matches is probably wrong (a false positive: the photo really shows
a different brick). Those groups live in output/fp_candidates.csv.

This generator turns that CSV into one hosted page, one group per brick:
the official inscription and its scanned list row on top, then every
claiming photo with its read, score, and pallet. For each photo the
reviewer picks:

  * "Correct" -- the photo really shows this brick (decision 'match',
    re-affirming the current catalogue row as human-verified), or
  * "Wrong brick" -- a false positive; decision 'mismatch' sends the
    photo BACK to the review queue (match_status 'unmatched') so the next
    review round can find its real identity. The note box is the place to
    write whatever words the reviewer can read.

Decisions autosave to the browser and (with --receiver-url) to the hosted
receiver, exactly like review.html -- apply_decisions.py folds them in
with everything else, no new plumbing.

The page is hosted-only (--photo-base-url required): every photo in these
groups is already matched, so the derivative trees are already uploaded.

Usage:
    python make_fp_page.py --fp output/fp_candidates.csv \
        --matched output/pallets_final.csv \
        --master reference/master_list.csv \
        --photo-base-url https://example.com/bricks/photos \
        --output output/dreamhost_upload/fp_review.html
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from urllib.parse import quote

from make_review_page import (_PAGE_BOTTOM, _hosted_photo_html,
                              _load_strip_map, _radio)

_PAGE_TOP = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Town Square duplicate-claim check</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0; background: #f4f2ee;
        color: #222; }}
 header {{ position: sticky; top: 0; background: #274156; color: #fff;
          padding: 10px 18px; z-index: 5; }}
 header h1 {{ font-size: 18px; margin: 0 0 4px; }}
 header p {{ margin: 2px 0; font-size: 13px; }}
 #bar {{ display: flex; gap: 14px; align-items: center; margin-top: 6px;
        flex-wrap: wrap; }}
 #bar input {{ padding: 5px 8px; border: none; border-radius: 4px;
              font-size: 14px; }}
 #bar button {{ padding: 7px 14px; border: none; border-radius: 4px;
               background: #e8a33d; font-weight: 600; cursor: pointer;
               font-size: 14px; }}
 #progress {{ font-size: 14px; }}
 .group {{ max-width: 1080px; margin: 26px auto 6px; font-size: 15px;
         color: #5a4a1f; background: #f6ecd4; border-radius: 8px;
         padding: 10px 16px; }}
 .group img.strip {{ display: block; width: 100%; max-width: 660px;
         margin-top: 6px; border: 1px solid #e0d6b8; border-radius: 4px; }}
 .group .meta {{ color: #8a7a4f; font-size: 12.5px; font-weight: normal; }}
 .item, .info {{ background: #fff; margin: 10px auto; max-width: 1080px;
         border-radius: 8px; padding: 14px 18px;
         box-shadow: 0 1px 4px rgba(0,0,0,.12); display: flex; gap: 18px;
         flex-wrap: wrap; }}
 .item.done {{ outline: 3px solid #7fb069; }}
 .photo img {{ width: 640px; max-width: 92vw; border-radius: 6px; }}
 .work {{ flex: 1; min-width: 300px; }}
 .work h2 {{ font-size: 15px; margin: 0 0 6px; }}
 .reads {{ font-size: 13px; color: #555; margin-bottom: 10px; }}
 .changed {{ font-size: 13px; color: #8a4f10; background: #fdf1e2;
         border-radius: 5px; padding: 6px 9px; margin-bottom: 8px; }}
 label.cand {{ display: block; padding: 7px 10px; margin: 4px 0;
              border: 1px solid #ddd; border-radius: 6px; cursor: pointer;
              font-size: 14px; }}
 label.cand:hover {{ background: #f0f6fb; }}
 label.cand input {{ margin-right: 8px; }}
 .meta {{ color: #777; font-size: 12px; margin-left: 8px; }}
 .note {{ width: 95%; margin-top: 8px; padding: 6px 8px; font-size: 13px;
         border: 1px solid #ccc; border-radius: 5px; }}
 button.clear {{ margin-top: 8px; padding: 4px 12px; font-size: 12.5px;
         border: 1px solid #ccc; border-radius: 5px; background: #f7f7f7;
         cursor: pointer; color: #555; }}
 button.clear:hover {{ background: #fbeaea; border-color: #d99; }}
</style>
</head>
<body>
<header>
 <h1>Town Square duplicate-claim check &mdash; {n_groups} brick(s),
     {n_photos} photo(s)</h1>
 <p>Each yellow band is ONE official brick that several photos matched,
    with reads that disagree &mdash; at least one photo is probably a
    different brick. Compare each photo against the scanned list row:
    click &ldquo;Correct&rdquo; if it really is this brick, or
    &ldquo;Wrong brick&rdquo; if not (then write whatever words you CAN
    read in the note). Choices save automatically.</p>
 <div id="bar">
  <label>Your name: <input id="reviewer" placeholder="required to export"></label>
  <button onclick="exportCsv()">Download decisions.csv</button>
  <span id="progress"></span>
  <span id="server"></span>
 </div>
</header>
"""


def _load_fp_groups(path: Path) -> list[tuple[tuple[str, str], list[dict]]]:
    """fp_candidates.csv -> ordered [(official_id, section), rows] groups."""
    groups: dict[tuple[str, str], list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["official_id"], row["official_section"])
            groups.setdefault(key, []).append(row)
    return list(groups.items())


def _load_matched_by_image(path: Path) -> dict[str, list[dict]]:
    by_image: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_image.setdefault(row.get("image", ""), []).append(row)
    return by_image


def _current_row(fp_row: dict, by_image: dict[str, list[dict]]) -> dict | None:
    """The photo's CURRENT catalogue row -- the match may have shifted
    since fp_candidates.csv was generated (rescans, reviewer decisions)."""
    rows = by_image.get(fp_row["image"], [])
    for row in rows:
        if (row.get("official_id"), row.get("official_section")) == \
                (fp_row["official_id"], fp_row["official_section"]):
            return row
    return rows[0] if rows else None


def _group_header(key: tuple[str, str], rows: list[dict],
                  hosted_base: str, strip_map: dict) -> str:
    official_id, section = key
    names = {r["official_name"] for r in rows}
    copies = rows[0].get("copies", "")
    title = (f'<b>#{html.escape(official_id)}</b> section '
             f'{html.escape(section or "?")}')
    if len(names) == 1:
        title += f' &mdash; {html.escape(next(iter(names)))}'
    else:
        # dup_orig collision: several list rows share this number, so the
        # claimed inscription differs per photo -- shown on each below.
        title += (' &mdash; <i>number collision: multiple list rows share '
                  'this number (each photo shows its own claimed '
                  'inscription)</i>')
    title += (f' <span class="meta">{html.escape(str(copies))} cop'
              f'{"y" if str(copies) == "1" else "ies"} sold &middot; '
              f'{len(rows)} photos matched</span>')
    orig = strip_map.get((section.upper(), official_id))
    if orig:
        title += (f'<img class="strip" loading="lazy" '
                  f'src="{hosted_base}/strips/{quote(orig)}.jpg" '
                  f'alt="scanned list row #{html.escape(orig)}" '
                  f'onerror="this.style.display=\'none\'">')
    return f'<h2 class="group">{title}</h2>'


def _photo_item(fp_row: dict, current: dict | None,
                hosted_base: str) -> str:
    image = fp_row["image"]
    photo_html = _hosted_photo_html(image, hosted_base)
    read_line = (f'claimed: {html.escape(fp_row["official_name"])} &middot; '
                 f'read: {html.escape(fp_row["matched_read"])} &middot; '
                 f'score {html.escape(fp_row["score"])}')
    if current and current.get("pallet"):
        read_line += f' &middot; pallet {html.escape(current["pallet"])}'

    # Info-only blocks must NOT be class="item": the shared page JS walks
    # every .item expecting a radio group and a note box.
    if current is None:
        return f"""
<section class="info">
 <div class="photo">{photo_html}</div>
 <div class="work">
  <h2>{html.escape(image)}</h2>
  <div class="reads">{read_line}</div>
  <div class="changed">Photo is no longer in the catalogue &mdash;
   nothing to decide.</div>
 </div>
</section>"""

    brick_id = str(current.get("brick_id", ""))
    if current.get("match_status") != "matched":
        return f"""
<section class="info">
 <div class="photo">{photo_html}</div>
 <div class="work">
  <h2>{html.escape(image)}</h2>
  <div class="reads">{read_line}</div>
  <div class="changed">No longer matched (status:
   {html.escape(current.get("match_status", ""))}) &mdash; already
   resolved, nothing to decide.</div>
 </div>
</section>"""

    changed = ""
    if (current.get("official_id"), current.get("official_section")) != \
            (fp_row["official_id"], fp_row["official_section"]):
        changed = (f'<div class="changed">Match has since changed: this '
                   f'photo now matches <b>#'
                   f'{html.escape(current.get("official_id", ""))}</b> '
                   f'{html.escape(current.get("official_name", ""))} &mdash; '
                   f'&ldquo;Correct&rdquo; confirms THAT match.</div>')

    group = f"d:{image}:{brick_id}"
    correct = _radio(
        group, "match",
        f'<b>Correct</b> &mdash; this photo shows '
        f'#{html.escape(current.get("official_id", ""))} '
        f'{html.escape(current.get("official_name", ""))}',
        id_=current.get("official_id", ""),
        section=current.get("official_section", ""),
        name=current.get("official_name", ""),
        keyword=current.get("official_keyword", ""))
    wrong = _radio(
        group, "mismatch",
        '<b>Wrong brick</b> &mdash; a different brick; goes back for '
        're-review. Write any words you can read in the note.',
        id_=current.get("official_id", ""),
        section=current.get("official_section", ""),
        name=current.get("official_name", ""))
    return f"""
<section class="item" data-image="{html.escape(image, quote=True)}"
         data-brick="{html.escape(brick_id, quote=True)}">
 <div class="photo">{photo_html}</div>
 <div class="work">
  <h2>{html.escape(image)}</h2>
  <div class="reads">{read_line}</div>
  {changed}
  {correct}
  {wrong}
  <input class="note" placeholder="note (optional)">
  <button type="button" class="clear" onclick="clearItem(this)">Clear
   choice</button>
 </div>
</section>"""


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp", required=True, type=Path,
                        help="fp_candidates.csv -- duplicate-claim groups "
                             "with disagreeing reads")
    parser.add_argument("--matched", required=True, type=Path,
                        help="Current matched catalogue "
                             "(output/pallets_final.csv) -- supplies each "
                             "photo's brick_id and CURRENT match")
    parser.add_argument("--master", type=Path,
                        help="reference/master_list.csv -- show each "
                             "group's scanned list-row image")
    parser.add_argument("--photo-base-url", required=True,
                        help="Base URL of the uploaded derivative trees")
    parser.add_argument("--output", required=True, type=Path,
                        help=".html file to write")
    parser.add_argument("--receiver-url", default="",
                        help="URL of the hosted receiver.php; adds autosave-"
                             "to-server and send-on-export to the page")
    parser.add_argument("--receiver-token", default="",
                        help="Shared token, must match TOKEN in receiver.php")
    args = parser.parse_args(argv)
    if bool(args.receiver_url) != bool(args.receiver_token):
        raise SystemExit("--receiver-url and --receiver-token go together")

    groups = _load_fp_groups(args.fp)
    if not groups:
        raise SystemExit(f"No rows in {args.fp} -- nothing to check.")
    by_image = _load_matched_by_image(args.matched)
    hosted_base = args.photo_base_url.rstrip("/")
    strip_map = _load_strip_map(args.master)

    sections, n_photos, n_decidable = [], 0, 0
    for key, rows in groups:
        sections.append(_group_header(key, rows, hosted_base, strip_map))
        for fp_row in rows:
            current = _current_row(fp_row, by_image)
            item = _photo_item(fp_row, current, hosted_base)
            n_photos += 1
            if "data-brick" in item:
                n_decidable += 1
            sections.append(item)

    receiver = ({"url": args.receiver_url, "token": args.receiver_token}
                if args.receiver_url else None)
    page = (_PAGE_TOP.format(n_groups=len(groups), n_photos=n_photos)
            + "".join(sections)
            + _PAGE_BOTTOM.replace("__RECEIVER__", json.dumps(receiver)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")

    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output}  ({len(groups)} group(s), {n_photos} "
          f"photo(s), {n_decidable} decidable, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
