#!/usr/bin/env python3
"""Build the pickup-counter search page: one self-contained search.html.

Front-desk staff type a name, words from the inscription, or a brick
number; the page answers "does this brick exist, which section, which
pallet" from the baked-in master list -- no server, no install, works
offline and hosted alike (Dreamhost plan: behind basic auth on the
bricks subdomain).

The fuzzy matching is the SAME logic the OCR matcher uses, ported to
JavaScript with the fold tables emitted straight from consensus.py so
they cannot drift: scan-confusable folding (I/L/T/1/J ...), phonetic
folding (people SAY names at the counter and parts of the list were
voice-transcribed: BRIAN/BRYAN, CATHY/KATHY), boilerplate-word dropping,
and per-word similarity scoring. A numeric query searches BOTH numbering
eras -- original certificate numbers and 2008 renumbers collide, so all
hits are shown with their era spelled out.

With --matched (one or more matched CSVs from match.py), each brick also
shows its photo status: photographed on which pallet, or not yet seen.
Photographed bricks get a click-to-expand VERIFICATION panel: the brick
photo (thumbnail; clicking overlays the 2500px zoom image, fetched only
then), the OCR read, and the scanned list-row image (click to magnify)
-- so staff can catch a false-positive match by eye before a claimant
arrives. Image URLs are relative (photos/...), so
they resolve when the page is hosted next to the derivative trees and
silently hide when offline -- search itself needs no network.

Usage:
    python make_search_page.py --master reference/master_list.csv \
        --matched output/singles_matched.csv \
        --output output/search.html
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

from consensus import _CONFUSABLE, _STOPWORDS, _normalise, _similar
from hostpaths import derivative_rel
from pagenav import NAV_CSS, nav_html

# Row layout baked into the page (arrays, not objects: ~40% smaller).
# og_display: the better-looking of the by-name list's two transcriptions --
# the scan parse has right rows but noisy glyphs (I'H for TH, IDVES for
# LOVES); the vision re-read (og_alt) has clean glyphs but occasional
# wrong-row slips, so it is shown only when it resembles the parse (same
# 0.40 rule match.py uses). og_extra keeps the other text SEARCHABLE
# without displaying it.
FIELDS = ["orig_id", "new_id", "section", "moved", "status", "buyer",
          "og_display", "new_inscription", "flag", "og_extra"]

_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Town Square bricks &mdash; pickup counter search</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 0; background: #f4f2ee;
        color: #1d2733; }
 header { background: #274156; color: #fff; padding: 14px 20px; }
 header h1 { font-size: 20px; margin: 0; }
 header .stamp { font-size: 12px; opacity: .75; margin-top: 2px; }
 #wrap { max-width: 900px; margin: 0 auto; padding: 14px 16px 60px; }
 #q { width: 100%; box-sizing: border-box; font-size: 24px; padding: 12px 14px;
      border: 2px solid #274156; border-radius: 8px; margin-top: 14px; }
 #meta { color: #666; font-size: 14px; margin: 8px 2px; min-height: 1.2em; }
 details { background: #fff; border-radius: 8px; padding: 10px 14px;
           margin-top: 10px; font-size: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
 summary { cursor: pointer; font-weight: 600; }
 details li { margin: 6px 0; }
 .card { background: #fff; border-radius: 8px; padding: 12px 16px;
         margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,.12); }
 .insc { font-size: 18px; font-weight: 600; }
 .sub { color: #555; font-size: 14px; margin-top: 4px; }
 .chip { display: inline-block; border-radius: 12px; padding: 2px 10px;
         font-size: 12.5px; font-weight: 600; margin-right: 6px; }
 .ok    { background: #e2f0d9; color: #2d5a1b; }
 .warn  { background: #fdeeca; color: #7a5a00; }
 .gray  { background: #e8e8e8; color: #555; }
 .photo { background: #d9e8f0; color: #1b4a5a; }
 .ids { font-family: ui-monospace, Consolas, monospace; }
 mark { background: #ffe9a8; padding: 0 1px; }
 button.verify { margin-left: 6px; padding: 1px 10px; font-size: 12.5px;
        border: 1px solid #bcd; border-radius: 12px; background: #eef5fa;
        cursor: pointer; color: #1b4a5a; }
 .panel { margin-top: 10px; border-top: 1px solid #eee; padding-top: 8px; }
 .panel img.photo { max-width: 640px; width: 100%; border-radius: 6px;
        display: block; }
 .panel img.striprow { max-width: 660px; width: 100%; border: 1px solid #eee;
        border-radius: 4px; display: block; margin-top: 6px;
        cursor: zoom-in; }
 .panel .cap { font-size: 12.5px; color: #666; margin: 3px 0 6px; }
 .panel img.photo { cursor: zoom-in; }
 /* Click-to-magnify, one overlay for both panel images. The scan strips
    are 1400px wide but the panel shows them at 660px -- too small to read
    a worn row; the photo thumb is 640px but a 2500px zoom exists, loaded
    ONLY when clicked. Tap opens the overlay (hover-zoom alone fails on
    the counter tablets); tap again or Escape closes. */
 #zoomstrip { position: fixed; inset: 0; z-index: 10; cursor: zoom-out;
        background: rgba(20,28,36,.88); display: flex;
        align-items: center; justify-content: center; }
 #zoomstrip img { background: #fff; padding: 10px 6px; border-radius: 4px;
        box-sizing: border-box; }
 #zoomstrip img.wide { width: 97vw; max-width: none; }
 #zoomstrip img.fit { max-width: 97vw; max-height: 94vh;
        object-fit: contain; }
__NAVCSS__</style>
</head>
<body>
<header>
 __NAV__
 <h1>Town Square bricks &mdash; pickup counter search</h1>
 <div class="stamp">__STAMP__</div>
</header>
<div id="wrap">
<input id="q" placeholder="Type a name, words from the brick, or a brick number&hellip;" autofocus autocomplete="off">
<div id="meta"></div>
<details>
 <summary>Front desk &mdash; how to use this page</summary>
 <ol>
  <li><b>Ask for the name on the brick</b> (or words they remember from it),
      or the number from their certificate. Type it above &mdash; spelling
      does not have to be exact, and names spelled by sound (BRIAN/BRYAN,
      CATHY/KATHY) still match.</li>
  <li><b>Numbers:</b> the same number can exist twice &mdash; bricks moved in
      the 2008 renovation were given NEW numbers, so every numeric hit says
      which numbering it is. Certificates show the <i>original</i> number.</li>
  <li><b>Read the visitor the result:</b> the section letter is where the
      brick was in the park. An <span class="chip photo">at pickup site</span>
      chip means the brick was photographed at the pickup location &mdash; it
      made the move from Town Square and is on that pallet. No chip only
      means it has not been photographed yet, NOT that it is lost.</li>
  <li><b>Status meanings:</b>
      <span class="chip ok">on the list</span> the brick is in the official
      records; <span class="chip warn">needs verification</span> the two
      official lists disagree about this row &mdash; take the visitor's name
      and contact info for follow-up, do not promise the brick;
      <span class="chip gray">no brick made</span> the purchase is recorded
      but no brick was ever engraved.</li>
  <li><b>Checking a match before pickup:</b> the
      <span class="chip photo">at pickup site</span> bricks have a
      <b>verify &#128247;</b> button &mdash; it opens the warehouse photo,
      what the computer read from it, and the brick&rsquo;s row in the
      scanned official list. If the photo doesn&rsquo;t say what the list
      row says, the match may be wrong: note it for follow-up rather than
      promising the brick.</li>
  <li><b>&ldquo;Unofficial&rdquo; results</b> are bricks a reviewer
      confirmed exist at the pickup site but are missing from the official
      lists &mdash; they show what the brick reads and which pallet holds
      it. The photo is the record; the visitor can still claim it.</li>
  <li><b>Not found at all?</b> Try fewer words, or just the surname. If it
      is genuinely absent, take contact info and add it to the follow-up
      list &mdash; absence here is not final until every pallet is
      photographed.</li>
  <li><b>Every claimant</b> must also complete the Municipality&rsquo;s
      attestation form before taking a brick.</li>
 </ol>
</details>
<div id="out"></div>
</div>
<script>
"use strict";
const DATA = __DATA__;          // [orig,new,section,moved,status,buyer,og,newi,flag,extra]
const PHOTOS = __PHOTOS__;      // "SECTION|id" -> [pallet,image,extra,read,note]
const UNOFFICIAL = __UNOFFICIAL__;  // [image,pallet,read,note] no_match photos
const SCAN = __SCAN__;          // scan-OCR confusable fold map (from consensus.py)
const STOP = new Set(__STOP__); // boilerplate words dropped from match keys

function normalise(s) {
  return s.toLowerCase().replace(/\//g, " ").replace(/[^a-z0-9 ]+/g, " ")
          .trim().replace(/\s+/g, " ");
}
function phoneticFold(s) {
  s = s.split("ph").join("f").split("ck").join("k");
  s = s.replace(/c/g, "k").replace(/z/g, "s").replace(/y/g, "i");
  return s.replace(/(.)\1+/g, "$1");
}
function scanFold(s) {
  let out = "";
  for (const ch of s) out += (SCAN[ch] || ch);
  return out;
}
function fold(s) { return scanFold(phoneticFold(s)); }

// Similarity of two short words: 1 - levenshtein/maxlen, best of raw/folded.
function lev(a, b) {
  if (a === b) return 0;
  const m = a.length, n = b.length;
  if (!m || !n) return Math.max(m, n);
  let prev = new Array(n + 1), cur = new Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    cur[0] = i;
    for (let j = 1; j <= n; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1,
                        prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    [prev, cur] = [cur, prev];
  }
  return prev[n];
}
function ratio(a, b) {
  const L = Math.max(a.length, b.length);
  return L ? 1 - lev(a, b) / L : 1;
}
function wsim(a, b) {
  if (a === b) return 1;
  // Folded similarity gets only half credit above raw: the folds exist to
  // bridge scan noise and phonetic spellings (KATHY/CATHY raw .8 -> .9),
  // but stacking them can collide unrelated words (COY/COLL both fold to
  // "koi"); half credit keeps real variants high and sinks coincidences.
  const raw = ratio(a, b);
  return Math.max(raw, (raw + ratio(fold(a), fold(b))) / 2);
}

function wordsOf(text) {
  // Single characters are scan-split debris ("c oy") -- pure match noise.
  return [...new Set(text.split(" ")
                         .filter(w => w.length >= 2 && !STOP.has(w)))];
}

// Precompute each row's searchable words once: buyer + both inscriptions,
// PLUS the matched photo's OCR read and any review note -- so what a
// reviewer typed (or the granite actually says) is findable later.
const ROWS = DATA.map(r => {
  const p = PHOTOS[r[2].toUpperCase() + "|" + (r[1] || r[0])];
  const extra = p ? " " + (p[3] || "") + " " + (p[4] || "") : "";
  const text = normalise(r[5] + " " + r[6] + " " + r[7] + " " + r[9] + extra);
  return { r, text, words: wordsOf(text) };
});
// Human-confirmed no_match photos: bricks that exist at the pickup site
// but are absent from the official records. Searchable by their OCR read
// and review note; they carry a photo and a pallet, not an id.
for (const u of UNOFFICIAL) {
  const text = normalise((u[2] || "") + " " + (u[3] || ""));
  if (text) ROWS.push({ u, text, words: wordsOf(text) });
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                  .replace(/>/g, "&gt;");
}

function chips(r) {
  const status = r[4], sec = r[2], key = sec.toUpperCase() + "|" + (r[1] || r[0]);
  let h = "";
  if (status === "ok") h += '<span class="chip ok">on the list</span>';
  else if (status === "no_brick") h += '<span class="chip gray">no brick made</span>';
  else h += '<span class="chip warn">needs verification</span>';
  const p = PHOTOS[key];
  if (p) h += '<span class="chip photo">at pickup site &mdash; pallet ' +
              esc(p[0] || "?") + '</span>';
  return h;
}

function idLine(r) {
  const [orig, newid, sec, moved] = [r[0], r[1], r[2], r[3]];
  const s = sec ? "Section " + esc(sec.toUpperCase()) : "section unknown";
  if (newid && newid !== orig)
    return s + ' &middot; <span class="ids">original #' + esc(orig) +
           " &rarr; now #" + esc(newid) + "</span> (moved in 2008)";
  return s + ' &middot; <span class="ids">#' + esc(orig) + "</span>" +
         (moved === "no" ? " (original number, never renumbered)" : "");
}

const PHOTO_BASE = "photos";   // relative: works hosted, hides offline

function card(row, idx) {
  if (row.u) {
    const [img, pallet, read, note] = row.u;
    return '<div class="card" data-idx="' + idx + '">' +
           '<div class="insc">' + esc(read || "(unreadable)") + "</div>" +
           '<div class="sub"><span class="chip warn">unofficial &mdash; ' +
           "not in official records</span>" +
           '<span class="chip photo">at pickup site' +
           (pallet ? " &mdash; pallet " + esc(pallet) : "") + "</span>" +
           ' <button class="verify" onclick="togglePanel(this,' + idx +
           ')">verify &#128247;</button></div>' +
           '<div class="sub">Confirmed by a reviewer as present but ' +
           "absent from the lists" +
           (note ? " &middot; note: " + esc(note) : "") + "</div></div>";
  }
  const r = row.r;
  // Display preference: clean digital text (2008 workbook) > the matched
  // photo's OCR read (the engraving itself) > best scan transcription.
  const p = PHOTOS[r[2].toUpperCase() + "|" + (r[1] || r[0])];
  const insc = r[7] || (p && p[3]) || r[6] || "(no inscription recorded)";
  const buyer = r[5] ? "Buyer: " + esc(r[5]) + " &middot; " : "";
  // Staff verification: photographed bricks expand to photo + OCR read +
  // scanned list row, to catch a false-positive match by eye.
  const verify = p ? ' <button class="verify" onclick="togglePanel(this,' +
      idx + ')">verify &#128247;</button>' : "";
  return '<div class="card" data-idx="' + idx + '">' +
         '<div class="insc">' + esc(insc) + "</div>" +
         '<div class="sub">' + chips(r) + verify + "</div>" +
         '<div class="sub">' + buyer + idLine(r) + "</div></div>";
}

let lastHits = [];
function togglePanel(button, idx) {
  const cardEl = button.closest(".card");
  const open = cardEl.querySelector(".panel");
  if (open) { open.remove(); return; }
  const hit = lastHits[idx];
  const panel = document.createElement("div");
  panel.className = "panel";
  if (hit.u) {                 // unofficial: photo only, no list row
    const rel = encodeURI(hit.u[0]).replace(/'/g, "%27");
    panel.innerHTML =
      '<img class="photo" loading="lazy" src="' + PHOTO_BASE + '/thumbs/' +
      rel + '" onclick="magnifyPhoto(\'' + rel + '\')">' +
      '<div class="cap">Photo (click to zoom) &middot; ' +
      "no official list row exists for this brick</div>";
    cardEl.appendChild(panel);
    return;
  }
  const r = hit.r;
  const p = PHOTOS[r[2].toUpperCase() + "|" + (r[1] || r[0])];
  let h = "";
  if (p && p[1]) {
    const rel = encodeURI(p[1]).replace(/'/g, "%27");
    h += '<img class="photo" loading="lazy" src="' + PHOTO_BASE +
         '/thumbs/' + rel + '" onclick="magnifyPhoto(\'' + rel + '\')" ' +
         'onerror="this.style.display=\'none\'">' +
         '<div class="cap">Photo (click to zoom) &middot; OCR read: ' +
         esc(p[3] || "&mdash;") + "</div>";
  }
  h += '<img class="striprow" loading="lazy" src="' + PHOTO_BASE +
       '/strips/' + encodeURIComponent(r[0]) + '.jpg" ' +
       'onclick="magnifyStrip(this)" ' +
       'onerror="this.style.display=\'none\'">' +
       '<div class="cap">Row in the scanned official list ' +
       "(click to magnify)</div>";
  panel.innerHTML = h;
  cardEl.appendChild(panel);
}

function magnify(src, cls) {
  const ov = document.createElement("div");
  ov.id = "zoomstrip";
  const big = document.createElement("img");
  big.className = cls;
  big.src = src;
  ov.appendChild(big);
  ov.onclick = () => ov.remove();
  document.body.appendChild(ov);
}
function magnifyStrip(img) { magnify(img.src, "wide"); }
function magnifyPhoto(rel) {
  // The 2500px zoom image is fetched only here -- never on panel open.
  magnify(PHOTO_BASE + "/zoom/" + rel, "fit");
}

function searchNumber(digits) {
  const hits = [];
  for (const row of ROWS) {
    if (row.r && (row.r[0] === digits || row.r[1] === digits))
      hits.push(row);
  }
  return hits;
}

function searchText(q) {
  const qwords = normalise(q).split(" ").filter(w => w.length >= 2);
  if (!qwords.length) return [];
  const qnorm = normalise(q);
  const scored = [];
  for (const row of ROWS) {
    if (!row.words.length) continue;
    let weighted = 0, wsum = 0;
    for (const qw of qwords) {
      let best = 0;
      for (const rw of row.words) {
        const s = wsim(qw, rw);
        if (s > best) { best = s; if (best === 1) break; }
      }
      weighted += qw.length * best;
      wsum += qw.length;
    }
    let score = weighted / wsum;
    if (row.text.includes(qnorm)) score += 0.15;   // exact phrase boost
    if (score >= 0.72) scored.push([score, row]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  return scored.slice(0, 50).map(x => x[1]);
}

const out = document.getElementById("out");
const meta = document.getElementById("meta");
let timer = null;

function run() {
  const q = document.getElementById("q").value.trim();
  if (!q) { out.innerHTML = ""; meta.textContent = ""; return; }
  const digits = q.replace(/[#\s]/g, "");
  let hits, label;
  if (/^\d+$/.test(digits)) {
    hits = searchNumber(digits);
    label = hits.length + " brick(s) with number " + digits +
            " (both numbering eras searched)";
  } else {
    hits = searchText(q);
    label = hits.length ? "Top " + hits.length + " match(es) of " +
            ROWS.length + " bricks" : "No matches — try fewer words or just the surname";
  }
  meta.textContent = label;
  lastHits = hits;
  out.innerHTML = hits.map((h, i) => card(h, i)).join("");
}

document.getElementById("q").addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(run, 200);
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    // A magnified strip swallows the first Escape; search clears on the next.
    const ov = document.getElementById("zoomstrip");
    if (ov) { ov.remove(); return; }
    const box = document.getElementById("q");
    box.value = ""; box.focus(); run();
  }
});
</script>
</body>
</html>
"""


def _load_master(path: Path) -> list[list[str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            og = row.get("og_inscription", "")
            alt = row.get("og_alt", "")
            display, extra = og, ""
            if alt and _similar(_normalise(alt), _normalise(og)) >= 0.40:
                display, extra = alt, og
            # (an alt that does NOT resemble the parse read the wrong scan
            # row -- neither shown nor indexed, it is another brick's text)
            values = {"og_display": display, "og_extra": extra}
            rows.append([values.get(c, row.get(c, "")) for c in FIELDS])
    return rows


def _load_photos(paths: list[Path]) -> tuple[dict[str, list], list[list]]:
    """Photo data from matched CSVs (pass the APPLIED catalogue,
    pallets_final.csv, when it exists -- it carries review notes).

    Returns (photos, unofficial):
      photos: (SECTION|official_id) -> [pallet, thumb-ready image,
              extra_count, ocr_read, review_note]
      unofficial: [[thumb-ready image, pallet, ocr_read, note], ...] for
              human-confirmed no_match photos -- bricks that physically
              exist at the pickup site but are absent from the official
              records. They become searchable entries of their own.
    """
    photos: dict[str, list] = {}
    unofficial: list[list] = []
    seen_unofficial: set[str] = set()
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                status = row.get("match_status")
                # matched_read is the photo's OCR text -- the engraving
                # itself. Image paths are stored thumb-ready (.jpg suffix,
                # the hostpaths contract shared with make_derivatives).
                jpg = derivative_rel(row.get("image", ""))
                if status == "matched":
                    key = (row.get("official_section", "").upper() + "|"
                           + row.get("official_id", ""))
                    if key in photos:
                        photos[key][2] += 1
                    else:
                        photos[key] = [row.get("pallet", ""), jpg, 0,
                                       row.get("matched_read", ""),
                                       row.get("review_note", "")]
                elif status == "no_match" and jpg not in seen_unofficial:
                    seen_unofficial.add(jpg)
                    unofficial.append([jpg, row.get("pallet", ""),
                                       row.get("matched_read", ""),
                                       row.get("review_note", "")])
    return photos, unofficial


def _json(value) -> str:
    # </script> can never appear in the embedded data.
    return json.dumps(value, ensure_ascii=False,
                      separators=(",", ":")).replace("<", "\\u003c")


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", required=True, type=Path,
                        help="reference/master_list.csv from merge_lists.py")
    parser.add_argument("--matched", type=Path, nargs="*", default=[],
                        help="matched CSV(s) from match.py -- adds "
                             "photographed-on-pallet status per brick")
    parser.add_argument("--output", required=True, type=Path,
                        help="Self-contained .html file to write")
    args = parser.parse_args(argv)

    rows = _load_master(args.master)
    photos, unofficial = _load_photos(args.matched)

    scan_map = {chr(k): v for k, v in _CONFUSABLE.items()}
    stamp = (f"Built {date.today().isoformat()} · {len(rows):,} bricks"
             + (f" · {len(photos):,} confirmed at pickup site" if photos
                else "")
             + (f" · {len(unofficial)} unofficial" if unofficial else "")
             + " · works offline")

    page = (_PAGE
            .replace("__NAVCSS__", NAV_CSS)
            .replace("__NAV__", nav_html("search.html"))
            .replace("__STAMP__", stamp)
            .replace("__DATA__", _json(rows))
            .replace("__PHOTOS__", _json(photos))
            .replace("__UNOFFICIAL__", _json(unofficial))
            .replace("__SCAN__", _json(scan_map))
            .replace("__STOP__", _json(sorted(_STOPWORDS))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output}  ({len(rows):,} bricks, "
          f"{len(photos):,} with photos, {size_mb:.1f} MB)")
    print("Hosting note: keep the stable filenames (search.html, "
          "review.html, fp_review.html) -- the nav bar links them, and "
          "the .htaccess serves HTML no-cache so they never go stale.")


if __name__ == "__main__":
    main()
