#!/usr/bin/env python3
"""Build a self-contained review.html for the unmatched-brick queue.

The reclaim pipeline runs on one laptop; the people who can decide the hard
cases (Parks & Rec staff, volunteers) may not have Python, a server, or the
photos. This generator packs everything a reviewer needs into ONE .html file
that opens in any browser, fully offline:

  * each unmatched brick's photo, embedded as a base64 thumbnail
    (EXIF-upright for human eyes -- the OCR pipeline's bytes are untouched),
  * the OCR read(s), and the top candidates from match.py's review file as
    clickable choices, plus "None of these" / "Can't read the photo",
  * an optional note per brick and a reviewer-name box.

Choices autosave in the browser (localStorage), so the page can be closed and
reopened without losing work. When done, one button downloads decisions.csv;
that file comes back by email and apply_decisions.py folds it into the
matched catalogue. Nothing to install on the reviewer's machine.

With --receiver-url (the hosted web/receiver.php) the page ALSO autosaves
every decision to the server a moment after it is made, and sends a final
copy when the reviewer exports -- so a hosted review round needs no email
at all: pull the decisions files from the server's data directory. The
download button keeps working either way.

Embedding photos caps the page at a few hundred items (~40-90 KB each).
For hosted review rounds pass --photo-base-url instead: the page then
references the uploaded derivative trees (make_derivatives.py) -- lazy-
loaded thumbnails that click through to the 2500 px zoom image, which is
exactly what worn bricks need. The page is tiny at any queue size, but
needs the server (or the derivative folders alongside it) to show photos.

With --master (hosted mode only), each candidate additionally shows the
IMAGE of its row in the scanned by-name list (make_strips.py ->
<base>/strips/<orig_id>.jpg): when the list's transcription is mangled,
the reviewer reads the actual print -- granite on the left, scan row on
the right, and the garbled text stops mattering. Candidates with no scan
row (or a missing strip) simply show none.

Usage:
    python make_review_page.py --review output/review_matched.csv \
        --photos test_bricks test_images \
        --catalog output/singles.csv --output output/review.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import sys
from pathlib import Path
from urllib.parse import quote

from hostpaths import derivative_url

THUMB_WIDTH = 640      # px; ~40-90 KB/photo -> a few hundred fit in one file
JPEG_QUALITY = 72

# Catalogue columns that are not OCR-method reads (same set as match.py).
_NON_READ = {"image", "brick_id", "section", "pallet", "x", "y", "w", "h",
             "status"}

_PAGE_TOP = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Town Square brick review</title>
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
 .item {{ background: #fff; margin: 14px auto; max-width: 1080px;
         border-radius: 8px; padding: 14px 18px;
         box-shadow: 0 1px 4px rgba(0,0,0,.12); display: flex; gap: 18px;
         flex-wrap: wrap; }}
 .item.done {{ outline: 3px solid #7fb069; }}
 .photo img {{ width: {thumb}px; max-width: 92vw; border-radius: 6px; }}
 .work {{ flex: 1; min-width: 300px; }}
 .work h2 {{ font-size: 15px; margin: 0 0 6px; }}
 .reads {{ font-size: 13px; color: #555; margin-bottom: 10px; }}
 label.cand {{ display: block; padding: 7px 10px; margin: 4px 0;
              border: 1px solid #ddd; border-radius: 6px; cursor: pointer;
              font-size: 14px; }}
 label.cand:hover {{ background: #f0f6fb; }}
 label.cand input {{ margin-right: 8px; }}
 label.cand img.strip {{ display: block; width: 100%; max-width: 660px;
              margin-top: 5px; border: 1px solid #eee; border-radius: 4px;
              transition: transform .12s ease; }}
 label.cand img.strip:hover {{ transform: scale(2.1);
              transform-origin: left center; position: relative; z-index: 20;
              background: #fff; box-shadow: 0 3px 14px rgba(0,0,0,.35); }}
 .meta {{ color: #777; font-size: 12px; margin-left: 8px; }}
 .note {{ width: 95%; margin-top: 8px; padding: 6px 8px; font-size: 13px;
         border: 1px solid #ccc; border-radius: 5px; }}
 button.clear {{ margin-top: 8px; padding: 4px 12px; font-size: 12.5px;
         border: 1px solid #ccc; border-radius: 5px; background: #f7f7f7;
         cursor: pointer; color: #555; }}
 button.clear:hover {{ background: #fbeaea; border-color: #d99; }}
 h2.divider {{ max-width: 1080px; margin: 26px auto 6px; font-size: 15px;
         color: #5a4a1f; background: #f6ecd4; border-radius: 8px;
         padding: 10px 16px; }}
</style>
</head>
<body>
<header>
 <h1>Town Square brick review &mdash; {n_items} photo(s) need a decision</h1>
 <p>For each photo: click the brick it shows. If none of the choices match,
    pick &ldquo;None of these&rdquo;; if the photo is unreadable, pick
    &ldquo;Can&rsquo;t read the photo&rdquo;. Your choices save automatically
    in this browser &mdash; you can close the page and come back.</p>
 <div id="bar">
  <label>Your name: <input id="reviewer" placeholder="required to export"></label>
  <button onclick="exportCsv()">Download decisions.csv</button>
  <span id="progress"></span>
  <span id="server"></span>
 </div>
</header>
"""

_PAGE_BOTTOM = """
<script>
const KEY = 'brickreview:' + document.title;
function stateLoad() {
  try { return JSON.parse(localStorage.getItem(KEY) || '{}'); }
  catch (e) { return {}; }
}
function stateSave(s) { localStorage.setItem(KEY, JSON.stringify(s)); }

function itemKey(item) {
  return item.dataset.image + '|' + item.dataset.brick;
}
function refresh() {
  let done = 0;
  const items = document.querySelectorAll('.item');
  items.forEach(item => {
    const picked = item.querySelector('input[type=radio]:checked');
    item.classList.toggle('done', !!picked);
    if (picked) done++;
  });
  document.getElementById('progress').textContent =
    done + ' of ' + items.length + ' decided';
}
function persist() {
  const s = stateLoad();
  document.querySelectorAll('.item').forEach(item => {
    const picked = item.querySelector('input[type=radio]:checked');
    s[itemKey(item)] = {
      v: picked ? picked.value : '',
      n: item.querySelector('.note').value,
    };
  });
  s['reviewer'] = document.getElementById('reviewer').value;
  stateSave(s);
  refresh();
}
function restore() {
  const s = stateLoad();
  document.getElementById('reviewer').value = s['reviewer'] || '';
  document.querySelectorAll('.item').forEach(item => {
    const saved = s[itemKey(item)];
    if (!saved) return;
    if (saved.v) {
      const radio = item.querySelector('input[value="' + saved.v + '"]');
      if (radio) radio.checked = true;
    }
    item.querySelector('.note').value = saved.n || '';
  });
  refresh();
}
function csvField(text) {
  return '"' + String(text ?? '').replace(/"/g, '""') + '"';
}
function buildCsv(reviewer) {
  const rows = [['reviewer', 'image', 'brick_id', 'decision', 'official_id',
                 'official_section', 'official_name', 'official_keyword',
                 'note']];
  let undecided = 0;
  document.querySelectorAll('.item').forEach(item => {
    const picked = item.querySelector('input[type=radio]:checked');
    if (!picked) { undecided++; return; }
    rows.push([reviewer, item.dataset.image, item.dataset.brick,
               picked.dataset.decision, picked.dataset.id || '',
               picked.dataset.section || '', picked.dataset.name || '',
               picked.dataset.keyword || '',
               item.querySelector('.note').value]);
  });
  return {csv: rows.map(r => r.map(csvField).join(',')).join('\\r\\n'),
          undecided: undecided, decided: rows.length - 1};
}

const RECEIVER = __RECEIVER__;   // null, or {url, token}
let uploadTimer = null;
function serverNote(text) {
  document.getElementById('server').textContent = text;
}
function uploadCsv(final) {
  if (!RECEIVER) return;
  const reviewer = document.getElementById('reviewer').value.trim();
  const built = buildCsv(reviewer || 'anon');
  if (!built.decided) return;
  const url = RECEIVER.url + '?token=' + encodeURIComponent(RECEIVER.token) +
              '&label=' + encodeURIComponent(reviewer || 'anon') +
              '&final=' + (final ? '1' : '0');
  fetch(url, {method: 'POST', headers: {'Content-Type': 'text/csv'},
              body: built.csv})
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(() => serverNote((final ? 'sent to server ' : 'saved to server ') +
                           new Date().toLocaleTimeString()))
    .catch(() => serverNote('server unreachable -- your work is safe in ' +
                            'this browser; use the download button'));
}
function scheduleUpload() {
  if (!RECEIVER) return;
  clearTimeout(uploadTimer);
  uploadTimer = setTimeout(() => uploadCsv(false), 2500);
}

function exportCsv() {
  const reviewer = document.getElementById('reviewer').value.trim();
  if (!reviewer) { alert('Please enter your name first.'); return; }
  const built = buildCsv(reviewer);
  if (built.undecided &&
      !confirm(built.undecided + ' photo(s) are still undecided and will be ' +
               'left out. Download anyway?')) return;
  const blob = new Blob(['\\ufeff' + built.csv], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'decisions.csv';
  a.click();
  uploadCsv(true);
}
function clearItem(button) {
  // Radios can't be un-clicked natively; this un-decides the photo. The
  // next autosave sends the corrected state (undecided rows are omitted).
  const item = button.closest('.item');
  item.querySelectorAll('input[type=radio]').forEach(r => r.checked = false);
  persist();
  scheduleUpload();
}
document.addEventListener('change', () => { persist(); scheduleUpload(); });
document.addEventListener('input', () => { persist(); scheduleUpload(); });
restore();
</script>
</body>
</html>
"""


def _thumb_b64(photo: Path) -> str:
    """Photo -> base64 JPEG thumbnail, EXIF-upright for human eyes."""
    from PIL import Image, ImageOps

    if photo.suffix.lower() == ".heic":
        import pillow_heif
        pillow_heif.register_heif_opener()
    with Image.open(photo) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        if img.width > THUMB_WIDTH:
            scale = THUMB_WIDTH / img.width
            img = img.resize((THUMB_WIDTH, round(img.height * scale)),
                             Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _find_photo(name: str, photo_dirs: list[Path]) -> Path | None:
    for directory in photo_dirs:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _hosted_photo_html(image: str, base_url: str) -> str:
    """Lazy thumbnail linking to the zoom derivative (see hostpaths.py)."""
    alt = html.escape(image, quote=True)
    return (f'<a href="{derivative_url(base_url, "zoom", image)}" '
            f'target="_blank">'
            f'<img loading="lazy" '
            f'src="{derivative_url(base_url, "thumbs", image)}" '
            f'alt="{alt}" title="click for full size"></a>')


def _load_strip_map(master: Path | None) -> dict[tuple[str, str], str]:
    """(SECTION, official id) -> original brick number, for strip lookups.

    The strips are named by ORIGINAL number (the by-name scan's numbering);
    candidates reference the master's assigned id (new_id when moved).
    """
    if master is None:
        return {}
    strip_map = {}
    with open(master, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            orig = row.get("orig_id", "")
            key = (row.get("section", "").upper(),
                   row.get("new_id") or orig)
            if orig:
                strip_map[key] = orig
    return strip_map


def _load_reads(catalog: Path | None) -> dict[tuple[str, str], str]:
    """(image, brick_id) -> 'method: read / method: read' from the catalogue."""
    if catalog is None:
        return {}
    reads: dict[tuple[str, str], str] = {}
    with open(catalog, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parts = [f"{col}: {row[col]}" for col in row
                     if col not in _NON_READ and row.get(col)
                     and not row[col].startswith("ERROR:")]
            reads[(row.get("image", ""), str(row.get("brick_id", "")))] = \
                " | ".join(parts)
    return reads


def _radio(group: str, decision: str, label_html: str, *,
           id_="", section="", name="", keyword="") -> str:
    value = html.escape(f"{decision}:{id_}:{section}", quote=True)
    attrs = (f'data-decision="{html.escape(decision, quote=True)}" '
             f'data-id="{html.escape(id_, quote=True)}" '
             f'data-section="{html.escape(section, quote=True)}" '
             f'data-name="{html.escape(name, quote=True)}" '
             f'data-keyword="{html.escape(keyword, quote=True)}"')
    return (f'<label class="cand"><input type="radio" '
            f'name="{html.escape(group, quote=True)}" value="{value}" '
            f'{attrs}>{label_html}</label>')


def _collect_items(args) -> tuple[list, dict[str, str]]:
    """The review queue as (ordered items, photo_types).

    Items group candidate rows per (image, brick_id) in rank order, plus
    unmatched photos with NO candidates from --matched (blank reads,
    stack shots -- otherwise they never appear anywhere a human looks).
    Stack-labelled photos sort to the bottom.
    """
    with open(args.review, newline="", encoding="utf-8") as f:
        review = list(csv.DictReader(f))
    if not review:
        raise SystemExit(f"No rows in {args.review} -- nothing to review.")

    items: dict[tuple[str, str], list[dict]] = {}
    for row in review:
        items.setdefault((row["image"], row["brick_id"]), []).append(row)

    if args.matched:
        with open(args.matched, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("image", ""), str(row.get("brick_id", "")))
                if (row.get("match_status") != "matched" and key[0]
                        and key not in items):
                    items[key] = []

    photo_types: dict[str, str] = {}
    if args.photo_types and args.photo_types.is_file():
        with open(args.photo_types, newline="", encoding="utf-8") as f:
            photo_types = {r["image"]: r["label"]
                           for r in csv.DictReader(f)}

    ordered = sorted(items.items(),
                     key=lambda kv: photo_types.get(kv[0][0]) == "stack")
    return ordered, photo_types


def _candidate_choices(group: str, candidates: list[dict], is_stack: bool,
                       hosted_base: str, strip_map: dict) -> list[str]:
    """The radio choices for one photo, in display order."""
    choices = []
    for cand in candidates:
        label = (f'<b>#{html.escape(cand["official_id"])}</b> '
                 f'section {html.escape(cand["official_section"] or "?")} '
                 f'&mdash; {html.escape(cand["official_name"])}'
                 f'<span class="meta">{html.escape(cand["official_keyword"])}'
                 f' &middot; score {html.escape(cand["score"])}</span>')
        orig = strip_map.get((cand["official_section"].upper(),
                              cand["official_id"]))
        if orig:
            # The actual printed row from the scanned list; a missing
            # strip (colliding number, preamble row) hides itself.
            label += (f'<img class="strip" loading="lazy" '
                      f'src="{hosted_base}/strips/{quote(orig)}.jpg" '
                      f'alt="scanned list row #{html.escape(orig)}" '
                      f'onerror="this.style.display=\'none\'">')
        choices.append(_radio(
            group, "match", label,
            id_=cand["official_id"], section=cand["official_section"],
            name=cand["official_name"], keyword=cand["official_keyword"]))
    if is_stack:
        # Overview shots get a one-click confirmation FIRST -- but the
        # candidates stay: the classifier sometimes labels a photo
        # 'stack' when one brick fills the centre with stacks behind
        # it, and that brick still needs its match options.
        choices.insert(0, _radio(
            group, "stack",
            "<b>Stack / pallet overview photo</b> &mdash; "
            "not an individual brick"))
    choices.append(_radio(group, "none",
                          "<b>None of these</b> &mdash; if you can read "
                          "any words on the brick, write them in the "
                          "note below"))
    choices.append(_radio(group, "illegible",
                          "<b>Can&rsquo;t read the photo</b>"))
    return choices


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", required=True, type=Path,
                        help="review_*.csv written by match.py")
    parser.add_argument("--photos", type=Path, nargs="+", default=[],
                        help="Director(ies) holding the source photos "
                             "(required unless --photo-base-url is used)")
    parser.add_argument("--photo-base-url", default="",
                        help="Base URL of the uploaded derivative trees "
                             "(make_derivatives.py): reference hosted "
                             "thumbs/zoom images instead of embedding -- "
                             "keeps the page small at any queue size")
    parser.add_argument("--master", type=Path,
                        help="reference/master_list.csv -- show each "
                             "candidate's scanned list-row image "
                             "(<base>/strips/, from make_strips.py) so the "
                             "reviewer reads the print, not the OCR of it. "
                             "Hosted mode only.")
    parser.add_argument("--photo-types", type=Path,
                        help="classify_photos.py output: photos labelled "
                             "'stack' (pallet/stack overviews, not "
                             "individual bricks) sort into their own "
                             "section at the bottom with a one-click "
                             "'stack photo' decision")
    parser.add_argument("--matched", type=Path,
                        help="matched CSV: also include unmatched photos "
                             "that have NO candidates (blank reads, stack "
                             "shots) -- otherwise they never appear "
                             "anywhere a human looks")
    parser.add_argument("--catalog", type=Path,
                        help="Catalogue CSV (single_pipeline.py output) to "
                             "show each photo's OCR read(s)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Self-contained .html file to write")
    parser.add_argument("--receiver-url", default="",
                        help="URL of the hosted receiver.php; adds autosave-"
                             "to-server and send-on-export to the page")
    parser.add_argument("--receiver-token", default="",
                        help="Shared token, must match TOKEN in receiver.php")
    args = parser.parse_args(argv)
    if bool(args.receiver_url) != bool(args.receiver_token):
        raise SystemExit("--receiver-url and --receiver-token go together")
    if not args.photos and not args.photo_base_url:
        raise SystemExit("need --photos (embedded thumbnails) or "
                         "--photo-base-url (hosted derivatives)")

    ordered, photo_types = _collect_items(args)
    reads = _load_reads(args.catalog)

    hosted_base = args.photo_base_url.rstrip("/")
    strip_map = _load_strip_map(args.master) if hosted_base else {}
    sections, missing_photos = [], []
    stack_divider_done = False
    for (image, brick_id), candidates in ordered:
        is_stack = photo_types.get(image) == "stack"
        if is_stack and not stack_divider_done:
            sections.append(
                '<h2 class="divider">Pallet &amp; stack photos &mdash; '
                'overview shots, not individual bricks. Confirm each with '
                'one click (or override if one is actually a brick '
                'close-up).</h2>')
            stack_divider_done = True
        group = f"d:{image}:{brick_id}"
        if hosted_base:
            photo_html = _hosted_photo_html(image, hosted_base)
        else:
            photo = _find_photo(image, args.photos)
            if photo is None:
                missing_photos.append(image)
                photo_html = (f"<p><em>photo {html.escape(image)} "
                              f"not found</em></p>")
            else:
                photo_html = (f'<img src="data:image/jpeg;base64,'
                              f'{_thumb_b64(photo)}" alt="{html.escape(image)}">')

        choices = _candidate_choices(group, candidates, is_stack,
                                     hosted_base, strip_map)

        read_line = reads.get((image, str(brick_id)), "")
        sections.append(f"""
<section class="item" data-image="{html.escape(image, quote=True)}"
         data-brick="{html.escape(str(brick_id), quote=True)}">
 <div class="photo">{photo_html}</div>
 <div class="work">
  <h2>{html.escape(image)}</h2>
  <div class="reads">{('OCR read &mdash; ' + html.escape(read_line))
                      if read_line else ''}</div>
  {''.join(choices)}
  <input class="note" placeholder="note (optional)">
  <button type="button" class="clear" onclick="clearItem(this)">Clear
   choice</button>
 </div>
</section>""")

    receiver = ({"url": args.receiver_url, "token": args.receiver_token}
                if args.receiver_url else None)
    page = (_PAGE_TOP.format(thumb=THUMB_WIDTH, n_items=len(ordered))
            + "".join(sections)
            + _PAGE_BOTTOM.replace("__RECEIVER__", json.dumps(receiver)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")

    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output}  ({len(ordered)} photo(s), {size_mb:.1f} MB)")
    if missing_photos:
        print(f"  WARNING: {len(missing_photos)} photo(s) not found under "
              f"{', '.join(map(str, args.photos))}: "
              f"{', '.join(missing_photos[:5])}")
    print("Send the file to the reviewer; they open it in any browser and "
          "send back decisions.csv.")


if __name__ == "__main__":
    main()
