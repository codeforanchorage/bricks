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
import sys
from pathlib import Path

THUMB_WIDTH = 640      # px; ~40-90 KB/photo -> a few hundred fit in one file
JPEG_QUALITY = 72

# Catalogue columns that are not OCR-method reads (same set as match.py).
_NON_READ = {"image", "brick_id", "x", "y", "w", "h", "status"}

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
 .meta {{ color: #777; font-size: 12px; margin-left: 8px; }}
 .note {{ width: 95%; margin-top: 8px; padding: 6px 8px; font-size: 13px;
         border: 1px solid #ccc; border-radius: 5px; }}
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
function exportCsv() {
  const reviewer = document.getElementById('reviewer').value.trim();
  if (!reviewer) { alert('Please enter your name first.'); return; }
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
  if (undecided &&
      !confirm(undecided + ' photo(s) are still undecided and will be left ' +
               'out. Download anyway?')) return;
  const csv = rows.map(r => r.map(csvField).join(',')).join('\\r\\n');
  const blob = new Blob(['\\ufeff' + csv], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'decisions.csv';
  a.click();
}
document.addEventListener('change', persist);
document.addEventListener('input', persist);
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


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", required=True, type=Path,
                        help="review_*.csv written by match.py")
    parser.add_argument("--photos", required=True, type=Path, nargs="+",
                        help="Director(ies) holding the source photos")
    parser.add_argument("--catalog", type=Path,
                        help="Catalogue CSV (single_pipeline.py output) to "
                             "show each photo's OCR read(s)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Self-contained .html file to write")
    args = parser.parse_args(argv)

    with open(args.review, newline="", encoding="utf-8") as f:
        review = list(csv.DictReader(f))
    if not review:
        raise SystemExit(f"No rows in {args.review} -- nothing to review.")

    # Group candidate rows per (image, brick_id), preserving rank order.
    items: dict[tuple[str, str], list[dict]] = {}
    for row in review:
        items.setdefault((row["image"], row["brick_id"]), []).append(row)
    reads = _load_reads(args.catalog)

    sections, missing_photos = [], []
    for (image, brick_id), candidates in items.items():
        group = f"d:{image}:{brick_id}"
        photo = _find_photo(image, args.photos)
        if photo is None:
            missing_photos.append(image)
            photo_html = f"<p><em>photo {html.escape(image)} not found</em></p>"
        else:
            photo_html = (f'<img src="data:image/jpeg;base64,'
                          f'{_thumb_b64(photo)}" alt="{html.escape(image)}">')

        choices = []
        for cand in candidates:
            label = (f'<b>#{html.escape(cand["official_id"])}</b> '
                     f'section {html.escape(cand["official_section"] or "?")} '
                     f'&mdash; {html.escape(cand["official_name"])}'
                     f'<span class="meta">{html.escape(cand["official_keyword"])}'
                     f' &middot; score {html.escape(cand["score"])}</span>')
            choices.append(_radio(
                group, "match", label,
                id_=cand["official_id"], section=cand["official_section"],
                name=cand["official_name"], keyword=cand["official_keyword"]))
        choices.append(_radio(group, "none",
                              "<b>None of these</b> &mdash; the brick is not "
                              "in this list"))
        choices.append(_radio(group, "illegible",
                              "<b>Can&rsquo;t read the photo</b>"))

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
 </div>
</section>""")

    page = (_PAGE_TOP.format(thumb=THUMB_WIDTH, n_items=len(items))
            + "".join(sections) + _PAGE_BOTTOM)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")

    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output}  ({len(items)} photo(s), {size_mb:.1f} MB)")
    if missing_photos:
        print(f"  WARNING: {len(missing_photos)} photo(s) not found under "
              f"{', '.join(map(str, args.photos))}: "
              f"{', '.join(missing_photos[:5])}")
    print("Send the file to the reviewer; they open it in any browser and "
          "send back decisions.csv.")


if __name__ == "__main__":
    main()
