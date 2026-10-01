"""Black out the yellow claim tags (handwritten names/phones/emails) in the
City Hall derivatives. Reads derivatives/zoom, writes redacted zoom+thumb to
a staging dir plus contact sheets for eyeballing."""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

src = Path("derivatives/zoom/pallets/City Hall")
stage = Path(sys.argv[1]) / "redacted"
extra = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}
(stage / "zoom").mkdir(parents=True, exist_ok=True); (stage / "thumbs").mkdir(exist_ok=True)
FILL = (70, 70, 70)
files = sorted(src.glob("*.jpg")); report = []
for f in files:
    im = Image.open(f).convert("RGB")
    hsv = np.asarray(im.convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # neon yellow tag: PIL hue 0-255 -> yellow ~ 36-52 (50-73 deg)
    yel = (h >= 31) & (h <= 58)
    def blobs(mask, min_area):
        mask = ndi.binary_closing(mask, iterations=6)
        lab, n = ndi.label(mask)
        if not n:
            return mask, lab, []
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        return mask, lab, [i + 1 for i, a in enumerate(sizes) if a >= min_area]
    # seeds: bright saturated tag yellow (never fires on a brick face)
    sm, slab, sids = blobs(yel & (s >= 85) & (v >= 95), 1200)
    seeds = np.isin(slab, sids)
    # loose: shadowed / washed-out tag pieces -- kept only beside a seed,
    # because engraving paint and lichen on the brick also land here
    lm, llab, lids = blobs(yel & (((s >= 85) & (v >= 60)) | ((s >= 40) & (v >= 200))), 400)
    near = ndi.binary_dilation(seeds, iterations=45)
    touching = set(np.unique(llab[near])) & set(lids)
    keep = seeds | np.isin(llab, list(touching))
    if f.stem == "IMG_0166":   # lichen on the brick face reads as tag yellow
        H_, W_ = keep.shape
        keep[int(.03*H_):int(.9*H_), int(.12*W_):int(.9*W_)] = False
    # grow generously: covers ink strokes, shadowed folds and tag edges
    keep = ndi.binary_dilation(keep, iterations=28)
    keep = ndi.binary_fill_holes(keep)
    arr = np.asarray(im).copy(); arr[keep] = FILL
    out = Image.fromarray(arr)
    d = ImageDraw.Draw(out)
    for box in extra.get(f.stem, []):          # manual boxes, fractions of w/h
        x0, y0, x1, y1 = box
        d.rectangle([x0 * out.width, y0 * out.height, x1 * out.width, y1 * out.height], fill=FILL)
    out.save(stage / "zoom" / f.name, quality=84, optimize=True)
    t = out.copy(); t.thumbnail((640, 10**5), Image.LANCZOS)
    t.save(stage / "thumbs" / f.name, quality=78, optimize=True)
    report.append((f.stem, round(100 * keep.mean(), 1)))
print(" ".join(f"{a[4:]}:{b}" for a, b in report))
W, H, cols, rows = 400, 534, 6, 3
for sidx in range(0, len(files), cols * rows):
    sheet = Image.new("RGB", (W * cols, (H + 20) * rows), "white"); d = ImageDraw.Draw(sheet)
    for i, f in enumerate(files[sidx:sidx + cols * rows]):
        t = Image.open(stage / "zoom" / f.name); t.thumbnail((W, H))
        x, y = (i % cols) * W, (i // cols) * (H + 20)
        sheet.paste(t, (x, y + 20)); d.text((x + 4, y + 4), f.stem, fill="black")
    sheet.save(stage / f"sheet{sidx // (cols*rows) + 1}.jpg", quality=82)
