"""Split a brick photo into overlapping tiles for higher-resolution OCR.

A single wide photo spreads its pixels across dozens of bricks, and the
Anthropic vision API further caps each image at ~1.15 MP. Tiling crops the
photo into an R x C grid and OCRs each tile separately, so every brick is read
at much higher effective resolution (and each tile is a separate, full-detail
API call for Haiku).

Tiles overlap so a brick straddling a cut still appears whole in at least one
tile. That overlap makes boundary bricks get detected twice, so the pipeline
de-duplicates afterwards:

  - PaddleOCR detections carry boxes -> de-dup by spatial overlap.
  - Haiku inscriptions have no boxes -> de-dup by text similarity.

Detections found in a tile map back to full-image coordinates by adding the
tile's (x_offset, y_offset).
"""
from __future__ import annotations

import difflib
from pathlib import Path

from PIL import Image


def plan_tiles(width: int, height: int, rows: int, cols: int,
               overlap: float) -> list[tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) rectangles for an R x C overlapping grid."""
    tile_w, tile_h = width / cols, height / rows
    pad_x, pad_y = tile_w * overlap, tile_h * overlap
    rects = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(round(c * tile_w - pad_x)))
            y0 = max(0, int(round(r * tile_h - pad_y)))
            x1 = min(width, int(round((c + 1) * tile_w + pad_x)))
            y1 = min(height, int(round((r + 1) * tile_h + pad_y)))
            rects.append((x0, y0, x1, y1))
    return rects


def split(image_path, rows: int, cols: int, overlap: float, dest_dir):
    """Crop an image into rows*cols overlapping JPEG tiles inside dest_dir.

    Returns (tiles, image_size) where `tiles` is a list of
    (tile_path, x0, y0, x1, y1) -- the tile rectangle in full-image
    coordinates -- and `image_size` is (width, height).
    """
    dest = Path(dest_dir)
    out = []
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        size = (img.width, img.height)
        rects = plan_tiles(img.width, img.height, rows, cols, overlap)
        for i, (x0, y0, x1, y1) in enumerate(rects):
            tile_path = dest / f"tile_{i:03d}.jpg"
            img.crop((x0, y0, x1, y1)).save(tile_path, "JPEG", quality=95)
            out.append((tile_path, x0, y0, x1, y1))
    return out, size


def drop_clipped(detections: list[dict], tile_rect: tuple,
                 image_size: tuple, margin: float = 3.0) -> list[dict]:
    """Drop detections touching a tile edge that is interior to the image.

    A detection whose tile-local box runs into an interior cut was probably
    sliced by it; the overlapping neighbour tile holds a complete, centred
    copy. Edges lying on the true image border are kept (nothing is beyond
    them). `tile_rect` is the tile's full-image rectangle; detection boxes are
    tile-local. Run this before offsetting boxes to full-image coordinates.
    """
    tx0, ty0, tx1, ty1 = tile_rect
    img_w, img_h = image_size
    tile_w, tile_h = tx1 - tx0, ty1 - ty0
    drop_left, drop_top = tx0 > 0, ty0 > 0
    drop_right, drop_bottom = tx1 < img_w, ty1 < img_h

    kept = []
    for det in detections:
        box = det.get("box")
        if box is None:
            kept.append(det)
            continue
        x0, y0, x1, y1 = box
        clipped = ((drop_left and x0 <= margin)
                   or (drop_top and y0 <= margin)
                   or (drop_right and x1 >= tile_w - margin)
                   or (drop_bottom and y1 >= tile_h - margin))
        if not clipped:
            kept.append(det)
    return kept


def _overlap_ratio(a: tuple, b: tuple) -> float:
    """Intersection area as a fraction of the smaller box's area."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _box_area(box: tuple) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def dedup_detections(detections: list[dict], overlap_thresh: float = 0.6) -> list[dict]:
    """Drop PaddleOCR detections that overlap-region duplication produced.

    Greedy NMS over boxes in full-image coordinates: process detections
    largest-box-first (score breaks ties) and skip any whose box mostly
    coincides with an already-kept one. Largest-first matters because a tile
    edge can catch a *partial* brick -- keeping the bigger box preserves the
    complete detection ("MARY & TOM") over a clipped fragment ("TOM").
    """
    boxed = [d for d in detections if d.get("box") is not None]
    boxless = [d for d in detections if d.get("box") is None]
    boxed.sort(key=lambda d: (_box_area(d["box"]), d.get("raw_score") or 0.0),
               reverse=True)

    kept: list[dict] = []
    for det in boxed:
        if all(_overlap_ratio(det["box"], k["box"]) < overlap_thresh
               for k in kept):
            kept.append(det)
    return kept + boxless


_CONF_RANK = {"high": 0, "medium": 1, "low": 2}


def dedup_inscriptions(items: list[dict], similarity: float = 0.85) -> list[dict]:
    """Drop near-duplicate Haiku inscriptions from overlapping tiles.

    Keeps the highest-confidence copy; compares normalised text with difflib.
    """
    ordered = sorted(items, key=lambda d: _CONF_RANK.get(d.get("confidence"), 3))
    kept: list[dict] = []
    kept_norms: list[str] = []
    for item in ordered:
        norm = " ".join(item["inscription"].lower().split())
        if any(difflib.SequenceMatcher(None, norm, prev).ratio() >= similarity
               for prev in kept_norms):
            continue
        kept.append(item)
        kept_norms.append(norm)
    return kept
