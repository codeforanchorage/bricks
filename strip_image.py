#!/usr/bin/env python3
"""Strip image cleanup: deskew + ink-valley trimming.

The scanned list's row crops come from its embedded OCR text boxes, which
under-report descenders and sit on skewed baselines -- so a horizontal
band clipped at neighbour midpoints both bleeds neighbouring rows in AND
cuts the target row's ascenders/descenders off (measured: the same crop
feeds the models and the review page, so both suffered).

clean_strip() fixes both with one strategy: crop a deliberately
over-generous band (bleed is fine at this stage), straighten it, then cut
at the whitespace valleys the actual INK defines:

  1. expand the tight band by one band-height on each side
  2. deskew: search small rotations for the angle that maximises the
     sharpness (variance) of the horizontal ink profile -- a straight row
     of type gives a spiky profile, a tilted one smears it
  3. trim: find contiguous ink bands in the profile, keep the band whose
     centre lies closest to where the target row is expected, cut in the
     valleys around it with a few pixels of padding

The embedded text boxes only need to say roughly where the row is; the
ink decides the exact edges. Rows keep their descenders, neighbours
disappear, and characters reach the model upright. On any degenerate
input (blank band, no valleys) the original tight crop is returned
unchanged -- never worse than the old behaviour.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

INK_THRESHOLD = 160    # gray below this counts as ink (typewritten on white)
MAX_SKEW_DEG = 1.6     # scanner tilt observed is well under a degree
PAD_PX = 3             # breathing room kept beyond the ink band's edges
_WHITE = (255, 255, 255)


def _profile(image: Image.Image) -> np.ndarray:
    """Dark-pixel count per pixel row."""
    gray = np.asarray(image.convert("L"))
    return (gray < INK_THRESHOLD).sum(axis=1)


def _deskew(band: Image.Image) -> Image.Image:
    """Rotate by the small angle that makes the ink profile sharpest."""
    # Score on a width-reduced copy: rotation cost drops 4x and the
    # profile's variance ranks angles just as well.
    small = band.resize((max(64, band.width // 4), band.height))

    def sharpness(angle: float) -> float:
        img = (small.rotate(angle, resample=Image.BILINEAR, fillcolor=_WHITE)
               if angle else small)
        p = _profile(img).astype(float)
        return float(((p - p.mean()) ** 2).sum())

    best_angle, best_score = 0.0, sharpness(0.0)
    for step, span in ((0.4, MAX_SKEW_DEG), (0.08, 0.4)):
        centre = best_angle
        angle = centre - span
        while angle <= centre + span + 1e-9:
            a = round(angle, 3)
            if abs(a) <= MAX_SKEW_DEG and a != best_angle:
                score = sharpness(a)
                if score > best_score:
                    best_angle, best_score = a, score
            angle += step
    if best_angle:
        return band.rotate(best_angle, resample=Image.BICUBIC,
                           fillcolor=_WHITE)
    return band


def _trim(band: Image.Image, centre_frac: float,
          pitch: int) -> Image.Image | None:
    """Cut the band down to the ink band nearest the expected row centre.

    `pitch` is the expected single-row height in pixels (the tight crop's
    height). On tightly-leaded pages the gaps between printed rows never
    drop below the valley threshold, so adjacent rows fuse into one ink
    band -- measured on 6% of the list's strips (816/12,828, e.g. brick
    2854 shipping with three rows). When the chosen band is much taller
    than one row, cut it at the weakest points of the ink profile just
    above and below the expected row.
    """
    prof = _profile(band)
    if not prof.any():
        return None
    smooth = np.convolve(prof, np.ones(3) / 3, mode="same")
    ink = smooth > max(2.0, 0.03 * float(smooth.max()))

    bands: list[list[int]] = []
    for i, is_ink in enumerate(ink):
        if is_ink and (not bands or bands[-1][1] is not None):
            bands.append([i, None])
        elif not is_ink and bands and bands[-1][1] is None:
            bands[-1][1] = i
    if bands and bands[-1][1] is None:
        bands[-1][1] = len(ink)
    if not bands:
        return None
    # Specks can split one printed row; re-join bands separated by <=2 px.
    merged = [bands[0]]
    for start, end in bands[1:]:
        if start - merged[-1][1] <= 2:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    target_y = centre_frac * band.height
    start, end = min(merged, key=lambda b: abs((b[0] + b[1]) / 2 - target_y))
    pad_top = pad_bottom = PAD_PX

    if end - start > 1.55 * pitch:
        # Fused rows: cut at the weakest ink minimum in the windows where
        # the gaps above and below the target row should be. A forced cut
        # IS the boundary -- padding across it would re-import the very
        # neighbour sliver the cut removed.
        def weakest(lo: float, hi: float, default: int) -> int:
            lo_i, hi_i = max(start, int(lo)), min(end, int(hi))
            if hi_i - lo_i < 2:
                return default
            # Raw profile, not smoothed: smoothing spreads a 1px gap over
            # its neighbours and argmin can tie onto a bar row.
            window = prof[lo_i:hi_i]
            return lo_i + int(np.argmin(window))

        cut_top = weakest(target_y - 0.9 * pitch, target_y - 0.35 * pitch,
                          start)
        cut_bottom = weakest(target_y + 0.35 * pitch, target_y + 0.9 * pitch,
                             end)
        if cut_top != start:
            start, pad_top = cut_top, 0
        if cut_bottom != end:
            end, pad_bottom = cut_bottom, 0
        if end <= start:
            return None

    return band.crop((0, max(0, start - pad_top), band.width,
                      min(band.height, end + pad_bottom)))


def clean_strip(page_image: Image.Image, y0: int, y1: int) -> Image.Image:
    """Cleaned single-row strip for the tight pixel band [y0, y1).

    Returns the deskewed, ink-trimmed strip; falls back to the plain
    tight crop whenever the cleanup cannot find a usable ink band.
    """
    height = max(1, y1 - y0)
    ey0 = max(0, y0 - height)
    ey1 = min(page_image.height, y1 + height)
    band = _deskew(page_image.crop((0, ey0, page_image.width, ey1)))
    centre_frac = ((y0 + y1) / 2 - ey0) / max(1, ey1 - ey0)
    trimmed = _trim(band, centre_frac, height)
    # Slivers (speck-only bands) are as useless as fused rows: fall back.
    if trimmed is None or trimmed.height < max(6, 0.35 * height):
        return page_image.crop((0, y0, page_image.width, y1))
    return trimmed
