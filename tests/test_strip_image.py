"""Tests for strip_image.clean_strip: deskew + ink-valley trimming.

Synthetic pages: white background, black bars as printed rows. The tight
crop band is deliberately misplaced the way the scan's text boxes misplace
it -- clipping the target row's descender and bleeding the neighbour in --
and clean_strip must recover the whole row alone.
"""
import numpy as np
from PIL import Image, ImageDraw

from strip_image import INK_THRESHOLD, clean_strip


def _page(rows_y, skew_deg=0.0, size=(1200, 200)):
    """A page with a black bar per row y; middle row gets a 'descender'."""
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, y in enumerate(rows_y):
        draw.rectangle([40, y, size[0] - 40, y + 16], fill=(20, 20, 20))
        if i == 1:  # descender tail below the middle row's body
            draw.rectangle([300, y + 16, 320, y + 22], fill=(20, 20, 20))
    if skew_deg:
        img = img.rotate(skew_deg, resample=Image.BICUBIC,
                         fillcolor=(255, 255, 255))
    return img


def _ink_rows(strip):
    gray = np.asarray(strip.convert("L"))
    return (gray < INK_THRESHOLD).sum(axis=1)


def _bands(strip):
    """Count contiguous ink bands (rows of dark pixels) in the strip."""
    ink = _ink_rows(strip) > 2
    return int(np.diff(np.concatenate(([0], ink.view(np.int8),
                                       [0]))).clip(min=0).sum())


def test_trim_recovers_descender_and_drops_neighbours():
    page = _page([40, 90, 140])
    # Tight band misplaced the way text boxes misplace it: starts inside
    # the neighbour above, ends BEFORE the descender (y 106..112).
    strip = clean_strip(page, 60, 104)
    ink = _ink_rows(strip)
    assert ink.any()
    assert _bands(strip) == 1                 # neighbours gone
    # The descender's narrow tail is present: some rows have a small,
    # non-zero ink count (the 20px tail), distinct from the body's width.
    tail_rows = ((ink > 0) & (ink < 100)).sum()
    assert tail_rows >= 3                     # descender recovered
    assert strip.height < 50                  # a single row, not the band


def test_skewed_row_is_straightened():
    page = _page([40, 90, 140], skew_deg=1.0)
    strip = clean_strip(page, 60, 116)
    assert _bands(strip) == 1
    # A 1-degree tilt across 1120px smears the bar over ~45 rows; once
    # deskewed and trimmed the strip lands near the bar's true height
    # (16px bar + 6px descender + padding, with a little residual).
    assert strip.height <= 38


def test_fused_rows_are_split_at_weakest_gap():
    # Tightly-leaded pages: rows so close the valley threshold never
    # triggers (1px gaps), fusing 3 rows into one ink band -- the brick
    # 2854 case. The pitch-aware cut must recover just the middle row.
    img = Image.new("RGB", (1200, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in (30, 48, 66):                       # 17px bars, 1px gaps
        draw.rectangle([40, y, 1160, y + 16], fill=(20, 20, 20))
    strip = clean_strip(img, 46, 66)             # tight band = middle row
    assert strip.height <= 30                    # one row, not three
    ink = _ink_rows(strip)
    assert ink.any()
    assert _bands(strip) == 1


def test_blank_band_falls_back_to_original_crop():
    page = Image.new("RGB", (800, 200), (255, 255, 255))
    strip = clean_strip(page, 60, 100)
    assert strip.size == (800, 40)            # exact old-behaviour crop
