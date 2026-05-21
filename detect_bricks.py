#!/usr/bin/env python3
"""Classical-CV brick detector.

Finds individual paver rectangles by reconstructing the mortar-joint grid and
taking the brick faces as the cells between the lines:

  1. downscale, grayscale, blur
  2. black-hat morphology -> highlights dark thin structures (joints + speckle)
  3. directional morphology -> keep only long horizontal/vertical runs (the
     joint lines); the weathered-surface speckle is discarded
  4. brick faces = NOT grid; connected components -> candidate bricks
  5. filter by size / aspect / rectangularity, skipping partial bricks that
     are cut off by the image border

Writes two images for inspection: `_joints` (the reconstructed grid) and
`_detected` (the original with detected bricks boxed). Run it, eyeball, tune.

Usage:
    python detect_bricks.py --image test_images/IMG_7115.jpeg --outdir output/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

WORK_WIDTH = 1600       # downscale width for detection
JOINT_KERNEL = 51       # black-hat kernel -- must exceed the joint width
H_LINE = 90             # min horizontal run (px) to count as a joint line
V_LINE = 55             # min vertical run (px) to count as a joint line
GRID_DILATE = 7         # thicken / bridge gaps in the reconstructed grid
FACE_ERODE = 17         # shrink brick faces so neighbours separate cleanly
MIN_SIDE = 35           # ignore components thinner than this (work px)
EDGE_MARGIN = 14        # a brick within this of the image border is partial
AREA_LO, AREA_HI = 0.30, 4.0   # keep components within this x median area
MAX_ASPECT = 6.0        # longest:shortest side
MIN_EXTENT = 0.55       # area / bounding-box area (a brick fills its rect)


def detect(image_path: Path):
    """Return (original, work_image, grid_mask, bricks) -- bricks are (x,y,w,h)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    scale = WORK_WIDTH / w
    work = cv2.resize(img, (WORK_WIDTH, round(h * scale)),
                      interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # Black-hat lights up dark structures narrower than the kernel -- the
    # mortar joints, but also a lot of weathered-surface speckle.
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (JOINT_KERNEL, JOINT_KERNEL))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    _, raw = cv2.threshold(blackhat, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep only long horizontal / vertical runs: real joint lines survive,
    # isolated speckle does not. Their union is the joint grid.
    horiz = cv2.morphologyEx(
        raw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (H_LINE, 1)))
    vert = cv2.morphologyEx(
        raw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, V_LINE)))
    grid = cv2.bitwise_or(horiz, vert)
    grid = cv2.dilate(
        grid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (GRID_DILATE, GRID_DILATE)))

    # Brick faces are the cells between the joint lines; erode so touching
    # bricks split into separate connected components.
    faces = cv2.bitwise_not(grid)
    faces = cv2.erode(
        faces, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (FACE_ERODE, FACE_ERODE)))

    n, _, stats, _ = cv2.connectedComponentsWithStats(faces, connectivity=4)
    work_h, work_w = work.shape[:2]
    candidates = []
    for x, y, bw, bh, area in stats[1:]:  # skip the background label
        if bw < MIN_SIDE or bh < MIN_SIDE:
            continue
        # Skip partial bricks cut by the image border -- their crops are
        # incomplete; a whole copy appears in an overlapping photo.
        if (x <= EDGE_MARGIN or y <= EDGE_MARGIN
                or x + bw >= work_w - EDGE_MARGIN
                or y + bh >= work_h - EDGE_MARGIN):
            continue
        candidates.append((x, y, bw, bh, area))
    if not candidates:
        return img, work, grid, []

    median_area = float(np.median([c[4] for c in candidates]))
    bricks = []
    for x, y, bw, bh, area in candidates:
        if not (AREA_LO * median_area <= area <= AREA_HI * median_area):
            continue
        if max(bw, bh) / max(min(bw, bh), 1) > MAX_ASPECT:
            continue
        if area / float(bw * bh) < MIN_EXTENT:
            continue
        bricks.append((int(x), int(y), int(bw), int(bh)))
    return img, work, grid, bricks


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path,
                        help="Brick image to detect bricks in")
    parser.add_argument("--outdir", required=True, type=Path,
                        help="Directory for the _joints and _detected images")
    args = parser.parse_args(argv)

    img, work, grid, bricks = detect(args.image)
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    joints_path = args.outdir / f"{stem}_joints.jpg"
    cv2.imwrite(str(joints_path), grid)

    viz = work.copy()
    for idx, (x, y, bw, bh) in enumerate(bricks, 1):
        cv2.rectangle(viz, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(viz, str(idx), (x + 5, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    viz_path = args.outdir / f"{stem}_detected.jpg"
    cv2.imwrite(str(viz_path), viz)

    print(f"{args.image.name}: detected {len(bricks)} brick(s)")
    print(f"  joints   -> {joints_path}")
    print(f"  detected -> {viz_path}")


if __name__ == "__main__":
    main()
