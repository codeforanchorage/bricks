"""Group PaddleOCR text detections into individual bricks.

PaddleOCR returns each detected text region separately and has no concept of a
"brick", so a 3-line brick inscription comes back as 3 unrelated detections.
This module reassembles them by spatial layout, in two passes:

  1. Line assembly  -- merge detections that sit on the same horizontal line
                       (insurance for when detection splits a line into words).
  2. Brick assembly -- stack lines that are vertically adjacent AND horizontally
                       aligned into one brick (a brick is normally 1-3 lines).

Both passes connect detections with a union-find over every pair, so a cluster
forms transitively (word A-B close, B-C close => A,B,C are one line).

THRESHOLDS are expressed as multiples of the median detected text height, so
they scale with image resolution. The defaults work on the synthetic test
grid; they will likely need tuning against real brick photos -- the brick-stack
gap is the most layout-sensitive one and is exposed on the CLI as --brick-gap.
"""
from __future__ import annotations

import statistics

# --- Tunable thresholds (multiples of the median text height) --------------

# Pass 1: two same-row detections merge into one line when the horizontal gap
# between them is below this. A normal inter-word space is ~0.3-0.5x text
# height; the gap between two side-by-side bricks is much larger.
LINE_MERGE_GAP = 1.0

# Pass 2: two lines stack into the same brick when the vertical gap between
# them is below this. Engraved lines within a brick sit tight together; the
# gap across a mortar joint (plus each brick's blank margin) is larger.
BRICK_STACK_GAP = 1.5

# Pass 1: minimum vertical-overlap fraction for two detections to be "same row".
SAME_ROW_OVERLAP = 0.4

# Pass 2: minimum horizontal-overlap fraction for two lines to share a brick.
# Lines of a brick are left-aligned or centred, so their x-ranges overlap a lot.
SAME_BRICK_OVERLAP = 0.25


# --- Geometry helpers ------------------------------------------------------
# A box is a 4-tuple (x0, y0, x1, y1).

def _span_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the overlap between intervals [a0,a1] and [b0,b1] (>= 0)."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _span_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Gap between intervals [a0,a1] and [b0,b1]; negative if they overlap."""
    return max(a0, b0) - min(a1, b1)


def _union_box(boxes: list[tuple]) -> tuple:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _connected_components(count: int, connected) -> list[list[int]]:
    """Union-find over all pairs; returns index groups via the `connected` test."""
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(count):
        for j in range(i + 1, count):
            if connected(i, j):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# --- Grouping passes -------------------------------------------------------

def _assemble_lines(units: list[dict], median_h: float) -> list[dict]:
    """Pass 1: merge same-row, horizontally-adjacent detections into lines."""
    def same_line(i: int, j: int) -> bool:
        bi, bj = units[i]["box"], units[j]["box"]
        hi, hj = bi[3] - bi[1], bj[3] - bj[1]
        v_overlap = _span_overlap(bi[1], bi[3], bj[1], bj[3])
        if v_overlap < SAME_ROW_OVERLAP * min(hi, hj):
            return False
        return _span_gap(bi[0], bi[2], bj[0], bj[2]) < LINE_MERGE_GAP * median_h

    lines = []
    for comp in _connected_components(len(units), same_line):
        members = sorted((units[k] for k in comp), key=lambda u: u["box"][0])
        worst = min(members, key=lambda u: u["score"])
        lines.append({
            "text": " ".join(u["text"] for u in members),
            "score": worst["score"],
            "conf": worst["conf"],
            "box": _union_box([u["box"] for u in members]),
        })
    return lines


def _assemble_bricks(lines: list[dict], median_h: float,
                     brick_gap: float) -> list[dict]:
    """Pass 2: stack vertically-adjacent, horizontally-aligned lines into bricks."""
    def same_brick(i: int, j: int) -> bool:
        li, lj = lines[i]["box"], lines[j]["box"]
        wi, wj = li[2] - li[0], lj[2] - lj[0]
        h_overlap = _span_overlap(li[0], li[2], lj[0], lj[2])
        if h_overlap < SAME_BRICK_OVERLAP * min(wi, wj):
            return False
        return _span_gap(li[1], li[3], lj[1], lj[3]) < brick_gap * median_h

    bricks = []
    for comp in _connected_components(len(lines), same_brick):
        members = sorted((lines[k] for k in comp), key=lambda ln: ln["box"][1])
        texts = [ln["text"] for ln in members]
        worst = min(members, key=lambda ln: ln["score"])
        bricks.append(_make_brick(
            texts, worst["conf"], worst["score"],
            _union_box([ln["box"] for ln in members])))
    return bricks


def _make_brick(lines: list[str], confidence: str, raw_score, box) -> dict:
    return {
        "inscription": "\n".join(lines),
        "lines": lines,
        "confidence": confidence,
        "raw_score": raw_score,
        "box": box,
        "n_lines": len(lines),
    }


def _reading_order(bricks: list[dict]) -> list[dict]:
    """Sort bricks top-to-bottom, then left-to-right within each row band."""
    placed = [b for b in bricks if b["box"] is not None]
    boxless = [b for b in bricks if b["box"] is None]
    if not placed:
        return boxless

    def cy(b):
        return (b["box"][1] + b["box"][3]) / 2

    placed.sort(key=cy)
    band = statistics.median(b["box"][3] - b["box"][1] for b in placed) * 0.7
    rows: list[list[dict]] = []
    for b in placed:
        if rows and cy(b) - cy(rows[-1][-1]) <= band:
            rows[-1].append(b)
        else:
            rows.append([b])

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b["box"][0]))
    return ordered + boxless


# --- Public entry point ----------------------------------------------------

def group(detections: list[dict], brick_gap: float = BRICK_STACK_GAP) -> list[dict]:
    """Group PaddleOCR line/word detections into bricks.

    `detections` -- list of {inscription, confidence, raw_score, box} as
    returned by ocr_paddle.run(); `box` is (x0,y0,x1,y1) or None.

    Returns a list of brick dicts in reading order, each:
        {inscription, lines, confidence, raw_score, box, n_lines}
    `inscription` joins the brick's lines with newlines. Detections that lack a
    box (no geometry to cluster on) are passed through as single-line bricks.
    """
    units, boxless = [], []
    for d in detections:
        box = d.get("box")
        if box is None:
            boxless.append(d)
        else:
            units.append({
                "text": d["inscription"],
                "score": d.get("raw_score") or 0.0,
                "conf": d.get("confidence", "low"),
                "box": tuple(float(v) for v in box),
            })

    bricks = []
    if units:
        median_h = max(
            statistics.median(u["box"][3] - u["box"][1] for u in units), 1.0)
        lines = _assemble_lines(units, median_h)
        bricks = _assemble_bricks(lines, median_h, brick_gap)

    for d in boxless:
        bricks.append(_make_brick(
            [d["inscription"]], d.get("confidence", "low"),
            d.get("raw_score"), None))

    return _reading_order(bricks)
