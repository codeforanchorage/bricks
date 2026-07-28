#!/usr/bin/env python3
"""Consensus / triage step for the brick OCR comparison.

Collapses the multi-method comparison CSV (pipeline.py output) into a draft
catalogue: one row per estimated brick, with a status that triages which
bricks are trustworthy and which need a human to check.

The LLM methods return no coordinates, so reads are matched across methods by
text similarity. For each matched cluster of reads:

  agreed     >= 3 methods produced closely-matching text   -> auto-acceptable
  conflict   >= 2 methods found the brick but reads differ -> needs review
  single     only 1 method found it                        -> needs review

This is best-effort: where the methods agree it reliably identifies the brick;
where they disagree badly the reads may not even cluster, so one physical
brick can surface as several 'single' rows. Everything that is not 'agreed'
lands in the review pile -- nothing dubious is auto-accepted.

Usage:
    python consensus.py --input output/results.csv --output output/catalog.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# Two reads are the same brick if their normalised text is at least this
# similar; they "agree" on the wording at the higher threshold.
MATCH_THRESHOLD = 0.60
AGREE_THRESHOLD = 0.80

# Preferred source for the consensus wording, best first (see project notes).
METHOD_PRIORITY = ["gemini-pro", "gemini-flash", "claude-sonnet",
                   "claude-haiku", "paddleocr"]

CATALOG_COLUMNS = ["image", "inscription", "status",
                   "methods_found", "methods_agree", "variants"]
SKIP_CONFIDENCE = {"none", "error"}


def _normalise(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace -- for fuzzy matching."""
    text = text.lower().replace("/", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


# Dedication boilerplate appears on many different bricks; ignoring it when
# matching keeps brick clustering keyed on the distinctive text (the names),
# so "IN MEMORY OF / WANDA" and "IN MEMORY OF / SID" are not merged.
_STOPWORDS = {
    "in", "of", "the", "and", "a", "an", "to", "at", "on", "is", "for",
    "with", "by", "we", "you", "our", "memory", "loving", "love", "family",
}


def _match_key(norm: str) -> str:
    """Drop boilerplate words so brick matching keys on the distinctive text."""
    key = " ".join(w for w in norm.split() if w not in _STOPWORDS)
    return key or norm


# The original by-name brick list is a scan whose own OCR confuses a small,
# very consistent set of letter shapes -- TOWN->IOWN, LOVES->IDVES, THE->'IHE,
# ALASKA->AIASKA, LOVES->IDVFS. Folding each confusable group to one character
# on *both* sides of a comparison lets a good photo read match a badly scanned
# list entry, without trying to guess a correction for the scan text itself.
_CONFUSABLE = str.maketrans({
    "l": "i", "t": "i", "1": "i", "j": "i",   # I / L / T / 1 / J
    "f": "e",                                 # E / F
    "d": "o", "q": "o", "0": "o",             # O / D / Q / 0
    "5": "s", "8": "b",                       # S / 5, B / 8
})


def scan_fold(text: str) -> str:
    """Collapse scan-OCR confusable letters so noisy text still compares."""
    return text.translate(_CONFUSABLE)


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# Parts of the official lists were transcribed by voice, so a reference entry
# can spell a name phonetically while the brick spells it properly (BRIAN vs
# BRYAN, CATHY vs KATHY, BRITANY vs BRITTANY). Folding the classic phonetic
# equivalences -- and collapsing doubled letters -- lets those pairs compare
# as equal without loosening comparisons between genuinely different names.
_PH_PAIRS = [("ph", "f"), ("ck", "k")]


def phonetic_fold(text: str) -> str:
    """Collapse voice-transcription spelling variants for comparison."""
    for a, b in _PH_PAIRS:
        text = text.replace(a, b)
    text = (text.replace("c", "k").replace("z", "s").replace("y", "i"))
    return re.sub(r"(.)\1+", r"\1", text)


def similar_spoken(a: str, b: str) -> float:
    """Similarity that forgives phonetic spelling differences."""
    return max(_similar(a, b), _similar(phonetic_fold(a), phonetic_fold(b)))


def token_containment(read_key: str, ref_key: str) -> float:
    """How well the read's words are found *somewhere* in the reference key.

    A worn brick often yields a read that is a noisy subset of the inscription
    ("GRAND BEATY BUDDY" for "HAROLD G BEATY 1938-1991 MY 'BUDDY'"). Whole-
    string similarity punishes the read for every word it missed; this instead
    scores each read word by its best match among the reference words, weighted
    by length (longer words are more discriminative).

    Words shorter than 3 characters are ignored on the read side -- a stray
    "g" or "91" would match almost anything -- and a read with fewer than two
    usable words returns 0.0: one word, however good, is not identification.
    """
    read_words = [w for w in read_key.split() if len(w) >= 3]
    ref_words = ref_key.split()
    if len(read_words) < 2 or not ref_words:
        return 0.0
    weighted = sum(len(w) * max(similar_spoken(w, r) for r in ref_words)
                   for w in read_words)
    return weighted / sum(len(w) for w in read_words)


def _priority(method: str) -> int:
    return METHOD_PRIORITY.index(method) if method in METHOD_PRIORITY else 99


def _cluster(items: list[dict]) -> list[list[dict]]:
    """Greedily group reads of the same brick by distinctive-text similarity."""
    clusters: list[list[dict]] = []
    for item in items:
        best, best_sim = None, MATCH_THRESHOLD
        for cluster in clusters:
            sim = max(_similar(item["key"], member["key"])
                      for member in cluster)
            if sim >= best_sim:
                best, best_sim = cluster, sim
        if best is None:
            clusters.append([item])
        else:
            best.append(item)
    return clusters


def _summarise(cluster: list[dict], image: str) -> dict:
    """Turn one cluster of matched reads into a catalogue row."""
    spine = min(cluster, key=lambda it: _priority(it["method"]))
    methods_found = {it["method"] for it in cluster}
    agree = {it["method"] for it in cluster
             if _similar(it["norm"], spine["norm"]) >= AGREE_THRESHOLD}

    if len(agree) >= 3:
        status = "agreed"
    elif len(methods_found) >= 2:
        status = "conflict"
    else:
        status = "single"

    variants = " | ".join(
        f"{it['method']}={it['text']}"
        for it in sorted(cluster, key=lambda it: _priority(it["method"])))
    return {
        "image": image,
        "inscription": spine["text"],
        "status": status,
        "methods_found": len(methods_found),
        "methods_agree": len(agree),
        "variants": variants,
    }


def main(argv=None) -> None:
    # Inscriptions can contain non-cp1252 characters; force UTF-8 stdout.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Comparison CSV produced by pipeline.py")
    parser.add_argument("--output", required=True, type=Path,
                        help="Catalogue CSV to write")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        raise SystemExit(f"Input CSV not found: {args.input}")

    # Group readable rows by image, preserving first-seen image order.
    by_image: dict[str, list[dict]] = defaultdict(list)
    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("brick_inscription") or "").strip()
            conf = (row.get("confidence") or "").strip().lower()
            if not text or conf in SKIP_CONFIDENCE:
                continue
            norm = _normalise(text)
            if not norm:
                continue
            by_image[row["filename"]].append({
                "method": row["method"], "text": text,
                "norm": norm, "key": _match_key(norm),
            })

    if not by_image:
        raise SystemExit(f"No readable rows in {args.input}")

    catalog = []
    for image, items in by_image.items():
        for cluster in _cluster(items):
            catalog.append(_summarise(cluster, image))

    # Within each image: agreed first, then conflict, then single.
    rank = {"agreed": 0, "conflict": 1, "single": 2}
    catalog.sort(key=lambda r: (r["image"], rank[r["status"]],
                                -r["methods_found"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        writer.writerows(catalog)

    counts = {"agreed": 0, "conflict": 0, "single": 0}
    for row in catalog:
        counts[row["status"]] += 1
    total = len(catalog)
    print(f"\nConsensus catalogue: {total} estimated brick(s) "
          f"across {len(by_image)} image(s)")
    print(f"  agreed   : {counts['agreed']:4d}  "
          f"(>=3 methods concur -- auto-acceptable)")
    print(f"  conflict : {counts['conflict']:4d}  "
          f"(found by 2+ methods but reads differ -- review)")
    print(f"  single   : {counts['single']:4d}  "
          f"(only 1 method saw it -- review)")
    if total:
        print(f"  -> {counts['agreed'] / total:.0%} of rows auto-acceptable")
    print(f"\nWrote {total} row(s) to {args.output}")


if __name__ == "__main__":
    main()
