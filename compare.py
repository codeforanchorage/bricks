"""Stacked terminal comparison of the OCR methods.

With three methods, side-by-side columns are too narrow for a terminal, so
each method's results are printed as its own indented block under the image.
"""
from __future__ import annotations

import textwrap

_RULE_WIDTH = 74
_TEXT_WIDTH = 68  # wrap width for inscription lines (indented 4 spaces)


def _count_label(items, error, unit: str) -> str:
    if error:
        return "FAILED"
    if items is None:
        return "not run"
    n = len(items)
    return f"{n} {unit}{'' if n == 1 else 's'}"


def _print_block(items, error) -> None:
    """Print one method's results as an indented block."""
    if error:
        for line in textwrap.wrap(f"ERROR: {error}", _TEXT_WIDTH) or ["ERROR"]:
            print(f"    {line}")
        return
    if items is None:
        print("    (method not run)")
        return
    if not items:
        print("    (no text detected)")
        return
    for item in items:
        conf = item["confidence"]
        score = item.get("raw_score")
        # PaddleOCR carries a raw 0-1 score; the LLM methods do not.
        tag = f"[{conf} {score:.2f}]" if score is not None else f"[{conf}]"
        text = item["inscription"].replace("\n", " / ")
        wrapped = textwrap.wrap(f"{tag} {text}", _TEXT_WIDTH,
                                subsequent_indent="      ")
        for line in wrapped or [tag]:
            print(f"    {line}")


def print_comparison(filename, methods) -> None:
    """Print one image's results, one stacked block per method.

    `methods` is a list of (label, items, error, unit) tuples in display order.
    Pass items=None for a method that did not run on this image.
    """
    print()
    print("=" * _RULE_WIDTH)
    print(f"  IMAGE: {filename}")
    print("=" * _RULE_WIDTH)
    for label, items, error, unit in methods:
        print(f"\n  {label}  --  {_count_label(items, error, unit)}")
        _print_block(items, error)
