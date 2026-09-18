#!/usr/bin/env python3
"""The ONE contract for derivative image paths and URLs.

make_derivatives.py writes every derivative as the source photo's
relative path with a .jpg suffix (HEIC and .JPG alike). Three consumers
must agree on that transform byte-for-byte or images silently 404:
the derivative writer, the review page's hosted URLs, and the search
page's photo map. They all call here.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import quote


def derivative_rel(image: str) -> str:
    """Catalogue image path -> derivative-tree relative path (.jpg)."""
    return str(PurePosixPath(image).with_suffix(".jpg"))


# Bumped whenever strip images change CONTENT at an unchanged URL -- which
# re-filing does by design (2026-09-18: 366 strips were renamed to the
# number printed on them). The photo tree is served with a 30-day
# Cache-Control, so without a new query string a visitor who saw a brick
# in the last month keeps the old, wrong-row image. Any new value works;
# it only has to differ from the last one.
STRIP_VERSION = "20260918"


def strip_url(base: str, orig_id: str) -> str:
    """Full URL for one scanned-list row image, cache-busted."""
    return f"{base}/strips/{quote(str(orig_id))}.jpg?v={STRIP_VERSION}"


def derivative_url(base: str, tree: str, image: str) -> str:
    """Full URL for one derivative (tree: thumbs / zoom / strips)."""
    return f"{base}/{tree}/{quote(derivative_rel(image))}"
