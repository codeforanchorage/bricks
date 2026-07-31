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


def derivative_url(base: str, tree: str, image: str) -> str:
    """Full URL for one derivative (tree: thumbs / zoom / strips)."""
    return f"{base}/{tree}/{quote(derivative_rel(image))}"
