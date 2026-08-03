#!/usr/bin/env python3
"""The ONE contract for the hosted pages' shared navigation bar.

The three staff pages (search, review queue, duplicate-claim check) live
side by side in the same basic-auth directory, so one login covers them
all -- the nav bar just lets staff click between them. Every generator
calls here so the filenames cannot drift: the links only stay valid
because run_pipeline.py writes these exact stable names (the .htaccess
serves HTML no-cache, so stable names never go stale in a browser).
"""
from __future__ import annotations

# (filename, label) -- filename is both the link target and the stable
# name run_pipeline.py writes into output/dreamhost_upload/.
PAGES = [
    ("search.html", "Search"),
    ("review.html", "Review queue"),
    ("fp_review.html", "Duplicate check"),
]

# Plain CSS. Safe in str.format() templates too: it arrives as a VALUE
# ({nav_css}), and format never re-scans substituted values for braces.
NAV_CSS = """
 #pagenav { display: flex; gap: 16px; align-items: baseline;
        font-size: 13.5px; margin-bottom: 6px; }
 #pagenav a { color: #cfe3f5; text-decoration: none;
        border-bottom: 1px dotted #7d99b3; padding-bottom: 1px; }
 #pagenav a:hover { color: #fff; border-bottom-color: #fff; }
 #pagenav .here { color: #fff; font-weight: 700;
        border-bottom: 2px solid #e8a33d; padding-bottom: 1px; }
 #pagenav .stamp { margin-left: auto; font-size: 12px; opacity: .75; }
"""


def nav_html(current: str, stamp: str = "") -> str:
    """The nav bar with `current` shown as the you-are-here page.

    All links are relative -- the pages sit in one directory behind one
    basic-auth realm, so following them never asks for a second login.
    """
    parts = []
    for fname, label in PAGES:
        if fname == current:
            parts.append(f'<span class="here">{label}</span>')
        else:
            parts.append(f'<a href="{fname}">{label}</a>')
    if stamp:
        parts.append(f'<span class="stamp">{stamp}</span>')
    return '<nav id="pagenav">' + " ".join(parts) + "</nav>"
