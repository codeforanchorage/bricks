#!/usr/bin/env python3
"""Re-file strip images that were cut under the wrong brick number.

make_strips.py names each strip from the PDF text layer's row number, but on
some pages that layer is offset from the printed rows, so the image filed as
<n>.jpg shows a neighbouring row (found 2026-09-17 via Ashley/Zjok Durst:
strips/6889.jpg shows the printed 6891 row). audit_strips.py reads the number
printed inside each strip, and its --text mode transcribes the whole row so a
misfile can be confirmed by NAME -- the number alone over-flags, because a
number inside an inscription or a clipped number column reads as a mismatch
on a correctly-filed strip.

This script renames only the CONFIRMED misfiles (verdict
'GENUINE MISFILE' in the recheck verdicts) to the number actually printed on
them. Renaming is two-phase, so a cycle of swaps cannot clobber itself, and
every move is written to a map CSV that --revert replays backwards.

    python audit_strips.py --output output/strip_audit.csv
    python audit_strips.py --text --only <flagged> --output output/strip_recheck_text.csv
    python refile_strips.py --verdicts output/strip_recheck_verdicts.csv

A strip whose target number is held by a strip confirmed CORRECT is dropped
(moved to ../strips_misfiled/) rather than overwriting it: no strip is
better than a wrong one. The holding folder sits BESIDE the strips tree,
never inside it -- anything inside gets swept up when the tree is synced to
the web host (2026-09-18: a WinSCP sync published 530 wrong-row images
because the holding folders were subdirectories). Re-sync the tree
afterwards -- the review page reads strips by brick number.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strips", type=Path, default=Path("derivatives/strips"))
    p.add_argument("--verdicts", type=Path,
                   default=Path("output/strip_recheck_verdicts.csv"))
    p.add_argument("--map", type=Path,
                   default=Path("output/strip_refile_map.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="re-file again despite the .refiled sentinel")
    p.add_argument("--revert", action="store_true",
                   help="undo the moves recorded in --map")
    a = p.parse_args(argv)

    if a.revert:
        moves = list(csv.DictReader(open(a.map, newline="", encoding="utf-8")))
        for m in reversed(moves):
            src, dst = a.strips / m["to"], a.strips / m["from"]
            if m["kind"] == "dropped":
                src = a.strips.parent / (a.strips.name + "_misfiled") / m["to"]
            if src.exists():
                src.rename(dst)
        print(f"reverted {len(moves)} move(s)")
        return

    # SINGLE PASS ONLY. The verdicts are keyed on the ORIGINAL filenames, so
    # a second pass would read an already-corrected strip as that number's
    # misfile and move the right image away again (seen 2026-09-17: five
    # cascading passes shuffled ~560 strips). The sentinel makes that
    # impossible; regenerate the tree with make_strips.py to start over.
    done = a.strips / ".refiled"
    if done.exists() and not a.force:
        raise SystemExit(
            f"{a.strips} was already re-filed ({done.read_text().strip()}). "
            "Re-filing twice corrupts the tree -- regenerate the strips and "
            "re-run the audit, or pass --force if you know why.")

    rows = list(csv.DictReader(open(a.verdicts, newline="", encoding="utf-8")))
    bad = {r["strip_file"]: r["printed_number"].lstrip("0") or "0"
           for r in rows if r["verdict"] == "GENUINE MISFILE"}
    confirmed_ok = {r["strip_file"] for r in rows
                    if r["verdict"].startswith("correctly")}
    present = {q.stem for q in a.strips.glob("*.jpg")}
    # A crashed earlier run leaves sources parked as __tmp_<n>.jpg: they are
    # still "present", just mid-move.
    present |= {q.stem[len("__tmp_"):] for q in a.strips.glob("__tmp_*.jpg")}
    # A number is "held" unless its holder is itself moving away.
    held = {q for q in present if q not in bad
            and not (a.strips / f"__tmp_{q}.jpg").exists()}

    moves, dropped, skipped = [], [], []
    for src, target in sorted(bad.items(), key=lambda t: int(t[0])):
        if src not in present:
            skipped.append((src, target, "source missing"))
            continue
        if target in held:
            dropped.append((src, target))
        elif any(t == target for _, t in moves):
            skipped.append((src, target, "two strips claim this number"))
        else:
            moves.append((src, target))

    print(f"confirmed misfiles {len(bad)}: renaming {len(moves)}, "
          f"dropping {len(dropped)} (number held by a correct strip), "
          f"skipping {len(skipped)}")
    if a.dry_run:
        for s, t in moves[:10]:
            print(f"  {s}.jpg -> {t}.jpg")
        return

    # Dropped strips vacate their numbers FIRST: a dropped source still
    # sitting on disk owns a number some rename needs (Windows rename
    # refuses an existing target).
    out = a.strips.parent / (a.strips.name + "_misfiled")
    if dropped:
        out.mkdir(exist_ok=True)
    aside_names = {}
    for src, _target in dropped:
        q = a.strips / f"{src}.jpg"
        if not q.exists():
            continue
        dst = out / f"{src}.jpg"
        n = 2
        while dst.exists():          # a number can be vacated twice across
            dst = out / f"{src}_{n}.jpg"   # re-runs; keep both images
            n += 1
        q.rename(dst)
        aside_names[src] = dst.name

    # Two-phase, and idempotent: a crashed earlier run leaves __tmp_ files,
    # which phase 2 below picks up.
    tmp = []
    for src, target in moves:
        q = a.strips / f"{src}.jpg"
        t = a.strips / f"__tmp_{src}.jpg"
        if t.exists():
            tmp.append((t, target, src))
        elif q.exists():
            q.rename(t)
            tmp.append((t, target, src))
    for t, target, src in tmp:
        dst = a.strips / f"{target}.jpg"
        if dst.exists():
            dst.unlink()          # target confirmed wrong; the mover is right
        t.rename(dst)

    with open(a.map, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from", "to", "kind"])
        for src, target in moves:
            w.writerow([f"{src}.jpg", f"{target}.jpg", "renamed"])
        for src, target in dropped:
            w.writerow([f"{src}.jpg", aside_names.get(src, f"{src}.jpg"),
                        "dropped"])
    import datetime
    done.write_text("%s %d renamed, %d set aside, map %s"
                    % (datetime.date.today(), len(moves), len(dropped), a.map))
    print(f"wrote {a.map}; revert with: python refile_strips.py --revert")


if __name__ == "__main__":
    main()
