"""Tests for make_derivatives.py: the hosted photo derivative builder."""
from pathlib import Path

from PIL import Image

import make_derivatives


def _photo(path, size=(3000, 2000)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 100, 90)).save(path, "JPEG")


def test_builds_mirrored_thumb_and_zoom_trees(tmp_path):
    photos = tmp_path / "photos"
    _photo(photos / "pallets" / "Pallet F1" / "GP010481.JPG")
    _photo(photos / "pallets" / "K" / "small.jpg", size=(500, 400))
    out = tmp_path / "derivatives"

    make_derivatives.main(["--input", str(photos), "--output", str(out)])

    thumb = out / "thumbs" / "pallets" / "Pallet F1" / "GP010481.jpg"
    zoom = out / "zoom" / "pallets" / "Pallet F1" / "GP010481.jpg"
    assert thumb.is_file() and zoom.is_file()
    with Image.open(thumb) as img:
        assert img.width == make_derivatives.THUMB_WIDTH
    with Image.open(zoom) as img:
        assert img.width == make_derivatives.ZOOM_WIDTH

    # A photo already smaller than the target is copied, never upscaled.
    with Image.open(out / "zoom" / "pallets" / "K" / "small.jpg") as img:
        assert img.width == 500


def test_rerun_skips_current_outputs(tmp_path, capsys):
    photos = tmp_path / "photos"
    _photo(photos / "a.jpg")
    out = tmp_path / "derivatives"
    make_derivatives.main(["--input", str(photos), "--output", str(out)])
    capsys.readouterr()

    make_derivatives.main(["--input", str(photos), "--output", str(out)])
    text = capsys.readouterr().out
    assert "0 built, 1 already current" in text

    make_derivatives.main(["--input", str(photos), "--output", str(out),
                           "--force"])
    text = capsys.readouterr().out
    assert "1 built" in text


def test_derivative_paths_match_review_page_urls(tmp_path):
    # All three consumers share the hostpaths contract: the writer's disk
    # path, the review page's URL, and the search page's photo map.
    from hostpaths import derivative_rel, derivative_url
    thumb, zoom = make_derivatives._derivative_paths(
        Path("pallets/Pallet F1/GP1.HEIC"), Path("d"))
    assert thumb == Path("d/thumbs/pallets/Pallet F1/GP1.jpg")
    assert zoom == Path("d/zoom/pallets/Pallet F1/GP1.jpg")
    assert derivative_rel("pallets/Pallet F1/GP1.HEIC") == \
        "pallets/Pallet F1/GP1.jpg"
    assert derivative_url("https://x/photos", "zoom",
                          "pallets/Pallet F1/GP1.HEIC") == \
        "https://x/photos/zoom/pallets/Pallet%20F1/GP1.jpg"
