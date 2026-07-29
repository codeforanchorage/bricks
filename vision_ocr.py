"""Shared helpers for the vision-LLM OCR methods (Claude, Gemini).

Both LLM methods send the same image and the same prompt and expect the same
JSON contract back, so the comparison stays fair. The provider-specific code
(which API client, how to attach the image) lives in ocr_anthropic.py and
ocr_google.py; everything common lives here.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

# Vision APIs downscale large images server-side; resizing here first avoids
# oversized uploads and keeps both providers on identical input.
MAX_IMAGE_EDGE = 1568

PROMPT = (
    "This is a downward-facing photo of commemorative paver bricks with "
    "engraved text. For each brick where you can read text, extract the full "
    "inscription exactly as engraved. Return results as a JSON array where "
    "each element is an object with 'inscription' (full text) and 'confidence' "
    "('high', 'medium', or 'low'). If no text is readable, return an empty array."
)

# Prompt for a single-brick crop (the detect -> crop -> per-brick pipeline).
BRICK_PROMPT = (
    "This image is a close-up of a single commemorative paver brick. Read the "
    "engraved inscription on the main brick that fills the image; ignore any "
    "partial text from adjacent bricks at the edges. The text may be at any "
    "rotation. Return a JSON array: if the brick has readable engraved text, a "
    "single object with 'inscription' (the full text, newline-separated for "
    "multiple lines) and 'confidence' ('high', 'medium', or 'low'); if no text "
    "is readable, an empty array."
)

_VALID_CONFIDENCE = {"high", "medium", "low"}


def load_jpeg_bytes(image_path: Path, max_edge: int = MAX_IMAGE_EDGE) -> bytes:
    """Load an image, downscale it to `max_edge`, return re-encoded JPEG bytes.

    EXIF orientation is deliberately NOT applied. Measured 2026-07-29 on the
    one labeled photo stored rotated (IMG_0099.jpg, orientation=6): 4 trials
    sideways vs 4 trials exif-transposed produced 8/8 byte-identical reads --
    the vision models are rotation-robust on engraved bricks, so transposing
    buys nothing and would only perturb a validated pipeline. (HEIC files
    arrive upright regardless: pillow-heif applies the container rotation on
    decode.)
    """
    from PIL import Image

    if image_path.suffix.lower() == ".heic":
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError as exc:
            raise RuntimeError(
                "Reading .heic images needs the 'pillow-heif' package. "
                "Run: pip install -r requirements.txt") from exc

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        longest = max(img.size)
        if longest > max_edge:
            scale = max_edge / longest
            new_size = (round(img.width * scale), round(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _extract_json_array(text: str) -> list:
    """Pull a JSON array out of a model reply, tolerating markdown fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Fall back to the span from the first '[' to the last ']'.
    if not text.startswith("["):
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def _normalise(raw: list) -> list[dict]:
    """Validate and clean parsed JSON into {inscription, confidence} dicts."""
    items: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        inscription = str(entry.get("inscription", "")).strip()
        if not inscription:
            continue
        confidence = str(entry.get("confidence", "")).strip().lower()
        if confidence not in _VALID_CONFIDENCE:
            confidence = "low"
        items.append({"inscription": inscription, "confidence": confidence})
    return items


def parse_response(reply: str) -> list[dict]:
    """Turn a model's raw text reply into a list of {inscription, confidence}.

    Returns [] for an empty reply; raises RuntimeError if the reply is not a
    parseable JSON array.
    """
    reply = (reply or "").strip()
    if not reply:
        return []
    try:
        raw = _extract_json_array(reply)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not parse JSON from the model reply: {exc}\n"
            f"Reply was: {reply[:500]}"
        ) from exc
    if isinstance(raw, dict):  # tolerate a bare object instead of an array
        raw = [raw]
    if not isinstance(raw, list):
        raise RuntimeError(
            f"Expected a JSON array from the model, got {type(raw).__name__}."
        )
    return _normalise(raw)
