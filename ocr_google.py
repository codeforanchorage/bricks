"""Google (Gemini) vision-OCR provider.

Runs any Gemini vision model (2.5 Pro, 2.5 Flash, ...) via the Google Gemini
API. The prompt, image handling and JSON parsing are shared with the Anthropic
provider (see vision_ocr.py); this file only does the Gemini API call.

The API key is read from the GEMINI_API_KEY environment variable.

Production notes: the client is built once and shared (it is thread-safe --
resolve_tsp_rows.py already shares one across 8 workers), and each call
retries transient failures with backoff. At 13,000+ images a run WILL hit
occasional 429s/5xxs; without retries each one becomes a permanent ERROR row
that later just reads as an unmatched brick.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import vision_ocr

_ATTEMPTS = 3        # tries per image before giving up
_BACKOFF = 4.0       # seconds before retry 1; doubles each retry

_client = None
_client_lock = threading.Lock()


def _get_client():
    """Build the Gemini client once; safe under concurrent first calls."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY environment variable is not set. "
                    "Set it before running the Gemini methods (see README.md)."
                )
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "The 'google-genai' package is not installed. "
                    "Run: pip install -r requirements.txt"
                ) from exc
            _client = genai.Client(api_key=api_key)
    return _client


def run(image_path: Path, model: str, prompt: str | None = None) -> list[dict]:
    """Extract brick inscriptions from one image with a Gemini vision model.

    `prompt` defaults to the whole-image prompt; pass vision_ocr.BRICK_PROMPT
    for a single-brick crop. Returns a list of {inscription, confidence} dicts.

    Transient API errors AND malformed replies are retried up to _ATTEMPTS
    times (a bad-JSON reply is model flakiness -- a second call usually
    parses); config errors (missing key / package) raise immediately.
    """
    prompt = prompt or vision_ocr.PROMPT
    client = _get_client()
    from google.genai import types

    image_bytes = vision_ocr.load_jpeg_bytes(image_path)
    part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    last_error: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=model, contents=[part, prompt])
            return vision_ocr.parse_response(response.text or "")
        except Exception as exc:  # noqa: BLE001 -- classify below, then retry
            last_error = exc
            if attempt < _ATTEMPTS - 1:
                time.sleep(_BACKOFF * (2 ** attempt))
    raise RuntimeError(
        f"Gemini call failed after {_ATTEMPTS} attempts: {last_error}"
    ) from last_error
