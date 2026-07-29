"""Anthropic (Claude) vision-OCR provider.

Runs any Claude vision model (Haiku, Sonnet, ...) via the Anthropic API. The
prompt, image handling and JSON parsing are shared with the Google provider
(see vision_ocr.py); this file only does the Anthropic API call.

The API key is read from the ANTHROPIC_API_KEY environment variable.

Production notes: the client is built once and shared (the Anthropic SDK
client is thread-safe), and each call retries transient failures with
backoff -- see ocr_google.py for the rationale.
"""
from __future__ import annotations

import base64
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
    """Build the Anthropic client once; safe under concurrent first calls."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Set it before running the Claude methods (see README.md)."
                )
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "The 'anthropic' package is not installed. "
                    "Run: pip install -r requirements.txt"
                ) from exc
            _client = Anthropic(api_key=api_key)
    return _client


def run(image_path: Path, model: str, prompt: str | None = None) -> list[dict]:
    """Extract brick inscriptions from one image with a Claude vision model.

    `prompt` defaults to the whole-image prompt; pass vision_ocr.BRICK_PROMPT
    for a single-brick crop. Returns a list of {inscription, confidence} dicts.

    Transient API errors AND malformed replies are retried up to _ATTEMPTS
    times; config errors (missing key / package) raise immediately.
    """
    prompt = prompt or vision_ocr.PROMPT
    client = _get_client()

    image_b64 = base64.standard_b64encode(
        vision_ocr.load_jpeg_bytes(image_path)).decode("ascii")
    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64,
            },
        },
        {"type": "text", "text": prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )
            reply = "".join(
                block.text for block in message.content
                if block.type == "text"
            )
            return vision_ocr.parse_response(reply)
        except Exception as exc:  # noqa: BLE001 -- retry transient failures
            last_error = exc
            if attempt < _ATTEMPTS - 1:
                time.sleep(_BACKOFF * (2 ** attempt))
    raise RuntimeError(
        f"Claude call failed after {_ATTEMPTS} attempts: {last_error}"
    ) from last_error
