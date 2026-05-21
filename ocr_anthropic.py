"""Anthropic (Claude) vision-OCR provider.

Runs any Claude vision model (Haiku, Sonnet, ...) via the Anthropic API. The
prompt, image handling and JSON parsing are shared with the Google provider
(see vision_ocr.py); this file only does the Anthropic API call.

The API key is read from the ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import vision_ocr


def run(image_path: Path, model: str, prompt: str | None = None) -> list[dict]:
    """Extract brick inscriptions from one image with a Claude vision model.

    `prompt` defaults to the whole-image prompt; pass vision_ocr.BRICK_PROMPT
    for a single-brick crop. Returns a list of {inscription, confidence} dicts.
    """
    prompt = prompt or vision_ocr.PROMPT
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

    image_b64 = base64.standard_b64encode(
        vision_ocr.load_jpeg_bytes(image_path)).decode("ascii")
    client = Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    reply = "".join(
        block.text for block in message.content if block.type == "text"
    )
    return vision_ocr.parse_response(reply)
