"""Google (Gemini) vision-OCR provider.

Runs any Gemini vision model (2.5 Pro, 2.5 Flash, ...) via the Google Gemini
API. The prompt, image handling and JSON parsing are shared with the Anthropic
provider (see vision_ocr.py); this file only does the Gemini API call.

The API key is read from the GEMINI_API_KEY environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

import vision_ocr


def run(image_path: Path, model: str, prompt: str | None = None) -> list[dict]:
    """Extract brick inscriptions from one image with a Gemini vision model.

    `prompt` defaults to the whole-image prompt; pass vision_ocr.BRICK_PROMPT
    for a single-brick crop. Returns a list of {inscription, confidence} dicts.
    """
    prompt = prompt or vision_ocr.PROMPT
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before running the Gemini methods (see README.md)."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "The 'google-genai' package is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    image_bytes = vision_ocr.load_jpeg_bytes(image_path)
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )
    return vision_ocr.parse_response(response.text or "")
