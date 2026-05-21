"""PaddleOCR wrapper -- Method 1 (local OCR engine).

PaddleOCR detects and recognises text one line at a time, so every detected
text region comes back as a separate result. The engine has no concept of a
"brick", so a 3-line brick inscription appears here as 3 separate lines --
grouping lines back into bricks is left to a later stage of the project.

NOTE: PaddleOCR depends on paddlepaddle, which currently publishes no wheels
for Python 3.14. Install and run this project under Python 3.11-3.13.
See README.md for environment setup.
"""
from __future__ import annotations

from pathlib import Path

# The engine is expensive to construct (loads detection + recognition models),
# so it is built once and reused across every image in a run.
_engine = None
_engine_device: str | None = None


def _score_to_confidence(score: float) -> str:
    """Map PaddleOCR's 0.0-1.0 recognition score to a high/medium/low label.

    Keeps the CSV's `confidence` column uniform with the Haiku method, which
    reports the same three labels.
    """
    if score >= 0.90:
        return "high"
    if score >= 0.70:
        return "medium"
    return "low"


def _build_engine(device: str, cpu_threads: int | None):
    """Construct a PaddleOCR engine for the given device ('gpu' or 'cpu')."""
    from paddleocr import PaddleOCR

    # PaddleOCR 3.x signature. The document-orientation and unwarping
    # sub-models are unnecessary for flat nadir brick crops, so they are
    # disabled for speed. Text-line orientation is kept on -- bricks can be
    # photographed at any rotation as the camera passes over them.
    #
    # enable_mkldnn=False is required: paddlepaddle 3.3.x's oneDNN (MKLDNN)
    # CPU executor crashes with "ConvertPirAttribute2RuntimeAttribute not
    # support" on the OCR models. Disabling oneDNN runs the plain CPU kernels.
    kwargs = dict(
        lang="en",
        device=device,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    if cpu_threads and device == "cpu":
        kwargs["cpu_threads"] = cpu_threads

    try:
        return PaddleOCR(**kwargs)
    except (TypeError, ValueError):
        # Retry without cpu_threads in case that kwarg is unsupported.
        kwargs.pop("cpu_threads", None)
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError):
            # Older PaddleOCR (2.x) used a different constructor signature.
            return PaddleOCR(lang="en", use_angle_cls=True,
                             use_gpu=(device == "gpu"))


def _gpu_available() -> bool:
    """True only if paddle is a CUDA build with a visible GPU.

    The CPU paddlepaddle build always returns False here. This check matters:
    passing device='gpu' to a CPU build does NOT raise -- PaddleOCR silently
    switches to CPU but loses the enable_mkldnn=False setting, which re-triggers
    the oneDNN executor bug. So the real device must be resolved up front.
    """
    try:
        import paddle
        return (paddle.is_compiled_with_cuda()
                and paddle.device.cuda.device_count() > 0)
    except Exception:  # noqa: BLE001
        return False


def _get_engine(use_gpu: bool, cpu_threads: int | None):
    """Return a cached PaddleOCR engine on the best available device."""
    global _engine, _engine_device
    if _engine is not None:
        return _engine

    try:
        import paddleocr  # noqa: F401  -- presence check only
    except ImportError as exc:
        raise RuntimeError(
            "paddleocr is not installed. PaddleOCR requires Python 3.11-3.13 "
            "(no paddlepaddle wheels exist for Python 3.14). See README.md."
        ) from exc

    device = "gpu" if (use_gpu and _gpu_available()) else "cpu"
    try:
        _engine = _build_engine(device, cpu_threads)
        _engine_device = device
    except Exception as exc:  # noqa: BLE001 -- any GPU failure -> CPU fallback
        if device == "cpu":
            raise
        print(f"    PaddleOCR GPU init failed ({exc}); falling back to CPU.")
        _engine = _build_engine("cpu", cpu_threads)
        _engine_device = "cpu"
    return _engine


def _box_from_poly(poly) -> tuple | None:
    """Axis-aligned (x0, y0, x1, y1) bounding box of a 4-point polygon."""
    try:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        return (min(xs), min(ys), max(xs), max(ys))
    except (TypeError, ValueError, IndexError):
        return None


def _extract_box(boxes, polys, idx: int) -> tuple | None:
    """Get detection `idx`'s box, preferring axis-aligned rec_boxes over polys."""
    if boxes is not None and idx < len(boxes):
        b = boxes[idx]
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    if polys is not None and idx < len(polys):
        return _box_from_poly(polys[idx])
    return None


def _parse_results(results) -> list[dict]:
    """Normalise PaddleOCR output into a list of inscription dicts.

    Each dict carries a `box` (x0, y0, x1, y1) so detections can later be
    grouped into bricks (see group_bricks.py). Handles both the 3.x
    `predict()` format (dict-like OCRResult) and the 2.x `ocr()` format.
    """
    items: list[dict] = []
    for res in results or []:
        # PaddleOCR 3.x: each result is a dict-like OCRResult.
        if hasattr(res, "get") and res.get("rec_texts") is not None:
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            boxes = res.get("rec_boxes")
            polys = res.get("rec_polys")
            for idx, text in enumerate(texts):
                text = (text or "").strip()
                if not text:
                    continue
                score = float(scores[idx]) if idx < len(scores) else 0.0
                items.append({
                    "inscription": text,
                    "confidence": _score_to_confidence(score),
                    "raw_score": score,
                    "box": _extract_box(boxes, polys, idx),
                })
            continue

        # PaddleOCR 2.x: result is [[poly, (text, score)], ...].
        for line in res or []:
            try:
                poly = line[0]
                text, score = line[1][0], float(line[1][1])
            except (IndexError, TypeError, ValueError):
                continue
            text = (text or "").strip()
            if not text:
                continue
            items.append({
                "inscription": text,
                "confidence": _score_to_confidence(score),
                "raw_score": score,
                "box": _box_from_poly(poly),
            })
    return items


def run(image_path: Path, use_gpu: bool = True,
        cpu_threads: int | None = 8) -> list[dict]:
    """Extract text lines from one brick image with PaddleOCR.

    `cpu_threads` sets the inference thread count when running on CPU. The
    engine is built once and cached, so this takes effect on the first call.

    Returns a list of dicts, each: {inscription, confidence, raw_score, box},
    where box is (x0, y0, x1, y1). These are line/word level -- pass them
    through group_bricks.group() to assemble per-brick inscriptions.
    """
    engine = _get_engine(use_gpu, cpu_threads)
    path = str(image_path)

    if hasattr(engine, "predict"):
        results = engine.predict(input=path)  # PaddleOCR 3.x
    else:
        results = engine.ocr(path)  # PaddleOCR 2.x
    return _parse_results(results)
