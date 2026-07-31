"""Tests for rescan_rows' read cache (free resume after interruptions)."""
import json

from rescan_rows import MODELS, _load_cache


def test_cache_roundtrip_and_model_guard(tmp_path):
    state = tmp_path / "state.jsonl"
    lines = [
        {"number": "1375", "models": MODELS,
         "reads": [{"line1": "THE WOOFTER FAMILY"}, {"line1": "same"}]},
        {"number": "9999", "models": ["some-old-model", "other"],
         "reads": [{"line1": "STALE"}, None]},
    ]
    state.write_text("\n".join(json.dumps(e) for e in lines)
                     + '\n{"torn line', encoding="utf-8")

    cache = _load_cache(state, MODELS)
    assert "1375" in cache                      # same models: served
    assert "9999" not in cache                  # model change: ignored
    assert cache["1375"][0]["line1"] == "THE WOOFTER FAMILY"


def test_missing_cache_is_empty(tmp_path):
    assert _load_cache(tmp_path / "nope.jsonl", MODELS) == {}
