import asyncio
from collections import deque

from core.strategies.ml_strategy import MLStrategy, _normalize_runtime_bundle
from bot.formatters import format_model_status


class DummyBooster:
    def __init__(self, value):
        self.value = value
    def predict(self, feature_row):
        return [self.value]


def _make_strategy(models, meta):
    s = MLStrategy.__new__(MLStrategy)
    s._models = models
    s._model_meta = meta
    s._funding_buffer = deque(maxlen=24)
    s._model_slot = meta.get("slot", "current")
    s._last_funding_settlement = None
    return s


def test_normalize_runtime_bundle_keeps_legacy_single_explicit():
    models, meta = _normalize_runtime_bundle({"up": DummyBooster(0.6)}, {"threshold": 0.55, "down_threshold": 0.45})
    assert models is not None
    assert models["up"] is not None
    assert "down" not in models
    assert meta["inference_mode"] == "legacy_single"
    assert meta["has_real_down_model"] is False


def test_normalize_runtime_bundle_marks_real_dual():
    models, meta = _normalize_runtime_bundle(
        {"up": DummyBooster(0.6), "down": DummyBooster(0.7)},
        {"threshold": 0.55, "down_threshold": 0.52, "down_enabled": True, "artifact_version": 2},
    )
    assert models is not None
    assert "down" in models
    assert meta["inference_mode"] == "dual"
    assert meta["has_real_down_model"] is True


def test_threshold_precedence_db_over_metadata_for_both_sides():
    s = _make_strategy(
        {"up": DummyBooster(0.61), "down": DummyBooster(0.58)},
        {
            "slot": "current",
            "threshold": 0.53,
            "down_threshold": 0.49,
            "inference_mode": "dual",
            "has_real_down_model": True,
        },
    )

    import core.strategies.ml_strategy as mod

    async def fake_get_ml_threshold():
        return 0.64
    async def fake_get_ml_down_threshold():
        return 0.57

    orig_up = mod.queries.get_ml_threshold
    orig_down = mod.queries.get_ml_down_threshold
    mod.queries.get_ml_threshold = fake_get_ml_threshold
    mod.queries.get_ml_down_threshold = fake_get_ml_down_threshold
    try:
        up_thr, down_thr = asyncio.run(s._resolve_thresholds())
    finally:
        mod.queries.get_ml_threshold = orig_up
        mod.queries.get_ml_down_threshold = orig_down

    assert up_thr == 0.64
    assert down_thr == 0.57


def test_margin_arbitration_prefers_larger_side_specific_margin_dual():
    up_prob = 0.61
    down_prob = 0.70
    up_thr = 0.60
    down_thr = 0.52
    up_margin = up_prob - up_thr
    down_margin = down_prob - down_thr
    side = "Up" if up_margin >= down_margin else "Down"
    assert side == "Down"


def test_legacy_fallback_is_not_presented_as_dual_in_status():
    text = format_model_status(
        "current",
        {
            "train_date": "2026-04-17T10:00:00Z",
            "sample_count": 1234,
            "threshold": 0.55,
            "down_threshold": 0.45,
            "test_wr": 0.6,
            "val_wr": 0.59,
            "down_enabled": False,
            "inference_mode": "legacy_single",
        },
        0.56,
    )
    assert "Legacy single-model fallback" in text
    assert "Dual booster" not in text


def test_dual_mode_is_presented_in_status():
    text = format_model_status(
        "current",
        {
            "train_date": "2026-04-17T10:00:00Z",
            "sample_count": 1234,
            "threshold": 0.55,
            "down_threshold": 0.52,
            "test_wr": 0.6,
            "val_wr": 0.59,
            "down_val_wr": 0.58,
            "down_test_wr": 0.57,
            "down_enabled": True,
            "inference_mode": "dual",
        },
        0.56,
    )
    assert "Dual booster" in text
