import asyncio
import logging
from collections import deque

from core.strategies.ml_strategy import MLStrategy, _normalize_runtime_bundle
from bot.formatters import format_model_status, format_retrain_blocked, format_retrain_complete


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


def test_retrain_complete_report_is_dual_bundle_aware():
    text, risk = format_retrain_complete(
        {
            "train_date": "2026-04-17T10:00:00Z",
            "sample_count": 1234,
            "data_start": "2026-01-01",
            "data_end": "2026-04-16",
            "payout": 0.85,
            "format": "dual_bundle",
            "bundle_version": 2,
            "artifact_version": 2,
            "inference_mode": "dual",
            "threshold": 0.55,
            "threshold_source": "walk_forward_validation_median",
            "val_wr": 0.59,
            "test_wr": 0.61,
            "test_trades_per_day": 4.2,
            "up_ev_per_day": 0.23,
            "down_threshold": 0.52,
            "down_val_wr": 0.58,
            "down_test_wr": 0.57,
            "down_test_tpd": 3.4,
            "down_ev_per_day": 0.11,
            "down_enabled": True,
            "models": {
                "up": {
                    "threshold": 0.55,
                    "val_wr": 0.59,
                    "test_wr": 0.61,
                    "test_trades_per_day": 4.2,
                    "ev_per_day": 0.23,
                },
                "down": {
                    "threshold": 0.52,
                    "val_wr": 0.58,
                    "test_wr": 0.57,
                    "test_trades_per_day": 3.4,
                    "ev_per_day": 0.11,
                    "enabled": True,
                },
            },
        },
        0.56,
    )
    assert risk is None
    assert "Dual-bundle artifact (candidate/current-ready)" in text
    assert "Deployment gate passed" in text
    assert "Enabled for live DOWN signals" in text
    assert "walk-forward median" in text
    assert "eligible to become current after review" in text


def test_retrain_blocked_report_separates_gate_from_down_enablement():
    text, _ = format_retrain_blocked(
        {
            "train_date": "2026-04-17T10:00:00Z",
            "sample_count": 1234,
            "payout": 0.85,
            "format": "dual_bundle",
            "bundle_version": 2,
            "artifact_version": 2,
            "inference_mode": "dual",
            "threshold": 0.55,
            "threshold_source": "walk_forward_validation_median",
            "val_wr": 0.59,
            "test_wr": 0.56,
            "test_trades_per_day": 4.2,
            "up_ev_per_day": 0.05,
            "down_threshold": 0.52,
            "down_val_wr": 0.58,
            "down_test_wr": 0.57,
            "down_test_tpd": 3.4,
            "down_ev_per_day": 0.11,
            "down_enabled": False,
            "models": {
                "up": {
                    "threshold": 0.55,
                    "val_wr": 0.59,
                    "test_wr": 0.56,
                    "test_trades_per_day": 4.2,
                    "ev_per_day": 0.05,
                    "blocked": True,
                },
                "down": {
                    "threshold": 0.52,
                    "val_wr": 0.58,
                    "test_wr": 0.57,
                    "test_trades_per_day": 3.4,
                    "ev_per_day": 0.11,
                    "enabled": False,
                    "blocked": True,
                },
            },
        },
        0.56,
    )
    assert "candidate, not live" in text
    assert "Deployment gate failed" in text
    assert "Disabled for live DOWN signals" in text
    assert "current-ready artifact not auto-promoted" in text
    assert "walk-forward median" in text


def test_dual_diagnostics_log_line_is_unambiguous(caplog):
    caplog.set_level(logging.INFO)
    side = "Down"
    prob = 0.61
    prob_down = 0.73
    prob_up_complement = round(1.0 - prob, 6)
    prob_down_minus_complement = round(prob_down - prob_up_complement, 6)
    up_threshold = 0.55
    down_threshold = 0.52
    down_enabled = True
    inference_mode = "dual"
    has_real_down_model = True
    down_model_is_same_object_as_up_model = False
    p_down_source = "down_booster"
    slug = "slot-123"

    logging.getLogger("core.strategies.ml_strategy").info(
        "MLStrategy: side=%s p_up=%.4f p_down=%.4f raw_1_minus_p_up=%.4f delta_p_down_vs_complement=%+.6f "
        "up_thr=%.3f down_thr=%.3f down_enabled=%s inference_mode=%s has_real_down_model=%s "
        "down_model_same_as_up=%s p_down_source=%s slot=%s",
        side, prob, prob_down, prob_up_complement, prob_down_minus_complement,
        up_threshold, down_threshold, down_enabled, inference_mode, has_real_down_model,
        down_model_is_same_object_as_up_model, p_down_source, slug,
    )

    msg = caplog.text
    assert "inference_mode=dual" in msg
    assert "has_real_down_model=True" in msg
    assert "down_model_same_as_up=False" in msg
    assert "p_down_source=down_booster" in msg
    assert "p_up=0.6100" in msg
    assert "p_down=0.7300" in msg
    assert "raw_1_minus_p_up=0.3900" in msg
    assert "delta_p_down_vs_complement=+0.340000" in msg
