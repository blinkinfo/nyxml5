import json
import tempfile
from pathlib import Path

import numpy as np

from ml import inference_logger
import config as cfg


def test_log_inference_writes_unambiguous_model_diagnostics(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "inference.jsonl"
        monkeypatch.setattr(cfg, "INFERENCE_LOG_PATH", str(log_path), raising=False)

        inference_logger.log_inference(
            slot_slug="slot-1",
            slot_ts=123,
            slot_start_str="10:00",
            slot_end_str="10:05",
            infer_time_utc="2026-04-17T16:00:00+00:00",
            df5_rows=400,
            df15_rows=100,
            df1h_rows=60,
            cvd_rows=400,
            funding_buf_len=24,
            candle_n1_ts="2026-04-17T15:55:00+00:00",
            candle_n1_close=100.0,
            candle_n1_vol=5.0,
            feature_names=["f1", "f2"],
            feature_row=np.array([[1.5, 2.5]]),
            nan_features=[],
            p_up=0.61,
            p_down=0.73,
            up_threshold=0.55,
            down_threshold=0.52,
            down_enabled=True,
            inference_mode="dual",
            has_real_down_model=True,
            down_model_is_same_object_as_up_model=False,
            p_down_source="down_booster",
            p_up_complement=0.39,
            p_down_minus_complement=0.34,
            fired=True,
            side="Down",
            skip_reason=None,
        )

        record = json.loads(log_path.read_text().strip())
        model = record["model"]
        assert model["inference_mode"] == "dual"
        assert model["has_real_down_model"] is True
        assert model["down_model_is_same_object_as_up_model"] is False
        assert model["p_down_source"] == "down_booster"
        assert model["p_up"] == 0.61
        assert model["p_down"] == 0.73
        assert model["p_up_complement"] == 0.39
        assert model["p_down_minus_complement"] == 0.34
