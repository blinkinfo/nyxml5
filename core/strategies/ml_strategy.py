"""ML strategy using a trained LightGBM model for BTC/USDT 5-min binary prediction.

Returns the IDENTICAL signal dict schema as PatternStrategy.
Uses get_next_slot_info() + get_slot_prices() exactly as PatternStrategy does.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any


from core.strategies.base import BaseStrategy
from ml import data_fetcher
from ml import features as feat_eng
from ml import model_store
from ml import inference_logger
from db import queries
from polymarket.markets import get_next_slot_info, get_slot_prices
import config as cfg

log = logging.getLogger(__name__)

FEATURE_COLS = feat_eng.FEATURE_COLS  # 42 features in exact order

# Module-level reload flag so cmd_promote_model can signal a reload
_RELOAD_REQUESTED = False

# Module-level preloaded model bundle — injected at startup via set_model_bundle()
_PRELOADED_MODEL_BUNDLE = None


def _normalize_runtime_bundle(models: dict[str, Any] | None, metadata: dict | None = None) -> tuple[dict[str, Any] | None, dict]:
    """Normalize runtime state without falsely upgrading legacy artifacts to dual-model inference."""
    normalized_meta = model_store._normalize_bundle_metadata(metadata or {})
    if not models:
        return None, normalized_meta

    normalized_models = dict(models)
    up_model = normalized_models.get("up") or normalized_models.get("model")
    down_model = normalized_models.get("down")
    if up_model is None:
        return None, normalized_meta

    has_real_down_model = down_model is not None and down_model is not up_model
    inference_mode = "dual" if has_real_down_model else "legacy_single"

    normalized_models["up"] = up_model
    if down_model is not None:
        normalized_models["down"] = down_model
    else:
        normalized_models.pop("down", None)

    models_meta = normalized_meta.setdefault("models", {})
    up_meta = models_meta.setdefault("up", {})
    down_meta = models_meta.setdefault("down", {})
    if up_meta.get("threshold") is None:
        up_meta["threshold"] = normalized_meta.get("threshold")
    if down_meta.get("threshold") is None:
        down_meta["threshold"] = normalized_meta.get("down_threshold")
    if down_meta.get("enabled") is None:
        down_meta["enabled"] = normalized_meta.get("down_enabled", False)

    normalized_meta["threshold"] = up_meta.get("threshold")
    normalized_meta["down_threshold"] = down_meta.get("threshold")
    normalized_meta["down_enabled"] = bool(down_meta.get("enabled", False))
    normalized_meta["inference_mode"] = inference_mode
    normalized_meta["has_real_down_model"] = has_real_down_model
    return normalized_models, normalized_meta


def set_model_bundle(models: dict[str, Any], metadata: dict | None = None) -> None:
    """Inject a pre-loaded model bundle at startup or after retrain/promote."""
    global _PRELOADED_MODEL_BUNDLE
    normalized_models, normalized_meta = _normalize_runtime_bundle(models, metadata)
    _PRELOADED_MODEL_BUNDLE = {"models": normalized_models, "metadata": normalized_meta}


def set_model(model) -> None:
    """Backward-compatible wrapper that injects a legacy single-model bundle."""
    set_model_bundle({"up": model}, {})


def _runtime_bundle_diagnostics(
    models: dict[str, Any] | None,
    metadata: dict | None,
    *,
    load_source: str,
    reload_requested: bool,
) -> dict[str, Any]:
    """Summarize runtime bundle state for focused load/reload diagnostics."""
    meta = dict(metadata or {})
    model_keys = sorted(models.keys()) if models else []
    up_model = models.get("up") if models else None
    down_model = models.get("down") if models else None
    has_any_models = bool(models)
    has_up_model = up_model is not None
    has_down_key = down_model is not None
    models_are_distinct = has_down_key and down_model is not up_model
    runtime_inference_mode = "dual" if models_are_distinct else "legacy_single"
    runtime_down_source = "down_model" if models_are_distinct else "complement_fallback"
    meta_inference_mode = meta.get("inference_mode")
    artifact_format = meta.get("format") or ("dual_bundle" if int(meta.get("artifact_version") or meta.get("bundle_version") or 1) >= 2 else "legacy_single")
    artifact_version = int(meta.get("artifact_version") or meta.get("bundle_version") or 1)
    down_enabled = bool(meta.get("down_enabled", False))
    inconsistencies: list[str] = []

    if meta_inference_mode == "dual" and not models_are_distinct:
        inconsistencies.append("metadata_dual_runtime_complement_fallback")
    if artifact_format == "dual_bundle" and not has_down_key:
        inconsistencies.append("dual_bundle_metadata_missing_runtime_down_model")
    if has_down_key and not models_are_distinct:
        inconsistencies.append("runtime_down_model_aliases_up_model")
    if has_down_key and artifact_format != "dual_bundle":
        inconsistencies.append("runtime_has_down_model_but_metadata_not_dual_bundle")
    if down_enabled and not models_are_distinct:
        inconsistencies.append("down_enabled_without_distinct_down_model")

    return {
        "load_source": load_source,
        "reload_requested": reload_requested,
        "slot": meta.get("slot", "current"),
        "artifact_format": artifact_format,
        "artifact_version": artifact_version,
        "meta_inference_mode": meta_inference_mode,
        "runtime_inference_mode": runtime_inference_mode,
        "down_enabled": down_enabled,
        "threshold": meta.get("threshold"),
        "down_threshold": meta.get("down_threshold"),
        "model_keys": model_keys,
        "has_any_models": has_any_models,
        "has_up_model": has_up_model,
        "has_down_model": has_down_key,
        "models_are_distinct": models_are_distinct,
        "runtime_down_source": runtime_down_source,
        "inconsistencies": inconsistencies,
    }


def _log_runtime_bundle_diagnostics(
    models: dict[str, Any] | None,
    metadata: dict | None,
    *,
    load_source: str,
    reload_requested: bool,
) -> None:
    diag = _runtime_bundle_diagnostics(
        models,
        metadata,
        load_source=load_source,
        reload_requested=reload_requested,
    )
    log.info(
        "MLStrategy: runtime model load source=%s reload_requested=%s slot=%s artifact_format=%s artifact_version=%s "
        "meta_inference_mode=%s runtime_inference_mode=%s down_enabled=%s threshold=%s down_threshold=%s "
        "model_keys=%s has_any_models=%s has_up_model=%s has_down_model=%s models_are_distinct=%s "
        "runtime_down_source=%s inconsistencies=%s",
        diag["load_source"],
        diag["reload_requested"],
        diag["slot"],
        diag["artifact_format"],
        diag["artifact_version"],
        diag["meta_inference_mode"],
        diag["runtime_inference_mode"],
        diag["down_enabled"],
        diag["threshold"],
        diag["down_threshold"],
        diag["model_keys"],
        diag["has_any_models"],
        diag["has_up_model"],
        diag["has_down_model"],
        diag["models_are_distinct"],
        diag["runtime_down_source"],
        diag["inconsistencies"],
    )


def request_model_reload() -> None:
    """Signal that the model should be reloaded on the next check_signal call."""
    global _RELOAD_REQUESTED
    _RELOAD_REQUESTED = True


class MLStrategy(BaseStrategy):
    """LightGBM-based signal strategy. Replaces PatternStrategy as the default."""

    def __init__(self):
        self._models: dict[str, Any] | None = None
        self._model_meta: dict[str, Any] = {}
        self._funding_buffer: deque = deque(maxlen=24)
        self._model_slot = "current"
        # Track the last funding settlement timestamp that was appended to the
        # buffer.  MEXC settles funding every 8h (00:00, 08:00, 16:00 UTC).
        # We only append to the buffer when a new settlement period has started,
        # matching the training data semantics where each buffer entry represents
        # one distinct 8h settlement — not a repeated 5m snapshot of the same rate.
        self._last_funding_settlement: datetime | None = None
        # Each step is individually guarded so a failure in one never prevents
        # the other from running, and a constructor crash can never propagate
        # up to _get_strategy() / the scheduler.
        try:
            self._load_model()
        except Exception:
            log.exception(
                "MLStrategy.__init__: _load_model failed — model will be None; "
                "signals will be skipped until a model is loaded via set_model() or /retrain"
            )
        try:
            self._seed_funding_buffer()
        except Exception:
            log.exception(
                "MLStrategy.__init__: _seed_funding_buffer failed — "
                "funding zscore will be undefined for the first live periods; "
                "inference will continue with an empty buffer"
            )

    def _seed_funding_buffer(self) -> None:
        """Seed the funding buffer with historical data on startup.

        Without seeding, the buffer starts empty and zscore is undefined for the
        first 8 days of operation (24 periods * 8h each). This pre-fills the buffer
        from MEXC historical funding so zscore is valid from the very first inference.
        """
        try:
            history = data_fetcher.fetch_live_funding_history(n_periods=24)
            if history:
                for rate in history:
                    self._funding_buffer.append(rate)
                log.info(
                    "MLStrategy: seeded funding_buffer with %d historical records",
                    len(self._funding_buffer),
                )
            else:
                log.warning("MLStrategy: could not seed funding_buffer — no historical data returned")
        except Exception as exc:
            log.warning("MLStrategy: funding_buffer seed failed: %s", exc)

    @staticmethod
    def _current_funding_settlement() -> datetime:
        """Return the most recent MEXC funding settlement timestamp (UTC).

        MEXC settles funding at 00:00, 08:00, and 16:00 UTC every day.
        This returns the floor of utcnow() to the nearest 8h boundary,
        giving a stable, deterministic key for deduplication.
        """
        now = datetime.now(timezone.utc)
        # Hours since midnight, floored to 8h block: 0, 8, or 16
        settlement_hour = (now.hour // 8) * 8
        return now.replace(hour=settlement_hour, minute=0, second=0, microsecond=0)

    def _load_model(self) -> None:
        """Load the current model, preferring the canonical runtime source for current."""
        global _RELOAD_REQUESTED, _PRELOADED_MODEL_BUNDLE
        reload_requested = _RELOAD_REQUESTED
        if _PRELOADED_MODEL_BUNDLE is not None:
            self._models, preloaded_meta = _normalize_runtime_bundle(
                _PRELOADED_MODEL_BUNDLE.get("models"),
                _PRELOADED_MODEL_BUNDLE.get("metadata"),
            )
            self._model_meta = preloaded_meta
            self._model_slot = preloaded_meta.get("slot", "current")
            _PRELOADED_MODEL_BUNDLE = None
            _RELOAD_REQUESTED = False
            _log_runtime_bundle_diagnostics(
                self._models,
                self._model_meta,
                load_source="preloaded",
                reload_requested=reload_requested,
            )
            return

        models, meta = model_store.load_model_bundle_for_runtime_sync("current")
        self._models, normalized_meta = _normalize_runtime_bundle(models, meta)
        self._model_meta = normalized_meta
        self._model_slot = normalized_meta.get("slot", "current")
        _RELOAD_REQUESTED = False
        if self._models is None:
            log.warning("MLStrategy: no trained model bundle found in current slot")
        else:
            _log_runtime_bundle_diagnostics(
                self._models,
                self._model_meta,
                load_source="runtime_loader",
                reload_requested=reload_requested,
            )

    async def _get_threshold(self) -> float:
        """Read UP threshold from ml_config table, fall back to cfg default."""
        try:
            val = await queries.get_ml_threshold()
            return val
        except Exception:
            pass
        # Legacy fallback: check settings table
        try:
            val = await queries.get_setting("ml_threshold")
            if val is not None:
                return float(val)
        except Exception:
            pass
        return cfg.ML_DEFAULT_THRESHOLD

    async def _get_down_threshold(self, up_threshold: float) -> float:
        """Read DOWN threshold from DB, then model metadata, then legacy complement."""
        try:
            val = await queries.get_ml_down_threshold()
            if val is not None:
                return val
        except Exception:
            pass
        try:
            meta = self._model_meta or model_store.load_metadata(self._model_slot)
            if meta is not None and meta.get("down_threshold") is not None:
                return float(meta["down_threshold"])
        except Exception:
            pass
        return round(1.0 - up_threshold, 4)

    async def _resolve_thresholds(self) -> tuple[float, float]:
        """Resolve runtime thresholds with DB override first, then metadata, then legacy defaults."""
        meta = self._model_meta or model_store.load_metadata(self._model_slot) or {}

        up_threshold = meta.get("threshold")
        try:
            db_up = await queries.get_ml_threshold()
            if db_up is not None:
                up_threshold = db_up
        except Exception:
            if up_threshold is None:
                try:
                    legacy_up = await queries.get_setting("ml_threshold")
                    if legacy_up is not None:
                        up_threshold = float(legacy_up)
                except Exception:
                    pass
        if up_threshold is None:
            up_threshold = cfg.ML_DEFAULT_THRESHOLD

        down_threshold = meta.get("down_threshold")
        try:
            db_down = await queries.get_ml_down_threshold()
            if db_down is not None:
                down_threshold = db_down
        except Exception:
            pass
        if down_threshold is None:
            down_threshold = round(1.0 - float(up_threshold), 4)

        return float(up_threshold), float(down_threshold)

    def _get_down_enabled(self) -> bool:
        """Read down_enabled flag from current model metadata.

        Returns False if metadata is missing or down_enabled is not set,
        ensuring backwards-compatibility with models trained before Option B.
        """
        try:
            meta = self._model_meta or model_store.load_metadata(self._model_slot)
            if meta is not None:
                if meta.get("down_override", False):
                    return True
                return bool(meta.get("down_enabled", False))
        except Exception:
            pass
        return False

    async def check_signal(self) -> dict[str, Any] | None:
        """Generate an ML-based signal for slot N+1.

        Called at T-85s before the current slot ends.

        Returns the same signal dict schema as PatternStrategy:
          - skipped=True dict when no trade (below threshold or data issues)
          - skipped=False dict with full trade fields when model fires
          - None on hard failure
        """
        global _RELOAD_REQUESTED

        # Reload model if requested (e.g., after promote_model)
        if _RELOAD_REQUESTED:
            self._load_model()

        # Get next slot info — identical pattern to PatternStrategy
        slot_n1 = get_next_slot_info()
        slug = slot_n1["slug"]
        slot_ts = slot_n1["slot_start_ts"]
        slot_start_str = slot_n1["slot_start_str"]
        slot_end_str = slot_n1["slot_end_str"]

        # Standard base fields used in all return dicts (matches PatternStrategy exactly)
        base_fields: dict[str, Any] = {
            "skipped": True,
            "pattern": None,
            "candles_used": 400,
            "slot_n1_start_full": slot_n1["slot_start_full"],
            "slot_n1_end_full":   slot_n1["slot_end_full"],
            "slot_n1_start_str":  slot_start_str,
            "slot_n1_end_str":    slot_end_str,
            "slot_n1_ts":         slot_ts,
            "slot_n1_slug":       slug,
        }

        if self._models is None:
            self._load_model()
            if self._models is None:
                log.error("MLStrategy: no model bundle loaded, skipping slot %s", slug)
                inference_logger.log_skipped_data(
                    slot_slug=slug,
                    slot_ts=slot_ts,
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    skip_reason="No model loaded",
                )
                return {**base_fields, "reason": "No model loaded"}

        try:
            # Fetch live data in parallel using executor (blocking ccxt calls)
            loop = asyncio.get_running_loop()
            df5, df15, df1h, funding_rate, cvd_live = await asyncio.gather(
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_5m(400)),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_15m(100)),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_1h(60)),
                loop.run_in_executor(None, data_fetcher.fetch_live_funding),
                loop.run_in_executor(None, lambda: data_fetcher.fetch_live_gate_cvd(400)),
            )

            # --- Data quality snapshot (before dropping the forming candle) ---
            df5_rows_raw  = len(df5)      if df5      is not None else 0
            df15_rows     = len(df15)     if df15     is not None else 0
            df1h_rows     = len(df1h)     if df1h     is not None else 0
            cvd_rows_raw  = len(cvd_live) if cvd_live is not None and not cvd_live.empty else 0

            # df5 is passed to build_live_features WITH the still-forming candle N
            # intact as the last row (index -1).  build_live_features always uses
            # safe(series, k=1) / iloc[-2] to reference the N-1 (last fully-closed)
            # candle — the forming candle is never read as a feature value.
            #
            # Keeping the forming candle present is required for parity with training:
            # build_features() operates on a full df5 that includes row i (the "current"
            # row whose target we are predicting), and all feature shifts are k>=1.
            # Trimming df5 before calling build_live_features shifts every index by one,
            # making safe(s,1) return N-2 instead of N-1 — a systematic one-candle lag
            # that corrupts all 42 features and inverts the model's predictions.
            #
            # 15m/1h/CVD are also NOT trimmed: build_live_features selects the most
            # recent candle with timestamp <= ts_n1 (the N-1 5m bar's timestamp) via
            # backward merge, so no trimming is needed or correct there either.

            # Row counts (what the model actually sees)
            df5_rows  = len(df5)
            cvd_rows  = len(cvd_live) if cvd_live is not None and not cvd_live.empty else 0

            # N-1 candle metadata for the log
            candle_n1_ts    = None
            candle_n1_close = None
            candle_n1_vol   = None
            if df5_rows >= 2:
                try:
                    import pandas as _pd
                    # df5[-1] is the still-forming candle N; df5[-2] is the last
                    # fully-closed candle N-1 — that is what we log as "candle_n1".
                    n1 = df5.iloc[-2]
                    ts_raw = n1["timestamp"]
                    if isinstance(ts_raw, _pd.Timestamp):
                        candle_n1_ts = str(ts_raw.tz_localize("UTC").isoformat() if ts_raw.tzinfo is None else ts_raw.isoformat())
                    elif ts_raw is not None:
                        candle_n1_ts = str(_pd.Timestamp(int(ts_raw), unit="ms", tz="UTC").isoformat())
                    else:
                        candle_n1_ts = None
                    candle_n1_close = float(n1["close"])
                    candle_n1_vol   = float(n1["volume"])
                except Exception as _e:
                    log.debug("inference_logger: candle_n1 extraction failed: %s", _e)

            # Update funding rolling buffer — only append when a new 8h settlement
            # has occurred, matching training data semantics (one entry per settlement
            # period, not one entry per 5m check_signal call).
            if funding_rate is not None:
                current_settlement = self._current_funding_settlement()
                if self._last_funding_settlement != current_settlement:
                    self._funding_buffer.append(funding_rate)
                    self._last_funding_settlement = current_settlement
                    log.debug(
                        "MLStrategy: funding_buffer updated for settlement=%s rate=%.6f buffer_len=%d",
                        current_settlement.isoformat(), funding_rate, len(self._funding_buffer),
                    )

            funding_buf_len = len(self._funding_buffer)

            # Build feature row — returns (row, nan_features) 2-tuple
            feature_row, nan_features = feat_eng.build_live_features(
                df5, df15, df1h, funding_rate, self._funding_buffer, cvd_live
            )
            if feature_row is None:
                log.warning("MLStrategy: insufficient data for features, skipping")
                inference_logger.log_inference(
                    slot_slug=slug,
                    slot_ts=slot_ts,
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    df5_rows=df5_rows,
                    df15_rows=df15_rows,
                    df1h_rows=df1h_rows,
                    cvd_rows=cvd_rows,
                    funding_buf_len=funding_buf_len,
                    candle_n1_ts=candle_n1_ts,
                    candle_n1_close=candle_n1_close,
                    candle_n1_vol=candle_n1_vol,
                    feature_names=FEATURE_COLS,
                    feature_row=None,
                    nan_features=nan_features,
                    p_up=None,
                    p_down=None,
                    up_threshold=None,
                    down_threshold=None,
                    down_enabled=False,
                    fired=False,
                    side=None,
                    skip_reason="Insufficient data for features"
                    + (f" (NaN: {nan_features})" if nan_features else ""),
                )
                return {**base_fields, "reason": "Insufficient data for features"}

            up_model = self._models.get("up") if self._models else None
            down_model = self._models.get("down") if self._models else None
            if up_model is None:
                raise RuntimeError("UP model missing from loaded model bundle")

            # Model inference: always use the UP booster for p_up. Use the DOWN
            # booster independently when present; otherwise fall back to legacy
            # complement semantics without pretending that fallback is a real DOWN model.
            prob = float(up_model.predict(feature_row)[0])
            inference_mode = (self._model_meta or {}).get("inference_mode", "legacy_single")
            down_model_is_same_object_as_up_model = down_model is up_model and down_model is not None
            has_real_down_model = bool((self._model_meta or {}).get("has_real_down_model", False))
            if has_real_down_model and down_model is not None:
                prob_down = float(down_model.predict(feature_row)[0])
                p_down_source = "down_booster"
            else:
                prob_down = round(1.0 - prob, 6)
                inference_mode = "legacy_single"
                has_real_down_model = False
                p_down_source = "complement_fallback"
            prob_up_complement = round(1.0 - prob, 6)
            prob_down_minus_complement = round(prob_down - prob_up_complement, 6)

            up_threshold, down_threshold = await self._resolve_thresholds()
            up_qualifies = prob >= up_threshold

            # DOWN gate: only fire if the model's DOWN side was independently
            # validated (down_enabled=True in metadata). If the model was trained
            # before Option B or failed the DOWN sweep, down_enabled=False and
            # no DOWN trade ever fires regardless of prob_down.
            down_enabled = self._get_down_enabled()

            # ------------------------------------------------------------------
            # Regime gate -- covariate shift guard (Blueprint Option 1).
            #
            # At training time, trainer.py records the 5th and 95th percentile
            # of vol_regime across the full training dataset and stores them as
            # "regime_vol_p5" / "regime_vol_p95" in the model metadata JSON.
            #
            # Here we compare the live vol_regime value against those bounds.
            # If the live regime falls OUTSIDE [p5, p95], the model is operating
            # in a volatility environment it rarely saw during training -- its
            # probability estimates are less calibrated and the signal is suppressed.
            #
            # Design decisions:
            #   - Gate fires AFTER model.predict() so the log always contains the
            #     full p_up/p_down values. This lets you audit "what the model
            #     wanted to do" vs "what the gate blocked" -- invaluable for tuning.
            #   - If metadata is missing or the keys are absent (e.g. older model
            #     trained before this feature), the gate is silently skipped.
            #     This guarantees full backwards compatibility with no config change.
            #   - If either bound is None (degenerate training set < 10 samples),
            #     the gate is skipped -- not the live bot's fault, don't punish it.
            #   - The gate itself is wrapped in try/except so a metadata read error
            #     never crashes inference -- the model fires normally if gate errors.
            # ------------------------------------------------------------------
            try:
                _meta = model_store.load_metadata(self._model_slot)
                if _meta is not None:
                    _regime_p5  = _meta.get("regime_vol_p5")
                    _regime_p95 = _meta.get("regime_vol_p95")
                    if _regime_p5 is not None and _regime_p95 is not None:
                        _vol_regime_idx = FEATURE_COLS.index("vol_regime")
                        _live_regime = float(feature_row[0, _vol_regime_idx])
                        if not (_regime_p5 <= _live_regime <= _regime_p95):
                            _regime_skip_reason = (
                                f"Regime gate: vol_regime={_live_regime:.4f} outside training "
                                f"distribution [{_regime_p5:.4f}, {_regime_p95:.4f}] -- "
                                f"signal suppressed (covariate shift guard)"
                            )
                            log.warning("MLStrategy: %s", _regime_skip_reason)
                            inference_logger.log_inference(
                                slot_slug=slug,
                                slot_ts=slot_ts,
                                slot_start_str=slot_start_str,
                                slot_end_str=slot_end_str,
                                df5_rows=df5_rows,
                                df15_rows=df15_rows,
                                df1h_rows=df1h_rows,
                                cvd_rows=cvd_rows,
                                funding_buf_len=funding_buf_len,
                                candle_n1_ts=candle_n1_ts,
                                candle_n1_close=candle_n1_close,
                                candle_n1_vol=candle_n1_vol,
                                feature_names=FEATURE_COLS,
                                feature_row=feature_row,
                                nan_features=[],
                                p_up=prob,
                                p_down=prob_down,
                                up_threshold=up_threshold,
                                down_threshold=down_threshold,
                                down_enabled=down_enabled,
                                inference_mode=inference_mode,
                                has_real_down_model=has_real_down_model,
                                down_model_is_same_object_as_up_model=down_model_is_same_object_as_up_model,
                                p_down_source=p_down_source,
                                p_up_complement=prob_up_complement,
                                p_down_minus_complement=prob_down_minus_complement,
                                fired=False,
                                side=None,
                                skip_reason=_regime_skip_reason,
                            )
                            return {
                                **base_fields,
                                "pattern": f"p={prob:.4f} [regime_gate]",
                                "reason": _regime_skip_reason,
                                "ml_p_up":           prob,
                                "ml_p_down":         prob_down,
                                "ml_up_threshold":   up_threshold,
                                "ml_down_threshold": down_threshold,
                                "ml_down_enabled":   down_enabled,
                                "ml_inference_mode": inference_mode,
                            }
            except Exception as _rge:
                # Never let the regime gate itself crash inference.
                # Log and continue -- the model fires normally if the gate errors.
                log.warning(
                    "MLStrategy: regime gate check failed (non-fatal, continuing): %s", _rge
                )

            down_qualifies = down_enabled and (prob_down >= down_threshold)

            # Determine direction:
            #   - Both qualify  -> pick the one with the larger margin over its threshold.
            #   - Only one      -> pick that one.
            #   - Neither       -> skip.
            # Independent UP and DOWN models can both qualify, so arbitration must
            # compare side-specific margin over threshold rather than assume parity.
            if up_qualifies and down_qualifies:
                up_margin   = prob      - up_threshold
                down_margin = prob_down - down_threshold
                side = "Up" if up_margin >= down_margin else "Down"
                log.info(
                    "MLStrategy: BOTH qualify — up_margin=%.4f down_margin=%.4f → side=%s",
                    up_margin, down_margin, side,
                )
            elif up_qualifies:
                side = "Up"
            elif down_qualifies:
                side = "Down"
            else:
                # Build skip reason — include DOWN gate status so logs are clear
                if not down_enabled:
                    down_reason = "DOWN disabled (not validated)"
                else:
                    down_reason = f"p_down={prob_down:.4f}<{down_threshold:.3f}"
                skip_reason = (
                    f"Below threshold (p_up={prob:.4f}<{up_threshold:.3f}, {down_reason})"
                )
                inference_logger.log_inference(
                    slot_slug=slug,
                    slot_ts=slot_ts,
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    df5_rows=df5_rows,
                    df15_rows=df15_rows,
                    df1h_rows=df1h_rows,
                    cvd_rows=cvd_rows,
                    funding_buf_len=funding_buf_len,
                    candle_n1_ts=candle_n1_ts,
                    candle_n1_close=candle_n1_close,
                    candle_n1_vol=candle_n1_vol,
                    feature_names=FEATURE_COLS,
                    feature_row=feature_row,
                    nan_features=[],
                    p_up=prob,
                    p_down=prob_down,
                    up_threshold=up_threshold,
                    down_threshold=down_threshold,
                    down_enabled=down_enabled,
                    fired=False,
                    side=None,
                    skip_reason=skip_reason,
                )
                return {
                    **base_fields,
                    "pattern": f"p={prob:.4f}<{up_threshold:.3f}",
                    "reason": skip_reason,
                    # Structured ML fields for rich Telegram formatting
                    "ml_p_up": prob,
                    "ml_p_down": prob_down,
                    "ml_up_threshold": up_threshold,
                    "ml_down_threshold": down_threshold,
                    "ml_down_enabled": down_enabled,
                    "ml_inference_mode": inference_mode,
                }

            log.info(
                "MLStrategy: side=%s p_up=%.4f p_down=%.4f raw_1_minus_p_up=%.4f delta_p_down_vs_complement=%+.6f "
                "up_thr=%.3f down_thr=%.3f down_enabled=%s inference_mode=%s has_real_down_model=%s "
                "down_model_same_as_up=%s p_down_source=%s slot=%s",
                side, prob, prob_down, prob_up_complement, prob_down_minus_complement,
                up_threshold, down_threshold, down_enabled, inference_mode, has_real_down_model,
                down_model_is_same_object_as_up_model, p_down_source, slug,
            )

            # Fetch Polymarket prices — identical to PatternStrategy
            prices = await get_slot_prices(slug)
            if prices is None:
                log.warning(
                    "MLStrategy: no Polymarket prices for slug=%s, skipping", slug
                )
                inference_logger.log_inference(
                    slot_slug=slug,
                    slot_ts=slot_ts,
                    slot_start_str=slot_start_str,
                    slot_end_str=slot_end_str,
                    df5_rows=df5_rows,
                    df15_rows=df15_rows,
                    df1h_rows=df1h_rows,
                    cvd_rows=cvd_rows,
                    funding_buf_len=funding_buf_len,
                    candle_n1_ts=candle_n1_ts,
                    candle_n1_close=candle_n1_close,
                    candle_n1_vol=candle_n1_vol,
                    feature_names=FEATURE_COLS,
                    feature_row=feature_row,
                    nan_features=[],
                    p_up=prob,
                    p_down=prob_down,
                    up_threshold=up_threshold,
                    down_threshold=down_threshold,
                    down_enabled=down_enabled,
                    inference_mode=inference_mode,
                    has_real_down_model=has_real_down_model,
                    down_model_is_same_object_as_up_model=down_model_is_same_object_as_up_model,
                    p_down_source=p_down_source,
                    p_up_complement=prob_up_complement,
                    p_down_minus_complement=prob_down_minus_complement,
                    fired=False,
                    side=side,
                    skip_reason="Market data unavailable (no Polymarket prices)",
                )
                return {
                    **base_fields,
                    "pattern": f"p={prob:.4f}",
                    "reason": "Market data unavailable",
                    # ML inference already completed — include structured fields so
                    # the scheduler can render the rich ML skip card instead of
                    # falling back to the generic format_skip() card.
                    "ml_p_up":           prob,
                    "ml_p_down":         prob_down,
                    "ml_up_threshold":   up_threshold,
                    "ml_down_threshold": down_threshold,
                    "ml_down_enabled":   down_enabled,
                    "ml_inference_mode": inference_mode,
                }

            entry_price    = prices["up_price"]    if side == "Up" else prices["down_price"]
            opposite_price = prices["down_price"]  if side == "Up" else prices["up_price"]
            token_id          = prices["up_token_id"] if side == "Up" else prices["down_token_id"]
            opposite_token_id = prices["down_token_id"] if side == "Up" else prices["up_token_id"]

            # Log the fired inference record
            inference_logger.log_inference(
                slot_slug=slug,
                slot_ts=slot_ts,
                slot_start_str=slot_start_str,
                slot_end_str=slot_end_str,
                df5_rows=df5_rows,
                df15_rows=df15_rows,
                df1h_rows=df1h_rows,
                cvd_rows=cvd_rows,
                funding_buf_len=funding_buf_len,
                candle_n1_ts=candle_n1_ts,
                candle_n1_close=candle_n1_close,
                candle_n1_vol=candle_n1_vol,
                feature_names=FEATURE_COLS,
                feature_row=feature_row,
                nan_features=[],
                p_up=prob,
                p_down=prob_down,
                up_threshold=up_threshold,
                down_threshold=down_threshold,
                down_enabled=down_enabled,
                inference_mode=inference_mode,
                has_real_down_model=has_real_down_model,
                down_model_is_same_object_as_up_model=down_model_is_same_object_as_up_model,
                p_down_source=p_down_source,
                p_up_complement=prob_up_complement,
                p_down_minus_complement=prob_down_minus_complement,
                fired=True,
                side=side,
                skip_reason=None,
            )

            return {
                **base_fields,
                "skipped":           False,
                "side":              side,
                "entry_price":       entry_price,
                "opposite_price":    opposite_price,
                "token_id":          token_id,
                "opposite_token_id": opposite_token_id,
                "pattern":        f"p_up={prob:.4f},p_down={prob_down:.4f}",
                # Structured ML fields for rich Telegram formatting
                "ml_p_up":          prob,
                "ml_p_down":        prob_down,
                "ml_up_threshold":  up_threshold,
                "ml_down_threshold": down_threshold,
                "ml_down_enabled":  down_enabled,
                "ml_inference_mode": inference_mode,
            }

        except Exception as exc:
            log.exception("MLStrategy.check_signal failed: %s", exc)
            return None
