"""Model store for dual-model ML bundles with legacy single-model fallback."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Any

import lightgbm as lgb

from ml.features import FEATURE_COLS

log = logging.getLogger(__name__)

_EXPECTED_NUM_FEATURES = len(FEATURE_COLS)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BUNDLE_VERSION = 2
MODEL_KEYS = ("up", "down")


def _ensure_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def _validate_feature_count(model: lgb.Booster, slot: str, source: str, model_key: str = "up") -> lgb.Booster | None:
    """Return the model if its feature count matches FEATURE_COLS, else None."""
    n = model.num_feature()
    if n != _EXPECTED_NUM_FEATURES:
        log.warning(
            "%s: model slot=%s key=%s has %d features but current FEATURE_COLS expects %d "
            "- discarding stale model (signals will be skipped until retrain)",
            source,
            slot,
            model_key,
            n,
            _EXPECTED_NUM_FEATURES,
        )
        return None
    return model


def _legacy_model_path(slot: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{slot}.lgb")


def _legacy_meta_path(slot: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{slot}_meta.json")


def _bundle_model_path(slot: str, model_key: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{slot}_{model_key}.lgb")


def _bundle_meta_path(slot: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{slot}_bundle_meta.json")


def _serialize_model(model: lgb.Booster) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".lgb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save_model(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _deserialize_model(blob: bytes, slot: str, source: str, model_key: str) -> lgb.Booster | None:
    with tempfile.NamedTemporaryFile(suffix=".lgb", delete=False) as tmp:
        tmp.write(blob)
        tmp_path = tmp.name
    try:
        model = lgb.Booster(model_file=tmp_path)
        return _validate_feature_count(model, slot, source, model_key=model_key)
    except Exception as e:
        log.error("%s: failed to load slot=%s key=%s: %s", source, slot, model_key, e)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _normalize_bundle_metadata(metadata: dict | None, slot: str = "current") -> dict:
    """Normalize metadata into bundle-first schema while preserving legacy fields."""
    metadata = dict(metadata or {})
    artifact_version = int(metadata.get("artifact_version") or metadata.get("bundle_version") or 1)
    models_meta = metadata.get("models")

    if isinstance(models_meta, dict) and "up" in models_meta:
        normalized = metadata
    else:
        legacy_up = {
            "threshold": metadata.get("threshold"),
            "val_wr": metadata.get("val_wr"),
            "val_trades_per_day": metadata.get("val_trades_per_day"),
            "test_wr": metadata.get("test_wr"),
            "test_precision": metadata.get("test_precision"),
            "test_trades": metadata.get("test_trades"),
            "test_trades_per_day": metadata.get("test_trades_per_day"),
            "ev_per_day": metadata.get("up_ev_per_day", metadata.get("ev_per_day")),
            "best_iteration": metadata.get("best_iteration"),
            "enabled": True,
            "blocked": metadata.get("blocked", False),
        }
        legacy_down = {
            "threshold": metadata.get("down_threshold"),
            "val_wr": metadata.get("down_val_wr"),
            "val_trades_per_day": metadata.get("down_val_tpd"),
            "test_wr": metadata.get("down_test_wr"),
            "test_precision": metadata.get("down_test_precision"),
            "test_trades": metadata.get("down_test_trades"),
            "test_trades_per_day": metadata.get("down_test_tpd"),
            "ev_per_day": metadata.get("down_ev_per_day"),
            "best_iteration": metadata.get("best_iteration"),
            "enabled": metadata.get("down_enabled", False),
            "blocked": not metadata.get("down_enabled", False),
        }
        normalized = dict(metadata)
        normalized["models"] = {"up": legacy_up, "down": legacy_down}

    normalized["artifact_version"] = max(artifact_version, 1)
    normalized.setdefault("bundle_version", BUNDLE_VERSION if normalized["artifact_version"] >= 2 else 1)
    normalized.setdefault("slot", slot)
    normalized.setdefault("format", "dual_bundle" if normalized["artifact_version"] >= 2 else "legacy_single")

    models = normalized.setdefault("models", {})
    up_meta = dict(models.get("up") or {})
    down_meta = dict(models.get("down") or {})

    if up_meta.get("threshold") is None and normalized.get("threshold") is not None:
        up_meta["threshold"] = normalized.get("threshold")
    if down_meta.get("threshold") is None and normalized.get("down_threshold") is not None:
        down_meta["threshold"] = normalized.get("down_threshold")
    if up_meta.get("enabled") is None:
        up_meta["enabled"] = True
    if down_meta.get("enabled") is None:
        down_meta["enabled"] = normalized.get("down_enabled", False)
    if down_meta.get("blocked") is None:
        down_meta["blocked"] = not bool(down_meta.get("enabled"))

    models["up"] = up_meta
    models["down"] = down_meta

    normalized["threshold"] = up_meta.get("threshold")
    normalized["down_threshold"] = down_meta.get("threshold")
    normalized["down_enabled"] = bool(down_meta.get("enabled"))

    return normalized


def _extract_legacy_model(meta: dict | None, model: lgb.Booster | None) -> dict[str, lgb.Booster] | None:
    if model is None:
        return None
    return {"up": model, "down": model}


def _load_bundle_models_from_disk(slot: str) -> dict[str, lgb.Booster] | None:
    models: dict[str, lgb.Booster] = {}
    for model_key in MODEL_KEYS:
        path = _bundle_model_path(slot, model_key)
        if not os.path.exists(path):
            return None
        try:
            loaded = lgb.Booster(model_file=path)
            loaded = _validate_feature_count(loaded, slot, "load_model_bundle", model_key=model_key)
        except Exception as e:
            log.error("load_model_bundle: failed to load %s: %s", path, e)
            return None
        if loaded is None:
            return None
        models[model_key] = loaded
    return models


def _load_bundle_meta_from_disk(slot: str) -> dict | None:
    path = _bundle_meta_path(slot)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return _normalize_bundle_metadata(json.load(f), slot=slot)
    except Exception as e:
        log.error("load_bundle_meta: failed to load %s: %s", path, e)
        return None


def save_model_bundle(models: dict[str, lgb.Booster], slot: str, metadata: dict) -> None:
    """Save a dual-model bundle to disk with shared metadata."""
    _ensure_dir()
    missing = [key for key in MODEL_KEYS if key not in models or models[key] is None]
    if missing:
        raise ValueError(f"save_model_bundle requires models for {missing}")

    normalized = _normalize_bundle_metadata(metadata, slot=slot)
    normalized["artifact_version"] = BUNDLE_VERSION
    normalized["bundle_version"] = BUNDLE_VERSION
    normalized["format"] = "dual_bundle"

    for model_key, model in models.items():
        model.save_model(_bundle_model_path(slot, model_key))

    with open(_bundle_meta_path(slot), "w") as f:
        json.dump(normalized, f, indent=2)

    log.info("save_model_bundle: saved slot=%s keys=%s", slot, sorted(models.keys()))


def load_model_bundle(slot: str = "current") -> tuple[dict[str, lgb.Booster], dict] | tuple[None, None]:
    """Load bundle models and metadata, falling back to legacy single-model artifacts."""
    bundle_models = _load_bundle_models_from_disk(slot)
    bundle_meta = _load_bundle_meta_from_disk(slot)
    if bundle_models and bundle_meta:
        log.info("load_model_bundle: loaded dual bundle slot=%s from disk", slot)
        return bundle_models, bundle_meta

    legacy_model = load_model(slot)
    legacy_meta = load_metadata(slot)
    if legacy_model is None:
        return None, None
    models = _extract_legacy_model(legacy_meta, legacy_model)
    if models is None:
        return None, None
    normalized = _normalize_bundle_metadata(legacy_meta, slot=slot)
    log.info("load_model_bundle: loaded legacy single-model artifact slot=%s", slot)
    return models, normalized


def save_model(model: lgb.Booster, slot: str, metadata: dict) -> None:
    """Backward-compatible wrapper that stores a legacy single-model artifact as a bundle."""
    save_model_bundle({"up": model, "down": model}, slot, metadata)


def load_model(slot: str = "current") -> lgb.Booster | None:
    """Load the UP model for legacy callers, with bundle-aware fallback."""
    bundle_models = _load_bundle_models_from_disk(slot)
    if bundle_models:
        log.info("load_model: loaded UP model from dual bundle slot=%s", slot)
        return bundle_models.get("up")

    path = _legacy_model_path(slot)
    if not os.path.exists(path):
        log.debug("load_model: no model file at %s", path)
        return None
    try:
        model = lgb.Booster(model_file=path)
        log.info("load_model: loaded legacy slot=%s", slot)
        return _validate_feature_count(model, slot, "load_model", model_key="up")
    except Exception as e:
        log.error("load_model: failed to load %s: %s", path, e)
        return None


def load_metadata(slot: str = "current") -> dict | None:
    """Load normalized metadata for a slot, preferring bundle metadata."""
    meta = _load_bundle_meta_from_disk(slot)
    if meta is not None:
        return meta

    path = _legacy_meta_path(slot)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return _normalize_bundle_metadata(json.load(f), slot=slot)
    except Exception as e:
        log.error("load_metadata: failed to load %s: %s", path, e)
        return None


def promote_candidate() -> None:
    """Copy candidate bundle files to current slot on disk."""
    _ensure_dir()
    bundle_models, bundle_meta = load_model_bundle("candidate")
    if bundle_models is None or bundle_meta is None:
        raise FileNotFoundError("Candidate model bundle not found")
    save_model_bundle(bundle_models, "current", bundle_meta)
    log.info("promote_candidate: copied candidate bundle -> current (disk only)")


async def promote_candidate_in_db() -> None:
    """Promote candidate model bundle to current slot in the database."""
    import aiosqlite
    import config as cfg

    async with aiosqlite.connect(cfg.DB_PATH) as db:
        cursor = await db.execute(
            "SELECT blob, metadata FROM model_blobs WHERE slot = ?",
            ("candidate",),
        )
        row = await cursor.fetchone()

    if not row:
        raise KeyError("No candidate model found in DB - cannot promote to current")

    blob, meta_json = row
    async with aiosqlite.connect(cfg.DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO model_blobs (slot, blob, metadata)
            VALUES (?, ?, ?)
            ON CONFLICT(slot) DO UPDATE SET
                blob=excluded.blob,
                metadata=excluded.metadata,
                updated_at=CURRENT_TIMESTAMP
            """,
            ("current", blob, meta_json),
        )
        await db.commit()

    log.info("promote_candidate_in_db: candidate promoted to current in DB (%d bytes)", len(blob))


def has_model(slot: str = "current") -> bool:
    """Return True if either a dual bundle or legacy single-model artifact exists."""
    if all(os.path.exists(_bundle_model_path(slot, key)) for key in MODEL_KEYS):
        return True
    return os.path.exists(_legacy_model_path(slot))


def delete_model(slot: str) -> None:
    """Delete model files and metadata for the given slot. Safe to call if missing."""
    paths = [_bundle_meta_path(slot), _legacy_meta_path(slot), _legacy_model_path(slot)]
    paths.extend(_bundle_model_path(slot, key) for key in MODEL_KEYS)
    for path in paths:
        try:
            os.remove(path)
            log.info("delete_model: removed %s", path)
        except FileNotFoundError:
            pass


async def save_model_bundle_to_db(models: dict[str, lgb.Booster], slot: str, metadata: dict) -> None:
    """Serialize a dual-model bundle into a single SQLite blob row."""
    import aiosqlite
    import config as cfg

    payload = {
        "artifact_version": BUNDLE_VERSION,
        "models": {key: _serialize_model(model).hex() for key, model in models.items()},
    }
    normalized = _normalize_bundle_metadata(metadata, slot=slot)
    normalized["artifact_version"] = BUNDLE_VERSION
    normalized["bundle_version"] = BUNDLE_VERSION
    normalized["format"] = "dual_bundle"

    blob = json.dumps(payload).encode("utf-8")
    meta_json = json.dumps(normalized)

    async with aiosqlite.connect(cfg.DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO model_blobs (slot, blob, metadata)
            VALUES (?, ?, ?)
            ON CONFLICT(slot) DO UPDATE SET
                blob=excluded.blob,
                metadata=excluded.metadata,
                updated_at=CURRENT_TIMESTAMP
            """,
            (slot, blob, meta_json),
        )
        await db.commit()
    log.info("save_model_bundle_to_db: saved slot=%s (%d bytes)", slot, len(blob))


async def save_model_to_db(model: lgb.Booster, slot: str, metadata: dict) -> None:
    """Backward-compatible wrapper that stores a legacy single-model artifact as a bundle."""
    await save_model_bundle_to_db({"up": model, "down": model}, slot, metadata)


def patch_metadata(slot: str, updates: dict) -> None:
    """Merge updates into normalized on-disk metadata for a slot."""
    meta = load_metadata(slot)
    if meta is None:
        log.debug("patch_metadata: no metadata file for slot=%s, skipping", slot)
        return
    meta.update(updates)
    if "down_override" in updates:
        meta.setdefault("models", {}).setdefault("down", {})
        meta["models"]["down"]["override"] = bool(updates["down_override"])
    meta = _normalize_bundle_metadata(meta, slot=slot)
    path = _bundle_meta_path(slot)
    try:
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info("patch_metadata: patched slot=%s keys=%s", slot, list(updates.keys()))
    except Exception as e:
        log.error("patch_metadata: failed to write %s: %s", path, e)


async def load_model_bundle_from_db(slot: str = "current") -> tuple[dict[str, lgb.Booster], dict] | tuple[None, None]:
    """Load a model bundle from SQLite, supporting legacy single-model rows."""
    import aiosqlite
    import config as cfg

    async with aiosqlite.connect(cfg.DB_PATH) as db:
        cursor = await db.execute(
            "SELECT blob, metadata FROM model_blobs WHERE slot = ?",
            (slot,),
        )
        row = await cursor.fetchone()

    if not row:
        log.info("load_model_bundle_from_db: no blob found for slot=%s", slot)
        return None, None

    blob, meta_json = row
    metadata = None
    try:
        metadata = _normalize_bundle_metadata(json.loads(meta_json), slot=slot) if meta_json else None
    except Exception as e:
        log.error("load_model_bundle_from_db: bad metadata for slot=%s: %s", slot, e)

    try:
        payload = json.loads(blob.decode("utf-8"))
        if payload.get("artifact_version", 1) >= 2 and isinstance(payload.get("models"), dict):
            models: dict[str, lgb.Booster] = {}
            for model_key in MODEL_KEYS:
                model_hex = payload["models"].get(model_key)
                if not model_hex:
                    log.error("load_model_bundle_from_db: missing %s model in slot=%s", model_key, slot)
                    return None, None
                models[model_key] = _deserialize_model(bytes.fromhex(model_hex), slot, "load_model_bundle_from_db", model_key)
                if models[model_key] is None:
                    return None, None
            if metadata is None:
                metadata = _normalize_bundle_metadata({}, slot=slot)
            log.info("load_model_bundle_from_db: loaded dual bundle slot=%s (%d bytes)", slot, len(blob))
            return models, metadata
    except Exception:
        pass

    legacy_model = _deserialize_model(blob, slot, "load_model_from_db", "up")
    if legacy_model is None:
        return None, None
    models = _extract_legacy_model(metadata, legacy_model)
    if models is None:
        return None, None
    if metadata is None:
        metadata = _normalize_bundle_metadata({}, slot=slot)
    log.info("load_model_bundle_from_db: loaded legacy single-model slot=%s (%d bytes)", slot, len(blob))
    return models, metadata


async def load_model_from_db(slot: str = "current") -> lgb.Booster | None:
    """Legacy helper returning the UP model from the bundle loaded from DB."""
    models, _meta = await load_model_bundle_from_db(slot)
    if not models:
        return None
    return models.get("up")
