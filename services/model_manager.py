"""
Handles the full ML model lifecycle:
- Training from scratch (first boot or manual trigger)
- Loading from pkl
- Periodic retraining with accumulated DB data
- Version tracking + rollback
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy.orm import Session
from loguru import logger

from exceptions import ModelNotLoadedError
from services.data_collection_service import build_training_dataframe
from db_models import ModelVersion, ActualReading, PredictionLog

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CURRENT_MODEL_PATH = MODELS_DIR / "battery_model.pkl"
MAX_VERSIONS_TO_KEEP = 3

_model_data = None

FEATURE_COLS = ["power_usage_kW", "electricity_tariff_INR_per_kWh", "ac_power_output_kW"]
TARGET_COL = "battery_level_kWh"


def _prepare_features(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_model(
    db: Session,
    latitude: float = 28.6,
    longitude: float = 77.2,
    months_back: int = 3,
    notes: str = "Initial training",
) -> dict:
    """
    Trains a new Random Forest model on:
      1. Real ActualReading rows (legacy)
      2. PredictionLog rows (new — full per-request snapshots; each generates 24 samples)
      3. API-generated historical data (top-up / fallback)
    """
    logger.info(f"Training new model — {notes}")

    db_df = _load_actual_readings_from_db(db)
    log_df = _load_prediction_logs_as_training(db)

    logger.info("Fetching historical data from Open-Meteo API...")
    try:
        api_df = build_training_dataframe(
            latitude=latitude, longitude=longitude, months_back=months_back
        )
    except Exception as e:
        logger.warning(f"API top-up failed: {e} — proceeding without it")
        api_df = pd.DataFrame()

    frames = []
    if len(api_df) > 0:
        frames.append(api_df[FEATURE_COLS + [TARGET_COL]] if all(c in api_df.columns for c in FEATURE_COLS + [TARGET_COL]) else api_df)
    if len(db_df) > 0:
        # Real readings weighted heavier (duplicate 3x)
        frames.extend([db_df, db_df, db_df])
    if len(log_df) > 0:
        # Per-request logs weighted heavier too (duplicate 2x)
        frames.extend([log_df, log_df])

    if not frames:
        msg = "No training data available — aborting"
        logger.error(msg)
        return {"promoted": False, "message": msg, "training_samples": 0}

    combined_df = pd.concat(frames, ignore_index=True).dropna()
    logger.info(
        f"Total training samples: {len(combined_df)} "
        f"(api={len(api_df)}, readings={len(db_df)}, logs={len(log_df)})"
    )

    if len(combined_df) < 10:
        msg = f"Too few training samples ({len(combined_df)}) — aborting"
        logger.error(msg)
        return {"promoted": False, "message": msg, "training_samples": len(combined_df)}

    X, y = _prepare_features(combined_df)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    logger.info("Fitting Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    logger.info(f"Model metrics — RMSE: {rmse:.4f} | R²: {r2:.4f}")

    should_promote = _should_promote_new_model(db, r2)
    if not should_promote:
        logger.warning("New model is WORSE than current. Keeping existing model.")
        return {
            "promoted": False,
            "rmse": rmse,
            "r2": r2,
            "training_samples": len(combined_df),
            "message": "New model did not improve on current — existing model kept",
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_number = _get_next_version(db)
    pkl_path = MODELS_DIR / f"battery_model_v{version_number}_{timestamp}.pkl"

    model_data = {"model": model, "scaler": scaler}
    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f)
    with open(CURRENT_MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved: {pkl_path}")

    try:
        db.query(ModelVersion).filter(ModelVersion.is_active == True).update(
            {"is_active": False}
        )
        version_record = ModelVersion(
            version=version_number,
            training_samples=len(combined_df),
            rmse=rmse,
            r2_score=r2,
            is_active=True,
            pkl_path=str(pkl_path),
            notes=notes,
        )
        db.add(version_record)

        # Mark prediction logs as consumed
        db.query(PredictionLog).filter(PredictionLog.used_for_training == False).update(
            {"used_for_training": True, "model_version": version_number}
        )
        db.commit()
        logger.info(f"Model version {version_number} recorded in DB")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to record model version in DB: {e}")

    load_model()
    _cleanup_old_versions(db)

    return {
        "promoted": True,
        "version": version_number,
        "rmse": rmse,
        "r2": r2,
        "training_samples": len(combined_df),
        "pkl_path": str(pkl_path),
        "message": f"Model v{version_number} trained and promoted successfully",
    }


# ─── LOADING ──────────────────────────────────────────────────────────────────

def load_model():
    global _model_data
    if not CURRENT_MODEL_PATH.exists():
        raise ModelNotLoadedError()
    try:
        with open(CURRENT_MODEL_PATH, "rb") as f:
            _model_data = pickle.load(f)
        assert "model" in _model_data
        assert "scaler" in _model_data
        logger.info(f"✅ Model loaded from {CURRENT_MODEL_PATH}")
    except ModelNotLoadedError:
        raise
    except Exception as e:
        raise ModelNotLoadedError() from e


def ensure_model_loaded(db: Session, latitude: float = 28.6, longitude: float = 77.2):
    global _model_data
    if CURRENT_MODEL_PATH.exists():
        logger.info("pkl found — loading existing model")
        load_model()
    else:
        logger.warning("pkl NOT found — training from scratch (this takes 1-2 minutes)")
        train_model(
            db=db, latitude=latitude, longitude=longitude,
            notes="Auto-trained on first boot",
        )


# ─── PREDICTION ───────────────────────────────────────────────────────────────

def predict_battery_levels(
    power_usage: list,
    tariffs: list,
    solar_outputs: list,
    initial_battery_pct: float = 50.0,
    battery_capacity: float = 15.0,
) -> list:
    if _model_data is None:
        raise ModelNotLoadedError()

    model = _model_data["model"]
    scaler = _model_data["scaler"]
    battery_levels = []

    for i in range(24):
        features = np.array([[power_usage[i], tariffs[i], solar_outputs[i]]])
        features_scaled = scaler.transform(features)
        predicted = float(model.predict(features_scaled)[0])
        predicted = max(0.0, min(predicted, battery_capacity))
        battery_levels.append(round(predicted, 3))

    return battery_levels


# ─── DATA LOADERS ─────────────────────────────────────────────────────────────

def save_actual_reading(
    db: Session,
    location: str,
    hour: int,
    power_usage_kw: float,
    solar_output_kw: float,
    tariff_value: float,
    battery_level_kwh: float,
    source: str = "api",
):
    try:
        reading = ActualReading(
            recorded_at=datetime.utcnow(),
            location=location,
            hour=hour,
            power_usage_kw=power_usage_kw,
            solar_output_kw=solar_output_kw,
            tariff_value=tariff_value,
            battery_level_kwh=battery_level_kwh,
            source=source,
        )
        db.add(reading)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"Failed to save actual reading: {e}")


def _load_actual_readings_from_db(db: Session) -> pd.DataFrame:
    try:
        readings = db.query(ActualReading).all()
        if not readings:
            return pd.DataFrame()
        df = pd.DataFrame([
            {
                "power_usage_kW": r.power_usage_kw,
                "electricity_tariff_INR_per_kWh": r.tariff_value,
                "ac_power_output_kW": r.solar_output_kw,
                "battery_level_kWh": r.battery_level_kwh,
            } for r in readings
        ])
        logger.info(f"Loaded {len(df)} actual readings from DB")
        return df
    except Exception as e:
        logger.error(f"Failed to load actual readings from DB: {e}")
        return pd.DataFrame()


def _load_prediction_logs_as_training(db: Session) -> pd.DataFrame:
    """
    Each PredictionLog row contains 24 hours of (consumption + device load),
    tariff, solar, and battery forecast. We explode each into 24 training rows.
    """
    try:
        logs = db.query(PredictionLog).all()
        if not logs:
            return pd.DataFrame()

        rows = []
        for log in logs:
            consumption = log.consumption or []
            tariffs = log.tariffs or []
            solar = log.solar_outputs or []
            battery = log.battery_forecast or []
            devices = log.devices or []
            schedule = log.hourly_schedule or []

            if not (len(consumption) == len(tariffs) == len(solar) == len(battery) == 24):
                continue

            for h in range(24):
                # Add the device load running at hour h (from the optimised schedule)
                device_load = 0.0
                if h < len(schedule):
                    dev_flags = schedule[h].get("devices", {})
                    for d in devices:
                        if dev_flags.get(d.get("name")):
                            device_load += float(d.get("power_rating_kw", 0.0))

                rows.append({
                    "power_usage_kW": consumption[h] + device_load,
                    "electricity_tariff_INR_per_kWh": tariffs[h],
                    "ac_power_output_kW": solar[h],
                    "battery_level_kWh": battery[h],
                })

        df = pd.DataFrame(rows)
        logger.info(f"Loaded {len(df)} training samples from {len(logs)} prediction logs")
        return df
    except Exception as e:
        logger.error(f"Failed to load prediction logs as training data: {e}")
        return pd.DataFrame()


def _should_promote_new_model(db: Session, new_r2: float) -> bool:
    current_active = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    if not current_active:
        return True
    improvement = new_r2 - current_active.r2_score
    logger.info(
        f"Model comparison — current R²: {current_active.r2_score:.4f} vs new R²: {new_r2:.4f} "
        f"(improvement: {improvement:+.4f})"
    )
    return improvement >= 0.01


def _get_next_version(db: Session) -> int:
    latest = db.query(ModelVersion).order_by(ModelVersion.version.desc()).first()
    return (latest.version + 1) if latest else 1


def _cleanup_old_versions(db: Session):
    versions = db.query(ModelVersion).order_by(ModelVersion.trained_at.desc()).all()
    versions_to_delete = versions[MAX_VERSIONS_TO_KEEP:]
    for v in versions_to_delete:
        pkl_path = Path(v.pkl_path)
        if pkl_path.exists() and pkl_path != CURRENT_MODEL_PATH:
            pkl_path.unlink()
            logger.info(f"Deleted old model version: {pkl_path}")
    logger.info(f"Keeping {min(len(versions), MAX_VERSIONS_TO_KEEP)} model versions")
