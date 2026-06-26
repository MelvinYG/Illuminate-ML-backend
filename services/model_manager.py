"""
Handles the full ML model lifecycle:
- Training from scratch (first boot or manual trigger)
- Loading from pkl
- Periodic retraining with accumulated DB data
- Version tracking + rollback
"""

import pickle
import os
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

from exceptions import ModelNotLoadedError, DatabaseError
from services.data_collection_service import build_training_dataframe
from db_models import ModelVersion, ActualReading

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CURRENT_MODEL_PATH = MODELS_DIR / "battery_model.pkl"
MAX_VERSIONS_TO_KEEP = 3   # Keep last 3 pkl files

# Module-level — one copy in memory, shared across all requests
_model_data = None


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

FEATURE_COLS = ["power_usage_kW", "electricity_tariff_INR_per_kWh", "ac_power_output_kW"]
TARGET_COL = "battery_level_kWh"

def _prepare_features(df: pd.DataFrame):
    """Extracts and scales features + target from DataFrame."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_model(
    db: Session,
    latitude: float = 28.6,
    longitude: float = 77.2,
    months_back: int = 3,
    notes: str = "Initial training"
) -> dict:
    """
    Trains a new Random Forest model.
    Data sources (in order of preference):
      1. Accumulated real readings from DB (most valuable)
      2. API-generated historical data (fallback / top-up)

    Saves new pkl, records version in DB.
    Returns metrics dict.
    """
    logger.info(f"Training new model — {notes}")

    # ── Step 1: Get accumulated real data from DB ──
    db_df = _load_actual_readings_from_db(db)

    # ── Step 2: Get API data to supplement ──
    logger.info("Fetching historical data from Open-Meteo API...")
    api_df = build_training_dataframe(
        latitude=latitude,
        longitude=longitude,
        months_back=months_back
    )

    # ── Step 3: Combine both sources ──
    # DB data is more valuable (real readings) so we weight it
    # by duplicating it — a simple but effective technique
    if len(db_df) > 0:
        logger.info(f"Combining {len(db_df)} DB readings + {len(api_df)} API rows")
        # Duplicate DB readings 3x to give them more weight in training
        combined_df = pd.concat([api_df, db_df, db_df, db_df], ignore_index=True)
    else:
        logger.info(f"No DB readings yet — training on {len(api_df)} API rows only")
        combined_df = api_df

    combined_df = combined_df.dropna()
    logger.info(f"Total training samples: {len(combined_df)}")

    # ── Step 4: Prepare features ──
    X, y = _prepare_features(combined_df)

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ── Step 5: Train ──
    logger.info("Fitting Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1      # Use all CPU cores
    )
    model.fit(X_train, y_train)

    # ── Step 6: Evaluate ──
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    logger.info(f"Model metrics — RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # ── Step 7: Decide whether to promote ──
    # Compare against current active model if one exists
    should_promote = _should_promote_new_model(db, r2)

    if not should_promote:
        logger.warning(
            "New model is WORSE than current. Keeping existing model."
        )
        return {
            "promoted": False,
            "rmse": rmse,
            "r2": r2,
            "training_samples": len(combined_df),
            "message": "New model did not improve on current — existing model kept"
        }

    # ── Step 8: Save pkl ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_number = _get_next_version(db)
    pkl_path = MODELS_DIR / f"battery_model_v{version_number}_{timestamp}.pkl"

    model_data = {"model": model, "scaler": scaler}
    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f)

    # Also save as current (overwrite)
    with open(CURRENT_MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved: {pkl_path}")

    # ── Step 9: Record version in DB ──
    try:
        # Deactivate all previous versions
        db.query(ModelVersion).filter(ModelVersion.is_active == True)\
          .update({"is_active": False})

        # Record new version
        version_record = ModelVersion(
            version=version_number,
            training_samples=len(combined_df),
            rmse=rmse,
            r2_score=r2,
            is_active=True,
            pkl_path=str(pkl_path),
            notes=notes
        )
        db.add(version_record)
        db.commit()
        logger.info(f"Model version {version_number} recorded in DB")

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to record model version in DB: {e}")

    # ── Step 10: Load new model into memory ──
    load_model()

    # ── Step 11: Clean up old pkl files ──
    _cleanup_old_versions(db)

    return {
        "promoted": True,
        "version": version_number,
        "rmse": rmse,
        "r2": r2,
        "training_samples": len(combined_df),
        "pkl_path": str(pkl_path),
        "message": f"Model v{version_number} trained and promoted successfully"
    }


# ─── LOADING ──────────────────────────────────────────────────────────────────

def load_model():
    """
    Loads model from pkl into memory.
    Call at startup. Fails loudly if pkl missing.
    """
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
    """
    Called at startup.
    If pkl exists → load it.
    If pkl missing → train from scratch first, then load.
    This is the 'smart startup' behaviour you asked for.
    """
    global _model_data

    if CURRENT_MODEL_PATH.exists():
        logger.info("pkl found — loading existing model")
        load_model()
    else:
        logger.warning("pkl NOT found — training from scratch (this takes 1-2 minutes)")
        logger.warning("This only happens on first boot or after clearing models/")
        train_model(
            db=db,
            latitude=latitude,
            longitude=longitude,
            notes="Auto-trained on first boot"
        )
        # train_model calls load_model() at the end, so model is in memory


# ─── PREDICTION ───────────────────────────────────────────────────────────────

def predict_battery_levels(
    power_usage: list,
    tariffs: list,
    solar_outputs: list,
    initial_battery_pct: float = 50.0,
    battery_capacity: float = 15.0
) -> list:
    """Predict battery levels for 24 hours."""
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


# ─── INCREMENTAL LEARNING HELPERS ─────────────────────────────────────────────

def save_actual_reading(
    db: Session,
    location: str,
    hour: int,
    power_usage_kw: float,
    solar_output_kw: float,
    tariff_value: float,
    battery_level_kwh: float,
    source: str = "api"
):
    """
    Saves one real hourly reading to DB.
    Called every time we get real data — this is how the model
    accumulates real-world data to learn from over time.
    """
    try:
        reading = ActualReading(
            recorded_at=datetime.utcnow(),
            location=location,
            hour=hour,
            power_usage_kw=power_usage_kw,
            solar_output_kw=solar_output_kw,
            tariff_value=tariff_value,
            battery_level_kwh=battery_level_kwh,
            source=source
        )
        db.add(reading)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"Failed to save actual reading: {e}")


def _load_actual_readings_from_db(db: Session) -> pd.DataFrame:
    """
    Loads all accumulated real readings from DB.
    Converts to DataFrame for model training.
    """
    try:
        readings = db.query(ActualReading).all()
        if not readings:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "power_usage_kW": r.power_usage_kw,
            "electricity_tariff_INR_per_kWh": r.tariff_value,
            "ac_power_output_kW": r.solar_output_kw,
            "battery_level_kWh": r.battery_level_kwh
        } for r in readings])

        logger.info(f"Loaded {len(df)} actual readings from DB")
        return df

    except Exception as e:
        logger.error(f"Failed to load actual readings from DB: {e}")
        return pd.DataFrame()


def _should_promote_new_model(db: Session, new_r2: float) -> bool:
    """
    Checks if new model is better than current.
    Uses R² score — higher is better.
    """
    current_active = db.query(ModelVersion)\
                       .filter(ModelVersion.is_active == True)\
                       .first()

    if not current_active:
        return True  # No existing model — always promote first one

    improvement = new_r2 - current_active.r2_score
    logger.info(
        f"Model comparison — current R²: {current_active.r2_score:.4f} "
        f"vs new R²: {new_r2:.4f} (improvement: {improvement:+.4f})"
    )

    # Promote if new model is at least 1% better (avoids noise-based changes)
    return improvement >= 0.01


def _get_next_version(db: Session) -> int:
    """Gets the next version number."""
    latest = db.query(ModelVersion)\
               .order_by(ModelVersion.version.desc())\
               .first()
    return (latest.version + 1) if latest else 1


def _cleanup_old_versions(db: Session):
    """
    Keeps only last MAX_VERSIONS_TO_KEEP pkl files on disk.
    Deletes older ones to save storage.
    """
    versions = db.query(ModelVersion)\
                 .order_by(ModelVersion.trained_at.desc())\
                 .all()

    versions_to_delete = versions[MAX_VERSIONS_TO_KEEP:]

    for v in versions_to_delete:
        pkl_path = Path(v.pkl_path)
        if pkl_path.exists() and pkl_path != CURRENT_MODEL_PATH:
            pkl_path.unlink()  # Delete file
            logger.info(f"Deleted old model version: {pkl_path}")

    logger.info(f"Keeping {min(len(versions), MAX_VERSIONS_TO_KEEP)} model versions")


def predict_battery_levels(
    power_usage: list,
    tariffs: list,
    solar_outputs: list,
    initial_battery_pct: float = 50.0,
    battery_capacity: float = 15.0
) -> list:
    logger.info(f"_model_data = {_model_data}")
    
    if _model_data is None:
        raise ModelNotLoadedError()

    model = _model_data["model"]
    scaler = _model_data["scaler"]

    battery_levels = []

    for i in range(24):
        features = np.array([[power_usage[i], tariffs[i], solar_outputs[i]]])
        features_scaled = scaler.transform(features)
        predicted = model.predict(features_scaled)[0]
        predicted = max(0, min(predicted, battery_capacity))
        battery_levels.append(round(predicted, 3))

    return battery_levels
