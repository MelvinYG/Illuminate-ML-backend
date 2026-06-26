import sys
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from loguru import logger
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from database import SessionLocal

from logger_setup import setup_logging
from middleware import RequestIDMiddleware
from database import get_db, test_connection, engine, Base
from db_models import OptimizationResult, Notification, ModelVersion
from exceptions import (
    IlluminateError,
    WeatherAPIError,
    ModelNotLoadedError,
    OptimizationFailedError,
    InvalidModeError,
    UserNotFoundError
)
from services.weather_service import fetch_weather
from services.solar_service import calculate_solar_output
from services.optimizer_service import run_optimization, MODE_CONFIG
from services.model_manager import ensure_model_loaded, train_model, predict_battery_levels

# --- LOGGING SETUP ---
setup_logging()
load_dotenv()

scheduler = AsyncIOScheduler()

# --- APP LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""

    logger.info("=" * 50)
    logger.info("🚀 Illuminate ML Service starting up")
    logger.info("=" * 50)

    # 1. Test DB connection
    try:
        test_connection()
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database ready")
    except Exception as e:
        logger.critical(f"❌ Database failed at startup: {e}")
        sys.exit(1)  # Hard fail — no point running without DB

    # 2. Model — train if pkl missing, otherwise just load
    db = SessionLocal()

    try:
        ensure_model_loaded(db)
        logger.info("✅ ML model ready")
    except Exception as e:
        logger.critical(f"❌ Model failed: {e}")
        sys.exit(1)
    finally:
        db.close()

    # 3. Scheduler — weekly retrain every Sunday at 2 AM
    @scheduler.scheduled_job("cron", day_of_week="sun", hour=2, minute=0)
    def weekly_retrain():
        logger.info("⏰ Weekly retrain triggered by scheduler")
        db = SessionLocal()
        try:
            result = train_model(db=db, notes="Scheduled weekly retrain")
            logger.info(f"Weekly retrain result: {result['message']}")
        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}")
        finally:
            db.close()
    
    scheduler.start()
    logger.info("✅ Weekly retrainer scheduled (Sundays 2 AM)")
    logger.info("✅ All systems go")
    logger.info("=" * 50)

    yield  # Server runs here
    
    scheduler.shutdown()
    logger.info("🛑 Illuminate ML Service shutting down")

app = FastAPI(
    title="Illuminate ML Service",
    description="Smart Home Energy Optimization API",
    version="1.0.0",
    lifespan=lifespan
)

# todo: change this to backend url of illuminate core service
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL EXCEPTION HANDLERS ---
# These catch exceptions and return clean JSON instead of ugly 500 pages

@app.exception_handler(IlluminateError)
async def illuminate_error_handler(request: Request, exc: IlluminateError):
    """Handles all our custom exceptions."""
    logger.error(f"IlluminateError [{exc.status_code}]: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Catches anything that slips through — last resort."""
    logger.exception(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "Something went wrong. Our team has been notified.",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


# --- REQUEST / RESPONSE MODELS ---
VALID_MODES = list(MODE_CONFIG.keys())
VALID_APPLIANCES = ["washing_machine", "dishwasher", "pump"]

# --- REQUEST / RESPONSE MODELS ---
class UserSettings(BaseModel):
    """
    All configurable user settings.
    Maps directly to the Settings page in the frontend.
    """
    # Mode from settings page
    mode: str = "tou_savings"
    # Options: self_consumption | tou_savings | full_backup | low_power

    # Battery settings
    battery_capacity_kwh: float = 15.0
    min_battery_reserve_pct: float = 20.0  # Never go below this %
    # Used in full_backup mode especially

    # Solar settings
    solar_capacity_kw: float = 100.0
    has_net_metering: bool = True    # Can user sell back to grid?

    # Appliances the user has registered
    appliances: List[str] = ["washing_machine", "dishwasher", "pump"]

    # Local tariff preferences
    peak_start_hour: int = 17        # 5 PM default
    peak_end_hour: int = 21          # 9 PM default
    peak_tariff: float = 7.5
    offpeak_tariff: float = 3.5
    normal_tariff: float = 5.5

    @validator("mode")
    def mode_must_be_valid(cls, v):
        valid = ["self_consumption", "tou_savings", "full_backup", "low_power"]
        if v not in valid:
            raise ValueError(f"mode must be one of {valid}")
        return v

    @validator("min_battery_reserve_pct")
    def reserve_must_be_valid(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("min_battery_reserve_pct must be 0-100")
        return v


class OptimizeRequest(BaseModel):
    # Who is requesting
    user_id: Optional[str] = None

    # Where they are
    location: str = "Delhi,IN"
    latitude: float = 28.6
    longitude: float = 77.2

    # Current state
    current_battery_level_pct: float = 50.0

    # All settings — can be passed as a nested object
    settings: UserSettings = UserSettings()

    @validator("current_battery_level_pct")
    def battery_pct_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("current_battery_level_pct must be 0-100")
        return v

class OptimizeResponse(BaseModel):
    status: str
    mode: str
    mode_description: str
    total_grid_cost: float
    recommendations: dict
    hourly_schedule: List[dict]
    weather: dict
    tariffs: List[float]
    execution_time_ms: float

@app.get("/health")
def health(db: Session = Depends(get_db)):
    """Health check — tests DB connection too."""
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "service": "illuminate-ml",
        "version": "1.0.0",
        "database": db_status,
    }

@app.post("/predict")
def predict(request: OptimizeRequest, db: Session = Depends(get_db)):
    start_time = time.time()

    logger.info(
        f"Predict request | location={request.location} "
        f"mode={request.settings.mode} user={request.user_id or 'anonymous'}"
    )

    # Step 1 — Weather
    logger.info("Step 1/5: Fetching weather...")
    weather = fetch_weather(request.location, db)
    # fetch_weather raises WeatherAPIError internally if it fails

    # Step 2 — Solar
    logger.info("Step 2/5: Calculating solar output...")
    solar_outputs = calculate_solar_output(weather, request.settings.solar_capacity_kw)

    # Step 3 — Tariffs
    logger.info("Step 3/5: Building tariff schedule...")
    tariffs = [
        4.0 if (0 <= h < 6 or 14 <= h < 17)
        else 7.5 if (17 <= h < 21)
        else 5.5
        for h in range(24)
    ]

    # Step 4 — Consumption
    consumption = [
        5.0 if (18 <= h <= 21)
        else 3.5 if (22 <= h <= 23)
        else 2.0 if (0 <= h <= 4)
        else 3.0 if (5 <= h <= 8)
        else 2.5
        for h in range(24)
    ]

    # Step 5 — Battery prediction
    logger.info("Step 4/5: Predicting battery levels...")
    initial_battery_kwh = (
        request.current_battery_level_pct / 100
    ) * request.settings.battery_capacity_kwh

    battery_levels = predict_battery_levels(
        power_usage=consumption,
        tariffs=tariffs,
        solar_outputs=solar_outputs,
        initial_battery_pct=request.current_battery_level_pct,
        battery_capacity=request.settings.battery_capacity_kwh
    )

    # Step 6 — Optimize
    logger.info(f"Step 5/5: Running optimizer in '{request.settings.mode}' mode...")
    result = run_optimization(
        tariffs=tariffs,
        solar_outputs=solar_outputs,
        consumption=consumption,
        initial_battery=initial_battery_kwh,
        battery_capacity=request.settings.battery_capacity_kwh,
        appliances=request.settings.appliances,
        mode=request.settings.mode
    )

    if result["status"] not in ["Optimal", "optimal"]:
        raise OptimizationFailedError(result["status"])

    # Step 7 — Persist to DB
    if request.user_id:
        try:
            opt_record = OptimizationResult(
                user_id=request.user_id,
                mode=request.settings.mode,
                total_grid_cost=result["total_grid_cost"],
                recommendations=result["recommendations"],
                hourly_schedule=result["hourly_schedule"],
                estimated_savings=abs(result["total_grid_cost"])
            )
            db.add(opt_record)

            # Save recommendations as notifications
            for appliance, recommendation in result["recommendations"].items():
                if "Not scheduled" not in recommendation:
                    notif = Notification(
                        user_id=request.user_id,
                        message=f"{appliance.replace('_', ' ').title()}: {recommendation}",
                        type="suggestion"
                    )
                    db.add(notif)

            db.commit()
            logger.info("Results persisted to DB")

        except Exception as e:
            # DB save failing shouldn't crash the response
            # Log it but still return the optimization result
            db.rollback()
            logger.error(f"Failed to save results to DB: {e} — continuing anyway")

    execution_ms = round((time.time() - start_time) * 1000, 2)
    current_weather = weather[0] if weather else {}

    logger.info(f"✅ Predict complete in {execution_ms}ms")

    return {
        **result,
        "weather": {
            "temp": current_weather.get("temp"),
            "humidity": current_weather.get("humidity"),
            "cloud_cover": current_weather.get("cloud_cover"),
        },
        "tariffs": tariffs,
        "battery_forecast": battery_levels,
        "execution_time_ms": execution_ms
    }


@app.get("/notifications/{user_id}")
def get_notifications(user_id: str, db: Session = Depends(get_db)):
    notifications = db.query(Notification)\
                      .filter(
                          Notification.user_id == user_id,
                          Notification.is_read == False
                      )\
                      .order_by(Notification.created_at.desc())\
                      .all()

    logger.info(f"Fetched {len(notifications)} notifications for user {user_id}")
    return [
        {
            "id": str(n.id),
            "message": n.message,
            "type": n.type,
            "created_at": n.created_at.isoformat()
        }
        for n in notifications
    ]


@app.get("/history/{user_id}")
def get_history(user_id: str, limit: int = 30, db: Session = Depends(get_db)):
    results = db.query(OptimizationResult)\
                .filter(OptimizationResult.user_id == user_id)\
                .order_by(OptimizationResult.created_at.desc())\
                .limit(min(limit, 100))\
                .all()

    logger.info(f"Fetched {len(results)} history records for user {user_id}")
    return [
        {
            "id": str(r.id),
            "created_at": r.created_at.isoformat(),
            "mode": r.mode,
            "total_grid_cost": r.total_grid_cost,
            "estimated_savings": r.estimated_savings,
            "recommendations": r.recommendations
        }
        for r in results
    ]

@app.post("/retrain")
def trigger_retrain(
    latitude: float = 28.6,
    longitude: float = 77.2,
    months_back: int = 3,
    db: Session = Depends(get_db)
):
    """
    Manually trigger model retraining.
    Useful after accumulating significant new data,
    or if you want to retrain before the weekly schedule.
    """
    logger.info("Manual retrain triggered via API")
    result = train_model(
        db=db,
        latitude=latitude,
        longitude=longitude,
        months_back=months_back,
        notes="Manual retrain via API"
    )
    return result


@app.get("/model/versions")
def get_model_versions(db: Session = Depends(get_db)):
    """Shows all trained model versions — useful to track improvement."""
    versions = db.query(ModelVersion)\
                 .order_by(ModelVersion.trained_at.desc())\
                 .all()
    return [
        {
            "version": v.version,
            "trained_at": v.trained_at.isoformat(),
            "training_samples": v.training_samples,
            "rmse": v.rmse,
            "r2_score": v.r2_score,
            "is_active": v.is_active,
            "notes": v.notes
        }
        for v in versions
    ]