import sys
import time
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict
import uuid as _uuid


def _is_uuid(value: str) -> bool:
    try:
        _uuid.UUID(str(value))
        return True
    except (ValueError, TypeError, AttributeError):
        return False

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from logger_setup import setup_logging
from middleware import RequestIDMiddleware
from database import SessionLocal, get_db, test_connection, engine, Base
from db_models import (
    OptimizationResult,
    Notification,
    ModelVersion,
    PredictionLog,
    AdminUser,
)
from exceptions import (
    IlluminateError,
    OptimizationFailedError,
)
from services.weather_service import fetch_weather
from services.solar_service import calculate_solar_output
from services.optimizer_service import run_optimization, MODE_CONFIG
from services.model_manager import ensure_model_loaded, train_model, predict_battery_levels
from services.auth_service import (
    seed_admin,
    verify_password,
    create_access_token,
    get_current_admin,
)

setup_logging()
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("🚀 Illuminate ML Service starting up")
    logger.info("=" * 50)

    try:
        test_connection()
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database ready")
    except Exception as e:
        logger.critical(f"❌ Database failed at startup: {e}")
        sys.exit(1)

    db = SessionLocal()
    try:
        seed_admin(db)
        ensure_model_loaded(db)
        logger.info("✅ ML model ready")
    except Exception as e:
        logger.critical(f"❌ Model failed: {e}")
        sys.exit(1)
    finally:
        db.close()

    @scheduler.scheduled_job("cron", day_of_week="sun", hour=2, minute=0)
    def weekly_retrain():
        logger.info("⏰ Weekly retrain triggered by scheduler")
        db = SessionLocal()
        try:
            result = train_model(db=db, notes="Scheduled weekly retrain")
            logger.info(f"Weekly retrain result: {result.get('message')}")
        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}")
        finally:
            db.close()

    scheduler.start()
    logger.info("✅ Weekly retrainer scheduled (Sundays 2 AM)")
    logger.info("=" * 50)

    yield
    scheduler.shutdown()
    logger.info("🛑 Illuminate ML Service shutting down")


app = FastAPI(
    title="Illuminate ML Service",
    description="Smart Home Energy Optimization API",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Global handlers ---------------------------------------------------------
@app.exception_handler(IlluminateError)
async def illuminate_error_handler(request: Request, exc: IlluminateError):
    logger.error(f"IlluminateError [{exc.status_code}]: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "Something went wrong. Our team has been notified.",
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


# --- Pydantic schemas --------------------------------------------------------
VALID_MODES = list(MODE_CONFIG.keys())


class Device(BaseModel):
    """A single home appliance / device the user wants to schedule."""
    name: str
    power_rating_kw: float = Field(..., gt=0, description="Device power rating in kW")
    usage_hours: int = Field(1, ge=0, le=24, description="Total hours/day device should run")
    priority: str = Field("normal", description="low | normal | high")
    earliest_hour: Optional[int] = Field(0, ge=0, le=23)
    latest_hour: Optional[int] = Field(24, ge=1, le=24)
    contiguous: bool = Field(False, description="Must run in consecutive hours")
    category: Optional[str] = Field(None, description="ac|fan|light|bulb|washing_machine|dishwasher|fridge|water_heater|microwave|tv|ev_charger|pump|other")

    @validator("priority")
    def _priority_valid(cls, v):
        if v.lower() not in ("low", "normal", "high"):
            raise ValueError("priority must be one of: low, normal, high")
        return v.lower()


class UserSettings(BaseModel):
    mode: str = "tou_savings"
    battery_capacity_kwh: float = 15.0
    min_battery_reserve_pct: float = 20.0
    solar_capacity_kw: float = 100.0
    has_net_metering: bool = True
    peak_start_hour: int = 17
    peak_end_hour: int = 21
    peak_tariff: float = 7.5
    offpeak_tariff: float = 3.5
    normal_tariff: float = 5.5

    @validator("mode")
    def mode_must_be_valid(cls, v):
        if v not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        return v

    @validator("min_battery_reserve_pct")
    def reserve_must_be_valid(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("min_battery_reserve_pct must be 0-100")
        return v


class OptimizeRequest(BaseModel):
    user_id: Optional[str] = None
    location: str = "Delhi,IN"
    latitude: float = 28.6
    longitude: float = 77.2
    current_battery_level_pct: float = 50.0
    settings: UserSettings = UserSettings()
    devices: List[Device] = Field(default_factory=list)

    @validator("current_battery_level_pct")
    def battery_pct_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("current_battery_level_pct must be 0-100")
        return v


class AdminLoginRequest(BaseModel):
    email: str
    password: str


class RetrainRequest(BaseModel):
    latitude: float = 28.6
    longitude: float = 77.2
    months_back: int = 3
    notes: str = "Manual retrain via API"


# --- Health ------------------------------------------------------------------
@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "service": "illuminate-ml",
        "version": "1.1.0",
        "database": db_status,
    }


# --- Admin auth --------------------------------------------------------------
@app.post("/admin/login")
def admin_login(payload: AdminLoginRequest, db: Session = Depends(get_db)):
    admin = db.query(AdminUser).filter(AdminUser.email == payload.email.lower()).first()
    if admin is None or not verify_password(payload.password, admin.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(str(admin.id), admin.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "admin": {"id": str(admin.id), "email": admin.email, "name": admin.name},
    }


@app.get("/admin/me")
def admin_me(admin: AdminUser = Depends(get_current_admin)):
    return {"id": str(admin.id), "email": admin.email, "name": admin.name, "role": admin.role}


# --- Predict -----------------------------------------------------------------
@app.post("/predict")
def predict(request: OptimizeRequest, db: Session = Depends(get_db)):
    start_time = time.time()

    logger.info(
        f"Predict request | location={request.location} mode={request.settings.mode} "
        f"devices={len(request.devices)} user={request.user_id or 'anonymous'}"
    )

    # 1. Weather (already cached in DB on each fetch)
    logger.info("Step 1/5: Fetching weather...")
    weather = fetch_weather(request.location, db)

    # 2. Solar
    logger.info("Step 2/5: Calculating solar output...")
    solar_outputs = calculate_solar_output(weather, request.settings.solar_capacity_kw)

    # 3. Tariffs
    logger.info("Step 3/5: Building tariff schedule...")
    tariffs = [
        4.0 if (0 <= h < 6 or 14 <= h < 17)
        else 7.5 if (17 <= h < 21)
        else 5.5
        for h in range(24)
    ]

    # 4. Baseline consumption (always-on background load: lights, fridge etc.)
    consumption = [
        5.0 if (18 <= h <= 21)
        else 3.5 if (22 <= h <= 23)
        else 2.0 if (0 <= h <= 4)
        else 3.0 if (5 <= h <= 8)
        else 2.5
        for h in range(24)
    ]

    # 5. Battery forecast (ML)
    logger.info("Step 4/5: Predicting battery levels...")
    initial_battery_kwh = (
        request.current_battery_level_pct / 100
    ) * request.settings.battery_capacity_kwh
    battery_levels = predict_battery_levels(
        power_usage=consumption,
        tariffs=tariffs,
        solar_outputs=solar_outputs,
        initial_battery_pct=request.current_battery_level_pct,
        battery_capacity=request.settings.battery_capacity_kwh,
    )

    # 6. Optimise (with user devices)
    logger.info(f"Step 5/5: Running optimizer in '{request.settings.mode}' mode...")
    devices_list = [d.dict() for d in request.devices]
    result = run_optimization(
        tariffs=tariffs,
        solar_outputs=solar_outputs,
        consumption=consumption,
        initial_battery=initial_battery_kwh,
        battery_capacity=request.settings.battery_capacity_kwh,
        devices=devices_list,
        mode=request.settings.mode,
    )

    if result["status"] not in ("Optimal", "optimal"):
        raise OptimizationFailedError(result["status"])

    execution_ms = round((time.time() - start_time) * 1000, 2)
    current_weather = weather[0] if weather else {}

    # 7. Persist full snapshot for retraining (always — uses String user_id)
    try:
        plog = PredictionLog(
            user_id=request.user_id,
            location=request.location,
            latitude=request.latitude,
            longitude=request.longitude,
            mode=request.settings.mode,
            current_battery_pct=request.current_battery_level_pct,
            battery_capacity_kwh=request.settings.battery_capacity_kwh,
            solar_capacity_kw=request.settings.solar_capacity_kw,
            settings_snapshot=request.settings.dict(),
            devices=devices_list,
            weather_snapshot=weather,
            solar_outputs=solar_outputs,
            tariffs=tariffs,
            consumption=consumption,
            battery_forecast=battery_levels,
            total_grid_cost=result["total_grid_cost"],
            recommendations=result["recommendations"],
            hourly_schedule=result["hourly_schedule"],
            execution_time_ms=execution_ms,
        )
        db.add(plog)
        db.commit()
        logger.info(f"Persisted PredictionLog id={plog.id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist prediction snapshot: {e}")

    # 8. Persist user-scoped legacy tables only when user_id looks like a UUID
    if request.user_id and _is_uuid(request.user_id):
        try:
            opt_record = OptimizationResult(
                user_id=request.user_id,
                mode=request.settings.mode,
                total_grid_cost=result["total_grid_cost"],
                recommendations=result["recommendations"],
                hourly_schedule=result["hourly_schedule"],
                estimated_savings=abs(result["total_grid_cost"]),
            )
            db.add(opt_record)

            for device_name, recommendation in result["recommendations"].items():
                if "Not scheduled" not in recommendation:
                    notif = Notification(
                        user_id=request.user_id,
                        message=f"{device_name}: {recommendation}",
                        type="suggestion",
                    )
                    db.add(notif)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to persist user-scoped records: {e}")

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
        "execution_time_ms": execution_ms,
    }


# --- History / notifications -------------------------------------------------
@app.get("/notifications/{user_id}")
def get_notifications(user_id: str, db: Session = Depends(get_db)):
    notifications = (
        db.query(Notification)
        .filter(Notification.user_id == user_id, Notification.is_read == False)
        .order_by(Notification.created_at.desc())
        .all()
    )
    return [
        {
            "id": str(n.id),
            "message": n.message,
            "type": n.type,
            "created_at": n.created_at.isoformat(),
        }
        for n in notifications
    ]


@app.get("/history/{user_id}")
def get_history(user_id: str, limit: int = 30, db: Session = Depends(get_db)):
    results = (
        db.query(OptimizationResult)
        .filter(OptimizationResult.user_id == user_id)
        .order_by(OptimizationResult.created_at.desc())
        .limit(min(limit, 100))
        .all()
    )
    return [
        {
            "id": str(r.id),
            "created_at": r.created_at.isoformat(),
            "mode": r.mode,
            "total_grid_cost": r.total_grid_cost,
            "estimated_savings": r.estimated_savings,
            "recommendations": r.recommendations,
        }
        for r in results
    ]


# --- Admin: retrain (protected) ----------------------------------------------
@app.post("/retrain")
def trigger_retrain(
    payload: RetrainRequest = RetrainRequest(),
    admin: AdminUser = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Manually trigger model retraining using accumulated DB data + API top-up.
    Admin JWT required.
    """
    logger.info(f"Manual retrain triggered by admin {admin.email}")
    result = train_model(
        db=db,
        latitude=payload.latitude,
        longitude=payload.longitude,
        months_back=payload.months_back,
        notes=payload.notes,
    )
    return result


@app.get("/model/versions")
def get_model_versions(db: Session = Depends(get_db)):
    versions = db.query(ModelVersion).order_by(ModelVersion.trained_at.desc()).all()
    return [
        {
            "version": v.version,
            "trained_at": v.trained_at.isoformat(),
            "training_samples": v.training_samples,
            "rmse": v.rmse,
            "r2_score": v.r2_score,
            "is_active": v.is_active,
            "notes": v.notes,
        }
        for v in versions
    ]


@app.get("/prediction-logs")
def list_prediction_logs(
    limit: int = 50,
    admin: AdminUser = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: view recent prediction logs that will be used for retraining."""
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(min(limit, 500))
        .all()
    )
    return [
        {
            "id": str(r.id),
            "created_at": r.created_at.isoformat(),
            "user_id": r.user_id,
            "location": r.location,
            "mode": r.mode,
            "device_count": len(r.devices or []),
            "total_grid_cost": r.total_grid_cost,
            "used_for_training": r.used_for_training,
            "model_version": r.model_version,
        }
        for r in rows
    ]
