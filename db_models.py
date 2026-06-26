from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    settings = relationship("UserSettings", back_populates="user", uselist=False)
    optimizations = relationship("OptimizationResult", back_populates="user")
    notifications = relationship("Notification", back_populates="user")


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    location = Column(String(255), default="Delhi,IN")
    battery_capacity_kwh = Column(Float, default=15.0)
    solar_capacity_kw = Column(Float, default=100.0)
    current_battery_pct = Column(Float, default=50.0)
    mode = Column(String(50), default="tou_savings")
    appliances = Column(JSON, default=["washing_machine", "dishwasher", "pump"])
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="settings")


class WeatherCache(Base):
    __tablename__ = "weather_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    location = Column(String(255), nullable=False)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    data = Column(JSON, nullable=False)


class TariffHistory(Base):
    __tablename__ = "tariff_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recorded_at = Column(DateTime, nullable=False)
    hour = Column(Integer, nullable=False)
    tariff_value = Column(Float, nullable=False)
    location = Column(String(255))


class OptimizationResult(Base):
    __tablename__ = "optimization_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    mode = Column(String(50))
    total_grid_cost = Column(Float)
    recommendations = Column(JSON)
    hourly_schedule = Column(JSON)
    estimated_savings = Column(Float)

    user = relationship("User", back_populates="optimizations")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    message = Column(String, nullable=False)
    type = Column(String(50))
    is_read = Column(Boolean, default=False)

    user = relationship("User", back_populates="notifications")


class ActualReading(Base):
    """
    Stores real hourly readings as they accumulate.
    The ever-growing training dataset for retraining.
    """
    __tablename__ = "actual_readings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recorded_at = Column(DateTime, nullable=False)
    location = Column(String(255), nullable=False)
    hour = Column(Integer, nullable=False)

    power_usage_kw = Column(Float)
    solar_output_kw = Column(Float)
    tariff_value = Column(Float)
    battery_level_kwh = Column(Float)

    source = Column(String(50), default="api")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(Integer, nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)
    training_samples = Column(Integer)
    rmse = Column(Float)
    r2_score = Column(Float)
    is_active = Column(Boolean, default=False)
    pkl_path = Column(String(255))
    notes = Column(String(500))


# ─── NEW MODELS ───────────────────────────────────────────────────────────────

class AdminUser(Base):
    """Admin user for protected endpoints like /retrain."""
    __tablename__ = "admin_users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), default="Admin")
    role = Column(String(50), default="admin")
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """
    Complete snapshot of every /predict request and its result.
    This is the master log used by retraining to build a richer dataset.

    Captures: user inputs (settings, devices), all derived data (weather, solar,
    tariffs, consumption), the model prediction (battery forecast) and the
    optimizer output (schedule, recommendations, grid cost).
    """
    __tablename__ = "prediction_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user_id = Column(String(255), nullable=True)
    location = Column(String(255), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)

    # Inputs
    mode = Column(String(50))
    current_battery_pct = Column(Float)
    battery_capacity_kwh = Column(Float)
    solar_capacity_kw = Column(Float)
    settings_snapshot = Column(JSON)
    devices = Column(JSON)               # User-supplied devices w/ ratings & usage

    # Derived 24h arrays
    weather_snapshot = Column(JSON)      # Full 24h weather data
    solar_outputs = Column(JSON)         # 24h solar generation (kW)
    tariffs = Column(JSON)               # 24h tariff (INR/kWh)
    consumption = Column(JSON)           # 24h consumption (kW)
    battery_forecast = Column(JSON)      # 24h predicted battery level (kWh)

    # Output
    total_grid_cost = Column(Float)
    recommendations = Column(JSON)
    hourly_schedule = Column(JSON)
    execution_time_ms = Column(Float)

    # Bookkeeping
    used_for_training = Column(Boolean, default=False)
    model_version = Column(Integer, nullable=True)
