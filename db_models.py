from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from database import Base

# Think of these classes as table definitions
# Each attribute = one column

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships — like JOINs but object-oriented
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
    data = Column(JSON, nullable=False)  # Stores all 24 hourly readings


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
    type = Column(String(50))  # suggestion | alert | info
    is_read = Column(Boolean, default=False)

    user = relationship("User", back_populates="notifications")

class ActualReading(Base):
    """
    Stores real hourly readings as they accumulate.
    This is what the model learns from over time.
    Think of this as your ever-growing training dataset.
    """
    __tablename__ = "actual_readings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recorded_at = Column(DateTime, nullable=False)
    location = Column(String(255), nullable=False)
    hour = Column(Integer, nullable=False)           # 0-23

    # Features (inputs to model)
    power_usage_kw = Column(Float)                  # Actual consumption
    solar_output_kw = Column(Float)                 # Actual solar generation
    tariff_value = Column(Float)                    # Actual tariff at this hour

    # Target (what model predicts)
    battery_level_kwh = Column(Float)               # Actual battery level

    # Metadata
    source = Column(String(50), default="api")      # api | user_input | sensor
    location = Column(String(255))


class ModelVersion(Base):
    """
    Tracks every model version ever trained.
    Lets you rollback if a new model is worse.
    """
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(Integer, nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)
    training_samples = Column(Integer)              # How many rows used
    rmse = Column(Float)                            # Root Mean Square Error — lower is better
    r2_score = Column(Float)                        # 0-1, higher is better
    is_active = Column(Boolean, default=False)      # Only one active at a time
    pkl_path = Column(String(255))                  # Path to saved pkl file
    notes = Column(String(500))                     # e.g. "Weekly retrain #3"