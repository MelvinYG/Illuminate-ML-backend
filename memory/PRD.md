# Illuminate ML Service — PRD

## Original Problem Statement
Extend the existing Home Energy Management System ML backend (FastAPI + PostgreSQL):
1. Accept user-supplied devices (lights, bulbs, fans, AC, washing machine, EV charger, etc.) in `/predict` with their power ratings, planned usage hours, priority, and time windows.
2. Persist every prediction request (full weather, solar, tariff, battery, devices, predictions, schedule) to the DB so the model can learn from accumulated real-world data over time.
3. Add JWT-based admin auth to protect the `/retrain` endpoint.
4. Update the retraining flow to consume the newly accumulated data.

## Architecture
- **Stack**: FastAPI + SQLAlchemy 2.0 + PostgreSQL + scikit-learn (RandomForestRegressor) + PuLP optimizer + APScheduler.
- **Entry**: `/app/app.py` (re-exported from `/app/backend/server.py` for supervisor).
- **Auth**: JWT (HS256, Bearer) via `services/auth_service.py`; bcrypt password hashing; idempotent admin seed from `.env` (ADMIN_EMAIL / ADMIN_PASSWORD).
- **DB**: PostgreSQL (`illuminate` user/db, port 5432); tables auto-created via `Base.metadata.create_all()`.

## What's Been Implemented (2026-06-26)
### New endpoints
- `POST /admin/login` → returns JWT (12h expiry)
- `GET /admin/me`
- `POST /retrain` *(admin JWT required)*
- `GET /prediction-logs` *(admin JWT required)*

### `/predict` upgrades
- Now accepts a `devices: List[Device]` field. Each device has: `name`, `power_rating_kw`, `usage_hours`, `priority` (low/normal/high), optional `earliest_hour`, `latest_hour`, `contiguous`, `category`.
- Optimizer (`services/optimizer_service.py`) rewritten to support **any number of arbitrary devices** with priority-weighted objective, time windows, and contiguous-run constraints (previously hardcoded to washing_machine / dishwasher / pump).
- Every request now logs a full snapshot to `prediction_logs` (settings, devices, weather, solar, tariffs, consumption, battery forecast, recommendations, hourly schedule, execution time).

### New DB models (`db_models.py`)
- `AdminUser` — id, email, password_hash, name, role
- `PredictionLog` — full per-request snapshot (24h arrays for weather/solar/tariff/consumption/battery + devices JSON + schedule JSON + cost + used_for_training flag)

### Retraining (`services/model_manager.py`)
- `train_model()` now combines three data sources:
  1. `actual_readings` (legacy)
  2. `prediction_logs` exploded into 24 training rows per request (consumption + scheduled device load → battery forecast)
  3. Open-Meteo API top-up (with synthetic fallback when API unreachable)
- Marks consumed `PredictionLog` rows with `used_for_training=true` + `model_version`.
- Promote-only-if-better policy preserved (≥1% R² improvement).

### Other fixes
- Weather service has synthetic fallback when `WEATHER_API_KEY` is unconfigured.
- Data collection service falls back to synthetic weather if Open-Meteo is unreachable.
- Persistence split: `PredictionLog` always saved; legacy `OptimizationResult`/`Notification` only when `user_id` is a UUID belonging to `users` table.

## End-to-end Verification
- Admin login → JWT ✅
- `/retrain` 401 without token, 401 with bad token, 200 with valid token ✅
- `/predict` returns optimal schedule for 5+ dynamic device categories (AC, washing machine, ceiling fan, EV charger, LED lights) ✅
- 2 predict calls → 2 rows in `prediction_logs` (weather/solar/tariff/battery arrays of length 24, devices stored) ✅
- After `/retrain`: `prediction_logs.used_for_training = true`, `model_version = 2`, new row in `model_versions` table ✅

## Frontend / Node-backend integration notes (separate repos)
The frontend (`Illuminate`) and Node backend (`Illuminate-Backend`) live in separate GitHub repos. To wire them up:
- Add a **device dialog** in the frontend (`src/components/.../AddDeviceDialog.jsx`) with fields: name, power rating (kW), usage hours, priority, optional time-window, category.
- Node backend forwards the devices payload in the `/predict` request to this ML service.
- A future enhancement: cache user device library in the Node Mongo DB so users don't re-enter the same devices.

## Credentials
See `/app/memory/test_credentials.md`.

## Backlog / Future
- P1: Pull `tariff_history` per location (currently hardcoded curve).
- P1: Bulk `/admin/seed-actual-readings` endpoint to backfill the model with real sensor data.
- P2: Refresh-token flow for admin sessions.
- P2: Dashboards over `prediction_logs` (cost trends, device usage histograms).

## Potential Improvement
Add a **"household savings leaderboard"** comparing this user's optimized cost vs neighbours in the same city/tariff zone — turns the dashboard into something shareable and creates organic word-of-mouth growth for a B2C launch.
