import requests
import os
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger
from dotenv import load_dotenv
from db_models import WeatherCache

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")
BASE_URL = os.getenv("WEATHER_API_BASE_URL")
CACHE_TTL_MINUTES = 30


def _mock_24h_weather() -> list:
    """Synthetic 24h weather fallback when the upstream API is unavailable."""
    hourly = []
    for h in range(24):
        # Solar peaks midday; UV index follows similar curve
        if 6 <= h <= 18:
            sun = max(0.0, 1.0 - abs(12 - h) / 6.0)
        else:
            sun = 0.0
        hourly.append({
            "hour": h,
            "temp": 22 + sun * 12,
            "humidity": 60 - sun * 25,
            "cloud_cover": 30,
            "solar_radiation": sun * 900,
            "uv_index": sun * 10,
        })
    return hourly

def fetch_weather(location: str, db: Session) -> list:
    """
    Fetches next 24 hours of weather data for a given location.
    Returns a list of 24 hourly dicts.
    """

    # Check DB cache first
    cached = db.query(WeatherCache)\
               .filter(WeatherCache.location == location)\
               .order_by(WeatherCache.fetched_at.desc())\
               .first()
    if cached:
        age_minutes = (datetime.utcnow() - cached.fetched_at).total_seconds() / 60
        if age_minutes < CACHE_TTL_MINUTES:
            logger.info(f"Weather cache HIT for {location} (age: {age_minutes:.1f} mins)")
            return cached.data  # Return cached JSON directly
    
    # Cache miss — hit the API
    logger.info(f"Weather cache MISS for {location} — fetching from API")

    if not API_KEY or API_KEY == "demo_key" or not BASE_URL:
        logger.warning("Weather API not configured — using synthetic mock data")
        hourly = _mock_24h_weather()
        try:
            db.add(WeatherCache(location=location, data=hourly))
            db.commit()
        except Exception:
            db.rollback()
        return hourly

    url = f"{BASE_URL}/{location}/next24hours" 

    params = {
        "unitGroup": "metric",
        "include": "hours",
        "key": API_KEY,
        "contentType": "json"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
    except Exception as e:
        logger.error(f"Weather API request failed: {e}")
        if cached:
            return cached.data
        return _mock_24h_weather()

    if response.status_code != 200:
        logger.error(f"Weather API failed with status {response.status_code}")

        # Fallback to stale cache if API fails — better than crashing
        if cached:
            logger.warning("Using stale weather cache as fallback")
            return cached.data

        logger.warning("Using mock weather data as final fallback")
        return _mock_24h_weather()

    data = response.json()

    # Parse hourly data — list comprehension
    hourly = []
    for day in data["days"]:
        for hour in day["hours"]:
            hourly.append({
                "hour": int(hour["datetime"].split(":")[0]),  # "14:00:00" → 14
                "temp": hour["temp"],
                "humidity": hour["humidity"],
                "cloud_cover": hour["cloudcover"],
                "solar_radiation": hour.get("solarradiation", 0),
                "uv_index": hour.get("uvindex", 0),
            })

    hourly = hourly[:24] # Ensure exactly 24 hours — like .slice(0, 24)

    # Save to DB cache
    new_cache = WeatherCache(location=location, data=hourly)
    db.add(new_cache)
    db.commit()
    logger.info(f"Weather data cached for {location}")

    return hourly


def weather_to_dataframe(hourly_weather: list) -> pd.DataFrame:
    """
    Converts weather list → Pandas DataFrame.
    """
    df = pd.DataFrame(hourly_weather)
    logger.debug(f"Weather DataFrame shape: {df.shape}")
    return df