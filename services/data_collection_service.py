"""
Fetches training data entirely from APIs.
No CSV files needed.

Open-Meteo Archive API:
- Completely free, no API key
- Historical weather back to 1940
- Has solar radiation data
- No rate limits
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from exceptions import IlluminateError


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Realistic Indian electricity tariff pattern (INR/kWh)
# Based on KSEB/BESCOM published rates
TARIFF_PATTERN = {
    range(0, 6):   3.5,    # 12 AM - 6 AM: off-peak, cheapest
    range(6, 9):   5.0,    # 6 AM - 9 AM: morning ramp
    range(9, 14):  5.5,    # 9 AM - 2 PM: normal
    range(14, 17): 4.0,    # 2 PM - 5 PM: afternoon dip
    range(17, 21): 7.5,    # 5 PM - 9 PM: evening peak, most expensive
    range(21, 24): 5.0,    # 9 PM - 12 AM: evening taper
}

def get_tariff_for_hour(hour: int) -> float:
    """Returns realistic Indian electricity tariff for a given hour."""
    for hour_range, rate in TARIFF_PATTERN.items():
        if hour in hour_range:
            return rate
    return 5.5  # fallback


def get_consumption_for_hour(hour: int) -> float:
    """
    Returns realistic household power consumption for a given hour.
    Based on the pattern from userconsumption_battery.ipynb
    """
    if 18 <= hour <= 21:   return 5.0   # Peak evening (AC, TV, cooking)
    elif 22 <= hour <= 23: return 3.5   # Late night wind-down
    elif 0 <= hour <= 4:   return 2.0   # Night, minimal usage
    elif 5 <= hour <= 8:   return 3.0   # Morning ramp-up
    elif 9 <= hour <= 12:  return 2.5   # Day, moderate
    elif 13 <= hour <= 15: return 3.0   # Afternoon
    elif 16 <= hour <= 17: return 2.5   # Pre-evening
    return 2.0


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,    # "YYYY-MM-DD"
    end_date: str
) -> pd.DataFrame:
    """
    Fetches historical weather from Open-Meteo Archive API.
    Completely free, no API key needed.

    Returns DataFrame with hourly: solar_radiation, temperature, cloud_cover
    """
    logger.info(f"Fetching historical weather from Open-Meteo: {start_date} to {end_date}")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "cloudcover",
            "direct_radiation",          # W/m² — direct solar radiation
            "diffuse_radiation",         # W/m² — scattered solar radiation
            "uv_index",
        ],
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params)

    if response.status_code != 200:
        raise IlluminateError(
            f"Open-Meteo API failed: {response.status_code} — {response.text}",
            status_code=503
        )

    data = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "cloud_cover": hourly["cloudcover"],
        "direct_radiation": hourly["direct_radiation"],
        "diffuse_radiation": hourly["diffuse_radiation"],
        "uv_index": hourly["uv_index"],
    })

    # Total solar radiation = direct + diffuse
    df["solar_radiation"] = df["direct_radiation"] + df["diffuse_radiation"]
    df["hour"] = df["datetime"].dt.hour

    logger.info(f"Fetched {len(df)} hourly weather records")
    return df


def calculate_solar_output_from_weather(
    df: pd.DataFrame,
    panel_capacity_kw: float = 100.0
) -> pd.Series:
    """
    Calculates AC power output from solar panel based on weather.
    Same logic as solar_madeupdata.ipynb, but as a clean function.
    """
    max_radiation = df["solar_radiation"].max()
    if max_radiation == 0:
        return pd.Series(0.0, index=df.index)

    cloud_factor = (100 - df["cloud_cover"]) / 100
    radiation_factor = df["solar_radiation"] / max_radiation
    uv_factor = (df["uv_index"] / 11).clip(0, 1)   # UV max is 11

    output = panel_capacity_kw * cloud_factor * radiation_factor * uv_factor

    # No solar at night (7 PM to 5 AM)
    night_mask = (df["hour"] >= 19) | (df["hour"] < 5)
    output[night_mask] = 0.0

    return output.round(3)


def simulate_battery_levels(
    solar_output: pd.Series,
    consumption: pd.Series,
    tariff: pd.Series,
    battery_capacity: float = 15.0,
    initial_pct: float = 0.5
) -> pd.Series:
    """
    Simulates battery level hour by hour.
    Ported from userconsumption_battery.ipynb logic.

    This gives us the 'battery_level_kwh' column we need to train the model.
    """
    battery_level = initial_pct * battery_capacity
    battery_levels = []
    hours = solar_output.index

    for i, idx in enumerate(hours):
        hour = i % 24   # 0-23 within the day
        solar = solar_output.iloc[i]
        tariff_val = tariff.iloc[i]
        usage = consumption.iloc[i]

        # Battery management logic — same as notebook
        if hour >= 23 or hour < 5:
            # Nighttime — no charging/discharging
            pass
        elif solar > 0:
            # Daytime solar — charge battery
            battery_level = min(battery_level + solar, battery_capacity)
        elif tariff_val <= 4.0 and battery_level < battery_capacity:
            # Low tariff — charge from grid
            battery_level = min(battery_level + 1, battery_capacity)
        else:
            # Discharge to meet consumption
            discharge = min(usage, battery_level)
            battery_level = max(0, battery_level - discharge)

        battery_levels.append(round(battery_level, 3))

    return pd.Series(battery_levels, index=hours)


def build_training_dataframe(
    latitude: float = 28.6,     # Default: Delhi
    longitude: float = 77.2,
    months_back: int = 3,       # How many months of historical data
    panel_capacity_kw: float = 100.0,
    battery_capacity_kwh: float = 15.0
) -> pd.DataFrame:
    """
    Builds the complete training DataFrame from APIs only.
    No CSV files needed.

    Returns DataFrame ready for model training with columns:
    - power_usage_kw (feature)
    - electricity_tariff_INR_per_kWh (feature)
    - ac_power_output_kw (feature)
    - battery_level_kwh (target)
    """
    # Date range — last N months
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30 * months_back)

    logger.info(f"Building training data from {start_date} to {end_date}")

    # 1. Fetch weather
    weather_df = fetch_historical_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=str(start_date),
        end_date=str(end_date)
    )

    # 2. Solar output from weather
    weather_df["ac_power_output_kW"] = calculate_solar_output_from_weather(
        weather_df, panel_capacity_kw
    )

    # 3. Tariff pattern — per hour
    weather_df["electricity_tariff_INR_per_kWh"] = weather_df["hour"].apply(
        get_tariff_for_hour
    )

    # 4. Consumption pattern — per hour
    weather_df["power_usage_kW"] = weather_df["hour"].apply(
        get_consumption_for_hour
    )

    # 5. Simulate battery levels
    weather_df["battery_level_kWh"] = simulate_battery_levels(
        solar_output=weather_df["ac_power_output_kW"],
        consumption=weather_df["power_usage_kW"],
        tariff=weather_df["electricity_tariff_INR_per_kWh"],
        battery_capacity=battery_capacity_kwh
    )

    # Drop rows with NaN (can happen at edges of API response)
    weather_df = weather_df.dropna(subset=[
        "power_usage_kW",
        "electricity_tariff_INR_per_kWh",
        "ac_power_output_kW",
        "battery_level_kWh"
    ])

    logger.info(f"Training DataFrame built: {len(weather_df)} rows")

    return weather_df[[
        "datetime",
        "power_usage_kW",
        "electricity_tariff_INR_per_kWh",
        "ac_power_output_kW",
        "battery_level_kWh"
    ]]