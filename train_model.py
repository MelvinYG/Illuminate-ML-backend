"""
Run this ONCE to train and save the Random Forest model.
After this, app.py loads the saved model — no retraining needed per request.

To run,use: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def train_battery_model():
    print("Loading training data...")

    # Load CSVs — put these in a /data folder
    consumption_data = pd.read_csv("data/consumption_battery_data.csv")
    solar_tariff_data = pd.read_csv("data/solar_tariff_generated_data.csv")

    consumption_data["datetime"] = pd.to_datetime(
        consumption_data["datetime"], errors="coerce"
    )
    solar_tariff_data["datetime"] = pd.to_datetime(
        solar_tariff_data["datetime"], errors="coerce"
    )

    # Features
    features = pd.concat([
        consumption_data[["power_usage_kW"]],
        solar_tariff_data[["electricity_tariff_INR_per_kWh", "ac_power_output_kW"]]
    ], axis=1)

    target = consumption_data["battery_level_kWh"]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)
    y = target.values

    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Save model AND scaler — you need both for predictions
    os.makedirs("models", exist_ok=True)
    with open("models/battery_model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    print("Model saved to models/battery_model.pkl")

if __name__ == "__main__":
    train_battery_model()