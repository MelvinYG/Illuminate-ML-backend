

def calculate_solar_output(hourly_weather: list, panel_capacity_kw: float = 100.0) -> list:
    """
    Calculates hourly AC power output from solar panel.
    Based on your friend's logic from solar_madeupdata.ipynb
    """
    outputs = []

    # Find max radiation for normalization
    max_radiation = max(
        (h["solar_radiation"] for h in hourly_weather),
        default=1  # Avoid division by zero
    )

    for hour_data in hourly_weather:
        hour = hour_data["hour"]

        # No solar output at night (7PM to 5AM)
        # Same logic as your friend's notebook
        if hour >= 19 or hour < 5:
            outputs.append(0.0)
            continue

        # Calculate output based on weather factors
        cloud_factor = (100 - hour_data["cloud_cover"]) / 100
        radiation_factor = hour_data["solar_radiation"] / max(max_radiation, 1)
        uv_factor = hour_data["uv_index"] / 11  # UV index max is 11

        ac_output = panel_capacity_kw * cloud_factor * radiation_factor * uv_factor
        outputs.append(round(ac_output, 3))

    return outputs