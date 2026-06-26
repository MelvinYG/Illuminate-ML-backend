import pulp
from loguru import logger

# Mode configs — how each mode changes optimizer behaviour
MODE_CONFIG = {
    "self_consumption": {
        "grid_buy_penalty": 2.0,      # Penalise buying from grid heavily
        "battery_drain_allowed": True,
        "appliances_active": True,
        "description": "Prioritise solar and battery over grid"
    },
    "tou_savings": {
        "grid_buy_penalty": 1.0,      # Default — just follow tariffs
        "battery_drain_allowed": True,
        "appliances_active": True,
        "description": "Optimise based on time-of-use tariff rates"
    },
    "full_backup": {
        "grid_buy_penalty": 0.5,      # Allow grid buying to keep battery full
        "battery_drain_allowed": False, # Never drain battery
        "appliances_active": True,
        "description": "Keep battery fully charged at all times"
    },
    "low_power": {
        "grid_buy_penalty": 1.0,
        "battery_drain_allowed": True,
        "appliances_active": False,   # Don't run heavy appliances
        "description": "Minimise heavy appliance usage"
    }
}


def run_optimization(
    tariffs: list,          # 24 hourly buying tariffs
    solar_outputs: list,    # 24 hourly solar output
    consumption: list,      # 24 hourly consumption
    initial_battery: float, # Starting battery level in kWh
    battery_capacity: float = 15.0,
    appliances: list = ["washing_machine", "dishwasher", "pump"],
    mode: str = "tou_savings"   
) -> dict:
    """
    Runs PuLP linear programming optimizer.
    Returns 24-hour optimal schedule.
    """
    config = MODE_CONFIG.get(mode, MODE_CONFIG["tou_savings"])
    logger.info(f"Running optimizer in mode: {mode} — {config['description']}")

    tariff_sell = [t - 1 for t in tariffs] # Sell at 1 less than buy price
    penalty = config["grid_buy_penalty"]

    model = pulp.LpProblem("HEMS_Optimization", pulp.LpMinimize)

    # Decision variables — binary for appliances
    active_appliances = appliances if config["appliances_active"] else []
    has_wm = "washing_machine" in active_appliances
    has_dw = "dishwasher" in active_appliances
    has_pump = "pump" in active_appliances

    x_wm   = [pulp.LpVariable(f"x_wm_{t}", cat="Binary") for t in range(24)]
    x_dw   = [pulp.LpVariable(f"x_dw_{t}", cat="Binary") for t in range(24)]
    x_pump = [pulp.LpVariable(f"x_pump_{t}", cat="Binary") for t in range(24)]

    battery_charge    = [pulp.LpVariable(f"bc_{t}", lowBound=0, upBound=3) for t in range(24)]
    battery_discharge = [pulp.LpVariable(f"bd_{t}", lowBound=0,
                          upBound=3 if config["battery_drain_allowed"] else 0)  # ← full_backup mode
                         for t in range(24)]
    battery_level     = [pulp.LpVariable(f"bl_{t}", lowBound=0, upBound=battery_capacity) for t in range(24)]
    grid_buy          = [pulp.LpVariable(f"gb_{t}", lowBound=0) for t in range(24)]
    grid_sell         = [pulp.LpVariable(f"gs_{t}", lowBound=0) for t in range(24)]

    # Objective — minimize grid cost
    model += pulp.lpSum(
        tariffs[t] * grid_buy[t] - tariff_sell[t] * grid_sell[t]
        for t in range(24)
    )

    # Constraints
    for t in range(24):
        # Power balance
        model += (
            1.5 * x_wm[t] + 1.2 * x_dw[t] + 1.0 * x_pump[t] + consumption[t]
            <= grid_buy[t] - grid_sell[t] + battery_discharge[t] + solar_outputs[t] - battery_charge[t]
        )

        # Battery state
        if t == 0:
            model += battery_level[t] == initial_battery + battery_charge[t] - battery_discharge[t]
        else:
            model += battery_level[t] == battery_level[t-1] + battery_charge[t] - battery_discharge[t]

    # Appliance constraints
    if has_wm:
        model += pulp.lpSum(x_wm) == 1
    else:
        model += pulp.lpSum(x_wm) == 0

    if has_dw:
        model += pulp.lpSum(x_dw) == 2
        for t in range(23):
            model += x_dw[t] - x_dw[t+1] <= 0
    else:
        model += pulp.lpSum(x_dw) == 0

    if has_pump:
        model += pulp.lpSum(x_pump) == 1
    else:
        model += pulp.lpSum(x_pump) == 0

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))   # msg=False = silent mode

    # Build result
    schedule = []
    for t in range(24):
        schedule.append({
            "hour": t,
            "grid_buy_kw": round(max(0, pulp.value(grid_buy[t])), 3),
            "grid_sell_kw": round(max(0, pulp.value(grid_sell[t])), 3),
            "battery_level_kwh": round(pulp.value(battery_level[t]), 3),
            "solar_output_kw": round(solar_outputs[t], 3),
            "run_washing_machine": bool(round(pulp.value(x_wm[t]))),
            "run_dishwasher": bool(round(pulp.value(x_dw[t]))),
            "run_pump": bool(round(pulp.value(x_pump[t])))
        })

    def fmt_hour(h):
        if h is None: return "Not scheduled"
        return f"{'12' if h == 12 else h % 12 or 12} {'AM' if h < 12 else 'PM'}"
    
    # Human readable recommendations
    wm_hour = next((s["hour"] for s in schedule if s["run_washing_machine"]), None)
    dw_hours = [s["hour"] for s in schedule if s["run_dishwasher"]]
    pump_hour = next((s["hour"] for s in schedule if s["run_pump"]), None)


    return {
        "status": pulp.LpStatus[model.status],
        "total_grid_cost": round(pulp.value(model.objective), 3),
        "recommendations": {
            "washing_machine": f"Run at {fmt_hour(wm_hour)}",
            "dishwasher": f"Run at {fmt_hour(dw_hours[0])} - {fmt_hour(dw_hours[-1]+1)}" if dw_hours else "Not scheduled",
            "pump": f"Run at {fmt_hour(pump_hour)}"
        },
        "hourly_schedule": schedule
    }