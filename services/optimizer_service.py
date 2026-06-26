"""
Mixed-integer optimizer for any number of user-supplied devices.
Each device has: name, power rating (kW), planned daily run hours,
priority (low/normal/high) and optional preferred time window.
"""
import pulp
from loguru import logger
from typing import List, Dict, Optional

MODE_CONFIG = {
    "self_consumption": {
        "grid_buy_penalty": 2.0,
        "battery_drain_allowed": True,
        "appliances_active": True,
        "description": "Prioritise solar and battery over grid",
    },
    "tou_savings": {
        "grid_buy_penalty": 1.0,
        "battery_drain_allowed": True,
        "appliances_active": True,
        "description": "Optimise based on time-of-use tariff rates",
    },
    "full_backup": {
        "grid_buy_penalty": 0.5,
        "battery_drain_allowed": False,
        "appliances_active": True,
        "description": "Keep battery fully charged at all times",
    },
    "low_power": {
        "grid_buy_penalty": 1.0,
        "battery_drain_allowed": True,
        "appliances_active": False,
        "description": "Minimise heavy appliance usage",
    },
}

PRIORITY_WEIGHT = {"high": 0.0, "normal": 0.1, "low": 0.5}


def _fmt_hour(h: Optional[int]) -> str:
    if h is None:
        return "Not scheduled"
    suffix = "AM" if h < 12 else "PM"
    twelve = 12 if h % 12 == 0 else h % 12
    return f"{twelve} {suffix}"


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def run_optimization(
    tariffs: List[float],
    solar_outputs: List[float],
    consumption: List[float],
    initial_battery: float,
    battery_capacity: float = 15.0,
    devices: Optional[List[Dict]] = None,
    mode: str = "tou_savings",
) -> Dict:
    """
    devices: list of {
        name, power_rating_kw, usage_hours (int, total hrs/day),
        priority ("high"|"normal"|"low"),
        earliest_hour (0-23, optional),
        latest_hour  (0-23, optional, exclusive end),
        contiguous (bool, default False) - must run in consecutive hours
    }
    """
    config = MODE_CONFIG.get(mode, MODE_CONFIG["tou_savings"])
    devices = devices or []
    logger.info(f"Running optimizer in mode: {mode} — {config['description']} | {len(devices)} device(s)")

    tariff_sell = [t - 1 for t in tariffs]
    model = pulp.LpProblem("HEMS_Optimization", pulp.LpMinimize)

    # --- Device decision variables ---------------------------------------
    use_devices = config["appliances_active"] and len(devices) > 0
    device_vars: Dict[str, List[pulp.LpVariable]] = {}
    device_meta: Dict[str, Dict] = {}

    if use_devices:
        for idx, d in enumerate(devices):
            slug = f"{_slugify(d.get('name', f'dev{idx}'))}_{idx}"
            x = [pulp.LpVariable(f"x_{slug}_{t}", cat="Binary") for t in range(24)]
            device_vars[slug] = x
            device_meta[slug] = d

    # --- Energy variables -------------------------------------------------
    battery_charge = [pulp.LpVariable(f"bc_{t}", lowBound=0, upBound=3) for t in range(24)]
    battery_discharge = [
        pulp.LpVariable(
            f"bd_{t}", lowBound=0,
            upBound=3 if config["battery_drain_allowed"] else 0,
        ) for t in range(24)
    ]
    battery_level = [pulp.LpVariable(f"bl_{t}", lowBound=0, upBound=battery_capacity) for t in range(24)]
    grid_buy = [pulp.LpVariable(f"gb_{t}", lowBound=0) for t in range(24)]
    grid_sell = [pulp.LpVariable(f"gs_{t}", lowBound=0) for t in range(24)]

    # --- Objective: minimise grid cost + priority weighting on device hours
    grid_cost_terms = [tariffs[t] * grid_buy[t] - tariff_sell[t] * grid_sell[t] for t in range(24)]
    priority_terms = []
    for slug, x in device_vars.items():
        d = device_meta[slug]
        w = PRIORITY_WEIGHT.get(str(d.get("priority", "normal")).lower(), 0.1)
        # high priority devices get small/zero penalty so they run as planned;
        # low priority devices get penalty so optimiser may shift their hours.
        priority_terms.extend(w * x[t] for t in range(24))
    model += pulp.lpSum(grid_cost_terms) + pulp.lpSum(priority_terms)

    # --- Power balance per hour ------------------------------------------
    for t in range(24):
        device_load_t = pulp.lpSum(
            float(device_meta[slug].get("power_rating_kw", 0.0)) * device_vars[slug][t]
            for slug in device_vars
        )
        model += (
            device_load_t + consumption[t]
            <= grid_buy[t] - grid_sell[t] + battery_discharge[t]
            + solar_outputs[t] - battery_charge[t]
        )

        if t == 0:
            model += battery_level[t] == initial_battery + battery_charge[t] - battery_discharge[t]
        else:
            model += battery_level[t] == battery_level[t - 1] + battery_charge[t] - battery_discharge[t]

    # --- Device constraints ----------------------------------------------
    for slug, x in device_vars.items():
        d = device_meta[slug]
        hours = max(0, int(d.get("usage_hours", 1)))
        model += pulp.lpSum(x) == hours

        earliest = int(d.get("earliest_hour", 0) or 0)
        latest = int(d.get("latest_hour", 24) or 24)
        for t in range(24):
            if t < earliest or t >= latest:
                model += x[t] == 0

        if d.get("contiguous"):
            # Force consecutive run: x[t] - x[t+1] <= 0 then >=0 once block ends
            # Simpler: require sum of |x[t+1]-x[t]| transitions <= 2 via binary z
            # We'll approximate with: enforce monotonic block via z[t]
            z = [pulp.LpVariable(f"z_{slug}_{t}", cat="Binary") for t in range(24)]
            for t in range(23):
                model += x[t + 1] - x[t] <= z[t + 1]
            model += pulp.lpSum(z) <= 1

    # --- Solve ------------------------------------------------------------
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # --- Build schedule ---------------------------------------------------
    schedule = []
    for t in range(24):
        row = {
            "hour": t,
            "grid_buy_kw": round(max(0, pulp.value(grid_buy[t]) or 0), 3),
            "grid_sell_kw": round(max(0, pulp.value(grid_sell[t]) or 0), 3),
            "battery_level_kwh": round(pulp.value(battery_level[t]) or 0, 3),
            "solar_output_kw": round(solar_outputs[t], 3),
            "devices": {},
        }
        for slug, x in device_vars.items():
            row["devices"][device_meta[slug].get("name", slug)] = bool(round(pulp.value(x[t]) or 0))
        schedule.append(row)

    # --- Human-readable recommendations ----------------------------------
    recommendations: Dict[str, str] = {}
    for slug, x in device_vars.items():
        name = device_meta[slug].get("name", slug)
        hours_on = [t for t in range(24) if round(pulp.value(x[t]) or 0) == 1]
        if not hours_on:
            recommendations[name] = "Not scheduled"
            continue
        # contiguous block?
        contiguous = all(hours_on[i] + 1 == hours_on[i + 1] for i in range(len(hours_on) - 1))
        if contiguous and len(hours_on) > 1:
            recommendations[name] = f"Run from {_fmt_hour(hours_on[0])} to {_fmt_hour(hours_on[-1] + 1)}"
        elif len(hours_on) == 1:
            recommendations[name] = f"Run at {_fmt_hour(hours_on[0])}"
        else:
            slots = ", ".join(_fmt_hour(h) for h in hours_on)
            recommendations[name] = f"Run at: {slots}"

    return {
        "status": pulp.LpStatus[model.status],
        "total_grid_cost": round(pulp.value(model.objective) or 0, 3),
        "recommendations": recommendations,
        "hourly_schedule": schedule,
    }
