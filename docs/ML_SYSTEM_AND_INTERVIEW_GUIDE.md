# Illuminate ML, Energy Calculations, and SDE-2 Interview Guide

> A source-backed explanation of the machine learning, deterministic energy
> calculations, mixed-integer optimization, assumptions, metrics, limitations,
> redesign options, and likely interview questions for this repository.

## 1. How to use this guide

Read sections 2–9 to understand what the system actually calculates. Read sections
10–13 before presenting the project in an interview. The answers are intentionally
candid: explain the current implementation, identify its limits, and then describe
the production-grade design you would build next.

This guide was audited on **2026-08-30** at commit
`d8f55ca8d7394d9b87e98dc189d2783485e337e2`. Metrics below are local historical
evidence, not claims about a currently deployed production model.

## 2. The 90-second mental model

Illuminate plans one day of household energy use in 24 one-hour slots.

1. It obtains 24 weather observations, using a PostgreSQL cache and synthetic
   fallback.
2. It estimates photovoltaic output with a hand-built formula.
3. It constructs deterministic tariff and baseline-load curves.
4. A 100-tree Random Forest predicts battery energy for each hour independently.
5. Separately, a PuLP/CBC mixed-integer program chooses appliance hours, battery
   charge/discharge, and grid import/export to minimize its objective.
6. It returns both the ML forecast and optimizer schedule and logs a complete
   snapshot for possible retraining.

The key distinction is:

- **Machine learning** estimates battery level, currently for advisory/output use.
- **Operations research** produces the schedule and cost.
- **Deterministic formulas** generate most inputs and the synthetic training target.

The Random Forest forecast does not feed the optimizer. This is a hybrid system,
but not yet an ML-driven optimizer.

## 3. Notation and units

| Symbol | Meaning | Unit in code |
|---|---|---|
| `t` | Hour index, 0 through 23 | one-hour slot |
| `L_t` | Baseline household load | kW |
| `P_d` | Rated power of device `d` | kW |
| `x_d,t` | Device on/off decision | binary |
| `S_t` | Estimated solar output | kW |
| `T_t` | Grid import tariff | INR/kWh |
| `T_sell,t` | Export tariff, implemented as `T_t - 1` | INR/kWh |
| `G+_t` | Grid import | kW |
| `G-_t` | Grid export | kW |
| `C_t` | Battery charge power | kW |
| `D_t` | Battery discharge power | kW |
| `E_t` | Stored battery energy/state of charge | kWh |
| `E_max` | Battery capacity | kWh |
| `H_d` | Required daily run slots for device `d` | hours/slots |

The implementation omits an explicit `delta_t`. Because every slot is one hour,
the same numeric value is used when adding kW for one hour to kWh. A more general
formula must multiply power by `delta_t`.

SOC often means a fraction or percentage:

```text
SOC_pct = 100 * E_t / E_max
E_t     = SOC_pct / 100 * E_max
```

The API converts the current percentage to energy before optimization:

```text
initial_battery_kwh = current_battery_level_pct / 100 * battery_capacity_kwh
```

## 4. Exact live `/predict` calculations

### 4.1 Weather and fallback

The live weather service queries the newest cache row for the location string. A
row younger than 30 minutes is returned. Otherwise it calls the configured API.
Missing configuration produces deterministic mock weather; request failure uses a
stale cache when present, otherwise mock weather. That fallback does not cover a
malformed successful JSON/schema response or a failed cache commit; those can still
surface as HTTP 500.

The mock sun factor is triangular:

```text
sun_h = max(0, 1 - |12 - h| / 6), for 06:00 through 18:00
sun_h = 0, otherwise
```

It then sets:

```text
temperature   = 22 + 12 * sun_h
humidity      = 60 - 25 * sun_h
cloud_cover   = 30
solar_radiation = 900 * sun_h
uv_index      = 10 * sun_h
```

Source: `services/weather_service.py:17-34,36-117`.

### 4.2 Live solar estimate

For the 24 returned weather items, let `G_max` be the largest
`solar_radiation`. For hours 05:00–18:59, the implementation calculates:

```text
cloud_factor     = (100 - cloud_cover_t) / 100
radiation_factor = solar_radiation_t / max(G_max, 1)
uv_factor        = uv_index_t / 11

S_t = panel_capacity_kw * cloud_factor * radiation_factor * uv_factor
```

For hours 19:00–04:59, `S_t = 0`.

With the built-in noon mock and a 100 kW panel:

```text
S_noon = 100 * 0.70 * 1.00 * (10 / 11) = 63.636 kW
```

Assumptions and issues:

- `panel_capacity_kw` behaves like peak DC/AC capacity, although its meaning is
  not formally defined.
- Radiation and UV are both multiplied even though they are correlated and
  observed radiation already reflects atmospheric conditions.
- Output is normalized relative to the maximum of the returned 24-item horizon, so
  identical irradiance can map differently across responses.
- Live UV and cloud factors are not clipped; malformed provider values can yield
  output outside `[0, panel_capacity]`.
- Tilt, azimuth, panel area, STC irradiance, temperature derating, inverter
  efficiency, shading, snow/soiling, and wiring losses are absent.
- The 100 kW default is unusually large for a normal home and dwarfs the 15 kWh
  default battery.

Source: `services/solar_service.py:3-31`.

### 4.3 Live tariff curve

The API ignores tariff fields supplied in `settings` and creates:

| Hours | Live tariff `T_t` (INR/kWh) |
|---|---:|
| 00–05 | 4.0 |
| 06–13 | 5.5 |
| 14–16 | 4.0 |
| 17–20 | 7.5 |
| 21–23 | 5.5 |

The optimizer defines export price as `T_sell,t = T_t - 1`.

Source: `app.py:275-282`; `services/optimizer_service.py:74`.

### 4.4 Live baseline consumption

The API creates this fixed background load:

| Hours | `L_t` (kW) |
|---|---:|
| 00–04 | 2.0 |
| 05–08 | 3.0 |
| 09–17 | 2.5 |
| 18–21 | 5.0 |
| 22–23 | 3.5 |

User devices are not included in the Random Forest input. They are added only as
decision-dependent load inside the optimizer.

Source: `app.py:284-318`.

### 4.5 Random Forest inference

The fixed feature vector for hour `t` is:

```text
X_t = [L_t, T_t, S_t]
```

The saved `MinMaxScaler` applies each feature's training min/max:

```text
x_scaled_j = (x_j - min_j) / (max_j - min_j)
```

The 100 regression trees each return a leaf mean, and the forest averages them:

```text
y_hat_t = (1 / 100) * sum(tree_k(X_scaled_t), k=1..100)
```

Finally:

```text
battery_forecast_t = round(clamp(y_hat_t, 0, battery_capacity_kwh), 3)
```

Every hour is predicted independently. The following are **not model features**:

- hour/day/season;
- current or previous battery level;
- initial battery percentage;
- battery capacity, except post-prediction clipping;
- battery charge/discharge decisions;
- device schedule/load;
- location or household identity;
- temperature, humidity, or cloud cover directly.

`initial_battery_pct` is accepted by the Python function but unused. Therefore the
same `[load, tariff, solar]` sequence gives the same raw forecast for 0%, 50%, or
100% starting charge.

Source: `services/model_manager.py:33-40,210-231`.

### 4.6 Optimization and response

The optimizer receives tariff, solar, baseline consumption, initial battery energy,
capacity, devices, and mode. It does not receive `battery_forecast`. Its exact
mathematical formulation is in section 8.

The response combines the optimizer result with the first weather observation,
tariffs, the independent ML forecast, and timing. The full snapshot is then
best-effort persisted for retraining.

## 5. How the training target is manufactured

### 5.1 Historical weather

Retraining requests hourly temperature, cloud cover, direct radiation, diffuse
radiation, and UV from Open-Meteo with timezone `Asia/Kolkata`:

```text
G_t = direct_radiation_t + diffuse_radiation_t
```

If the API path fails, it creates deterministic synthetic weather instead.

Source: `services/data_collection_service.py:56-110,181-225`.

### 5.2 Historical solar

Training solar uses:

```text
S_t = panel_capacity
      * (1 - cloud_cover_t / 100)
      * (G_t / G_max_over_entire_training_window)
      * clip(uv_index_t / 11, 0, 1)
```

and forces zero output for hours 19–23 and 00–04.

This differs from live inference: training normalizes over the entire multi-month
window, while live calculation normalizes over one 24-hour response; training clips
UV while live code does not.

Source: `services/data_collection_service.py:113-135`.

### 5.3 Synthetic tariff and consumption

Training tariff differs from the live curve:

| Hours | Training INR/kWh | Live INR/kWh |
|---|---:|---:|
| 00–05 | 3.5 | 4.0 |
| 06–08 | 5.0 | 5.5 |
| 09–13 | 5.5 | 5.5 |
| 14–16 | 4.0 | 4.0 |
| 17–20 | 7.5 | 7.5 |
| 21–23 | 5.0 | 5.5 |

Training consumption is also different at 13:00–15:59: it is 3.0 kW versus the
live 2.5 kW. These are direct train/serve distribution shifts.

Source: `services/data_collection_service.py:22-53`; `app.py:275-292`.

### 5.4 Synthetic battery label recurrence

The training generator starts at half capacity:

```text
E_initial = 0.5 * battery_capacity
```

For row `i`, it uses `hour = i mod 24`, not the datetime's hour. Its target is:

```text
if hour >= 23 or hour < 5:
    E_t = E_previous
elif S_t > 0:
    E_t = min(E_previous + S_t, E_max)
elif T_t <= 4 and E_previous < E_max:
    E_t = min(E_previous + 1, E_max)
else:
    discharge = min(L_t, E_previous)
    E_t = max(0, E_previous - discharge)
```

Source: `services/data_collection_service.py:138-178`.

This recurrence assumes a one-hour step and 100% efficiency. It is not a physical
energy balance:

- Night load from 23:00 through 04:59 does not reduce the battery.
- When solar is positive, all solar is added and house load is ignored.
- During cheap grid charging, house load is ignored.
- There is no 3 kW charge/discharge limit like the optimizer has.
- There is no self-discharge, efficiency, degradation, reserve, or control policy
  state.
- Missing rows still affect the sequential state before rows with nulls are
  removed.

The model is therefore trained to approximate a deterministic but stateful rule
without receiving the previous state. The same current `L_t`, `T_t`, and `S_t` can
correctly map to many battery levels depending on history, so the learning problem
is under-specified.

## 6. Model training, evaluation, and promotion

### 6.1 Training inputs and weights

The lifecycle trainer combines:

| Data source | Effective treatment | Label quality |
|---|---|---|
| Open-Meteo/synthetic top-up | One copy | Deterministic simulated SOC |
| `actual_readings` | Three identical copies | Potential ground truth, but no active ingestion caller |
| All `prediction_logs` | Two identical copies | Old model prediction, not measured SOC |

Each `PredictionLog` is exploded into 24 rows. Scheduled device load is added to
the stored baseline consumption feature, but its battery target remains the old
forecast produced before scheduling. This makes those feature/label pairs
internally inconsistent.

All logs are loaded again on every retrain. `used_for_training` is marked only after
promotion and is never a query filter.

Promotion also bulk-stamps every previously unmarked log with the new
`model_version`, including logs skipped during row construction. The field therefore
does not reliably identify either the model that generated a prediction or the
specific training rows consumed by the stamped version.

Source: `services/model_manager.py:45-96,236-327`.

### 6.2 Preprocessing and split

Current sequence:

1. Concatenate and duplicate/weight rows.
2. Drop rows containing nulls.
3. Fit `MinMaxScaler` on **all** rows.
4. Randomly split scaled rows 80/20 with seed 42.
5. Fit `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`.
6. Calculate RMSE and R² on the random test rows.

Scaling is unnecessary for ordinary decision trees because monotonic scaling
preserves ordering and possible split partitions. If a shared pipeline retains it,
fit it on training data only.

### 6.3 Promotion rule

The first model is promoted automatically. Later, the rule is:

```text
promote if new_R2 - active_R2 >= 0.01
```

This compares scores from changing data and different random test populations,
not champion and challenger on the same immutable holdout. It does not check RMSE,
business impact, physical constraint violations, inference compatibility, or
latency.

Source: `services/model_manager.py:98-174,330-339`.

## 7. Metrics and artifact evidence

### 7.1 Metric definitions

For actual values `y_i`, predictions `y_hat_i`, and mean target `y_bar`:

```text
RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))

R2 = 1 - sum((y_i - y_hat_i)^2) / sum((y_i - y_bar)^2)
```

- RMSE is in kWh. Lower is better and large errors are penalized quadratically.
- R² compares squared error with the constant-mean baseline. `1` is perfect, `0`
  matches that baseline, and a negative value is worse than predicting the mean.
- R² is not “accuracy” and RMSE is not meaningful without target scale, a baseline,
  and trustworthy labels.

MAE would be:

```text
MAE = (1/n) * sum(abs(y_i - y_hat_i))
```

It is easier to interpret and less sensitive to outliers than RMSE. MAPE is a bad
primary SOC metric because battery energy can be zero, causing division by zero or
extreme percentages.

### 7.2 Recorded training attempts

Local logs record:

| Attempt | Effective rows | RMSE | R² | Outcome |
|---|---:|---:|---:|---|
| Earlier retrain | 910 | 0.0694 kWh | 0.0000 | Rejected by improvement rule |
| Version 2 | 454 = 310 API + 2 × 72 log rows | 0.7728 kWh | 0.0209 | Promoted |
| Later challenger | 502 = 310 API + 2 × 96 log rows | 1.0338 kWh | 0.0169 | Rejected |

The promoted score explains only about 2.1% of variance in its nominal test labels.
Those labels/splits are compromised, so even that weak number is not an unbiased
estimate of real-world performance. A low RMSE paired with R² near zero can occur
when target variance is very small; it does not prove the first attempt was good.

Sources: `logs/illuminate_2026-06-26.log`; training formulas in
`services/model_manager.py:98-124`.

### 7.3 Current local model artifact

Read-only inspection found that `models/battery_model.pkl` is byte-identical to the
local version-2 file and contains:

| Property | Observed value |
|---|---|
| Estimator | `RandomForestRegressor` |
| Trees | 100 |
| Random seed | 42 |
| Max depth | Unlimited/default |
| Min samples split/leaf | 2 / 1 |
| Bootstrap | Enabled |
| Features considered per split | All three (`max_features=1.0`) |
| Split criterion | Squared error |
| Out-of-bag scoring | Disabled |
| Scaler rows seen | 454 |
| Feature minima `[load, tariff, solar]` | `[2.0, 3.5, 0.0]` |
| Feature maxima `[load, tariff, solar]` | `[5.0, 7.5, 61.364]` |
| Impurity importance: load | 0.175767 |
| Impurity importance: tariff | 0.816161 |
| Impurity importance: solar | 0.008072 |

Impurity importance describes split usage in this fitted forest. It is not causal,
can be biased by feature cardinality/correlation, and is especially unreliable as
a domain conclusion when labels and validation are weak. Tariff dominance and
near-zero solar importance are suspicious for a physical SOC model.

With fallback weather and built-in live load/tariffs, the current local artifact's
24-hour output was approximately 0.000–0.105 kWh regardless of initial charge—a
maximum of 0.7% of a 15 kWh battery. That is diagnostic evidence, not a production
SLA.

The earlier local v1 artifact's scaler saw 910 rows but solar minimum and maximum
were both zero, making solar unusable to that fit. The corresponding log says 2,184
weather rows were fetched and only 910 remained in the training frame. Missing
daylight weather/UV fields followed by `dropna()` is a plausible explanation, but
the retained rows were not preserved, so treat the cause as an inference rather
than a proven fact.

### 7.4 Tracked legacy CSV

`data/consumption_battery_data.csv` is not used by the current lifecycle trainer.
Its observed profile is:

| Property | Value |
|---|---:|
| Data rows | 2,624 |
| Range | 2024-07-01 00:00 through 2024-10-20 23:00 |
| Unique timestamps | 2,600 |
| Missing hourly timestamps within range | 88 |
| Timestamp values duplicated | 24 |
| Completely duplicated rows | 18 |
| Null values | 0 |
| Load min / mean / max | 2.0 / 3.059 / 5.0 kW |
| Solar min / mean / max | 0 / 6.003 / 79.045 kW |
| Battery min / mean / max | 0 / 13.918 / 30 kWh |

The 30 kWh target maximum conflicts with the API's 15 kWh default capacity. The
legacy `train_model.py` also requires an absent
`data/solar_tariff_generated_data.csv`, concatenates rows by index rather than
timestamp, and computes no evaluation metric. Do not use it as the supported
training command.

An empirical reconstruction, not tracked generator code, explains almost the whole
CSV: treat the first 15 kWh value as initial state; hold battery before 05:00; add
all positive solar and cap at 30 kWh; otherwise subtract load and floor at zero.
Including the initial row, that rule matches 2,622 of 2,624 rows. At CSV lines
1016–1017 the stored battery instead rises `15 -> 16 -> 17 kWh` despite zero solar
and positive load. Duplicate timestamps can also apply a sequential transition more
than once. These anomalies require cleaning/provenance, not a model workaround.

## 8. Exact mixed-integer optimizer

### 8.1 Why this is a MILP

Energy flows and costs are modeled with linear equations. Device decisions are
binary. Combining continuous linear variables with integer variables makes this a
mixed-integer linear program (MILP), solved here by CBC through PuLP. MILPs are
generally NP-hard, although this 24-hour instance is small under normal device
counts.

### 8.2 Decision variables and bounds

For every hour `t`:

```text
x_d,t in {0,1}             device d off/on
0 <= C_t <= 3              battery charge power
0 <= D_t <= 3              discharge power; upper bound is 0 in full_backup
0 <= E_t <= E_max          stored battery energy
G+_t >= 0                  grid import
G-_t >= 0                  grid export
```

Device binary variables are omitted entirely in `low_power` mode.

Source: `services/optimizer_service.py:77-100`.

### 8.3 Objective

The implementation minimizes:

```text
sum_t(T_t * G+_t - (T_t - 1) * G-_t)
+ sum_d sum_t(w_d * x_d,t)
```

where:

```text
w_high   = 0.0
w_normal = 0.1
w_low    = 0.5
```

The first term is import cost minus export revenue. The second is labeled a
priority penalty but is not in monetary units.

Because every active device is constrained to run exactly `H_d` slots:

```text
sum_t(x_d,t) = H_d
```

its priority contribution is always `w_d * H_d`, a constant over all feasible
schedules. It cannot change which hours are selected. It only changes the number
reported as `total_grid_cost`.

`MODE_CONFIG.grid_buy_penalty` is never multiplied into this objective. This makes
`self_consumption` and `tou_savings` the same effective optimization model.

Source: `services/optimizer_service.py:10-37,101-110,129-134`.

### 8.4 Energy balance

For each hour, code enforces:

```text
L_t + sum_d(P_d * x_d,t)
    <= G+_t - G-_t + D_t + S_t - C_t
```

Battery state is:

```text
E_0 = E_initial + C_0 - D_0
E_t = E_(t-1) + C_t - D_t, for t > 0
```

Source: `services/optimizer_service.py:112-128`.

The one-hour assumption makes the numeric kW-to-kWh transition appear valid, but
the production form should be:

```text
E_(t+1) = E_t + eta_charge * C_t * delta_t
                - D_t / eta_discharge * delta_t
```

A clearer physical balance uses equality and explicit curtailment:

```text
G+_t + S_t + D_t
    = L_t + device_load_t + C_t + G-_t + curtailment_t
```

### 8.5 Device constraints

For each active device:

```text
sum_t(x_d,t) = usage_hours_d
x_d,t = 0 when t < earliest_hour or t >= latest_hour
```

The contiguity implementation creates transition variables and limits positive
off-to-on transitions after hour 0 to one. It does not count a block that begins at
hour 0, so a device may run at hour 0, turn off, and start a second block later.

A robust fixed-duration formulation introduces start variables `s_d,t`:

```text
sum_t(s_d,t) = 1
x_d,h = sum of feasible s_d,t whose duration block covers h
```

or uses complete start/stop transition equations including the boundary state at
`t = -1`. Validate that `usage_hours <= latest - earliest` before solving.

Source: `services/optimizer_service.py:129-149`.

### 8.6 What each mode really does

| Mode name | Actual code effect | Intended-name mismatch |
|---|---|---|
| `tou_savings` | Devices modeled; discharge allowed up to 3 kW | Baseline |
| `self_consumption` | Same as `tou_savings` | Configured import penalty is unused |
| `full_backup` | Discharge upper bound becomes zero | Battery need not become or remain full |
| `low_power` | All device decision variables disappear | Heavy loads are not selectively reduced/scheduled |

The schema also accepts reserve percentage, net metering, custom peak window, and
custom rates, but none are passed into the optimizer.

### 8.7 Worked one-hour energy example

Assume one hour with:

```text
baseline load       = 2.5 kW
scheduled device    = 1.0 kW
solar               = 2.0 kW
battery discharge   = 0.5 kW
battery charge      = 0.0 kW
tariff              = 5.5 INR/kWh
grid export         = 0.0 kW
```

Required grid import at equality is:

```text
G+ = 2.5 + 1.0 - 2.0 - 0.5 = 1.0 kW
```

For a one-hour slot, import energy is 1 kWh and its cost is:

```text
1 kWh * 5.5 INR/kWh = 5.5 INR
```

Battery energy falls by 0.5 kWh under the code's lossless recurrence. With 90%
discharge efficiency, supplying 0.5 kWh would instead reduce stored energy by
`0.5 / 0.9 = 0.556 kWh`.

### 8.8 Optimizer correctness and business-metric gaps

1. `grid_buy_penalty` is unused.
2. Priority is constant and does not influence schedule placement.
3. `total_grid_cost` contains a dimensionless priority constant and is therefore
   not a pure bill.
4. Legacy `estimated_savings = abs(total_grid_cost)` has no baseline and is not
   savings.
5. `full_backup` forbids discharge but does not enforce full/reserve/terminal SOC.
6. `low_power` silently ignores all supplied devices.
7. Minimum reserve, custom tariff, and net-metering settings are ignored.
8. Grid import/export and battery charge/discharge are not explicitly mutually
   exclusive. Current prices discourage some simultaneous flows but do not make the
   physical model complete.
9. Grid charging followed by peak export is allowed; whether that is legal depends
   on the tariff/net-metering contract.
10. The inequality permits energy to disappear as implicit curtailment without
    naming or pricing it.
11. There is no terminal SOC constraint/value, encouraging the solver to empty the
    battery at the end of the horizon when economical.
12. Battery efficiency, self-discharge, degradation, cycle count, export limits,
    inverter limits, and demand charges are absent.
13. Duplicate device names collide in response maps even though solver variables
    are unique.
14. Device count is unbounded on a public endpoint, creating a solver/DB resource
    exhaustion path.
15. Impossible usage windows become solver infeasibility instead of an early 422
    validation message.

## 9. Assumptions catalogue

### 9.1 Explicit or directly encoded

- Forecast horizon is exactly 24 one-hour periods.
- All list inputs to model inference are assumed to contain at least 24 elements.
- Battery capacity defaults to 15 kWh; solar capacity defaults to 100 kW.
- Battery charging/discharging is capped at 3 kW in the optimizer.
- Export tariff is always import tariff minus 1 INR/kWh.
- Weather cache freshness is 30 minutes.
- Live weather can silently fall back to stale or synthetic values.
- Training can silently fall back to fully synthetic weather.
- Historical training timezone is Asia/Kolkata.
- The weekly job runs Sunday 02:00 in scheduler/host default timezone.
- Model promotion uses a +0.01 absolute R² threshold.
- Real readings receive 3x duplication and prediction logs receive 2x duplication.

### 9.2 Implicit and risky

- Weather list index, embedded weather hour, tariff index, and consumption index all
  refer to the same local clock hour. A rolling next-24-hours response can violate
  this.
- kW over every slot can be treated numerically as kWh because `delta_t = 1 hour`.
- Battery/panel/site characteristics are homogeneous enough for one global model.
- Synthetic targets are adequate substitutes for actual meter readings.
- A battery state can be inferred without initial/previous state.
- Provider radiation, cloud, and UV fields are always present, numeric, and sane.
- Export is always permitted and unlimited.
- User IDs supplied to public routes are safe UUIDs and callers may read those rows.
- Local filesystem model state is persistent and consistent across processes.
- A successful response may still be considered successful if persistence failed.

### 9.3 Questions requiring a domain owner

1. What are the real battery capacity, usable SOC bounds, efficiencies, charge and
   discharge limits, and degradation cost?
2. Is 100 kW an intentional panel default or a prototype value?
3. What location/tariff source defines imports, exports, taxes, and demand charges?
4. Can grid-charged energy legally be exported?
5. What exact mathematical outcomes distinguish each mode?
6. Must the terminal battery match the initial level or a reserve?
7. Which meters provide actual load, PV, SOC, import/export, and device state?
8. What timestamp, interval, timezone, missing-data, and late-arrival rules apply?
9. Is the forecast global, regional, site-specific, or battery-specific?
10. Should degraded/synthetic weather be visible to clients?
11. What baseline defines savings: unscheduled appliances, no battery, fixed user
    schedule, or historical bill?
12. Which quality, cost, feasibility, latency, and availability thresholds decide
    production readiness?

## 10. A production-grade technical direction

### 10.1 Put learning and physics in the right places

The strongest redesign is:

```text
measured history + weather/calendar
          |
          +--> load forecast + uncertainty
          +--> PV forecast + uncertainty
                         |
current measured SOC --> physical battery state model --> MPC/MILP
                         |
                         +--> schedule, cost, confidence, constraints
```

Use ML for quantities that are uncertain—future household load and PV production.
Use conservation equations for battery state. This avoids asking a model to learn a
known recurrence and guarantees that the optimizer respects physical bounds.

Useful forecast features include:

- lagged load/PV and rolling averages;
- hour-of-day and day-of-week encoded as sine/cosine;
- season/holiday/occupancy indicators;
- weather forecast values and forecast horizon;
- panel capacity, orientation, and site losses;
- household/site identifier or a hierarchical strategy;
- planned device load;
- missingness and data-quality indicators.

Current SOC belongs in the state equation and may also help detect meter/model
inconsistency. Capacity and hardware limits must be explicit configuration.

### 10.2 Candidate model strategy

Start with baselines before more complex models:

1. persistence (`next hour = recent value`);
2. hour-of-week mean/median;
3. deterministic clear-sky/physics PV baseline;
4. Random Forest or gradient-boosted trees for tabular lag/weather features;
5. sequence models only when enough clean multi-site time-series data and a proven
   incremental benefit justify their complexity.

Random Forest remains a reasonable tabular baseline because it handles nonlinear
interactions, tolerates mixed feature scales, and reduces the variance of a single
tree through bootstrap aggregation. It is not inherently temporal and generally
cannot extrapolate beyond observed target regimes.

### 10.3 Data contract and ground truth

An actual-reading event should have, at minimum:

```text
site_id, device/meter_id, interval_start_utc, interval_minutes,
timezone, load_kwh, pv_kwh, grid_import_kwh, grid_export_kwh,
battery_soc_pct or battery_energy_kwh, charge_kwh, discharge_kwh,
quality_flags, source, ingestion_time, schema_version
```

Define deduplication keys, late-data corrections, missing intervals, sensor
calibration, unit conversion, timezone/DST behavior, and privacy/retention before
training. Prediction snapshots are useful context but never replace outcomes.

### 10.4 Evaluation design

1. Freeze an untouched chronological holdout by complete days and sites.
2. Use rolling-origin backtests to measure different seasons and forecast horizons.
3. Keep duplicate/site/request groups entirely in one split.
4. Fit preprocessing on training folds only.
5. Apply real sample weights only to the training loss; never duplicate into the
   validation set.
6. Evaluate champion and challenger on the exact same holdout.
7. Slice metrics by horizon, site, capacity, season, weather regime, missingness,
   and fallback provenance.
8. Backtest the full optimizer with identical constraints and compare realized
   costs against a defined baseline.

Recommended metrics:

| Layer | Metrics |
|---|---|
| Load/PV forecasts | MAE, RMSE, normalized RMSE, bias, horizon-wise error, prediction-interval coverage |
| Battery state | MAE/RMSE in kWh and SOC points, bound violations, state-equation residual |
| Optimizer | feasibility rate, optimality gap, p50/p95 solve time, reserve violations, terminal SOC |
| Business | realized cost versus baseline, cost regret versus perfect foresight, peak reduction, self-consumption ratio |
| Data/model health | missingness, feature/label drift, stale/synthetic input share, residual drift |
| Service | request rate, p50/p95/p99 latency, error/degraded rate, DB/solver/weather dependency health |

Normalized RMSE by capacity is useful across different batteries:

```text
NRMSE_capacity = RMSE_kwh / battery_capacity_kwh
```

Cost regret measures downstream consequence:

```text
cost_regret = realized_cost_using_forecasts
              - realized_cost_with_perfect_future_information
```

### 10.5 Correct savings definition

Run the optimized and baseline schedules under the **same realized** load, PV,
tariff, initial/terminal SOC, and export policy:

```text
savings_INR = baseline_realized_cost - optimized_realized_cost
savings_pct = savings_INR / baseline_realized_cost * 100
```

Define the baseline explicitly. Examples are the user's original device schedule or
a rule-based “no shifting” schedule. `abs(optimized_cost)` is never a valid savings
calculation.

### 10.6 Safer model promotion and serving

A production model version should record:

- artifact URI and checksum/signature;
- estimator and feature schema versions;
- training code/data revision and time range;
- site/location/capacity population;
- all validation metrics and slice metrics;
- creation/promoter identity and reason;
- dependency/runtime versions;
- status transitions and rollback target.

Promotion should use an immutable holdout, pass physical/business guardrails, write
an artifact atomically to a trusted registry, commit version metadata, and update a
single alias only after verification. Serving workers should load by immutable
version and receive a coordinated reload or deploy. Retraining requires a
distributed lock or a single external job worker.

### 10.7 Test strategy

High-value automated tests are:

- boundary tests for every tariff/load hour;
- solar tests for night, zero radiation, malformed/extreme weather, and capacity;
- property tests for exact energy conservation, SOC bounds, and no impossible
  simultaneous flows;
- optimizer tests for each mode, windows, zero-hour devices, contiguity, duplicate
  names, infeasibility, reserve, terminal SOC, and known golden schedules;
- leakage tests proving a group never crosses train/test;
- deterministic backtest and artifact-schema compatibility tests;
- API validation/error/auth/ownership/idempotency tests;
- weather timeout, stale-cache, malformed response, and provenance tests;
- DB migration/rollback, stale-connection, and best-effort persistence tests;
- concurrent retrain, atomic promotion, multi-worker reload, and rollback tests;
- load tests with bounded device counts and solver time limits.

## 11. How to present this project in an SDE-2 interview

### 11.1 Two-minute project explanation

> Illuminate is a smart-home energy planning service built with FastAPI,
> PostgreSQL, scikit-learn, and PuLP/CBC. A prediction request retrieves cached or
> live weather, estimates solar output, constructs a 24-hour tariff/load horizon,
> produces an advisory Random Forest battery forecast, and solves a mixed-integer
> program for appliance timing and grid/battery flows. The service best-effort
> persists a detailed prediction snapshot, versions retrained models, protects
> administrative operations with bcrypt and JWT, and attaches request IDs for
> observability. During review I identified that the forecast is independent of the
> optimizer, provenance is incomplete, and much of the training data is simulated
> or pseudo-labeled. My production plan is to forecast uncertain load/PV from
> measured outcomes, preserve battery dynamics as hard physics constraints,
> evaluate chronologically against fixed baselines, and make model promotion atomic
> and single-writer.

This answer demonstrates architecture, boundaries, and engineering judgment without
overstating model quality.

### 11.2 A strong deep-dive structure

Use this order when the interviewer asks for detail:

1. **Problem:** shift flexible loads and battery use to reduce cost while respecting
   energy and user constraints.
2. **Contract:** one request describes site state/settings/devices; one response
   provides a 24-hour plan and forecast.
3. **Pipeline:** weather → solar/load/tariff → forecast + MILP → persistence.
4. **Why MILP:** binary device decisions plus linear power/cost constraints.
5. **Why RF baseline:** nonlinear tabular baseline, easy deployment; acknowledge
   missing temporal/state features.
6. **Correctness:** units, conservation, SOC/reserve/terminal constraints, explicit
   baseline savings.
7. **Reliability:** cache/fallback, DB lifecycle, idempotency, single-writer model
   promotion, multi-worker behavior.
8. **Measurement:** chronological evaluation, quality slices, cost regret, solver
   feasibility/latency, service SLOs.
9. **Security:** user ownership, rate limits, secrets, JWT limitations, retention.
10. **Next iteration:** real telemetry, physics-based state, load/PV forecasts,
    robust/MPC optimization.

### 11.3 What not to claim

Do not say:

- the model learns from real-world prediction outcomes—the active ingestion path
  mainly uses simulations and old predictions;
- R² 0.0209 is good accuracy;
- the optimizer uses the battery forecast;
- priority currently changes schedule placement;
- `full_backup` keeps the battery full;
- `estimated_savings` is measured savings;
- all request settings are implemented;
- weekly retraining or local pickle files are safe across multiple replicas;
- the current service is production-ready.

A stronger answer is: “That is the current limitation, here is why it matters, and
here is the concrete design/test I would use to fix it.”

## 12. Interview questions and model answers

### Architecture and project framing

#### 1. Is this an ML project or an optimization project?

It is a hybrid service. The Random Forest emits an advisory battery forecast; the
PuLP MILP performs scheduling. In the current code the two branches share load,
tariff, and solar inputs, but the forecast does not feed the MILP. I would describe
that honestly and then explain how load/PV forecasts should feed a physics-based
optimizer.

#### 2. Walk me through one `/predict` request.

FastAPI parses the request and performs limited schema validation; several practical
bounds are still absent. The weather service checks a 30-minute DB cache and calls
its provider or fallback; solar, fixed tariffs, and fixed baseline load are
calculated; the model produces 24 independent SOC-energy predictions; CBC solves
the device/energy MILP; the API best-effort logs the snapshot and returns the
schedule, objective, weather summary, tariffs, forecast, and request time.

#### 3. Why FastAPI?

It provides typed Pydantic validation, dependency injection for DB sessions,
automatic OpenAPI, and a low-friction Python path to scikit-learn and PuLP. Current
handlers are synchronous, so FastAPI runs them in its threadpool; external I/O and
CPU/solver limits still need explicit capacity planning.

#### 4. Why PostgreSQL instead of a document database?

The service has relational identity/history/version metadata and needs transactions,
constraints, indexing, and auditable model state. PostgreSQL also supports JSON for
24-hour snapshots. JSON is useful for raw lineage, but commonly queried values
should be normalized or indexed rather than hiding everything in blobs.

#### 5. What is the source of truth for a prediction?

Ideally an immutable record containing request, input provenance, model version,
feature schema, optimizer version, output, and later actual outcome. Today
`PredictionLog` stores most request/derived/output data but does not record the
generating model version correctly and may fail to persist without failing the API.

#### 6. What would you split into separate services first?

I would not split merely for fashion. First isolate retraining/scheduled jobs from
request serving because they have different resource, concurrency, and failure
profiles. At higher scale I would separate telemetry ingestion and perhaps forecast
serving, while keeping request orchestration and the tightly coupled small MILP
together until measurements justify another boundary.

#### 7. What are the service's main failure domains?

PostgreSQL availability/stale pooled connections, weather timeout or malformed
data, missing/corrupt/incompatible pickle, solver infeasibility/resource exhaustion,
filesystem failure, concurrent retraining races, and swallowed persistence failure.
Each needs an explicit timeout, status/provenance, metric, and recovery policy.

#### 8. Is it production-ready?

No. It is a functional prototype with a useful end-to-end skeleton. P0 gaps include
ordinary-user authorization, rate/input limits, migrations/tests, physically correct
constraints, real labels, reliable artifact promotion, and multi-worker job/model
coordination.

### Machine learning and data

#### 9. How does Random Forest regression work?

Each CART regression tree trains on a bootstrap sample and recursively selects
feature thresholds that reduce squared-error impurity. Bootstrap differences are
the main decorrelation source here because `max_features=1.0` considers all three
features at every split; random tie-breaking can add variation. The forest averages
leaf predictions, reducing the variance of a single deep tree at the cost of
computation and some bias.

#### 10. Why choose Random Forest here?

It is a quick baseline for nonlinear tabular interactions, handles mixed feature
scales, requires little preprocessing, and is easy to serve. Regression trees with
squared-error splits and leaf means can still be sensitive to target outliers.
More importantly, battery SOC is stateful and the current feature set has no
state/history, so the problem formulation—not only the algorithm—is the bigger
limitation.

#### 11. Why is MinMax scaling unnecessary for this model?

Trees compare ordering against thresholds. A monotonic linear rescale preserves
order and all possible partitions, unlike k-nearest neighbors, SVMs, neural nets, or
regularized linear models where scale affects geometry or gradient/penalty size.

#### 12. Why can three current-hour features not determine SOC?

SOC is a state variable. Two homes can have identical current load, tariff, and PV
but different SOC due to their initial charge and every earlier charge/discharge
decision. Without previous SOC/state and controls, the mapping is one-to-many.

#### 13. What does R² = 0.0209 mean?

On that nominal random test set, squared error improved only slightly over predicting
the test-target mean—about 2.1% of variance under the R² definition. It is a weak
score, and the duplicated/pseudo-labeled split makes it an unreliable estimate of
real performance.

#### 14. Why can RMSE look small while R² is near zero?

RMSE is absolute and depends on target scale/variance. If targets are nearly
constant, even tiny errors can fail to improve on the mean, producing R² near zero.
Always show target distribution and naive-baseline RMSE alongside both metrics.

#### 15. What leakage exists?

Rows are duplicated before a random split, so exact copies can cross train/test;
the scaler is fit on all rows; temporal neighbors and repeated daily patterns are
randomized; and model promotion compares changing holdouts. Scaler leakage has
little practical effect on this RF, but the evaluation design remains invalid.

#### 16. What is the most serious label problem?

Prediction logs use the previous model's forecast as the next training target, not
measured battery SOC. Device load is then added to the feature after that target was
generated without it. This self-training loop can reinforce errors and cannot show
improvement against reality.

#### 17. How should you split the data?

Use chronological rolling-origin evaluation, keeping all rows from a day/request and
ideally a site in one fold. Reserve a final untouched recent period. If generalizing
to new homes, also test a site-held-out split. Champion and challenger must share
the same fixed evaluation population.

#### 18. What baseline models would you use?

For load/PV: persistence, recent moving average, hour-of-week median, and a simple
weather/clear-sky rule. For battery state: the physical recurrence from measured SOC
and actions. A complex model must beat these on forecast metrics and realized
optimizer outcomes.

#### 19. What features are missing?

Previous measured SOC, charge/discharge actions, capacity/limits, lags and rolling
load/PV, forecast horizon, cyclic time/season/holiday, detailed forecast weather,
site/panel configuration, planned device load, and quality/fallback flags.

#### 20. Can Random Forest extrapolate to a new capacity?

Generally no. Tree predictions are averages of observed training labels and do not
learn a continuous physical capacity scaling. Post-hoc clipping prevents impossible
upper values but does not adapt predictions to 5, 15, or 30 kWh behavior.

#### 21. Does feature importance show tariff causes SOC?

No. Impurity importance measures how much fitted splits reduce training impurity.
It is affected by cardinality, correlation, repeated synthetic patterns, and label
construction. Use permutation/SHAP cautiously for association and controlled domain
experiments for causal claims.

#### 22. How would you monitor drift?

Track input missingness/ranges and distributions by site/season/provider, share of
stale/synthetic weather, prediction residuals by horizon once outcomes arrive,
capacity-normalized errors, bias, bound/state-equation violations, optimizer cost
regret, and recommendation acceptance. Alert on sustained changes, not one noisy
sample.

#### 23. What should trigger retraining?

Enough new validated ground truth plus measured drift or scheduled review. A weekly
clock alone is not evidence that a new model is needed. Training should run through
data-quality gates, backtests, champion/challenger evaluation, approval, and atomic
promotion.

#### 24. How would you explain bias versus variance here?

A deep single tree has low training bias and high variance. Bagging many decorrelated
trees reduces variance. But no ensemble fixes specification bias: omitting previous
SOC makes the best learnable conditional average physically ambiguous.

#### 25. Why not use an LSTM immediately?

Sequence models can represent temporal state, but they need clean, sufficiently
large, representative sequences and add serving/tuning complexity. First fix labels,
state equations, baselines, and evaluation. Then adopt a sequence model only if it
improves fixed backtests and downstream cost.

### Optimization and energy theory

#### 26. Why is the scheduler a MILP?

Power balances, SOC dynamics, limits, and energy cost are linear; device on/off and
possibly mutually exclusive operating modes are binary. Integer decisions make it
mixed-integer rather than a plain linear program.

#### 27. Why does priority have no effect today?

The objective adds `w_d * sum_t(x_d,t)`, while a constraint fixes that sum to the
required usage hours. It is constant for that device. Priority must instead affect
allowed curtailment, lateness/deviation from preference, interruption, or a
time-dependent utility/penalty.

#### 28. Why are self-consumption and TOU modes identical?

The only configured distinction is `grid_buy_penalty`, but the objective never uses
it. A real self-consumption objective could minimize exports/imports or maximize
on-site PV use, while TOU should minimize monetary bill under tariff and terminal-SOC
constraints.

#### 29. What does full-backup mode actually guarantee?

Only that modeled discharge is zero. It neither requires charging nor full/reserve
SOC. A correct backup mode needs `E_t >= reserve_t`, possibly a terminal target, and
an objective/value for increasing SOC before outage risk.

#### 30. How would you prevent simultaneous charge/discharge or import/export?

Add binary mode variables with tight big-M bounds, for example
`C_t <= M_c z_t` and `D_t <= M_d(1-z_t)`. Do similarly where import/export must be
exclusive. Sometimes prices/efficiency already make simultaneity suboptimal, but
explicit constraints protect against pathological tariffs and clarify physics.

#### 31. Why require terminal SOC?

Without a terminal value/constraint, a finite-horizon optimizer treats remaining
energy as worthless and often empties the battery at hour 23. Constrain final SOC to
at least initial/reserve SOC or add a credible terminal energy value.

#### 32. What is cost regret?

It is realized cost using forecast-driven decisions minus the cost achievable with
perfect future information under the same physical rules. It translates forecast
error into the decision/business impact the optimizer actually cares about.

#### 33. How do you model uncertainty?

At increasing complexity: safety margins/reserves, scenario optimization over load
and PV quantiles, chance constraints, robust bounds, or model-predictive control that
re-solves as observations arrive. Measure the tradeoff between cost and constraint
violation risk.

#### 34. What is MPC and why does it fit?

Model Predictive Control repeatedly forecasts a horizon, optimizes it, applies only
the next action, observes the new state, and re-solves. That corrects forecast error
and changing weather/usage instead of committing once to an open-loop 24-hour plan.

### Backend, reliability, and security

#### 35. Why is `create_all()` not database migration?

It can create missing tables but does not safely add/alter/drop existing columns,
constraints, or indexes with ordered rollout/rollback. Use versioned Alembic
migrations tested against the prior production schema.

#### 36. What transaction inconsistency exists in model promotion?

Artifact files are written before DB metadata commits. If the commit fails, the
current file and in-memory model can advance while the database still names the old
version. Use immutable upload, verification, metadata transaction, then atomic alias
switch; compensate/rollback on failure.

#### 37. What happens with multiple Uvicorn workers?

Each process has its own model, DB pool, and scheduler. Sunday jobs can race,
version numbers/files can collide, and only the retraining process reloads its
in-memory model. Put scheduling/retraining in one external worker and deploy or
broadcast immutable model versions.

#### 38. How should `/predict` retries work?

Today retries recompute and append another log. Define an idempotency key scoped to
caller/request, store result status atomically, and return the prior successful
result. Retry weather/transient DB work only within a bounded deadline; never blindly
retry an expensive solver request indefinitely.

#### 39. What is wrong with swallowing persistence errors?

The caller believes the operation fully succeeded while lineage/history/retraining
data may be absent. Depending on product semantics, either make persistence part of
the transaction and fail, or return an explicit `persistence_status/warnings` field
and metric so degraded success is observable.

#### 40. How would you improve health checks?

Separate liveness from readiness. Liveness says the process/event loop is alive.
Readiness checks DB connectivity, loaded compatible model/version, writable/available
artifact dependency as needed, and solver availability. Dependency degradation and
fallback rates belong in metrics, not necessarily a blocking health call.

#### 41. How would you handle stale PostgreSQL SSL connections?

Enable `pool_pre_ping`, choose connection recycle/keepalive settings appropriate to
the provider, bound connect/query timeouts, retry only safe operations, expose pool
metrics, and test provider idle-timeout behavior. The local logs contain an example
of a closed SSL connection failing `/predict`.

#### 42. What authentication gaps matter?

Admin login has no rate limit/lockout and JWTs lack issuer, audience, revocation,
and rotation design. More critically, ordinary user history/notifications are
public by supplied UUID, an IDOR risk. Use real user auth, ownership checks, scoped
tokens, rate limits, configurable CORS, audit logs, and secret management.

#### 43. Why is loading pickle risky?

Pickle deserialization can execute code. Treat artifacts like executables: restrict
write access, store in a trusted registry, verify checksum/signature and compatible
schema/runtime, and never load user-provided or untrusted pickle files.

#### 44. What observability would you add?

Structured metrics/traces correlated with request ID: latency by weather/model/solve/
DB phase, cache and fallback state, solver status/gap, model version, feature/data
quality, prediction distributions, persistence outcome, dependency errors, and
retraining/promotion audit events. Define SLOs and actionable alerts.

#### 45. How would you protect a public MILP endpoint?

Authenticate or quota clients; bound devices, names, power, windows, and payload
size; validate infeasibility early; set solver time/gap limits; isolate solver CPU;
apply rate/concurrency limits; cache when safe; and monitor complexity/timeout abuse.

#### 46. How would you evolve the API without breaking clients?

Make models strict, add typed response schemas and contract tests, publish a versioned
route such as `/v1`, support an explicit deprecation window or adapter for the stale
shape, generate clients from OpenAPI, and use additive changes until a major version.

#### 47. Which indexes would you consider?

Based on current queries: weather cache `(location, fetched_at DESC)`, optimization
history `(user_id, created_at DESC)`, unread notifications `(user_id, is_read,
created_at DESC)`, model versions `(is_active)` and `(trained_at DESC)`, and
prediction logs `(created_at DESC)`, `(used_for_training)`, or partitioning by time.
Validate with query plans and workload before finalizing.

#### 48. What are good service SLOs?

They must be product-defined, but propose measurable targets for successful
non-degraded prediction rate, p95/p99 latency, solver feasibility/timeout rate,
weather fallback share, persistence success, readiness, and model/business quality.
Separate online prediction SLOs from slower retraining-job objectives.

### Ownership and engineering judgment

#### 49. What would you fix first and why?

First secure user-scoped data and public resource use. In parallel, establish tests
and migrations so later changes are safe. Then fix physical/optimizer correctness
and ground-truth ingestion before tuning the model; optimizing a flawed target or
unmeasured system produces misleading progress.

#### 50. What was the hardest technical insight in this review?

The advertised ML and optimization stages are independent, and the retraining loop
uses predictions as labels. Recognizing those data-flow semantics matters more than
adjusting hyperparameters because it changes what the system can legitimately learn
and claim.

#### 51. How would you demonstrate impact after redesign?

Show an immutable backtest plus controlled pilot: forecast errors versus baselines,
zero physical constraint violations, solver feasibility/latency, realized bill
savings against a declared baseline, peak/self-consumption improvements, fallback
rates, and operational error/SLO results. Include negative or neutral findings.

## 13. Rapid calculation and revision sheet

### 13.1 Core formulas to remember

```text
Battery energy from SOC:
E_kwh = SOC_pct / 100 * capacity_kwh

MinMax scaling:
x' = (x - training_min) / (training_max - training_min)

Random Forest regression:
y_hat = average prediction of all trees

RMSE:
sqrt(mean((actual - predicted)^2))

R2:
1 - SSE / SST

One-hour import cost:
cost_INR = grid_import_kw * 1 hour * tariff_INR_per_kwh

Physical battery update:
E_next = E + eta_c * charge_kw * delta_t
           - discharge_kw / eta_d * delta_t

Savings:
baseline realized cost - optimized realized cost

Cost regret:
forecast-driven realized cost - perfect-foresight realized cost
```

### 13.2 Quick numeric examples

**SOC conversion:** 60% of a 15 kWh battery is:

```text
0.60 * 15 = 9 kWh
```

**Current artifact scaling:** using observed min/max values, a feature row
`[load=3, tariff=5.5, solar=30.682]` scales approximately to:

```text
load    = (3 - 2) / (5 - 2)          = 0.333
tariff  = (5.5 - 3.5) / (7.5 - 3.5) = 0.500
solar   = 30.682 / 61.364             = 0.500
```

`MinMaxScaler` does not clamp unseen values by default, so a live solar value above
61.364 can scale above 1. Trees still produce a leaf average; they do not extrapolate
the target physically.

**Train/test row counts:** with 454 rows and `test_size=0.2`, scikit-learn assigns
91 test rows and 363 training rows.

**RMSE normalization:** the logged 0.7728 kWh RMSE relative to 15 kWh is:

```text
0.7728 / 15 = 0.05152 = 5.152% of capacity
```

This arithmetic does not rescue the invalid label/split; it only normalizes scale.

**Efficiency:** charging at 3 kW for one hour with 90% efficiency stores:

```text
3 * 1 * 0.9 = 2.7 kWh
```

Discharging 3 kW to the load for one hour at 90% efficiency removes:

```text
3 * 1 / 0.9 = 3.333 kWh from storage
```

**Savings:** if a declared baseline costs INR 300 and optimized operation under the
same realized conditions costs INR 240:

```text
savings = 300 - 240 = INR 60
savings_pct = 60 / 300 * 100 = 20%
```

### 13.3 Complexity talking points

- Random Forest inference is roughly proportional to number of trees times tree
  depth per sample. Here it predicts 24 samples across 100 trees; this is small.
- Training cost grows roughly with trees, samples, features considered, and tree
  construction depth; `n_jobs=-1` uses available CPU cores.
- The optimizer always has 120 base continuous variables: five 24-hour arrays.
- Each modeled device adds 24 binary `x` variables. Each contiguous device adds 24
  more binary transition variables in the present formulation.
- MILP worst-case time can grow exponentially with integer variables; bound device
  count and solver time even if ordinary cases are fast.
- Each configured application process can open 5 pooled plus 10 overflow DB
  connections, so `workers * 15` is the theoretical per-service ceiling before
  considering other processes.

### 13.4 Common terminology

| Term | Meaning in this project |
|---|---|
| SOC | Battery state of charge, expressed as percentage or energy |
| HEMS | Home Energy Management System |
| PV | Photovoltaic solar generation |
| TOU | Time-of-use electricity pricing |
| MILP | Linear objective/constraints with continuous and integer variables |
| CBC | Open-source solver invoked by PuLP |
| Feature | Model input: live load, tariff, or solar |
| Target/label | Battery energy value the model learns to predict |
| Leakage | Evaluation information improperly reaches training |
| Pseudo-label | A model-generated value reused as if it were truth |
| Drift | Production input/relationship distribution changes over time |
| Champion/challenger | Current model compared with a candidate on the same data |
| Backtest | Historical simulation that respects time ordering |
| MPC | Receding-horizon optimization that observes and re-solves repeatedly |
| Cost regret | Decision cost gap versus perfect future information |
| Idempotency | Retrying the same request does not duplicate its effect |
| IDOR | Insecure direct object reference; missing ownership authorization |

## 14. Final readiness checklist

Before saying “the model improved”:

- labels are measured outcomes, not prior predictions;
- unit/timezone/site schemas are validated;
- duplicates/groups cannot cross the split;
- champion and challenger use the same chronological holdout;
- naive and physics baselines are reported;
- RMSE, MAE, normalized/horizon/slice metrics are reported;
- physical violation and downstream cost-regret metrics pass thresholds;
- artifact lineage/checksum/schema are recorded;
- promotion/reload is atomic, single-writer, observable, and reversible.

Before saying “the optimizer saves money”:

- tariff/import/export rules are real and location-correct;
- energy equality, efficiency, reserve, rate, and terminal SOC are enforced;
- all four modes have tested mathematical definitions;
- priority/preference changes decisions as intended;
- device schedules are feasible and contiguity is correct;
- optimized and baseline costs use identical realized conditions;
- degradation and legal/export constraints are included where applicable;
- savings are measured over representative periods with uncertainty.

Before saying “the service is production-ready”:

- public data and compute are authenticated/authorized/rate-limited;
- strict request/response contracts and versioning exist;
- migrations, tests, CI, deployment, rollback, retention, and secret handling exist;
- multi-worker scheduling/model state is coordinated;
- liveness/readiness, metrics, traces, alerts, SLOs, and incident procedures exist;
- dependency failures and degraded data provenance are explicit to callers.

## 15. Source index

- Runtime orchestration and request schema: `app.py:56-114,146-503`
- Live weather and fallback: `services/weather_service.py:12-126`
- Live solar: `services/solar_service.py:3-33`
- Training-data formulas: `services/data_collection_service.py:20-266`
- Model lifecycle/inference/metrics: `services/model_manager.py:25-355`
- MILP formulation: `services/optimizer_service.py:10-191`
- ORM lineage/schema: `db_models.py:9-175`
- Database lifecycle: `database.py:9-46`
- Admin authentication: `services/auth_service.py:16-102`
- Request IDs/logging: `middleware.py:26-60`, `logger_setup.py:20-155`
- Historical metric evidence: `logs/illuminate_2026-06-26.log`
- Legacy data/trainer: `data/consumption_battery_data.csv`, `train_model.py`
- Full system/integration context: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)

When code changes, update this guide alongside the implementation and its tests.
For interviews, remember the strongest story is not “we used ML”; it is “we traced
the decision system end to end, measured it honestly, preserved physical invariants,
and designed a safe path from prototype to production.”
