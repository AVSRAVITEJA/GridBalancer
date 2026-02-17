````markdown
# PVGRIDBALANCER
# Regime-Aware Adaptive Virtual Synchronous Machine (RA-AVSM)
## Complete System Architecture & Implementation Blueprint

**Version:** 1.0  
**Status:** Design Specification  
**Purpose:** Full redesign of grid balancing system using adaptive, regime-aware synthetic inertia and predictive control.

---

# 1. Project Vision

PVGRIDBALANCER aims to simulate and control a renewable-dominant microgrid using:

- Solar and Wind forecasting (TCN models)
- Battery Energy Storage System (BESS)
- Physics-based grid frequency dynamics
- Adaptive control informed by renewable regimes

The objective is not merely frequency stabilization.

The objective is to create:

> A Regime-Aware Adaptive Virtual Synchronous Machine (RA-AVSM)

This transforms the battery from a passive balancing device into a context-aware synthetic generator.

---

# 2. Core Concept

Traditional Virtual Synchronous Machine (VSM):
- Fixed inertia
- Fixed droop
- Purely reactive control

RA-AVSM:
- Detect renewable operating regime
- Adapt control gains dynamically
- Weight control aggressiveness by volatility & complementarity
- Integrate forecast-based pre-positioning
- Protect battery health
- Reduce curtailment
- Maintain stable 50 Hz frequency

---

# 3. System Architecture Overview

The system is divided into 4 logical layers:

Layer 1: Data & Forecasting  
Layer 2: Renewable Regime Detection  
Layer 3: Adaptive VSM Controller  
Layer 4: Battery Dispatch & Frequency Update  

---

# 4. Layer 1 – Data & Forecasting

## 4.1 Available Dataset Features

Weather & Environment:
- Temperature
- Humidity
- Ground Radiation Intensity
- Upper Atmosphere Radiation Intensity
- Wind Speed
- Air Density

Renewable Physics:
- clearness_index_kt
- wind_power_coefficient_cp
- complementarity_index
- renewable_capacity_factor
- solar_efficiency
- wind_power_per_speed

Dynamics & Volatility:
- solar_rate_of_change
- wind_rate_of_change
- total_rate_of_change
- solar_volatility
- wind_volatility
- total_volatility
- rolling_mean_6h
- rolling_std_6h
- lag features (1h, 2h, 3h, 6h, 12h, 24h)

Temporal:
- hour_sin, hour_cos
- day_sin, day_cos
- month, week_of_year
- day_of_week, day_of_year

## 4.2 Forecasting

Use trained TCN models to produce:

- PV forecast (T+1)
- Wind forecast (T+1)
- Optional forecast uncertainty (rolling prediction error std)

Outputs:
- forecast_renewable
- forecast_uncertainty

---

# 5. Layer 2 – Renewable Regime Detection

Purpose:
Classify renewable operating condition to adapt control gains.

## 5.1 Regime Indicators

Use normalized values of:
- total_volatility
- total_rate_of_change
- complementarity_index
- rolling_std_6h
- forecast_uncertainty

## 5.2 Regime Definitions

Define five operating regimes:

1. STABLE  
   Low volatility, low ramp rate.

2. RAMPING  
   Moderate volatility, significant ramp.

3. TURBULENT  
   High volatility and high ramp rate.

4. COMPLEMENTARY  
   High complementarity_index (> threshold).

5. EXTREME  
   Very high volatility combined with high forecast uncertainty.

## 5.3 Example Rule-Based Detection

```python
if total_volatility < v1:
    regime = "STABLE"

elif complementarity_index > 0.7:
    regime = "COMPLEMENTARY"

elif total_volatility > v3 and forecast_uncertainty > u1:
    regime = "EXTREME"

elif total_volatility > v2:
    regime = "TURBULENT"

else:
    regime = "RAMPING"
````

Thresholds determined from dataset percentiles.

---

# 6. Layer 3 – Adaptive Virtual Synchronous Machine (RA-AVSM Core)

---

# 6.1 Per-Unit System Implementation

Define system base power:

```python
S_base = mean(total_renewable_power)
```

Convert power to per-unit:

```python
P_pu = P_kW / S_base
```

Define nominal frequency:

```python
f_nom = 50.0
```

Define inertia constant (seconds):

```python
H = 5.0
```

Define damping coefficient:

```python
D = 1.0
```

---

# 6.2 Swing Equation (Per-Unit)

Continuous form:

df/dt = (f_nom / (2H)) * (P_m - P_e - D * Δf)

Discrete form:

```python
df_dt = (f_nom / (2 * H)) * (P_m - P_e - D * delta_f)
frequency = frequency + df_dt * dt
```

No artificial scaling factors allowed.

---

# 6.3 Adaptive Gain Scheduling

Define base gains:

```python
k_inertia_base = K1
k_droop_base   = K2
```

Normalize volatility:

```python
vol_norm = total_volatility / max_volatility
```

Adaptive gains:

```python
k_inertia = k_inertia_base * (1 + alpha1 * vol_norm)
k_droop   = k_droop_base   * (1 + alpha2 * vol_norm)
```

Complementarity modulation:

```python
k_inertia *= (1 - complementarity_index)
k_droop   *= (1 - complementarity_index)
```

High complementarity reduces control intervention.

---

# 6.4 Virtual Inertia

```python
P_inertia = -k_inertia * (df_dt)
```

---

# 6.5 Droop Control

```python
P_droop = -k_droop * (frequency - f_nom)
```

---

# 6.6 AGC (Secondary Control)

Area Control Error:

```python
ACE = beta * (frequency - f_nom)
```

Integral control with anti-windup:

```python
if not battery_saturated:
    ace_integral += ACE * dt

P_agc = -k_i * ace_integral
```

---

# 6.7 Forecast-Aware Predictive Control

```python
expected_imbalance = forecast_renewable - forecast_load
P_predictive = -k_forecast * expected_imbalance
P_predictive *= (1 - forecast_uncertainty_norm)
```

---

# 6.8 SOC-Aware Modulation

Define:

```python
SOC_target = 0.5
SOC_deviation = SOC - SOC_target
P_soc = -k_soc * SOC_deviation
```

If SOC > 0.85:
reduce charging gains

If SOC < 0.15:
reduce discharging gains

---

# 7. Layer 4 – Battery Dispatch

Total command:

```python
P_battery = (
    P_inertia
  + P_droop
  + P_agc
  + P_predictive
  + P_soc
)
```

Apply limits:

```python
P_battery = clip(P_battery, -P_max, P_max)
```

Curtailment only if:

```python
if renewable_excess > P_max:
    curtailment = renewable_excess - P_max
```

---

# 8. Volatility-Weighted Objective Function

Define cost:

J =
(Δf²)

* λ1 * total_volatility * (Δf²)
* λ2 * battery_degradation_cost
* λ3 * curtailment_cost

This makes control more conservative during volatile periods.

---

# 9. Frequency Risk Index (Optional Advanced Feature)

Define:

FRI =
w1 * vol_norm

* w2 * forecast_uncertainty_norm
* w3 * (1 - complementarity_index)
* w4 * SOC_risk

If FRI > threshold:

* Increase inertia gain
* Increase droop gain
* Pre-charge battery

This creates proactive stabilization.

---

# 10. Implementation Roadmap

Phase 1:

* Implement proper per-unit swing equation

Phase 2:

* Add regime detection

Phase 3:

* Implement adaptive gain scheduling

Phase 4:

* Integrate forecast-weighted predictive control

Phase 5:

* Tune parameters statistically

---

# 11. Expected Improvements

Compared to fixed VSM:

* Improved frequency stability
* Reduced ACE
* Reduced curtailment
* Reduced battery stress
* Increased novelty and publishability

---

# 12. Project Identity

PVGRIDBALANCER becomes:

A Regime-Aware Adaptive Virtual Synchronous Machine
for Renewable-Dominant Microgrids.

This integrates:

Physics
Machine Learning
Volatility Structure
Battery Health Awareness

Into one unified adaptive grid balancing framework.

---

# End of RA-AVSM Design Specification

```
```
