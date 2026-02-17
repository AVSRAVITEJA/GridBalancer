# Hybrid Microgrid Simulation - Technical Documentation

**Version:** 1.0  
**Date:** February 18, 2026  
**Status:** Production Ready

---

## Overview

Research-grade hybrid microgrid simulation combining renewable generation (solar + wind), thermal generator, and battery storage for grid frequency regulation.

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   HYBRID MICROGRID                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Renewable (Solar + Wind)  →  Variable Generation       │
│  Thermal Generator         →  Slow Base Load Following  │
│  Battery Storage          →  Fast Frequency Regulation  │
│                                                          │
│  Total Generation = Renewable + Thermal + Battery       │
│  Grid Frequency regulated via per-unit swing equation   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Thermal Generator
- **Purpose:** Slow dispatch for base load following
- **Capacity:** 5000 kW
- **Ramp Rate:** 50 kW/min
- **Control:** Follows moving average of (Load - Renewable)
- **Window:** 12 timesteps (24 minutes)

### 2. Battery Energy Storage System (BESS)
- **Purpose:** Fast frequency regulation
- **Capacity:** 10,000 kWh
- **Power:** 7,000 kW
- **Efficiency:** 95%
- **SOC Range:** 10-90%
- **Sizing:** ≥ 2× std deviation of net imbalance

### 3. Grid Simulator
- **Physics:** Per-unit swing equation
- **Base Power:** 2000 kW
- **Inertia (H):** 10 seconds
- **Damping (D):** 6 kW/Hz
- **Frequency Limits:** 49.5-50.5 Hz (stability band)

### 4. Hierarchical Controller

**Layer 1: VSM (Fast - Seconds Scale)**
- Virtual Inertia: k_inertia = 4000 kW/(Hz/s)
- Droop Control: k_droop = 6000 kW/Hz
- Responds to frequency deviations and RoCoF

**Layer 2: AGC (Medium - Minutes Scale)**
- Integral gain: k_i = 5.0
- Back-calculation anti-windup: k_aw = 0.5
- Restores frequency to 50 Hz

**Layer 3: SOC Correction (Slow - Hours Scale)**
- Proportional gain: k_soc = 1000
- Target SOC: 50%
- Prevents battery saturation

---

## Performance Metrics

### Achieved Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Mean Frequency** | **50.012 Hz** | 50.0 Hz | ✅ **0.012 Hz error** |
| **Stability Rate** | **95.7%** | >95% | ✅ **Achieved** |
| **Battery SOC** | **52.5%** | 50% | ✅ **2.5% error** |
| **Mean ACE** | **3.94 kW** | <300 kW | ✅ **76x better** |
| **Max RoCoF** | **0.0014 Hz/s** | <0.5 Hz/s | ✅ **357x margin** |

**Targets Achieved: 4/4 ✅**

### System Utilization

- **Thermal Generator:** 28.5% average utilization (1422.7 kW)
- **Battery Power:** 1189.5 kW average, 5935.7 kW peak
- **Battery Sizing:** 7000 kW (2.3× required minimum)

---

## Configuration

### Load Profile
```python
base_load = avg_renewable × 1.70  # 70% higher than renewable average
daily_pattern = 1.0 + 0.25 × sin(2π(t-6)/24)
noise = N(0, 40 kW)
```

### Control Gains
```python
# VSM
k_inertia = 4000  # kW/(Hz/s)
k_droop = 6000    # kW/Hz

# AGC
k_i_agc = 5.0
k_aw = 0.5

# SOC Correction
k_soc = 1000
soc_target = 0.5
```

### Simulation Parameters
```python
dt_hours = 1/30  # 2-minute timestep
n_samples = 2208  # 73.6 hours total
random_seed = 42  # Reproducibility
```

---

## Usage

### Basic Simulation

```python
from scripts.hybrid_microgrid import HybridMicrogrid

# Initialize
microgrid = HybridMicrogrid(verbose=True)

# Run simulation
results_df, metrics = microgrid.run(dt_hours=1/30)

# Results saved to:
# - results/hybrid_microgrid_results.csv
# - results/hybrid_microgrid_metrics.json
```

### Custom Configuration

```python
# Initialize with custom paths
microgrid = HybridMicrogrid(
    data_path='data/preprocessed',
    results_path='results',
    verbose=True
)

# Access components
battery = microgrid.battery
thermal = microgrid.thermal
grid = microgrid.grid
controller = microgrid.controller

# Run with custom timestep
results_df, metrics = microgrid.run(dt_hours=1/60)  # 1-minute timestep
```

---

## File Structure

```
gridBalancer/
├── scripts/
│   └── hybrid_microgrid.py          # Main simulation (450 lines)
├── data/
│   ├── preprocessed/
│   │   └── engineered_features.csv  # Input data
│   └── raw/                          # Original datasets
├── models/
│   ├── pv_tcn_model.keras           # PV forecasting model
│   ├── wind_tcn_model.keras         # Wind forecasting model
│   ├── pv_scalers.pkl               # PV scalers
│   └── wind_scalers.pkl             # Wind scalers
├── results/
│   ├── hybrid_microgrid_results.csv # Simulation results
│   └── hybrid_microgrid_metrics.json# Performance metrics
├── docs/
│   └── HYBRID_MICROGRID_DOCUMENTATION.md  # This file
├── PROJECT_STATUS.md                 # Original requirements
├── project_reference.md              # Theoretical background
├── MODEL_TRAINING_SUMMARY.md         # TCN model documentation
├── FINAL_RESULTS.md                  # Historical results
└── README.md                         # Project overview
```

---

## Technical Details

### Per-Unit Swing Equation

```
df/dt = (f_nom / 2H) × (ΔP_pu - D_pu × Δf)

where:
  f_nom = 50 Hz (nominal frequency)
  H = 10 seconds (inertia constant)
  D_pu = D / S_base (per-unit damping)
  ΔP_pu = (P_gen - P_load) / S_base
```

### Battery Sizing Criterion

```
P_battery_required = 2 × σ(P_renewable - P_load)

Actual: 7000 kW
Required: 3101 kW
Margin: 2.3×
```

### Thermal Dispatch Logic

```
MA_renewable = moving_average(P_renewable, window=12)
P_thermal_setpoint = max(0, P_load - MA_renewable)
P_thermal_actual = ramp_limited(P_thermal_setpoint, 50 kW/min)
```

---

## Validation

### Frequency Regulation
- ✅ Mean frequency within 0.012 Hz of target
- ✅ 95.7% of time within stability band (49.5-50.5 Hz)
- ✅ Frequency std dev: 0.253 Hz (low oscillations)

### Battery Performance
- ✅ SOC centered at 52.5% (target: 50%)
- ✅ SOC std dev: 8.1% (reasonable variation)
- ✅ No SOC violations (stayed within 10-90%)

### Grid Stability
- ✅ ACE: 3.94 kW (excellent regulation)
- ✅ RoCoF: 0.0014 Hz/s (well below limit)
- ✅ No frequency violations outside 40-60 Hz hard limits

---

## References

1. **Swing Equation:** Kundur, P. "Power System Stability and Control"
2. **Virtual Synchronous Machine:** Zhong, Q. et al. "Synchronverters"
3. **AGC Control:** Wood, A. & Wollenberg, B. "Power Generation, Operation, and Control"
4. **Battery Sizing:** EPRI "Energy Storage Handbook"

---

## Changelog

### Version 1.0 (February 18, 2026)
- ✅ Initial production release
- ✅ All 4 performance targets achieved
- ✅ Clean minimal architecture (450 lines)
- ✅ Comprehensive documentation
- ✅ Validated battery sizing
- ✅ Reproducible results (seed=42)

---

## License

Research and educational use only.

---

## Contact

For questions or issues, refer to PROJECT_STATUS.md for original requirements and theoretical background.
