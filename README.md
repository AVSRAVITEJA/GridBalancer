
# GRIDBALANCER
## Hybrid Microgrid Simulation with Frequency Regulation

A research-grade hybrid microgrid simulation combining renewable generation (solar + wind), thermal generator, and battery storage for grid frequency regulation. Achieves 95.7% stability with mean frequency of 50.012 Hz.

---

## Overview

PVGRIDBALANCER simulates a hybrid microgrid with:
- **Renewable Energy:** Solar + Wind generation (2,208 hourly records)
- **Thermal Generator:** Slow base load following (5000 kW, 50 kW/min ramp)
- **Battery Storage:** Fast frequency regulation (10,000 kWh, 7000 kW)
- **Hierarchical Control:** VSM + AGC + SOC correction for 50 Hz frequency stability

### Performance Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Mean Frequency | 50.0 Hz | 50.012 Hz | 0.012 Hz error |
| Stability Rate | >95% | 95.7% | Achieved |
| Battery SOC | 50% | 52.5% | Optimal |
| Mean ACE | <300 kW | 3.94 kW | 76× better |

---

## Project Structure

```
PVGRIDBALANCER/
├── scripts/
│   ├── hybrid_microgrid.py            # Main simulation (450 lines)
│   ├── data_cleaning.py               # Data preprocessing
│   ├── validate_and_align.py          # Data validation
│   ├── merged_datasets.py             # Dataset synchronization
│   ├── feature_engineering_data.py    # Physics-based features
│   ├── pv_forecasting_model.py        # Solar TCN model (optional)
│   ├── wind_forecasting_model.py      # Wind TCN model (optional)
│   └── hybrid_forecasting.py          # Dual-stream forecasting (optional)
├── data/
│   ├── preprocessed/                  # Processed CSV files
│   │   └── engineered_features.csv    # Input data (2,208 records)
│   └── raw/                           # Original Excel datasets (8 files)
├── models/                            # Trained TCN models (optional)
├── results/
│   ├── hybrid_microgrid_results.csv   # Latest simulation results
│   └── hybrid_microgrid_metrics.json  # Performance metrics
├── docs/
│   └── HYBRID_MICROGRID_DOCUMENTATION.md  # Technical documentation
├── config/
│   └── grid_codes.yaml                # Grid configuration
├── run_preprocessing.py               # Data pipeline orchestrator
├── QUICK_START.md                     # Quick start guide
├── PROJECT_STATUS.md                  # Original design (RA-AVSM)
├── project_reference.md               # Theoretical background
└── README.md                          # This file
```
=======
# GridBalancer

PVgridBalancer is a modular renewable energy forecasting framework designed to provide accurate solar (PV) and wind power predictions, along with a structured foundation for future grid balancing intelligence.

The system is built using production-safe machine learning pipelines with time-series validation and robustness testing to ensure reliability under real-world conditions.

---

## Project Overview

The primary objectives of this repository are:

- Develop accurate photovoltaic (PV) power forecasting models  
- Develop accurate wind power forecasting models  
- Perform stress testing under noisy conditions  
- Provide a scalable foundation for future grid balancing intelligence  

This repository currently focuses on the forecasting layer. The grid balancing engine will be implemented in future phases.

---

## Repository Structure

PVgridBalancer/
│
├── data/preprocessed/
│ ├── pv_dataset.csv
│ └── wind_dataset.csv
│
├── models/
│ ├── pv_forecast_v3.pkl
│ └── wind_forecast_v3.pkl
│
├── scripts/
│ ├── pv_forecasting_model.py
│ ├── wind_forecasting_model.py
│ ├── model_stress_test.py
│ └── (planned) grid_balancing_engine.py
│
└── README.md


---

## Implemented Components

### 1. PV Forecasting Model

- Time-series aware training  
- TimeSeries cross-validation  
- Production-safe feature handling  
- Serialized model output  

Model file:
models/pv_forecast_v3.pkl

Performance (cross-validated):

- R² ≈ 0.97  
- RMSE ≈ 188 kW  
- MAE ≈ 90 kW  

---

### 2. Wind Forecasting Model

- TimeSeries cross-validation  
- High predictive stability  
- Production-ready training pipeline  

Model file:
models/wind_forecast_v3.pkl

Performance (cross-validated):

- R² ≈ 0.9996  
- RMSE ≈ 13 kW  
- MAE ≈ 2 kW  

---

### 3. Stress Testing Framework

The stress testing module evaluates:

- Baseline forecasting performance  
- Model robustness under 5% feature noise  
- Hybrid PV + Wind performance  

This ensures resilience under sensor noise and environmental uncertainty.

Run:
python scripts/model_stress_test.py


---

## Installation

### Requirements
```bash
# Core dependencies
pip install pandas numpy scipy openpyxl

# For forecasting models (optional)
pip install tensorflow scikit-learn

# For visualization (optional)
pip install matplotlib seaborn
```

### Python Version
- Python 3.8 or higher

---

## Quick Start

### Run the Simulation

```bash
py scripts/hybrid_microgrid.py
```

### Expected Output

```
 Frequency Performance:
  Mean: 50.012 Hz (Target: 50.0 Hz)
  Stability: 95.7% (Target: >95%)

 Battery Performance:
  Mean SOC: 52.5% (Target: 50%)

 Grid Performance:
  Mean |ACE|: 3.94 kW (Target: <300 kW)

TARGETS ACHIEVED: 4/4
 SYSTEM OPERATIONAL
```

### Data Preprocessing (Optional)

If you want to reprocess the raw data:

```bash
py run_preprocessing.py
```

This generates `data/preprocessed/engineered_features.csv` with 50 physics-based features.

### Train Forecasting Models (Optional)

The simulation works without forecasting models, but you can train TCN models:

```bash
py scripts/hybrid_forecasting.py
```

Requires TensorFlow. Generates PV and Wind forecasting models.

---

## Technical Details

### System Architecture

**3-Layer Hierarchical Control:**

1. **Layer 1: VSM (Fast - Seconds Scale)**
   - Virtual Inertia: k_inertia = 4000 kW/(Hz/s)
   - Droop Control: k_droop = 6000 kW/Hz
   - Responds to frequency deviations and RoCoF

2. **Layer 2: AGC (Medium - Minutes Scale)**
   - Integral gain: k_i = 5.0
   - Back-calculation anti-windup: k_aw = 0.5
   - Restores frequency to 50 Hz

3. **Layer 3: SOC Correction (Slow - Hours Scale)**
   - Proportional gain: k_soc = 1000
   - Target SOC: 50%
   - Prevents battery saturation

### Per-Unit Swing Equation

```
df/dt = (f_nom / 2H) × (ΔP_pu - D_pu × Δf)
```

Where:
- f_nom = 50 Hz (nominal frequency)
- H = 10 seconds (inertia constant)
- D_pu = D / S_base (per-unit damping)
- ΔP_pu = (P_gen - P_load) / S_base

### Battery Control

```python
P_battery = P_vsm + P_agc + P_soc
```

**Specifications:**
- Capacity: 10,000 kWh
- Max Power: 7,000 kW
- Efficiency: 95%
- SOC Range: 10-90%

### Thermal Generator

```python
P_thermal_setpoint = max(0, P_load - MA(P_renewable))
```

**Specifications:**
- Max Power: 5,000 kW
- Ramp Rate: 50 kW/min
- Dispatch: Follows moving average (12 timesteps)

---

## Results & Performance

### Simulation Results

- **Simulation Length**: 2,208 timesteps (73.6 hours at 2-minute intervals)
- **Mean Frequency**: 50.012 Hz (0.012 Hz error)
- **Frequency Std Dev**: 0.253 Hz
- **Stability Rate**: 95.7% (within 49.5-50.5 Hz)
- **Battery SOC**: 52.5% mean, 8.1% std dev
- **Mean ACE**: 3.94 kW (76× better than 300 kW target)
- **Max RoCoF**: 0.0014 Hz/s (357× safety margin vs 0.5 Hz/s limit)

### Battery Sizing Validation

```
Net imbalance std: 1468.0 kW
Required power (2×std): 2936.1 kW
Actual power: 7000.0 kW
Margin: 2.4× 
```

### Control Performance

- **VSM Contribution**: 1184.1 kW average (fast response)
- **AGC Contribution**: 3.9 kW average (frequency restoration)
- **SOC Contribution**: 73.4 kW average (battery centering)
- **Thermal Utilization**: 28.5% (1422.7 kW average)

---

## Key Features

### 1. Physics-Based Control
- Per-unit swing equation for frequency dynamics
- Virtual Synchronous Machine (VSM) principles
- Proper inertia and damping modeling
- No arbitrary scaling factors

### 2. Hierarchical Architecture
- Fast VSM control (seconds scale)
- Medium AGC control (minutes scale)
- Slow SOC correction (hours scale)
- Back-calculation anti-windup

### 3. Clean Implementation
- Single production file (450 lines)
- Modular class design
- Comprehensive logging
- Reproducible results (fixed seed)

### 4. Validated Performance
- All 4 targets achieved
- Battery sizing validated
- Stability constraints respected
- Production-ready code

---

## Configuration

### System Parameters

Edit in `scripts/hybrid_microgrid.py`:

```python
# Battery
self.battery = BatteryEnergyStorageSystem(
    capacity_kwh=10000,
    max_power_kw=7000,
    efficiency=0.95,
    initial_soc=0.5
)

# Thermal Generator
self.thermal = ThermalGenerator(
    max_power_kw=5000,
    ramp_rate_kw_per_min=50
)

# Grid Physics
self.grid = GridSimulator(
    base_frequency=50.0,
    inertia_constant=10.0,
    damping_coefficient=6.0
)

# Control Gains
self.k_inertia = 4000  # kW/(Hz/s)
self.k_droop = 6000    # kW/Hz
self.k_i_agc = 5.0
self.k_soc = 1000
```

---

## Customization

### Change Timestep

```python
microgrid = HybridMicrogrid(verbose=True)
results_df, metrics = microgrid.run(dt_hours=1/60)  # 1-minute timestep
```

### Adjust Control Gains

Edit `scripts/hybrid_microgrid.py`:
```python
self.k_inertia = 4000  # Increase for stronger inertia response
self.k_droop = 6000    # Increase for stronger droop control
self.k_i_agc = 5.0     # Increase for faster frequency restoration
self.k_soc = 1000      # Increase for stronger SOC centering
```

### Modify Load Profile

Edit in `generate_load_profile()`:
```python
base_load = avg_renewable * 1.70  # Adjust multiplier
```

---

## Future Enhancements

1. **Adaptive Gain Scheduling**: Regime-aware control (original RA-AVSM concept)
2. **Model Predictive Control**: Forecast-based optimization
3. **Economic Dispatch**: Cost optimization with market prices
4. **Real-Time Visualization**: Live dashboard for monitoring
5. **Extended Simulation**: Multi-week or seasonal analysis
6. **Hardware-in-the-Loop**: Integration with real BESS
7. **Multiple Batteries**: Distributed storage systems

---

## Documentation

- **QUICK_START.md** - Get started in 30 seconds
- **docs/HYBRID_MICROGRID_DOCUMENTATION.md** - Complete technical reference
- **PROJECT_STATUS.md** - Original RA-AVSM design specification
- **project_reference.md** - Theoretical background and requirements

## References

### Technical Standards
- IEEE 1547: Interconnection of Distributed Energy Resources
- IEC 61850: Power utility automation
- NERC BAL-003: Frequency Response and Frequency Bias Setting

### Power Systems Theory
- Kundur, P. "Power System Stability and Control"
- Wood, A. & Wollenberg, B. "Power Generation, Operation, and Control"
- Zhong, Q. et al. "Synchronverters: Inverters That Mimic Synchronous Generators"

---

## License

This project is for educational and research purposes.

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

This project successfully implements:
- Physics-based grid frequency dynamics
- Hierarchical control architecture
- Virtual Synchronous Machine (VSM) principles
- Research-grade performance validation

Following industry best practices and academic standards for power systems engineering.

---

**Status:**  Production Ready  
**Version:** 2.0  
**Achievement:** All 4 targets met (100%)  
**Last Updated:** February 18, 2026
