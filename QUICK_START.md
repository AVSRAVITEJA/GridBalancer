# PVGRIDBALANCER - Quick Start Guide

**Status:** ✅ Production Ready | **Version:** 1.0 | **Date:** February 18, 2026

---

## 🎯 What This Project Does

Simulates a hybrid microgrid with:
- **Renewable Energy:** Solar + Wind generation
- **Thermal Generator:** Slow base load following (5000 kW)
- **Battery Storage:** Fast frequency regulation (10,000 kWh, 7000 kW)
- **Grid Control:** Maintains 50 Hz frequency with 95.7% stability

---

## ⚡ Quick Start (30 seconds)

### Run the Simulation

```bash
py scripts/hybrid_microgrid.py
```

### Expected Output

```
✅ SYSTEM OPERATIONAL
Frequency: 50.012 Hz
Stability: 95.7%
Battery SOC: 52.5%
ACE: 3.94 kW
```

---

## 📊 Performance Summary

| Metric | Result | Status |
|--------|--------|--------|
| Frequency | 50.012 Hz | ✅ Perfect |
| Stability | 95.7% | ✅ Achieved |
| Battery SOC | 52.5% | ✅ Optimal |
| ACE | 3.94 kW | ✅ Excellent |

**All 4 targets achieved!**

---

## 📁 Project Structure

```
PVGRIDBALANCER/
├── scripts/
│   └── hybrid_microgrid.py          # Main simulation (run this!)
├── data/
│   └── preprocessed/
│       └── engineered_features.csv  # Input data (2,208 records)
├── results/
│   ├── hybrid_microgrid_results.csv # Latest results
│   └── hybrid_microgrid_metrics.json# Performance metrics
├── docs/
│   └── HYBRID_MICROGRID_DOCUMENTATION.md  # Technical docs
├── models/                           # Trained TCN models (optional)
├── README.md                         # Full project overview
├── FINAL_PROJECT_STATUS.md          # Current status
└── QUICK_START.md                   # This file
```

---

## 🔧 System Configuration

### Control Parameters (Already Tuned)

```python
k_inertia = 4000  # Virtual inertia
k_droop = 6000    # Droop control
k_i_agc = 5.0     # AGC integral
k_soc = 1000      # SOC correction
dt = 2 minutes    # Timestep
```

### Battery Specifications

```python
Capacity: 10,000 kWh
Max Power: 7,000 kW
Efficiency: 95%
SOC Range: 10-90%
```

### Thermal Generator

```python
Max Power: 5,000 kW
Ramp Rate: 50 kW/min
Dispatch: Follows moving average
```

---

## 📖 Documentation

### For Quick Reference
- **QUICK_START.md** (this file) - Get started in 30 seconds
- **FINAL_PROJECT_STATUS.md** - Current status and results

### For Technical Details
- **docs/HYBRID_MICROGRID_DOCUMENTATION.md** - Complete technical reference
- **PROJECT_COMPLETION_SUMMARY.md** - Full project summary

### For Understanding
- **README.md** - Project overview and architecture
- **PROJECT_STATUS.md** - Original design (RA-AVSM concept)
- **project_reference.md** - Theoretical background

---

## 🎓 How It Works

### 3-Layer Control Hierarchy

**Layer 1: VSM (Fast - Seconds)**
- Virtual inertia responds to frequency changes
- Droop control provides proportional response

**Layer 2: AGC (Medium - Minutes)**
- Integral control restores frequency to 50 Hz
- Back-calculation anti-windup prevents saturation

**Layer 3: SOC (Slow - Hours)**
- Proportional control centers battery at 50% SOC
- Prevents battery saturation

### Power Flow

```
Renewable (Solar + Wind)
    ↓
Thermal Generator (slow dispatch)
    ↓
Battery (fast regulation)
    ↓
Grid (50 Hz frequency)
    ↓
Load (variable demand)
```

---

## 🔬 Technical Highlights

### Per-Unit Swing Equation
```
df/dt = (f_nom / 2H) × (ΔP_pu - D_pu × Δf)
```

### Battery Control
```
P_battery = P_vsm + P_agc + P_soc
```

### Thermal Dispatch
```
P_thermal = max(0, Load - MA(Renewable))
```

---

## 📈 Results Breakdown

### Frequency Performance
- Mean: 50.012 Hz (0.012 Hz error)
- Std Dev: 0.253 Hz
- Range: 49.33-51.00 Hz
- Stability: 95.7% within 49.5-50.5 Hz

### Battery Performance
- Mean SOC: 52.5% (target: 50%)
- SOC Std Dev: 8.1%
- Mean Power: 1189 kW
- Max Power: 5936 kW (85% of capacity)

### Grid Performance
- Mean ACE: 3.94 kW (76× better than target)
- Max RoCoF: 0.0014 Hz/s (357× safety margin)
- Thermal Utilization: 28.5%

---

## 🛠️ Customization

### Change Battery Size

Edit `scripts/hybrid_microgrid.py`:
```python
self.battery = BatteryEnergyStorageSystem(
    capacity_kwh=10000,  # Change this
    max_power_kw=7000,   # Change this
    efficiency=0.95,
    initial_soc=0.5
)
```

### Change Control Gains

Edit `scripts/hybrid_microgrid.py`:
```python
self.k_inertia = 4000  # Virtual inertia
self.k_droop = 6000    # Droop control
self.k_i_agc = 5.0     # AGC integral
self.k_soc = 1000      # SOC correction
```

### Change Timestep

Run with custom timestep:
```python
microgrid = HybridMicrogrid(verbose=True)
results_df, metrics = microgrid.run(dt_hours=1/60)  # 1-minute
```

---

## 🎯 Key Achievements

✅ **Mean Frequency:** 50.012 Hz (0.012 Hz error)  
✅ **Stability:** 95.7% (target >95%)  
✅ **Battery SOC:** 52.5% (target 50%)  
✅ **ACE:** 3.94 kW (target <300 kW)  
✅ **Clean Code:** 450 lines, production-ready  
✅ **Validated:** Battery sizing, performance metrics  

---

## 🚀 Next Steps (Optional)

### Run Extended Simulation
```python
# Modify to run longer
microgrid.run(dt_hours=1/30)  # Current: 73.6 hours
```

### Analyze Results
```python
import pandas as pd
results = pd.read_csv('results/hybrid_microgrid_results.csv')
print(results.describe())
```

### Visualize Performance
```python
import matplotlib.pyplot as plt
results['frequency'].plot()
plt.axhline(50, color='r', linestyle='--')
plt.show()
```

---

## 📞 Support

### Documentation
- Technical: `docs/HYBRID_MICROGRID_DOCUMENTATION.md`
- Status: `FINAL_PROJECT_STATUS.md`
- Summary: `PROJECT_COMPLETION_SUMMARY.md`

### Results
- Metrics: `results/hybrid_microgrid_metrics.json`
- Data: `results/hybrid_microgrid_results.csv`

---

## ✨ Summary

The PVGRIDBALANCER is a **complete, production-ready** hybrid microgrid simulation that achieves:

- ✅ Excellent frequency regulation
- ✅ High stability
- ✅ Optimal battery management
- ✅ Clean, documented code

**Ready to run. Ready to extend. Ready to deploy.**

---

**Last Updated:** February 18, 2026  
**Status:** Production Ready  
**Achievement:** 100% (All targets met)
