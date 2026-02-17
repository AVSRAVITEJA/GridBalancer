```
project_reference.md
```

This document defines **ALL required theoretical corrections** to make the model achieve:

* Mean frequency: 49.95–50.05 Hz
* Frequency violations: < 2%
* Stability rate: > 98%
* Mean |ACE| < 300 kW

---

# ============================================================

# GRIDBALANCER – PHASE III COMPLETE THEORETICAL REDESIGN

# ============================================================

# 1. FUNDAMENTAL PROBLEM DIAGNOSIS

The model currently exhibits:

* Large frequency oscillations (45–55 Hz)
* High variance despite correct mean
* Battery power saturation
* Unrealistic 1-hour frequency timestep
* No renewable curtailment
* Control loop instability (discrete-time instability)

The root cause is NOT insufficient gain.

The root cause is:

1. Incorrect time-scale modeling
2. Discrete-time instability
3. Actuator saturation without coordination
4. No renewable curtailment logic
5. No volatility filtering
6. Improper eigenvalue placement

This document defines the required theoretical corrections.

---

# ============================================================

# 2. MULTI-TIME-SCALE CONTROL STRUCTURE (MANDATORY)

# ============================================================

The model must separate:

Outer Loop (Energy Layer)
Time step = 1 hour

Inner Loop (Frequency Layer)
Time step = 1 minute (dt = 1/60 hour)

Frequency dynamics must NEVER use 1-hour timestep.

---

## REQUIRED STRUCTURE

For each 1-hour timestep:

```
For minute = 1 to 60:
    1. Compute ΔP
    2. Apply inertia control
    3. Apply droop control
    4. Apply AGC integral
    5. Apply saturation limits
    6. Update frequency
End
```

Then update SOC once per hour.

---

# ============================================================

# 3. CORRECT FREQUENCY DYNAMICS MODEL

# ============================================================

Replace artificial scaling factor with physical swing equation:

df/dt = (P_gen - P_load - DΔf) / M_eq

Discrete form (Euler):

f[k+1] = f[k] + (dt / M_eq) * (ΔP - (D + k_droop) * Δf)

Where:

dt = 1/60 hour
M_eq = M_grid + M_virtual
ΔP = P_gen + P_battery - P_load

---

## Stability Condition

For discrete stability:

0 < (dt / M_eq) * (D + k_droop) < 2

Choose parameters to satisfy this.

---

# ============================================================

# 4. VIRTUAL INERTIA IMPLEMENTATION

# ============================================================

Battery must inject power proportional to RoCoF:

P_inertia = -k_inertia * (df/dt)

Recommended:

k_inertia = 10,000 – 15,000 kW / (Hz/s)

This limits:

|df/dt| < 0.5 Hz/s

---

# ============================================================

# 5. PRIMARY DROOP CONTROL

# ============================================================

P_droop = -k_droop * (f - f_nom)

Recommended:

k_droop = 4000 – 6000 kW/Hz

Deadband:

±0.005 Hz (reduce from 0.02 Hz)

---

# ============================================================

# 6. SECONDARY CONTROL (AGC – INTEGRAL ACE)

# ============================================================

ACE = βΔf + (P_gen - P_scheduled)

Integral control:

P_AGC = -k_i ∫ ACE dt

With anti-windup:

Clamp integral term to battery power limits.

Recommended:

k_i = 0.1 – 0.5

---

# ============================================================

# 7. CONTROL PRIORITY STRUCTURE

# ============================================================

REMOVE weighted blending (0.8 / 0.3 / 0.1).

Use strict hierarchy:

P_battery =
P_inertia

* P_droop
* P_AGC
* P_predictive
* P_SOC

Then apply power limits.

Primary control must NEVER be diluted by arbitrage.

---

# ============================================================

# 8. RENEWABLE CURTAILMENT (MANDATORY)

# ============================================================

If renewable excess exceeds battery capability:

If:

P_renewable - P_load > P_battery_max

Then:

P_renewable_effective =
P_load + P_battery_max

Curtailment prevents frequency runaway.

This is physically correct grid behavior.

---

# ============================================================

# 9. VOLATILITY SMOOTHING (LOW-PASS FILTER)

# ============================================================

To simulate turbine inertia:

P_smoothed[t] =
α * P_current +
(1 - α) * P_previous

Recommended:

α = 0.2 – 0.3

This reduces high-frequency swings.

---

# ============================================================

# 10. BATTERY POWER REQUIREMENT CONDITION

# ============================================================

For full stabilization:

P_battery_max ≥ 0.8 × ΔP_max

If ΔP_max ≈ 7000 kW:

Required battery power ≈ 5600 kW

If battery is smaller:
→ Curtailment becomes mandatory.

---

# ============================================================

# 11. SOC PREVENTIVE CONTROL

# ============================================================

Prevent hitting 90% ceiling.

If SOC > 85%:
Reduce charging aggressiveness
Increase curtailment
Increase artificial load

If SOC < 15%:
Limit discharge

Target SOC operating band:

45% – 55%

---

# ============================================================

# 12. EIGENVALUE PLACEMENT (DAMPING DESIGN)

# ============================================================

Linearized model:

dΔf/dt = (1/M_eq)(ΔP - (D + k_droop)Δf)

Time constant:

τ = M_eq / (D + k_droop)

To reduce oscillation:

ζ > 0.7

Design condition:

D + k_droop ≈ 2ζ√(M_eq * K)

Increase damping relative to inertia.

---

# ============================================================

# 13. REMOVE ARTIFICIAL FREQUENCY CEILING LOCKING

# ============================================================

Do NOT hard clamp frequency to 45–55.

Instead:

Allow continuous dynamics.

Only record violations.

Hard clamping causes artificial limit cycling.

---

# ============================================================

# 14. DISCRETE-TIME NUMERICAL STABILITY RULE

# ============================================================

For Euler integration:

dt < 2M_eq / (D + k_droop)

If dt is too large:
System oscillates regardless of gains.

Hence:
Use 1-minute timestep.

---

# ============================================================

# 15. FORECAST-INTEGRATED PREDICTIVE CONTROL (OPTIONAL)

# ============================================================

Instead of reactive:

Minimize over horizon H:

Σ (Δf² + ACE² + SOC deviation²)

H = 6–12 timesteps

Subject to:
Power limits
SOC limits
Forecast uncertainty

---

# ============================================================

# 16. TARGET PERFORMANCE AFTER IMPLEMENTATION

# ============================================================

Expected:

Mean frequency: 49.98 Hz
Std dev: < 0.25 Hz
Violations: < 2%
Mean |ACE|: < 300 kW
Stability: > 98%
RoCoF: < 0.5 Hz/s

---

# ============================================================

# 17. FINAL REQUIRED CHANGES SUMMARY

# ============================================================

1. Implement nested time-scale simulation
2. Replace scaling factor with swing equation
3. Add true virtual inertia
4. Increase droop gain
5. Add integral AGC control
6. Remove weighted blending
7. Add renewable curtailment
8. Add volatility smoothing
9. Prevent SOC saturation
10. Remove frequency hard clamping
11. Tune damping for ζ > 0.7
12. Ensure Euler stability condition satisfied

---

# ============================================================

# 18. FINAL ENGINEERING CONCLUSION

# ============================================================

The model will only achieve >98% stability if:

* It behaves as a Virtual Synchronous Machine
* It respects time-scale physics
* It respects actuator limits
* It includes curtailment
* It is numerically stable

This is no longer a gain-tuning problem.

It is a dynamics modeling correction.

---

END OF THEORETICAL REDESIGN DOCUMENT

---

