"""
Hybrid Microgrid Simulation
Research-grade implementation with thermal generator and battery storage

Architecture:
- Thermal Generator: Slow dispatch, balances moving average
- Battery: Fast frequency regulation (VSM + AGC)
- Grid: Per-unit swing equation physics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple

# Fixed random seed for reproducibility
np.random.seed(42)

logger = logging.getLogger(__name__)


class ThermalGenerator:
    """Slow dispatchable thermal generator for base load following"""
    
    def __init__(self, max_power_kw=5000, ramp_rate_kw_per_min=50):
        self.max_power_kw = max_power_kw
        self.ramp_rate_kw_per_min = ramp_rate_kw_per_min
        self.power_output = 0.0
        self.setpoint = 0.0
    
    def set_dispatch(self, target_power: float, dt_hours: float) -> float:
        """
        Dispatch thermal generator with ramp rate limits
        
        Args:
            target_power: Desired power output (kW)
            dt_hours: Time step (hours)
        
        Returns:
            actual_power: Actual power output after ramp limits (kW)
        """
        # Apply power limits
        target_power = np.clip(target_power, 0, self.max_power_kw)
        
        # Apply ramp rate limits
        dt_minutes = dt_hours * 60
        max_ramp = self.ramp_rate_kw_per_min * dt_minutes
        
        power_change = target_power - self.power_output
        power_change = np.clip(power_change, -max_ramp, max_ramp)
        
        self.power_output += power_change
        self.setpoint = target_power
        
        return self.power_output
    
    def get_state(self) -> Dict[str, float]:
        """Get current generator state"""
        return {
            'power_output': self.power_output,
            'setpoint': self.setpoint,
            'ramp_margin': self.max_power_kw - self.power_output
        }


class BatteryEnergyStorageSystem:
    """Battery for fast frequency regulation"""
    
    def __init__(self, capacity_kwh=10000, max_power_kw=7000, efficiency=0.95, initial_soc=0.5):
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.efficiency = efficiency
        self.soc = initial_soc
        self.energy_kwh = self.soc * self.capacity_kwh
        
        self.min_soc = 0.1
        self.max_soc = 0.9
    
    def charge(self, power_kw: float, dt_hours: float) -> float:
        """Charge battery and return actual power charged"""
        power_kw = min(power_kw, self.max_power_kw)
        energy_to_add = power_kw * dt_hours * self.efficiency
        
        max_energy = self.max_soc * self.capacity_kwh
        available_capacity = max_energy - self.energy_kwh
        
        if energy_to_add > available_capacity:
            energy_to_add = available_capacity
            power_kw = energy_to_add / (dt_hours * self.efficiency)
        
        self.energy_kwh += energy_to_add
        self.soc = self.energy_kwh / self.capacity_kwh
        
        return power_kw
    
    def discharge(self, power_kw: float, dt_hours: float) -> float:
        """Discharge battery and return actual power discharged"""
        power_kw = min(power_kw, self.max_power_kw)
        energy_to_remove = power_kw * dt_hours / self.efficiency
        
        min_energy = self.min_soc * self.capacity_kwh
        available_energy = self.energy_kwh - min_energy
        
        if energy_to_remove > available_energy:
            energy_to_remove = available_energy
            power_kw = energy_to_remove * self.efficiency / dt_hours
        
        self.energy_kwh -= energy_to_remove
        self.soc = self.energy_kwh / self.capacity_kwh
        
        return power_kw
    
    def get_state(self) -> Dict[str, float]:
        """Get current battery state"""
        return {
            'soc': self.soc,
            'energy_kwh': self.energy_kwh,
            'available_charge_power': self.max_power_kw if self.soc < self.max_soc else 0,
            'available_discharge_power': self.max_power_kw if self.soc > self.min_soc else 0
        }



class GridSimulator:
    """Grid physics with per-unit swing equation"""
    
    def __init__(self, base_frequency=50.0, inertia_constant=10.0, damping_coefficient=6.0):
        self.base_frequency = base_frequency
        self.frequency = base_frequency
        self.prev_frequency = base_frequency
        
        # Per-unit system parameters
        self.S_base = 2000.0  # System base power (kW)
        self.H = inertia_constant  # Inertia constant (seconds)
        self.D = damping_coefficient  # Damping coefficient (kW/Hz)
        
        self.voltage_pu = 1.0
        self.freq_min = 49.5
        self.freq_max = 50.5
    
    def update_frequency(self, p_gen: float, p_load: float, dt_hours: float) -> float:
        """Update grid frequency using per-unit swing equation"""
        self.prev_frequency = self.frequency
        
        # Convert to per-unit
        p_gen_pu = p_gen / self.S_base
        p_load_pu = p_load / self.S_base
        delta_p_pu = p_gen_pu - p_load_pu
        
        # Frequency deviation
        delta_f = self.frequency - self.base_frequency
        
        # Damping in per-unit
        D_pu = self.D / self.S_base
        
        # Swing equation: df/dt = (f_nom / 2H) * (ΔP_pu - D_pu * Δf)
        df_dt = (self.base_frequency / (2 * self.H)) * (delta_p_pu - D_pu * delta_f)
        
        # Update frequency
        dt_seconds = dt_hours * 3600
        delta_freq = df_dt * dt_seconds
        
        # Frequency limiter for numerical stability
        max_freq_change = 5.0 * dt_hours
        delta_freq = np.clip(delta_freq, -max_freq_change, max_freq_change)
        
        self.frequency += delta_freq
        self.frequency = np.clip(self.frequency, 40.0, 60.0)
        
        return self.frequency
    
    def get_rocof(self, dt_hours: float) -> float:
        """Calculate rate of change of frequency (RoCoF)"""
        dt_seconds = dt_hours * 3600
        if dt_seconds > 0:
            return (self.frequency - self.prev_frequency) / dt_seconds
        return 0.0
    
    def is_stable(self) -> bool:
        """Check if grid is within stability limits"""
        return self.freq_min <= self.frequency <= self.freq_max



class HybridController:
    """
    Hierarchical controller for hybrid microgrid
    
    - Thermal: Slow dispatch (moving average)
    - Battery: Fast regulation (VSM + AGC + SOC correction)
    """
    
    def __init__(self, battery: BatteryEnergyStorageSystem, thermal: ThermalGenerator, 
                 grid: GridSimulator):
        self.battery = battery
        self.thermal = thermal
        self.grid = grid
        
        # VSM gains
        self.k_inertia = 4000  # kW/(Hz/s)
        self.k_droop = 6000    # kW/Hz
        
        # AGC gains
        self.k_i_agc = 5.0
        self.k_aw = 0.5  # Back-calculation anti-windup
        self.ace_integral = 0.0
        self.integral_limit = 5000.0
        
        # SOC correction (proportional only)
        self.k_soc = 1000  # Increased for stronger SOC centering
        self.soc_target = 0.5
        
        # Moving average window for thermal dispatch
        self.ma_window = 12  # 12 timesteps = 24 minutes at 2-min timestep
        self.renewable_history = []
    
    def compute_thermal_dispatch(self, p_renewable: float, p_load: float) -> float:
        """
        Compute thermal generator setpoint based on moving average
        
        Thermal follows: base_load - MA(renewable)
        """
        # Update renewable history
        self.renewable_history.append(p_renewable)
        if len(self.renewable_history) > self.ma_window:
            self.renewable_history.pop(0)
        
        # Moving average of renewable
        ma_renewable = np.mean(self.renewable_history)
        
        # Thermal setpoint: base load minus average renewable
        thermal_setpoint = p_load - ma_renewable
        thermal_setpoint = max(0, thermal_setpoint)  # Non-negative
        
        return thermal_setpoint
    
    def compute_vsm(self, freq: float, df_dt: float) -> float:
        """Virtual Synchronous Machine control"""
        freq_dev = freq - self.grid.base_frequency
        
        # Virtual inertia
        p_inertia = -self.k_inertia * df_dt
        
        # Droop control
        p_droop = -self.k_droop * freq_dev
        
        return p_inertia + p_droop
    
    def compute_agc(self, freq: float, dt_hours: float, p_command: float, 
                    p_actual: float) -> Tuple[float, float]:
        """AGC with back-calculation anti-windup"""
        freq_error = self.grid.base_frequency - freq
        
        # Back-calculation anti-windup
        tracking_error = p_actual - p_command
        integral_update = self.k_i_agc * freq_error + self.k_aw * tracking_error
        
        # Update integral
        self.ace_integral += integral_update * dt_hours
        self.ace_integral = np.clip(self.ace_integral, -self.integral_limit, self.integral_limit)
        
        # AGC power
        p_agc = self.ace_integral
        
        # Calculate ACE for monitoring
        beta = 20
        ace = beta * (freq - self.grid.base_frequency)
        
        return p_agc, ace
    
    def compute_soc_correction(self, soc: float) -> float:
        """Simple proportional SOC correction"""
        soc_error = soc - self.soc_target
        return self.k_soc * soc_error
    
    def decide_battery_action(self, p_renewable: float, p_load: float, p_thermal: float,
                              battery_state: Dict[str, float], dt_hours: float) -> Tuple[float, Dict]:
        """
        Decide battery action for fast frequency regulation
        
        Battery handles: net_imbalance = (renewable + thermal) - load
        """
        freq = self.grid.frequency
        df_dt = self.grid.get_rocof(dt_hours)
        soc = battery_state['soc']
        
        # VSM control (fast)
        p_vsm = self.compute_vsm(freq, df_dt)
        
        # SOC correction (slow)
        p_soc = self.compute_soc_correction(soc)
        
        # Preliminary command
        p_command_prelim = p_vsm + p_soc
        
        # Apply physical constraints
        if p_command_prelim > 0:
            p_actual_prelim = min(p_command_prelim, battery_state['available_discharge_power'])
        else:
            p_actual_prelim = max(p_command_prelim, -battery_state['available_charge_power'])
        
        # AGC control (medium)
        p_agc, ace = self.compute_agc(freq, dt_hours, p_command_prelim, p_actual_prelim)
        
        # Total command
        p_command_total = p_vsm + p_agc + p_soc
        
        # Apply physical constraints
        if p_command_total > 0:
            p_battery = min(p_command_total, battery_state['available_discharge_power'])
        else:
            p_battery = max(p_command_total, -battery_state['available_charge_power'])
        
        # Control breakdown
        breakdown = {
            'p_vsm': p_vsm,
            'p_agc': p_agc,
            'p_soc': p_soc,
            'ace': ace
        }
        
        return p_battery, breakdown



class HybridMicrogrid:
    """Main hybrid microgrid simulator"""
    
    def __init__(self, data_path='data/preprocessed', results_path='results', verbose=False):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(level=logging.INFO, 
                              format='%(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING)
        
        # Initialize components
        self.battery = BatteryEnergyStorageSystem(
            capacity_kwh=10000,
            max_power_kw=7000,
            efficiency=0.95,
            initial_soc=0.5
        )
        
        self.thermal = ThermalGenerator(
            max_power_kw=5000,
            ramp_rate_kw_per_min=50
        )
        
        self.grid = GridSimulator(
            base_frequency=50.0,
            inertia_constant=10.0,
            damping_coefficient=6.0
        )
        
        self.controller = HybridController(self.battery, self.thermal, self.grid)
        
        logger.info("Hybrid Microgrid initialized")
        logger.info(f"Battery: {self.battery.capacity_kwh} kWh, {self.battery.max_power_kw} kW")
        logger.info(f"Thermal: {self.thermal.max_power_kw} kW, {self.thermal.ramp_rate_kw_per_min} kW/min")
    
    def load_data(self) -> pd.DataFrame:
        """Load renewable generation data"""
        file_path = self.data_path / 'engineered_features.csv'
        df = pd.read_csv(file_path)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded data: {df.shape}")
        return df
    
    def generate_load_profile(self, n_samples: int, dt_hours: float, p_renewable: np.ndarray) -> np.ndarray:
        """Generate load profile for hybrid microgrid"""
        hours = (np.arange(n_samples) * dt_hours) % 24
        
        # Base load should be higher than renewable average
        # This allows thermal to provide base power
        avg_renewable = np.mean(p_renewable)
        base_load = avg_renewable * 1.70  # 70% higher than renewable average
        
        # Daily variation
        daily_pattern = 1.0 + 0.25 * np.sin(2 * np.pi * (hours - 6) / 24)
        noise = np.random.normal(0, 40, n_samples)
        
        load = base_load * daily_pattern + noise
        load = np.maximum(load, 800)
        
        logger.info(f"Load profile: base={base_load:.1f} kW (renewable avg={avg_renewable:.1f} kW)")
        
        return load
    
    def validate_battery_sizing(self, p_renewable: np.ndarray, p_load: np.ndarray):
        """Validate battery power is sized for renewable fluctuations"""
        net_imbalance = p_renewable - p_load
        std_imbalance = np.std(net_imbalance)
        
        required_power = 2 * std_imbalance
        actual_power = self.battery.max_power_kw
        
        logger.info(f"Net imbalance std: {std_imbalance:.1f} kW")
        logger.info(f"Required battery power (2×std): {required_power:.1f} kW")
        logger.info(f"Actual battery power: {actual_power:.1f} kW")
        
        if actual_power >= required_power:
            logger.info("✓ Battery adequately sized")
        else:
            logger.warning(f"⚠ Battery undersized by {required_power - actual_power:.1f} kW")

    
    def run_simulation(self, df: pd.DataFrame, dt_hours=1/30) -> pd.DataFrame:
        """Run hybrid microgrid simulation"""
        logger.info(f"Starting simulation: dt={dt_hours*60:.0f} min")
        
        n_samples = len(df)
        
        # Get renewable generation
        p_solar = df['PV Generation (KW)'].values
        p_wind = df['Power Generation '].values
        p_renewable = p_solar + p_wind
        
        # Generate load
        p_load = self.generate_load_profile(n_samples, dt_hours, p_renewable)
        
        # Validate battery sizing
        self.validate_battery_sizing(p_renewable, p_load)
        
        # Initialize results
        results = {
            'timestamp': df['timestamp'].values if 'timestamp' in df.columns else np.arange(n_samples),
            'p_renewable': p_renewable,
            'p_load': p_load,
            'p_thermal': np.zeros(n_samples),
            'p_battery': np.zeros(n_samples),
            'p_total_gen': np.zeros(n_samples),
            'frequency': np.zeros(n_samples),
            'battery_soc': np.zeros(n_samples),
            'thermal_setpoint': np.zeros(n_samples),
            'ace': np.zeros(n_samples),
            'rocof': np.zeros(n_samples),
            'is_stable': np.zeros(n_samples, dtype=bool),
            'p_vsm': np.zeros(n_samples),
            'p_agc': np.zeros(n_samples),
            'p_soc': np.zeros(n_samples)
        }
        
        # Simulation loop
        for i in range(n_samples):
            # Thermal dispatch (slow)
            thermal_setpoint = self.controller.compute_thermal_dispatch(
                p_renewable[i], p_load[i]
            )
            p_thermal = self.thermal.set_dispatch(thermal_setpoint, dt_hours)
            
            # Battery action (fast)
            battery_state = self.battery.get_state()
            p_battery_cmd, breakdown = self.controller.decide_battery_action(
                p_renewable[i], p_load[i], p_thermal, battery_state, dt_hours
            )
            
            # Execute battery action
            if p_battery_cmd > 0:
                p_battery = self.battery.discharge(p_battery_cmd, dt_hours)
            else:
                p_battery = -self.battery.charge(-p_battery_cmd, dt_hours)
            
            # Total generation
            p_total_gen = p_renewable[i] + p_thermal + p_battery
            
            # Update grid
            frequency = self.grid.update_frequency(p_total_gen, p_load[i], dt_hours)
            rocof = self.grid.get_rocof(dt_hours)
            
            # Store results
            results['p_thermal'][i] = p_thermal
            results['p_battery'][i] = p_battery
            results['p_total_gen'][i] = p_total_gen
            results['frequency'][i] = frequency
            results['battery_soc'][i] = self.battery.soc
            results['thermal_setpoint'][i] = thermal_setpoint
            results['ace'][i] = breakdown['ace']
            results['rocof'][i] = rocof
            results['is_stable'][i] = self.grid.is_stable()
            results['p_vsm'][i] = breakdown['p_vsm']
            results['p_agc'][i] = breakdown['p_agc']
            results['p_soc'][i] = breakdown['p_soc']
        
        logger.info(f"Simulation complete: {n_samples} timesteps")
        
        return pd.DataFrame(results)
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze simulation results"""
        metrics = {
            # Frequency
            'frequency_mean_hz': float(results_df['frequency'].mean()),
            'frequency_std_hz': float(results_df['frequency'].std()),
            'frequency_min_hz': float(results_df['frequency'].min()),
            'frequency_max_hz': float(results_df['frequency'].max()),
            'stability_rate': float(results_df['is_stable'].sum() / len(results_df)),
            
            # Battery
            'battery_soc_mean': float(results_df['battery_soc'].mean()),
            'battery_soc_std': float(results_df['battery_soc'].std()),
            'battery_power_mean': float(results_df['p_battery'].abs().mean()),
            'battery_power_max': float(results_df['p_battery'].abs().max()),
            
            # Thermal
            'thermal_power_mean': float(results_df['p_thermal'].mean()),
            'thermal_utilization': float(results_df['p_thermal'].mean() / self.thermal.max_power_kw),
            
            # Performance
            'ace_mean_kw': float(results_df['ace'].abs().mean()),
            'ace_max_kw': float(results_df['ace'].abs().max()),
            'rocof_mean': float(results_df['rocof'].abs().mean()),
            'rocof_max': float(results_df['rocof'].abs().max()),
            
            # Control
            'p_vsm_mean': float(results_df['p_vsm'].abs().mean()),
            'p_agc_mean': float(results_df['p_agc'].abs().mean()),
            'p_soc_mean': float(results_df['p_soc'].abs().mean())
        }
        
        if self.verbose:
            self._print_results(metrics, len(results_df))
        
        return metrics
    
    def _print_results(self, metrics: Dict[str, float], n_samples: int):
        """Print results summary"""
        print("\n" + "="*70)
        print("HYBRID MICROGRID SIMULATION RESULTS")
        print("="*70)
        
        print(f"\n📊 Frequency Performance:")
        print(f"  Mean: {metrics['frequency_mean_hz']:.3f} Hz (Target: 50.0 Hz)")
        print(f"  Std Dev: {metrics['frequency_std_hz']:.3f} Hz")
        print(f"  Range: [{metrics['frequency_min_hz']:.3f}, {metrics['frequency_max_hz']:.3f}] Hz")
        print(f"  Stability: {100*metrics['stability_rate']:.1f}% (Target: >95%)")
        
        print(f"\n🔋 Battery Performance:")
        print(f"  Mean SOC: {100*metrics['battery_soc_mean']:.1f}% (Target: 50%)")
        print(f"  SOC Std Dev: {100*metrics['battery_soc_std']:.1f}%")
        print(f"  Mean Power: {metrics['battery_power_mean']:.1f} kW")
        print(f"  Max Power: {metrics['battery_power_max']:.1f} kW")
        
        print(f"\n🔥 Thermal Generator:")
        print(f"  Mean Power: {metrics['thermal_power_mean']:.1f} kW")
        print(f"  Utilization: {100*metrics['thermal_utilization']:.1f}%")
        
        print(f"\n⚡ Grid Performance:")
        print(f"  Mean |ACE|: {metrics['ace_mean_kw']:.2f} kW (Target: <300 kW)")
        print(f"  Max |RoCoF|: {metrics['rocof_max']:.4f} Hz/s (Target: <0.5 Hz/s)")
        
        print(f"\n🎛️ Control Contributions:")
        print(f"  VSM: {metrics['p_vsm_mean']:.1f} kW (fast)")
        print(f"  AGC: {metrics['p_agc_mean']:.1f} kW (medium)")
        print(f"  SOC: {metrics['p_soc_mean']:.1f} kW (slow)")
        
        # Assessment
        targets_met = 0
        if 49.5 <= metrics['frequency_mean_hz'] <= 50.5:
            targets_met += 1
        if metrics['stability_rate'] > 0.95:
            targets_met += 1
        if 45 <= metrics['battery_soc_mean']*100 <= 55:
            targets_met += 1
        if metrics['ace_mean_kw'] < 300:
            targets_met += 1
        
        print(f"\n{'='*70}")
        print(f"TARGETS ACHIEVED: {targets_met}/4")
        if targets_met >= 3:
            print("✅ SYSTEM OPERATIONAL")
        else:
            print("⚠️ SYSTEM NEEDS TUNING")
        print("="*70 + "\n")
    
    def save_results(self, results_df: pd.DataFrame, metrics: Dict[str, float]):
        """Save simulation results"""
        results_file = self.results_path / 'hybrid_microgrid_results.csv'
        results_df.to_csv(results_file, index=False)
        
        metrics_file = self.results_path / 'hybrid_microgrid_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {self.results_path}")
    
    def run(self, dt_hours=1/30) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Execute complete simulation"""
        logger.info("Starting hybrid microgrid simulation")
        
        df = self.load_data()
        results_df = self.run_simulation(df, dt_hours)
        metrics = self.analyze_results(results_df)
        self.save_results(results_df, metrics)
        
        logger.info("Simulation complete")
        
        return results_df, metrics


if __name__ == '__main__':
    # Run simulation
    microgrid = HybridMicrogrid(verbose=True)
    results_df, metrics = microgrid.run(dt_hours=1/30)  # 2-minute timestep
    
    print(f"📈 FINAL SUMMARY:")
    print(f"  Frequency: {metrics['frequency_mean_hz']:.3f} Hz")
    print(f"  Stability: {100*metrics['stability_rate']:.1f}%")
    print(f"  Battery SOC: {100*metrics['battery_soc_mean']:.1f}%")
    print(f"  ACE: {metrics['ace_mean_kw']:.2f} kW")
