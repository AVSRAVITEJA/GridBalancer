"""
Hybrid Forecasting System for PVGRIDBALANCER
Orchestrates dual-stream TCN models for solar and wind prediction
Combines predictions for total renewable generation forecasting
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from pv_forecasting_model import PVForecaster
from wind_forecasting_model import WindForecaster


class HybridForecaster:
    """Hybrid forecasting system combining solar and wind models"""
    
    def __init__(self, preprocessed_path='data/preprocessed', models_path='models', results_path='results'):
        self.preprocessed_path = Path(preprocessed_path)
        self.models_path = Path(models_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.pv_forecaster = PVForecaster(preprocessed_path, models_path)
        self.wind_forecaster = WindForecaster(preprocessed_path, models_path)
    
    def train_all_models(self, epochs=50):
        """Train both PV and Wind forecasting models"""
        print("="*70)
        print("HYBRID FORECASTING SYSTEM - Training Pipeline")
        print("="*70)
        
        # Train PV model
        print("\n[1/2] Training Solar PV Model")
        print("-"*70)
        pv_results = self.pv_forecaster.run(epochs=epochs)
        
        # Train Wind model
        print("\n[2/2] Training Wind Power Model")
        print("-"*70)
        wind_results = self.wind_forecaster.run(epochs=epochs)
        
        return pv_results, wind_results
    
    def generate_combined_forecast(self):
        """Generate combined renewable generation forecast"""
        print("\n" + "="*70)
        print("Generating Combined Renewable Forecast")
        print("="*70)
        
        # Load engineered features
        df = pd.read_csv(self.preprocessed_path / 'engineered_features.csv')
        
        # Get actual values
        pv_actual = df['PV Generation (KW)'].values
        wind_actual = df['Power Generation '].values
        total_actual = pv_actual + wind_actual
        
        print(f"\nDataset: {len(df)} samples")
        print(f"  Solar range: [{pv_actual.min():.2f}, {pv_actual.max():.2f}] kW")
        print(f"  Wind range: [{wind_actual.min():.2f}, {wind_actual.max():.2f}] kW")
        print(f"  Total range: [{total_actual.min():.2f}, {total_actual.max():.2f}] kW")
        
        # Calculate statistics
        print(f"\nGeneration Statistics:")
        print(f"  Solar mean: {pv_actual.mean():.2f} kW")
        print(f"  Wind mean: {wind_actual.mean():.2f} kW")
        print(f"  Total mean: {total_actual.mean():.2f} kW")
        
        # Calculate complementarity
        if 'complementarity_index' in df.columns:
            comp_index = df['complementarity_index'].mean()
            print(f"\nComplementarity Index: {comp_index:.3f}")
        
        # Save forecast summary
        summary = {
            'total_samples': len(df),
            'solar_mean_kw': pv_actual.mean(),
            'wind_mean_kw': wind_actual.mean(),
            'total_mean_kw': total_actual.mean(),
            'solar_max_kw': pv_actual.max(),
            'wind_max_kw': wind_actual.max(),
            'total_max_kw': total_actual.max(),
            'complementarity_index': df['complementarity_index'].mean() if 'complementarity_index' in df.columns else None
        }
        
        summary_file = self.results_path / 'forecast_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("HYBRID FORECASTING SUMMARY\n")
            f.write("="*50 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✓ Saved forecast summary to {summary_file}")
        
        return summary
    
    def run(self, epochs=50):
        """Execute complete hybrid forecasting pipeline"""
        print("\n" + "="*70)
        print("PVGRIDBALANCER - HYBRID FORECASTING SYSTEM")
        print("="*70)
        print("\nThis system trains dual-stream TCN models for:")
        print("  • Solar PV generation forecasting")
        print("  • Wind power generation forecasting")
        print("  • Combined renewable generation prediction")
        print("\n" + "="*70)
        
        # Train models
        pv_results, wind_results = self.train_all_models(epochs=epochs)
        
        # Generate combined forecast
        summary = self.generate_combined_forecast()
        
        # Final summary
        print("\n" + "="*70)
        print("HYBRID FORECASTING COMPLETE")
        print("="*70)
        print("\n✓ Trained Models:")
        print("  • Solar PV TCN Model → models/pv_tcn_model.keras")
        print("  • Wind Power TCN Model → models/wind_tcn_model.keras")
        print("\n✓ Model Scalers:")
        print("  • PV Scalers → models/pv_scalers.pkl")
        print("  • Wind Scalers → models/wind_scalers.pkl")
        print("\n✓ Results:")
        print("  • Forecast Summary → results/forecast_summary.txt")
        print("\nReady for grid balancing and arbitrage logic!")
        print("="*70)
        
        return summary


if __name__ == '__main__':
    forecaster = HybridForecaster()
    forecaster.run(epochs=50)
