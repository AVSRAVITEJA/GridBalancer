"""
Complete PVGRIDBALANCER Pipeline Orchestrator
Executes all stages from data preprocessing to model training
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from data_cleaning import DataCleaner
from validate_and_align import DataValidator
from merged_datasets import DatasetMerger
from feature_engineering_data import FeatureEngineer
from hybrid_forecasting import HybridForecaster
from grid_balancer import GridBalancer
from model_stress_test import StressTester


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)


def main():
    """Execute complete PVGRIDBALANCER pipeline"""
    print_header("PVGRIDBALANCER - COMPLETE PIPELINE")
    print("\nThis pipeline executes:")
    print("  1. Data Cleaning & Preprocessing")
    print("  2. Data Validation & Alignment")
    print("  3. Dataset Merging & Synchronization")
    print("  4. Physics-Based Feature Engineering")
    print("  5. Hybrid Forecasting Model Training")
    print("  6. Grid Balancing Simulation (MARL)")
    print("  7. Stress Testing & Resilience Validation")
    print("\n" + "="*70)
    
    input("\nPress Enter to start the pipeline...")
    
    # Stage 1: Data Cleaning
    print_header("STAGE 1/5: DATA CLEANING")
    cleaner = DataCleaner()
    solar_cleaned, wind_cleaned = cleaner.run()
    
    # Stage 2: Validation & Alignment
    print_header("STAGE 2/5: VALIDATION & ALIGNMENT")
    validator = DataValidator()
    solar_validated, wind_validated = validator.run()
    
    # Stage 3: Dataset Merging
    print_header("STAGE 3/5: DATASET MERGING")
    merger = DatasetMerger()
    merged_data = merger.run()
    
    # Stage 4: Feature Engineering
    print_header("STAGE 4/5: FEATURE ENGINEERING")
    engineer = FeatureEngineer()
    engineered_data = engineer.run()
    
    # Stage 5: Hybrid Forecasting
    print_header("STAGE 5/7: HYBRID FORECASTING")
    forecaster = HybridForecaster()
    forecast_summary = forecaster.run(epochs=50)
    
    # Stage 6: Grid Balancing
    print_header("STAGE 6/7: GRID BALANCING (MARL)")
    balancer = GridBalancer()
    balancing_results, balancing_metrics = balancer.run()
    
    # Stage 7: Stress Testing
    print_header("STAGE 7/7: STRESS TESTING")
    tester = StressTester()
    stress_results = tester.run_all_tests()
    
    # Final Summary
    print_header("PIPELINE COMPLETE")
    print("\n✓ All stages completed successfully!")
    print("\nGenerated Outputs:")
    print("  • Cleaned Datasets → data/preprocessed/")
    print("  • Unified Dataset → data/preprocessed/unified_renewable_data.csv")
    print("  • Engineered Features → data/preprocessed/engineered_features.csv")
    print("  • Trained Models → models/")
    print("  • Grid Balancing Results → results/grid_balancing_results.csv")
    print("  • Stress Test Results → results/stress_test_results.json")
    print("\nKey Metrics:")
    print(f"  • Grid Stability Rate: {100*balancing_metrics['stability_rate']:.1f}%")
    print(f"  • Mean Frequency: {balancing_metrics['frequency_mean_hz']:.3f} Hz")
    print(f"  • Stress Tests Passed: {sum(1 for r in stress_results if r['passed'])}/{len(stress_results)}")
    print("\nNext Steps:")
    print("  • Review results in results/ directory")
    print("  • Run main_simulation.py for interactive demonstration")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
