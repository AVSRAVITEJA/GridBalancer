"""
Main Simulation Orchestrator for PVGRIDBALANCER
Integrates all components for complete system demonstration
"""

import sys
from pathlib import Path
import argparse

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from grid_balancer import GridBalancer
from model_stress_test import StressTester


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)


def run_grid_balancing():
    """Run grid balancing simulation"""
    print_header("GRID BALANCING SIMULATION")
    print("\nSimulating MARL Arbitrator with BESS control")
    print("Objectives:")
    print("  • Maintain frequency: 49.5-50.5 Hz")
    print("  • Maintain voltage: 0.95-1.05 pu")
    print("  • Minimize Area Control Error (ACE)")
    print("  • Optimize battery SOC management")
    
    balancer = GridBalancer()
    results_df, metrics = balancer.run()
    
    return results_df, metrics


def run_stress_testing():
    """Run stress testing scenarios"""
    print_header("STRESS TESTING & RESILIENCE VALIDATION")
    print("\nTesting grid resilience under worst-case scenarios:")
    print("  • Wind drop during sunset")
    print("  • Rapid cloud cover")
    print("  • Combined renewable failure")
    print("  • High load spikes")
    print("  • Oscillating wind power")
    
    tester = StressTester()
    stress_results = tester.run_all_tests()
    
    return stress_results


def main():
    """Main simulation orchestrator"""
    parser = argparse.ArgumentParser(description='PVGRIDBALANCER Main Simulation')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'balance', 'stress'],
                       help='Simulation mode: all, balance, or stress')
    
    args = parser.parse_args()
    
    print_header("PVGRIDBALANCER - MAIN SIMULATION")
    print("\nMulti-Agent Dynamic Arbitrage & Grid Resilience Framework")
    print("\nThis simulation demonstrates:")
    print("  1. MARL-based battery arbitrage for grid stability")
    print("  2. Frequency and voltage regulation")
    print("  3. Stress testing under worst-case scenarios")
    print("  4. Grid resilience validation")
    
    print("\n" + "="*70)
    input("\nPress Enter to start simulation...")
    
    # Run simulations based on mode
    if args.mode in ['all', 'balance']:
        results_df, metrics = run_grid_balancing()
    
    if args.mode in ['all', 'stress']:
        stress_results = run_stress_testing()
    
    # Final summary
    print_header("SIMULATION COMPLETE")
    
    if args.mode in ['all', 'balance']:
        print("\n✓ Grid Balancing Results:")
        print(f"  • Frequency stability: {100*metrics['stability_rate']:.1f}%")
        print(f"  • Mean frequency: {metrics['frequency_mean_hz']:.3f} Hz")
        print(f"  • Mean |ACE|: {metrics['ace_mean_kw']:.2f} kW")
        print(f"  • Battery SOC: {100*metrics['battery_soc_mean']:.1f}%")
    
    if args.mode in ['all', 'stress']:
        passed = sum(1 for r in stress_results if r['passed'])
        total = len(stress_results)
        print(f"\n✓ Stress Testing Results:")
        print(f"  • Tests passed: {passed}/{total} ({100*passed/total:.1f}%)")
    
    print("\n✓ All results saved to results/ directory")
    print("\nGenerated Files:")
    print("  • results/grid_balancing_results.csv")
    print("  • results/grid_balancing_metrics.json")
    print("  • results/stress_test_results.json")
    print("  • results/stress_test_report.txt")
    
    print("\n" + "="*70)
    print("Thank you for using PVGRIDBALANCER!")
    print("="*70)


if __name__ == '__main__':
    main()
