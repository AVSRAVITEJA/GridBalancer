"""
Main preprocessing pipeline orchestrator for PVGRIDBALANCER
Executes the complete data processing workflow:
1. Data Cleaning
2. Validation & Alignment
3. Dataset Merging
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from data_cleaning import DataCleaner
from validate_and_align import DataValidator
from merged_datasets import DatasetMerger


def main():
    """Execute complete preprocessing pipeline"""
    print("="*60)
    print("PVGRIDBALANCER - Data Preprocessing Pipeline")
    print("="*60)
    
    # Step 1: Data Cleaning
    print("\n[STEP 1/3] Data Cleaning")
    print("-"*60)
    cleaner = DataCleaner()
    solar_cleaned, wind_cleaned = cleaner.run()
    
    # Step 2: Validation & Alignment
    print("\n[STEP 2/3] Validation & Alignment")
    print("-"*60)
    validator = DataValidator()
    solar_validated, wind_validated = validator.run()
    
    # Step 3: Dataset Merging
    print("\n[STEP 3/3] Dataset Merging")
    print("-"*60)
    merger = DatasetMerger()
    merged_data = merger.run()
    
    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"✓ Cleaned datasets saved in: data/preprocessed/")
    print(f"✓ Solar datasets: {len(solar_cleaned)}")
    print(f"✓ Wind datasets: {len(wind_cleaned)}")
    if merged_data is not None:
        print(f"✓ Unified dataset: {merged_data.shape[0]} records, {merged_data.shape[1]} features")
    print("\nReady for feature engineering and model training!")
    print("="*60)


if __name__ == '__main__':
    main()
