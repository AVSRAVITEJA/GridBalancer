"""
Data Validation and Alignment Module for PVGRIDBALANCER
Ensures data integrity and temporal alignment across datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataValidator:
    """Validates and aligns cleaned renewable energy datasets"""
    
    def __init__(self, preprocessed_path='data/preprocessed'):
        self.preprocessed_path = Path(preprocessed_path)
    
    def load_cleaned_data(self, pattern):
        """Load cleaned CSV files"""
        files = sorted(self.preprocessed_path.glob(pattern))
        dataframes = []
        
        for file in files:
            print(f"Loading {file.name}...")
            df = pd.read_csv(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            dataframes.append(df)
        
        return dataframes
    
    def validate_data_quality(self, df, name):
        """Validate data quality metrics"""
        print(f"\n--- Validation Report: {name} ---")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"Missing values:\n{missing[missing > 0]}")
        else:
            print("✓ No missing values")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"⚠ Found {duplicates} duplicate rows")
        else:
            print("✓ No duplicate rows")
        
        # Check timestamp continuity
        if 'timestamp' in df.columns:
            time_diff = df['timestamp'].diff()
            mode_diff = time_diff.mode()[0] if len(time_diff.mode()) > 0 else None
            gaps = time_diff[time_diff != mode_diff].dropna()
            if len(gaps) > 0:
                print(f"⚠ Found {len(gaps)} time gaps")
            else:
                print("✓ Continuous timestamps")
        
        # Check value ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'power' in col.lower() or 'generation' in col.lower():
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"  {col}: [{min_val:.2f}, {max_val:.2f}]")
                
                if min_val < 0:
                    print(f"  ⚠ Negative values in {col}")
        
        return True
    
    def align_timestamps(self, dataframes, freq='15min'):
        """Align all dataframes to common timestamp grid"""
        print(f"\n=== Aligning timestamps to {freq} frequency ===")
        
        # Find common time range
        start_times = [df['timestamp'].min() for df in dataframes if 'timestamp' in df.columns]
        end_times = [df['timestamp'].max() for df in dataframes if 'timestamp' in df.columns]
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        print(f"Common time range: {common_start} to {common_end}")
        
        # Create common timestamp index
        common_index = pd.date_range(start=common_start, end=common_end, freq=freq)
        
        aligned_dfs = []
        for i, df in enumerate(dataframes):
            if 'timestamp' in df.columns:
                df_aligned = df.set_index('timestamp').reindex(common_index)
                df_aligned.index.name = 'timestamp'
                df_aligned = df_aligned.reset_index()
                aligned_dfs.append(df_aligned)
                print(f"  Aligned dataset {i+1}: {len(df_aligned)} records")
            else:
                aligned_dfs.append(df)
        
        return aligned_dfs
    
    def validate_all_solar(self):
        """Validate all solar datasets"""
        print("\n=== Validating Solar Datasets ===")
        solar_data = self.load_cleaned_data('solar_cleaned_*.csv')
        
        for i, df in enumerate(solar_data, 1):
            self.validate_data_quality(df, f"Solar Dataset {i}")
        
        return solar_data
    
    def validate_all_wind(self):
        """Validate all wind datasets"""
        print("\n=== Validating Wind Datasets ===")
        wind_data = self.load_cleaned_data('wind_cleaned_*.csv')
        
        for i, df in enumerate(wind_data, 1):
            self.validate_data_quality(df, f"Wind Dataset {i}")
        
        return wind_data
    
    def run(self):
        """Execute full validation pipeline"""
        print("Starting data validation pipeline...")
        solar_data = self.validate_all_solar()
        wind_data = self.validate_all_wind()
        
        # Align timestamps if needed
        if len(solar_data) > 1:
            solar_aligned = self.align_timestamps(solar_data)
        else:
            solar_aligned = solar_data
        
        if len(wind_data) > 1:
            wind_aligned = self.align_timestamps(wind_data)
        else:
            wind_aligned = wind_data
        
        print("\n✓ Data validation completed successfully!")
        return solar_aligned, wind_aligned


if __name__ == '__main__':
    validator = DataValidator()
    validator.run()
