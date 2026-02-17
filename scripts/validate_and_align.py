<<<<<<< HEAD
"""
Data Validation and Alignment Module for PVGRIDBALANCER
Ensures data integrity and temporal alignment across datasets
"""

=======
>>>>>>> 0ba306db939cd0c78f9380d603453b23345abac5
import pandas as pd
import numpy as np
from pathlib import Path

<<<<<<< HEAD

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
=======
# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"

PV_PATH = DATA_DIR / "pv_master_dataset.csv"
WIND_PATH = DATA_DIR / "wind_master_dataset.csv"

PV_OUT = DATA_DIR / "pv_validated.csv"
WIND_OUT = DATA_DIR / "wind_validated.csv"

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def parse_datetime(df, col_name):
    df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
    invalid = df[col_name].isna().sum()
    if invalid > 0:
        print(f"  {invalid} invalid datetime entries in {col_name}")
    return df


def check_missing_timestamps(df, time_col_name):
    """Check for missing timestamps. df should have datetime index."""
    if time_col_name in df.columns:
        # If time column is still a regular column
        deltas = df[time_col_name].diff().value_counts()
    else:
        # If time column is already the index
        deltas = df.index.to_series().diff().value_counts()
    
    print("  Time interval distribution:")
    print(deltas.head())
    return df

# --------------------------------------------------
# PV DATASET VALIDATION
# --------------------------------------------------
print("\nVALIDATING PV DATASET")
print("-" * 50)

pv = pd.read_csv(PV_PATH)
print(f"  Loaded {len(pv)} rows, {len(pv.columns)} columns")
print(f"  Columns: {list(pv.columns)}")

# Parse datetime
pv = parse_datetime(pv, "Date - Time")

# Sort & set index
pv = pv.sort_values("Date - Time").set_index("Date - Time")
print(f"  Index set to: {pv.index.name}")

# Basic sanity checks
print("\n  Performing sanity checks...")
try:
    assert (pv["Ground Radiation Intensity (W/m^2)"] >= 0).all(), "❌ Negative ground radiation found"
    print("   Ground radiation values are non-negative")
except AssertionError as e:
    print(f"  {e}")
    
try:
    assert (pv["Upper Atmosphere Radiation Intensity (W/m^2)"] >= 0).all(), "❌ Negative upper radiation found"
    print("   Upper atmosphere radiation values are non-negative")
except AssertionError as e:
    print(f"  {e}")

# Night-time PV check
night_pv = pv[
    (pv["Ground Radiation Intensity (W/m^2)"] == 0) &
    (pv["PV Generation (KW)"] > 0)
]

if not night_pv.empty:
    print(f"   Night-time PV generation detected: {len(night_pv)} rows")
else:
    print("   No night-time PV generation detected")

# Efficiency sanity (avoid divide-by-zero)
pv["pv_efficiency"] = np.where(
    pv["Ground Radiation Intensity (W/m^2)"] > 0,
    pv["PV Generation (KW)"] / pv["Ground Radiation Intensity (W/m^2)"],
    0
)

high_efficiency = (pv["pv_efficiency"] > 1).sum()
if high_efficiency > 0:
    print(f"   PV efficiency > 1 detected in {high_efficiency} rows (check units)")
else:
    print(" PV efficiency within reasonable bounds")

# Missing timestamp check
print("\n  Checking timestamp distribution...")
pv = check_missing_timestamps(pv, "Date - Time")

# Check for duplicates in index
duplicates = pv.index.duplicated().sum()
if duplicates > 0:
    print(f"  Found {duplicates} duplicate timestamps")
    pv = pv[~pv.index.duplicated(keep='first')]

# Save
pv.drop(columns=["pv_efficiency"], inplace=True)
pv.to_csv(PV_OUT)

print(f"\n  PV dataset validated and saved to {PV_OUT}")
print(f"  Final shape: {pv.shape}")

# --------------------------------------------------
# WIND DATASET VALIDATION
# --------------------------------------------------
print("\n" + "=" * 50)
print("VALIDATING WIND DATASET")
print("-" * 50)

wind = pd.read_csv(WIND_PATH)
print(f"  Loaded {len(wind)} rows, {len(wind.columns)} columns")
print(f"  Columns: {list(wind.columns)}")

# Parse datetime
wind = parse_datetime(wind, "Date-Time")

# Sort & set index
wind = wind.sort_values("Date-Time").set_index("Date-Time")
print(f"  Index set to: {wind.index.name}")

# Physical sanity checks
print("\n  Performing sanity checks...")
try:
    assert (wind["Wind Speed"] >= 0).all(), " Negative wind speed found"
    print("  ✓ Wind speed values are non-negative")
except AssertionError as e:
    print(f"  {e}")

try:
    assert (wind["Air Density"] > 0).all(), "Non-positive air density found"
    print("  ✓ Air density values are positive")
except AssertionError as e:
    print(f"  {e}")

try:
    assert (wind["Power Generation"] >= 0).all(), " Negative power generation found"
    print("   Power generation values are non-negative")
except AssertionError as e:
    print(f"  {e}")

# Check for unrealistic values
high_wind_speed = (wind["Wind Speed"] > 50).sum()  # > 50 m/s is extreme
if high_wind_speed > 0:
    print(f"   {high_wind_speed} rows with wind speed > 50 m/s (extreme values)")

low_air_density = (wind["Air Density"] < 0.5).sum()  # Unusually low for surface air
if low_air_density > 0:
    print(f"   {low_air_density} rows with air density < 0.5 kg/m³")

# Missing timestamp check
print("\n  Checking timestamp distribution...")
wind = check_missing_timestamps(wind, "Date-Time")

# Check for duplicates in index
duplicates_wind = wind.index.duplicated().sum()
if duplicates_wind > 0:
    print(f"   Found {duplicates_wind} duplicate timestamps")
    wind = wind[~wind.index.duplicated(keep='first')]

# Save
wind.to_csv(WIND_OUT)

print(f"\n  Wind dataset validated and saved to {WIND_OUT}")
print(f"  Final shape: {wind.shape}")

print("\n" + "=" * 50)
print("VALIDATION COMPLETE!")
print("=" * 50)
>>>>>>> 0ba306db939cd0c78f9380d603453b23345abac5
