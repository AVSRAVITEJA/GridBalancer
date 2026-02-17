<<<<<<< HEAD
"""
Dataset Merging Module for PVGRIDBALANCER
Aligns "Winter Solar" and "Summer Wind" datasets into a unified 
24-hour simulation cycle through statistical signature extraction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


class DatasetMerger:
    """Merges and synchronizes solar and wind datasets for unified simulation"""
    
    def __init__(self, preprocessed_path='data/preprocessed'):
        self.preprocessed_path = Path(preprocessed_path)
    
    def load_cleaned_data(self):
        """Load all cleaned datasets"""
        solar_files = sorted(self.preprocessed_path.glob('solar_cleaned_*.csv'))
        wind_files = sorted(self.preprocessed_path.glob('wind_cleaned_*.csv'))
        
        solar_data = []
        for file in solar_files:
            df = pd.read_csv(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            solar_data.append(df)
        
        wind_data = []
        for file in wind_files:
            df = pd.read_csv(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            wind_data.append(df)
        
        print(f"Loaded {len(solar_data)} solar datasets and {len(wind_data)} wind datasets")
        return solar_data, wind_data
    
    def concatenate_datasets(self, dataframes, source_type):
        """Concatenate multiple datasets of same type"""
        if len(dataframes) == 0:
            return None
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        # Concatenate all dataframes
        combined = pd.concat(dataframes, ignore_index=True)
        
        # Sort by timestamp if available
        if 'timestamp' in combined.columns:
            combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        if 'timestamp' in combined.columns:
            combined = combined.drop_duplicates(subset=['timestamp'], keep='first')
        
        print(f"Combined {source_type}: {len(combined)} records")
        return combined
    
    def extract_statistical_signature(self, df, power_col):
        """Extract statistical patterns for temporal synchronization"""
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
            df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Hourly patterns
        hourly_stats = df.groupby('hour')[power_col].agg(['mean', 'std', 'max'])
        
        # Daily patterns
        if 'day_of_year' in df.columns:
            daily_stats = df.groupby('day_of_year')[power_col].agg(['mean', 'std'])
        else:
            daily_stats = None
        
        return {
            'hourly': hourly_stats,
            'daily': daily_stats,
            'overall_mean': df[power_col].mean(),
            'overall_std': df[power_col].std()
        }
    
    def create_unified_timeline(self, solar_df, wind_df):
        """Create unified 24-hour simulation timeline"""
        print("\n=== Creating Unified Timeline ===")
        
        # Ensure both have timestamps
        if 'timestamp' not in solar_df.columns or 'timestamp' not in wind_df.columns:
            print("⚠ Missing timestamp columns, using index-based merge")
            # Align by index
            min_len = min(len(solar_df), len(wind_df))
            solar_df = solar_df.iloc[:min_len].reset_index(drop=True)
            wind_df = wind_df.iloc[:min_len].reset_index(drop=True)
            
            merged = pd.concat([solar_df, wind_df], axis=1)
            return merged
        
        # Find overlapping time period
        solar_start, solar_end = solar_df['timestamp'].min(), solar_df['timestamp'].max()
        wind_start, wind_end = wind_df['timestamp'].min(), wind_df['timestamp'].max()
        
        common_start = max(solar_start, wind_start)
        common_end = min(solar_end, wind_end)
        
        print(f"Solar range: {solar_start} to {solar_end}")
        print(f"Wind range: {wind_start} to {wind_end}")
        print(f"Common range: {common_start} to {common_end}")
        
        # Filter to common period
        solar_filtered = solar_df[(solar_df['timestamp'] >= common_start) & 
                                   (solar_df['timestamp'] <= common_end)].copy()
        wind_filtered = wind_df[(wind_df['timestamp'] >= common_start) & 
                                 (wind_df['timestamp'] <= common_end)].copy()
        
        # Merge on timestamp
        merged = pd.merge(solar_filtered, wind_filtered, 
                         on='timestamp', how='inner', 
                         suffixes=('_solar', '_wind'))
        
        print(f"Merged dataset: {len(merged)} records")
        return merged
    
    def calculate_complementarity_index(self, merged_df):
        """Calculate complementarity between solar and wind generation"""
        # Find power columns
        solar_cols = [col for col in merged_df.columns if 'solar' in col.lower() and 
                     ('power' in col.lower() or 'generation' in col.lower())]
        wind_cols = [col for col in merged_df.columns if 'wind' in col.lower() and 
                    ('power' in col.lower() or 'generation' in col.lower())]
        
        if len(solar_cols) > 0 and len(wind_cols) > 0:
            solar_power = merged_df[solar_cols[0]]
            wind_power = merged_df[wind_cols[0]]
            
            # Normalize to 0-1 range
            solar_norm = (solar_power - solar_power.min()) / (solar_power.max() - solar_power.min() + 1e-6)
            wind_norm = (wind_power - wind_power.min()) / (wind_power.max() - wind_power.min() + 1e-6)
            
            # Complementarity: high when one is low and other is high
            merged_df['complementarity_index'] = 1 - np.abs(solar_norm - wind_norm)
            
            # Combined generation
            merged_df['total_renewable_power'] = solar_power + wind_power
            
            print(f"\nComplementarity Index: {merged_df['complementarity_index'].mean():.3f}")
            print(f"Total Renewable Power Range: [{merged_df['total_renewable_power'].min():.2f}, "
                  f"{merged_df['total_renewable_power'].max():.2f}]")
        
        return merged_df
    
    def save_merged_dataset(self, merged_df, filename='unified_renewable_data.csv'):
        """Save the merged dataset"""
        output_file = self.preprocessed_path / filename
        merged_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved merged dataset to {output_file}")
        return output_file
    
    def run(self):
        """Execute full merging pipeline"""
        print("Starting dataset merging pipeline...")
        
        # Load cleaned data
        solar_data, wind_data = self.load_cleaned_data()
        
        # Concatenate individual datasets
        print("\n=== Concatenating Datasets ===")
        solar_combined = self.concatenate_datasets(solar_data, "Solar")
        wind_combined = self.concatenate_datasets(wind_data, "Wind")
        
        if solar_combined is None or wind_combined is None:
            print("⚠ Missing data, cannot merge")
            return None
        
        # Create unified timeline
        merged = self.create_unified_timeline(solar_combined, wind_combined)
        
        # Calculate complementarity
        merged = self.calculate_complementarity_index(merged)
        
        # Save merged dataset
        output_file = self.save_merged_dataset(merged)
        
        print("\n✓ Dataset merging completed successfully!")
        print(f"Final dataset shape: {merged.shape}")
        
        return merged


if __name__ == '__main__':
    merger = DatasetMerger()
    merger.run()
=======
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path.cwd()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

pv_files = sorted(RAW_DIR.glob("*solar*.xlsx"))
wind_files = sorted(RAW_DIR.glob("*wind*.xlsx"))

# Define expected column names
PV_COLUMNS = [
    "Date - Time",
    "Time",
    "Temperature©",
    "Humidity",
    "Ground Radiation Intensity (W/m^2)",
    "Upper Atmosphere Radiation Intensity (W/m^2)",
    "PV Generation (KW)"
]

WIND_COLUMNS = [
    "Date-Time",
    "Time",
    "Air Density",
    "Wind Speed",
    "Power Generation"
]

pv_dfs = []

for file in pv_files:
    df = pd.read_excel(file)
    print(f"\nProcessing PV file: {file.name}")
    print(f"Available columns: {list(df.columns)}")
    
    # Try to handle different column name variations
    # First, strip any extra whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Check which PV columns are available
    available_columns = []
    for col in PV_COLUMNS:
        # Try exact match
        if col in df.columns:
            available_columns.append(col)
        # Try with different spacing variations
        elif col.replace(" - ", "-") in df.columns:  # "Date - Time" -> "Date-Time"
            available_columns.append(col.replace(" - ", "-"))
        elif col.replace("-", " - ") in df.columns:  # "Date-Time" -> "Date - Time"
            available_columns.append(col.replace("-", " - "))
        else:
            print(f"Warning: Column '{col}' not found in {file.name}")
    
    # Use only available columns
    df = df[available_columns]
    
    # Rename columns to standard format if needed
    column_mapping = {}
    for col in df.columns:
        if "Date" in col and "Time" in col:
            if "-" in col and " - " not in col:
                column_mapping[col] = "Date - Time"
            elif " - " in col:
                column_mapping[col] = "Date - Time"
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    pv_dfs.append(df)

if pv_dfs:
    pv_master = pd.concat(pv_dfs, ignore_index=True)
    print(f"\nPV Master dataset columns: {list(pv_master.columns)}")
else:
    print("No PV data processed!")

# -----------------------------
# Merge Wind files
# -----------------------------
wind_dfs = []

for file in wind_files:
    df = pd.read_excel(file)
    print(f"\nProcessing Wind file: {file.name}")
    print(f"Available columns: {list(df.columns)}")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Check which WIND columns are available
    available_columns = []
    for col in WIND_COLUMNS:
        if col in df.columns:
            available_columns.append(col)
        # Handle Date-Time variations
        elif "Date" in col and "Time" in col:
            # Check for common variations
            variations = ["Date-Time", "Date - Time", "DateTime", "Date_Time"]
            for var in variations:
                if var in df.columns:
                    available_columns.append(var)
                    # Map to standard name
                    df = df.rename(columns={var: "Date-Time"})
                    break
        else:
            print(f"Warning: Column '{col}' not found in {file.name}")
    
    # Use only available columns
    df = df[available_columns]
    
    wind_dfs.append(df)

if wind_dfs:
    wind_master = pd.concat(wind_dfs, ignore_index=True)
    print(f"\nWind Master dataset columns: {list(wind_master.columns)}")
else:
    print("No Wind data processed!")

# -----------------------------
# Save outputs
# -----------------------------
if pv_dfs:
    pv_output_path = PREPROCESSED_DIR / "pv_master_dataset.csv"
    pv_master.to_csv(pv_output_path, index=False)
    print(f"\nPV dataset saved to: {pv_output_path}")

if wind_dfs:
    wind_output_path = PREPROCESSED_DIR / "wind_master_dataset.csv"
    wind_master.to_csv(wind_output_path, index=False)
    print(f"Wind dataset saved to: {wind_output_path}")

print("\nProcessing complete!")
>>>>>>> 0ba306db939cd0c78f9380d603453b23345abac5
