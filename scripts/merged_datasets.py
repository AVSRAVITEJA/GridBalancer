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