import pandas as pd
import numpy as np
from pathlib import Path

def add_time_features(df, datetime_col="Date-Time"):
    """Add temporal features to the dataframe."""
    # Try different datetime column names
    possible_names = ["Date-Time", "Date - Time", "datetime", "timestamp"]
    
    actual_col = None
    for name in possible_names:
        if name in df.columns:
            actual_col = name
            break
    
    if actual_col is None:
        # Try to find any column with 'date' or 'time' in name
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                actual_col = col
                break
    
    if actual_col is None:
        raise KeyError(f"No datetime column found. Available columns: {list(df.columns)}")
    
    print(f"  Using datetime column: '{actual_col}'")
    df[actual_col] = pd.to_datetime(df[actual_col])
    df["hour"] = df[actual_col].dt.hour
    df["day"] = df[actual_col].dt.day
    df["month"] = df[actual_col].dt.month
    df["weekday"] = df[actual_col].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Cyclical encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df, actual_col  # Return the actual column name used

def engineer_pv_features(df):
    """Engineer features for PV dataset."""
    print("  Adding time features...")
    df, datetime_col = add_time_features(df)
    
    print("  Engineering radiation features...")
    # Radiation-based features
    df["radiation_ratio"] = (
        df["Ground Radiation Intensity (W/m^2)"] /
        df["Upper Atmosphere Radiation Intensity (W/m^2)"]
    ).replace([np.inf, -np.inf], np.nan)

    df["radiation_change"] = df["Ground Radiation Intensity (W/m^2)"].diff()

    print("  Engineering temperature features...")
    # Temperature effects
    df["temp_squared"] = df["Temperature©"] ** 2
    df["temp_derating"] = np.maximum(0, df["Temperature©"] - 25)

    print("  Creating lag features...")
    # Lag features
    df["pv_lag_1"] = df["PV Generation (KW)"].shift(1)
    df["pv_lag_3"] = df["PV Generation (KW)"].shift(3)

    print("  Creating rolling statistics...")
    # Rolling statistics
    df["pv_roll_mean_3"] = df["PV Generation (KW)"].rolling(3, min_periods=1).mean()
    df["pv_roll_std_3"] = df["PV Generation (KW)"].rolling(3, min_periods=1).std()

    print("  Creating ramp rate...")
    # Ramp rate
    df["pv_ramp_rate"] = df["PV Generation (KW)"].diff()

    print("  Calculating efficiency metric...")
    # Efficiency metric
    df["pv_per_wm2"] = (
        df["PV Generation (KW)"] /
        df["Ground Radiation Intensity (W/m^2)"]
    ).replace([np.inf, -np.inf], np.nan)
    
    # Also add a more reasonable efficiency metric (capped)
    df["pv_efficiency"] = np.clip(df["pv_per_wm2"], 0, 0.3)

    return df

def engineer_wind_features(df):
    """Engineer features for wind dataset."""
    print("  Adding time features...")
    df, datetime_col = add_time_features(df)

    print("  Engineering physics-based features...")
    # Physics-based features
    df["wind_speed_cubed"] = df["Wind Speed"] ** 3
    df["wind_power_density"] = (
        0.5 * df["Air Density"] * df["Wind Speed"] ** 3
    )

    print("  Creating lag features...")
    # Lag features
    df["wind_lag_1"] = df["Power Generation"].shift(1)

    print("  Creating rolling features...")
    # Rolling features
    df["wind_roll_mean_3"] = df["Power Generation"].rolling(3, min_periods=1).mean()
    df["wind_roll_std_3"] = df["Power Generation"].rolling(3, min_periods=1).std()

    print("  Creating ramp rate...")
    # Ramp rate
    df["wind_ramp"] = df["Power Generation"].diff()

    print("  Calculating turbulence intensity...")
    # Turbulence proxy
    df["turbulence_intensity"] = (
        df["wind_roll_std_3"] / df["wind_roll_mean_3"]
    ).replace([np.inf, -np.inf], np.nan)

    print("  Calculating power efficiency...")
    # Efficiency
    df["power_per_ws"] = (
        df["Power Generation"] / df["Wind Speed"]
    ).replace([np.inf, -np.inf], np.nan)

    return df

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
def main():
    # Set up paths
    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
    
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load validated datasets
    print("\n1. Loading datasets...")
    pv = pd.read_csv(DATA_DIR / "pv_validated.csv")
    wind = pd.read_csv(DATA_DIR / "wind_validated.csv")
    
    print(f"   PV dataset shape: {pv.shape}")
    print(f"   PV columns: {list(pv.columns)}")
    print(f"   Wind dataset shape: {wind.shape}")
    print(f"   Wind columns: {list(wind.columns)}")
    
    # Apply feature engineering
    print("\n2. Engineering PV features...")
    pv_featured = engineer_pv_features(pv)
    print(f"   PV featured shape: {pv_featured.shape}")
    print(f"   PV total features: {len(pv_featured.columns)}")
    
    print("\n3. Engineering wind features...")
    wind_featured = engineer_wind_features(wind)
    print(f"   Wind featured shape: {wind_featured.shape}")
    print(f"   Wind total features: {len(wind_featured.columns)}")
    
    # Save engineered datasets
    print("\n4. Saving engineered datasets...")
    
    # Handle datetime column name for PV
    pv_datetime_col = None
    for col in ['Date-Time', 'Date - Time', 'datetime', 'timestamp']:
        if col in pv_featured.columns:
            pv_datetime_col = col
            break
    
    if pv_datetime_col:
        pv_featured = pv_featured.rename(columns={pv_datetime_col: "timestamp"})
    
    # Handle datetime column name for Wind
    wind_datetime_col = None
    for col in ['Date-Time', 'Date - Time', 'datetime', 'timestamp']:
        if col in wind_featured.columns:
            wind_datetime_col = col
            break
    
    if wind_datetime_col:
        wind_featured = wind_featured.rename(columns={wind_datetime_col: "timestamp"})
    
    pv_featured.to_csv(DATA_DIR / "pv_engineered.csv", index=False)
    wind_featured.to_csv(DATA_DIR / "wind_engineered.csv", index=False)
    
    print(f"   ✓ PV engineered saved to: {DATA_DIR / 'pv_engineered.csv'}")
    print(f"   ✓ Wind engineered saved to: {DATA_DIR / 'wind_engineered.csv'}")
    
    # Show feature summary
    print("\n5. Feature Summary")
    print("-" * 40)
    
    # PV features
    pv_original = set(pv.columns)
    pv_new = set(pv_featured.columns) - pv_original
    print(f"\n   PV New Features ({len(pv_new)}):")
    for i, feature in enumerate(sorted(pv_new), 1):
        print(f"     {i:2}. {feature}")
    
    # Wind features
    wind_original = set(wind.columns)
    wind_new = set(wind_featured.columns) - wind_original
    print(f"\n   Wind New Features ({len(wind_new)}):")
    for i, feature in enumerate(sorted(wind_new), 1):
        print(f"     {i:2}. {feature}")
    
    # Create merged dataset
    print("\n6. Creating merged dataset...")
    if 'timestamp' in pv_featured.columns and 'timestamp' in wind_featured.columns:
        # Ensure timestamps are datetime type
        pv_featured['timestamp'] = pd.to_datetime(pv_featured['timestamp'])
        wind_featured['timestamp'] = pd.to_datetime(wind_featured['timestamp'])
        
        # Merge datasets
        merged = pd.merge(
            pv_featured, 
            wind_featured, 
            on="timestamp", 
            how="inner",
            suffixes=("_pv", "_wind")
        )
        
        print(f"   Merged dataset shape: {merged.shape}")
        print(f"   Merged dataset columns: {len(merged.columns)}")
        
        # Save merged dataset
        merged.to_csv(DATA_DIR / "merged_engineered.csv", index=False)
        print(f"   ✓ Merged dataset saved to: {DATA_DIR / 'merged_engineered.csv'}")
        
        # Show merged sample
        print("\n   First 3 rows of merged dataset:")
        print(merged[['timestamp', 'PV Generation (KW)', 'Power Generation']].head(3))
    else:
        print("   ✗ Could not merge - timestamp columns not found")
        print(f"   PV columns: {list(pv_featured.columns)}")
        print(f"   Wind columns: {list(wind_featured.columns)}")
    
    # Quick statistics
    print("\n7. Quick Statistics")
    print("-" * 40)
    
    # Check for NaN values
    pv_nan = pv_featured.isna().sum().sum()
    wind_nan = wind_featured.isna().sum().sum()
    print(f"   PV NaN values: {pv_nan}")
    print(f"   Wind NaN values: {wind_nan}")
    
    # Check efficiency ranges
    if 'pv_efficiency' in pv_featured.columns:
        pv_eff_stats = pv_featured['pv_efficiency'].describe()
        print(f"\n   PV Efficiency Stats:")
        print(f"     Mean: {pv_eff_stats['mean']:.4f}")
        print(f"     Std: {pv_eff_stats['std']:.4f}")
        print(f"     Min: {pv_eff_stats['min']:.4f}")
        print(f"     Max: {pv_eff_stats['max']:.4f}")
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()