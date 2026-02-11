# scripts/data_cleaning.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def clean_engineered_data():
    """Clean the engineered dataset for machine learning."""
    
    # Set up paths
    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
    
    print("=" * 60)
    print("DATA CLEANING FOR MACHINE LEARNING")
    print("=" * 60)
    
    # Load engineered datasets
    print("\n1. Loading engineered datasets...")
    pv = pd.read_csv(DATA_DIR / "pv_engineered.csv")
    wind = pd.read_csv(DATA_DIR / "wind_engineered.csv")
    merged = pd.read_csv(DATA_DIR / "merged_engineered.csv")
    
    print(f"   PV shape: {pv.shape}")
    print(f"   Wind shape: {wind.shape}")
    print(f"   Merged shape: {merged.shape}")
    
    # Check data types
    print("\n   Data types in PV dataset:")
    print(pv.dtypes.value_counts())
    
    print("\n   Data types in Wind dataset:")
    print(wind.dtypes.value_counts())
    
    # 1. Handle missing values
    print("\n2. Handling missing values...")
    
    # Check NaN counts before cleaning
    print(f"\n   NaN values before cleaning:")
    print(f"   PV: {pv.isna().sum().sum()}")
    print(f"   Wind: {wind.isna().sum().sum()}")
    print(f"   Merged: {merged.isna().sum().sum()}")
    
    # Show columns with most NaN values
    print(f"\n   Top 5 columns with most NaN values in PV:")
    pv_nan_counts = pv.isna().sum().sort_values(ascending=False)
    print(pv_nan_counts.head())
    
    print(f"\n   Top 5 columns with most NaN values in Wind:")
    wind_nan_counts = wind.isna().sum().sort_values(ascending=False)
    print(wind_nan_counts.head())
    
    # Strategy for handling NaN values
    print("\n3. Applying cleaning strategies...")
    
    # For PV dataset
    print("\n   Cleaning PV dataset...")
    pv_clean = pv.copy()
    
    # First, identify non-numeric columns
    non_numeric_cols = pv_clean.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"   Non-numeric columns: {list(non_numeric_cols)}")
    
    # Check why we have so many NaN in efficiency
    print("\n   Analyzing PV efficiency NaN causes...")
    if 'pv_efficiency' in pv_clean.columns:
        # Check if it's due to division by zero
        zero_radiation = (pv_clean['Ground Radiation Intensity (W/m^2)'] == 0).sum()
        print(f"   Rows with zero ground radiation: {zero_radiation}")
        
        # Also check pv_per_wm2
        if 'pv_per_wm2' in pv_clean.columns:
            zero_pv_gen = (pv_clean['PV Generation (KW)'] == 0).sum()
            print(f"   Rows with zero PV generation: {zero_pv_gen}")
    
    # Handle NaN values with better strategies
    
    # 1. For temporal features (lag, rolling, diff, ramp)
    lag_cols = [col for col in pv_clean.columns if any(x in col.lower() for x in ['lag', 'roll', 'diff', 'ramp'])]
    print(f"\n   Handling {len(lag_cols)} temporal features...")
    
    for col in lag_cols:
        if col in pv_clean.columns and pv_clean[col].dtype in [np.float64, np.float32, np.int64]:
            # Forward fill (carry last observation forward)
            pv_clean[col] = pv_clean[col].ffill()
            # Backward fill for any remaining NaN at beginning
            pv_clean[col] = pv_clean[col].bfill()
            print(f"     {col}: filled {pv_clean[col].isna().sum()} NaN values")
    
    # 2. For efficiency and ratio features
    eff_cols = [col for col in pv_clean.columns if any(x in col.lower() for x in ['efficiency', 'per_wm2', 'ratio', 'per_'])]
    print(f"\n   Handling {len(eff_cols)} efficiency/ratio features...")
    
    for col in eff_cols:
        if col in pv_clean.columns and pv_clean[col].dtype in [np.float64, np.float32]:
            # Fill with median (more robust than mean)
            median_val = pv_clean[col].median(skipna=True)
            pv_clean[col] = pv_clean[col].fillna(median_val)
            print(f"     {col}: filled with median {median_val:.4f}")
    
    # 3. Fill remaining numeric columns with column median
    print("\n   Handling remaining numeric columns...")
    numeric_cols = pv_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if pv_clean[col].isna().any():
            median_val = pv_clean[col].median(skipna=True)
            pv_clean[col] = pv_clean[col].fillna(median_val)
    
    # 4. For non-numeric columns (if any), fill with mode
    non_numeric_cols = pv_clean.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if pv_clean[col].isna().any():
            mode_val = pv_clean[col].mode()[0] if not pv_clean[col].mode().empty else ''
            pv_clean[col] = pv_clean[col].fillna(mode_val)
    
    print(f"   PV NaN after cleaning: {pv_clean.isna().sum().sum()}")
    
    # For Wind dataset
    print("\n   Cleaning Wind dataset...")
    wind_clean = wind.copy()
    
    # Handle NaN values
    # 1. Temporal features
    wind_lag_cols = [col for col in wind_clean.columns if any(x in col.lower() for x in ['lag', 'roll', 'ramp', 'diff'])]
    for col in wind_lag_cols:
        if col in wind_clean.columns and wind_clean[col].dtype in [np.float64, np.float32, np.int64]:
            wind_clean[col] = wind_clean[col].ffill().bfill()
    
    # 2. Special features
    wind_special_cols = [col for col in wind_clean.columns if any(x in col.lower() for x in ['turbulence', 'intensity', 'per_ws', 'density'])]
    for col in wind_special_cols:
        if col in wind_clean.columns and wind_clean[col].dtype in [np.float64, np.float32]:
            median_val = wind_clean[col].median(skipna=True)
            wind_clean[col] = wind_clean[col].fillna(median_val)
    
    # 3. Fill remaining numeric columns
    numeric_cols = wind_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if wind_clean[col].isna().any():
            median_val = wind_clean[col].median(skipna=True)
            wind_clean[col] = wind_clean[col].fillna(median_val)
    
    # 4. Non-numeric columns
    non_numeric_cols = wind_clean.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if wind_clean[col].isna().any():
            mode_val = wind_clean[col].mode()[0] if not wind_clean[col].mode().empty else ''
            wind_clean[col] = wind_clean[col].fillna(mode_val)
    
    print(f"   Wind NaN after cleaning: {wind_clean.isna().sum().sum()}")
    
    # For Merged dataset
    print("\n   Cleaning Merged dataset...")
    merged_clean = merged.copy()
    
    # Apply similar strategies
    # 1. Temporal features
    temp_cols = [col for col in merged_clean.columns if any(x in col.lower() for x in ['lag', 'roll', 'ramp', 'diff'])]
    for col in temp_cols:
        if col in merged_clean.columns and merged_clean[col].dtype in [np.float64, np.float32, np.int64]:
            merged_clean[col] = merged_clean[col].ffill().bfill()
    
    # 2. Efficiency/ratio features
    ratio_cols = [col for col in merged_clean.columns if any(x in col.lower() for x in ['efficiency', 'ratio', 'per_', 'intensity'])]
    for col in ratio_cols:
        if col in merged_clean.columns and merged_clean[col].dtype in [np.float64, np.float32]:
            median_val = merged_clean[col].median(skipna=True)
            merged_clean[col] = merged_clean[col].fillna(median_val)
    
    # 3. Fill remaining numeric columns
    numeric_cols = merged_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if merged_clean[col].isna().any():
            median_val = merged_clean[col].median(skipna=True)
            merged_clean[col] = merged_clean[col].fillna(median_val)
    
    # 4. Non-numeric columns
    non_numeric_cols = merged_clean.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if merged_clean[col].isna().any():
            mode_val = merged_clean[col].mode()[0] if not merged_clean[col].mode().empty else ''
            merged_clean[col] = merged_clean[col].fillna(mode_val)
    
    print(f"   Merged NaN after cleaning: {merged_clean.isna().sum().sum()}")
    
    # 4. Check for infinite values
    print("\n4. Checking for infinite values...")
    
    def count_inf(df, df_name):
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        print(f"   {df_name}: {inf_count} infinite values")
        return inf_count
    
    pv_inf = count_inf(pv_clean, "PV")
    wind_inf = count_inf(wind_clean, "Wind")
    merged_inf = count_inf(merged_clean, "Merged")
    
    # Replace infinite values
    if pv_inf > 0:
        pv_clean = pv_clean.replace([np.inf, -np.inf], np.nan)
        # Fill with median
        numeric_cols = pv_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if pv_clean[col].isna().any():
                median_val = pv_clean[col].median(skipna=True)
                pv_clean[col] = pv_clean[col].fillna(median_val)
    
    if wind_inf > 0:
        wind_clean = wind_clean.replace([np.inf, -np.inf], np.nan)
        numeric_cols = wind_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if wind_clean[col].isna().any():
                median_val = wind_clean[col].median(skipna=True)
                wind_clean[col] = wind_clean[col].fillna(median_val)
    
    if merged_inf > 0:
        merged_clean = merged_clean.replace([np.inf, -np.inf], np.nan)
        numeric_cols = merged_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if merged_clean[col].isna().any():
                median_val = merged_clean[col].median(skipna=True)
                merged_clean[col] = merged_clean[col].fillna(median_val)
    
    # 5. Save cleaned datasets
    print("\n5. Saving cleaned datasets...")
    pv_clean.to_csv(DATA_DIR / "pv_cleaned.csv", index=False)
    wind_clean.to_csv(DATA_DIR / "wind_cleaned.csv", index=False)
    merged_clean.to_csv(DATA_DIR / "merged_cleaned.csv", index=False)
    
    print(f"   ✓ PV cleaned saved to: {DATA_DIR / 'pv_cleaned.csv'}")
    print(f"   ✓ Wind cleaned saved to: {DATA_DIR / 'wind_cleaned.csv'}")
    print(f"   ✓ Merged cleaned saved to: {DATA_DIR / 'merged_cleaned.csv'}")
    
    # 6. Create ML-ready datasets (scaled)
    print("\n6. Creating ML-ready datasets (scaled)...")
    
    # Separate features and target for PV
    # First ensure timestamp is properly handled
    timestamp_cols = [col for col in pv_clean.columns if 'time' in col.lower() or 'date' in col.lower()]
    timestamp_col = timestamp_cols[0] if timestamp_cols else None
    
    pv_features_cols = [col for col in pv_clean.columns if col not in ['PV Generation (KW)'] and col != timestamp_col]
    pv_features = pv_clean[pv_features_cols]
    pv_target = pv_clean['PV Generation (KW)']
    
    # Separate features and target for Wind
    wind_timestamp_cols = [col for col in wind_clean.columns if 'time' in col.lower() or 'date' in col.lower()]
    wind_timestamp_col = wind_timestamp_cols[0] if wind_timestamp_cols else None
    
    wind_features_cols = [col for col in wind_clean.columns if col not in ['Power Generation'] and col != wind_timestamp_col]
    wind_features = wind_clean[wind_features_cols]
    wind_target = wind_clean['Power Generation']
    
    # For merged dataset
    merged_timestamp_cols = [col for col in merged_clean.columns if 'time' in col.lower() or 'date' in col.lower()]
    merged_timestamp_col = merged_timestamp_cols[0] if merged_timestamp_cols else None
    
    merged_features_cols = [col for col in merged_clean.columns if col not in ['PV Generation (KW)', 'Power Generation'] and col != merged_timestamp_col]
    merged_features = merged_clean[merged_features_cols]
    merged_target_pv = merged_clean['PV Generation (KW)']
    merged_target_wind = merged_clean['Power Generation']
    
    # Check for any non-numeric columns remaining
    print("\n   Checking for non-numeric columns in features...")
    for df_name, df in [("PV", pv_features), ("Wind", wind_features), ("Merged", merged_features)]:
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"   {df_name} has non-numeric columns: {list(non_numeric)}")
            # Convert to numeric if possible
            for col in non_numeric:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"     Converted {col} to numeric")
                except:
                    print(f"     Could not convert {col}")
    
    # Scale features (only numeric columns)
    scaler = StandardScaler()
    
    # PV scaling
    pv_numeric_cols = pv_features.select_dtypes(include=[np.number]).columns
    pv_features_numeric = pv_features[pv_numeric_cols]
    pv_features_scaled_numeric = pd.DataFrame(
        scaler.fit_transform(pv_features_numeric),
        columns=pv_numeric_cols,
        index=pv_features_numeric.index
    )
    
    # Combine scaled numeric with any non-numeric columns
    pv_features_non_numeric = pv_features.select_dtypes(exclude=[np.number])
    pv_features_scaled = pd.concat([pv_features_scaled_numeric, pv_features_non_numeric], axis=1)
    
    # Wind scaling
    wind_numeric_cols = wind_features.select_dtypes(include=[np.number]).columns
    wind_features_numeric = wind_features[wind_numeric_cols]
    wind_features_scaled_numeric = pd.DataFrame(
        scaler.fit_transform(wind_features_numeric),
        columns=wind_numeric_cols,
        index=wind_features_numeric.index
    )
    
    wind_features_non_numeric = wind_features.select_dtypes(exclude=[np.number])
    wind_features_scaled = pd.concat([wind_features_scaled_numeric, wind_features_non_numeric], axis=1)
    
    # Merged scaling
    merged_numeric_cols = merged_features.select_dtypes(include=[np.number]).columns
    merged_features_numeric = merged_features[merged_numeric_cols]
    merged_features_scaled_numeric = pd.DataFrame(
        scaler.fit_transform(merged_features_numeric),
        columns=merged_numeric_cols,
        index=merged_features_numeric.index
    )
    
    merged_features_non_numeric = merged_features.select_dtypes(exclude=[np.number])
    merged_features_scaled = pd.concat([merged_features_scaled_numeric, merged_features_non_numeric], axis=1)
    
    # Create final ML datasets
    pv_ml = pd.concat([pv_features_scaled, pv_target.reset_index(drop=True)], axis=1)
    wind_ml = pd.concat([wind_features_scaled, wind_target.reset_index(drop=True)], axis=1)
    merged_ml = pd.concat([
        merged_features_scaled, 
        merged_target_pv.reset_index(drop=True),
        merged_target_wind.reset_index(drop=True)
    ], axis=1)
    
    # Save ML-ready datasets
    pv_ml.to_csv(DATA_DIR / "pv_ml_ready.csv", index=False)
    wind_ml.to_csv(DATA_DIR / "wind_ml_ready.csv", index=False)
    merged_ml.to_csv(DATA_DIR / "merged_ml_ready.csv", index=False)
    
    print(f"   ✓ PV ML-ready saved to: {DATA_DIR / 'pv_ml_ready.csv'}")
    print(f"   ✓ Wind ML-ready saved to: {DATA_DIR / 'wind_ml_ready.csv'}")
    print(f"   ✓ Merged ML-ready saved to: {DATA_DIR / 'merged_ml_ready.csv'}")
    
    # 7. Summary statistics
    print("\n7. Final Dataset Statistics")
    print("-" * 40)
    
    print(f"\n   PV Dataset:")
    print(f"     Total features: {len(pv_features.columns)}")
    print(f"     Numeric features: {len(pv_numeric_cols)}")
    print(f"     Non-numeric features: {len(pv_features_non_numeric.columns)}")
    print(f"     Samples: {len(pv_ml)}")
    print(f"     Target range: [{pv_target.min():.2f}, {pv_target.max():.2f}]")
    print(f"     Target mean: {pv_target.mean():.2f}")
    
    print(f"\n   Wind Dataset:")
    print(f"     Total features: {len(wind_features.columns)}")
    print(f"     Numeric features: {len(wind_numeric_cols)}")
    print(f"     Non-numeric features: {len(wind_features_non_numeric.columns)}")
    print(f"     Samples: {len(wind_ml)}")
    print(f"     Target range: [{wind_target.min():.2f}, {wind_target.max():.2f}]")
    print(f"     Target mean: {wind_target.mean():.2f}")
    
    print(f"\n   Merged Dataset:")
    print(f"     Total features: {len(merged_features.columns)}")
    print(f"     Numeric features: {len(merged_numeric_cols)}")
    print(f"     Non-numeric features: {len(merged_features_non_numeric.columns)}")
    print(f"     Samples: {len(merged_ml)}")
    print(f"     PV Target range: [{merged_target_pv.min():.2f}, {merged_target_pv.max():.2f}]")
    print(f"     Wind Target range: [{merged_target_wind.min():.2f}, {merged_target_wind.max():.2f}]")
    
    # 8. Feature importance analysis (correlation)
    print("\n8. Feature Correlation Analysis")
    print("-" * 40)
    
    # For PV - only consider numeric columns for correlation
    if len(pv_numeric_cols) > 0:
        print(f"\n   Top 5 features correlated with PV Generation:")
        pv_corr_df = pd.concat([pv_features[pv_numeric_cols], pv_target], axis=1)
        pv_corr = pv_corr_df.corr()['PV Generation (KW)'].abs().sort_values(ascending=False)
        print(pv_corr.head(6))  # Include target itself
    
    # For Wind
    if len(wind_numeric_cols) > 0:
        print(f"\n   Top 5 features correlated with Wind Power Generation:")
        wind_corr_df = pd.concat([wind_features[wind_numeric_cols], wind_target], axis=1)
        wind_corr = wind_corr_df.corr()['Power Generation'].abs().sort_values(ascending=False)
        print(wind_corr.head(6))
    
    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE!")
    print("=" * 60)
    
    return pv_clean, wind_clean, merged_clean

if __name__ == "__main__":
    # Run cleaning
    pv_clean, wind_clean, merged_clean = clean_engineered_data()