<<<<<<< HEAD
"""
Feature Engineering Module for PVGRIDBALANCER
Physics-based feature extraction including:
- Complementarity Index
- Power Curves
- Clearness Index (Kt)
- Wind Power Coefficient (Cp)
- Temporal features
"""

=======
>>>>>>> 0ba306db939cd0c78f9380d603453b23345abac5
import pandas as pd
import numpy as np
from pathlib import Path

<<<<<<< HEAD

class FeatureEngineer:
    """Extracts physics-based features for renewable energy forecasting"""
    
    def __init__(self, preprocessed_path='data/preprocessed'):
        self.preprocessed_path = Path(preprocessed_path)
        
        # Physical constants
        self.SOLAR_CONSTANT = 1367  # W/m^2 (extraterrestrial solar radiation)
        self.AIR_DENSITY_STD = 1.225  # kg/m^3 at sea level
        self.WIND_TURBINE_AREA = 100  # m^2 (assumed rotor swept area)
    
    def load_unified_data(self):
        """Load the unified renewable dataset"""
        file_path = self.preprocessed_path / 'unified_renewable_data.csv'
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Loaded unified dataset: {df.shape}")
        return df
    
    def calculate_clearness_index(self, df):
        """
        Calculate Clearness Index (Kt) for solar generation
        Kt = GHI / Extraterrestrial_Radiation
        Indicates atmospheric transparency
        """
        print("\n=== Calculating Clearness Index (Kt) ===")
        
        # Find radiation columns
        ghi_col = None
        for col in df.columns:
            if 'ground' in col.lower() and 'radiation' in col.lower():
                ghi_col = col
                break
        
        if ghi_col and 'hour' in df.columns:
            # Calculate extraterrestrial radiation based on hour
            # Simplified model: peak at solar noon (hour 12)
            hour = df['hour'].fillna(12)
            
            # Solar elevation angle approximation
            solar_elevation = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
            extraterrestrial_rad = self.SOLAR_CONSTANT * solar_elevation
            
            # Clearness index
            df['clearness_index_kt'] = np.where(
                extraterrestrial_rad > 0,
                df[ghi_col] / (extraterrestrial_rad + 1e-6),
                0
            )
            
            # Clip to valid range [0, 1]
            df['clearness_index_kt'] = df['clearness_index_kt'].clip(0, 1)
            
            print(f"  Kt range: [{df['clearness_index_kt'].min():.3f}, {df['clearness_index_kt'].max():.3f}]")
            print(f"  Kt mean: {df['clearness_index_kt'].mean():.3f}")
        
        return df
    
    def calculate_wind_power_coefficient(self, df):
        """
        Calculate Wind Power Coefficient (Cp)
        Theoretical: P_wind = 0.5 * ρ * A * v^3 * Cp
        Cp = P_actual / (0.5 * ρ * A * v^3)
        """
        print("\n=== Calculating Wind Power Coefficient (Cp) ===")
        
        # Find wind speed and power columns
        wind_speed_col = None
        wind_power_col = None
        
        for col in df.columns:
            if 'wind' in col.lower() and 'speed' in col.lower():
                wind_speed_col = col
            if 'power' in col.lower() and 'generation' in col.lower():
                wind_power_col = col
        
        if wind_speed_col and wind_power_col and 'Air Density' in df.columns:
            # Theoretical maximum power
            wind_speed = df[wind_speed_col]
            air_density = df['Air Density']
            
            # P_theoretical = 0.5 * ρ * A * v^3 (in Watts)
            p_theoretical = 0.5 * air_density * self.WIND_TURBINE_AREA * (wind_speed ** 3)
            
            # Convert to kW
            p_theoretical_kw = p_theoretical / 1000
            
            # Calculate Cp
            df['wind_power_coefficient_cp'] = np.where(
                p_theoretical_kw > 0.1,
                df[wind_power_col] / (p_theoretical_kw + 1e-6),
                0
            )
            
            # Clip to Betz limit (0.593) and physical range
            df['wind_power_coefficient_cp'] = df['wind_power_coefficient_cp'].clip(0, 0.593)
            
            print(f"  Cp range: [{df['wind_power_coefficient_cp'].min():.3f}, {df['wind_power_coefficient_cp'].max():.3f}]")
            print(f"  Cp mean: {df['wind_power_coefficient_cp'].mean():.3f}")
        
        return df
    
    def calculate_complementarity_index(self, df):
        """
        Enhanced Complementarity Index
        Measures how well solar and wind complement each other
        High when one is high and other is low
        """
        print("\n=== Calculating Enhanced Complementarity Index ===")
        
        # Find power columns
        solar_col = None
        wind_col = None
        
        for col in df.columns:
            if 'pv' in col.lower() and 'generation' in col.lower():
                solar_col = col
            if 'power' in col.lower() and 'generation' in col.lower() and 'pv' not in col.lower():
                wind_col = col
        
        if solar_col and wind_col:
            # Normalize to [0, 1]
            solar_norm = (df[solar_col] - df[solar_col].min()) / (df[solar_col].max() - df[solar_col].min() + 1e-6)
            wind_norm = (df[wind_col] - df[wind_col].min()) / (df[wind_col].max() - df[wind_col].min() + 1e-6)
            
            # Complementarity: 1 - |solar - wind|
            df['complementarity_index'] = 1 - np.abs(solar_norm - wind_norm)
            
            # Total renewable power
            df['total_renewable_power'] = df[solar_col] + df[wind_col]
            
            # Renewable capacity factor
            max_capacity = df[solar_col].max() + df[wind_col].max()
            df['renewable_capacity_factor'] = df['total_renewable_power'] / (max_capacity + 1e-6)
            
            print(f"  Complementarity range: [{df['complementarity_index'].min():.3f}, {df['complementarity_index'].max():.3f}]")
            print(f"  Complementarity mean: {df['complementarity_index'].mean():.3f}")
            print(f"  Total power range: [{df['total_renewable_power'].min():.2f}, {df['total_renewable_power'].max():.2f}] kW")
        
        return df
    
    def calculate_power_curves(self, df):
        """
        Calculate normalized power curves for solar and wind
        """
        print("\n=== Calculating Power Curves ===")
        
        # Solar power curve (normalized by radiation)
        ghi_col = None
        solar_col = None
        
        for col in df.columns:
            if 'ground' in col.lower() and 'radiation' in col.lower():
                ghi_col = col
            if 'pv' in col.lower() and 'generation' in col.lower():
                solar_col = col
        
        if ghi_col and solar_col:
            # Solar efficiency curve
            df['solar_efficiency'] = np.where(
                df[ghi_col] > 10,
                df[solar_col] / (df[ghi_col] + 1e-6),
                0
            )
            df['solar_efficiency'] = df['solar_efficiency'].clip(0, 1)
            
            print(f"  Solar efficiency range: [{df['solar_efficiency'].min():.3f}, {df['solar_efficiency'].max():.3f}]")
        
        # Wind power curve (normalized by wind speed)
        wind_speed_col = None
        wind_power_col = None
        
        for col in df.columns:
            if 'wind' in col.lower() and 'speed' in col.lower():
                wind_speed_col = col
            if 'power' in col.lower() and 'generation' in col.lower():
                wind_power_col = col
        
        if wind_speed_col and wind_power_col:
            # Wind power per unit speed
            df['wind_power_per_speed'] = np.where(
                df[wind_speed_col] > 0.5,
                df[wind_power_col] / (df[wind_speed_col] + 1e-6),
                0
            )
            
            print(f"  Wind power/speed range: [{df['wind_power_per_speed'].min():.2f}, {df['wind_power_per_speed'].max():.2f}]")
        
        return df
    
    def add_temporal_features(self, df):
        """
        Add temporal features for time-series forecasting
        """
        print("\n=== Adding Temporal Features ===")
        
        if 'timestamp' in df.columns:
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['month'] = df['timestamp'].dt.month
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week
            
            # Cyclical encoding for hour
            if 'hour' not in df.columns:
                df['hour'] = df['timestamp'].dt.hour
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Cyclical encoding for day of year
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            
            print(f"  Added temporal features: day_of_week, month, hour_sin/cos, day_sin/cos")
        
        return df
    
    def add_lag_features(self, df, lags=[1, 2, 3, 6, 12, 24]):
        """
        Add lagged features for time-series prediction
        """
        print(f"\n=== Adding Lag Features (lags: {lags}) ===")
        
        # Find power columns
        solar_col = None
        wind_col = None
        
        for col in df.columns:
            if 'pv' in col.lower() and 'generation' in col.lower():
                solar_col = col
            if 'power' in col.lower() and 'generation' in col.lower() and 'pv' not in col.lower():
                wind_col = col
        
        # Add lags for solar
        if solar_col:
            for lag in lags:
                df[f'solar_lag_{lag}'] = df[solar_col].shift(lag)
        
        # Add lags for wind
        if wind_col:
            for lag in lags:
                df[f'wind_lag_{lag}'] = df[wind_col].shift(lag)
        
        # Add rolling statistics
        if solar_col:
            df['solar_rolling_mean_6h'] = df[solar_col].rolling(window=6, min_periods=1).mean()
            df['solar_rolling_std_6h'] = df[solar_col].rolling(window=6, min_periods=1).std()
        
        if wind_col:
            df['wind_rolling_mean_6h'] = df[wind_col].rolling(window=6, min_periods=1).mean()
            df['wind_rolling_std_6h'] = df[wind_col].rolling(window=6, min_periods=1).std()
        
        print(f"  Added {len(lags)*2} lag features and 4 rolling statistics")
        
        return df
    
    def add_volatility_features(self, df):
        """
        Calculate generation volatility (rate of change)
        Important for grid stability analysis
        """
        print("\n=== Calculating Volatility Features ===")
        
        # Find power columns
        solar_col = None
        wind_col = None
        
        for col in df.columns:
            if 'pv' in col.lower() and 'generation' in col.lower():
                solar_col = col
            if 'power' in col.lower() and 'generation' in col.lower() and 'pv' not in col.lower():
                wind_col = col
        
        if solar_col:
            # Rate of change (dP/dt)
            df['solar_rate_of_change'] = df[solar_col].diff()
            df['solar_volatility'] = df['solar_rate_of_change'].abs()
            
            print(f"  Solar volatility range: [{df['solar_volatility'].min():.2f}, {df['solar_volatility'].max():.2f}] kW/h")
        
        if wind_col:
            # Rate of change (dP/dt)
            df['wind_rate_of_change'] = df[wind_col].diff()
            df['wind_volatility'] = df['wind_rate_of_change'].abs()
            
            print(f"  Wind volatility range: [{df['wind_volatility'].min():.2f}, {df['wind_volatility'].max():.2f}] kW/h")
        
        # Combined volatility
        if 'total_renewable_power' in df.columns:
            df['total_rate_of_change'] = df['total_renewable_power'].diff()
            df['total_volatility'] = df['total_rate_of_change'].abs()
            
            print(f"  Total volatility range: [{df['total_volatility'].min():.2f}, {df['total_volatility'].max():.2f}] kW/h")
        
        return df
    
    def save_engineered_features(self, df, filename='engineered_features.csv'):
        """Save the feature-engineered dataset"""
        output_file = self.preprocessed_path / filename
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved engineered features to {output_file}")
        print(f"  Final shape: {df.shape}")
        print(f"  Total features: {len(df.columns)}")
        return output_file
    
    def run(self):
        """Execute full feature engineering pipeline"""
        print("="*60)
        print("Starting Feature Engineering Pipeline")
        print("="*60)
        
        # Load data
        df = self.load_unified_data()
        
        # Physics-based features
        df = self.calculate_clearness_index(df)
        df = self.calculate_wind_power_coefficient(df)
        df = self.calculate_complementarity_index(df)
        df = self.calculate_power_curves(df)
        
        # Temporal features
        df = self.add_temporal_features(df)
        
        # Time-series features
        df = self.add_lag_features(df)
        df = self.add_volatility_features(df)
        
        # Save results
        output_file = self.save_engineered_features(df)
        
        print("\n" + "="*60)
        print("✓ Feature Engineering Completed Successfully!")
        print("="*60)
        print("\nKey Features Created:")
        print("  • Clearness Index (Kt) - Solar atmospheric transparency")
        print("  • Wind Power Coefficient (Cp) - Turbine efficiency")
        print("  • Complementarity Index - Solar-wind synergy")
        print("  • Power Curves - Normalized efficiency metrics")
        print("  • Temporal Features - Cyclical time encoding")
        print("  • Lag Features - Historical patterns")
        print("  • Volatility Metrics - Rate of change (dP/dt)")
        print("\nReady for forecasting model training!")
        print("="*60)
        
        return df


if __name__ == '__main__':
    engineer = FeatureEngineer()
    engineer.run()
=======
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
>>>>>>> 0ba306db939cd0c78f9380d603453b23345abac5
