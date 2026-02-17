"""
Data Cleaning Module for PVGRIDBALANCER
Handles noise removal, outlier detection, and distinguishes between 
physical zero-generation (night) and sensor failure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Cleans and preprocesses raw renewable energy datasets"""
    
    def __init__(self, raw_data_path='data/raw', output_path='data/preprocessed'):
        self.raw_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def load_excel_files(self, pattern):
        """Load all Excel files matching pattern"""
        files = sorted(self.raw_path.glob(pattern))
        dataframes = []
        
        for file in files:
            print(f"Loading {file.name}...")
            df = pd.read_excel(file)
            dataframes.append(df)
        
        return dataframes
    
    def detect_outliers_iqr(self, series, multiplier=3.0):
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def clean_solar_data(self, df):
        """Clean solar PV data with physics-aware logic"""
        df = df.copy()
        
        # Drop unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        
        # Ensure datetime column - handle various column names
        timestamp_col = None
        for col in ['Date-Time ', 'Date-Time', 'timestamp', 'date', 'Date', 'datetime']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            df['timestamp'] = pd.to_datetime(df[timestamp_col])
            if timestamp_col != 'timestamp':
                df = df.drop(columns=[timestamp_col])
        
        # Extract hour for night detection
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
        
        # Identify power columns
        power_cols = [col for col in df.columns if 'power' in col.lower() or 'generation' in col.lower()]
        
        for col in power_cols:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Night hours (physical zero is expected)
            if 'hour' in df.columns:
                night_mask = (df['hour'] < 6) | (df['hour'] > 20)
                df.loc[night_mask & (df[col] < 0), col] = 0
            
            # Remove negative values during day
            df.loc[df[col] < 0, col] = np.nan
            
            # Detect and remove outliers (only during daylight)
            if 'hour' in df.columns:
                day_mask = (df['hour'] >= 6) & (df['hour'] <= 20)
                day_data = df.loc[day_mask, col]
                if len(day_data) > 0:
                    outliers = self.detect_outliers_iqr(day_data.dropna())
                    df.loc[day_mask & outliers, col] = np.nan
            
            # Forward fill small gaps (max 3 consecutive)
            df[col] = df[col].ffill(limit=3)
            
            # Interpolate remaining gaps
            df[col] = df[col].interpolate(method='linear', limit=10)
        
        return df
    
    def clean_wind_data(self, df):
        """Clean wind data with physics-aware logic"""
        df = df.copy()
        
        # Drop unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        
        # Ensure datetime column - handle various column names
        timestamp_col = None
        for col in ['Date-Time ', 'Date-Time', 'timestamp', 'date', 'Date', 'datetime']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            df['timestamp'] = pd.to_datetime(df[timestamp_col])
            if timestamp_col != 'timestamp':
                df = df.drop(columns=[timestamp_col])
        
        # Identify power and wind speed columns
        power_cols = [col for col in df.columns if 'power' in col.lower() or 'generation' in col.lower()]
        speed_cols = [col for col in df.columns if 'speed' in col.lower() or 'wind' in col.lower()]
        
        for col in power_cols:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove negative values
            df.loc[df[col] < 0, col] = 0
            
            # Detect and remove outliers
            outliers = self.detect_outliers_iqr(df[col].dropna())
            df.loc[outliers, col] = np.nan
            
            # Forward fill small gaps
            df[col] = df[col].ffill(limit=3)
            
            # Interpolate remaining gaps
            df[col] = df[col].interpolate(method='linear', limit=10)
        
        for col in speed_cols:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Physical constraints: wind speed >= 0
            df.loc[df[col] < 0, col] = 0
            
            # Detect outliers
            outliers = self.detect_outliers_iqr(df[col].dropna())
            df.loc[outliers, col] = np.nan
            
            # Interpolate
            df[col] = df[col].interpolate(method='linear', limit=10)
        
        return df
    
    def process_all_solar(self):
        """Process all solar datasets"""
        print("\n=== Processing Solar Datasets ===")
        solar_files = self.load_excel_files('solar_dataset_*.xlsx')
        
        cleaned_solar = []
        for i, df in enumerate(solar_files, 1):
            print(f"Cleaning solar dataset {i}...")
            cleaned = self.clean_solar_data(df)
            cleaned_solar.append(cleaned)
            
            # Save individual cleaned file
            output_file = self.output_path / f'solar_cleaned_{i}.csv'
            cleaned.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
        
        return cleaned_solar
    
    def process_all_wind(self):
        """Process all wind datasets"""
        print("\n=== Processing Wind Datasets ===")
        wind_files = self.load_excel_files('wind_dataset_*.xlsx')
        
        cleaned_wind = []
        for i, df in enumerate(wind_files, 1):
            print(f"Cleaning wind dataset {i}...")
            cleaned = self.clean_wind_data(df)
            cleaned_wind.append(cleaned)
            
            # Save individual cleaned file
            output_file = self.output_path / f'wind_cleaned_{i}.csv'
            cleaned.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
        
        return cleaned_wind
    
    def run(self):
        """Execute full cleaning pipeline"""
        print("Starting data cleaning pipeline...")
        solar_data = self.process_all_solar()
        wind_data = self.process_all_wind()
        print("\n✓ Data cleaning completed successfully!")
        return solar_data, wind_data


if __name__ == '__main__':
    cleaner = DataCleaner()
    cleaner.run()
