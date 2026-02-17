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
