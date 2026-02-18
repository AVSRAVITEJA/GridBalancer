# PVGRIDBALANCER Dataset Metadata

## Overview
This document describes the datasets used in the PVGRIDBALANCER project for renewable energy forecasting and grid balancing.

## Raw Data Structure

### Location
`data/raw/`

### Files
- **Solar Datasets**: `solar_dataset_2019.xlsx`, `solar_dataset_2019.1.xlsx`, `solar_dataset_2019.2.xlsx`, `solar_dataset_2019.3.xlsx`
- **Wind Datasets**: `wind_dataset_2019.xlsx`, `wind_dataset_2019.1.xlsx`, `wind_dataset_2019.2.xlsx`, `wind_dataset_2019.3.xlsx`

### Expected Columns
The preprocessing pipeline automatically detects and processes columns containing:
- **Timestamp/Date**: `timestamp`, `date`, `Date`
- **Power Generation**: Columns with `power` or `generation` in name
- **Wind Speed**: Columns with `speed` or `wind` in name (wind datasets only)
- **Solar Irradiance**: Columns with `irradiance` or `ghi` in name (solar datasets only)

## Preprocessing Pipeline

### 1. Data Cleaning (`data_cleaning.py`)
- **Noise Removal**: IQR-based outlier detection (3x multiplier)
- **Missing Value Handling**: Forward fill (max 3 steps) + linear interpolation (max 10 steps)
- **Solar-Specific**:
  - Night detection (hours < 6 or > 20)
  - Physical zero vs sensor failure distinction
  - Negative value removal during daylight
- **Wind-Specific**:
  - Physical constraints (speed ≥ 0, power ≥ 0)
  - Outlier detection across all hours

### 2. Validation & Alignment (`validate_and_align.py`)
- **Quality Checks**:
  - Missing value detection
  - Duplicate row identification
  - Timestamp continuity verification
  - Value range validation
- **Temporal Alignment**: Resampling to common frequency (default: 15 minutes)

### 3. Dataset Merging (`merged_datasets.py`)
- **Temporal Synchronization**: Aligns solar and wind data to common timeline
- **Statistical Signature Extraction**: Hourly and daily patterns
- **Complementarity Index**: Measures how well solar and wind complement each other
  - Formula: `1 - |solar_normalized - wind_normalized|`
  - Range: [0, 1], higher = better complementarity
- **Combined Generation**: `total_renewable_power = solar_power + wind_power`

## Preprocessed Data Structure

### Location
`data/preprocessed/`

### Files
- **Individual Cleaned**: `solar_cleaned_1.csv`, `solar_cleaned_2.csv`, etc.
- **Individual Cleaned**: `wind_cleaned_1.csv`, `wind_cleaned_2.csv`, etc.
- **Unified Dataset**: `unified_renewable_data.csv`

### Unified Dataset Columns
- `timestamp`: Datetime index
- `*_solar`: Solar-related features (power, irradiance, etc.)
- `*_wind`: Wind-related features (power, speed, etc.)
- `complementarity_index`: Solar-wind complementarity metric [0-1]
- `total_renewable_power`: Combined generation (MW or kW)

## Data Quality Metrics

### Solar Data
- **Temporal Resolution**: 15-minute intervals (typical)
- **Generation Range**: 0 to rated capacity
- **Night Hours**: 0 generation expected (20:00 - 06:00)
- **Peak Hours**: 10:00 - 16:00

### Wind Data
- **Temporal Resolution**: 15-minute intervals (typical)
- **Generation Range**: 0 to rated capacity
- **Cut-in Speed**: ~3 m/s
- **Rated Speed**: ~12-15 m/s
- **Cut-out Speed**: ~25 m/s

## Usage

### Run Complete Pipeline
```bash
python run_preprocessing.py
```

### Run Individual Steps
```python
from src.data_cleaning import DataCleaner
from src.validate_and_align import DataValidator
from src.merged_datasets import DatasetMerger

# Step 1: Clean
cleaner = DataCleaner()
cleaner.run()

# Step 2: Validate
validator = DataValidator()
validator.run()

# Step 3: Merge
merger = DatasetMerger()
merger.run()
```

## Notes
- All timestamps are converted to pandas datetime format
- Missing values after interpolation are flagged for review
- Outliers beyond 3x IQR are removed and interpolated
- Physical constraints are enforced (no negative generation)
- Night-time solar generation is set to 0 (not NaN)

## Next Steps
After preprocessing, the data is ready for:
1. Feature Engineering (`feature_engineering_data.py`)
2. Forecasting Model Training (`hybrid_forecasting.py`)
3. Grid Balancing Simulation (`grid_balancer.py`)
