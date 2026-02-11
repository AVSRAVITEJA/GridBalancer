# PVWindGridBalancer

PVgridBalancer is a modular renewable energy forecasting framework designed to provide accurate solar (PV) and wind power predictions, along with a structured foundation for future grid balancing intelligence.

The system is built using production-safe machine learning pipelines with time-series validation and robustness testing to ensure reliability under real-world conditions.

---

## Project Overview

The primary objectives of this repository are:

- Develop accurate photovoltaic (PV) power forecasting models  
- Develop accurate wind power forecasting models  
- Perform stress testing under noisy conditions  
- Provide a scalable foundation for future grid balancing intelligence  

This repository currently focuses on the forecasting layer. The grid balancing engine will be implemented in future phases.

---

## Repository Structure
PVgridBalancer/
│
├── data/preprocessed/
│ ├── pv_dataset.csv
│ └── wind_dataset.csv
│
├── models/
│ ├── pv_forecast_v3.pkl
│ └── wind_forecast_v3.pkl
│
├── scripts/
│ ├── pv_forecasting_model.py
│ ├── wind_forecasting_model.py
│ ├── model_stress_test.py
│ └── (planned) grid_balancing_engine.py
│
└── README.md


---

## Implemented Components

### 1. PV Forecasting Model

- Time-series aware training  
- TimeSeries cross-validation  
- Production-safe feature handling  
- Serialized model output  

Model file:
models/pv_forecast_v3.pkl

Performance (cross-validated):

- R² ≈ 0.97  
- RMSE ≈ 188 kW  
- MAE ≈ 90 kW  

---

### 2. Wind Forecasting Model

- TimeSeries cross-validation  
- High predictive stability  
- Production-ready training pipeline  

Model file:
models/wind_forecast_v3.pkl

Performance (cross-validated):

- R² ≈ 0.9996  
- RMSE ≈ 13 kW  
- MAE ≈ 2 kW  

---

### 3. Stress Testing Framework

The stress testing module evaluates:

- Baseline forecasting performance  
- Model robustness under 5% feature noise  
- Hybrid PV + Wind performance  

This ensures resilience under sensor noise and environmental uncertainty.

Run:
python scripts/model_stress_test.py


---

## Installation

### Clone Repository
git clone https://github.com/yourusername/PVgridBalancer.git

cd PVgridBalancer


### Create Virtual Environment
python -m venv venv
venv\Scripts\activate


### Install Dependencies
pip install -r requirements.txt


---

## Running the Models

### Train PV Model
python scripts/pv_forecasting_model.py


### Train Wind Model
python scripts/wind_forecasting_model.py

### Run Stress Test
python scripts/model_stress_test.py


Planned capabilities:

- Net load prediction  
- Storage dispatch optimization  
- Renewable curtailment control  
- Economic dispatch optimization  
- Reinforcement learning-based adaptive control  
- Real-time simulation integration  

---

## Technical Characteristics

- Time-series safe model training  
- Cross-validation performance reporting  
- Noise robustness validation  
- Modular architecture  
- Serialized model deployment support  

---

## Potential Applications

- Microgrid energy management  
- Renewable generation coordination  
- Smart grid research  
- EV charging infrastructure optimization  
- Academic research in renewable forecasting  
- Hybrid solar-wind farm management  

---

## Author

Ravi Teja AVS  
Renewable Energy Systems | Machine Learning | Intelligent Power Systems






