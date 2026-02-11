# scripts/model_stress_test.py

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from pv_forecasting_model import PVForecaster
from wind_forecasting_model import WindForecaster


PV_MODEL = "models/pv_forecast_v3.pkl"
WIND_MODEL = "models/wind_forecast_v3.pkl"

PV_DATA = "data/preprocessed/pv_ml_ready.csv"
WIND_DATA = "data/preprocessed/wind_ml_ready.csv"

PV_TARGET = "PV Generation (KW)"
WIND_TARGET = "Power Generation"


print("="*60)
print("RENEWABLE FORECASTING - STRESS TEST")
print("="*60)

pv_df = pd.read_csv(PV_DATA)
wind_df = pd.read_csv(WIND_DATA)

# Chronological split (70% train / 30% test)
split_index = int(len(pv_df) * 0.7)

pv_test = pv_df.iloc[split_index:]
wind_test = wind_df.iloc[split_index:]

print(f"\nPV Test samples: {len(pv_test)}")
print(f"Wind Test samples: {len(wind_test)}")

pv_model = PVForecaster(PV_MODEL)
wind_model = WindForecaster(WIND_MODEL)

# ======================
# BASELINE PERFORMANCE
# ======================

print("\nBASELINE PERFORMANCE")

pv_preds = pv_model.predict_batch(pv_test)
wind_preds = wind_model.predict_batch(wind_test)

def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"{name} -> R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    return r2

pv_r2 = evaluate(pv_test[PV_TARGET], pv_preds, "PV")
wind_r2 = evaluate(wind_test[WIND_TARGET], wind_preds, "Wind")

# ======================
# NOISE STRESS TEST
# ======================

print("\nNOISE STRESS TEST (5% feature noise)")

pv_noisy = pv_test.copy()
wind_noisy = wind_test.copy()

pv_noisy += np.random.normal(0, 0.05, pv_noisy.shape)
wind_noisy += np.random.normal(0, 0.05, wind_noisy.shape)

pv_preds_noise = pv_model.predict_batch(pv_noisy)
wind_preds_noise = wind_model.predict_batch(wind_noisy)

evaluate(pv_test[PV_TARGET], pv_preds_noise, "PV (Noise)")
evaluate(wind_test[WIND_TARGET], wind_preds_noise, "Wind (Noise)")

# ======================
# HYBRID PERFORMANCE
# ======================

print("\nHYBRID PERFORMANCE")

hybrid_true = pv_test[PV_TARGET].values + wind_test[WIND_TARGET].values
hybrid_pred = pv_preds + wind_preds

evaluate(hybrid_true, hybrid_pred, "Hybrid")
