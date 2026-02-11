# scripts/wind_forecasting_model.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


MODEL_PATH = "models/wind_forecast_v3.pkl"
DATA_PATH = "data/preprocessed/wind_ml_ready.csv"
TARGET = "Power Generation"


def train_wind_model():

    print("="*60)
    print("WIND FORECASTING MODEL (PRODUCTION SAFE)")
    print("="*60)

    df = pd.read_csv(DATA_PATH)

    leakage_keywords = ["lag", "roll", "ramp"]
    drop_cols = [col for col in df.columns if any(k in col.lower() for k in leakage_keywords)]
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"\nDataset shape: {df.shape}")
    print(f"Features used: {X.shape[1]}")

    tscv = TimeSeriesSplit(n_splits=5)

    r2_scores = []
    rmse_scores = []
    mae_scores = []

    print("\nTraining with TimeSeries CV...")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        print(f" Fold {fold+1}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

    final_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X, y)

    os.makedirs("models", exist_ok=True)

    model_package = {
        "model": final_model,
        "features": list(X.columns)
    }

    joblib.dump(model_package, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)

    print("\nPerformance Summary:")
    print(f"  R²   = {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"  RMSE = {np.mean(rmse_scores):.2f} kW")
    print(f"  MAE  = {np.mean(mae_scores):.2f} kW")

    print("\nWind Forecaster ready.\n")


class WindForecaster:

    def __init__(self, model_path):
        self.model_data = joblib.load(model_path)

        if isinstance(self.model_data, dict):
            self.model = self.model_data["model"]
            self.features = self.model_data["features"]
        else:
            self.model = self.model_data
            self.features = None

    def predict_batch(self, X):
        if self.features is not None:
            X = X[self.features]
        return self.model.predict(X)


if __name__ == "__main__":
    train_wind_model()
