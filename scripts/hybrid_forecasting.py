import pandas as pd
import numpy as np
import joblib

def hybrid_forecast():

    print("=" * 60)
    print("HYBRID PV + WIND FORECAST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load models
    # --------------------------------------------------
    pv_model = joblib.load("models/pv_forecast_v2.pkl")
    wind_model = joblib.load("models/wind_forecast_v2.pkl")

    # --------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------
    pv_df = pd.read_csv("data/preprocessed/pv_ml_ready.csv")
    wind_df = pd.read_csv("data/preprocessed/wind_ml_ready.csv")

    # Targets
    pv_target = "PV Generation (KW)"
    wind_target = "Power Generation"

    # Remove leakage columns same as training
    pv_leakage = [
        "pv_roll_mean_3", "pv_roll_std_3",
        "pv_lag_1", "pv_lag_3", "pv_ramp_rate"
    ]

    wind_leakage = [
        "wind_roll_mean_3", "wind_roll_std_3",
        "wind_lag_1", "wind_lag_3", "wind_ramp"
    ]

    pv_leakage = [c for c in pv_leakage if c in pv_df.columns]
    wind_leakage = [c for c in wind_leakage if c in wind_df.columns]

    # Feature matrices
    X_pv = pv_df.drop(columns=[pv_target] + pv_leakage)
    X_wind = wind_df.drop(columns=[wind_target] + wind_leakage)

    # --------------------------------------------------
    # 3. Predict
    # --------------------------------------------------
    pv_pred = pv_model.predict(X_pv)
    wind_pred = wind_model.predict(X_wind)

    total_pred = pv_pred + wind_pred
    total_actual = pv_df[pv_target].values + wind_df[wind_target].values

    # --------------------------------------------------
    # 4. Evaluation
    # --------------------------------------------------
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    r2 = r2_score(total_actual, total_pred)
    rmse = np.sqrt(mean_squared_error(total_actual, total_pred))
    mae = mean_absolute_error(total_actual, total_pred)

    print("\nHybrid System Performance:")
    print(f"  R²   = {r2:.4f}")
    print(f"  RMSE = {rmse:.2f} kW")
    print(f"  MAE  = {mae:.2f} kW")

    print("\nHybrid forecaster ready.")

if __name__ == "__main__":
    hybrid_forecast()
