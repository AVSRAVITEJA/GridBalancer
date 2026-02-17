"""
Solar PV Forecasting Model for PVGRIDBALANCER
Implements Temporal Convolutional Network (TCN) for solar generation prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("⚠ Warning: TensorFlow/sklearn not installed. Install with: pip install tensorflow scikit-learn")


class PVForecaster:
    """Solar PV generation forecasting using TCN"""
    
    def __init__(self, preprocessed_path='data/preprocessed', models_path='models'):
        self.preprocessed_path = Path(preprocessed_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler_X = StandardScaler() if DEPS_AVAILABLE else None
        self.scaler_y = StandardScaler() if DEPS_AVAILABLE else None
        self.feature_cols = []
        self.target_col = 'PV Generation (KW)'
    
    def load_data(self):
        """Load engineered features"""
        file_path = self.preprocessed_path / 'engineered_features.csv'
        df = pd.read_csv(file_path)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Loaded data: {df.shape}")
        return df
    
    def prepare_features(self, df):
        """Select relevant features for PV forecasting"""
        print("\n=== Preparing PV Features ===")
        
        # Solar-specific features
        solar_features = [
            'Temperature©', 'Humidity ', 
            'Ground Radiation Intensity (W/m^2)',
            'Upper Atmosphere Radiation Intensity (W/m^2)',
            'clearness_index_kt', 'solar_efficiency',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'day_of_week', 'month'
        ]
        
        # Add lag features
        lag_features = [col for col in df.columns if 'solar_lag' in col]
        rolling_features = [col for col in df.columns if 'solar_rolling' in col]
        
        # Combine all features
        self.feature_cols = [col for col in solar_features + lag_features + rolling_features 
                            if col in df.columns]
        
        print(f"  Selected {len(self.feature_cols)} features")
        
        # Remove rows with NaN (from lag features)
        df_clean = df[self.feature_cols + [self.target_col]].dropna()
        
        X = df_clean[self.feature_cols].values
        y = df_clean[self.target_col].values
        
        print(f"  Clean dataset: {X.shape[0]} samples")
        
        return X, y
    
    def build_tcn_model(self, input_shape, filters=64, kernel_size=3, dropout=0.2):
        """
        Build Temporal Convolutional Network
        TCN uses dilated causal convolutions for time-series
        """
        print("\n=== Building TCN Model ===")
        
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # TCN Block 1 (dilation=1)
            layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=1, activation='relu'),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            
            # TCN Block 2 (dilation=2)
            layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=2, activation='relu'),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            
            # TCN Block 3 (dilation=4)
            layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=4, activation='relu'),
            layers.Dropout(dropout),
            layers.BatchNormalization(),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"  Model parameters: {model.count_params():,}")
        
        return model
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the TCN model"""
        print("\n=== Training PV Forecasting Model ===")
        
        if not DEPS_AVAILABLE:
            print("⚠ Cannot train: TensorFlow not available")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Reshape for TCN (samples, timesteps, features)
        X_train_tcn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_tcn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        # Build model
        self.model = self.build_tcn_model(input_shape=(X_train_tcn.shape[1], 1))
        
        # Train
        history = self.model.fit(
            X_train_tcn, y_train_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        # Evaluate
        y_pred_scaled = self.model.predict(X_test_tcn, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n  Test Metrics:")
        print(f"    RMSE: {rmse:.2f} kW")
        print(f"    MAE: {mae:.2f} kW")
        print(f"    R²: {r2:.4f}")
        
        return history, (y_test, y_pred)
    
    def save_model(self):
        """Save trained model and scalers"""
        if self.model and DEPS_AVAILABLE:
            model_file = self.models_path / 'pv_tcn_model.keras'
            self.model.save(model_file)
            print(f"\n✓ Saved model to {model_file}")
            
            # Save scalers
            scaler_file = self.models_path / 'pv_scalers.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump({
                    'scaler_X': self.scaler_X,
                    'scaler_y': self.scaler_y,
                    'feature_cols': self.feature_cols
                }, f)
            print(f"✓ Saved scalers to {scaler_file}")
    
    def run(self, epochs=50):
        """Execute full PV forecasting pipeline"""
        print("="*60)
        print("PV Forecasting Model Training")
        print("="*60)
        
        if not DEPS_AVAILABLE:
            print("\n⚠ Skipping training: Dependencies not installed")
            print("Install with: pip install tensorflow scikit-learn")
            return None
        
        # Load and prepare data
        df = self.load_data()
        X, y = self.prepare_features(df)
        
        # Train model
        history, predictions = self.train(X, y, epochs=epochs)
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("✓ PV Forecasting Model Training Complete!")
        print("="*60)
        
        return history, predictions


if __name__ == '__main__':
    forecaster = PVForecaster()
    forecaster.run(epochs=50)
