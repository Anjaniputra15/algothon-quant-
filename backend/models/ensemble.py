import numpy as np
import pandas as pd
import logging
from typing import Optional, Any, Dict, Union
from backend.strategies.base import TradingStrategy
from backend.models.time_series import ARIMAModel

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnsembleModel(TradingStrategy):
    """
    Ensemble model blending XGBoost, ARIMA, and LSTM via stacking.
    Weights are determined by rolling 20-day RMSE of each model.
    Inherits TradingStrategy interface.
    """
    def __init__(self, name="EnsembleModel", arima_order=(1,1,1), lstm_units=32, lookback=20, **kwargs):
        super().__init__(name=name, **kwargs)
        self.arima_order = arima_order
        self.lstm_units = lstm_units
        self.lookback = lookback
        self.xgb = XGBRegressor() if XGBOOST_AVAILABLE else None
        self.arima = ARIMAModel(order=arima_order)
        self.lstm = None
        self.weights_ = None
        self.rmse_history_ = None
        self.is_fitted = False

    def _build_lstm(self, input_shape):
        if not KERAS_AVAILABLE:
            raise ImportError("Keras/TensorFlow not available for LSTM.")
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(self.lstm_units, activation='tanh'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray]=None, **kwargs):
        """
        Fit all base models and compute rolling RMSEs for stacking weights.
        Args:
            X: Price matrix (days x instruments)
            y: Target (if supervised, else None)
        """
        run = None
        if MLFLOW_AVAILABLE:
            run = mlflow.start_run(run_name=self.name)
            mlflow.log_param("arima_order", self.arima_order)
            mlflow.log_param("lstm_units", self.lstm_units)
            mlflow.log_param("lookback", self.lookback)
        try:
            if isinstance(X, pd.DataFrame):
                X = X.values
            n_days, n_instruments = X.shape
            preds = {"xgb": [], "arima": [], "lstm": []}
            actuals = []
            rmse_window = self.lookback
            # Fit and predict for each instrument
            for i in range(n_instruments):
                series = X[:, i]
                # Prepare supervised learning data
                X_lag = np.array([series[j:j+rmse_window] for j in range(n_days - rmse_window)])
                y_lag = series[rmse_window:]
                # XGBoost
                if XGBOOST_AVAILABLE:
                    self.xgb.fit(X_lag, y_lag)
                    preds["xgb"].append(self.xgb.predict(X_lag))
                else:
                    preds["xgb"].append(np.zeros_like(y_lag))
                # ARIMA
                self.arima.fit(series[:-(rmse_window)])
                arima_pred = self.arima.predict(steps=len(y_lag))
                preds["arima"].append(arima_pred)
                # LSTM
                if KERAS_AVAILABLE:
                    if self.lstm is None:
                        self.lstm = self._build_lstm((rmse_window, 1))
                    X_lstm = X_lag[..., np.newaxis]
                    self.lstm.fit(X_lstm, y_lag, epochs=10, batch_size=16, verbose=0)
                    lstm_pred = self.lstm.predict(X_lstm, verbose=0).flatten()
                    preds["lstm"].append(lstm_pred)
                else:
                    preds["lstm"].append(np.zeros_like(y_lag))
                actuals.append(y_lag)
            # Compute rolling RMSEs
            actuals = np.array(actuals)
            rmse = {k: [] for k in preds}
            for k in preds:
                for i in range(n_instruments):
                    err = preds[k][i] - actuals[i]
                    rmse[k].append(np.sqrt(np.mean(err**2)))
            # Compute weights (inverse RMSE, normalized)
            rmse_arr = np.array([rmse[k] for k in ["xgb", "arima", "lstm"]])  # shape: (3, n_instruments)
            inv_rmse = 1.0 / (rmse_arr + 1e-8)
            weights = inv_rmse / inv_rmse.sum(axis=0, keepdims=True)
            self.weights_ = weights  # shape: (3, n_instruments)
            self.rmse_history_ = rmse_arr
            self.is_fitted = True
            logger.info(f"EnsembleModel fitted. Weights shape: {weights.shape}")
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("mean_rmse_xgb", float(np.mean(rmse["xgb"])))
                mlflow.log_metric("mean_rmse_arima", float(np.mean(rmse["arima"])))
                mlflow.log_metric("mean_rmse_lstm", float(np.mean(rmse["lstm"])))
        finally:
            if MLFLOW_AVAILABLE and run is not None:
                mlflow.end_run()
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Blend predictions from all base models using rolling RMSE weights.
        Args:
            X: Price matrix (days x instruments)
        Returns:
            np.ndarray: Ensemble predictions (days x instruments)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_days, n_instruments = X.shape
        rmse_window = self.lookback
        preds = {"xgb": [], "arima": [], "lstm": []}
        for i in range(n_instruments):
            series = X[:, i]
            X_lag = np.array([series[j:j+rmse_window] for j in range(n_days - rmse_window)])
            # XGBoost
            if XGBOOST_AVAILABLE:
                xgb_pred = self.xgb.predict(X_lag)
            else:
                xgb_pred = np.zeros(n_days - rmse_window)
            preds["xgb"].append(xgb_pred)
            # ARIMA
            self.arima.fit(series[:-(rmse_window)])
            arima_pred = self.arima.predict(steps=n_days - rmse_window)
            preds["arima"].append(arima_pred)
            # LSTM
            if KERAS_AVAILABLE and self.lstm is not None:
                X_lstm = X_lag[..., np.newaxis]
                lstm_pred = self.lstm.predict(X_lstm, verbose=0).flatten()
            else:
                lstm_pred = np.zeros(n_days - rmse_window)
            preds["lstm"].append(lstm_pred)
        # Blend predictions
        ensemble_preds = []
        for i in range(n_instruments):
            w = self.weights_[:, i]
            stacked = np.stack([preds["xgb"][i], preds["arima"][i], preds["lstm"][i]], axis=0)
            blended = np.tensordot(w, stacked, axes=1)
            # Pad with nan for initial rmse_window days
            blended = np.concatenate([np.full(rmse_window, np.nan), blended])
            ensemble_preds.append(blended)
        return np.array(ensemble_preds).T  # shape: (days, instruments)

    def get_positions(self, day_idx: int) -> Dict[str, float]:
        """
        Return positions for a specific day based on ensemble predictions.
        Args:
            day_idx: Day index
        Returns:
            Dict[str, float]: Instrument -> position value
        """
        if not self.is_fitted or self.weights_ is None:
            raise RuntimeError("Model must be fitted before calling get_positions().")
        if self.price_data is None:
            raise RuntimeError("No price data loaded.")
        preds = self.predict(self.price_data)
        if day_idx >= preds.shape[0]:
            raise ValueError(f"day_idx {day_idx} out of range.")
        # Long top 10, zero others (example logic)
        day_preds = preds[day_idx]
        n = len(day_preds)
        top_idx = np.argsort(day_preds)[-10:]
        positions = {f"instrument_{i:03d}": (self.max_position_value if i in top_idx else 0.0) for i in range(n)}
        return self.apply_position_constraints(positions)

    def update(self, day_idx: int, new_row: Union[np.ndarray, pd.Series]) -> 'EnsembleModel':
        """
        Update the ensemble with new data (no-op for stateless model).
        """
        # For a stateless ensemble, just record the new row
        if self.price_data is not None:
            if isinstance(new_row, pd.Series):
                new_row = new_row.values
            self.price_data = np.vstack([self.price_data, new_row[np.newaxis, :]])
        return self 