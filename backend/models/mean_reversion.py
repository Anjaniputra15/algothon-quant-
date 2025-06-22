import numpy as np
import pandas as pd
import logging
from typing import Optional, Any, Dict

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class MeanReversionModel:
    """
    Short-term cross-sectional mean reversion (reversal) model.

    - Ranks prior-day returns for all instruments each day.
    - Goes long the bottom decile and short the top decile (decile_width).
    - Uses a lookback window N for return calculation.
    - Supports Optuna hyperparameter tuning and MLflow logging.
    """
    def __init__(self, lookback: int = 1, decile_width: float = 0.1):
        """
        Args:
            lookback (int): Number of days for return calculation (N).
            decile_width (float): Fraction of instruments in each decile (e.g., 0.1 for 10%).
        """
        self.lookback = lookback
        self.decile_width = decile_width
        self.fitted_ = False
        self.params_: Dict[str, Any] = {}

    def fit(self, X, y=None, optuna_trial: Optional[Any] = None):
        """
        Fit the model and optionally tune hyperparameters with Optuna and log to MLflow.

        Args:
            X (np.ndarray or pd.DataFrame): Price matrix (days x instruments or instruments x days).
            y: Ignored.
            optuna_trial: Optuna trial object for hyperparameter tuning.
        """
        run = None
        if MLFLOW_AVAILABLE:
            run = mlflow.start_run(run_name="MeanReversionModel")
            mlflow.log_param("lookback", self.lookback)
            mlflow.log_param("decile_width", self.decile_width)
        try:
            if optuna_trial is not None:
                self.lookback = optuna_trial.suggest_int("lookback", 1, 10)
                self.decile_width = optuna_trial.suggest_float("decile_width", 0.05, 0.5)
                logger.info(f"Optuna trial: lookback={self.lookback}, decile_width={self.decile_width}")
            self.fitted_ = True
            self.params_ = {"lookback": self.lookback, "decile_width": self.decile_width}
            logger.info(f"MeanReversionModel fitted with lookback={self.lookback}, decile_width={self.decile_width}")
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("n_samples", X.shape[0] if hasattr(X, 'shape') else None)
        finally:
            if MLFLOW_AVAILABLE and run is not None:
                mlflow.end_run()
        return self

    def predict(self, X):
        """
        Generate positions based on cross-sectional mean reversion.

        Args:
            X (np.ndarray or pd.DataFrame): Price matrix (days x instruments or instruments x days).

        Returns:
            np.ndarray: Positions (same shape as X, but only for days >= lookback)
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling predict().")

        # Accept both (days x instruments) and (instruments x days)
        if isinstance(X, pd.DataFrame):
            prices = X.values
        else:
            prices = np.asarray(X)

        if prices.shape[0] < prices.shape[1]:
            # Transpose to (days x instruments)
            prices = prices.T

        n_days, n_instruments = prices.shape
        lookback = self.lookback
        decile = int(np.floor(self.decile_width * n_instruments))
        if decile < 1:
            decile = 1

        positions = np.zeros((n_days, n_instruments), dtype=np.float32)

        for t in range(lookback, n_days):
            # Calculate prior-day returns
            rets = (prices[t, :] - prices[t - lookback, :]) / prices[t - lookback, :]
            # Rank returns
            ranks = np.argsort(rets)
            # Long bottom decile, short top decile
            long_idx = ranks[:decile]
            short_idx = ranks[-decile:]
            pos = np.zeros(n_instruments, dtype=np.float32)
            pos[long_idx] = 1.0
            pos[short_idx] = -1.0
            positions[t, :] = pos

        return positions

    def get_params(self, deep=True):
        return {"lookback": self.lookback, "decile_width": self.decile_width}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self 