"""
Time series models for quantitative finance.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import TimeSeriesModel
from loguru import logger


class ARIMAModel(TimeSeriesModel):
    """
    ARIMA model wrapper.
    """
    
    def __init__(self, order: tuple = (1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.model = None
    
    def fit(self, y: Union[np.ndarray, pd.Series], 
            X: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> 'ARIMAModel':
        """Fit the ARIMA model."""
        super().fit(y, X)
        self.model = ARIMA(y, order=self.order)
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.fitted_model.forecast(steps=steps)


class SARIMAXModel(TimeSeriesModel):
    """
    SARIMAX model wrapper.
    """
    
    def __init__(self, order: tuple = (1, 1, 1), 
                 seasonal_order: tuple = (1, 1, 1, 12), **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
    
    def fit(self, y: Union[np.ndarray, pd.Series], 
            X: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> 'SARIMAXModel':
        """Fit the SARIMAX model."""
        super().fit(y, X)
        self.model = SARIMAX(y, exog=X, order=self.order, 
                           seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = 1, 
                exog: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.fitted_model.forecast(steps=steps, exog=exog)


class MovingAverageModel(TimeSeriesModel):
    """
    Simple moving average model.
    """
    
    def __init__(self, window: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.last_values = None
    
    def fit(self, y: Union[np.ndarray, pd.Series], 
            X: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> 'MovingAverageModel':
        """Fit the moving average model."""
        super().fit(y, X)
        self.last_values = y[-self.window:].values
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        current_values = self.last_values.copy()
        
        for _ in range(steps):
            prediction = np.mean(current_values)
            predictions.append(prediction)
            current_values = np.roll(current_values, -1)
            current_values[-1] = prediction
        
        return np.array(predictions) 