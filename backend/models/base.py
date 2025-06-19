"""
Base model classes for machine learning models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
from sklearn.base import BaseEstimator
from loguru import logger


class BaseModel(ABC, BaseEstimator):
    """
    Base class for all models in the algothon-quant package.
    """
    
    def __init__(self, **kwargs):
        self.is_fitted = False
        self.model_params = kwargs
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'BaseModel':
        """
        Fit the model to the data.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature data
            
        Returns:
            Predictions
        """
        pass
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model_params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        self.model_params.update(params)
        return self


class TimeSeriesModel(BaseModel):
    """
    Base class for time series models.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []
    
    def fit(self, y: Union[np.ndarray, pd.Series], 
            X: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> 'TimeSeriesModel':
        """
        Fit the time series model.
        
        Args:
            y: Target time series
            X: Optional exogenous variables
            
        Returns:
            Self
        """
        self._validate_time_series_data(y)
        return self
    
    def _validate_time_series_data(self, data: Union[np.ndarray, pd.Series]) -> None:
        """Validate time series data."""
        if len(data) < 2:
            raise ValueError("Time series must have at least 2 observations")
        
        if isinstance(data, pd.Series) and not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Time series data should have DatetimeIndex") 