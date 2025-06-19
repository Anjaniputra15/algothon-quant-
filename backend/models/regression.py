"""
Regression models for quantitative finance.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .base import BaseModel
from loguru import logger


class LinearRegressionModel(BaseModel):
    """
    Linear regression model wrapper.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression(**kwargs)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'LinearRegressionModel':
        """Fit the linear regression model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class RidgeRegressionModel(BaseModel):
    """
    Ridge regression model wrapper.
    """
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.model = Ridge(alpha=alpha, **kwargs)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'RidgeRegressionModel':
        """Fit the ridge regression model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class RandomForestRegressionModel(BaseModel):
    """
    Random forest regression model wrapper.
    """
    
    def __init__(self, n_estimators: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'RandomForestRegressionModel':
        """Fit the random forest regression model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class XGBoostRegressionModel(BaseModel):
    """
    XGBoost regression model wrapper.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = XGBRegressor(**kwargs)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'XGBoostRegressionModel':
        """Fit the XGBoost regression model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X) 