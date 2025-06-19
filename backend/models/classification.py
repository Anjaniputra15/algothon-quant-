"""
Classification models for quantitative finance.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .base import BaseModel
from loguru import logger


class LogisticRegressionModel(BaseModel):
    """
    Logistic regression model wrapper.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression(**kwargs)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'LogisticRegressionModel':
        """Fit the logistic regression model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)


class RandomForestClassificationModel(BaseModel):
    """
    Random forest classification model wrapper.
    """
    
    def __init__(self, n_estimators: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'RandomForestClassificationModel':
        """Fit the random forest classification model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)


class XGBoostClassificationModel(BaseModel):
    """
    XGBoost classification model wrapper.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = XGBClassifier(**kwargs)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'XGBoostClassificationModel':
        """Fit the XGBoost classification model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X) 