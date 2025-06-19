"""
Data validation utilities for the data module.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any
from loguru import logger


def validate_data_format(data: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    Validate data format for processing.
    
    Args:
        data: Data to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if data.empty:
        raise ValueError("Data is empty")
    
    if data.isnull().all().all() if hasattr(data, 'columns') else data.isnull().all():
        raise ValueError("Data contains only null values")
    
    return True


def validate_time_series_data(data: pd.Series) -> bool:
    """
    Validate time series data.
    
    Args:
        data: Time series data to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be DatetimeIndex")
    
    if not data.index.is_monotonic_increasing:
        raise ValueError("Data index must be sorted in ascending order")
    
    return True


def validate_feature_data(data: pd.DataFrame) -> bool:
    """
    Validate feature data for machine learning.
    
    Args:
        data: Feature data to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if data.empty:
        raise ValueError("Feature data is empty")
    
    if data.isnull().any().any():
        raise ValueError("Feature data contains null values")
    
    if data.isin([np.inf, -np.inf]).any().any():
        raise ValueError("Feature data contains infinite values")
    
    return True 