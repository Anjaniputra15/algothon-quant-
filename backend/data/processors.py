"""
Data processing utilities for quantitative finance.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any
from loguru import logger


def clean_price_data(prices: pd.Series, method: str = "forward_fill") -> pd.Series:
    """
    Clean price data by handling missing values.
    
    Args:
        prices: Price series
        method: Method for handling missing values
        
    Returns:
        Cleaned price series
    """
    if method == "forward_fill":
        return prices.fillna(method='ffill')
    elif method == "backward_fill":
        return prices.fillna(method='bfill')
    elif method == "interpolate":
        return prices.interpolate()
    else:
        raise ValueError(f"Unknown method: {method}")


def resample_data(data: pd.Series, freq: str) -> pd.Series:
    """
    Resample time series data.
    
    Args:
        data: Time series data
        freq: Resampling frequency
        
    Returns:
        Resampled data
    """
    return data.resample(freq).last()


def calculate_technical_indicators(prices: pd.Series, 
                                 indicators: List[str] = None) -> pd.DataFrame:
    """
    Calculate technical indicators.
    
    Args:
        prices: Price series
        indicators: List of indicators to calculate
        
    Returns:
        DataFrame with technical indicators
    """
    if indicators is None:
        indicators = ["sma_20", "sma_50", "rsi_14"]
    
    result = pd.DataFrame(index=prices.index)
    
    for indicator in indicators:
        if indicator == "sma_20":
            result["SMA_20"] = prices.rolling(window=20).mean()
        elif indicator == "sma_50":
            result["SMA_50"] = prices.rolling(window=50).mean()
        elif indicator == "rsi_14":
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result["RSI_14"] = 100 - (100 / (1 + rs))
    
    return result


def create_features_from_returns(returns: pd.Series, 
                                lags: List[int] = None) -> pd.DataFrame:
    """
    Create features from returns using lagged values.
    
    Args:
        returns: Returns series
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    
    features = pd.DataFrame(index=returns.index)
    features["returns"] = returns
    
    for lag in lags:
        features[f"returns_lag_{lag}"] = returns.shift(lag)
    
    return features 