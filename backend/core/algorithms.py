"""
Core quantitative finance algorithms.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
from loguru import logger


def calculate_returns(prices: Union[pd.Series, np.ndarray], method: str = "log") -> Union[pd.Series, np.ndarray]:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: Return calculation method ("log" or "simple")
        
    Returns:
        Returns series
    """
    if method == "log":
        return np.log(prices / prices.shift(1))
    elif method == "simple":
        return (prices - prices.shift(1)) / prices.shift(1)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_volatility(returns: Union[pd.Series, np.ndarray], window: int = 252) -> Union[pd.Series, np.ndarray]:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Returns series
        window: Rolling window size
        
    Returns:
        Volatility series
    """
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_max_drawdown(returns: Union[pd.Series, np.ndarray]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Returns series
        
    Returns:
        Tuple of (max_drawdown, start_idx, end_idx)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    end_idx = drawdown.idxmin()
    start_idx = cumulative.loc[:end_idx].idxmax()
    
    return max_dd, start_idx, end_idx 