"""
Data validation utilities for quantitative finance.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any
from loguru import logger


def validate_price_data(prices: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    Validate price data for common issues.
    
    Args:
        prices: Price data to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if prices.empty:
        raise ValueError("Price data is empty")
    
    if prices.isnull().any().any():
        raise ValueError("Price data contains null values")
    
    if (prices <= 0).any().any():
        raise ValueError("Price data contains non-positive values")
    
    return True


def validate_returns_data(returns: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    Validate returns data for common issues.
    
    Args:
        returns: Returns data to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if returns.empty:
        raise ValueError("Returns data is empty")
    
    if returns.isnull().any().any():
        raise ValueError("Returns data contains null values")
    
    # Check for extreme outliers (more than 10 standard deviations)
    for col in returns.columns if hasattr(returns, 'columns') else [returns.name]:
        col_data = returns[col] if hasattr(returns, 'columns') else returns
        mean_val = col_data.mean()
        std_val = col_data.std()
        outliers = np.abs(col_data - mean_val) > 10 * std_val
        if outliers.any():
            logger.warning(f"Extreme outliers detected in {col}")
    
    return True


def validate_correlation_matrix(corr_matrix: np.ndarray) -> bool:
    """
    Validate correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not np.allclose(corr_matrix, corr_matrix.T):
        raise ValueError("Correlation matrix is not symmetric")
    
    if not np.allclose(np.diag(corr_matrix), 1.0):
        raise ValueError("Correlation matrix diagonal is not 1.0")
    
    eigenvalues = np.linalg.eigvals(corr_matrix)
    if np.any(eigenvalues < -1e-10):  # Allow for small numerical errors
        raise ValueError("Correlation matrix is not positive semi-definite")
    
    return True


def validate_covariance_matrix(cov_matrix: np.ndarray) -> bool:
    """
    Validate covariance matrix.
    
    Args:
        cov_matrix: Covariance matrix to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not np.allclose(cov_matrix, cov_matrix.T):
        raise ValueError("Covariance matrix is not symmetric")
    
    if np.any(np.diag(cov_matrix) <= 0):
        raise ValueError("Covariance matrix has non-positive diagonal elements")
    
    eigenvalues = np.linalg.eigvals(cov_matrix)
    if np.any(eigenvalues < -1e-10):  # Allow for small numerical errors
        raise ValueError("Covariance matrix is not positive semi-definite")
    
    return True


def check_data_consistency(data1: Union[pd.Series, pd.DataFrame], 
                          data2: Union[pd.Series, pd.DataFrame]) -> bool:
    """
    Check if two datasets are consistent (same length, same index).
    
    Args:
        data1: First dataset
        data2: Second dataset
        
    Returns:
        True if consistent, raises ValueError otherwise
    """
    if len(data1) != len(data2):
        raise ValueError("Datasets have different lengths")
    
    if hasattr(data1, 'index') and hasattr(data2, 'index'):
        if not data1.index.equals(data2.index):
            raise ValueError("Datasets have different indices")
    
    return True 