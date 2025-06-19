"""
Mathematical utilities for quantitative finance.
"""

import numpy as np
from typing import Union, Optional, Tuple
from scipy import stats


def normalize_data(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize data using various methods.
    
    Args:
        data: Input data
        method: Normalization method ("zscore", "minmax", "robust")
        
    Returns:
        Normalized data
    """
    if method == "zscore":
        return (data - np.mean(data)) / np.std(data)
    elif method == "minmax":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == "robust":
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return (data - median) / mad
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculate correlation matrix.
    
    Args:
        data: Input data matrix (samples x features)
        
    Returns:
        Correlation matrix
    """
    return np.corrcoef(data.T)


def calculate_covariance_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculate covariance matrix.
    
    Args:
        data: Input data matrix (samples x features)
        
    Returns:
        Covariance matrix
    """
    return np.cov(data.T)


def calculate_eigenvalues_eigenvectors(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate eigenvalues and eigenvectors.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def calculate_principal_components(data: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
    """
    Calculate principal components.
    
    Args:
        data: Input data matrix (samples x features)
        n_components: Number of components to return
        
    Returns:
        Principal components
    """
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Calculate covariance matrix
    cov_matrix = calculate_covariance_matrix(data_centered)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = calculate_eigenvalues_eigenvectors(cov_matrix)
    
    # Project data onto principal components
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
    
    return data_centered @ eigenvectors 