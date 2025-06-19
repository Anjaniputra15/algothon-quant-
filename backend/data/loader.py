"""
Data loading utilities for quantitative finance.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
from pathlib import Path
import yfinance as yf
from loguru import logger
import re


def load_csv_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        Loaded data as DataFrame
    """
    return pd.read_csv(file_path, **kwargs)


def load_yahoo_finance_data(symbols: Union[str, list], 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from Yahoo Finance.
    
    Args:
        symbols: Stock symbol(s)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Stock data as DataFrame
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    
    data = yf.download(symbols, start=start_date, end=end_date)
    return data


def load_price_data_from_file(file_path: Union[str, Path]) -> pd.Series:
    """
    Load price data from text file.
    
    Args:
        file_path: Path to price data file
        
    Returns:
        Price data as Series
    """
    with open(file_path, 'r') as f:
        prices = [float(line.strip()) for line in f if line.strip()]
    
    return pd.Series(prices)


def load_price_matrix(path, validate=True, sep=None):
    """
    Load price data as a numpy float32 matrix (days × instruments).
    Accepts a file path, numpy array, or pandas DataFrame.
    """
    # If already a numpy array, validate and return
    if isinstance(path, np.ndarray):
        arr = path.astype(np.float32)
        if validate:
            if arr.ndim != 2:
                raise ValueError(f"Price data must be 2-dimensional, got {arr.ndim}")
            n_days, n_instruments = arr.shape
            if n_days == 0 or n_instruments == 0:
                raise ValueError(f"Price data cannot be empty, got shape {arr.shape}")
            if not (50 <= n_instruments <= 100):
                raise ValueError(f"Number of instruments ({n_instruments}) must be between 50 and 100")
            if not np.all(np.isfinite(arr)):
                raise ValueError("Price data contains non-finite values")
            if np.any(arr <= 0):
                raise ValueError("Price data contains non-positive values")
        return arr
    # If already a DataFrame, convert to numpy and validate
    if isinstance(path, pd.DataFrame):
        arr = path.values.astype(np.float32)
        if validate:
            if arr.ndim != 2:
                raise ValueError(f"Price data must be 2-dimensional, got {arr.ndim}")
            n_days, n_instruments = arr.shape
            if n_days == 0 or n_instruments == 0:
                raise ValueError(f"Price data cannot be empty, got shape {arr.shape}")
            if not (50 <= n_instruments <= 100):
                raise ValueError(f"Number of instruments ({n_instruments}) must be between 50 and 100")
            if not np.all(np.isfinite(arr)):
                raise ValueError("Price data contains non-finite values")
            if np.any(arr <= 0):
                raise ValueError("Price data contains non-positive values")
        return arr
    # Otherwise, treat as file path
    path = Path(path)
    if sep is None:
        sep = r"\s+"
    
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")
    
    # Read all lines and parse
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Parse each line into a list of floats
    price_data = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        try:
            # Split by separator and convert to floats
            values = re.split(sep, line)
            row = [float(val) for val in values if val.strip()]
            price_data.append(row)
        except ValueError as e:
            raise ValueError(f"Error parsing line {i}: {line}. {str(e)}")
    
    if not price_data:
        raise ValueError("No valid price data found in file")
    
    # Convert to numpy array
    try:
        price_matrix = np.array(price_data, dtype=np.float32)
    except ValueError as e:
        raise ValueError(f"Error creating price matrix: {str(e)}")
    
    # Validate number of instruments (columns)
    n_instruments = price_matrix.shape[1]
    if not (50 <= n_instruments <= 100):
        raise ValueError(f"Number of instruments ({n_instruments}) must be between 50 and 100")
    
    # Forward-fill weekends (assuming daily data)
    # For simplicity, we'll forward-fill any NaN values
    price_matrix = pd.DataFrame(price_matrix).fillna(method='ffill').values.astype(np.float32)
    
    logger.info(f"Loaded price matrix: {price_matrix.shape[0]} days × {price_matrix.shape[1]} instruments")
    
    return price_matrix


def load_price_df(path, validate=True, sep=None):
    """
    Load price data as a pandas DataFrame (days × instruments).
    Accepts a file path, numpy array, or pandas DataFrame.
    """
    # If already a DataFrame, validate and return
    if isinstance(path, pd.DataFrame):
        df = path.copy()
        if validate:
            arr = df.values
            if arr.ndim != 2:
                raise ValueError(f"Price data must be 2-dimensional, got {arr.ndim}")
            n_days, n_instruments = arr.shape
            if n_days == 0 or n_instruments == 0:
                raise ValueError(f"Price data cannot be empty, got shape {arr.shape}")
            if not (50 <= n_instruments <= 100):
                raise ValueError(f"Number of instruments ({n_instruments}) must be between 50 and 100")
            if not np.all(np.isfinite(arr)):
                raise ValueError("Price data contains non-finite values")
            if np.any(arr <= 0):
                raise ValueError("Price data contains non-positive values")
        return df
    # If already a numpy array, convert to DataFrame and validate
    if isinstance(path, np.ndarray):
        arr = path.astype(np.float32)
        if validate:
            if arr.ndim != 2:
                raise ValueError(f"Price data must be 2-dimensional, got {arr.ndim}")
            n_days, n_instruments = arr.shape
            if n_days == 0 or n_instruments == 0:
                raise ValueError(f"Price data cannot be empty, got shape {arr.shape}")
            if not (50 <= n_instruments <= 100):
                raise ValueError(f"Number of instruments ({n_instruments}) must be between 50 and 100")
            if not np.all(np.isfinite(arr)):
                raise ValueError("Price data contains non-finite values")
            if np.any(arr <= 0):
                raise ValueError("Price data contains non-positive values")
        columns = [f"instrument_{i:03d}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=columns)
        return df
    # Otherwise, treat as file path
    path = Path(path)
    if sep is None:
        sep = r"\s+"
    
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")
    
    # Read all lines and parse
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Parse each line into a list of floats
    price_data = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        try:
            # Split by separator and convert to floats
            values = re.split(sep, line)
            row = [float(val) for val in values if val.strip()]
            price_data.append(row)
        except ValueError as e:
            raise ValueError(f"Error parsing line {i}: {line}. {str(e)}")
    
    if not price_data:
        raise ValueError("No valid price data found in file")
    
    # Create DataFrame
    try:
        df = pd.DataFrame(price_data, dtype=np.float32)
    except ValueError as e:
        raise ValueError(f"Error creating price DataFrame: {str(e)}")
    
    # Validate number of instruments (columns)
    n_instruments = df.shape[1]
    if not (50 <= n_instruments <= 100):
        raise ValueError(f"Number of instruments ({n_instruments}) must be between 50 and 100")
    
    # Forward-fill weekends (assuming daily data)
    # For simplicity, we'll forward-fill any NaN values
    df = df.fillna(method='ffill')
    
    # Set column names as instrument IDs
    df.columns = [f"instrument_{i:03d}" for i in range(n_instruments)]
    
    # Set index as dates (assuming daily data starting from a reasonable date)
    # For now, we'll use a simple integer index, but this could be enhanced
    # to use actual dates if provided
    df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    
    logger.info(f"Loaded price DataFrame: {df.shape[0]} days × {df.shape[1]} instruments")
    
    return df


def load_price_data_with_dates(path: Union[str, Path], 
                              sep: str = r"\s+",
                              date_col: int = 0,
                              date_format: str = "%Y-%m-%d") -> pd.DataFrame:
    """
    Load price data with explicit date parsing.
    
    Args:
        path: Path to price data file
        sep: Separator pattern for splitting lines
        date_col: Column index containing dates (0-based)
        date_format: Date format string
        
    Returns:
        Price data as pandas DataFrame with datetime index
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")
    
    # Read all lines and parse
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Parse each line
    dates = []
    price_data = []
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        try:
            # Split by separator
            values = re.split(sep, line)
            values = [val.strip() for val in values if val.strip()]
            
            if len(values) <= date_col:
                raise ValueError(f"Line {i} has insufficient columns")
            
            # Parse date
            date_str = values[date_col]
            date = pd.to_datetime(date_str, format=date_format)
            dates.append(date)
            
            # Parse prices (skip date column)
            prices = [float(val) for j, val in enumerate(values) if j != date_col]
            price_data.append(prices)
            
        except ValueError as e:
            raise ValueError(f"Error parsing line {i}: {line}. {str(e)}")
    
    if not price_data:
        raise ValueError("No valid price data found in file")
    
    # Create DataFrame
    try:
        df = pd.DataFrame(price_data, dtype=np.float32)
    except ValueError as e:
        raise ValueError(f"Error creating price DataFrame: {str(e)}")
    
    # Validate number of instruments
    n_instruments = df.shape[1]
    if not (50 <= n_instruments <= 100):
        raise ValueError(f"Number of instruments ({n_instruments}) must be between 50 and 100")
    
    # Set index and column names
    df.index = pd.DatetimeIndex(dates)
    df.columns = [f"instrument_{i:03d}" for i in range(n_instruments)]
    
    # Forward-fill weekends
    df = df.fillna(method='ffill')
    
    logger.info(f"Loaded price DataFrame with dates: {df.shape[0]} days × {df.shape[1]} instruments")
    
    return df 