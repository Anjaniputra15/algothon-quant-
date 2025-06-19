"""
Data loading utilities for quantitative finance.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
from pathlib import Path
import yfinance as yf
from loguru import logger


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