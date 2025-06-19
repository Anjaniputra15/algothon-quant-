"""
Performance metrics for trading strategies.
"""

import numpy as np
from typing import Dict, List, Union


def calculate_sharpe_ratio(returns: Union[List[float], np.ndarray], 
                          risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (default: 0.0)
        
    Returns:
        Sharpe ratio
    """
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(cumulative_returns: Union[List[float], np.ndarray]) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        cumulative_returns: List of cumulative returns
        
    Returns:
        Maximum drawdown
    """
    if not cumulative_returns:
        return 0.0
    
    cumulative_array = np.array(cumulative_returns)
    running_max = np.maximum.accumulate(cumulative_array)
    drawdown = cumulative_array - running_max
    
    return np.min(drawdown)


def calculate_calmar_ratio(returns: Union[List[float], np.ndarray],
                          cumulative_returns: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: List of returns
        cumulative_returns: List of cumulative returns
        
    Returns:
        Calmar ratio
    """
    if not returns or not cumulative_returns:
        return 0.0
    
    annualized_return = np.mean(returns) * 252
    max_dd = abs(calculate_max_drawdown(cumulative_returns))
    
    if max_dd == 0:
        return 0.0
    
    return annualized_return / max_dd


def calculate_sortino_ratio(returns: Union[List[float], np.ndarray],
                           risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (default: 0.0)
        
    Returns:
        Sortino ratio
    """
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    # Only consider negative returns for downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.std(negative_returns)
    
    if downside_deviation == 0:
        return 0.0
    
    return np.mean(excess_returns) / downside_deviation


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Calculate win rate from trade history.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Win rate as a fraction
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    return winning_trades / len(trades)


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
    gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss 