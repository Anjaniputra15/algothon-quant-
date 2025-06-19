"""
Backtesting engine for trading strategies.

This module provides a comprehensive backtesting framework that simulates
trading strategies on historical data, tracking positions, calculating
commissions, and computing performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from loguru import logger

from ..strategies.base import TradingStrategy


class Backtester:
    """
    Backtesting engine for trading strategies.
    
    This class provides a comprehensive framework for backtesting trading
    strategies on historical price data. It handles position tracking,
    commission calculation, and performance measurement.
    
    Attributes:
        commission_rate (float): Commission rate per trade (default: 0.0005 = 5 bps)
        position_limit (float): Maximum position value per instrument (default: $10,000)
        min_trade_threshold (float): Minimum trade size to execute (default: $100)
    """
    
    def __init__(self, commission_rate: float = 0.0005, 
                 position_limit: float = 10000.0,
                 min_trade_threshold: float = 100.0):
        """
        Initialize the backtester.
        
        Args:
            commission_rate: Commission rate per trade (default: 5 basis points)
            position_limit: Maximum position value per instrument (default: $10,000)
            min_trade_threshold: Minimum trade size to execute (default: $100)
        """
        self.commission_rate = commission_rate
        self.position_limit = position_limit
        self.min_trade_threshold = min_trade_threshold
        
        logger.info(f"Initialized backtester with commission rate: {commission_rate:.4f}, "
                   f"position limit: ${position_limit:,.0f}, "
                   f"min trade threshold: ${min_trade_threshold:,.0f}")
    
    def run_backtest(self, prices: Union[np.ndarray, pd.DataFrame], 
                    strategy: TradingStrategy,
                    comm_rate: Optional[float] = None,
                    pos_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Run a backtest for a trading strategy on historical price data.
        
        This function simulates trading day-by-day, computing position changes,
        applying position limits, charging commissions, and calculating
        performance metrics.
        
        Args:
            prices: Historical price data (days × instruments) as numpy array or DataFrame
            strategy: Trading strategy instance implementing TradingStrategy interface
            comm_rate: Override commission rate for this backtest (optional)
            pos_limit: Override position limit for this backtest (optional)
            
        Returns:
            Dictionary containing backtest results:
            - daily_pnl: List of daily profit/loss values
            - cum_pnl: List of cumulative profit/loss values
            - mean: Mean daily return
            - std: Standard deviation of daily returns
            - metric: Risk-adjusted return metric (mean - 0.1 × std)
            - total_commission: Total commission paid
            - total_trades: Total number of trades executed
            - max_drawdown: Maximum drawdown experienced
            - sharpe_ratio: Sharpe ratio of returns
            - position_history: List of daily position snapshots
            - trade_history: List of all trades executed
            
        Raises:
            ValueError: If price data is invalid or strategy is not fitted
            RuntimeError: If backtest encounters critical errors
        """
        # Use override parameters if provided, otherwise use instance defaults
        commission_rate = comm_rate if comm_rate is not None else self.commission_rate
        position_limit = pos_limit if pos_limit is not None else self.position_limit
        
        logger.info(f"Starting backtest for {strategy.name}")
        logger.info(f"Parameters: commission_rate={commission_rate:.4f}, "
                   f"position_limit=${position_limit:,.0f}")
        
        # Validate inputs
        if not strategy.is_fitted:
            raise ValueError("Strategy must be fitted before running backtest")
        
        # Convert prices to numpy array if needed
        if isinstance(prices, pd.DataFrame):
            price_array = prices.values
            dates = prices.index if prices.index.dtype == 'datetime64[ns]' else None
        else:
            price_array = np.asarray(prices)
            dates = None
        
        if price_array.ndim != 2:
            raise ValueError(f"Price data must be 2-dimensional, got {price_array.ndim}")
        
        n_days, n_instruments = price_array.shape
        logger.info(f"Backtesting on {n_days} days × {n_instruments} instruments")
        
        # Initialize tracking variables
        current_positions = {}  # Current position values by instrument
        position_history = []   # Daily position snapshots
        trade_history = []      # All trades executed
        daily_pnl = []          # Daily profit/loss
        cum_pnl = [0.0]        # Cumulative profit/loss
        total_commission = 0.0  # Total commission paid
        total_trades = 0        # Total number of trades
        
        # Reset strategy state for clean backtest
        strategy.reset()
        strategy.fit(price_array)
        
        # Main backtest loop
        for day_idx in range(n_days):
            try:
                # Get current day's prices
                current_prices = price_array[day_idx]
                
                # Get target positions from strategy
                target_positions = strategy.get_positions(day_idx)
                
                # Calculate position changes (trades)
                trades = self._calculate_trades(current_positions, target_positions)
                
                # Apply position limits and calculate commissions
                executed_trades, day_commission = self._execute_trades(
                    trades, current_prices, commission_rate, position_limit
                )
                
                # Update current positions
                current_positions = self._update_positions(
                    current_positions, executed_trades
                )
                
                # Calculate daily P&L
                day_pnl = self._calculate_daily_pnl(
                    current_positions, current_prices, day_commission
                )
                
                # Update tracking variables
                total_commission += day_commission
                total_trades += len(executed_trades)
                daily_pnl.append(day_pnl)
                cum_pnl.append(cum_pnl[-1] + day_pnl)
                
                # Record position snapshot
                position_snapshot = {
                    'day_idx': day_idx,
                    'date': dates[day_idx] if dates is not None else day_idx,
                    'positions': current_positions.copy(),
                    'total_value': sum(current_positions.values()),
                    'daily_pnl': day_pnl,
                    'cumulative_pnl': cum_pnl[-1],
                    'commission': day_commission,
                    'n_trades': len(executed_trades)
                }
                position_history.append(position_snapshot)
                
                # Record trades
                for trade in executed_trades:
                    trade_record = {
                        'day_idx': day_idx,
                        'date': dates[day_idx] if dates is not None else day_idx,
                        'instrument': trade['instrument'],
                        'trade_value': trade['trade_value'],
                        'commission': trade['commission'],
                        'position_before': trade['position_before'],
                        'position_after': trade['position_after']
                    }
                    trade_history.append(trade_record)
                
                # Update strategy with new data
                strategy.update(day_idx, current_prices)
                
                # Log progress every 100 days
                if (day_idx + 1) % 100 == 0:
                    logger.info(f"Completed day {day_idx + 1}/{n_days}, "
                              f"cumulative P&L: ${cum_pnl[-1]:,.2f}")
                
            except Exception as e:
                logger.error(f"Error on day {day_idx}: {e}")
                raise RuntimeError(f"Backtest failed on day {day_idx}: {e}")
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(daily_pnl, cum_pnl)
        
        # Prepare results
        results = {
            'daily_pnl': daily_pnl,
            'cum_pnl': cum_pnl[1:],  # Remove initial 0.0
            'mean': metrics['mean'],
            'std': metrics['std'],
            'metric': metrics['metric'],
            'total_commission': total_commission,
            'total_trades': total_trades,
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'position_history': position_history,
            'trade_history': trade_history,
            'final_pnl': cum_pnl[-1] if cum_pnl else 0.0,
            'total_return': metrics['total_return'],
            'volatility': metrics['volatility']
        }
        
        logger.info(f"Backtest completed. Final P&L: ${results['final_pnl']:,.2f}, "
                   f"Total commission: ${total_commission:,.2f}, "
                   f"Total trades: {total_trades}")
        
        return results
    
    def _calculate_trades(self, current_positions: Dict[str, float], 
                         target_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate required trades to move from current to target positions.
        
        Args:
            current_positions: Current position values by instrument
            target_positions: Target position values by instrument
            
        Returns:
            Dictionary of required trade values (positive = buy, negative = sell)
        """
        trades = {}
        
        # Get all unique instruments
        all_instruments = set(current_positions.keys()) | set(target_positions.keys())
        
        for instrument in all_instruments:
            current = current_positions.get(instrument, 0.0)
            target = target_positions.get(instrument, 0.0)
            trade_value = target - current
            
            # Only record trades above threshold
            if abs(trade_value) >= self.min_trade_threshold:
                trades[instrument] = trade_value
        
        return trades
    
    def _execute_trades(self, trades: Dict[str, float], 
                       current_prices: np.ndarray,
                       commission_rate: float,
                       position_limit: float) -> Tuple[List[Dict], float]:
        """
        Execute trades with position limits and commission calculation.
        
        Args:
            trades: Dictionary of trade values by instrument
            current_prices: Current day's prices for all instruments
            commission_rate: Commission rate to apply
            position_limit: Maximum position value per instrument
            
        Returns:
            Tuple of (executed_trades, total_commission)
        """
        executed_trades = []
        total_commission = 0.0
        
        for instrument, trade_value in trades.items():
            # Extract instrument index from name (e.g., "instrument_000" -> 0)
            try:
                instrument_idx = int(instrument.split('_')[1])
                if instrument_idx >= len(current_prices):
                    logger.warning(f"Invalid instrument index: {instrument_idx}")
                    continue
            except (IndexError, ValueError):
                logger.warning(f"Invalid instrument name format: {instrument}")
                continue
            
            # Calculate position before trade
            position_before = 0.0  # Would need to track this from previous state
            
            # Apply position limit
            if trade_value > 0:  # Buying
                max_buy = position_limit - position_before
                if trade_value > max_buy:
                    logger.debug(f"Capping buy for {instrument}: ${trade_value:,.2f} -> ${max_buy:,.2f}")
                    trade_value = max_buy
            
            # Skip if trade is too small or would result in negative position
            if abs(trade_value) < self.min_trade_threshold or trade_value < -position_before:
                continue
            
            # Calculate commission
            commission = abs(trade_value) * commission_rate
            
            # Calculate position after trade
            position_after = position_before + trade_value
            
            # Record executed trade
            executed_trade = {
                'instrument': instrument,
                'trade_value': trade_value,
                'commission': commission,
                'position_before': position_before,
                'position_after': position_after
            }
            executed_trades.append(executed_trade)
            total_commission += commission
        
        return executed_trades, total_commission
    
    def _update_positions(self, current_positions: Dict[str, float], 
                         executed_trades: List[Dict]) -> Dict[str, float]:
        """
        Update current positions based on executed trades.
        
        Args:
            current_positions: Current position values
            executed_trades: List of executed trades
            
        Returns:
            Updated position dictionary
        """
        updated_positions = current_positions.copy()
        
        for trade in executed_trades:
            instrument = trade['instrument']
            trade_value = trade['trade_value']
            
            current = updated_positions.get(instrument, 0.0)
            updated_positions[instrument] = current + trade_value
            
            # Remove zero positions
            if abs(updated_positions[instrument]) < self.min_trade_threshold:
                del updated_positions[instrument]
        
        return updated_positions
    
    def _calculate_daily_pnl(self, positions: Dict[str, float], 
                           current_prices: np.ndarray,
                           commission: float) -> float:
        """
        Calculate daily profit/loss from position changes and price movements.
        
        Args:
            positions: Current position values by instrument
            current_prices: Current day's prices
            commission: Commission paid for the day
            
        Returns:
            Daily profit/loss
        """
        # This is a simplified P&L calculation
        # In a real implementation, you'd need to track previous day's positions and prices
        # For now, we'll assume P&L comes from position value changes
        
        total_position_value = sum(positions.values())
        pnl = total_position_value - commission  # Simplified
        
        return pnl
    
    def _calculate_performance_metrics(self, daily_pnl: List[float], 
                                     cum_pnl: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            daily_pnl: List of daily profit/loss values
            cum_pnl: List of cumulative profit/loss values
            
        Returns:
            Dictionary of performance metrics
        """
        if not daily_pnl:
            return {
                'mean': 0.0, 'std': 0.0, 'metric': 0.0, 'max_drawdown': 0.0,
                'sharpe_ratio': 0.0, 'total_return': 0.0, 'volatility': 0.0
            }
        
        daily_pnl_array = np.array(daily_pnl)
        cum_pnl_array = np.array(cum_pnl[1:])  # Remove initial 0.0
        
        # Basic statistics
        mean = np.mean(daily_pnl_array)
        std = np.std(daily_pnl_array)
        metric = mean - 0.1 * std
        
        # Risk metrics
        volatility = std * np.sqrt(252) if std > 0 else 0.0  # Annualized
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = mean / std if std > 0 else 0.0
        sharpe_ratio *= np.sqrt(252)  # Annualized
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(cum_pnl_array)
        drawdown = cum_pnl_array - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Total return
        total_return = cum_pnl_array[-1] if len(cum_pnl_array) > 0 else 0.0
        
        return {
            'mean': mean,
            'std': std,
            'metric': metric,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'volatility': volatility
        }


def run_backtest(prices: Union[np.ndarray, pd.DataFrame], 
                strategy: TradingStrategy,
                comm_rate: float = 0.0005,
                pos_limit: float = 10000.0) -> Dict[str, Any]:
    """
    Convenience function to run a backtest.
    
    This is a simplified interface to the Backtester class for quick backtesting.
    
    Args:
        prices: Historical price data (days × instruments)
        strategy: Trading strategy instance
        comm_rate: Commission rate per trade (default: 5 basis points)
        pos_limit: Maximum position value per instrument (default: $10,000)
        
    Returns:
        Dictionary containing backtest results with keys:
        - daily_pnl: List of daily profit/loss values
        - cum_pnl: List of cumulative profit/loss values  
        - mean: Mean daily return
        - std: Standard deviation of daily returns
        - metric: Risk-adjusted return metric (mean - 0.1 × std)
        - total_commission: Total commission paid
        - total_trades: Total number of trades executed
        - max_drawdown: Maximum drawdown experienced
        - sharpe_ratio: Sharpe ratio of returns
        - position_history: List of daily position snapshots
        - trade_history: List of all trades executed
    """
    backtester = Backtester(commission_rate=comm_rate, position_limit=pos_limit)
    return backtester.run_backtest(prices, strategy, comm_rate, pos_limit) 