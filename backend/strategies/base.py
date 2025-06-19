"""
Base trading strategy classes and interfaces.

This module provides the foundation for implementing trading strategies
with standardized interfaces and common functionality.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger


class TradingStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    This class defines the interface that all trading strategies must implement.
    It provides a standardized way to fit strategies to historical data,
    generate trading positions, and update strategies with new market data.
    
    Trading Constraints:
    - Position Size: Maximum $10,000 per stock position
    - Commission: 10 basis points (0.001) per trade
    - All positions are assumed to be long-only (no short selling)
    
    Attributes:
        name (str): Strategy name for identification
        is_fitted (bool): Whether the strategy has been fitted to data
        position_history (List[Dict]): History of all positions taken
        performance_metrics (Dict): Strategy performance statistics
        max_position_value (float): Maximum position value per stock ($10,000)
        commission_rate (float): Commission rate per trade (0.001 = 10 bps)
    """
    
    def __init__(self, name: str = "BaseStrategy", **kwargs):
        """
        Initialize the trading strategy.
        
        Args:
            name: Strategy name for identification and logging
            **kwargs: Additional strategy-specific parameters
        """
        self.name = name
        self.is_fitted = False
        self.position_history = []
        self.performance_metrics = {}
        
        # Trading constraints
        self.max_position_value = 10000.0  # $10,000 per stock cap
        self.commission_rate = 0.001       # 10 basis points (0.001) commission
        
        # Strategy state
        self.current_positions = {}
        self.total_commission_paid = 0.0
        self.total_trades = 0
        
        # Data storage
        self.price_data = None
        self.n_days = 0
        self.n_instruments = 0
        
        logger.info(f"Initialized {self.name} with max position value: ${self.max_position_value:,.0f}, "
                   f"commission rate: {self.commission_rate:.3f}")
    
    @abstractmethod
    def fit(self, prices: Union[np.ndarray, pd.DataFrame]) -> 'TradingStrategy':
        """
        Fit the strategy to historical price data.
        
        This method should analyze the historical data to determine strategy
        parameters, calculate initial signals, and prepare the strategy for
        live trading or backtesting.
        
        Args:
            prices: Historical price data as numpy array (days × instruments) 
                   or pandas DataFrame with datetime index and instrument columns
                   
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If price data is invalid or insufficient
            NotImplementedError: If strategy-specific fitting logic is not implemented
        """
        pass
    
    @abstractmethod
    def get_positions(self, day_idx: int) -> Dict[str, float]:
        """
        Get trading positions for a specific day.
        
        This method should return the target positions for each instrument
        based on the strategy's current state and the specified day index.
        Positions should respect the $10,000 per-stock cap and be ready
        for commission calculation.
        
        Args:
            day_idx: Day index (0-based) for which to generate positions
            
        Returns:
            Dictionary mapping instrument identifiers to position values.
            Position values should be in dollars and respect the max_position_value
            constraint. Example: {"instrument_000": 5000.0, "instrument_001": 10000.0}
            
        Raises:
            ValueError: If day_idx is invalid or strategy is not fitted
            NotImplementedError: If strategy-specific position logic is not implemented
        """
        pass
    
    @abstractmethod
    def update(self, day_idx: int, new_row: Union[np.ndarray, pd.Series]) -> 'TradingStrategy':
        """
        Update the strategy with new market data.
        
        This method should incorporate new price data into the strategy's
        state, potentially triggering position changes. The method should
        handle commission calculations and position history tracking.
        
        Args:
            day_idx: Day index (0-based) for the new data
            new_row: New price data for all instruments as numpy array or pandas Series
                    
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If day_idx is invalid or new_row data is invalid
            NotImplementedError: If strategy-specific update logic is not implemented
        """
        pass
    
    def calculate_commission(self, position_value: float) -> float:
        """
        Calculate commission for a position trade.
        
        Commission is calculated as 10 basis points (0.001) of the position value.
        
        Args:
            position_value: Dollar value of the position
            
        Returns:
            Commission amount in dollars
        """
        return position_value * self.commission_rate
    
    def apply_position_constraints(self, positions: Dict[str, float]) -> Dict[str, float]:
        """
        Apply trading constraints to position values.
        
        This method ensures that:
        1. No position exceeds $10,000 per stock
        2. All positions are non-negative (long-only)
        3. Position values are properly formatted
        
        Args:
            positions: Raw position dictionary
            
        Returns:
            Constrained position dictionary
        """
        constrained_positions = {}
        
        for instrument, value in positions.items():
            # Ensure non-negative positions (long-only)
            if value < 0:
                logger.warning(f"Negative position for {instrument} set to 0 (long-only constraint)")
                value = 0.0
            
            # Apply maximum position value constraint
            if value > self.max_position_value:
                logger.info(f"Position for {instrument} capped at ${self.max_position_value:,.0f} "
                           f"(was ${value:,.0f})")
                value = self.max_position_value
            
            # Round to 2 decimal places for currency
            value = round(value, 2)
            
            if value > 0:  # Only include non-zero positions
                constrained_positions[instrument] = value
        
        return constrained_positions
    
    def record_trade(self, day_idx: int, positions: Dict[str, float], 
                    prices: Optional[Union[np.ndarray, pd.Series]] = None) -> None:
        """
        Record a trade in the position history.
        
        Args:
            day_idx: Day index when the trade occurred
            positions: Final positions after constraints
            prices: Optional price data for the day
        """
        trade_record = {
            'day_idx': day_idx,
            'positions': positions.copy(),
            'total_position_value': sum(positions.values()),
            'commission': sum(self.calculate_commission(pos) for pos in positions.values()),
            'n_positions': len(positions),
            'prices': prices
        }
        
        self.position_history.append(trade_record)
        self.total_commission_paid += trade_record['commission']
        self.total_trades += 1
        
        logger.debug(f"Recorded trade for day {day_idx}: {len(positions)} positions, "
                    f"total value: ${trade_record['total_position_value']:,.2f}, "
                    f"commission: ${trade_record['commission']:,.2f}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of strategy performance.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.position_history:
            return {"message": "No trading history available"}
        
        total_trades = len(self.position_history)
        total_commission = sum(record['commission'] for record in self.position_history)
        avg_position_value = np.mean([record['total_position_value'] 
                                    for record in self.position_history])
        max_position_value = max([record['total_position_value'] 
                                for record in self.position_history])
        
        return {
            'strategy_name': self.name,
            'total_trades': total_trades,
            'total_commission_paid': total_commission,
            'average_position_value': avg_position_value,
            'max_position_value': max_position_value,
            'commission_rate': self.commission_rate,
            'max_position_constraint': self.max_position_value,
            'is_fitted': self.is_fitted
        }
    
    def reset(self) -> 'TradingStrategy':
        """
        Reset the strategy to initial state.
        
        Returns:
            Self for method chaining
        """
        self.is_fitted = False
        self.position_history = []
        self.performance_metrics = {}
        self.current_positions = {}
        self.total_commission_paid = 0.0
        self.total_trades = 0
        self.price_data = None
        self.n_days = 0
        self.n_instruments = 0
        
        logger.info(f"Reset {self.name} to initial state")
        return self
    
    def validate_price_data(self, prices: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, int, int]:
        """
        Validate and prepare price data for the strategy.
        
        Args:
            prices: Price data to validate
            
        Returns:
            Tuple of (validated_prices, n_days, n_instruments)
            
        Raises:
            ValueError: If price data is invalid
        """
        if isinstance(prices, pd.DataFrame):
            # Convert DataFrame to numpy array
            price_array = prices.values
            if prices.index.dtype == 'datetime64[ns]':
                logger.info(f"Using DataFrame with datetime index: {prices.index[0]} to {prices.index[-1]}")
        else:
            price_array = np.asarray(prices)
        
        if price_array.ndim != 2:
            raise ValueError(f"Price data must be 2-dimensional, got {price_array.ndim}")
        
        n_days, n_instruments = price_array.shape
        
        if n_days == 0 or n_instruments == 0:
            raise ValueError(f"Price data cannot be empty, got shape {price_array.shape}")
        
        if not (50 <= n_instruments <= 100):
            raise ValueError(f"Number of instruments ({n_instruments}) must be between 50 and 100")
        
        if not np.all(np.isfinite(price_array)):
            raise ValueError("Price data contains non-finite values")
        
        if np.any(price_array <= 0):
            raise ValueError("Price data contains non-positive values")
        
        logger.info(f"Validated price data: {n_days} days × {n_instruments} instruments")
        return price_array, n_days, n_instruments
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name}(fitted={self.is_fitted}, trades={self.total_trades})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the strategy."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"fitted={self.is_fitted}, trades={self.total_trades}, "
                f"commission_paid=${self.total_commission_paid:.2f})") 