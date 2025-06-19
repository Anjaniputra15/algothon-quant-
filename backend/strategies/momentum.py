"""
Momentum-based trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from .base import TradingStrategy
from loguru import logger


class MomentumStrategy(TradingStrategy):
    """
    Simple momentum-based trading strategy.
    
    This strategy invests in instruments that have shown positive momentum
    over a specified lookback period. It ranks instruments by their returns
    and invests in the top performers, subject to the $10,000 per-stock cap
    and 10 bps commission constraints.
    
    Attributes:
        lookback_period (int): Number of days to calculate momentum
        top_n (int): Number of top performers to invest in
        rebalance_frequency (int): Days between rebalancing
    """
    
    def __init__(self, name: str = "MomentumStrategy", 
                 lookback_period: int = 20,
                 top_n: int = 10,
                 rebalance_frequency: int = 5,
                 **kwargs):
        """
        Initialize the momentum strategy.
        
        Args:
            name: Strategy name
            lookback_period: Days to calculate momentum
            top_n: Number of top performers to invest in
            rebalance_frequency: Days between rebalancing
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        
        self.lookback_period = lookback_period
        self.top_n = min(top_n, 50)  # Ensure we don't exceed instrument count
        self.rebalance_frequency = rebalance_frequency
        
        # Strategy-specific state
        self.momentum_scores = None
        self.last_rebalance_day = -1
        
        logger.info(f"Initialized {self.name} with lookback={lookback_period}, "
                   f"top_n={self.top_n}, rebalance_freq={rebalance_frequency}")
    
    def fit(self, prices: np.ndarray) -> 'MomentumStrategy':
        """
        Fit the momentum strategy to historical price data.
        
        Args:
            prices: Historical price data (days × instruments)
            
        Returns:
            Self for method chaining
        """
        # Validate and store price data
        self.price_data, self.n_days, self.n_instruments = self.validate_price_data(prices)
        
        # Calculate initial momentum scores
        self._calculate_momentum_scores()
        
        self.is_fitted = True
        logger.info(f"Fitted {self.name} to {self.n_days} days of data")
        
        return self
    
    def get_positions(self, day_idx: int) -> Dict[str, float]:
        """
        Get trading positions for a specific day.
        
        Args:
            day_idx: Day index (0-based)
            
        Returns:
            Dictionary of instrument positions
        """
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before getting positions")
        
        if day_idx < 0 or day_idx >= self.n_days:
            raise ValueError(f"Invalid day_idx: {day_idx}")
        
        # Check if we need to rebalance
        if day_idx - self.last_rebalance_day >= self.rebalance_frequency:
            self._calculate_momentum_scores(day_idx)
            self.last_rebalance_day = day_idx
        
        # Get top performers
        if self.momentum_scores is None or len(self.momentum_scores) == 0:
            return {}
        
        # Sort by momentum score and get top N
        sorted_indices = np.argsort(self.momentum_scores)[::-1]  # Descending order
        top_indices = sorted_indices[:self.top_n]
        
        # Create positions
        positions = {}
        position_value = self.max_position_value / self.top_n  # Equal weight
        
        for i, instrument_idx in enumerate(top_indices):
            if self.momentum_scores[instrument_idx] > 0:  # Only positive momentum
                instrument_name = f"instrument_{instrument_idx:03d}"
                positions[instrument_name] = position_value
        
        # Apply constraints
        constrained_positions = self.apply_position_constraints(positions)
        
        return constrained_positions
    
    def update(self, day_idx: int, new_row: np.ndarray) -> 'MomentumStrategy':
        """
        Update the strategy with new market data.
        
        Args:
            day_idx: Day index for the new data
            new_row: New price data for all instruments
            
        Returns:
            Self for method chaining
        """
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before updating")
        
        if day_idx < 0 or day_idx >= self.n_days:
            raise ValueError(f"Invalid day_idx: {day_idx}")
        
        # Update price data
        if day_idx < len(self.price_data):
            self.price_data[day_idx] = new_row
        else:
            # Append new row if beyond current data
            self.price_data = np.vstack([self.price_data, new_row])
            self.n_days += 1
        
        # Get positions for this day
        positions = self.get_positions(day_idx)
        
        # Record the trade
        self.record_trade(day_idx, positions, new_row)
        
        # Update current positions
        self.current_positions = positions.copy()
        
        return self
    
    def _calculate_momentum_scores(self, day_idx: Optional[int] = None) -> None:
        """
        Calculate momentum scores for all instruments.
        
        Args:
            day_idx: Day index to calculate scores for (None for latest)
        """
        if day_idx is None:
            day_idx = self.n_days - 1
        
        if day_idx < self.lookback_period:
            logger.warning(f"Insufficient data for momentum calculation: "
                          f"need {self.lookback_period}, have {day_idx + 1}")
            self.momentum_scores = np.zeros(self.n_instruments)
            return
        
        # Calculate returns over lookback period
        start_idx = day_idx - self.lookback_period
        end_idx = day_idx
        
        start_prices = self.price_data[start_idx]
        end_prices = self.price_data[end_idx]
        
        # Calculate returns (avoid division by zero)
        returns = np.where(start_prices > 0, 
                          (end_prices - start_prices) / start_prices, 
                          0.0)
        
        self.momentum_scores = returns
        
        logger.debug(f"Calculated momentum scores for day {day_idx}: "
                    f"mean={np.mean(returns):.4f}, std={np.std(returns):.4f}")
    
    def get_strategy_info(self) -> Dict:
        """
        Get strategy-specific information.
        
        Returns:
            Dictionary with strategy parameters
        """
        return {
            'strategy_type': 'momentum',
            'lookback_period': self.lookback_period,
            'top_n': self.top_n,
            'rebalance_frequency': self.rebalance_frequency,
            'momentum_scores_mean': np.mean(self.momentum_scores) if self.momentum_scores is not None else None,
            'momentum_scores_std': np.std(self.momentum_scores) if self.momentum_scores is not None else None
        } 