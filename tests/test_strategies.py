"""
Tests for trading strategies.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from backend.strategies.base import TradingStrategy
from backend.strategies.momentum import MomentumStrategy


class TestTradingStrategy:
    """Test the base TradingStrategy class."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data for testing."""
        # Create 10 days × 60 instruments of realistic price data
        np.random.seed(42)
        n_days, n_instruments = 10, 60
        base_prices = 100 + np.random.normal(0, 10, n_instruments)
        
        prices = np.zeros((n_days, n_instruments))
        prices[0] = base_prices
        
        for day in range(1, n_days):
            # Add some random walk behavior
            returns = np.random.normal(0, 0.02, n_instruments)
            prices[day] = prices[day-1] * (1 + returns)
        
        return prices
    
    @pytest.fixture
    def strategy(self):
        """Create a base strategy instance."""
        return TradingStrategy(name="TestStrategy")
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "TestStrategy"
        assert not strategy.is_fitted
        assert strategy.max_position_value == 10000.0
        assert strategy.commission_rate == 0.001
        assert len(strategy.position_history) == 0
        assert strategy.total_commission_paid == 0.0
        assert strategy.total_trades == 0
    
    def test_calculate_commission(self, strategy):
        """Test commission calculation."""
        # Test various position values
        assert strategy.calculate_commission(1000.0) == 1.0  # 10 bps of 1000
        assert strategy.calculate_commission(10000.0) == 10.0  # 10 bps of 10000
        assert strategy.calculate_commission(50000.0) == 50.0  # 10 bps of 50000
        assert strategy.calculate_commission(0.0) == 0.0  # No commission for zero position
    
    def test_apply_position_constraints(self, strategy):
        """Test position constraint application."""
        # Test normal positions
        positions = {"instrument_000": 5000.0, "instrument_001": 8000.0}
        constrained = strategy.apply_position_constraints(positions)
        assert constrained == positions
        
        # Test position exceeding cap
        positions = {"instrument_000": 15000.0, "instrument_001": 8000.0}
        constrained = strategy.apply_position_constraints(positions)
        assert constrained["instrument_000"] == 10000.0
        assert constrained["instrument_001"] == 8000.0
        
        # Test negative positions (should be set to 0)
        positions = {"instrument_000": -1000.0, "instrument_001": 5000.0}
        constrained = strategy.apply_position_constraints(positions)
        assert "instrument_000" not in constrained  # Should be excluded
        assert constrained["instrument_001"] == 5000.0
        
        # Test zero positions (should be excluded)
        positions = {"instrument_000": 0.0, "instrument_001": 5000.0}
        constrained = strategy.apply_position_constraints(positions)
        assert "instrument_000" not in constrained
        assert constrained["instrument_001"] == 5000.0
    
    def test_record_trade(self, strategy):
        """Test trade recording."""
        positions = {"instrument_000": 5000.0, "instrument_001": 8000.0}
        prices = np.array([100.0, 101.0])
        
        strategy.record_trade(5, positions, prices)
        
        assert len(strategy.position_history) == 1
        assert strategy.total_trades == 1
        assert strategy.total_commission_paid == 13.0  # 10 bps of 13000
        
        trade_record = strategy.position_history[0]
        assert trade_record['day_idx'] == 5
        assert trade_record['positions'] == positions
        assert trade_record['total_position_value'] == 13000.0
        assert trade_record['commission'] == 13.0
        assert trade_record['n_positions'] == 2
        assert np.array_equal(trade_record['prices'], prices)
    
    def test_get_performance_summary(self, strategy):
        """Test performance summary generation."""
        # Test with no history
        summary = strategy.get_performance_summary()
        assert summary["message"] == "No trading history available"
        
        # Test with some history
        positions1 = {"instrument_000": 5000.0, "instrument_001": 8000.0}
        positions2 = {"instrument_002": 10000.0}
        
        strategy.record_trade(1, positions1)
        strategy.record_trade(2, positions2)
        
        summary = strategy.get_performance_summary()
        assert summary['strategy_name'] == "TestStrategy"
        assert summary['total_trades'] == 2
        assert summary['total_commission_paid'] == 23.0  # 10 bps of 23000
        assert summary['average_position_value'] == 11500.0
        assert summary['max_position_value'] == 13000.0
        assert summary['commission_rate'] == 0.001
        assert summary['max_position_constraint'] == 10000.0
        assert not summary['is_fitted']
    
    def test_reset(self, strategy):
        """Test strategy reset."""
        # Add some state
        strategy.is_fitted = True
        strategy.position_history = [{"test": "data"}]
        strategy.total_commission_paid = 100.0
        strategy.total_trades = 5
        
        strategy.reset()
        
        assert not strategy.is_fitted
        assert len(strategy.position_history) == 0
        assert strategy.total_commission_paid == 0.0
        assert strategy.total_trades == 0
    
    def test_validate_price_data(self, strategy):
        """Test price data validation."""
        # Test valid data
        valid_prices = np.random.rand(10, 60) * 100 + 50  # 10 days × 60 instruments
        prices, n_days, n_instruments = strategy.validate_price_data(valid_prices)
        assert n_days == 10
        assert n_instruments == 60
        
        # Test DataFrame input
        df = pd.DataFrame(valid_prices, 
                         index=pd.date_range('2023-01-01', periods=10),
                         columns=[f"instrument_{i:03d}" for i in range(60)])
        prices, n_days, n_instruments = strategy.validate_price_data(df)
        assert n_days == 10
        assert n_instruments == 60
        
        # Test invalid dimensions
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            strategy.validate_price_data(np.array([1, 2, 3]))
        
        # Test empty data
        with pytest.raises(ValueError, match="cannot be empty"):
            strategy.validate_price_data(np.array([]))
        
        # Test insufficient instruments
        with pytest.raises(ValueError, match="must be between 50 and 100"):
            strategy.validate_price_data(np.random.rand(10, 30))
        
        # Test non-finite values
        invalid_prices = np.random.rand(10, 60)
        invalid_prices[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite values"):
            strategy.validate_price_data(invalid_prices)
        
        # Test non-positive values
        invalid_prices = np.random.rand(10, 60)
        invalid_prices[0, 0] = -1.0
        with pytest.raises(ValueError, match="non-positive values"):
            strategy.validate_price_data(invalid_prices)


class TestMomentumStrategy:
    """Test the MomentumStrategy implementation."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        n_days, n_instruments = 30, 60  # More days for momentum calculation
        base_prices = 100 + np.random.normal(0, 10, n_instruments)
        
        prices = np.zeros((n_days, n_instruments))
        prices[0] = base_prices
        
        for day in range(1, n_days):
            # Add some momentum behavior
            returns = np.random.normal(0.001, 0.02, n_instruments)  # Slight positive drift
            prices[day] = prices[day-1] * (1 + returns)
        
        return prices
    
    @pytest.fixture
    def strategy(self):
        """Create a momentum strategy instance."""
        return MomentumStrategy(name="TestMomentum", 
                               lookback_period=10,
                               top_n=5,
                               rebalance_frequency=3)
    
    def test_momentum_strategy_initialization(self, strategy):
        """Test momentum strategy initialization."""
        assert strategy.name == "TestMomentum"
        assert strategy.lookback_period == 10
        assert strategy.top_n == 5
        assert strategy.rebalance_frequency == 3
        assert strategy.momentum_scores is None
        assert strategy.last_rebalance_day == -1
    
    def test_momentum_strategy_fit(self, strategy, sample_prices):
        """Test momentum strategy fitting."""
        strategy.fit(sample_prices)
        
        assert strategy.is_fitted
        assert strategy.n_days == 30
        assert strategy.n_instruments == 60
        assert strategy.momentum_scores is not None
        assert len(strategy.momentum_scores) == 60
    
    def test_momentum_strategy_get_positions(self, strategy, sample_prices):
        """Test momentum strategy position generation."""
        strategy.fit(sample_prices)
        
        # Get positions for a day after lookback period
        positions = strategy.get_positions(15)
        
        # Should have positions for top 5 instruments with positive momentum
        assert len(positions) <= 5
        
        # Check position values
        for instrument, value in positions.items():
            assert 0 < value <= 10000.0  # Within constraints
            assert value == 2000.0  # Equal weight: 10000/5
        
        # Check instrument names
        for instrument in positions.keys():
            assert instrument.startswith("instrument_")
    
    def test_momentum_strategy_update(self, strategy, sample_prices):
        """Test momentum strategy update."""
        strategy.fit(sample_prices)
        
        # Update with new data
        new_row = sample_prices[15] + np.random.normal(0, 1, 60)
        strategy.update(15, new_row)
        
        # Check that trade was recorded
        assert len(strategy.position_history) == 1
        assert strategy.total_trades == 1
        assert strategy.current_positions is not None
    
    def test_momentum_strategy_rebalancing(self, strategy, sample_prices):
        """Test momentum strategy rebalancing logic."""
        strategy.fit(sample_prices)
        
        # Get positions on consecutive days
        positions1 = strategy.get_positions(15)
        positions2 = strategy.get_positions(16)
        positions3 = strategy.get_positions(17)
        positions4 = strategy.get_positions(18)
        
        # Should rebalance every 3 days (rebalance_frequency)
        # Day 15: initial calculation
        # Day 16: no rebalance (1 day since last)
        # Day 17: no rebalance (2 days since last)
        # Day 18: rebalance (3 days since last)
        
        # The positions should be the same for days 15-17, then change on day 18
        assert positions1 == positions2
        assert positions2 == positions3
        # Note: positions4 might be different due to rebalancing
    
    def test_momentum_strategy_constraints(self, strategy, sample_prices):
        """Test that momentum strategy respects constraints."""
        strategy.fit(sample_prices)
        
        positions = strategy.get_positions(15)
        
        # Check position constraints
        for instrument, value in positions.items():
            assert value > 0  # Long-only
            assert value <= 10000.0  # Max position cap
            assert value == round(value, 2)  # Proper currency formatting
    
    def test_momentum_strategy_info(self, strategy, sample_prices):
        """Test momentum strategy information retrieval."""
        strategy.fit(sample_prices)
        
        info = strategy.get_strategy_info()
        assert info['strategy_type'] == 'momentum'
        assert info['lookback_period'] == 10
        assert info['top_n'] == 5
        assert info['rebalance_frequency'] == 3
        assert info['momentum_scores_mean'] is not None
        assert info['momentum_scores_std'] is not None


class TestStrategyIntegration:
    """Test integration between strategies and data loading."""
    
    @pytest.fixture
    def temp_price_file(self):
        """Create a temporary price file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Create 5 days × 60 instruments
            np.random.seed(42)
            for day in range(5):
                prices = 100 + np.random.normal(0, 5, 60)
                line = ' '.join(f"{price:.2f}" for price in prices)
                f.write(line + '\n')
            f.flush()
            return Path(f.name)
    
    def test_strategy_with_loaded_data(self, temp_price_file):
        """Test strategy with data loaded from file."""
        from backend.data.loader import load_price_matrix
        
        # Load price data
        prices = load_price_matrix(temp_price_file)
        
        # Create and fit strategy
        strategy = MomentumStrategy(lookback_period=3, top_n=3)
        strategy.fit(prices)
        
        # Get positions
        positions = strategy.get_positions(4)  # Last day
        
        # Verify positions
        assert len(positions) <= 3
        for instrument, value in positions.items():
            assert 0 < value <= 10000.0
        
        # Clean up
        temp_price_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__]) 