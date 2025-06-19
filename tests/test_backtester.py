"""
Tests for backtesting engine.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from backend.evaluation.backtester import Backtester, run_backtest
from backend.strategies.momentum import MomentumStrategy
from backend.strategies.base import TradingStrategy


class TestBacktester:
    """Test the Backtester class."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        n_days, n_instruments = 20, 60
        base_prices = 100 + np.random.normal(0, 10, n_instruments)
        
        prices = np.zeros((n_days, n_instruments))
        prices[0] = base_prices
        
        for day in range(1, n_days):
            returns = np.random.normal(0.001, 0.02, n_instruments)
            prices[day] = prices[day-1] * (1 + returns)
        
        return prices
    
    @pytest.fixture
    def strategy(self):
        """Create a strategy for testing."""
        return MomentumStrategy(
            name="TestMomentum",
            lookback_period=5,
            top_n=3,
            rebalance_frequency=2
        )
    
    @pytest.fixture
    def backtester(self):
        """Create a backtester instance."""
        return Backtester(
            commission_rate=0.0005,
            position_limit=10000.0,
            min_trade_threshold=100.0
        )
    
    def test_backtester_initialization(self, backtester):
        """Test backtester initialization."""
        assert backtester.commission_rate == 0.0005
        assert backtester.position_limit == 10000.0
        assert backtester.min_trade_threshold == 100.0
    
    def test_calculate_trades(self, backtester):
        """Test trade calculation."""
        current_positions = {"instrument_000": 1000.0, "instrument_001": 2000.0}
        target_positions = {"instrument_000": 1500.0, "instrument_001": 1000.0, "instrument_002": 500.0}
        
        trades = backtester._calculate_trades(current_positions, target_positions)
        
        assert trades["instrument_000"] == 500.0  # Buy 500
        assert trades["instrument_001"] == -1000.0  # Sell 1000
        assert trades["instrument_002"] == 500.0  # Buy 500
    
    def test_calculate_trades_small_threshold(self, backtester):
        """Test that small trades are filtered out."""
        current_positions = {"instrument_000": 1000.0}
        target_positions = {"instrument_000": 1050.0}  # Only 50 difference
        
        trades = backtester._calculate_trades(current_positions, target_positions)
        
        # Should be empty since 50 < min_trade_threshold (100)
        assert len(trades) == 0
    
    def test_execute_trades(self, backtester):
        """Test trade execution with position limits."""
        trades = {
            "instrument_000": 5000.0,  # Buy 5000
            "instrument_001": 15000.0,  # Buy 15000 (should be capped)
            "instrument_002": -2000.0   # Sell 2000
        }
        current_prices = np.array([100.0, 101.0, 102.0])
        
        executed_trades, commission = backtester._execute_trades(
            trades, current_prices, 0.001, 10000.0
        )
        
        # Should have 3 trades
        assert len(executed_trades) == 3
        
        # Check commission calculation
        expected_commission = (5000 + 10000 + 2000) * 0.001  # 17.0
        assert commission == expected_commission
        
        # Check that position limit was applied
        trade_values = [t['trade_value'] for t in executed_trades]
        assert max(trade_values) <= 10000.0
    
    def test_update_positions(self, backtester):
        """Test position updates."""
        current_positions = {"instrument_000": 1000.0, "instrument_001": 2000.0}
        executed_trades = [
            {"instrument": "instrument_000", "trade_value": 500.0},
            {"instrument": "instrument_001", "trade_value": -1000.0},
            {"instrument": "instrument_002", "trade_value": 300.0}
        ]
        
        updated_positions = backtester._update_positions(current_positions, executed_trades)
        
        assert updated_positions["instrument_000"] == 1500.0
        assert updated_positions["instrument_001"] == 1000.0
        assert updated_positions["instrument_002"] == 300.0
    
    def test_calculate_daily_pnl(self, backtester):
        """Test daily P&L calculation."""
        positions = {"instrument_000": 1000.0, "instrument_001": 2000.0}
        current_prices = np.array([100.0, 101.0])
        commission = 15.0
        
        pnl = backtester._calculate_daily_pnl(positions, current_prices, commission)
        
        # Simplified P&L calculation: total position value - commission
        expected_pnl = 3000.0 - 15.0
        assert pnl == expected_pnl
    
    def test_calculate_performance_metrics(self, backtester):
        """Test performance metrics calculation."""
        daily_pnl = [100.0, -50.0, 200.0, -100.0, 150.0]
        cum_pnl = [0.0, 100.0, 50.0, 250.0, 150.0, 300.0]
        
        metrics = backtester._calculate_performance_metrics(daily_pnl, cum_pnl)
        
        assert abs(metrics['mean'] - 60.0) < 1e-10  # (100-50+200-100+150)/5
        assert metrics['std'] > 0
        assert metrics['metric'] == metrics['mean'] - 0.1 * metrics['std']
        assert metrics['total_return'] == 300.0
        assert metrics['max_drawdown'] < 0  # Should be negative
    
    def test_calculate_performance_metrics_empty(self, backtester):
        """Test performance metrics with empty data."""
        metrics = backtester._calculate_performance_metrics([], [])
        
        assert metrics['mean'] == 0.0
        assert metrics['std'] == 0.0
        assert metrics['metric'] == 0.0
        assert metrics['max_drawdown'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0


class TestRunBacktest:
    """Test the run_backtest function."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        n_days, n_instruments = 15, 60
        base_prices = 100 + np.random.normal(0, 10, n_instruments)
        
        prices = np.zeros((n_days, n_instruments))
        prices[0] = base_prices
        
        for day in range(1, n_days):
            returns = np.random.normal(0.001, 0.02, n_instruments)
            prices[day] = prices[day-1] * (1 + returns)
        
        return prices
    
    @pytest.fixture
    def strategy(self):
        """Create a strategy for testing."""
        return MomentumStrategy(
            name="TestMomentum",
            lookback_period=3,
            top_n=2,
            rebalance_frequency=2
        )
    
    def test_run_backtest_basic(self, sample_prices, strategy):
        """Test basic backtest functionality."""
        results = run_backtest(sample_prices, strategy)
        
        # Check required keys
        required_keys = ['daily_pnl', 'cum_pnl', 'mean', 'std', 'metric']
        for key in required_keys:
            assert key in results
        
        # Check data types and basic properties
        assert isinstance(results['daily_pnl'], list)
        assert isinstance(results['cum_pnl'], list)
        assert isinstance(results['mean'], float)
        assert isinstance(results['std'], float)
        assert isinstance(results['metric'], float)
        
        # Check that we have results for each day
        assert len(results['daily_pnl']) == len(sample_prices)
        assert len(results['cum_pnl']) == len(sample_prices)
        
        # Check that metric is calculated correctly
        expected_metric = results['mean'] - 0.1 * results['std']
        assert abs(results['metric'] - expected_metric) < 1e-10
    
    def test_run_backtest_with_custom_params(self, sample_prices, strategy):
        """Test backtest with custom commission rate and position limit."""
        results = run_backtest(
            sample_prices, 
            strategy, 
            comm_rate=0.001,  # 10 bps
            pos_limit=5000.0   # $5,000 limit
        )
        
        # Check that custom parameters were used
        assert results['total_commission'] > 0
        assert results['total_trades'] > 0
        
        # Check position history for position limits
        for snapshot in results['position_history']:
            for position_value in snapshot['positions'].values():
                assert position_value <= 5000.0
    
    def test_run_backtest_with_dataframe(self, sample_prices, strategy):
        """Test backtest with pandas DataFrame input."""
        # Create DataFrame with datetime index
        dates = pd.date_range('2023-01-01', periods=len(sample_prices))
        df = pd.DataFrame(sample_prices, index=dates)
        
        results = run_backtest(df, strategy)
        
        # Should work the same as numpy array
        assert len(results['daily_pnl']) == len(sample_prices)
        assert len(results['cum_pnl']) == len(sample_prices)
        
        # Check that dates were preserved in position history
        for snapshot in results['position_history']:
            assert 'date' in snapshot
            assert isinstance(snapshot['date'], pd.Timestamp)
    
    def test_run_backtest_strategy_not_fitted(self, sample_prices):
        """Test that backtest fails with unfitted strategy."""
        strategy = MomentumStrategy(name="UnfittedStrategy")
        
        with pytest.raises(ValueError, match="must be fitted"):
            run_backtest(sample_prices, strategy)
    
    def test_run_backtest_invalid_prices(self, strategy):
        """Test that backtest fails with invalid price data."""
        # Test 1D array
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            run_backtest(np.array([1, 2, 3]), strategy)
        
        # Test empty array
        with pytest.raises(ValueError, match="cannot be empty"):
            run_backtest(np.array([]), strategy)
    
    def test_run_backtest_comprehensive_results(self, sample_prices, strategy):
        """Test that backtest returns comprehensive results."""
        results = run_backtest(sample_prices, strategy)
        
        # Check all expected keys
        expected_keys = [
            'daily_pnl', 'cum_pnl', 'mean', 'std', 'metric',
            'total_commission', 'total_trades', 'max_drawdown',
            'sharpe_ratio', 'position_history', 'trade_history',
            'final_pnl', 'total_return', 'volatility'
        ]
        
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        # Check data consistency
        assert len(results['position_history']) == len(sample_prices)
        assert results['final_pnl'] == results['cum_pnl'][-1]
        assert results['total_return'] == results['cum_pnl'][-1]
        
        # Check that commission is reasonable
        assert results['total_commission'] >= 0
        assert results['total_trades'] >= 0
        
        # Check that Sharpe ratio is calculated
        assert isinstance(results['sharpe_ratio'], float)
        assert not np.isnan(results['sharpe_ratio'])
    
    def test_run_backtest_position_tracking(self, sample_prices, strategy):
        """Test that position tracking works correctly."""
        results = run_backtest(sample_prices, strategy)
        
        # Check position history structure
        for i, snapshot in enumerate(results['position_history']):
            assert snapshot['day_idx'] == i
            assert 'positions' in snapshot
            assert 'total_value' in snapshot
            assert 'daily_pnl' in snapshot
            assert 'cumulative_pnl' in snapshot
            assert 'commission' in snapshot
            assert 'n_trades' in snapshot
            
            # Check that total_value matches sum of positions
            total_value = sum(snapshot['positions'].values())
            assert abs(snapshot['total_value'] - total_value) < 1e-10
    
    def test_run_backtest_trade_history(self, sample_prices, strategy):
        """Test that trade history is recorded correctly."""
        results = run_backtest(sample_prices, strategy)
        
        # Check trade history structure
        for trade in results['trade_history']:
            assert 'day_idx' in trade
            assert 'instrument' in trade
            assert 'trade_value' in trade
            assert 'commission' in trade
            assert 'position_before' in trade
            assert 'position_after' in trade
            
            # Check that commission is calculated correctly
            expected_commission = abs(trade['trade_value']) * 0.0005
            assert abs(trade['commission'] - expected_commission) < 1e-10


class TestBacktesterIntegration:
    """Test integration between backtester and data loading."""
    
    @pytest.fixture
    def temp_price_file(self):
        """Create a temporary price file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Create 10 days × 60 instruments
            np.random.seed(42)
            for day in range(10):
                prices = 100 + np.random.normal(0, 5, 60)
                line = ' '.join(f"{price:.2f}" for price in prices)
                f.write(line + '\n')
            f.flush()
            return Path(f.name)
    
    def test_backtester_with_loaded_data(self, temp_price_file):
        """Test backtester with data loaded from file."""
        from backend.data.loader import load_price_matrix
        
        # Load price data
        prices = load_price_matrix(temp_price_file)
        
        # Create strategy
        strategy = MomentumStrategy(lookback_period=3, top_n=3)
        
        # Run backtest
        results = run_backtest(prices, strategy)
        
        # Verify results
        assert len(results['daily_pnl']) == 10
        assert len(results['cum_pnl']) == 10
        assert results['total_trades'] >= 0
        assert results['total_commission'] >= 0
        
        # Clean up
        temp_price_file.unlink()
    
    def test_backtester_performance_metrics(self, temp_price_file):
        """Test that performance metrics are reasonable."""
        from backend.data.loader import load_price_matrix
        
        # Load price data
        prices = load_price_matrix(temp_price_file)
        
        # Create strategy
        strategy = MomentumStrategy(lookback_period=3, top_n=3)
        
        # Run backtest
        results = run_backtest(prices, strategy)
        
        # Check that metrics are reasonable
        assert isinstance(results['mean'], float)
        assert isinstance(results['std'], float)
        assert isinstance(results['metric'], float)
        assert isinstance(results['sharpe_ratio'], float)
        assert isinstance(results['max_drawdown'], float)
        assert isinstance(results['volatility'], float)
        
        # Check that metric calculation is correct
        expected_metric = results['mean'] - 0.1 * results['std']
        assert abs(results['metric'] - expected_metric) < 1e-10
        
        # Clean up
        temp_price_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__]) 