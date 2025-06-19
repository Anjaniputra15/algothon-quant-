#!/usr/bin/env python3
"""
Demonstration script for backtesting engine.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.evaluation.backtester import run_backtest, Backtester
from backend.strategies.momentum import MomentumStrategy
from backend.data.loader import load_price_matrix


def demo_basic_backtest():
    """Demonstrate basic backtesting functionality."""
    print("=" * 60)
    print("DEMONSTRATION: Basic Backtesting")
    print("=" * 60)
    
    # Create sample price data
    print("Creating sample price data...")
    np.random.seed(42)
    n_days, n_instruments = 30, 60
    base_prices = 100 + np.random.normal(0, 10, n_instruments)
    
    prices = np.zeros((n_days, n_instruments))
    prices[0] = base_prices
    
    for day in range(1, n_days):
        returns = np.random.normal(0.001, 0.02, n_instruments)
        prices[day] = prices[day-1] * (1 + returns)
    
    print(f"Generated {n_days} days × {n_instruments} instruments of price data")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Create strategy
    print("\nCreating momentum strategy...")
    strategy = MomentumStrategy(
        name="DemoMomentum",
        lookback_period=10,
        top_n=5,
        rebalance_frequency=3
    )
    
    # Fit the strategy before running backtest
    strategy.fit(prices)
    
    # Run backtest
    print("\nRunning backtest...")
    results = run_backtest(prices, strategy)
    
    # Display results
    print("\nBACKTEST RESULTS:")
    print("-" * 30)
    print(f"Final P&L: ${results['final_pnl']:,.2f}")
    print(f"Total Return: ${results['total_return']:,.2f}")
    print(f"Total Commission: ${results['total_commission']:,.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Mean Daily Return: {results['mean']:,.4f}")
    print(f"Standard Deviation: {results['std']:,.4f}")
    print(f"Risk-Adjusted Metric: {results['metric']:,.4f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:,.4f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:,.2f}")
    print(f"Volatility (Annualized): {results['volatility']:,.4f}")


def demo_custom_parameters():
    """Demonstrate backtesting with custom parameters."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Custom Backtest Parameters")
    print("=" * 60)
    
    # Create sample price data
    np.random.seed(42)
    n_days, n_instruments = 20, 60
    base_prices = 100 + np.random.normal(0, 10, n_instruments)
    
    prices = np.zeros((n_days, n_instruments))
    prices[0] = base_prices
    
    for day in range(1, n_days):
        returns = np.random.normal(0.001, 0.02, n_instruments)
        prices[day] = prices[day-1] * (1 + returns)
    
    # Create strategy
    strategy = MomentumStrategy(
        name="CustomMomentum",
        lookback_period=5,
        top_n=3,
        rebalance_frequency=2
    )
    
    # Test different parameter combinations
    parameter_sets = [
        {"comm_rate": 0.0005, "pos_limit": 10000.0, "name": "Default (5 bps, $10k)"},
        {"comm_rate": 0.0010, "pos_limit": 10000.0, "name": "High Commission (10 bps, $10k)"},
        {"comm_rate": 0.0005, "pos_limit": 5000.0, "name": "Low Position Limit (5 bps, $5k)"},
        {"comm_rate": 0.0001, "pos_limit": 20000.0, "name": "Low Commission (1 bps, $20k)"}
    ]
    
    print("Testing different parameter combinations:")
    print("-" * 50)
    
    for params in parameter_sets:
        print(f"\n{params['name']}:")
        
        # Fit the strategy before each backtest
        strategy.fit(prices)
        results = run_backtest(
            prices, 
            strategy, 
            comm_rate=params['comm_rate'],
            pos_limit=params['pos_limit']
        )
        
        print(f"  Final P&L: ${results['final_pnl']:,.2f}")
        print(f"  Total Commission: ${results['total_commission']:,.2f}")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Risk-Adjusted Metric: {results['metric']:,.4f}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:,.4f}")


def demo_backtester_class():
    """Demonstrate using the Backtester class directly."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Backtester Class")
    print("=" * 60)
    
    # Create sample price data
    np.random.seed(42)
    n_days, n_instruments = 15, 60
    base_prices = 100 + np.random.normal(0, 10, n_instruments)
    
    prices = np.zeros((n_days, n_instruments))
    prices[0] = base_prices
    
    for day in range(1, n_days):
        returns = np.random.normal(0.001, 0.02, n_instruments)
        prices[day] = prices[day-1] * (1 + returns)
    
    # Create strategy
    strategy = MomentumStrategy(
        name="ClassMomentum",
        lookback_period=3,
        top_n=2,
        rebalance_frequency=2
    )
    
    # Create backtester with custom settings
    print("Creating backtester with custom settings...")
    backtester = Backtester(
        commission_rate=0.0003,  # 3 basis points
        position_limit=8000.0,   # $8,000 limit
        min_trade_threshold=200.0  # $200 minimum trade
    )
    
    print(f"Commission rate: {backtester.commission_rate:.4f}")
    print(f"Position limit: ${backtester.position_limit:,.0f}")
    print(f"Min trade threshold: ${backtester.min_trade_threshold:,.0f}")
    
    # Fit the strategy before running backtest
    strategy.fit(prices)
    
    # Run backtest
    print("\nRunning backtest with custom backtester...")
    results = backtester.run_backtest(prices, strategy)
    
    # Display results
    print("\nBACKTEST RESULTS:")
    print("-" * 30)
    print(f"Final P&L: ${results['final_pnl']:,.2f}")
    print(f"Total Commission: ${results['total_commission']:,.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Risk-Adjusted Metric: {results['metric']:,.4f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:,.4f}")
    
    # Show position history
    print(f"\nPosition history entries: {len(results['position_history'])}")
    if results['position_history']:
        last_snapshot = results['position_history'][-1]
        print(f"Final day positions: {len(last_snapshot['positions'])} instruments")
        print(f"Final total value: ${last_snapshot['total_value']:,.2f}")
    
    # Show trade history
    print(f"\nTrade history entries: {len(results['trade_history'])}")
    if results['trade_history']:
        avg_trade_value = sum(abs(t['trade_value']) for t in results['trade_history']) / len(results['trade_history'])
        print(f"Average trade value: ${avg_trade_value:,.2f}")


def demo_dataframe_input():
    """Demonstrate backtesting with pandas DataFrame input."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: DataFrame Input")
    print("=" * 60)
    
    # Create sample price data with datetime index
    np.random.seed(42)
    n_days, n_instruments = 10, 60
    base_prices = 100 + np.random.normal(0, 10, n_instruments)
    
    prices = np.zeros((n_days, n_instruments))
    prices[0] = base_prices
    
    for day in range(1, n_days):
        returns = np.random.normal(0.001, 0.02, n_instruments)
        prices[day] = prices[day-1] * (1 + returns)
    
    # Create DataFrame with datetime index
    dates = pd.date_range('2023-01-01', periods=n_days)
    df = pd.DataFrame(prices, index=dates)
    
    print(f"Created DataFrame with {len(df)} days × {len(df.columns)} instruments")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create strategy
    strategy = MomentumStrategy(
        name="DataFrameMomentum",
        lookback_period=3,
        top_n=2,
        rebalance_frequency=2
    )
    
    # Run backtest
    print("\nRunning backtest with DataFrame...")
    strategy.fit(df.values)
    results = run_backtest(df, strategy)
    
    # Display results
    print("\nBACKTEST RESULTS:")
    print("-" * 30)
    print(f"Final P&L: ${results['final_pnl']:,.2f}")
    print(f"Total Commission: ${results['total_commission']:,.2f}")
    print(f"Total Trades: {results['total_trades']}")
    
    # Check that dates were preserved
    print(f"\nDate preservation check:")
    for i, snapshot in enumerate(results['position_history'][:3]):  # First 3 days
        print(f"  Day {i}: {snapshot['date']}")


def demo_integration_with_loader():
    """Demonstrate integration with data loader."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Integration with Data Loader")
    print("=" * 60)
    
    # Check if prices.txt exists
    sample_file = Path("prices.txt")
    if not sample_file.exists():
        print("prices.txt not found. Please ensure the file exists in the current directory.")
        return
    
    print(f"Loading price data from {sample_file}...")
    try:
        prices = load_price_matrix(sample_file)
        print(f"Successfully loaded: {prices.shape[0]} days × {prices.shape[1]} instruments")
        
        # Create strategy
        strategy = MomentumStrategy(lookback_period=3, top_n=3)
        strategy.fit(prices)
        
        # Run backtest
        print("\nRunning backtest with loaded data...")
        results = run_backtest(prices, strategy)
        
        # Display results
        print("\nBACKTEST RESULTS:")
        print("-" * 30)
        print(f"Final P&L: ${results['final_pnl']:,.2f}")
        print(f"Total Commission: ${results['total_commission']:,.2f}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Risk-Adjusted Metric: {results['metric']:,.4f}")
        
    except Exception as e:
        print(f"Error: {e}")


def demo_performance_analysis():
    """Demonstrate performance analysis features."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Performance Analysis")
    print("=" * 60)
    
    # Create sample price data
    np.random.seed(42)
    n_days, n_instruments = 25, 60
    base_prices = 100 + np.random.normal(0, 10, n_instruments)
    
    prices = np.zeros((n_days, n_instruments))
    prices[0] = base_prices
    
    for day in range(1, n_days):
        returns = np.random.normal(0.001, 0.02, n_instruments)
        prices[day] = prices[day-1] * (1 + returns)
    
    # Create strategy
    strategy = MomentumStrategy(
        name="AnalysisMomentum",
        lookback_period=5,
        top_n=4,
        rebalance_frequency=3
    )
    
    # Fit the strategy before running backtest
    strategy.fit(prices)
    
    # Run backtest
    results = run_backtest(prices, strategy)
    
    # Analyze performance
    print("PERFORMANCE ANALYSIS:")
    print("-" * 30)
    
    # Daily P&L analysis
    daily_pnl = results['daily_pnl']
    print(f"Daily P&L Statistics:")
    print(f"  Mean: ${np.mean(daily_pnl):,.2f}")
    print(f"  Std Dev: ${np.std(daily_pnl):,.2f}")
    print(f"  Min: ${np.min(daily_pnl):,.2f}")
    print(f"  Max: ${np.max(daily_pnl):,.2f}")
    print(f"  Positive days: {sum(1 for pnl in daily_pnl if pnl > 0)}/{len(daily_pnl)}")
    
    # Cumulative P&L analysis
    cum_pnl = results['cum_pnl']
    print(f"\nCumulative P&L Analysis:")
    print(f"  Final value: ${cum_pnl[-1]:,.2f}")
    print(f"  Peak value: ${max(cum_pnl):,.2f}")
    print(f"  Maximum drawdown: ${results['max_drawdown']:,.2f}")
    
    # Trade analysis
    trades = results['trade_history']
    if trades:
        print(f"\nTrade Analysis:")
        print(f"  Total trades: {len(trades)}")
        print(f"  Average trade value: ${sum(abs(t['trade_value']) for t in trades) / len(trades):,.2f}")
        print(f"  Average commission: ${results['total_commission'] / len(trades):,.2f}")
        
        # Analyze trade sizes
        trade_values = [abs(t['trade_value']) for t in trades]
        print(f"  Largest trade: ${max(trade_values):,.2f}")
        print(f"  Smallest trade: ${min(trade_values):,.2f}")
    
    # Risk metrics
    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:,.4f}")
    print(f"  Volatility (Annualized): {results['volatility']:,.4f}")
    print(f"  Risk-Adjusted Metric: {results['metric']:,.4f}")


def main():
    """Run all demonstrations."""
    print("ALGOTHON-QUANT BACKTESTING ENGINE DEMONSTRATION")
    print("=" * 60)
    
    try:
        demo_basic_backtest()
        demo_custom_parameters()
        demo_backtester_class()
        demo_dataframe_input()
        demo_integration_with_loader()
        demo_performance_analysis()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Day-by-day backtesting simulation")
        print("✓ Position tracking and trade calculation")
        print("✓ Commission calculation (5 bps default)")
        print("✓ Position size capping ($10,000 default)")
        print("✓ Performance metrics calculation")
        print("✓ Risk-adjusted return metric (mean - 0.1 × std)")
        print("✓ Custom parameter support")
        print("✓ DataFrame input support")
        print("✓ Integration with data loader")
        print("✓ Comprehensive performance analysis")
        
    except Exception as e:
        print(f"\nDEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 