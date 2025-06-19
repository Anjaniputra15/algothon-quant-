#!/usr/bin/env python3
"""
Demonstration script for trading strategies.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.strategies.base import TradingStrategy
from backend.strategies.momentum import MomentumStrategy
from backend.data.loader import load_price_matrix


def demo_base_strategy():
    """Demonstrate the base TradingStrategy class."""
    print("=" * 60)
    print("DEMONSTRATION: Base TradingStrategy Class")
    print("=" * 60)
    
    # Create a base strategy (this will fail since it's abstract)
    print("Creating base strategy (should fail since it's abstract):")
    try:
        strategy = TradingStrategy(name="TestStrategy")
        print("SUCCESS: Base strategy created")
    except TypeError as e:
        print(f"EXPECTED ERROR: {e}")
    
    print("\nBase strategy features:")
    print("- Maximum position value: $10,000 per stock")
    print("- Commission rate: 10 basis points (0.001)")
    print("- Long-only positions (no short selling)")
    print("- Automatic position constraint application")
    print("- Trade history tracking")
    print("- Performance metrics calculation")


def demo_momentum_strategy():
    """Demonstrate the MomentumStrategy implementation."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: MomentumStrategy Implementation")
    print("=" * 60)
    
    # Create sample price data
    print("Creating sample price data...")
    np.random.seed(42)
    n_days, n_instruments = 30, 60
    base_prices = 100 + np.random.normal(0, 10, n_instruments)
    
    prices = np.zeros((n_days, n_instruments))
    prices[0] = base_prices
    
    for day in range(1, n_days):
        # Add some momentum behavior
        returns = np.random.normal(0.001, 0.02, n_instruments)
        prices[day] = prices[day-1] * (1 + returns)
    
    print(f"Generated {n_days} days × {n_instruments} instruments of price data")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Create momentum strategy
    print("\nCreating momentum strategy...")
    strategy = MomentumStrategy(
        name="DemoMomentum",
        lookback_period=10,
        top_n=5,
        rebalance_frequency=3
    )
    
    print(f"Strategy: {strategy}")
    print(f"Lookback period: {strategy.lookback_period} days")
    print(f"Top N instruments: {strategy.top_n}")
    print(f"Rebalance frequency: {strategy.rebalance_frequency} days")
    
    # Fit the strategy
    print("\nFitting strategy to historical data...")
    strategy.fit(prices)
    print(f"Strategy fitted: {strategy.is_fitted}")
    print(f"Data shape: {strategy.n_days} days × {strategy.n_instruments} instruments")
    
    # Get positions for different days
    print("\nGetting positions for different days:")
    for day in [10, 15, 20, 25]:
        positions = strategy.get_positions(day)
        total_value = sum(positions.values())
        commission = sum(strategy.calculate_commission(pos) for pos in positions.values())
        
        print(f"Day {day}: {len(positions)} positions, "
              f"total value: ${total_value:,.2f}, "
              f"commission: ${commission:.2f}")
        
        if positions:
            print(f"  Top positions: {list(positions.items())[:3]}")
    
    # Update strategy with new data
    print("\nUpdating strategy with new data...")
    new_row = prices[25] + np.random.normal(0, 1, n_instruments)
    strategy.update(26, new_row)
    
    # Get performance summary
    print("\nPerformance summary:")
    summary = strategy.get_performance_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Get strategy-specific info
    print("\nStrategy-specific information:")
    info = strategy.get_strategy_info()
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def demo_constraints():
    """Demonstrate trading constraints."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Trading Constraints")
    print("=" * 60)
    
    # Create a strategy to test constraints
    strategy = MomentumStrategy(name="ConstraintTest")
    
    print("Testing position constraints:")
    
    # Test normal positions
    positions = {"instrument_000": 5000.0, "instrument_001": 8000.0}
    constrained = strategy.apply_position_constraints(positions)
    print(f"Normal positions: {positions} -> {constrained}")
    
    # Test position exceeding cap
    positions = {"instrument_000": 15000.0, "instrument_001": 8000.0}
    constrained = strategy.apply_position_constraints(positions)
    print(f"Exceeding cap: {positions} -> {constrained}")
    
    # Test negative positions
    positions = {"instrument_000": -1000.0, "instrument_001": 5000.0}
    constrained = strategy.apply_position_constraints(positions)
    print(f"Negative positions: {positions} -> {constrained}")
    
    # Test commission calculation
    print(f"\nCommission calculation (10 bps):")
    for value in [1000, 5000, 10000, 50000]:
        commission = strategy.calculate_commission(value)
        print(f"  ${value:,.0f} position -> ${commission:.2f} commission")


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
        
        # Create and fit strategy
        strategy = MomentumStrategy(lookback_period=3, top_n=3)
        strategy.fit(prices)
        
        # Get positions
        positions = strategy.get_positions(4)  # Last day
        print(f"Generated positions: {positions}")
        
        # Performance summary
        summary = strategy.get_performance_summary()
        print(f"Total trades: {summary['total_trades']}")
        print(f"Total commission: ${summary['total_commission_paid']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all demonstrations."""
    print("ALGOTHON-QUANT TRADING STRATEGIES DEMONSTRATION")
    print("=" * 60)
    
    try:
        demo_base_strategy()
        demo_momentum_strategy()
        demo_constraints()
        demo_integration_with_loader()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Abstract base class with standardized interface")
        print("✓ $10,000 per-stock position cap")
        print("✓ 10 basis points (0.001) commission rate")
        print("✓ Long-only position constraints")
        print("✓ Trade history tracking")
        print("✓ Performance metrics calculation")
        print("✓ Momentum strategy implementation")
        print("✓ Integration with data loader")
        
    except Exception as e:
        print(f"\nDEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 