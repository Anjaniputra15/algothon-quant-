import numpy as np
import pandas as pd
from main import getMyPosition
import time
import warnings
warnings.filterwarnings('ignore')

def generate_test_data(nInst=50, nt=252):
    """Generate realistic test price data"""
    np.random.seed(42)
    
    # Generate correlated price movements
    base_trend = np.cumsum(np.random.randn(nt) * 0.01)
    
    prices = np.zeros((nInst, nt))
    for i in range(nInst):
        # Individual stock trend
        stock_trend = np.cumsum(np.random.randn(nt) * 0.02)
        # Market correlation
        market_component = base_trend * (0.3 + 0.4 * np.random.random())
        # Idiosyncratic noise
        noise = np.cumsum(np.random.randn(nt) * 0.015)
        
        # Combine components
        returns = stock_trend + market_component + noise
        prices[i] = 100 * np.exp(returns)
    
    return prices

def calculate_returns(prices):
    """Calculate daily returns"""
    return np.diff(prices, axis=1) / prices[:, :-1]

def calculate_metrics(positions, prices, commission_rate=0.0005):
    """Calculate performance metrics"""
    returns = calculate_returns(prices)
    
    # Calculate P&L
    position_returns = positions.reshape(-1, 1) * returns
    daily_pnl = np.sum(position_returns, axis=0)
    
    # Calculate transaction costs
    position_changes = np.diff(positions.reshape(-1, 1), axis=1, prepend=positions.reshape(-1, 1))
    transaction_volume = np.abs(position_changes) * prices
    transaction_costs = np.sum(transaction_volume, axis=0) * commission_rate
    
    # Ensure transaction costs has same length as daily_pnl
    if len(transaction_costs) > len(daily_pnl):
        transaction_costs = transaction_costs[:len(daily_pnl)]
    elif len(transaction_costs) < len(daily_pnl):
        # Pad with zeros if needed
        transaction_costs = np.pad(transaction_costs, (0, len(daily_pnl) - len(transaction_costs)), 'constant')
    
    # Net P&L
    net_pnl = daily_pnl - transaction_costs
    
    # Calculate metrics
    mean_pl = np.mean(net_pnl)
    std_pl = np.std(net_pnl)
    sharpe = mean_pl / (std_pl + 1e-8)
    
    # Competition metric: mean(PL) - 0.1 * StdDev(PL)
    competition_score = mean_pl - 0.1 * std_pl
    
    return {
        'mean_pl': mean_pl,
        'std_pl': std_pl,
        'sharpe': sharpe,
        'competition_score': competition_score,
        'total_return': np.sum(net_pnl),
        'max_drawdown': calculate_max_drawdown(net_pnl),
        'win_rate': np.sum(net_pnl > 0) / len(net_pnl)
    }

def calculate_max_drawdown(pnl):
    """Calculate maximum drawdown"""
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return np.min(drawdown)

def test_position_limits(positions, prices, limit=10000):
    """Test if positions respect $10k limit"""
    current_prices = prices[:, -1]
    dollar_positions = positions * current_prices
    
    violations = np.abs(dollar_positions) > limit
    return {
        'violations': np.sum(violations),
        'max_position': np.max(np.abs(dollar_positions)),
        'avg_position': np.mean(np.abs(dollar_positions))
    }

def test_algorithm():
    """Comprehensive algorithm testing"""
    print("🚀 Testing Algothon Trading Algorithm")
    print("=" * 50)
    
    # Test 1: Basic functionality
    print("\n1. Testing basic functionality...")
    nInst, nt = 50, 100
    test_prices = generate_test_data(nInst, nt)
    
    start_time = time.time()
    positions = getMyPosition(test_prices)
    execution_time = time.time() - start_time
    
    print(f"✅ Execution time: {execution_time:.3f} seconds")
    print(f"✅ Positions shape: {positions.shape}")
    print(f"✅ Position range: {positions.min()} to {positions.max()}")
    print(f"✅ Non-zero positions: {np.count_nonzero(positions)}")
    
    # Test 2: Position limits
    print("\n2. Testing position limits...")
    limit_test = test_position_limits(positions, test_prices)
    print(f"✅ Max position: ${limit_test['max_position']:.2f}")
    print(f"✅ Average position: ${limit_test['avg_position']:.2f}")
    print(f"✅ Limit violations: {limit_test['violations']}")
    
    if limit_test['violations'] == 0:
        print("✅ All positions within $10k limit!")
    else:
        print("❌ Position limit violations detected!")
    
    # Test 3: Performance metrics
    print("\n3. Testing performance metrics...")
    metrics = calculate_metrics(positions, test_prices)
    
    print(f"✅ Mean P&L: ${metrics['mean_pl']:.2f}")
    print(f"✅ P&L Std Dev: ${metrics['std_pl']:.2f}")
    print(f"✅ Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"✅ Competition Score: {metrics['competition_score']:.3f}")
    print(f"✅ Total Return: ${metrics['total_return']:.2f}")
    print(f"✅ Max Drawdown: ${metrics['max_drawdown']:.2f}")
    print(f"✅ Win Rate: {metrics['win_rate']:.1%}")
    
    # Test 4: Multiple time periods
    print("\n4. Testing multiple time periods...")
    periods = [50, 100, 200, 500]
    
    for nt in periods:
        test_prices = generate_test_data(50, nt)
        positions = getMyPosition(test_prices)
        metrics = calculate_metrics(positions, test_prices)
        print(f"   {nt} days: Score = {metrics['competition_score']:.3f}, Sharpe = {metrics['sharpe']:.3f}")
    
    # Test 5: Runtime performance
    print("\n5. Testing runtime performance...")
    large_prices = generate_test_data(50, 1000)
    
    start_time = time.time()
    positions = getMyPosition(large_prices)
    execution_time = time.time() - start_time
    
    print(f"✅ 1000-day execution time: {execution_time:.3f} seconds")
    
    if execution_time < 600:  # 10 minutes
        print("✅ Runtime within 10-minute limit!")
    else:
        print("❌ Runtime exceeds 10-minute limit!")
    
    # Test 6: Edge cases
    print("\n6. Testing edge cases...")
    
    # Very short data
    short_prices = generate_test_data(50, 5)
    try:
        positions = getMyPosition(short_prices)
        print("✅ Handles short data correctly")
    except Exception as e:
        print(f"❌ Error with short data: {e}")
    
    # Extreme price movements
    extreme_prices = generate_test_data(50, 100)
    extreme_prices *= np.random.uniform(0.1, 10, extreme_prices.shape)
    try:
        positions = getMyPosition(extreme_prices)
        print("✅ Handles extreme price movements")
    except Exception as e:
        print(f"❌ Error with extreme prices: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Algorithm Testing Complete!")
    
    return metrics

if __name__ == "__main__":
    test_algorithm() 