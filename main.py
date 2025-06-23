import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

nInst = 50
currentPos = np.zeros(nInst)

# Global variables for strategy state
strategy_state = {
    'volatility_lookback': 20,
    'momentum_lookback': 10,
    'mean_reversion_lookback': 15,
    'risk_aversion': 0.1,
    'max_position_value': 10000,
    'min_volatility': 0.001,
    'max_volatility': 0.1,
    'correlation_threshold': 0.7,
    'trend_strength_threshold': 0.6,
    'position_smoothing': 0.8
}

def calculate_returns(prices):
    """Calculate log returns"""
    return np.diff(np.log(prices), axis=1)

def calculate_volatility(prices, lookback=20):
    """Calculate rolling volatility"""
    if prices.shape[1] < lookback + 1:
        return np.ones(prices.shape[0]) * 0.02
    
    returns = calculate_returns(prices)
    volatility = np.std(returns[:, -lookback:], axis=1)
    return np.maximum(volatility, 0.001)

def calculate_momentum_signal(prices, short_lookback=5, long_lookback=20):
    """Calculate momentum signal using multiple timeframes"""
    if prices.shape[1] < long_lookback + 1:
        return np.zeros(prices.shape[0])
    
    # Short-term momentum (5 days)
    short_momentum = np.log(prices[:, -1] / prices[:, -short_lookback])
    
    # Long-term momentum (20 days)
    long_momentum = np.log(prices[:, -1] / prices[:, -long_lookback])
    
    # Combine with weights
    momentum_signal = 0.7 * short_momentum + 0.3 * long_momentum
    
    # Normalize by volatility
    volatility = calculate_volatility(prices, short_lookback)
    momentum_signal = momentum_signal / (volatility + 1e-8)
    
    return momentum_signal

def calculate_mean_reversion_signal(prices, lookback=15):
    """Calculate mean reversion signal using Bollinger Bands"""
    if prices.shape[1] < lookback + 1:
        return np.zeros(prices.shape[0])
    
    # Calculate Bollinger Bands
    sma = np.mean(prices[:, -lookback:], axis=1)
    std = np.std(prices[:, -lookback:], axis=1)
    
    current_prices = prices[:, -1]
    
    # Z-score for mean reversion
    z_score = (current_prices - sma) / (std + 1e-8)
    
    # Mean reversion signal (stronger at extremes)
    mean_reversion_signal = -z_score * np.exp(-np.abs(z_score))
    
    return mean_reversion_signal

def calculate_trend_signal(prices, lookback=20):
    """Calculate trend signal using linear regression"""
    if prices.shape[1] < lookback + 1:
        return np.zeros(prices.shape[0])
    
    trend_signal = np.zeros(prices.shape[0])
    
    for i in range(prices.shape[0]):
        y = prices[i, -lookback:]
        x = np.arange(lookback)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Use slope and R-squared for trend signal
        trend_signal[i] = slope * r_value**2
    
    return trend_signal

def calculate_volatility_signal(prices, lookback=20):
    """Calculate volatility-based signal"""
    if prices.shape[1] < lookback + 1:
        return np.zeros(prices.shape[0])
    
    # Calculate rolling volatility
    returns = calculate_returns(prices)
    volatility = np.std(returns[:, -lookback:], axis=1)
    
    # Volatility mean reversion (high volatility tends to mean revert)
    avg_volatility = np.mean(volatility)
    volatility_signal = (avg_volatility - volatility) / (avg_volatility + 1e-8)
    
    return volatility_signal

def detect_market_regime(prices):
    """Detect market regime (trending vs mean-reverting)"""
    if prices.shape[1] < 30:
        return 'normal'
    
    # Calculate trend strength
    trend_strength = calculate_trend_signal(prices, 20)
    avg_trend = np.mean(np.abs(trend_strength))
    
    # Calculate volatility
    volatility = calculate_volatility(prices, 20)
    avg_volatility = np.mean(volatility)
    
    # Market regime classification
    if avg_trend > 0.1 and avg_volatility < 0.03:
        return 'trending'
    elif avg_volatility > 0.05:
        return 'volatile'
    else:
        return 'normal'

def calculate_position_sizes(signals, prices, regime):
    """Calculate position sizes based on signals and market regime"""
    current_prices = prices[:, -1]
    volatility = calculate_volatility(prices, 20)
    
    # Base position size
    base_size = 1000  # Base dollar amount
    
    # Adjust for volatility
    volatility_adjustment = 1 / (volatility + 1e-8)
    volatility_adjustment = np.clip(volatility_adjustment, 0.1, 10)
    
    # Regime-specific adjustments
    if regime == 'trending':
        regime_multiplier = 1.5
    elif regime == 'volatile':
        regime_multiplier = 0.5
    else:
        regime_multiplier = 1.0
    
    # Calculate position sizes
    position_sizes = signals * base_size * volatility_adjustment * regime_multiplier / current_prices
    
    # Apply position limits ($10k per instrument)
    max_positions = 10000 / current_prices
    position_sizes = np.clip(position_sizes, -max_positions, max_positions)
    
    return position_sizes

def apply_risk_management(positions, prices):
    """Apply risk management rules"""
    current_prices = prices[:, -1]
    
    # Calculate total portfolio value
    portfolio_value = np.abs(positions * current_prices).sum()
    
    # Maximum portfolio risk (5% of total portfolio)
    max_risk = portfolio_value * 0.05
    
    # Calculate individual position risks
    volatility = calculate_volatility(prices, 20)
    position_risks = np.abs(positions * current_prices * volatility)
    
    # Scale down if total risk exceeds limit
    if position_risks.sum() > max_risk:
        scaling_factor = max_risk / position_risks.sum()
        positions = positions * scaling_factor
    
    # Ensure position limits
    max_positions = 10000 / current_prices
    positions = np.clip(positions, -max_positions, max_positions)
    
    return positions

def smooth_positions(new_positions, current_positions, alpha=0.7):
    """Smooth position changes to reduce transaction costs"""
    smoothed = alpha * current_positions + (1 - alpha) * new_positions
    return smoothed.astype(int)

def getMyPosition(prcSoFar):
    """
    Advanced trading algorithm with multiple strategies and risk management
    
    Args:
        prcSoFar: numpy array of shape (nInst, nt) containing price history
        
    Returns:
        numpy array of integer positions for each instrument
    """
    global currentPos
    
    (nins, nt) = prcSoFar.shape
    
    # Need at least 2 days of data
    if nt < 2:
        return np.zeros(nins)
    
    # Detect market regime
    regime = detect_market_regime(prcSoFar)
    
    # Calculate various signals
    momentum_signal = calculate_momentum_signal(prcSoFar)
    mean_reversion_signal = calculate_mean_reversion_signal(prcSoFar)
    trend_signal = calculate_trend_signal(prcSoFar)
    volatility_signal = calculate_volatility_signal(prcSoFar)
    
    # Combine signals based on market regime
    if regime == 'trending':
        # In trending markets, favor momentum and trend signals
        combined_signal = (0.5 * momentum_signal + 
                          0.3 * trend_signal + 
                          0.1 * mean_reversion_signal + 
                          0.1 * volatility_signal)
    elif regime == 'volatile':
        # In volatile markets, favor mean reversion and volatility signals
        combined_signal = (0.2 * momentum_signal + 
                          0.1 * trend_signal + 
                          0.5 * mean_reversion_signal + 
                          0.2 * volatility_signal)
    else:
        # In normal markets, use balanced approach
        combined_signal = (0.4 * momentum_signal + 
                          0.2 * trend_signal + 
                          0.3 * mean_reversion_signal + 
                          0.1 * volatility_signal)
    
    # Normalize combined signal
    signal_norm = np.sqrt(combined_signal.dot(combined_signal))
    if signal_norm > 0:
        combined_signal = combined_signal / signal_norm
    
    # Calculate position sizes
    target_positions = calculate_position_sizes(combined_signal, prcSoFar, regime)
    
    # Apply risk management
    target_positions = apply_risk_management(target_positions, prcSoFar)
    
    # Smooth position changes
    currentPos = smooth_positions(target_positions, currentPos)
    
    # Ensure positions are integers
    currentPos = currentPos.astype(int)
    
    return currentPos
