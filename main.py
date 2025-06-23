import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

class AdvancedTradingAlgorithm:
    def __init__(self):
        self.position_limit = 10000  # $10k limit
        self.commission_rate = 0.0005  # 5 bps
        self.lookback_short = 3
        self.lookback_medium = 10
        self.lookback_long = 20
        self.volatility_window = 15
        self.correlation_window = 20
        self.max_positions = 15  # Max number of active positions
        
    def calculate_returns(self, prices):
        """Calculate log returns"""
        return np.diff(np.log(prices), axis=1)
    
    def calculate_volatility(self, returns):
        """Calculate rolling volatility"""
        return np.std(returns, axis=1, ddof=1)
    
    def calculate_momentum(self, prices, lookback):
        """Calculate momentum signal"""
        if prices.shape[1] < lookback + 1:
            return np.zeros(prices.shape[0])
        
        momentum = (prices[:, -1] / prices[:, -lookback-1]) - 1
        return momentum
    
    def calculate_mean_reversion(self, prices, lookback):
        """Calculate mean reversion signal using z-score"""
        if prices.shape[1] < lookback:
            return np.zeros(prices.shape[0])
        
        # Calculate rolling mean and std
        rolling_mean = np.mean(prices[:, -lookback:], axis=1)
        rolling_std = np.std(prices[:, -lookback:], axis=1, ddof=1)
        
        # Avoid division by zero
        rolling_std = np.where(rolling_std == 0, 1e-8, rolling_std)
        
        # Z-score for mean reversion
        z_score = (prices[:, -1] - rolling_mean) / rolling_std
        return -z_score  # Negative because we want to buy when price is low
    
    def calculate_volatility_regime(self, returns):
        """Detect high/low volatility regimes"""
        if returns.shape[1] < self.volatility_window:
            return np.ones(returns.shape[0])
        
        recent_vol = np.std(returns[:, -self.volatility_window:], axis=1, ddof=1)
        historical_vol = np.std(returns, axis=1, ddof=1)
        
        # Volatility regime indicator (1 for high vol, 0.5 for normal, 0.2 for low vol)
        vol_ratio = recent_vol / (historical_vol + 1e-8)
        regime = np.where(vol_ratio > 1.5, 0.2, np.where(vol_ratio < 0.7, 1.0, 0.5))
        return regime
    
    def calculate_correlation_risk(self, returns):
        """Calculate correlation-based risk adjustment"""
        if returns.shape[1] < self.correlation_window:
            return np.ones(returns.shape[0])
        
        # Calculate correlation matrix
        recent_returns = returns[:, -self.correlation_window:]
        corr_matrix = np.corrcoef(recent_returns)
        
        # Average correlation for each instrument
        avg_corr = np.mean(np.abs(corr_matrix), axis=1)
        
        # Risk adjustment: higher correlation = lower position size
        risk_adj = 1 - (avg_corr - np.min(avg_corr)) / (np.max(avg_corr) - np.min(avg_corr) + 1e-8)
        return risk_adj
    
    def calculate_trend_strength(self, prices, lookback):
        """Calculate trend strength using linear regression"""
        if prices.shape[1] < lookback:
            return np.zeros(prices.shape[0])
        
        trend_strength = np.zeros(prices.shape[0])
        x = np.arange(lookback)
        
        for i in range(prices.shape[0]):
            y = prices[i, -lookback:]
            if len(y) == lookback:
                slope, _, r_value, _, _ = stats.linregress(x, y)
                trend_strength[i] = slope * r_value**2  # Slope weighted by R-squared
        
        return trend_strength
    
    def calculate_volume_weighted_signal(self, prices):
        """Calculate volume-weighted price signal (simulated)"""
        if prices.shape[1] < 10:
            return np.zeros(prices.shape[0])
        
        # Simulate volume using price volatility
        price_changes = np.abs(np.diff(prices, axis=1))
        simulated_volume = np.mean(price_changes[:, -10:], axis=1)
        
        # Volume-weighted momentum
        momentum = self.calculate_momentum(prices, 10)
        volume_weighted = momentum * (simulated_volume / (np.mean(simulated_volume) + 1e-8))
        
        return volume_weighted
    
    def calculate_risk_adjusted_position(self, signal, prices, returns):
        """Calculate risk-adjusted position sizes"""
        # Base position from signal
        base_position = signal
        
        # Volatility adjustment
        vol = self.calculate_volatility(returns)
        vol_adj = 1 / (vol + 1e-8)
        vol_adj = vol_adj / (np.mean(vol_adj) + 1e-8)  # Normalize
        
        # Volatility regime adjustment
        vol_regime = self.calculate_volatility_regime(returns)
        
        # Correlation risk adjustment
        corr_risk = self.calculate_correlation_risk(returns)
        
        # Combine all adjustments
        adjusted_signal = base_position * vol_adj * vol_regime * corr_risk
        
        return adjusted_signal
    
    def optimize_positions(self, signals, prices):
        """Optimize positions considering constraints and risk"""
        # Convert signals to dollar positions
        current_prices = prices[:, -1]
        dollar_positions = signals * current_prices
        
        # Apply position limits
        max_shares = self.position_limit / (current_prices + 1e-8)
        dollar_positions = np.clip(dollar_positions, -self.position_limit, self.position_limit)
        
        # Convert back to shares
        share_positions = dollar_positions / (current_prices + 1e-8)
        
        # Round to integers
        share_positions = np.round(share_positions).astype(int)
        
        # Ensure we don't exceed position limits in shares
        share_positions = np.clip(share_positions, -max_shares, max_shares)
        
        return share_positions
    
    def generate_signals(self, prices):
        """Generate comprehensive trading signals"""
        if prices.shape[1] < 5:  # Need at least 5 days
            return np.zeros(prices.shape[0])
        
        returns = self.calculate_returns(prices)
        
        # Multiple signal components with adaptive lookbacks
        available_days = prices.shape[1]
        
        # Adjust lookbacks based on available data
        short_lookback = min(self.lookback_short, available_days - 1)
        medium_lookback = min(self.lookback_medium, available_days - 1)
        long_lookback = min(self.lookback_long, available_days - 1)
        
        momentum_short = self.calculate_momentum(prices, short_lookback)
        momentum_medium = self.calculate_momentum(prices, medium_lookback)
        momentum_long = self.calculate_momentum(prices, long_lookback)
        
        mean_rev_short = self.calculate_mean_reversion(prices, short_lookback)
        mean_rev_medium = self.calculate_mean_reversion(prices, medium_lookback)
        
        trend_strength = self.calculate_trend_strength(prices, medium_lookback)
        volume_weighted = self.calculate_volume_weighted_signal(prices)
        
        # Combine signals with weights
        combined_signal = (
            0.30 * momentum_short +
            0.25 * momentum_medium +
            0.15 * momentum_long +
            0.15 * mean_rev_short +
            0.10 * mean_rev_medium +
            0.03 * trend_strength +
            0.02 * volume_weighted
        )
        
        # Apply risk adjustments
        risk_adjusted = self.calculate_risk_adjusted_position(combined_signal, prices, returns)
        
        return risk_adjusted

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

def getMyPosition(prices):
    """
    Main function for Algothon competition
    
    Args:
        prices: numpy array of shape (nInst, nt) where nInst=50 instruments, nt=number of days
    
    Returns:
        numpy array of 50 integers representing desired positions for each instrument
    """
    # Initialize algorithm
    algo = AdvancedTradingAlgorithm()
    
    # Generate signals
    signals = algo.generate_signals(prices)
    
    # Debug prints
    print(f"Signal range: {signals.min():.6f} to {signals.max():.6f}")
    print(f"Non-zero signals: {np.count_nonzero(signals)}")
    
    # Optimize positions
    positions = algo.optimize_positions(signals, prices)
    
    # Debug prints
    print(f"Position range: {positions.min()} to {positions.max()}")
    print(f"Non-zero positions: {np.count_nonzero(positions)}")
    
    # Apply final constraints
    current_prices = prices[:, -1]
    max_shares = algo.position_limit / (current_prices + 1e-8)
    positions = np.clip(positions, -max_shares, max_shares)
    
    # Ensure integer positions
    positions = np.round(positions).astype(int)
    
    return positions

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    nInst, nt = 50, 100
    test_prices = np.random.randn(nInst, nt).cumsum(axis=1) + 100
    
    positions = getMyPosition(test_prices)
    print(f"Generated positions: {positions}")
    print(f"Position range: {positions.min()} to {positions.max()}")
    print(f"Non-zero positions: {np.count_nonzero(positions)}")
