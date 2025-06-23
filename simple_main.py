#!/usr/bin/env python3
"""
Simplified Quantitative Trading Algorithm
This is a standalone version that works without complex dependencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

class SimpleTradingStrategy:
    """A simplified momentum-based trading strategy"""
    
    def __init__(self, n_instruments: int = 50, position_limit: int = 10000):
        self.n_instruments = n_instruments
        self.position_limit = position_limit
        self.current_positions = np.zeros(n_instruments)
        self.commission_rate = 0.0005  # 5 basis points
        
    def get_position(self, price_history: np.ndarray) -> np.ndarray:
        """
        Calculate trading positions based on price momentum
        
        Args:
            price_history: Array of shape (n_instruments, n_days)
            
        Returns:
            Array of positions for each instrument
        """
        if price_history.shape[1] < 2:
            return np.zeros(self.n_instruments)
        
        # Calculate returns
        returns = np.log(price_history[:, -1] / price_history[:, -2])
        
        # Normalize returns
        norm = np.sqrt(returns.dot(returns))
        if norm > 0:
            returns = returns / norm
        
        # Calculate target positions (simplified momentum strategy)
        target_positions = 5000 * returns / price_history[:, -1]
        
        # Apply position limits
        position_limits = self.position_limit / price_history[:, -1]
        target_positions = np.clip(target_positions, -position_limits, position_limits)
        
        # Update current positions
        self.current_positions = target_positions.astype(int)
        
        return self.current_positions
    
    def backtest(self, price_data: np.ndarray, test_days: int = 200) -> Tuple[float, float, float, float, float]:
        """
        Run backtest on historical price data
        
        Args:
            price_data: Price data of shape (n_instruments, n_days)
            test_days: Number of days to test
            
        Returns:
            Tuple of (mean_pl, return_rate, pl_std, sharpe_ratio, total_volume)
        """
        cash = 0
        current_positions = np.zeros(self.n_instruments)
        total_volume = 0
        daily_pl = []
        
        n_days = price_data.shape[1]
        start_day = max(0, n_days - test_days)
        
        print(f"Running backtest for {test_days} days...")
        
        for day in range(start_day, n_days):
            # Get price history up to current day
            price_history = price_data[:, :day+1]
            current_prices = price_history[:, -1]
            
            if day < n_days - 1:  # Don't trade on last day
                # Get new positions
                new_positions = self.get_position(price_history)
                
                # Calculate position changes
                position_changes = new_positions - current_positions
                
                # Calculate trading volume and costs
                volumes = current_prices * np.abs(position_changes)
                daily_volume = np.sum(volumes)
                total_volume += daily_volume
                
                # Calculate costs
                commission = daily_volume * self.commission_rate
                trade_cost = current_prices.dot(position_changes) + commission
                cash -= trade_cost
                
                current_positions = new_positions.copy()
            
            # Calculate portfolio value
            position_value = current_positions.dot(current_prices)
            total_value = cash + position_value
            
            # Calculate daily P&L
            if day > start_day:
                daily_pl.append(total_value - prev_value)
            
            prev_value = total_value
            
            # Print progress every 50 days
            if (day - start_day) % 50 == 0:
                print(f"Day {day}: Value = ${total_value:.2f}, Volume = ${daily_volume:.0f}")
        
        # Calculate statistics
        daily_pl = np.array(daily_pl)
        mean_pl = np.mean(daily_pl)
        pl_std = np.std(daily_pl)
        return_rate = mean_pl / total_volume if total_volume > 0 else 0
        sharpe_ratio = np.sqrt(249) * mean_pl / pl_std if pl_std > 0 else 0
        
        return mean_pl, return_rate, pl_std, sharpe_ratio, total_volume
    
    def plot_results(self, price_data: np.ndarray, test_days: int = 200):
        """Plot backtest results"""
        try:
            # Run backtest
            mean_pl, return_rate, pl_std, sharpe, total_volume = self.backtest(price_data, test_days)
            
            # Print results
            print("\n" + "="*50)
            print("BACKTEST RESULTS")
            print("="*50)
            print(f"Mean P&L: ${mean_pl:.2f}")
            print(f"Return Rate: {return_rate:.5f}")
            print(f"P&L Std Dev: ${pl_std:.2f}")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Total Volume: ${total_volume:.0f}")
            print(f"Score: {mean_pl - 0.1*pl_std:.2f}")
            
            # Create simple plot
            plt.figure(figsize=(12, 8))
            
            # Plot price evolution for first few instruments
            plt.subplot(2, 2, 1)
            for i in range(min(5, self.n_instruments)):
                plt.plot(price_data[i, -test_days:], label=f'Instrument {i+1}')
            plt.title('Price Evolution (First 5 Instruments)')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            
            # Plot position distribution
            plt.subplot(2, 2, 2)
            final_positions = self.current_positions
            plt.bar(range(len(final_positions)), final_positions)
            plt.title('Final Positions')
            plt.xlabel('Instrument')
            plt.ylabel('Position')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting results: {e}")
            print("Continuing without plots...")

def load_price_data(filename: str = "prices.txt") -> np.ndarray:
    """Load price data from file"""
    try:
        if os.path.exists(filename):
            # Try to load the existing prices.txt file
            df = pd.read_csv(filename, sep='\s+', header=None, index_col=None)
            return df.values.T
        else:
            # Generate synthetic data for testing
            print(f"File {filename} not found. Generating synthetic data...")
            n_instruments = 50
            n_days = 500
            np.random.seed(42)
            
            # Generate random walk prices
            returns = np.random.normal(0, 0.02, (n_instruments, n_days))
            prices = 100 * np.exp(np.cumsum(returns, axis=1))
            
            return prices
            
    except Exception as e:
        print(f"Error loading price data: {e}")
        print("Generating synthetic data...")
        
        # Fallback to synthetic data
        n_instruments = 50
        n_days = 500
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, (n_instruments, n_days))
        prices = 100 * np.exp(np.cumsum(returns, axis=1))
        return prices

def main():
    """Main function to run the simplified trading strategy"""
    print("🚀 Starting Simplified Quantitative Trading Algorithm")
    print("="*60)
    
    # Load or generate price data
    price_data = load_price_data()
    n_instruments, n_days = price_data.shape
    print(f"Loaded {n_instruments} instruments for {n_days} days")
    
    # Create and run strategy
    strategy = SimpleTradingStrategy(n_instruments=n_instruments)
    
    # Run backtest
    mean_pl, return_rate, pl_std, sharpe, total_volume = strategy.backtest(price_data, test_days=200)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Mean P&L: ${mean_pl:.2f}")
    print(f"Return Rate: {return_rate:.5f}")
    print(f"P&L Std Dev: ${pl_std:.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Total Volume: ${total_volume:.0f}")
    print(f"Score: {mean_pl - 0.1*pl_std:.2f}")
    
    # Try to plot results
    try:
        strategy.plot_results(price_data, test_days=200)
    except:
        print("Plotting not available (matplotlib not installed)")

if __name__ == "__main__":
    main() 