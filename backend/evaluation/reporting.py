"""
Reporting utilities for backtest results.
"""

import pandas as pd
from typing import Dict, List, Any
from pathlib import Path


def generate_backtest_report(results: Dict[str, Any], 
                           output_file: Path = None) -> str:
    """
    Generate a comprehensive backtest report.
    
    Args:
        results: Backtest results dictionary
        output_file: Optional file path to save report
        
    Returns:
        Formatted report string
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 60)
    report_lines.append("BACKTEST REPORT")
    report_lines.append("=" * 60)
    
    # Summary statistics
    report_lines.append("\nPERFORMANCE SUMMARY:")
    report_lines.append("-" * 30)
    report_lines.append(f"Final P&L: ${results.get('final_pnl', 0):,.2f}")
    report_lines.append(f"Total Return: {results.get('total_return', 0):,.2f}")
    report_lines.append(f"Total Commission: ${results.get('total_commission', 0):,.2f}")
    report_lines.append(f"Total Trades: {results.get('total_trades', 0)}")
    
    # Risk metrics
    report_lines.append("\nRISK METRICS:")
    report_lines.append("-" * 30)
    report_lines.append(f"Mean Daily Return: {results.get('mean', 0):,.4f}")
    report_lines.append(f"Standard Deviation: {results.get('std', 0):,.4f}")
    report_lines.append(f"Risk-Adjusted Metric: {results.get('metric', 0):,.4f}")
    report_lines.append(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):,.4f}")
    report_lines.append(f"Maximum Drawdown: {results.get('max_drawdown', 0):,.2f}")
    report_lines.append(f"Volatility (Annualized): {results.get('volatility', 0):,.4f}")
    
    # Trade analysis
    if 'trade_history' in results:
        report_lines.append("\nTRADE ANALYSIS:")
        report_lines.append("-" * 30)
        trades = results['trade_history']
        if trades:
            avg_trade_value = sum(abs(t.get('trade_value', 0)) for t in trades) / len(trades)
            report_lines.append(f"Average Trade Value: ${avg_trade_value:,.2f}")
            report_lines.append(f"Average Commission per Trade: ${results.get('total_commission', 0) / len(trades):,.2f}")
    
    report = "\n".join(report_lines)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
    
    return report


def create_performance_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from backtest results.
    
    Args:
        results: Backtest results dictionary
        
    Returns:
        DataFrame with performance data
    """
    if 'position_history' not in results:
        return pd.DataFrame()
    
    # Extract daily performance data
    daily_data = []
    for snapshot in results['position_history']:
        daily_data.append({
            'day_idx': snapshot['day_idx'],
            'date': snapshot.get('date', snapshot['day_idx']),
            'total_value': snapshot['total_value'],
            'daily_pnl': snapshot['daily_pnl'],
            'cumulative_pnl': snapshot['cumulative_pnl'],
            'commission': snapshot['commission'],
            'n_trades': snapshot['n_trades']
        })
    
    return pd.DataFrame(daily_data)


def plot_performance(results: Dict[str, Any], 
                    save_path: Path = None) -> None:
    """
    Create performance plots from backtest results.
    
    Args:
        results: Backtest results dictionary
        save_path: Optional path to save plots
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Performance Analysis', fontsize=16)
        
        # Cumulative P&L
        if 'cum_pnl' in results:
            axes[0, 0].plot(results['cum_pnl'])
            axes[0, 0].set_title('Cumulative P&L')
            axes[0, 0].set_xlabel('Day')
            axes[0, 0].set_ylabel('Cumulative P&L ($)')
            axes[0, 0].grid(True)
        
        # Daily P&L
        if 'daily_pnl' in results:
            axes[0, 1].plot(results['daily_pnl'])
            axes[0, 1].set_title('Daily P&L')
            axes[0, 1].set_xlabel('Day')
            axes[0, 1].set_ylabel('Daily P&L ($)')
            axes[0, 1].grid(True)
        
        # Position value over time
        if 'position_history' in results:
            total_values = [snapshot['total_value'] for snapshot in results['position_history']]
            axes[1, 0].plot(total_values)
            axes[1, 0].set_title('Total Position Value')
            axes[1, 0].set_xlabel('Day')
            axes[1, 0].set_ylabel('Position Value ($)')
            axes[1, 0].grid(True)
        
        # Commission over time
        if 'position_history' in results:
            commissions = [snapshot['commission'] for snapshot in results['position_history']]
            axes[1, 1].plot(commissions)
            axes[1, 1].set_title('Daily Commission')
            axes[1, 1].set_xlabel('Day')
            axes[1, 1].set_ylabel('Commission ($)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("matplotlib not available for plotting")
    except Exception as e:
        print(f"Error creating plots: {e}") 