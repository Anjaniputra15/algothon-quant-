"""
Command-line interface for algothon-quant.
"""

import click
import sys
from pathlib import Path
from loguru import logger
from . import __version__, config
from .utils.logging import setup_logging


@click.group()
@click.version_option(version=__version__, prog_name="algothon-quant")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", type=click.Path(), help="Log file path")
def main(log_level: str, log_file: str):
    """
    Algothon-Quant: Polyglot monorepo for quantitative finance algorithms.
    
    This CLI provides access to various quantitative finance tools and algorithms
    written in Python, Rust (via PyO3), and Julia (via PyJulia).
    """
    # Setup logging
    log_file_path = Path(log_file) if log_file else None
    setup_logging(log_level=log_level, log_file=log_file_path)
    
    logger.info(f"Algothon-Quant v{__version__} started")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Rust enabled: {config.get('polyglot.rust_enabled', True)}")
    logger.info(f"Julia enabled: {config.get('polyglot.julia_enabled', True)}")


@main.command()
@click.argument("symbol", type=str)
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def download_data(symbol: str, start_date: str, end_date: str, output: str):
    """Download financial data for a given symbol."""
    try:
        from .data.loaders import load_yahoo_finance_data
        
        logger.info(f"Downloading data for {symbol}")
        data = load_yahoo_finance_data(symbol, start_date, end_date)
        
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(output_path)
            logger.info(f"Data saved to {output_path}")
        else:
            click.echo(data.head())
            
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        sys.exit(1)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--method", default="log", type=click.Choice(["log", "simple"]), 
              help="Return calculation method")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def calculate_returns(input_file: str, method: str, output: str):
    """Calculate returns from price data."""
    try:
        from .data.loaders import load_price_data_from_file
        from .core.algorithms import calculate_returns
        
        logger.info(f"Loading price data from {input_file}")
        prices = load_price_data_from_file(input_file)
        
        logger.info(f"Calculating {method} returns")
        returns = calculate_returns(prices, method=method)
        
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            returns.to_csv(output_path)
            logger.info(f"Returns saved to {output_path}")
        else:
            click.echo(returns.head())
            
    except Exception as e:
        logger.error(f"Failed to calculate returns: {e}")
        sys.exit(1)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--window", default=252, type=int, help="Volatility window")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def calculate_volatility(input_file: str, window: int, output: str):
    """Calculate volatility from returns data."""
    try:
        import pandas as pd
        from .core.algorithms import calculate_volatility
        
        logger.info(f"Loading returns data from {input_file}")
        returns = pd.read_csv(input_file, index_col=0, squeeze=True)
        
        logger.info(f"Calculating volatility with {window}-day window")
        volatility = calculate_volatility(returns, window=window)
        
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            volatility.to_csv(output_path)
            logger.info(f"Volatility saved to {output_path}")
        else:
            click.echo(volatility.tail())
            
    except Exception as e:
        logger.error(f"Failed to calculate volatility: {e}")
        sys.exit(1)


@main.command()
def info():
    """Display package information and configuration."""
    click.echo(f"Algothon-Quant v{__version__}")
    click.echo(f"Python version: {sys.version}")
    click.echo(f"Project root: {config.PROJECT_ROOT}")
    click.echo(f"Backend directory: {config.BACKEND_DIR}")
    click.echo(f"Rust enabled: {config.rust_enabled}")
    click.echo(f"Julia enabled: {config.julia_enabled}")


if __name__ == "__main__":
    main() 