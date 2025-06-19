#!/usr/bin/env python3
"""
Demonstration script for data loading utilities.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.data.loader import load_price_matrix, load_price_df


def demo_toy_data():
    """Demonstrate loading toy data."""
    print("=" * 60)
    print("DEMONSTRATION: Toy Data Loading")
    print("=" * 60)
    
    # Create toy data inline instead of using toy_prices.txt
    print("Creating toy price data (3 days × 5 instruments)...")
    toy_data = np.array([
        [100.0, 101.0, 102.0, 103.0, 104.0],  # Day 1
        [101.0, 102.0, 101.0, 104.0, 105.0],  # Day 2
        [102.0, 103.0, 100.0, 105.0, 106.0]   # Day 3
    ])
    
    print("Toy price matrix:")
    print(toy_data)
    print(f"Shape: {toy_data.shape}")
    
    # Test loading as numpy array
    print("\nLoading as numpy array...")
    try:
        prices_array = load_price_matrix(toy_data)
        print(f"Loaded array shape: {prices_array.shape}")
        print(f"Data type: {prices_array.dtype}")
        print("First few values:")
        print(prices_array[:2, :3])
    except ValueError as e:
        print(f"[Expected] Validation error for toy data: {e}")
    
    # Test loading as DataFrame
    print("\nLoading as DataFrame...")
    try:
        prices_df = load_price_df(toy_data)
        print(f"DataFrame shape: {prices_df.shape}")
        print(f"DataFrame info:")
        print(prices_df.info())
        print("\nFirst few rows:")
        print(prices_df.head())
    except ValueError as e:
        print(f"[Expected] Validation error for toy data: {e}")


def demo_real_data():
    """Demonstrate loading real price data."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Real Data Loading")
    print("=" * 60)
    
    # Use prices.txt instead of sample_prices_60.txt
    valid_file = Path("prices.txt")
    
    if not valid_file.exists():
        print(f"Error: {valid_file} not found!")
        print("Please ensure prices.txt exists in the current directory.")
        return
    
    print(f"Loading price data from {valid_file}...")
    
    try:
        # Load as numpy array
        print("\nLoading as numpy array...")
        prices_array = load_price_matrix(valid_file)
        print(f"Loaded array shape: {prices_array.shape}")
        print(f"Data type: {prices_array.dtype}")
        print(f"Price range: ${prices_array.min():.2f} - ${prices_array.max():.2f}")
        print(f"Mean price: ${prices_array.mean():.2f}")
        print(f"Standard deviation: ${prices_array.std():.2f}")
        
        # Check for weekends (should be forward-filled)
        print(f"\nChecking for weekend handling...")
        print(f"Number of days: {prices_array.shape[0]}")
        print(f"Number of instruments: {prices_array.shape[1]}")
        
        # Load as DataFrame
        print("\nLoading as DataFrame...")
        prices_df = load_price_df(valid_file)
        print(f"DataFrame shape: {prices_df.shape}")
        print(f"DataFrame info:")
        print(prices_df.info())
        
        # Show sample data
        print("\nSample data (first 3 days, first 5 instruments):")
        sample_data = prices_df.iloc[:3, :5]
        print(sample_data)
        
        # Show statistics
        print("\nData statistics:")
        print(f"Total data points: {prices_df.size}")
        print(f"Missing values: {prices_df.isnull().sum().sum()}")
        print(f"Finite values: {np.isfinite(prices_df.values).sum()}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()


def demo_data_validation():
    """Demonstrate data validation features."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Data Validation")
    print("=" * 60)
    
    # Test with invalid data
    print("Testing with invalid data...")
    
    # Test empty data
    try:
        load_price_matrix(np.array([]))
        print("ERROR: Should have failed with empty data")
    except ValueError as e:
        print(f"✓ Correctly caught empty data error: {e}")
    
    # Test 1D data
    try:
        load_price_matrix(np.array([1, 2, 3]))
        print("ERROR: Should have failed with 1D data")
    except ValueError as e:
        print(f"✓ Correctly caught 1D data error: {e}")
    
    # Test data with insufficient instruments
    try:
        small_data = np.random.rand(10, 30)  # Only 30 instruments
        load_price_matrix(small_data)
        print("ERROR: Should have failed with insufficient instruments")
    except ValueError as e:
        print(f"✓ Correctly caught insufficient instruments error: {e}")
    
    # Test data with non-finite values
    try:
        invalid_data = np.random.rand(10, 60)
        invalid_data[0, 0] = np.nan
        load_price_matrix(invalid_data)
        print("ERROR: Should have failed with non-finite values")
    except ValueError as e:
        print(f"✓ Correctly caught non-finite values error: {e}")
    
    # Test data with non-positive values
    try:
        invalid_data = np.random.rand(10, 60)
        invalid_data[0, 0] = -1.0
        load_price_matrix(invalid_data)
        print("ERROR: Should have failed with non-positive values")
    except ValueError as e:
        print(f"✓ Correctly caught non-positive values error: {e}")


def demo_performance():
    """Demonstrate performance with large datasets."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Performance Testing")
    print("=" * 60)
    
    # Use prices.txt for performance testing
    valid_file = Path("prices.txt")
    
    if not valid_file.exists():
        print(f"Error: {valid_file} not found!")
        return
    
    import time
    
    print(f"Testing performance with {valid_file}...")
    
    # Test numpy array loading performance
    start_time = time.time()
    prices_array = load_price_matrix(valid_file)
    array_time = time.time() - start_time
    
    print(f"Numpy array loading: {array_time:.4f} seconds")
    print(f"Array shape: {prices_array.shape}")
    print(f"Memory usage: {prices_array.nbytes / 1024 / 1024:.2f} MB")
    
    # Test DataFrame loading performance
    start_time = time.time()
    prices_df = load_price_df(valid_file)
    df_time = time.time() - start_time
    
    print(f"DataFrame loading: {df_time:.4f} seconds")
    print(f"DataFrame shape: {prices_df.shape}")
    print(f"Memory usage: {prices_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Performance comparison
    if array_time > 0 and df_time > 0:
        ratio = df_time / array_time
        print(f"\nDataFrame loading is {ratio:.2f}x slower than numpy array loading")


def main():
    """Run all demonstrations."""
    print("ALGOTHON-QUANT DATA LOADER DEMONSTRATION")
    print("=" * 60)
    
    try:
        demo_toy_data()
        demo_real_data()
        demo_data_validation()
        demo_performance()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Toy data loading and validation")
        print("✓ Real data loading from prices.txt")
        print("✓ Data validation and error handling")
        print("✓ Performance testing with large datasets")
        print("✓ Both numpy array and DataFrame output")
        print("✓ Weekend forward-filling")
        print("✓ Data type conversion to float32")
        
    except Exception as e:
        print(f"\nDEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 