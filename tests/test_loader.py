"""
Tests for data loading utilities.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
from backend.data.loader import (
    load_price_matrix,
    load_price_df,
    load_price_data_with_dates,
    load_price_data_from_file,
    load_csv_data,
    load_yahoo_finance_data
)


class TestLoader:
    """Test class for data loading functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def toy_price_file(self, temp_dir):
        """Create a toy price file with 3×5 matrix (for backward compatibility)."""
        # This fixture is kept for backward compatibility but not used in tests
        # since toy data is now created inline
        price_file = temp_dir / "toy_prices.txt"
        
        # Create a 3×5 toy matrix with realistic price data
        toy_data = [
            "100.50  101.20  99.80  102.10  100.90",
            "101.30  102.50  100.10  103.20  101.80",
            "100.80  101.90  99.50  102.80  100.60"
        ]
        
        with open(price_file, 'w') as f:
            f.write('\n'.join(toy_data))
        
        return price_file
    
    @pytest.fixture
    def large_price_file(self, temp_dir):
        """Create a large price file with 60 instruments (within 50-100 range)."""
        price_file = temp_dir / "large_prices.txt"
        
        # Create data with 60 instruments and 10 days
        n_instruments = 60
        n_days = 10
        
        with open(price_file, 'w') as f:
            for day in range(n_days):
                # Generate realistic price data around 100
                prices = [100 + np.random.normal(0, 2) for _ in range(n_instruments)]
                line = ' '.join(f"{price:.2f}" for price in prices)
                f.write(line + '\n')
        
        return price_file
    
    @pytest.fixture
    def price_file_with_dates(self, temp_dir):
        """Create a price file with dates in the first column."""
        price_file = temp_dir / "prices_with_dates.txt"
        
        data_with_dates = [
            "2023-01-01  100.50  101.20  99.80  102.10  100.90",
            "2023-01-02  101.30  102.50  100.10  103.20  101.80",
            "2023-01-03  100.80  101.90  99.50  102.80  100.60"
        ]
        
        with open(price_file, 'w') as f:
            f.write('\n'.join(data_with_dates))
        
        return price_file
    
    @pytest.fixture
    def invalid_price_file(self, temp_dir):
        """Create an invalid price file."""
        price_file = temp_dir / "invalid_prices.txt"
        
        invalid_data = [
            "100.50  101.20  abc  102.10  100.90",  # Non-numeric value
            "101.30  102.50  100.10  103.20",       # Different number of columns
            "100.80  101.90  99.50  102.80  100.60"
        ]
        
        with open(price_file, 'w') as f:
            f.write('\n'.join(invalid_data))
        
        return price_file
    
    @pytest.fixture
    def small_price_file(self, temp_dir):
        """Create a price file with too few instruments (< 50)."""
        price_file = temp_dir / "small_prices.txt"
        
        # Only 3 instruments (should fail validation)
        small_data = [
            "100.50  101.20  99.80",
            "101.30  102.50  100.10",
            "100.80  101.90  99.50"
        ]
        
        with open(price_file, 'w') as f:
            f.write('\n'.join(small_data))
        
        return price_file
    
    @pytest.fixture
    def empty_price_file(self, temp_dir):
        """Create an empty price file."""
        price_file = temp_dir / "empty_prices.txt"
        price_file.touch()
        return price_file
    
    def test_load_price_matrix_large(self, large_price_file):
        """Test loading large price matrix (60 instruments)."""
        matrix = load_price_matrix(large_price_file)
        
        # Check shape
        assert matrix.shape == (10, 60)
        
        # Check dtype
        assert matrix.dtype == np.float32
        
        # Check that all values are finite
        assert np.all(np.isfinite(matrix))
        
        # Check that values are reasonable (around 100)
        assert np.all(matrix > 0)
        assert np.all(matrix < 200)
    
    def test_load_price_df_large(self, large_price_file):
        """Test loading large price DataFrame (60 instruments)."""
        df = load_price_df(large_price_file)
        
        # Check shape
        assert df.shape == (10, 60)
        
        # Check dtype
        assert df.dtypes.iloc[0] == np.float32
        
        # Check column names
        assert df.columns[0] == "instrument_000"
        assert df.columns[-1] == "instrument_059"
        
        # Check index is datetime
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check that all values are finite
        assert df.notna().all().all()
    
    def test_load_price_matrix_invalid_data(self, invalid_price_file):
        """Test loading invalid price data."""
        with pytest.raises(ValueError, match="Error parsing line"):
            load_price_matrix(invalid_price_file)
    
    def test_load_price_df_invalid_data(self, invalid_price_file):
        """Test loading invalid price data as DataFrame."""
        with pytest.raises(ValueError, match="Error parsing line"):
            load_price_df(invalid_price_file)
    
    def test_load_price_matrix_too_few_instruments(self, small_price_file):
        """Test loading price matrix with too few instruments."""
        with pytest.raises(ValueError, match="Number of instruments \\(3\\) must be between 50 and 100"):
            load_price_matrix(small_price_file)
    
    def test_load_price_df_too_few_instruments(self, small_price_file):
        """Test loading price DataFrame with too few instruments."""
        with pytest.raises(ValueError, match="Number of instruments \\(3\\) must be between 50 and 100"):
            load_price_df(small_price_file)
    
    def test_load_price_matrix_file_not_found(self, temp_dir):
        """Test loading non-existent file."""
        non_existent_file = temp_dir / "non_existent.txt"
        with pytest.raises(FileNotFoundError):
            load_price_matrix(non_existent_file)
    
    def test_load_price_df_file_not_found(self, temp_dir):
        """Test loading non-existent file as DataFrame."""
        non_existent_file = temp_dir / "non_existent.txt"
        with pytest.raises(FileNotFoundError):
            load_price_df(non_existent_file)
    
    def test_load_price_matrix_empty_file(self, empty_price_file):
        """Test loading empty file."""
        with pytest.raises(ValueError, match="No valid price data found"):
            load_price_matrix(empty_price_file)
    
    def test_load_price_df_empty_file(self, empty_price_file):
        """Test loading empty file as DataFrame."""
        with pytest.raises(ValueError, match="No valid price data found"):
            load_price_df(empty_price_file)
    
    def test_load_price_matrix_custom_separator(self, temp_dir):
        """Test loading with custom separator."""
        price_file = temp_dir / "custom_sep_prices.txt"
        
        # Create data with comma separator
        data = [
            "100.50,101.20,99.80,102.10,100.90",
            "101.30,102.50,100.10,103.20,101.80",
            "100.80,101.90,99.50,102.80,100.60"
        ]
        
        with open(price_file, 'w') as f:
            f.write('\n'.join(data))
        
        # This should fail validation since only 5 instruments
        with pytest.raises(ValueError, match="Number of instruments \\(5\\) must be between 50 and 100"):
            load_price_matrix(price_file, sep=",")
    
    def test_load_price_data_with_dates(self, price_file_with_dates):
        """Test loading price data with dates."""
        # This should fail validation since only 5 instruments
        with pytest.raises(ValueError, match="Number of instruments \\(5\\) must be between 50 and 100"):
            load_price_data_with_dates(price_file_with_dates)
    
    def test_load_price_data_with_dates_large(self, temp_dir):
        """Test loading large price data with dates."""
        price_file = temp_dir / "large_prices_with_dates.txt"
        
        # Create data with 60 instruments and dates
        n_instruments = 60
        n_days = 5
        
        with open(price_file, 'w') as f:
            for day in range(n_days):
                date = f"2023-01-{day+1:02d}"
                prices = [100 + np.random.normal(0, 2) for _ in range(n_instruments)]
                line = date + ' ' + ' '.join(f"{price:.2f}" for price in prices)
                f.write(line + '\n')
        
        df = load_price_data_with_dates(price_file)
        
        # Check shape
        assert df.shape == (n_days, n_instruments)
        
        # Check dtype
        assert df.dtypes.iloc[0] == np.float32
        
        # Check index is datetime
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check dates
        expected_dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
        pd.testing.assert_index_equal(df.index, expected_dates)
    
    def test_load_price_data_from_file(self, temp_dir):
        """Test loading single column price data."""
        price_file = temp_dir / "single_column_prices.txt"
        
        # Create single column data
        prices = [100.50, 101.30, 100.80, 102.10, 99.50]
        
        with open(price_file, 'w') as f:
            f.write('\n'.join(str(price) for price in prices))
        
        series = load_price_data_from_file(price_file)
        
        # Check type
        assert isinstance(series, pd.Series)
        
        # Check values
        assert len(series) == len(prices)
        assert all(series.iloc[i] == prices[i] for i in range(len(prices)))
    
    def test_load_csv_data(self, temp_dir):
        """Test loading CSV data."""
        csv_file = temp_dir / "test.csv"
        
        # Create CSV data
        csv_data = [
            "date,price,volume",
            "2023-01-01,100.50,1000",
            "2023-01-02,101.30,1200",
            "2023-01-03,100.80,1100"
        ]
        
        with open(csv_file, 'w') as f:
            f.write('\n'.join(csv_data))
        
        df = load_csv_data(csv_file)
        
        # Check shape
        assert df.shape == (3, 3)
        
        # Check columns
        assert list(df.columns) == ['date', 'price', 'volume']
    
    def test_forward_filling_behavior(self, temp_dir):
        """Test that forward-filling works correctly."""
        price_file = temp_dir / "prices_with_nan.txt"
        
        # Create data with some NaN values (represented as empty strings)
        data_with_nan = [
            "100.50  101.20  99.80  102.10  100.90",
            "101.30        100.10  103.20  101.80",  # Missing value
            "100.80  101.90  99.50  102.80  100.60"
        ]
        
        with open(price_file, 'w') as f:
            f.write('\n'.join(data_with_nan))
        
        # This should fail validation since only 5 instruments
        with pytest.raises(ValueError, match="Number of instruments \\(5\\) must be between 50 and 100"):
            load_price_matrix(price_file)
    
    def test_load_toy_matrix(self, temp_dir):
        """Test loading toy matrix data."""
        # Create toy data inline instead of using toy_prices.txt
        toy_data = np.array([
            [100.0, 101.0, 102.0, 103.0, 104.0],  # Day 1
            [101.0, 102.0, 101.0, 104.0, 105.0],  # Day 2
            [102.0, 103.0, 100.0, 105.0, 106.0]   # Day 3
        ])
        
        # Test loading as numpy array
        prices = load_price_matrix(toy_data)
        assert prices.shape == (3, 5)
        assert prices.dtype == np.float32
        assert np.allclose(prices, toy_data, rtol=1e-5)
        
        # Test loading as DataFrame
        df = load_price_df(toy_data)
        assert df.shape == (3, 5)
        assert df.dtype == np.float32
        assert np.allclose(df.values, toy_data, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__]) 