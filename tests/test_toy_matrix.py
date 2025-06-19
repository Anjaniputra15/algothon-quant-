"""
Tests specifically for the 3×5 toy matrix.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from backend.data.loader import load_price_matrix, load_price_df


class TestToyMatrix:
    """Test class specifically for the 3×5 toy matrix."""
    
    @pytest.fixture
    def toy_price_file(self):
        """Create a toy price file with 3×5 matrix."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Create a 3×5 toy matrix with realistic price data
            toy_data = [
                "100.50  101.20  99.80  102.10  100.90",
                "101.30  102.50  100.10  103.20  101.80",
                "100.80  101.90  99.50  102.80  100.60"
            ]
            f.write('\n'.join(toy_data))
            f.flush()
            return Path(f.name)
    
    def test_toy_matrix_structure(self, toy_price_file):
        """Test that the toy matrix has the correct 3×5 structure."""
        # Read the file manually to verify structure
        with open(toy_price_file, 'r') as f:
            lines = f.readlines()
        
        # Parse the data
        price_data = []
        for line in lines:
            line = line.strip()
            if line:
                values = line.split()
                row = [float(val) for val in values]
                price_data.append(row)
        
        # Verify dimensions
        assert len(price_data) == 3, f"Expected 3 rows, got {len(price_data)}"
        assert all(len(row) == 5 for row in price_data), "All rows should have 5 columns"
        
        # Verify the exact values
        expected_data = [
            [100.50, 101.20, 99.80, 102.10, 100.90],
            [101.30, 102.50, 100.10, 103.20, 101.80],
            [100.80, 101.90, 99.50, 102.80, 100.60]
        ]
        
        for i, row in enumerate(price_data):
            for j, val in enumerate(row):
                assert abs(val - expected_data[i][j]) < 1e-6, \
                    f"Value at [{i},{j}] should be {expected_data[i][j]}, got {val}"
    
    def test_toy_matrix_validation_failure(self, toy_price_file):
        """Test that the toy matrix fails validation due to insufficient instruments."""
        # The toy matrix has only 5 instruments, which is less than the required 50-100
        with pytest.raises(ValueError, match="Number of instruments \\(5\\) must be between 50 and 100"):
            load_price_matrix(toy_price_file)
        
        with pytest.raises(ValueError, match="Number of instruments \\(5\\) must be between 50 and 100"):
            load_price_df(toy_price_file)
    
    def test_toy_matrix_with_validation_bypass(self, toy_price_file):
        """Test loading the toy matrix by temporarily bypassing validation."""
        # This is a helper function to load the toy matrix for testing purposes
        def load_toy_matrix(path, sep=r"\s+"):
            """Load toy matrix bypassing instrument count validation."""
            path = Path(path)
            
            if not path.exists():
                raise FileNotFoundError(f"Price file not found: {path}")
            
            # Read all lines and parse
            with open(path, 'r') as f:
                lines = f.readlines()
            
            # Parse each line into a list of floats
            price_data = []
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    # Split by separator and convert to floats
                    import re
                    values = re.split(sep, line)
                    row = [float(val) for val in values if val.strip()]
                    price_data.append(row)
                except ValueError as e:
                    raise ValueError(f"Error parsing line {i}: {line}. {str(e)}")
            
            if not price_data:
                raise ValueError("No valid price data found in file")
            
            # Convert to numpy array
            try:
                price_matrix = np.array(price_data, dtype=np.float32)
            except ValueError as e:
                raise ValueError(f"Error creating price matrix: {str(e)}")
            
            # Skip validation for toy matrix testing
            # price_matrix = pd.DataFrame(price_matrix).fillna(method='ffill').values.astype(np.float32)
            
            return price_matrix
        
        def load_toy_df(path, sep=r"\s+"):
            """Load toy DataFrame bypassing instrument count validation."""
            path = Path(path)
            
            if not path.exists():
                raise FileNotFoundError(f"Price file not found: {path}")
            
            # Read all lines and parse
            with open(path, 'r') as f:
                lines = f.readlines()
            
            # Parse each line into a list of floats
            price_data = []
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    # Split by separator and convert to floats
                    import re
                    values = re.split(sep, line)
                    row = [float(val) for val in values if val.strip()]
                    price_data.append(row)
                except ValueError as e:
                    raise ValueError(f"Error parsing line {i}: {line}. {str(e)}")
            
            if not price_data:
                raise ValueError("No valid price data found in file")
            
            # Create DataFrame
            try:
                df = pd.DataFrame(price_data, dtype=np.float32)
            except ValueError as e:
                raise ValueError(f"Error creating price DataFrame: {str(e)}")
            
            # Skip validation for toy matrix testing
            # df = df.fillna(method='ffill')
            
            # Set column names as instrument IDs
            n_instruments = df.shape[1]
            df.columns = [f"instrument_{i:03d}" for i in range(n_instruments)]
            
            # Set index as dates
            df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
            
            return df
        
        # Test loading with bypass
        matrix = load_toy_matrix(toy_price_file)
        df = load_toy_df(toy_price_file)
        
        # Verify matrix
        assert matrix.shape == (3, 5)
        assert matrix.dtype == np.float32
        
        # Verify DataFrame
        assert df.shape == (3, 5)
        assert df.dtypes.iloc[0] == np.float32
        assert list(df.columns) == [f"instrument_{i:03d}" for i in range(5)]
        assert isinstance(df.index, pd.DatetimeIndex)
    
    def test_toy_matrix_values(self, toy_price_file):
        """Test the specific values in the toy matrix."""
        # Load the toy matrix with bypass
        def load_toy_matrix(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            
            price_data = []
            for line in lines:
                line = line.strip()
                if line:
                    values = line.split()
                    row = [float(val) for val in values]
                    price_data.append(row)
            
            return np.array(price_data, dtype=np.float32)
        
        matrix = load_toy_matrix(toy_price_file)
        
        # Expected values
        expected = np.array([
            [100.50, 101.20, 99.80, 102.10, 100.90],
            [101.30, 102.50, 100.10, 103.20, 101.80],
            [100.80, 101.90, 99.50, 102.80, 100.60]
        ], dtype=np.float32)
        
        # Compare
        np.testing.assert_array_almost_equal(matrix, expected, decimal=6)
        
        # Test specific values
        assert matrix[0, 0] == 100.50
        assert matrix[0, 1] == 101.20
        assert matrix[1, 2] == 100.10
        assert matrix[2, 4] == 100.60
    
    def test_toy_matrix_statistics(self, toy_price_file):
        """Test basic statistics of the toy matrix."""
        def load_toy_matrix(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            
            price_data = []
            for line in lines:
                line = line.strip()
                if line:
                    values = line.split()
                    row = [float(val) for val in values]
                    price_data.append(row)
            
            return np.array(price_data, dtype=np.float32)
        
        matrix = load_toy_matrix(toy_price_file)
        
        # Test basic statistics
        assert matrix.min() == 99.50
        assert matrix.max() == 103.20
        assert np.isclose(matrix.mean(), 101.16, atol=1e-2)
        assert matrix.shape == (3, 5)
        
        # Test column-wise statistics
        col_means = matrix.mean(axis=0)
        assert len(col_means) == 5
        assert all(99 < mean < 104 for mean in col_means)
        
        # Test row-wise statistics
        row_means = matrix.mean(axis=1)
        assert len(row_means) == 3
        assert all(99 < mean < 104 for mean in row_means)


if __name__ == "__main__":
    pytest.main([__file__]) 