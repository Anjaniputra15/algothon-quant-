#!/usr/bin/env python3
"""
Test runner for the algothon-quant loader functions.
"""

import pytest
import sys
from pathlib import Path


def main():
    """Run the loader tests."""
    # Add the project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Run tests
    test_files = [
        "tests/test_loader.py",
        "tests/test_toy_matrix.py"
    ]
    
    print("Running loader tests...")
    print("=" * 50)
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\nRunning tests in {test_file}")
            print("-" * 30)
            result = pytest.main([test_file, "-v"])
            if result != 0:
                print(f"Tests in {test_file} failed!")
                return result
        else:
            print(f"Test file {test_file} not found!")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 