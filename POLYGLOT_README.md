# Algothon-Quant: Polyglot Monorepo

A polyglot monorepo for quantitative finance algorithms, combining the power of Python, Rust, and Julia.

## Overview

This repository provides a unified interface for quantitative finance algorithms written in multiple languages:

- **Python (3.12)**: Core algorithms, data processing, and ML models
- **Rust (via PyO3)**: High-performance numerical computations
- **Julia (via PyJulia)**: Advanced mathematical modeling

## Project Structure

```
algothon-quant/
├── backend/                 # Python backend package
│   ├── __init__.py         # Main package initialization
│   ├── core/               # Core algorithms
│   ├── data/               # Data processing
│   ├── models/             # ML models
│   ├── utils/              # Utilities
│   └── cli.py              # Command-line interface
├── pyproject.toml          # Python project configuration
├── Cargo.toml              # Rust project configuration
├── JuliaProject.toml       # Julia project configuration
└── README.md               # This file
```

## Installation

### Prerequisites

1. **Python 3.12+**
2. **Rust** (for Rust components)
3. **Julia 1.9+** (for Julia components)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/algothon/algothon-quant.git
   cd algothon-quant
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -e .
   # Or install with specific components:
   pip install -e ".[dev,rust,julia]"
   ```

3. **Setup Rust components:**
   ```bash
   # Build Rust library
   maturin develop
   ```

4. **Setup Julia components:**
   ```bash
   # Install Julia dependencies
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

## Usage

### Python API

```python
from backend import core, data, models

# Load data
prices = data.loaders.load_price_data_from_file("prices.txt")

# Calculate returns
returns = core.algorithms.calculate_returns(prices)

# Calculate volatility
volatility = core.algorithms.calculate_volatility(returns)

# Train a model
from backend.models.regression import XGBoostRegressionModel
model = XGBoostRegressionModel()
model.fit(X, y)
predictions = model.predict(X_test)
```

### Command Line Interface

```bash
# Download financial data
algothon-quant download-data AAPL --start-date 2023-01-01 --end-date 2023-12-31

# Calculate returns
algothon-quant calculate-returns prices.txt --method log

# Calculate volatility
algothon-quant calculate-volatility returns.csv --window 252

# Show package info
algothon-quant info
```

### Rust Integration

The Rust components provide high-performance numerical computations:

```python
# Rust components are automatically imported if available
from backend import config

if config.rust_enabled:
    # Use Rust-accelerated functions
    pass
```

### Julia Integration

The Julia components provide advanced mathematical modeling:

```python
# Julia components are automatically imported if available
from backend import config

if config.julia_enabled:
    # Use Julia mathematical functions
    pass
```

## Development

### Python Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black backend/
isort backend/

# Type checking
mypy backend/
```

### Rust Development

```bash
# Build in development mode
maturin develop

# Run tests
cargo test

# Build for release
maturin build --release
```

### Julia Development

```bash
# Enter Julia REPL with project
julia --project=.

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Configuration

The package uses a configuration system that can be customized:

```python
from backend.utils.config import config

# Get configuration values
cache_dir = config.get("data.cache_dir")
log_level = config.get("logging.level", "INFO")

# Set configuration values
config.set("models.default_random_state", 42)
config.save()
```

## Polyglot Features

### Rust (PyO3)
- High-performance numerical computations
- Memory-efficient array operations
- Parallel processing capabilities
- Zero-copy data sharing with Python

### Julia (PyJulia)
- Advanced mathematical modeling
- Statistical computing
- Optimization algorithms
- Time series analysis

### Python
- Core algorithms and data processing
- Machine learning models
- Data visualization
- API and CLI interfaces

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub. 