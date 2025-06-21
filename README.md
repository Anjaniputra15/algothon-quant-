# Algothon-Quant: Polyglot Quantitative Trading Backend

A polyglot monorepo for quantitative finance algorithms, combining the power of Python, Rust, and Julia.

---

## 🚀 Project Overview

This repository provides a unified, high-performance backend for developing, testing, and evaluating trading strategies. It is designed for both competition and open-source use, supporting:

- **Python 3.12+**: Core algorithms, data processing, and ML models
- **Rust (via PyO3)**: High-performance numerical computations
- **Julia (via PyJulia)**: Advanced mathematical modeling

---

## 🗂️ Directory Structure

```
algothon-quant/
├── backend/                 # Python backend package
│   ├── __init__.py         # Main package initialization
│   ├── core/               # Core algorithms
│   ├── data/               # Data processing
│   ├── models/             # ML models
│   ├── utils/              # Utilities
│   ├── strategies/         # Trading strategies
│   ├── evaluation/         # Backtesting and metrics
│   └── cli.py              # Command-line interface
├── pyproject.toml          # Python project configuration
├── Cargo.toml              # Rust project configuration
├── JuliaProject.toml       # Julia project configuration
├── prices.txt              # Example price data
├── demo_loader.py          # Data loader demo
├── demo_strategies.py      # Strategies demo
├── demo_backtester.py      # Backtester demo
└── README.md               # This file
```

---

## 🏁 Competition Objective

Develop a trading strategy algorithm to perform optimally on provided price data, subject to realistic trading constraints:
- **$10,000 per-stock position cap** (long-only)
- **10 basis points (0.001) commission rate**
- **Day-by-day backtesting**
- **Performance metric:** `mean(P&L) - 0.1 * std(P&L)`

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.12+
- Rust (for Rust components)
- Julia 1.9+ (for Julia components)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anjaniputra15/algothon-quant.git
   cd algothon-quant
   ```
2. **Install Python dependencies:**
   ```bash
   pip install -e .
   # Or with extras:
   pip install -e ".[dev,rust,julia]"
   ```
3. **Build Rust components:**
   ```bash
   maturin develop
   ```
4. **Setup Julia components:**
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

---

## 🧑‍💻 Usage

### Python API Example
```python
from backend.data.loader import load_price_matrix
from backend.strategies.momentum import MomentumStrategy
from backend.evaluation.backtester import run_backtest

# Load price data
prices = load_price_matrix("prices.txt")

# Initialize and fit strategy
strategy = MomentumStrategy(lookback=10, top_n=5)
strategy.fit(prices)

# Run backtest
results = run_backtest(strategy, prices)
print(results)
```

### Command Line Interface
```bash
# Download data, calculate returns, run backtest, etc.
python -m backend.cli --help
```

### Rust & Julia Integration
- Rust and Julia modules are auto-imported if available for high-performance and advanced modeling.

---

## 🧪 Development

### Python
```bash
pip install -e ".[dev]"
pytest                # Run tests
black backend/        # Format code
isort backend/        # Sort imports
mypy backend/         # Type checking
```

### Rust
```bash
maturin develop       # Build Rust extension
cargo test            # Run Rust tests
```

### Julia
```bash
julia --project=.     # Enter Julia REPL
julia --project=. -e 'using Pkg; Pkg.test()'  # Run Julia tests
```

---

## ⚖️ Trading Constraints & Features
- **$10,000 per-stock position cap** (enforced automatically)
- **10 bps (0.001) commission** on all trades
- **Long-only positions** (no short selling)
- **Day-by-day backtesting** with realistic trading simulation
- **Performance metrics:** Sharpe ratio, risk-adjusted return, drawdown, etc.
- **Forward-filling** for missing data (weekends/holidays)
- **Extensive validation** and error handling

---

## 🛠️ Configuration

The backend uses a flexible configuration system:
```python
from backend.utils.config import config
config.set("models.default_random_state", 42)
config.save()
```

---

## 📈 Example Demos

- `demo_loader.py`: Data loading and validation
- `demo_strategies.py`: Strategy interface and constraints
- `demo_backtester.py`: Backtesting engine and performance metrics

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## 📄 License

This project is for educational and competition use. See LICENSE file for details.

---

## 🔗 Links
- [GitHub Repository](https://github.com/Anjaniputra15/algothon-quant)

---

## 👩‍💻 Authors & Credits
- Algothon Quant Team
- Polyglot backend by (aayush parashar)

---

For any questions, please open an issue on GitHub.
