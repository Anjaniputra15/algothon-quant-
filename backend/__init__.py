"""
Algothon-Quant: Polyglot Monorepo for Quantitative Finance

This package provides a unified interface for quantitative finance algorithms
written in Python, Rust (via PyO3), and Julia (via PyJulia).

Main components:
- Python: Core algorithms and data processing
- Rust: High-performance numerical computations via PyO3
- Julia: Advanced mathematical modeling via PyJulia
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Version information
__version__ = "0.1.0"
__author__ = "Algothon Team"
__description__ = "Polyglot monorepo for quantitative finance algorithms"

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = Path(__file__).parent

# Ensure project root is in Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration for polyglot components
class PolyglotConfig:
    """Configuration for polyglot components (Rust, Julia, Python)."""
    
    def __init__(self):
        self.python_version = "3.12"
        self.rust_enabled = True
        self.julia_enabled = True
        
        # Rust/PyO3 configuration
        self.rust_target_dir = PROJECT_ROOT / "target"
        self.rust_lib_name = "algothon_quant"
        
        # Julia/PyJulia configuration
        self.julia_env_path = PROJECT_ROOT / "julia_env"
        self.julia_depot_path = PROJECT_ROOT / "julia_depot"
        
    def get_rust_lib_path(self) -> Optional[Path]:
        """Get the path to the compiled Rust library."""
        if not self.rust_enabled:
            return None
            
        # Try different possible locations for the compiled library
        possible_paths = [
            self.rust_target_dir / "release" / f"lib{self.rust_lib_name}.so",
            self.rust_target_dir / "debug" / f"lib{self.rust_lib_name}.so",
            self.rust_target_dir / "release" / f"{self.rust_lib_name}.dll",
            self.rust_target_dir / "debug" / f"{self.rust_lib_name}.dll",
            self.rust_target_dir / "release" / f"lib{self.rust_lib_name}.dylib",
            self.rust_target_dir / "debug" / f"lib{self.rust_lib_name}.dylib",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None

# Global configuration instance
config = PolyglotConfig()

# Import polyglot components
def _import_rust_components():
    """Import Rust components via PyO3 if available."""
    try:
        if config.rust_enabled and config.get_rust_lib_path():
            # This would be replaced with actual PyO3 imports
            # from .rust import algothon_quant_rust
            pass
    except ImportError as e:
        print(f"Warning: Rust components not available: {e}")

def _import_julia_components():
    """Import Julia components via PyJulia if available."""
    try:
        if config.julia_enabled:
            # This would be replaced with actual PyJulia imports
            # from .julia import algothon_quant_julia
            pass
    except ImportError as e:
        print(f"Warning: Julia components not available: {e}")

# Initialize polyglot components
_import_rust_components()
_import_julia_components()

# Core Python modules
from . import core
from . import data
from . import models
from . import utils

# Export main components
__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "config",
    "PROJECT_ROOT",
    "BACKEND_DIR",
    "core",
    "data", 
    "models",
    "utils",
] 