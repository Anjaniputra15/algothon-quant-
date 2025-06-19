"""
Configuration utilities for the algothon-quant package.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
from loguru import logger


class Config:
    """
    Configuration manager for the algothon-quant package.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix == '.yaml':
                    return yaml.safe_load(f)
                elif self.config_path.suffix == '.json':
                    return json.load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data": {
                "cache_dir": "data/cache",
                "raw_dir": "data/raw",
                "processed_dir": "data/processed"
            },
            "models": {
                "save_dir": "models",
                "default_random_state": 42
            },
            "logging": {
                "level": "INFO",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
            },
            "polyglot": {
                "rust_enabled": True,
                "julia_enabled": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save configuration to file."""
        os.makedirs(self.config_path.parent, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            if self.config_path.suffix == '.yaml':
                yaml.dump(self.config, f, default_flow_style=False)
            elif self.config_path.suffix == '.json':
                json.dump(self.config, f, indent=2)


# Global configuration instance
config = Config() 