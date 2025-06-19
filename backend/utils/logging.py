"""
Logging utilities for the algothon-quant package.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[Path] = None,
                 log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        log_format: Log message format
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation="10 MB",
            retention="30 days"
        )


def get_logger(name: str = "algothon_quant"):
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Setup default logging
setup_logging() 