"""
Centralized logging configuration for the OandaScalper trading system.

This module implements Phase 1 of the High-Observability Trading System plan,
establishing a professional-grade, structured logging framework using Loguru.
"""

import sys
import os
from loguru import logger


def configure_logging():
    """
    Configure the centralized logging system with multiple sinks.
    
    This function sets up:
    1. Daily rotating system logs with JSON serialization
    2. Persistent error logs
    3. Colorized console output for development
    
    Should be called once at application startup.
    """
    # Remove the default handler to ensure clean configuration
    logger.remove()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # System Log: Daily rotating, structured JSON for comprehensive debugging
    logger.add(
        "logs/system_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",  # Rotate at midnight
        retention="7 days",  # Keep logs for 7 days
        serialize=True,  # JSON format for machine parsing
        enqueue=True,  # Non-blocking, thread-safe logging
        backtrace=True,  # Rich exception information
        diagnose=True,  # Variable values in stack traces
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    # Error Log: Persistent, non-rotating log for critical failures
    logger.add(
        "logs/errors.log",
        level="ERROR",
        serialize=True,  # JSON format for machine parsing
        enqueue=True,  # Non-blocking, thread-safe logging
        backtrace=True,  # Rich exception information
        diagnose=True,  # Variable values in stack traces
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    # Console Stream: Real-time, human-readable output for development
    logger.add(
        sys.stderr,
        level="INFO",  # Reduce console verbosity
        colorize=True,  # Color-coded output
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    logger.info("Logging system initialized with structured JSON output and multi-destination sinks")
    logger.debug("Debug logging enabled - detailed system activity will be captured")


def get_logger():
    """
    Get the configured logger instance.
    
    Returns:
        loguru.Logger: The configured logger instance
    """
    return logger


def create_trade_logger(trade_id: int, symbol: str, strategy_name: str):
    """
    Create a contextual logger bound to a specific trade.
    
    This creates a logger with persistent context that will automatically
    include trade-specific information in all log messages.
    
    Args:
        trade_id: Unique identifier for the trade
        symbol: Trading instrument (e.g., 'EUR/USD')
        strategy_name: Name of the strategy that initiated the trade
        
    Returns:
        loguru.Logger: A bound logger with trade context
    """
    return logger.bind(
        trade_id=trade_id,
        symbol=symbol,
        strategy=strategy_name
    )