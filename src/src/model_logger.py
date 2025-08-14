#!/usr/bin/env python3
"""
Model Logger - Centralized Logging System

This module provides a comprehensive logging system for the LLM training application.
It implements a singleton pattern for global logger access with thread-safe initialization,
colored console output, and structured logging capabilities.

Key Features:
- Singleton pattern with thread-safe global access
- Colored console output (INFO=green, WARNING=orange, ERROR=red)
- File logging without colors for clean log files
- Context managers for operation tracking
- Specialized logging methods for model operations
- Global convenience functions for easy access

Author: Ruben Gerad Mathew
Email: ruben_mathew@hotmail.com
Date: August 14, 2025
License: MIT License 2.0

Copyright (c) 2025 Ruben Gerad Mathew

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import logging
import threading
from contextlib import contextmanager
from typing import Optional
from functools import wraps

class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Orange/Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset to default
    }
    
    def format(self, record):
        # Add color to the log level name
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Format the message
        formatted_message = super().format(record)
        
        # Apply color to the entire message for console output
        return f"{log_color}{formatted_message}{reset_color}"

class ModelLogger:
    """
    Centralized logging class with singleton pattern for global logger instance.
    Provides structured logging for model training operations.
    """
    
    _instance = None
    _lock = threading.Lock()
    _global_logger = None
    
    def __init__(self, name: str = "ModelLogger", log_file: str = "model.log", level: int = logging.INFO):
        """Initialize logger with specified configuration"""
        self.name = name
        self.log_file = log_file
        self.level = level
        self.logger = self._setup_logger()
    
    @classmethod
    def get_global_logger(cls, name: str = "GlobalModelLogger", log_file: str = "global_model.log", level: int = logging.INFO) -> 'ModelLogger':
        """
        Get or create a global singleton logger instance.
        Thread-safe implementation.
        """
        if cls._global_logger is None:
            with cls._lock:
                if cls._global_logger is None:
                    cls._global_logger = cls(name, log_file, level)
        return cls._global_logger
    
    @classmethod
    def create_logger(cls, name: str, log_file: str, level: int = logging.INFO) -> 'ModelLogger':
        """Create a new logger instance (non-singleton)"""
        return cls(name, log_file, level)
    
    @classmethod
    def initialize_global(cls, name: str = "GlobalModelLogger", log_file: str = "global_model.log", level: int = logging.INFO):
        """Initialize the global logger with specific configuration"""
        with cls._lock:
            cls._global_logger = cls(name, log_file, level)
        return cls._global_logger
    
    def _setup_logger(self):
        """Setup logging configuration with colored console output"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatters
        # Colored formatter for console
        colored_formatter = ColorFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Regular formatter for file (no colors)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
        
        # File handler without colors
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file, mode='a')
            file_handler.setLevel(self.level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    # Basic logging methods
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_warn(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    # Specialized logging methods
    def log_model_info(self, model_name: str, operation: str, status: str, details: str = ""):
        """Log model-specific information"""
        message = f"Model: {model_name} | Operation: {operation} | Status: {status}"
        if details:
            message += f" | Details: {details}"
        self.log_info(message)
    
    def log_file_operation(self, operation: str, file_path: str, status: str, details: str = ""):
        """Log file operation information"""
        message = f"File: {operation} | Path: {file_path} | Status: {status}"
        if details:
            message += f" | Details: {details}"
        self.log_info(message)
    
    def log_environment_info(self, key: str, value: str, status: str = "loaded"):
        """Log environment information"""
        message = f"Environment: {key} | Value: {value} | Status: {status}"
        self.log_info(message)
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """Context manager for logging operation start and end"""
        self.log_info(f"Starting operation: {operation_name}")
        try:
            yield
            self.log_info(f"Completed operation: {operation_name}")
        except Exception as e:
            self.log_error(f"Failed operation: {operation_name} - Error: {str(e)}")
            raise
    
    def log_operation(self, operation_name: str):
        """Decorator for logging method operations"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.operation_context(operation_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# Global convenience functions using singleton logger
def get_logger() -> ModelLogger:
    """Get the global logger instance"""
    return ModelLogger.get_global_logger()

def initialize_global_logger(name: str = "GlobalModelLogger", log_file: str = "global_model.log", level: int = logging.INFO) -> ModelLogger:
    """
    Initialize global logger with specific configuration.
    Call this once in main.py, then use log_info, log_warn, etc. everywhere else.
    """
    return ModelLogger.initialize_global(name, log_file, level)

def log_info(message: str):
    """Global convenience function for info logging"""
    get_logger().log_info(message)

def log_warn(message: str):
    """Global convenience function for warning logging"""
    get_logger().log_warn(message)

def log_error(message: str):
    """Global convenience function for error logging"""
    get_logger().log_error(message)

def log_debug(message: str):
    """Global convenience function for debug logging"""
    get_logger().log_debug(message)

def log_critical(message: str):
    """Global convenience function for critical logging"""
    get_logger().log_critical(message)

@contextmanager
def operation_context(operation_name: str):
    """Global convenience context manager"""
    with get_logger().operation_context(operation_name):
        yield

def log_operation(operation_name: str):
    """Global convenience decorator"""
    return get_logger().log_operation(operation_name)
