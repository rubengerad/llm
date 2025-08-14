#!/usr/bin/env python3
"""
Model Base - Abstract Base Class for Model Training

This module provides the abstract base class for all model training operations.
It implements comprehensive exception handling, environment setup, validation methods,
and common functionality shared across different model trainers.

Key Features:
- Abstract base class with common model training methods
- Comprehensive exception handling with custom ModelException
- Environment variable loading and HuggingFace authentication
- File and model state validation
- Context managers for error handling
- Logging integration with global logger

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
import torch
import glob
import json
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from functools import wraps
from contextlib import contextmanager
from huggingface_hub import login
from unsloth import FastLanguageModel
from transformers import TextStreamer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from model_logger import log_info, log_warn, log_error, log_debug, operation_context


class ModelException(Exception):
    """Custom exception for model operations"""
    def __init__(self, message: str, error_code: str = None, original_exception: Exception = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_exception = original_exception


class ModelTrainerBase(ABC):
    """
    Base class for model training with comprehensive exception handling
    """
    
    def __init__(self, model_config: Union[Dict, List], train_args: TrainingArguments):
        """
        Initialize the ModelTrainerBase with configuration
        
        Args:
            model_config: Dictionary or list containing model configuration
            train_args: TrainingArguments for training configuration
        """
        # Initialize base attributes
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
        # Parse configuration
        self._parse_config(model_config)
        self.train_args = train_args
        
        # Load environment variables
        self._load_env_file_with_paths()
        
        # Log initialization
        log_info("ModelTrainerBase initialized successfully")
    
    def _parse_config(self, model_config: Union[Dict, List]):
        """Parse model configuration from dict or list"""
        try:
            if isinstance(model_config, list):
                if len(model_config) < 4:
                    raise ModelException(
                        "model_config list must contain at least 4 elements: [model_id, max_seq_length, dtype, load_in_4bit]",
                        "CONFIG_PARSE_ERROR"
                    )
                self.model_id = model_config[0]
                self.max_seq_length = model_config[1]
                self.dtype = model_config[2]
                self.load_in_4bit = model_config[3]
                self.use_tokenizer = model_config[4] 
                
                # Default QLoRA settings for list config
                self.qlora_full = False
                self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                self.lora_alpha = 16
                self.lora_dropout = 0.1
                self.r = 16
                
            elif isinstance(model_config, dict):
                required_keys = ["model_id", "max_seq_length", "dtype", "load_in_4bit"]
                missing_keys = [key for key in required_keys if key not in model_config]
                if missing_keys:
                    raise ModelException(
                        f"model_config dictionary missing required keys: {missing_keys}",
                        "MISSING_CONFIG_KEYS"
                    )
                
                self.model_id = model_config["model_id"]
                self.max_seq_length = model_config["max_seq_length"]
                self.dtype = model_config["dtype"]
                self.load_in_4bit = model_config["load_in_4bit"]
                self.use_tokenizer = model_config.get("use_tokenizer", True)
                
                # QLoRA configuration
                self.qlora_full = model_config.get("qlora_full", False)
                self.target_modules = model_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
                self.lora_alpha = model_config.get("lora_alpha", 16)
                self.lora_dropout = model_config.get("lora_dropout", 0.1)
                self.r = model_config.get("r", 16)
            else:
                raise ModelException(
                    "model_config must be a list or dictionary",
                    "INVALID_CONFIG_TYPE"
                )
                
        except Exception as e:
            if isinstance(e, ModelException):
                raise
            raise ModelException(f"Failed to parse model configuration: {str(e)}", "CONFIG_PARSE_ERROR", e)
    

    # Exception handling decorators
    @staticmethod
    def handle_exceptions(operation_name: str = "operation"):
        """Decorator for handling exceptions in methods"""
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                try:
                    log_info(f"Starting operation: {operation_name}")
                    result = func(self, *args, **kwargs)
                    log_info(f"Completed operation: {operation_name}")
                    return result
                except ModelException as e:
                    log_error(f"{operation_name} failed with ModelException: {e}")
                    raise
                except Exception as e:
                    log_error(f"Operation failed: {operation_name} - Error: {str(e)}")
                    raise ModelException(
                        f"Failed during {operation_name}: {str(e)}",
                        "OPERATION_FAILED",
                        e
                    )
            return wrapper
        return decorator
    
    @contextmanager
    def error_context(self, operation_name: str):
        """Context manager for error handling using global logger"""
        with operation_context(operation_name):
            yield
    
    # Environment and setup methods
    def setup_environment(self):
        """Setup training environment"""
        with self.error_context("environment setup"):
            # Load environment variables
            self._load_env_file()
            
            # Setup HuggingFace authentication
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ModelException("HF_TOKEN environment variable not set", "MISSING_HF_TOKEN")
            
            try:
                login(token=hf_token)
                log_info("HuggingFace authentication successful")
            except Exception as e:
                # If it's a network error, provide a helpful message but don't fail
                if "Failed to resolve" in str(e) or "NameResolutionError" in str(e):
                    log_warn(f"HuggingFace authentication failed due to network issues: {str(e)}")
                    log_warn("Continuing in offline mode - you may need internet access to download models")
                else:
                    raise ModelException(f"HuggingFace authentication failed: {str(e)}", "HF_AUTH_FAILED", e)
            
            # Disable TorchDynamo to prevent recompilation errors
            os.environ["TORCH_COMPILE"] = "0"
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 1000
            
            log_info(f"Environment setup complete for model: {self.model_id}")
    
    def _load_env_file_with_paths(self):
        """Load environment variables using ModelUtils path management"""
        try:
            from model_utils import ModelUtils
            if ModelUtils._paths_initialized:
                env_path = ModelUtils.get_env_file_path()
                self._load_env_file(env_path)
            else:
                # Fallback to default behavior if paths not initialized
                self._load_env_file()
        except ImportError:
            # Fallback if ModelUtils not available
            self._load_env_file()
    
    def _load_env_file(self, env_file_path: str = None):
        """Load environment variables from .env file with explicit path"""
        if env_file_path is None:
            # Fallback to default path calculation
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        else:
            env_path = env_file_path
            
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            value = value.strip('"\'')
                            os.environ[key] = value
            log_info(f"Loaded environment variables from: {env_path}")
        else:
            log_warn(f"Warning: .env file not found at: {env_path}")
    
    # Validation methods
    def validate_model_state(self, require_model: bool = True, require_tokenizer: bool = True, require_dataset: bool = False):
        """Validate that required components are loaded"""
        if require_model and self.model is None:
            raise ModelException("Model not loaded", "MODEL_NOT_LOADED")
        if require_tokenizer and self.tokenizer is None:
            raise ModelException("Tokenizer not loaded", "TOKENIZER_NOT_LOADED")
        if require_dataset and self.dataset is None:
            raise ModelException("Dataset not loaded", "DATASET_NOT_LOADED")
    
    # Convenience logging methods that delegate to global logger
    def log_info(self, message: str):
        """Log info message using global logger"""
        log_info(message)
    
    def log_warn(self, message: str):
        """Log warning message using global logger"""
        log_warn(message)
    
    def log_error(self, message: str):
        """Log error message using global logger"""
        log_error(message)
    
    def log_debug(self, message: str):
        """Log debug message using global logger"""
        log_debug(message)
    
    def validate_file_exists(self, file_path: str, description: str = "file"):
        """Validate that a file exists"""
        if not os.path.exists(file_path):
            raise ModelException(f"{description} not found: {file_path}", "FILE_NOT_FOUND")
    
    def validate_directory_exists(self, dir_path: str, description: str = "directory"):
        """Validate that a directory exists"""
        if not os.path.exists(dir_path):
            raise ModelException(f"{description} not found: {dir_path}", "DIRECTORY_NOT_FOUND")
        if not os.path.isdir(dir_path):
            raise ModelException(f"Path is not a directory: {dir_path}", "NOT_A_DIRECTORY")
    
    # Utility methods
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            if self.dataset is not None:
                del self.dataset
                self.dataset = None
            if self.trainer is not None:
                del self.trainer
                self.trainer = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            log_info("Cleanup completed successfully")
        except Exception as e:
            log_error(f"Error during cleanup: {str(e)}")
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def load_model(self):
        """Load and prepare the model for training"""
        pass
    
    @abstractmethod
    def load_data(self, training_data_path: Union[str, List[str]], model_dir: Optional[str] = None):
        """Load and prepare training data"""
        pass
    
    @abstractmethod
    def train(self, output_dir: str = "outputs"):
        """Train the model"""
        pass
    
    @abstractmethod
    def save_model(self, save_dir: str, push_to_hub: bool = False, hub_name: Optional[str] = None):
        """Save the trained model"""
        pass
    
    # Context manager support
    def __enter__(self):
        """Enter context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        self.cleanup()
        
        if exc_type is not None:
            log_error(f"Exception occurred: {exc_type.__name__}: {exc_val}")
            if exc_type == ModelException:
                return False  # Don't suppress ModelException
            else:
                # Convert other exceptions to ModelException
                raise ModelException(f"Unexpected error: {str(exc_val)}", "CONTEXT_ERROR", exc_val)
        
        return False
