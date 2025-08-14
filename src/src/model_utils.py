#!/usr/bin/env python3
"""
Model Utils - Utility Functions and Path Management

This module provides utility functions for model training operations including environment
loading, JSON file cleaning, GGUF conversion, and centralized path management. It serves
as a comprehensive toolkit for common operations in the LLM training pipeline.

Key Features:
- Centralized path management with static methods
- Environment variable loading and validation
- JSON file cleaning and validation
- GGUF model conversion with error handling
- Directory structure management
- Model subdirectory organization

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
import json
import torch
from pathlib import Path
from typing import Dict, List
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from model_logger import log_info, log_warn, log_error, log_debug, operation_context


class ModelUtils:
    """
    Utility class for environment loading, JSON cleaning, GGUF conversion, and path management.
    Uses global ModelLogger for logging functionality.
    """
    
    # Class variables for path management
    _project_root = None
    _paths_initialized = False
    
    def __init__(self):
        # Load environment variables on initialization
        self._load_env_variables()
    
    @staticmethod
    def initialize_paths(project_root: str = None):
        """Initialize all project paths with explicit project root"""
        if project_root is None:
            # Auto-detect project root (parent of src directory)
            ModelUtils._project_root = Path(__file__).parent.parent.absolute()
        else:
            ModelUtils._project_root = Path(project_root).absolute()
        
        # Ensure project root exists
        if not ModelUtils._project_root.exists():
            raise FileNotFoundError(f"Project root directory not found: {ModelUtils._project_root}")
        
        ModelUtils._paths_initialized = True
        log_info(f"Paths initialized with project root: {ModelUtils._project_root}")
        
        # Create essential directories
        ModelUtils._create_essential_directories()
    
    @staticmethod
    def _create_essential_directories():
        """Create essential directories if they don't exist"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        
        essential_dirs = [
            ModelUtils.get_outputs_dir(),
            ModelUtils.get_dist_dir(),
            ModelUtils.get_alpaca_training_dir(),
        ]
        
        for directory in essential_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_project_root() -> str:
        """Get project root directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root)
    
    @staticmethod
    def get_src_dir() -> str:
        """Get src directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / "src")
    
    @staticmethod
    def get_training_dir() -> str:
        """Get training directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / "training")
    
    @staticmethod
    def get_outputs_dir() -> str:
        """Get outputs directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / "outputs")
    
    @staticmethod
    def get_dist_dir() -> str:
        """Get dist directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / "dist")
    
    @staticmethod
    def get_resources_dir() -> str:
        """Get resources directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / "resources")
    
    @staticmethod
    def get_llama_cpp_dir() -> str:
        """Get llama.cpp directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / "llama.cpp")
    
    @staticmethod
    def get_alpaca_training_dir() -> str:
        """Get alpaca training directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / "training" / "alpaca")
    
    @staticmethod
    def get_mcp_resources_dir() -> str:
        """Get MCP resources directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / "resources" / "mcp")
    
    @staticmethod
    def get_n8n_resources_dir() -> str:
        """Get n8n resources directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / "resources" / "n8n")
    
    @staticmethod
    def get_llama_cpp_build_dir() -> str:
        """Get llama.cpp build directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / "llama.cpp" / "build")
    
    @staticmethod
    def get_env_file_path() -> str:
        """Get path to .env file"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root / ".env")
    
    @staticmethod
    def get_log_dir() -> str:
        """Get log directory"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        return str(ModelUtils._project_root)  # Logs go in project root by default
    
    @staticmethod
    def get_training_data_paths() -> List[str]:
        """Get list of training data file paths"""
        alpaca_dir = ModelUtils.get_alpaca_training_dir()
        return [
            os.path.join(alpaca_dir, "n8n_training_data.json"),
            os.path.join(alpaca_dir, "n8n_train_workflows.json"),
            os.path.join(alpaca_dir, "context7_training_data.json"),
        ]
    
    @staticmethod
    def get_model_output_dir(model_name: str) -> str:
        """Get output directory for a specific model"""
        outputs_dir = ModelUtils.get_outputs_dir()
        model_output_dir = os.path.join(outputs_dir, model_name)
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)
        return model_output_dir
    
    @staticmethod
    def get_model_save_dir(model_name: str) -> str:
        """Get save directory for a specific model"""
        dist_dir = ModelUtils.get_dist_dir()
        model_save_dir = os.path.join(dist_dir, model_name)
        Path(model_save_dir).mkdir(parents=True, exist_ok=True)
        return model_save_dir
    
    @staticmethod
    def get_model_subdirs(model_save_dir: str) -> Dict[str, str]:
        """Get model subdirectories for separated model components"""
        return {
            "lora_adapter": os.path.join(model_save_dir, "lora_adapter"),
            "merged_model": os.path.join(model_save_dir, "merged_model"),
            "base_model": os.path.join(model_save_dir, "base_model"),
        }
    
    @staticmethod
    def get_gguf_output_dir(model_save_dir: str) -> str:
        """Get GGUF output directory for a specific model"""
        return f"{model_save_dir}_gguf"
    
    @staticmethod
    def get_log_file_path(log_filename: str) -> str:
        """Get full path for a log file"""
        log_dir = ModelUtils.get_log_dir()
        return os.path.join(log_dir, log_filename)
    
    @staticmethod
    def get_checkpoint_dirs(model_output_dir: str) -> List[str]:
        """Get list of checkpoint directories in model output directory"""
        output_path = Path(model_output_dir)
        if not output_path.exists():
            return []
        
        checkpoint_dirs = []
        for item in output_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                checkpoint_dirs.append(str(item))
        
        return sorted(checkpoint_dirs)
    
    @staticmethod
    def validate_training_data_paths() -> Dict[str, bool]:
        """Validate that training data files exist"""
        training_paths = ModelUtils.get_training_data_paths()
        validation_results = {}
        
        for path in training_paths:
            file_path = Path(path)
            validation_results[path] = file_path.exists() and file_path.is_file()
        
        return validation_results
    
    @staticmethod
    def validate_essential_directories() -> Dict[str, bool]:
        """Validate that essential directories exist"""
        essential_dirs = {
            "project_root": ModelUtils.get_project_root(),
            "src_dir": ModelUtils.get_src_dir(),
            "training_dir": ModelUtils.get_training_dir(),
            "outputs_dir": ModelUtils.get_outputs_dir(),
            "dist_dir": ModelUtils.get_dist_dir(),
            "alpaca_training_dir": ModelUtils.get_alpaca_training_dir(),
        }
        
        validation_results = {}
        for name, path_str in essential_dirs.items():
            path = Path(path_str)
            validation_results[name] = path.exists() and path.is_dir()
        
        return validation_results
    
    @staticmethod
    def get_path_summary() -> Dict[str, str]:
        """Get summary of all configured paths"""
        if not ModelUtils._paths_initialized:
            raise RuntimeError("Paths not initialized. Call initialize_paths() first.")
        
        return {
            "project_root": ModelUtils.get_project_root(),
            "src_dir": ModelUtils.get_src_dir(),
            "training_dir": ModelUtils.get_training_dir(),
            "outputs_dir": ModelUtils.get_outputs_dir(),
            "dist_dir": ModelUtils.get_dist_dir(),
            "resources_dir": ModelUtils.get_resources_dir(),
            "llama_cpp_dir": ModelUtils.get_llama_cpp_dir(),
            "alpaca_training_dir": ModelUtils.get_alpaca_training_dir(),
            "mcp_resources_dir": ModelUtils.get_mcp_resources_dir(),
            "n8n_resources_dir": ModelUtils.get_n8n_resources_dir(),
            "llama_cpp_build_dir": ModelUtils.get_llama_cpp_build_dir(),
            "env_file": ModelUtils.get_env_file_path(),
            "log_dir": ModelUtils.get_log_dir(),
        }
    
    @staticmethod
    def validate_all_paths() -> Dict[str, Dict[str, bool]]:
        """Validate all paths and return comprehensive status"""
        return {
            "essential_directories": ModelUtils.validate_essential_directories(),
            "training_data_files": ModelUtils.validate_training_data_paths(),
        }
    
    def __init__(self):
        # Load environment variables on initialization if paths are initialized
        if ModelUtils._paths_initialized:
            self._load_env_variables(ModelUtils.get_env_file_path())
        else:
            # Fallback to default behavior
            self._load_env_variables()
    
    @staticmethod
    def _load_env_variables(env_file_path: str = None):
        """Load environment variables from .env file manually with explicit path"""
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
                            # Remove quotes if present
                            value = value.strip('"\'')
                            os.environ[key] = value
            log_info(f"Loaded environment variables from: {env_path}")
        else:
            log_warn(f"Warning: .env file not found at: {env_path}")

    @staticmethod
    @operation_context("JSON file cleaning")
    def clean_json_file(json_file_path: str):
        """Clean a JSON file by fixing common formatting issues"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            try:
                json.loads(content)
                return  # Already valid
            except json.JSONDecodeError:
                # Try to fix newline-delimited JSON
                lines = [line for line in content.splitlines() if line.strip()]
                objs = []
                for line in lines:
                    try:
                        objs.append(json.loads(line))
                    except Exception:
                        pass
                if objs:
                    with open(json_file_path, 'w', encoding='utf-8') as f:
                        json.dump(objs, f, ensure_ascii=False, indent=2)
                    log_info(f"Fixed malformed JSON file: {json_file_path}")
                else:
                    log_warn(f"Could not fix malformed JSON file: {json_file_path}")
        except Exception as e:
            log_error(f"Error reading file {json_file_path}: {e}")
            raise

    @staticmethod
    @operation_context("GGUF conversion")
    def convert_to_gguf(model, tokenizer, model_output_dir, useToken):
        """Convert the fine-tuned model to GGUF format""" 
        log_warn(f'useToken: {useToken}')

        # Create dedicated GGUF output directory
        gguf_output_dir = f"{model_output_dir}_gguf"
        os.makedirs(gguf_output_dir, exist_ok=True)
        
        # Check for both expected naming conventions
        gguf_q8_path = os.path.join(gguf_output_dir, "unsloth.Q8_0.gguf")
        gguf_f16_path = os.path.join(gguf_output_dir, "unsloth.f16.gguf")
        # Also check for Unsloth's actual naming convention
        alt_q8_path = f"{gguf_output_dir}.Q8_0.gguf"
        alt_f16_path = f"{gguf_output_dir}.F16.gguf"
        
        # Check if GGUF files already exist (check both naming conventions)
        q8_exists = os.path.exists(gguf_q8_path) or os.path.exists(alt_q8_path)
        f16_exists = os.path.exists(gguf_f16_path) or os.path.exists(alt_f16_path)
        
        if q8_exists or f16_exists:
            log_info(f"GGUF files found in {gguf_output_dir}:")
            if os.path.exists(gguf_q8_path):
                log_info(f"  - Q8_0: {gguf_q8_path}")
            elif os.path.exists(alt_q8_path):
                log_info(f"  - Q8_0: {alt_q8_path}")
            if os.path.exists(gguf_f16_path):
                log_info(f"  - F16: {gguf_f16_path}")
            elif os.path.exists(alt_f16_path):
                log_info(f"  - F16: {alt_f16_path}")
            
            response = input("Do you want to overwrite existing GGUF files? (y/n): ").strip().lower()
            if response != 'y':
                log_info("Skipping GGUF conversion...")
                return
            else:
                # Clean existing GGUF files if user wants to overwrite
                log_info("Cleaning existing GGUF files...")
                files_to_remove = [gguf_q8_path, gguf_f16_path, alt_q8_path, alt_f16_path]
                for file_path in files_to_remove:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        log_info(f"Removed: {file_path}")
        
        log_info(f"Converting model to GGUF format in: {gguf_output_dir}")
        
        try:
            # Use Unsloth's proper merged saving to avoid bitsandbytes state dict issues
            temp_merged_dir = f"{model_output_dir}_temp_merged"
            
            # Try Unsloth's save_pretrained_merged first (cleanest approach)
            if hasattr(model, "save_pretrained_merged"):
                log_info("Using Unsloth save_pretrained_merged for clean 16-bit save...")
                try:
                    model.save_pretrained_merged(
                        temp_merged_dir,
                        tokenizer=tokenizer,
                        save_method="merged_16bit"
                    )
                    # Check if the merge actually created the directory and files
                    if not os.path.exists(temp_merged_dir) or not os.path.exists(os.path.join(temp_merged_dir, "config.json")):
                        log_info("Merge did not create expected files, falling back to regular save...")
                        os.makedirs(temp_merged_dir, exist_ok=True)
                        model.save_pretrained(temp_merged_dir)
                        tokenizer.save_pretrained(temp_merged_dir)
                except Exception as e:
                    log_warn(f"save_pretrained_merged failed: {e}, using fallback...")
                    os.makedirs(temp_merged_dir, exist_ok=True)
                    model.save_pretrained(temp_merged_dir)
                    tokenizer.save_pretrained(temp_merged_dir)
            else:
                # Fallback: manual merge but strip quantization config
                log_info("Manual merge with quantization config cleanup...")
                merged_model = model.merge_and_unload()
                
                # Strip quantization_config from config to prevent 4-bit loading issues
                try:
                    config_dict = merged_model.config.to_dict()
                    if "quantization_config" in config_dict:
                        del config_dict["quantization_config"]
                        log_info("Removed quantization_config from model config")
                    
                    # Create new config without quantization
                    new_config = type(merged_model.config).from_dict(config_dict)
                    
                    # Update config in all nested models
                    current_model = merged_model
                    while hasattr(current_model, "model"):
                        current_model.config = new_config
                        current_model = current_model.model
                    merged_model.config = new_config
                    
                except Exception as config_err:
                    log_warn(f"Could not strip quantization_config: {config_err}")
                
                # Save the cleaned merged model
                os.makedirs(temp_merged_dir, exist_ok=True)
                merged_model.save_pretrained(temp_merged_dir)
                tokenizer.save_pretrained(temp_merged_dir)
            
            # Now load with FastLanguageModel for GGUF conversion (without quantization)
            log_info("Loading merged model for GGUF conversion...")
            
            # Ensure temp_merged_dir exists and is a valid directory before loading
            if not os.path.exists(temp_merged_dir):
                raise FileNotFoundError(f"Temporary merged directory does not exist: {temp_merged_dir}")
            
            if not os.path.isdir(temp_merged_dir):
                raise ValueError(f"Path is not a directory: {temp_merged_dir}")
            
            # Check for required model files in the directory
            required_files = ["config.json"]
            for req_file in required_files:
                file_path = os.path.join(temp_merged_dir, req_file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required model file missing: {file_path}")
            
            # Convert to absolute path to avoid any path resolution issues
            abs_temp_merged_dir = os.path.abspath(temp_merged_dir)
            log_info(f"Loading model from: {abs_temp_merged_dir}")
            
            fl_model, fl_tokenizer = FastLanguageModel.from_pretrained(
                abs_temp_merged_dir,
                max_seq_length=2048,
                dtype=torch.float16,
                load_in_4bit=False,
            )
            
            # Convert using absolute paths to avoid path issues
            log_info("Converting to GGUF format...")
            abs_gguf_dir = os.path.abspath(gguf_output_dir)
            
            # Try Q8_0 conversion
            try:
                if useToken:    
                    log_warn("Using tokenizer for Q8_0 conversion")
                    fl_model.save_pretrained_gguf(
                        save_directory=abs_temp_merged_dir, 
                        quantization_method="q8_0",
                        tokenizer=fl_tokenizer
                    )
                else:
                    log_warn("Using model only for Q8_0 conversion")
                    fl_model.save_pretrained_gguf(
                        save_directory=abs_temp_merged_dir, 
                        quantization_method="q8_0"
                    )

                # Look for the generated GGUF files and move them to the correct location
                generated_q8_file = f"{abs_temp_merged_dir}.Q8_0.gguf"
                target_q8_file = f"{abs_gguf_dir}.Q8_0.gguf"
                
                if os.path.exists(generated_q8_file):
                    os.makedirs(os.path.dirname(target_q8_file), exist_ok=True)
                    import shutil
                    shutil.move(generated_q8_file, target_q8_file)
                    log_info(f"Model saved in Q8_0 GGUF format: {target_q8_file}")
                elif os.path.exists(gguf_q8_path):
                    log_info(f"Model saved in Q8_0 GGUF format: {gguf_q8_path}")
                elif os.path.exists(alt_q8_path):
                    log_info(f"Model saved in Q8_0 GGUF format: {alt_q8_path}")
                else:
                    log_warn("Warning: Q8_0 GGUF file not found in expected locations")
            except Exception as e:
                log_error(f"Q8_0 conversion failed: {e}")
            
            # Try F16 conversion
            try:
                if useToken:
                    log_warn("Using tokenizer for f16 conversion")
                    fl_model.save_pretrained_gguf(
                        save_directory=abs_temp_merged_dir, 
                        quantization_method="f16",
                        tokenizer=fl_tokenizer
                    )
                else:
                    log_warn("Using model only for f16 conversion")
                    fl_model.save_pretrained_gguf(
                        save_directory=abs_temp_merged_dir, 
                        quantization_method="f16"
                    )

                # Look for the generated GGUF files and move them to the correct location
                generated_f16_file = f"{abs_temp_merged_dir}.F16.gguf"
                target_f16_file = f"{abs_gguf_dir}.F16.gguf"
                
                if os.path.exists(generated_f16_file):
                    os.makedirs(os.path.dirname(target_f16_file), exist_ok=True)
                    import shutil
                    shutil.move(generated_f16_file, target_f16_file)
                    log_info(f"Model saved in f16 GGUF format: {target_f16_file}")
                elif os.path.exists(gguf_f16_path):
                    log_info(f"Model saved in f16 GGUF format: {gguf_f16_path}")
                elif os.path.exists(alt_f16_path):
                    log_info(f"Model saved in f16 GGUF format: {alt_f16_path}")
                else:
                    log_warn("Warning: f16 GGUF file not found in expected locations")
            except Exception as e:
                log_error(f"f16 conversion failed: {e}")
            
            log_info(f"\nGGUF conversion completed. Files saved to: {abs_gguf_dir}/")
            log_info("You can now use these files with Ollama by creating a Modelfile.")
            
            # Clean up temporary directory
            import shutil
            if os.path.exists(temp_merged_dir):
                shutil.rmtree(temp_merged_dir)
                log_info(f"Cleaned up temporary directory: {temp_merged_dir}")
            
        except Exception as e:
            # Clean up temporary directory on error too
            import shutil
            temp_merged_dir = f"{model_output_dir}_temp_merged"
            if os.path.exists(temp_merged_dir):
                shutil.rmtree(temp_merged_dir)
                log_info(f"Cleaned up temporary directory after error: {temp_merged_dir}")
            
            log_error(f"Error during GGUF conversion: {e}")
            raise

    # Convenience methods for logging (for backward compatibility)
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


# Backward compatibility functions
def clean_json_file(file_path: str):
    """Backward compatibility function"""
    return ModelUtils.clean_json_file(file_path)

def convert_to_gguf(model, tokenizer, model_output_dir, useToken):
    """Backward compatibility function"""
    return ModelUtils.convert_to_gguf(model, tokenizer, model_output_dir, useToken)
