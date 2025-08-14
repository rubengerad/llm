#!/usr/bin/env python3
"""
LLM Training Application - Main Entry Point

This module serves as the main entry point for the LLM (Large Language Model) training application.
It provides a comprehensive pipeline for training language models using Unsloth, with support for
QLoRA (Quantized Low-Rank Adaptation) fine-tuning, model saving, and GGUF conversion for deployment.

Key Features:
- Centralized path management and configuration
- Support for multiple model architectures (TinyLlama, Gemma3)
- QLoRA full training with customizable parameters
- Automatic model saving in multiple formats
- GGUF conversion for Ollama deployment
- Comprehensive logging and error handling

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
from pathlib import Path
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments
from model_trainer import ModelTrainer
from model_utils import ModelUtils
from model_logger import initialize_global_logger, log_info, log_warn, log_error, operation_context

# Explicitly configure all directory paths in main
def configure_project_paths():
    """Explicitly configure all project directory paths"""
    # Get the project root (parent of src directory where this file is located)
    current_file = Path(__file__).absolute()
    project_root = current_file.parent.parent
    
    # Explicitly define all directory paths
    paths_config = {
        "project_root": str(project_root),
        "src_dir": str(project_root / "src"),
        "training_dir": str(project_root / "training"),
        "outputs_dir": str(project_root / "training"),
        "dist_dir": str(project_root / "dist"),
        "resources_dir": str(project_root / "resources"),
        "llama_cpp_dir": str(project_root / "llama.cpp"),
        "alpaca_training_dir": str(project_root / "training" / "alpaca"),
        "mcp_resources_dir": str(project_root / "resources" / "mcp"),
        "n8n_resources_dir": str(project_root / "resources" / "n8n"),
        "llama_cpp_build_dir": str(project_root / "llama.cpp" / "build"),
        "env_file": str(project_root / ".env"),
        "log_dir": str(project_root),
    }
    
    # Create essential directories that must exist
    essential_dirs = [
        paths_config["outputs_dir"],
        paths_config["dist_dir"],
        paths_config["alpaca_training_dir"],
    ]
    
    for dir_path in essential_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    log_info(f"Project paths configured with root: {paths_config['project_root']}")
    return paths_config

# Configure project paths explicitly
project_paths = configure_project_paths()

# Initialize centralized path configuration using ModelUtils with explicit root
ModelUtils.initialize_paths(project_paths["project_root"])

# Initialize global logger once at startup - this is the ONLY place you need to initialize  
initialize_global_logger("TrainingApp", os.path.join(project_paths["log_dir"], "training_app.log"))

# Ensure environment variables are loaded with explicit path
ModelUtils._load_env_variables(project_paths["env_file"])

# Model configurations
MODELS = {
    "tiny_llama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_seq_length": 2048,
        "dtype": None,
        "load_in_4bit": True,
        "use_tokenizer": True,
    },
    "gemma3_base": {
        "model_id": "google/gemma-3-1b-it",
        "max_seq_length": 2048,
        "dtype": None,
        "load_in_4bit": False,
        "use_tokenizer": False,
    },
    "deepseek_qwen" : {
        "model_id": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
        "max_seq_length": 2048,
        "dtype": None,
        "load_in_4bit": False,
        "use_tokenizer": True,
    },
    "gemma3_unsloth": {
        "model_id": "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "max_seq_length": 2048,
        "dtype": None,
        "load_in_4bit": False,
        "use_tokenizer": False,
    },
    "gemma3_qlora_full": {
        "model_id": "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "max_seq_length": 2048,
        "dtype": None,
        "load_in_4bit": True,
        "use_tokenizer": False,
        "qlora_full": True,  # New flag for QLoRA full training
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "r": 64,  # Higher rank for full training
    }
}

SELECTED_MODEL = "deepseek_qwen" # Change this to select the model you want to train

def main(args=None):
    with operation_context("main_application"):
        try:
            log_info("Starting training application...")
            
            # Get training data paths from explicit configuration
            training_data_paths = [
                os.path.join(project_paths["alpaca_training_dir"], "n8n_training_data.json"),
                os.path.join(project_paths["alpaca_training_dir"], "n8n_train_workflows.json"),
                os.path.join(project_paths["alpaca_training_dir"], "context7_training_data.json"),
            ]
            log_info(f"Training data files configured: {len(training_data_paths)}")
            for path in training_data_paths:
                log_info(f"  - {path}")
            
            # Check GPU availability and memory
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                log_info(f"GPU Available: {gpu_name}")
                log_info(f"GPU Memory: {gpu_memory:.1f} GB")
            else:
                log_warn("CUDA not available, training will be very slow")
            
            # Get model configuration
            model_config = MODELS[SELECTED_MODEL]
            
            # Use explicit path configuration for model directories
            output_dir = os.path.join(project_paths["outputs_dir"], SELECTED_MODEL)
            save_dir = os.path.join(project_paths["dist_dir"], SELECTED_MODEL)
            
            # Create model directories explicitly
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            log_info(f"Selected model: {SELECTED_MODEL}")
            log_info(f"Model ID: {model_config['model_id']}")
            log_info(f"Output directory: {output_dir}")
            log_info(f"Save directory: {save_dir}")

            # Adjust training arguments based on QLoRA full training
            if model_config.get("qlora_full", False):
                # QLoRA full training typically needs different settings
                batch_size = 1  # Smaller batch size for full training
                gradient_accumulation = 16  # Higher accumulation
                learning_rate = 1e-4  # Lower learning rate for stability
                max_steps = 50  # More steps for full training
                warmup_steps = 100
                log_info("Using QLoRA full training configuration")
            else:
                # Standard training settings
                batch_size = 2
                gradient_accumulation = 8
                learning_rate = 2e-4
                max_steps = 50
                warmup_steps = 10
                log_info("Using standard training configuration")

            # Create trainer instance with optimized settings for smaller GPU
            log_info("Creating training arguments...")
            log_info(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation}")
            log_info(f"Learning rate: {learning_rate}, Max steps: {max_steps}")
            
            training_args = TrainingArguments(
                            per_device_train_batch_size=batch_size,
                            gradient_accumulation_steps=gradient_accumulation,
                            warmup_steps=warmup_steps,
                            max_steps=max_steps,
                            learning_rate=learning_rate,
                            fp16=not is_bfloat16_supported(),
                            bf16=is_bfloat16_supported(),
                            logging_steps=1,
                            optim="adamw_8bit",
                            weight_decay=0.01,
                            lr_scheduler_type="linear",
                            seed=3407,
                            output_dir=output_dir,
                        )
            
            log_info("Creating ModelTrainer instance...")
            # ModelTrainer will automatically use the global logger
            trainer = ModelTrainer(model_config, training_args)

            # Run full training pipeline
            log_info("Starting full training pipeline...")
            success = trainer.full_training_pipeline(
                training_data=training_data_paths,
                model_dir=output_dir,
                save_dir=save_dir,
                output_dir=output_dir,
                push_to_hub=False,
                hub_name=None,
                convert_gguf=True,
            )
            
            if success:
                log_info("Training completed successfully!")
            else:
                log_error("Training failed!")
                
        except Exception as e:
            log_error(f"Error during training: {e}")
            log_error("Full traceback:")
            import traceback
            log_error(traceback.format_exc())
            return False
    
    return True

if __name__ == "__main__":
    main()

