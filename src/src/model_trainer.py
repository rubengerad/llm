#!/usr/bin/env python3
"""
Model Trainer - Concrete Implementation of Model Training Pipeline

This module implements the complete model training pipeline for LLM fine-tuning using Unsloth.
It provides support for QLoRA training, model saving in multiple formats, GGUF conversion,
and comprehensive training workflow management.

Key Features:
- Complete QLoRA fine-tuning pipeline
- Support for multiple model architectures (TinyLlama, Gemma3)
- Separated model directory structure (LoRA, merged, base models)
- GGUF conversion for deployment
- Inference testing and validation
- Comprehensive error handling and logging

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
from typing import Optional, Union, List

from huggingface_hub import login
from unsloth import FastLanguageModel
from transformers import TextStreamer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from model_utils import ModelUtils 
from model_base import ModelTrainerBase, ModelException
from model_logger import log_info, log_warn, log_error, log_debug, operation_context

class ModelTrainer(ModelTrainerBase):
    def __init__(self, model_config: Union[dict, list], train_args: TrainingArguments):
        """
        Initialize the ModelTrainer with configuration
        
        Args:
            model_config: Dictionary or list containing model configuration
            train_args: TrainingArguments for training configuration
        """
        # Initialize base class - it will use global logger
        super().__init__(model_config, train_args)
        
        # Training configuration
        self.alpaca_prompt = """Below is an instruction that describes a task. Write a input that appropriately requests key information for completing the task as itemized list. Clean any field data, secure keys or client specific data within the output.
        ### Instruction:
        {}
        ### Input:
        {}
        ### Output:
        {}"""
        
        # Example data for testing
        self.instruction = "Create an n8n workflow with an HTTP Request node to fetch Airbnb property listings for a specific location. Configure the request parameters, headers, and handle the response data appropriately."
        self.input_example = "Location: Paris, France\nCheck-in: 2024-03-15\nCheck-out: 2024-03-20\nGuests: 2\nProperty type: Entire place"

    def setup_environment(self):
        """Setup environment variables and authentication"""
        with self.error_context("environment setup"):
            # Call parent setup first
            super().setup_environment()
            
            # Additional configuration specific to ModelTrainer
            concurrency = os.getenv("PARALLEL_TRAINING", "true")
            gpu = os.getenv("GPU", "0")
            
            os.environ["TOKENIZERS_PARALLELISM"] = concurrency
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                log_warn("CUDA is not available, training will be very slow on CPU")

    @ModelTrainerBase.handle_exceptions("data loading")
    def load_data(self, training_data_path: Union[str, List[str]], model_dir: Optional[str] = None):
        """Load and prepare training data"""
        # Check for existing model if model_dir is provided
        if model_dir and os.path.exists(model_dir):
            # Use ModelUtils to get model subdirectories
            model_subdirs = ModelUtils.get_model_subdirs(model_dir)
            lora_adapter_dir = model_subdirs["lora_adapter"]
            merged_model_dir = model_subdirs["merged_model"]
            base_model_dir = model_subdirs["base_model"]
            
            # Also check for checkpoint directories
            checkpoint_dirs = ModelUtils.get_checkpoint_dirs(model_dir)
            
            existing_model_found = False
            model_type = None
            model_path = None
            
            # Check for LoRA adapter
            if os.path.exists(lora_adapter_dir) and os.path.exists(os.path.join(lora_adapter_dir, "adapter_config.json")):
                existing_model_found = True
                model_type = "LoRA adapter"
                model_path = lora_adapter_dir
                log_info(f"Found LoRA adapter at: {lora_adapter_dir}")
                
            # Check for merged model
            elif os.path.exists(merged_model_dir) and os.path.exists(os.path.join(merged_model_dir, "config.json")):
                existing_model_found = True
                model_type = "merged model"
                model_path = merged_model_dir
                log_info(f"Found merged model at: {merged_model_dir}")
                
            # Check for base model
            elif os.path.exists(base_model_dir) and os.path.exists(os.path.join(base_model_dir, "config.json")):
                existing_model_found = True
                model_type = "base model"
                model_path = base_model_dir
                log_info(f"Found base model at: {base_model_dir}")
                
            # Check checkpoint directories for LoRA adapters
            else:
                for checkpoint_dir in checkpoint_dirs:
                    if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
                        existing_model_found = True
                        model_type = "LoRA checkpoint"
                        model_path = checkpoint_dir
                        log_info(f"Found LoRA checkpoint at: {checkpoint_dir}")
                        break
            
            if existing_model_found:
                try:
                    response = input(f"Do you want to overwrite the existing {model_type}? (y/n): ").strip().lower()
                    if response != 'y':
                        log_info("Skipping training...")
                        return "skip"  # Return "skip" instead of False to indicate user choice
                    else:
                        log_info(f"User chose to overwrite existing {model_type}. Continuing with training...")
                except (EOFError, KeyboardInterrupt):
                    log_info("User interrupted, skipping training...")
                    return "skip"  # Return "skip" instead of False to indicate user choice

        # Validate training data path
        if not training_data_path:
            raise ModelException("training_data_path cannot be None or empty", "INVALID_DATA_PATH")

        # Clean JSON files before loading - use ModelUtils with explicit paths
        if isinstance(training_data_path, list):
            for file in training_data_path:
                self.validate_file_exists(file, f"Training file")
                ModelUtils.clean_json_file(file)
        elif os.path.isdir(training_data_path):
            json_files = glob.glob(os.path.join(training_data_path, '*.json'))
            if not json_files:
                raise ModelException(f"No JSON files found in directory: {training_data_path}", "NO_JSON_FILES")
            for file in json_files:
                ModelUtils.clean_json_file(file)
        else:
            self.validate_file_exists(training_data_path, "Training file")
            ModelUtils.clean_json_file(training_data_path)

        # Load dataset
        if isinstance(training_data_path, list):
            log_info(f"Loading {len(training_data_path)} training files:")
            for file in training_data_path:
                log_info(f"  - {file}")
            self.dataset = load_dataset('json', data_files=training_data_path)
        elif os.path.isdir(training_data_path):
            json_files = glob.glob(os.path.join(training_data_path, '*.json'))
            if len(json_files) > 1:
                log_info(f"Found {len(json_files)} JSON files for training:")
                for file in json_files:
                    log_info(f"  - {file}")
                self.dataset = load_dataset('json', data_files=json_files)
            elif len(json_files) == 1:
                log_info(f"Found 1 JSON file for training: {json_files[0]}")
                self.dataset = load_dataset('json', data_files=json_files[0])
            else:
                raise ModelException(f"No JSON files found in directory: {training_data_path}", "NO_JSON_FILES")
        else:
            log_info(f"Loading single training file: {training_data_path}")
            self.dataset = load_dataset('json', data_files=training_data_path)

        if not self.dataset or 'train' not in self.dataset:
            raise ModelException("Failed to load dataset or dataset has no 'train' split", "DATASET_LOAD_FAILED")
        
        dataset_size = len(self.dataset['train'])
        if dataset_size == 0:
            raise ModelException("Dataset is empty", "EMPTY_DATASET")
        
        log_info(f"\nDataset loaded successfully!")
        log_info(f"Number of training examples: {dataset_size}")
        return True

    @ModelTrainerBase.handle_exceptions("model loading")
    def load_model(self):
        """Load and prepare the model for training"""
        log_info(f"Loading model: {self.model_id}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            token=os.getenv("HF_TOKEN")
        )
        
        if self.model is None or self.tokenizer is None:
            raise ModelException("Failed to load model or tokenizer", "MODEL_LOAD_FAILED")
        
        log_info(f"Model loaded successfully: {self.model_id}")

    @ModelTrainerBase.handle_exceptions("inference testing")
    def test_inference(self):
        """Test model inference before training"""
        self.validate_model_state(require_model=True, require_tokenizer=True)
        
        FastLanguageModel.for_inference(self.model)
        inputs = self.tokenizer([
            self.alpaca_prompt.format(
                self.instruction,
                self.input_example,
                "",
            )
        ], return_tensors="pt").to("cuda")

        text_streamer = TextStreamer(self.tokenizer)
        _ = self.model.generate(**inputs, streamer=text_streamer, max_new_tokens=1000)
        log_info("Inference test completed successfully")

    @ModelTrainerBase.handle_exceptions("dataset preparation")
    def prepare_dataset(self):
        """Prepare dataset with formatting using the loaded training data"""
        self.validate_model_state(require_tokenizer=True, require_dataset=True)
        
        EOS_TOKEN = self.tokenizer.eos_token
        if not EOS_TOKEN:
            log_warn("EOS token not found, using default")
            EOS_TOKEN = "</s>"

        def formatting_prompts_func(examples):
            try:
                instructions = examples["instruction"]
                inputs = examples["input"]
                outputs = examples["output"]
                texts = []
                
                for instruction, input_text, output in zip(instructions, inputs, outputs):
                    if not instruction or not output:
                        log_warn("Skipping empty instruction or output")
                        continue
                    text = self.alpaca_prompt.format(instruction, input_text or "", output) + EOS_TOKEN
                    texts.append(text)
                
                return {"text": texts}
            except Exception as e:
                raise ModelException(f"Error in formatting_prompts_func: {e}", "FORMATTING_ERROR", e)

        log_info("Using loaded dataset for training")
        # Use loaded dataset and apply formatting
        original_dataset = self.dataset['train'] if 'train' in self.dataset else self.dataset
        self.dataset = original_dataset.map(formatting_prompts_func, batched=True)
        
        if not self.dataset or len(self.dataset) == 0:
            raise ModelException("Dataset is empty after formatting", "EMPTY_FORMATTED_DATASET")
        
        log_info(f"Dataset prepared successfully with {len(self.dataset)} examples")

    @ModelTrainerBase.handle_exceptions("LoRA setup")
    def setup_lora(self):
        """Setup LoRA configuration for PEFT"""
        self.validate_model_state(require_model=True)
        
        if self.qlora_full:
            log_info(f"Setting up QLoRA full training with r={self.r}, alpha={self.lora_alpha}")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.r,  # Use configurable rank
                target_modules=self.target_modules,  # Use configurable target modules
                lora_alpha=self.lora_alpha,  # Use configurable alpha
                lora_dropout=self.lora_dropout,  # Use configurable dropout
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
            log_info("QLoRA full training configuration applied successfully")
        else:
            log_info("Setting up standard LoRA configuration")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
            log_info("Standard LoRA configuration applied successfully")

    @ModelTrainerBase.handle_exceptions("model training")
    def train(self, output_dir: str = "outputs"):
        """Train the model"""
        self.validate_model_state(require_model=True, require_tokenizer=True, require_dataset=True)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure compilation is disabled before creating trainer
        torch._dynamo.disable()
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=1,  # Reduced to avoid multiprocessing issues
            packing=False,  # Keep packing disabled to avoid dynamic shapes
            args=self.train_args,
        )

        # Show memory stats
        try:
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            log_info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            log_info(f"{start_gpu_memory} GB of memory reserved.")
        except Exception as e:
            log_warn(f"Could not get GPU stats: {e}")

        # Train
        log_info("Starting model training...")
        try:
            trainer_stats = self.trainer.train()
        except Exception as e:
            if "recompile_limit" in str(e):
                log_error("TorchDynamo recompilation error detected. Attempting workaround...")
                # Force disable compilation and try again
                torch._dynamo.reset()
                torch._dynamo.disable()
                trainer_stats = self.trainer.train()
            else:
                raise e

        # Show final stats
        try:
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
            
            log_info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
            log_info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
            log_info(f"Peak reserved memory = {used_memory} GB.")
            log_info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
            log_info(f"Peak reserved memory % of max memory = {used_percentage} %.")
            log_info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        except Exception as e:
            log_warn(f"Could not calculate memory stats: {e}")
        
        log_info("Training completed successfully")

    @ModelTrainerBase.handle_exceptions("post-training inference testing")
    def test_after_training(self):
        """Test model after training"""
        self.validate_model_state(require_model=True, require_tokenizer=True)
        
        FastLanguageModel.for_inference(self.model)
        inputs = self.tokenizer([
            self.alpaca_prompt.format(
                self.instruction,
                self.input_example,
                "",
            )
        ], return_tensors="pt").to("cuda")

        text_streamer = TextStreamer(self.tokenizer)
        _ = self.model.generate(**inputs, streamer=text_streamer, max_new_tokens=1000)
        log_info("Post-training inference test completed successfully")

    @ModelTrainerBase.handle_exceptions("model saving")
    def save_model(self, save_dir: str = "lora_model", push_to_hub: bool = False, hub_name: Optional[str] = None):
        """Save the trained model with proper separation of LoRA adapters and base models"""
        self.validate_model_state(require_model=True, require_tokenizer=True)
        
        # Determine if this is a LoRA adapter or full model
        is_lora_adapter = hasattr(self.model, 'peft_config') and self.model.peft_config is not None
        
        if is_lora_adapter:
            # Use ModelUtils to get model subdirectories
            model_subdirs = ModelUtils.get_model_subdirs(save_dir)
            lora_save_dir = model_subdirs["lora_adapter"]
            base_save_dir = model_subdirs["merged_model"]
            
            # Create directories
            os.makedirs(lora_save_dir, exist_ok=True)
            os.makedirs(base_save_dir, exist_ok=True)
            
            # Save only the LoRA adapter weights
            self.model.save_pretrained(lora_save_dir)
            self.tokenizer.save_pretrained(lora_save_dir)
            log_info(f"LoRA adapter saved to {lora_save_dir}")
            
            try:
                # Merge and save the full model
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(base_save_dir)
                self.tokenizer.save_pretrained(base_save_dir)
                log_info(f"Merged base model saved to {base_save_dir}")
                
                # Clean up merged model from memory
                del merged_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                log_warn(f"Could not save merged model: {e}")
                
        else:
            # Save full base model
            model_subdirs = ModelUtils.get_model_subdirs(save_dir)
            base_save_dir = model_subdirs["base_model"]
            
            os.makedirs(base_save_dir, exist_ok=True)
            
            self.model.save_pretrained(base_save_dir)
            self.tokenizer.save_pretrained(base_save_dir)
            log_info(f"Base model saved to {base_save_dir}")

        if push_to_hub:
            if not hub_name:
                raise ModelException("hub_name must be provided when push_to_hub is True", "MISSING_HUB_NAME")
            
            self.model.push_to_hub(hub_name, token=os.getenv("HF_TOKEN"))
            self.tokenizer.push_to_hub(hub_name, token=os.getenv("HF_TOKEN"))
            log_info(f"Model pushed to hub: {hub_name}")

    @ModelTrainerBase.handle_exceptions("GGUF conversion")
    def convert_to_gguf(self, model_output_dir: str):
        """Convert model to GGUF format using existing model directory"""
        self.validate_model_state(require_model=True, require_tokenizer=True)
        
        log_info("Starting GGUF conversion...")
        
        # For GGUF output, use the parent directory of model_output_dir to avoid nested paths
        # If model_output_dir is like "dist/model_name", use that as base
        # If model_output_dir is like "dist/model_name/subdir", use parent
        parent_dir = os.path.dirname(model_output_dir) if model_output_dir.endswith(('merged_model', 'base_model', 'lora_adapter')) else model_output_dir
        gguf_output_dir = ModelUtils.get_gguf_output_dir(parent_dir)
        
        # Check if GGUF files already exist
        existing_gguf_files = [
            f"{gguf_output_dir}.Q8_0.gguf",
            f"{gguf_output_dir}.F16.gguf"
        ]
        
        existing_files = [f for f in existing_gguf_files if os.path.exists(f)]
        if existing_files:
            log_info(f"Found existing GGUF files: {existing_files}")
            response = input("Do you want to overwrite existing GGUF files? (y/n): ").strip().lower()
            if response != 'y':
                log_info("Skipping GGUF conversion...")
                return
            # Clean existing files
            for file_path in existing_files:
                os.remove(file_path)
                log_info(f"Removed existing file: {file_path}")
        
        # For 4-bit models, try direct conversion first (no temporary files needed)
        if self.load_in_4bit:
            log_info("Attempting direct GGUF conversion from 4-bit model...")
            try:
                # Try direct conversion without tokenizer first (often works better)
                self.model.save_pretrained_gguf(
                    save_directory=gguf_output_dir,
                    quantization_method="q8_0"
                )
                log_info(f"Q8_0 GGUF conversion completed: {gguf_output_dir}.Q8_0.gguf")
                
                # Try F16 conversion
                try:
                    self.model.save_pretrained_gguf(
                        save_directory=gguf_output_dir,
                        quantization_method="f16"
                    )
                    log_info(f"F16 GGUF conversion completed: {gguf_output_dir}.F16.gguf")
                except Exception as f16_err:
                    log_warn(f"F16 conversion failed (this is normal): {f16_err}")
                
                return  # Success - no need for fallback
                
            except Exception as direct_err:
                log_info(f"Direct conversion failed: {direct_err}")
                log_info("Falling back to merge-based conversion...")
        
        # Fallback: Use existing model directory as source for conversion
        log_info(f"Using model from existing directory: {model_output_dir}")
        
        # Determine which model to use for GGUF conversion based on new directory structure
        conversion_source = None
        
        # Use ModelUtils to get model subdirectories
        model_subdirs = ModelUtils.get_model_subdirs(model_output_dir)
        merged_model_dir = model_subdirs["merged_model"]
        base_model_dir = model_subdirs["base_model"]
        lora_adapter_dir = model_subdirs["lora_adapter"]
        
        # Priority order: merged_model > base_model > lora_adapter
        if os.path.exists(merged_model_dir) and os.path.exists(os.path.join(merged_model_dir, "config.json")):
            conversion_source = merged_model_dir
            log_info(f"Using merged model for GGUF conversion: {merged_model_dir}")
        elif os.path.exists(base_model_dir) and os.path.exists(os.path.join(base_model_dir, "config.json")):
            conversion_source = base_model_dir
            log_info(f"Using base model for GGUF conversion: {base_model_dir}")
        elif os.path.exists(lora_adapter_dir) and os.path.exists(os.path.join(lora_adapter_dir, "adapter_config.json")):
            # For LoRA adapter, we need to use the in-memory model for conversion
            log_info("Detected LoRA adapter model, using in-memory model for GGUF conversion")
            
            try:
                log_info("Converting LoRA model to Q8_0 GGUF format...")
                if self.use_tokenizer:
                    self.model.save_pretrained_gguf(
                        save_directory=conversion_source,#gguf_output_dir,
                        quantization_method="q8_0",
                        tokenizer=self.tokenizer
                    )
                else:
                    self.model.save_pretrained_gguf(
                        save_directory=conversion_source,#gguf_output_dir,
                        quantization_method="q8_0"
                    )
                log_info(f"Q8_0 GGUF saved: {conversion_source}.Q8_0.gguf")
                
                # Try F16 conversion
                try:
                    log_info("Converting LoRA model to F16 GGUF format...")
                    if self.use_tokenizer:
                        self.model.save_pretrained_gguf(
                            save_directory=conversion_source,#gguf_output_dir,
                            quantization_method="f16",
                            tokenizer=self.tokenizer
                        )
                    else:
                        self.model.save_pretrained_gguf(
                            save_directory=conversion_source,#gguf_output_dir,
                            quantization_method="f16"
                        )
                    log_info(f"F16 GGUF saved: {conversion_source}.F16.gguf")
                except Exception as f16_err:
                    log_warn(f"F16 conversion failed (this is normal): {f16_err}")
                
                log_info("GGUF conversion completed successfully")
                return
                
            except Exception as lora_err:
                log_warn(f"LoRA direct conversion failed: {lora_err}")
                log_error("No suitable model found for GGUF conversion")
                raise ModelException(f"Failed to convert LoRA model to GGUF: {lora_err}", "LORA_CONVERSION_FAILED", lora_err)
        else:
            # Check for old-style checkpoint directories
            checkpoint_dirs = ModelUtils.get_checkpoint_dirs(model_output_dir)
                
            if checkpoint_dirs:
                # Use the latest checkpoint
                latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)
                if os.path.exists(os.path.join(latest_checkpoint, "adapter_config.json")):
                    log_info(f"Using checkpoint LoRA adapter: {latest_checkpoint}")
                    # Use in-memory model for LoRA checkpoint conversion
                    try:
                        log_info("Converting checkpoint LoRA model to GGUF format...")
                        merged_model = self.model.merge_and_unload()
                        if self.use_tokenizer:
                            merged_model.save_pretrained_gguf(
                                save_directory=latest_checkpoint, #gguf_output_dir,
                                quantization_method="q8_0",
                                tokenizer=self.tokenizer
                            )
                        else:
                            merged_model.save_pretrained_gguf(
                                save_directory=latest_checkpoint, #gguf_output_dir,
                                quantization_method="q8_0"
                            )
                        log_info(f"Q8_0 GGUF saved: {latest_checkpoint}.Q8_0.gguf")
                        
                        # Clean up merged model from memory
                        del merged_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        log_info("GGUF conversion completed successfully")
                        return
                        
                    except Exception as checkpoint_err:
                        log_error(f"Checkpoint LoRA conversion failed: {checkpoint_err}")
                        raise ModelException(f"Failed to convert checkpoint LoRA to GGUF: {checkpoint_err}", "CHECKPOINT_CONVERSION_FAILED", checkpoint_err)
                else:
                    conversion_source = latest_checkpoint
            else:
                raise ModelException(f"No suitable model found in directory: {model_output_dir}", "NO_MODEL_FOUND")
        
        if conversion_source:
            # Load the model from the determined source directory
            try:
                # Use the saved model directly for GGUF conversion
                conversion_model, conversion_tokenizer = FastLanguageModel.from_pretrained(
                    conversion_source,
                    max_seq_length=self.max_seq_length,
                    dtype=torch.float16,
                    load_in_4bit=False,  # Load as 16-bit for conversion
                    token=os.getenv("HF_TOKEN")
                )
                
                log_info("Converting to Q8_0 GGUF format...")
                if self.use_tokenizer:
                    conversion_model.save_pretrained_gguf(
                        save_directory=conversion_source, #gguf_output_dir,
                        quantization_method="q8_0",
                        tokenizer=conversion_tokenizer
                    )
                else:
                    conversion_model.save_pretrained_gguf(
                        save_directory=conversion_source, #gguf_output_dir,
                        quantization_method="q8_0"
                    )
                log_info(f"Q8_0 GGUF saved: {conversion_source}.Q8_0.gguf")

                # Try F16 conversion
                try:
                    log_info("Converting to F16 GGUF format...")
                    if self.use_tokenizer:
                        conversion_model.save_pretrained_gguf(
                            save_directory=conversion_source,
                            quantization_method="f16",
                            tokenizer=conversion_tokenizer
                        )
                    else:
                        conversion_model.save_pretrained_gguf(
                            save_directory=conversion_source,
                            quantization_method="f16"
                        )
                    log_info(f"F16 GGUF saved: {conversion_source}.F16.gguf")
                except Exception as f16_err:
                    log_warn(f"F16 conversion failed (this is normal): {f16_err}")
                
                # Clean up the loaded models from memory
                del conversion_model, conversion_tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as load_err:
                # Final fallback: use ModelUtils with the existing directory
                log_warn(f"FastLanguageModel loading failed: {load_err}")
                log_info("Using ModelUtils fallback conversion...")
                ModelUtils.convert_to_gguf(self.model, self.tokenizer, conversion_source, self.use_tokenizer)
        
        log_info("GGUF conversion completed successfully")

    @ModelTrainerBase.handle_exceptions("full training pipeline")
    def full_training_pipeline(self, training_data: Union[str, List[str]], model_dir: str, 
                             save_dir: str, output_dir: str, push_to_hub: bool = False, 
                             hub_name: Optional[str] = None, convert_gguf: bool = False):
        """Execute the complete training pipeline using loaded training data"""
        log_info("Starting full training pipeline...")
        
        # Setup
        log_info("Step 1: Setting up environment...")
        self.setup_environment()
        
        # Load data and check for existing model
        log_info("Step 2: Loading training data...")
        data_load_result = self.load_data(training_data, model_dir)
        
        if data_load_result == "skip":
            # User chose to skip training, use existing model
            log_info("Training skipped by user choice. Loading existing model for post-processing...")
            
            # Find the existing model using the new directory structure
            model_subdirs = ModelUtils.get_model_subdirs(model_dir)
            lora_adapter_dir = model_subdirs["lora_adapter"]
            merged_model_dir = model_subdirs["merged_model"]
            base_model_dir = model_subdirs["base_model"]
            checkpoint_dirs = ModelUtils.get_checkpoint_dirs(model_dir)
            
            model_to_load = None
            
            # Priority: merged > base > lora_adapter > checkpoint
            if os.path.exists(merged_model_dir) and os.path.exists(os.path.join(merged_model_dir, "config.json")):
                model_to_load = merged_model_dir
                log_info(f"Using existing merged model from: {merged_model_dir}")
            elif os.path.exists(base_model_dir) and os.path.exists(os.path.join(base_model_dir, "config.json")):
                model_to_load = base_model_dir
                log_info(f"Using existing base model from: {base_model_dir}")
            elif os.path.exists(lora_adapter_dir) and os.path.exists(os.path.join(lora_adapter_dir, "adapter_config.json")):
                model_to_load = lora_adapter_dir
                log_info(f"Using existing LoRA adapter from: {lora_adapter_dir}")
            elif checkpoint_dirs:
                # Use the latest checkpoint (highest number)
                latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)
                model_to_load = latest_checkpoint
                log_info(f"Using existing checkpoint from: {latest_checkpoint}")
            
            if model_to_load:
                # Load the existing model
                log_info(f"Loading existing model from: {model_to_load}")
                try:
                    from unsloth import FastLanguageModel
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_to_load,
                        max_seq_length=self.max_seq_length,
                        dtype=self.dtype,
                        load_in_4bit=False,  # Load existing model without quantization
                        token=os.getenv("HF_TOKEN")
                    )
                    log_info("Existing model loaded successfully")
                except Exception as e:
                    log_error(f"Failed to load existing model: {e}")
                    # Fallback to base model
                    log_info("Falling back to base model...")
                    self.load_model()
            else:
                log_info("No existing model found, loading base model...")
                self.load_model()
            
            # Skip to post-training steps
            log_info("Proceeding to post-training steps...")
            
            # Test inference with existing model
            log_info("Testing inference with existing model...")
            self.test_inference()
            
            # If save_dir is different from model_dir, save the model there
            if save_dir != model_dir:
                log_info(f"Saving model to: {save_dir}")
                self.save_model(save_dir, push_to_hub, hub_name)
            else:
                log_info("Model already in target location, skipping save step")
            
            # Convert to GGUF if requested
            if convert_gguf:
                log_info("Converting existing model to GGUF...")
                self.convert_to_gguf(save_dir)
            
            log_info("Pipeline completed successfully with existing model!")
            return True
            
        elif not data_load_result:
            # Data loading failed
            log_info("Training pipeline stopped (data loading failed)")
            return False
        
        # Continue with normal training pipeline
        # Load model
        log_info("Step 3: Loading model...")
        self.load_model()
        
        # Test inference before training
        log_info("Step 4: Testing inference before training...")
        self.test_inference()
        
        # Prepare dataset
        log_info("Step 5: Preparing dataset...")
        self.prepare_dataset()
        
        # Setup LoRA
        log_info("Step 6: Setting up LoRA...")
        self.setup_lora()
        
        # Train
        log_info("Step 7: Training model...")
        self.train(output_dir)
        
        # Test after training
        log_info("Step 8: Testing inference after training...")
        self.test_after_training()
        
        # Save model
        log_info("Step 9: Saving model...")
        self.save_model(save_dir, push_to_hub, hub_name)
        
        # Convert to GGUF if requested
        if convert_gguf:
            log_info("Step 10: Converting to GGUF...")
            self.convert_to_gguf(save_dir)
        
        log_info("Training pipeline completed successfully!")
        return True
