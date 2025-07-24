#!/usr/bin/env python3
"""
Prepare Training Data Utility

This script processes JSON workflow files from n8n and creates training data
by generating summaries using Ollama via the OpenAI API client.
"""
import argparse
import json
import os
import re
import sys
import logging
import requests
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def clean_filename(s):
    """
    Remove numbers, replace underscores with spaces, and remove .json from the end of a string.
    """
    s = re.sub(r'\d+', '', s)           # Remove all digits
    s = re.sub(r'_+', ' ', s)            # Replace underscores with spaces
    s = re.sub(r'\.json$', '', s)       # Remove .json from the end
    return s.strip()

def initialize_ollama_model(model_name: str):
    """
    Validate that Ollama is accessible with the specified model.
    
    Args:
        model_name: The name of the Ollama model to use
        
    Returns:
        The model name if successful
        
    Raises:
        Exception: If Ollama is not accessible or the model is not available
    """
    try:
        # Check if Ollama is running by making a simple request
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            raise Exception(f"Ollama server returned status code {response.status_code}")
            
        # Check if the model is available
        models = response.json().get("models", [])
        available_models = [model["name"] for model in models]
        
        if model_name not in available_models:
            logger.warning(f"Model {model_name} not found in available models: {available_models}")
            logger.info(f"Will attempt to use {model_name} anyway, Ollama will pull it if needed")
            
        return model_name
    except requests.RequestException as e:
        raise Exception(f"Error connecting to Ollama: {str(e)}")

def summarize_with_ollama(model_name: str, content: str) -> str:
    """
    Generate a summary of n8n workflow JSON using Ollama.
    
    Args:
        model_name: The name of the Ollama model to use
        content: The JSON content to summarize
    
    Returns:
        A string containing the summary of the workflow
    
    Raises:
        Exception: If there's an error communicating with Ollama
    """
    try:
          # Create the prompt for summarization
        prompt = f"Create task definition with purpose for the following n8n workflow JSON in 50 words or less. Focus on what the workflow does and its key components:\n\n{content}. Return no additional text and do not exceed 50 words in response."

        # Call Ollama API directly
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            return f"Error: Ollama returned status code {response.status_code}"
            
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        logger.error(f"Error generating summary with Ollama: {str(e)}")
        return f"Error summarizing workflow: {str(e)}"

def prepare_n8n_training_data(folder_path: str, model_name: str, output_path: str) -> bool:
    """
    Prepare n8n training data by processing JSON workflow files.
    
    Args:
        folder_path: Path to the folder containing JSON workflow files
        model_name: Name of the Ollama model to use
        output_path: Path where the output JSON file will be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    output_data = []
    
    # Validate the input folder exists
    if not os.path.exists(folder_path):
        logger.error(f"Folder path does not exist: {folder_path}")
        return False
        
    # Count total JSON files for progress reporting
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    total_files = len(json_files)
    
    if total_files == 0:
        logger.warning(f"No JSON files found in folder: {folder_path}")
        return False
        
    logger.info(f"Found {total_files} JSON files in {folder_path}")
    
    # Initialize Ollama model once
    try:
        logger.info(f"Initializing Ollama model: {model_name}")
        llm_model = initialize_ollama_model(model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Ollama model: {str(e)}")
        return False
    
    # Process each JSON file
    for i, filename in enumerate(json_files):
        filepath = os.path.join(folder_path, filename)
        logger.info(f"Processing file {i+1}/{total_files}: {filename}")
        
        try:
            # Read the JSON file
            with open(filepath, "r") as f:
                file_content = f.read()
                
            # Validate JSON structure
            try:
                json.loads(file_content)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in {filename}: {str(e)}")
                continue
            
            # Remove all digits from filename
            instructions = clean_filename(filename)
        
        
            # Generate summary using Ollama    
            summary = summarize_with_ollama(llm_model, file_content)
            logger.info(f"{instructions}: {summary}")

            # Add to output data
            output_data.append({
                "instruction": instructions,
                "input": summary,
                "output": json.dumps(json.loads(file_content), ensure_ascii=False)
            })
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Write output data to file
    try:
        with open(output_path, "w") as out_f:
            json.dump(output_data, out_f, indent=2)
        logger.info(f"Successfully wrote training data for {len(output_data)} files to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing output file: {str(e)}")
        return False

def main() -> None:
    """
    Main function to parse arguments and call the appropriate handler.
    """
    parser = argparse.ArgumentParser(
        description="Prepare training data for n8n workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--type", required=True, help="Type of training data (e.g., n8n)")
    parser.add_argument("--folder", required=True, help="Folder containing n8n workflow JSON files")
    parser.add_argument("--model", required=True, help="Ollama model name (e.g., llama3)")
    parser.add_argument("--output", default="training/n8n-training-data.json", 
                       help="Output file path for training data")
    args = parser.parse_args()

    # Call the appropriate handler based on the type
    if args.type.lower() == "n8n":
        success = prepare_n8n_training_data(args.folder, args.model, args.output)
        if not success:
            sys.exit(1)
    else:
        logger.error(f"Unsupported training data type: {args.type}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
