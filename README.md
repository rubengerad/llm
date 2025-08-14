# Universal LLM Trainer & Deployment Platform

A comprehensive, automated pipeline for downloading, training, and deploying specialized Large Language Models (LLMs) for any industry or use case. This platform transforms general-purpose models into domain-specific experts through intelligent data preparation, fine-tuning, and deployment automation.

## 🎯 What This Project Does

This is a **Universal Model Trainer** that enables you to:

### 📥 **Model Download & Management**
- **Automatic Model Discovery**: Download any model from Hugging Face Hub
- **Multi-Architecture Support**: TinyLlama, Gemma3, DeepSeek, Qwen, and custom models
- **Intelligent Model Selection**: Choose optimal models based on your hardware and requirements
- **Built-in Model Configurations**: Pre-configured settings for popular model families

### 🧠 **Specialized Training Pipeline**
- **QLoRA Fine-Tuning**: Efficient training with Quantized Low-Rank Adaptation
- **Industry-Specific Datasets**: Prepare training data for any domain (currently optimized for software development)
- **Automated Data Processing**: Convert raw data (JSON workflows, documentation, etc.) into training formats
- **Multi-Format Training Data**: Support for Alpaca, instruction-following, and custom formats

### 🚀 **Complete Deployment Automation**
- **GGUF Conversion**: Automatically convert trained models for Ollama deployment
- **Local Model Serving**: Integrated Ollama setup for immediate model testing
- **Performance Optimization**: Quantization and optimization for various hardware configurations
- **Model Versioning**: Track and manage different model iterations

## 🏗️ Core Architecture

### **Training Data Preparation** (`prepare_training_data.py`)
- **Intelligent Summarization**: Uses Ollama to create training instructions from raw data
- **JSON Workflow Processing**: Specialized for n8n workflows, extensible to other formats
- **Automated Quality Control**: Validates JSON structure and content quality
- **Scalable Processing**: Batch processing with progress tracking

### **Universal Model Trainer** (`model_trainer.py`)
- **Modular Design**: Easy to add new model architectures and training methods
- **Memory Optimization**: Efficient training on consumer GPUs (4GB+ VRAM)
- **Comprehensive Logging**: Detailed training metrics and error handling
- **Flexible Configuration**: Customizable training parameters for different scenarios

### **Deployment Pipeline** (`main.py`)
- **End-to-End Automation**: From raw data to deployed model in one command
- **Multi-Stage Validation**: Testing at each stage of the pipeline
- **Resource Management**: Intelligent GPU/CPU utilization
- **Error Recovery**: Robust error handling with detailed diagnostics

## 🎯 Industry Applications

While currently optimized for **software development** and **workflow automation**, this platform is designed to be universal:

### **Current Specializations**
- **n8n Workflow Generation**: Train models to create and understand automation workflows
- **Code Generation**: Specialized programming assistants
- **Documentation Analysis**: Extract and synthesize technical knowledge

### **Easily Extensible To**
- **Healthcare**: Medical record analysis, diagnostic assistance
- **Finance**: Risk assessment, regulatory compliance
- **Legal**: Contract analysis, legal research
- **Manufacturing**: Process optimization, quality control
- **Education**: Personalized tutoring, content generation
- **Any Domain**: Just prepare your training data!

## 🔄 The Universal Training Philosophy

### **Problem We Solve**
- **Outdated Knowledge**: Most open-source LLMs have training data 1-2 years behind
- **Generic Responses**: General models lack domain-specific expertise
- **High Computational Cost**: Large models require expensive infrastructure
- **Deployment Complexity**: Difficult to get models running in production

### **Our Solution**
- **Continuous Learning**: Regular retraining with latest domain data
- **Specialized Expertise**: Models trained specifically for your use case
- **Efficient Architecture**: Smaller, faster models that match larger model performance
- **One-Click Deployment**: Automated pipeline from training to production

## 🚀 Quick Start Guide

### 1. **Environment Setup**
```bash
# Create Python virtual environment
python3 -m venv llm
source llm/bin/activate

# Install CUDA (optional, for GPU acceleration)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# Install Python dependencies
pip install torch transformers unsloth requests huggingface_hub
pip install mistral-common python-dotenv trl datasets
```

### 2. **Build llama.cpp for Model Conversion**
```bash
cd llama.cpp
mkdir -p build && cd build
cmake .. -DLLAMA_CUDA=OFF -DCMAKE_BUILD_TYPE=Release  # Use CUDA=ON if you have NVIDIA GPU
make -j$(nproc)
```

### 3. **Prepare Your Training Data**
```bash
# For n8n workflows (example)
./src/prepare_training_data.py \
    --type=n8n \
    --folder=resources/n8n/workflows \
    --model=gemma3:latest \
    --output=training/alpaca/n8n-training-data.json

# For your custom data, modify the script or add new data types
```

### 4. **Train Your Specialized Model**
```bash
# Edit src/main.py to select your model and configure training
# Available models: tiny_llama, gemma3_base, deepseek_qwen, gemma3_qlora_full

python src/main.py
```

### 5. **Deploy with Ollama**
```bash
# Install Ollama
pip install ollama

# Create and run your model
ollama create my-specialized-model -f Modelfile
ollama run my-specialized-model
```

## 📁 Project Structure
```
llm/
├── src/                          # Core training modules
│   ├── main.py                   # Main training pipeline
│   ├── model_trainer.py          # Training implementation
│   ├── model_utils.py            # Utility functions
│   ├── prepare_training_data.py  # Data preparation pipeline
│   └── model_logger.py           # Logging system
├── training/                     # Training data and outputs
│   └── alpaca/                   # Alpaca format training data
├── resources/                    # Raw data sources
│   ├── n8n/workflows/           # n8n workflow JSON files
│   └── mcp/                     # Model Context Protocol data
├── llama.cpp/                   # Model conversion tools
├── outputs/                     # Training checkpoints
└── dist/                        # Final model distributions
```

## 🔧 Configuration & Customization

### **Model Selection**
In `src/main.py`, choose from pre-configured models:
```python
MODELS = {
    "tiny_llama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_seq_length": 2048,
        "load_in_4bit": True,
    },
    "gemma3_base": {
        "model_id": "google/gemma-3-1b-it",
        "max_seq_length": 2048,
        "load_in_4bit": False,
    },
    "deepseek_qwen": {
        "model_id": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
        "max_seq_length": 2048,
        "load_in_4bit": False,
    },
    # Add your custom model here
}

SELECTED_MODEL = "deepseek_qwen"  # Change this
```

### **Training Parameters**
Customize training for your hardware and requirements:
```python
# In main.py, adjust these based on your GPU memory
batch_size = 1          # Reduce for smaller GPUs
gradient_accumulation = 16  # Increase for effective larger batches
learning_rate = 1e-4    # Lower for stability
max_steps = 50          # Increase for more training
```

### **Add New Data Types**
Extend `prepare_training_data.py` to handle your domain-specific data:
```python
def prepare_medical_training_data(folder_path: str, model_name: str, output_path: str):
    # Your custom data processing logic
    pass

# Add to main() function
if args.type.lower() == "medical":
    success = prepare_medical_training_data(args.folder, args.model, args.output)
```

## 🎯 Training Workflow Deep Dive

### **Phase 1: Data Collection & Preparation**
1. **Raw Data Ingestion**: Place your domain-specific files in `resources/`
2. **Intelligent Processing**: The system analyzes and converts your data
3. **Quality Validation**: Automatic checks for data integrity and format
4. **Training Format Generation**: Creates instruction-response pairs for fine-tuning

### **Phase 2: Model Training**
1. **Model Download**: Automatically fetches the selected model from Hugging Face
2. **QLoRA Setup**: Configures efficient fine-tuning with quantization
3. **Training Execution**: Runs the training with progress monitoring
4. **Checkpoint Management**: Saves intermediate states for recovery

### **Phase 3: Model Optimization & Deployment**
1. **Model Merging**: Combines LoRA adapters with base model
2. **GGUF Conversion**: Optimizes for inference deployment
3. **Ollama Integration**: Sets up local model serving
4. **Performance Testing**: Validates model quality and speed

## 📊 Performance & Optimization

### **Hardware Requirements**
- **Minimum**: 8GB RAM, 4GB VRAM (with 4-bit quantization)
- **Recommended**: 16GB RAM, 8GB+ VRAM
- **Optimal**: 32GB RAM, 16GB+ VRAM

### **Memory Optimization Strategies**
- **4-bit Quantization**: Reduces memory usage by 75%
- **QLoRA**: Efficient fine-tuning with minimal memory overhead
- **Gradient Checkpointing**: Trades compute for memory
- **Mixed Precision**: Uses FP16/BF16 for faster training

## 🔮 Advanced Features

### **Continuous Learning Pipeline**
- **Scheduled Retraining**: Automatic model updates with new data
- **Performance Monitoring**: Track model degradation and improvement
- **A/B Testing**: Compare model versions in production
- **Incremental Learning**: Add new knowledge without full retraining

### **Multi-Model Management**
- **Model Ensemble**: Combine multiple specialized models
- **Version Control**: Git-like versioning for models
- **Resource Allocation**: Intelligent GPU scheduling for multiple training jobs
- **Distributed Training**: Scale across multiple machines

### **Production Integration**
- **API Deployment**: REST API for model serving
- **Model Context Protocol (MCP)**: Integration with various tools and systems
- **Monitoring & Analytics**: Real-time performance metrics
- **Auto-scaling**: Dynamic resource allocation based on load

## 🤝 Contributing & Extending

This platform is designed to be community-driven. Here's how you can contribute:

### **Add New Industries**
1. Create data preparation scripts for your domain
2. Add industry-specific model configurations
3. Contribute training datasets (where licensing permits)
4. Share performance benchmarks

### **Improve Training Methods**
1. Implement new fine-tuning techniques
2. Add support for new model architectures
3. Optimize for different hardware configurations
4. Enhance evaluation metrics

### **Extend Deployment Options**
1. Add support for cloud deployment platforms
2. Create integrations with MLOps tools
3. Implement new serving frameworks
4. Add monitoring and observability tools

## 📚 Example Use Cases

### **Software Development Assistant**
```bash
# Train on your codebase and documentation
./src/prepare_training_data.py --type=code --folder=your_project/ --model=deepseek_qwen
python src/main.py  # SELECTED_MODEL = "deepseek_qwen"
# Result: AI assistant that understands your specific codebase and patterns
```

### **Workflow Automation Expert**
```bash
# Train on n8n workflows
./src/prepare_training_data.py --type=n8n --folder=workflows/ --model=gemma3:latest
python src/main.py  # SELECTED_MODEL = "gemma3_qlora_full"
# Result: Model that can generate and explain automation workflows
```

### **Domain Expert Chatbot**
```bash
# Train on industry documentation
./src/prepare_training_data.py --type=docs --folder=industry_docs/ --model=tiny_llama
python src/main.py  # SELECTED_MODEL = "tiny_llama"
# Result: Specialized chatbot for your industry
```

## 🚨 Troubleshooting

### **Common Issues & Solutions**

**CUDA Errors**: Build llama.cpp without CUDA if you don't have NVIDIA GPU
```bash
cmake .. -DLLAMA_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
```

**Memory Issues**: Reduce batch size and enable 4-bit quantization
```python
batch_size = 1
load_in_4bit = True
```

**Slow Training**: Enable gradient checkpointing and mixed precision
```python
gradient_checkpointing = True
fp16 = True  # or bf16 = True
```

**Model Quality**: Increase training steps and data quality
```python
max_steps = 100  # More training
# Clean and validate your training data
```

## 📈 Future Roadmap

- **Multi-Modal Support**: Training with images, audio, and video
- **Federated Learning**: Collaborative training without data sharing
- **AutoML Integration**: Automatic hyperparameter optimization
- **Edge Deployment**: Optimize for mobile and IoT devices
- **Real-time Learning**: Continuous learning from user interactions

## 📄 License

MIT License - Feel free to use, modify, and distribute. See LICENSE file for details.

## 🙏 Acknowledgments

- **Unsloth**: Efficient fine-tuning framework
- **Hugging Face**: Model hub and transformers library
- **Ollama**: Local model serving platform
- **llama.cpp**: High-performance inference engine

---

**Ready to create your specialized AI?** Start with the Quick Start Guide above and join the community of developers building the next generation of domain-specific AI assistants!