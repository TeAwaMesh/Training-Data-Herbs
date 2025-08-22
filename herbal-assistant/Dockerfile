FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    software-properties-common \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML dependencies
RUN pip install \
    transformers>=4.50.0 \
    datasets>=2.12.0 \
    peft>=0.4.0 \
    accelerate>=0.20.0 \
    bitsandbytes>=0.39.0 \
    scipy \
    numpy \
    scikit-learn \
    tqdm \
    wandb \
    sentencepiece \
    protobuf \
    huggingface_hub

# Set working directory
WORKDIR /workspace

# Copy requirements (for reference)
COPY requirements.txt .

# Copy the herbal fine-tuning scripts
COPY herbal_preprocessor.py .
COPY qa_generator.py .
COPY herbal_finetune.py .
COPY herbal_inference.py .
COPY start_finetuning.py .
COPY run_pipeline.py .
COPY test_pipeline.py .
COPY README.md .

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "🌿 Herbal Fine-tuning Environment Ready!"\n\
echo "GPU Info:"\n\
nvidia-smi\n\
echo "\n🚀 To start training:"\n\
echo "1. Upload your herbal_qa_training.jsonl file"\n\
echo "2. Run: python herbal_finetune.py --model microsoft/phi-2"\n\
echo "\n💡 Or run the full pipeline: python run_pipeline.py"\n\
echo "\n📚 Check README.md for detailed instructions"\n\
bash' > /startup.sh && chmod +x /startup.sh

# Set the startup script as default command
CMD ["/startup.sh"]
