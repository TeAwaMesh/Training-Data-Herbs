#!/bin/bash
# Quick Start Script for RunPod Herbal Training
# Run this immediately after launching your RunPod container

echo "🚀 Quick Start: Herbal Model Training on RunPod"
echo "================================================"

# Check if we're in the right environment
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ This script requires a GPU-enabled RunPod container"
    echo "Please launch a container with NVIDIA GPU support"
    exit 1
fi

# Show GPU info
echo "✅ GPU Environment Detected:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Check if training data exists
if [ ! -f "herbal_qa_training.jsonl" ]; then
    echo ""
    echo "📁 Training data not found!"
    echo ""
    echo "To get started:"
    echo "1. Upload your herbal_qa_training.jsonl file to this container"
    echo "2. Re-run this script: ./quick_start_runpod.sh"
    echo ""
    echo "💡 You can upload files using RunPod's file manager"
    echo "   or drag & drop them into the container"
    exit 1
fi

echo ""
echo "✅ Training data found!"
echo "📊 File size: $(du -h herbal_qa_training.jsonl | cut -f1)"
echo "📝 Examples: $(wc -l < herbal_qa_training.jsonl)"

# Check if dependencies are installed
if ! python -c "import torch, transformers, peft" 2>/dev/null; then
    echo ""
    echo "🔧 Installing dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets peft accelerate bitsandbytes
    echo "✅ Dependencies installed!"
fi

echo ""
echo "🚀 Starting training in 5 seconds..."
echo "Press Ctrl+C to cancel"
sleep 5

# Start training
echo "🌿 Beginning Herbal Model Fine-tuning..."
python herbal_finetune.py --model microsoft/phi-2

# Check result
if [ -d "herbal-lora-model" ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo ""
    echo "🧪 Testing your herbal assistant..."
    python herbal_inference.py --question "What is rosemary used for?"
    echo ""
    echo "💾 To download your model:"
    echo "tar -czvf herbal-lora-model.tar.gz herbal-lora-model/"
    echo "Then download from RunPod file manager"
else
    echo ""
    echo "❌ Training failed or was interrupted"
    echo "Check the logs above for details"
fi
