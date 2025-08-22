#!/bin/bash
# RunPod Herbal Fine-tuning Setup Script
# This script sets up the environment and starts training on RunPod

echo "🌿 RunPod Herbal Fine-tuning Setup"
echo "=================================="

# Check if we're on RunPod
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "⚠️  No NVIDIA GPU detected - this script is designed for RunPod"
    exit 1
fi

# Check if training data exists
if [ ! -f "herbal_qa_training.jsonl" ]; then
    echo "❌ Training data not found: herbal_qa_training.jsonl"
    echo "Please upload your training data first."
    echo ""
    echo "📁 Files you need to upload:"
    echo "- herbal_qa_training.jsonl (required for training)"
    echo "- herbs_structured.json (optional, for inspection)"
    echo "- herbs_chunked.json (optional, for inspection)"
    echo ""
    echo "💡 You can upload files using RunPod's file manager or drag & drop."
    exit 1
fi

echo "✅ Training data found: herbal_qa_training.jsonl"

# Check file size
file_size=$(du -h herbal_qa_training.jsonl | cut -f1)
echo "📊 Training data size: $file_size"

# Count training examples
training_examples=$(wc -l < herbal_qa_training.jsonl)
echo "📝 Training examples: $training_examples"

# Check available disk space
available_space=$(df -h . | tail -1 | awk '{print $4}')
echo "💾 Available disk space: $available_space"

# Check if model directory already exists
if [ -d "herbal-lora-model" ]; then
    echo "⚠️  Found existing model directory: herbal-lora-model"
    echo "This suggests training was already started or completed."
    echo ""
    read -p "Do you want to continue with existing training? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "✅ Continuing with existing training..."
    else
        echo "🔄 Removing existing model directory..."
        rm -rf herbal-lora-model
        echo "✅ Clean slate ready for new training"
    fi
fi

echo ""
echo "🚀 Starting Herbal Model Fine-tuning..."
echo "Model: microsoft/phi-2"
echo "Training examples: $training_examples"
echo ""

# Start training
python herbal_finetune.py --model microsoft/phi-2

# Check if training completed successfully
if [ -d "herbal-lora-model" ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo ""
    echo "📁 Model saved to: herbal-lora-model/"
    echo ""
    echo "🧪 Testing the model..."
    echo "Q: What is rosemary used for?"
    python herbal_inference.py --question "What is rosemary used for?"
    echo ""
    echo "💾 To download your model:"
    echo "1. Compress: tar -czvf herbal-lora-model.tar.gz herbal-lora-model/"
    echo "2. Download from RunPod file manager"
    echo ""
    echo "🌿 Your herbal assistant is ready!"
else
    echo ""
    echo "❌ Training failed or was interrupted"
    echo "Check the logs above for error details"
    exit 1
fi
