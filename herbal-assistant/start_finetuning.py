#!/usr/bin/env python3
"""
Start Fine-tuning Helper Script
Provides instructions and checks for starting the herbal model fine-tuning
"""

import os
import sys

def check_environment():
    """Check if the environment is ready for fine-tuning"""
    print("🔍 Checking environment for fine-tuning...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check if required files exist
    required_files = [
        'herbs_structured.json',
        'herbs_chunked.json', 
        'herbal_qa_training.jsonl'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found")
            return False
    
    return True

def check_dependencies():
    """Check if ML dependencies are installed"""
    print("\n🔍 Checking ML dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - training will be very slow on CPU")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    try:
        import peft
        print(f"✅ PEFT: {peft.__version__}")
    except ImportError:
        print("❌ PEFT not installed")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError:
        print("❌ Datasets not installed")
        return False
    
    return True

def show_training_options():
    """Show available training options"""
    print("\n🚀 Training Options:")
    print("=" * 50)
    
    models = [
        ("microsoft/phi-2", "2.7B parameters", "Fast, good for testing", "4-8 GB VRAM"),
        ("meta-llama/Llama-2-7b-hf", "7B parameters", "Better quality", "8-16 GB VRAM"),
        ("EleutherAI/pythia-1.4b", "1.4B parameters", "Very fast", "2-4 GB VRAM"),
        ("microsoft/DialoGPT-medium", "345M parameters", "Fastest", "2-4 GB VRAM")
    ]
    
    for i, (model, params, desc, vram) in enumerate(models, 1):
        print(f"{i}. {model}")
        print(f"   Parameters: {params}")
        print(f"   Description: {desc}")
        print(f"   VRAM needed: {vram}")
        print()

def main():
    print("🌿 Herbal Model Fine-tuning Starter")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment not ready. Please run the preprocessing steps first:")
        print("   python herbal_preprocessor.py")
        print("   python qa_generator.py")
        return
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ ML dependencies not installed.")
        print("\nTo install dependencies:")
        print("   pip install -r requirements.txt")
        print("\nOr install manually:")
        print("   pip install torch transformers datasets peft accelerate")
        return
    
    # Show training options
    show_training_options()
    
    print("🎯 Ready to start fine-tuning!")
    print("\nQuick start commands:")
    print("=" * 50)
    print("# For testing (fastest):")
    print("python herbal_finetune.py --model microsoft/phi-2")
    print()
    print("# For better quality (slower):")
    print("python herbal_finetune.py --model meta-llama/Llama-2-7b-hf")
    print()
    print("# Test after training:")
    print("python herbal_inference.py --interactive")
    
    print("\n💡 Tips:")
    print("- Start with Phi-2 if you have limited GPU memory")
    print("- Training may take 2-8 hours depending on your hardware")
    print("- Monitor GPU memory usage during training")
    print("- The model will be saved to ./herbal-lora-model/")
    
    print("\n🚨 Important:")
    print("- Always consult healthcare professionals for medical advice")
    print("- This is for educational purposes only")
    print("- The herbal information is from the 1600s")

if __name__ == "__main__":
    main()
