#!/usr/bin/env python3
"""
Herbal Fine-tuning Pipeline Runner
Runs the complete pipeline from text processing to model training
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_file_exists(filename, description):
    """Check if a required file exists"""
    if not os.path.exists(filename):
        print(f"❌ {description} not found: {filename}")
        return False
    print(f"✅ {description} found: {filename}")
    return True

def main():
    print("🌿 Herbal Fine-tuning Pipeline")
    print("=" * 50)
    
    # Check if herbal.txt exists
    if not check_file_exists("herbal.txt", "Herbal text file"):
        print("\n❌ Please ensure herbal.txt exists before running the pipeline.")
        print("You can download it from Project Gutenberg or convert the HTML file.")
        return
    
    # Step 1: Process the herbal text
    if not run_command("python herbal_preprocessor.py", "Processing herbal text"):
        print("\n❌ Text processing failed. Please check the error above.")
        return
    
    # Check if processing created the required files
    if not check_file_exists("herbs_structured.json", "Structured herbs file"):
        print("\n❌ Text processing did not create required files.")
        return
    
    if not check_file_exists("herbs_chunked.json", "Chunked herbs file"):
        print("\n❌ Text processing did not create required files.")
        return
    
    # Step 2: Generate Q&A training data
    if not run_command("python qa_generator.py", "Generating Q&A training data"):
        print("\n❌ Q&A generation failed. Please check the error above.")
        return
    
    # Check if Q&A generation created the required file
    if not check_file_exists("herbal_qa_training.jsonl", "Q&A training data"):
        print("\n❌ Q&A generation did not create required files.")
        return
    
    # Step 3: Fine-tune the model
    print("\n🚀 Starting model fine-tuning...")
    print("This step may take several hours depending on your hardware.")
    print("You can monitor progress in the output above.")
    
    # Use Phi-2 by default (smaller, faster)
    model_name = "microsoft/phi-2"
    
    if not run_command(f"python herbal_finetune.py --model {model_name}", "Fine-tuning model"):
        print("\n❌ Model fine-tuning failed. Please check the error above.")
        print("Common issues:")
        print("- Insufficient GPU memory (try reducing batch size)")
        print("- Missing dependencies (run: pip install -r requirements.txt)")
        print("- CUDA/GPU compatibility issues")
        return
    
    # Check if fine-tuning created the model
    if not check_file_exists("./herbal-lora-model", "Fine-tuned model"):
        print("\n❌ Model fine-tuning did not create the expected output.")
        return
    
    # Step 4: Test the model
    print("\n🧪 Testing the fine-tuned model...")
    
    test_questions = [
        "What is rosemary used for?",
        "Who wrote The Complete Herbal?",
        "What herbs can help with headaches?"
    ]
    
    for question in test_questions:
        print(f"\nTesting: {question}")
        if not run_command(f'python herbal_inference.py --question "{question}"', f"Testing question: {question}"):
            print(f"❌ Failed to test question: {question}")
    
    # Final success message
    print("\n🎉 Pipeline completed successfully!")
    print("\nYour herbal assistant is ready to use!")
    print("\nTo start chatting:")
    print("python herbal_inference.py --interactive")
    print("\nTo ask a single question:")
    print('python herbal_inference.py --question "What is chamomile used for?"')
    
    print("\n📁 Files created:")
    print("- herbs_structured.json: Structured herb data")
    print("- herbs_chunked.json: Text chunks for training")
    print("- herbal_qa_training.jsonl: Q&A training data")
    print("- herbal-lora-model/: Your fine-tuned model")

if __name__ == "__main__":
    main()
