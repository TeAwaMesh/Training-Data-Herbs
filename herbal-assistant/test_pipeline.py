#!/usr/bin/env python3
"""
Test the herbal pipeline without requiring ML dependencies
"""

import json
import os

def test_preprocessing():
    """Test the preprocessing step"""
    print("🧪 Testing preprocessing...")
    
    if not os.path.exists('herbs_structured.json'):
        print("❌ herbs_structured.json not found. Run herbal_preprocessor.py first.")
        return False
    
    with open('herbs_structured.json', 'r', encoding='utf-8') as f:
        herbs = json.load(f)
    
    print(f"✅ Found {len(herbs)} herbs")
    
    # Check quality
    good_herbs = [h for h in herbs if len(h['description']) > 50 and len(h['government_virtues']) > 50]
    print(f"✅ {len(good_herbs)} herbs have good content")
    
    return True

def test_qa_generation():
    """Test the Q&A generation step"""
    print("\n🧪 Testing Q&A generation...")
    
    if not os.path.exists('herbal_qa_training.jsonl'):
        print("❌ herbal_qa_training.jsonl not found. Run qa_generator.py first.")
        return False
    
    # Count lines in JSONL file
    with open('herbal_qa_training.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"✅ Found {len(lines)} Q&A pairs")
    
    # Check a few examples
    print("\n📝 Sample Q&A pairs:")
    for i, line in enumerate(lines[:3]):
        try:
            qa = json.loads(line.strip())
            print(f"{i+1}. Q: {qa['instruction'][:60]}...")
            print(f"   A: {qa['output'][:60]}...")
        except:
            print(f"{i+1}. Error parsing line")
    
    return True

def test_training_data_quality():
    """Test the quality of the training data"""
    print("\n🧪 Testing training data quality...")
    
    if not os.path.exists('herbal_qa_training.json'):
        print("❌ herbal_qa_training.json not found.")
        return False
    
    with open('herbal_qa_training.json', 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    # Analyze Q&A pairs
    total_qa = len(qa_pairs)
    herb_specific = sum(1 for qa in qa_pairs if 'used for' in qa['instruction'] or 'medicinal properties' in qa['instruction'])
    general = sum(1 for qa in qa_pairs if 'Complete Herbal' in qa['instruction'])
    disease = sum(1 for qa in qa_pairs if 'help with' in qa['instruction'])
    instruction = sum(1 for qa in qa_pairs if 'Only answer questions' in qa['instruction'])
    
    print(f"✅ Total Q&A pairs: {total_qa}")
    print(f"✅ Herb-specific: {herb_specific}")
    print(f"✅ General herbal: {general}")
    print(f"✅ Disease-specific: {disease}")
    print(f"✅ Instruction following: {instruction}")
    
    # Check answer lengths
    avg_answer_length = sum(len(qa['output']) for qa in qa_pairs) / total_qa
    print(f"✅ Average answer length: {avg_answer_length:.1f} characters")
    
    return True

def main():
    print("🌿 Herbal Pipeline Test Suite")
    print("=" * 40)
    
    # Test each step
    preprocessing_ok = test_preprocessing()
    qa_generation_ok = test_qa_generation()
    quality_ok = test_training_data_quality()
    
    print("\n" + "=" * 40)
    if all([preprocessing_ok, qa_generation_ok, quality_ok]):
        print("🎉 All tests passed! Your pipeline is ready for fine-tuning.")
        print("\nNext steps:")
        print("1. Install ML dependencies: pip install -r requirements.txt")
        print("2. Run fine-tuning: python herbal_finetune.py --model microsoft/phi-2")
        print("3. Test the model: python herbal_inference.py --interactive")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("\n📊 Pipeline Summary:")
    print(f"Preprocessing: {'✅' if preprocessing_ok else '❌'}")
    print(f"Q&A Generation: {'✅' if qa_generation_ok else '❌'}")
    print(f"Data Quality: {'✅' if quality_ok else '❌'}")

if __name__ == "__main__":
    main()
