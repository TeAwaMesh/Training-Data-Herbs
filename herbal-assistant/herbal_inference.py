#!/usr/bin/env python3
"""
Herbal Model Inference Script
Simple script to use the fine-tuned herbal model for answering questions
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

class HerbalInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the fine-tuned herbal model"""
        print(f"Loading herbal model from: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
        return True
    
    def generate_response(self, question: str, max_length: int = 200) -> str:
        """Generate a response to a question"""
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded"
        
        # Format prompt
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.replace(prompt, "").strip()
        
        return response
    
    def interactive_chat(self):
        """Interactive chat mode"""
        print("\n🌿 Welcome to the Herbal Assistant!")
        print("Ask me anything about herbs from The Complete Herbal.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("🌿 You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("🌿 Goodbye! Stay healthy with herbs!")
                    break
                
                if not question:
                    continue
                
                print("🤖 Herbal Assistant: ", end="", flush=True)
                response = self.generate_response(question)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n🌿 Goodbye! Stay healthy with herbs!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Herbal Model Inference")
    parser.add_argument("--model", default="./herbal-lora-model", help="Path to fine-tuned model")
    parser.add_argument("--question", help="Single question to ask")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat mode")
    
    args = parser.parse_args()
    
    # Initialize inference
    herbal_bot = HerbalInference(args.model)
    
    # Load model
    if not herbal_bot.load_model():
        print("Failed to load model. Please check the model path.")
        return
    
    # Handle different modes
    if args.question:
        # Single question mode
        response = herbal_bot.generate_response(args.question)
        print(f"\nQ: {args.question}")
        print(f"A: {response}")
    elif args.interactive:
        # Interactive chat mode
        herbal_bot.interactive_chat()
    else:
        # Default to interactive mode
        herbal_bot.interactive_chat()

if __name__ == "__main__":
    main()
