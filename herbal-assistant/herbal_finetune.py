#!/usr/bin/env python3
"""
Herbal Model Fine-tuning Script
Fine-tunes a small local model (Phi-2 or LLaMA-7B) on The Complete Herbal data
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import argparse

class HerbalFineTuner:
    def __init__(self, model_name: str, training_data: str):
        self.model_name = model_name
        self.training_data = training_data
        self.tokenizer = None
        self.model = None
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully")
        print(f"Model parameters: {self.model.num_parameters():,}")
    
    def load_training_data(self) -> Dataset:
        """Load and format the training data"""
        print(f"Loading training data from: {self.training_data}")
        
        # Load JSONL file
        data = []
        with open(self.training_data, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        print(f"Loaded {len(data)} training examples")
        
        # Format data for instruction tuning
        formatted_data = []
        for item in data:
            # Create instruction format
            if item['input']:
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            
            formatted_data.append({
                "text": prompt,
                "instruction": item['instruction'],
                "input": item['input'],
                "output": item['output']
            })
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize the training examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def apply_lora(self):
        """Apply LoRA configuration to the model"""
        print("Applying LoRA configuration...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Rank
            lora_alpha=32,  # Alpha scaling
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def setup_training(self, dataset: Dataset):
        """Setup training arguments and trainer"""
        print("Setting up training...")
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./herbal-lora-model",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset.select(range(min(100, len(tokenized_dataset)))),  # Small eval set
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        return trainer
    
    def train(self):
        """Main training function"""
        print("Starting herbal model fine-tuning...")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load training data
        dataset = self.load_training_data()
        
        # Apply LoRA
        self.apply_lora()
        
        # Setup training
        trainer = self.setup_training(dataset)
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        print("Saving fine-tuned model...")
        trainer.save_model()
        self.tokenizer.save_pretrained("./herbal-lora-model")
        
        print("Training complete! Model saved to ./herbal-lora-model")
    
    def test_model(self, test_questions: list):
        """Test the fine-tuned model with some questions"""
        if not os.path.exists("./herbal-lora-model"):
            print("No fine-tuned model found. Please run training first.")
            return
        
        print("Loading fine-tuned model for testing...")
        
        # Load the fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            "./herbal-lora-model",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained("./herbal-lora-model")
        
        print("Testing model with sample questions...")
        
        for question in test_questions:
            prompt = f"### Instruction:\n{question}\n\n### Response:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            print(f"\nQ: {question}")
            print(f"A: {response}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model on herbal data")
    parser.add_argument("--model", default="microsoft/phi-2", help="Base model to fine-tune")
    parser.add_argument("--data", default="herbal_qa_training.jsonl", help="Training data file")
    parser.add_argument("--test", action="store_true", help="Test the model after training")
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = HerbalFineTuner(args.model, args.data)
    
    # Train the model
    fine_tuner.train()
    
    # Test if requested
    if args.test:
        test_questions = [
            "What is rosemary used for?",
            "How do you identify chamomile?",
            "What herbs can help with headaches?",
            "Who wrote The Complete Herbal?"
        ]
        fine_tuner.test_model(test_questions)

if __name__ == "__main__":
    main()
