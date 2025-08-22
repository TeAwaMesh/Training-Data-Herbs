#!/usr/bin/env python3
"""
Question-Answer Pair Generator for Herbal Training Data
Creates training examples from The Complete Herbal text
"""

import json
import random
from typing import List, Dict
import re

class HerbalQAGenerator:
    def __init__(self, herbs_file: str, chunks_file: str):
        self.herbs_file = herbs_file
        self.chunks_file = chunks_file
        self.herbs = []
        self.chunks = []
        
    def load_data(self):
        """Load the processed herbal data"""
        with open(self.herbs_file, 'r', encoding='utf-8') as f:
            self.herbs = json.load(f)
        
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"Loaded {len(self.herbs)} herbs and {len(self.chunks)} chunks")
    
    def generate_herb_specific_qa(self, herb: Dict) -> List[Dict]:
        """Generate Q&A pairs specific to a single herb"""
        qa_pairs = []
        herb_name = herb['name']
        
        # 1. What is [herb] used for?
        if herb['government_virtues']:
            qa_pairs.append({
                "instruction": f"What is {herb_name} used for?",
                "input": "",
                "output": f"{herb_name} is used for various medicinal purposes. According to The Complete Herbal: {herb['government_virtues'][:500]}..."
            })
        
        # 2. How do you identify [herb]?
        if herb['description']:
            qa_pairs.append({
                "instruction": f"How do you identify {herb_name}?",
                "input": "",
                "output": f"According to The Complete Herbal, {herb_name} can be identified by: {herb['description'][:500]}..."
            })
        
        # 3. Where does [herb] grow?
        if herb['place']:
            qa_pairs.append({
                "instruction": f"Where does {herb_name} grow?",
                "input": "",
                "output": f"{herb_name} grows {herb['place']}."
            })
        
        # 4. When does [herb] flower?
        if herb['time']:
            qa_pairs.append({
                "instruction": f"When does {herb_name} flower?",
                "input": "",
                "output": f"According to The Complete Herbal, {herb_name} {herb['time']}."
            })
        
        # 5. What are the medicinal properties of [herb]?
        if herb['government_virtues']:
            qa_pairs.append({
                "instruction": f"What are the medicinal properties of {herb_name}?",
                "input": "",
                "output": f"The Complete Herbal describes {herb_name} as having these medicinal properties: {herb['government_virtues'][:600]}..."
            })
        
        return qa_pairs
    
    def generate_general_herbal_qa(self) -> List[Dict]:
        """Generate general herbal medicine Q&A pairs"""
        qa_pairs = []
        
        # General questions about herbal medicine
        general_qa = [
            {
                "instruction": "What is The Complete Herbal?",
                "input": "",
                "output": "The Complete Herbal is a comprehensive guide to herbal medicine written by Nicholas Culpeper, M.D. It contains descriptions, medicinal properties, and uses of hundreds of herbs and plants for treating various ailments and disorders."
            },
            {
                "instruction": "Who wrote The Complete Herbal?",
                "input": "",
                "output": "The Complete Herbal was written by Nicholas Culpeper, M.D. It was first published in the 1600s and has been reprinted many times since."
            },
            {
                "instruction": "How should I use herbal remedies?",
                "input": "",
                "output": "According to The Complete Herbal, herbal remedies should be used according to the true system of nature. The book provides specific instructions for each herb, including preparation methods, dosages, and application techniques. Always consult with a qualified healthcare provider before using any herbal remedies."
            },
            {
                "instruction": "What are the main categories of herbs in The Complete Herbal?",
                "input": "",
                "output": "The Complete Herbal categorizes herbs by their medicinal and occult qualities, including herbs for the head, brain, memory, blood cleansing, liver health, digestive issues, respiratory problems, and many other bodily systems and disorders."
            }
        ]
        
        qa_pairs.extend(general_qa)
        return qa_pairs
    
    def generate_disease_specific_qa(self) -> List[Dict]:
        """Generate Q&A pairs about specific diseases and their herbal treatments"""
        qa_pairs = []
        
        # Common ailments and their herbal treatments
        disease_herbs = [
            ("headache", ["Rosemary", "Lavender", "Peppermint"]),
            ("digestive issues", ["Chamomile", "Peppermint", "Ginger"]),
            ("cough", ["Coltsfoot", "Marshmallow", "Horehound"]),
            ("wounds", ["Plantain", "Yarrow", "Comfrey"]),
            ("fever", ["Willow bark", "Elderflower", "Peppermint"]),
            ("insomnia", ["Valerian", "Chamomile", "Lavender"]),
            ("skin problems", ["Calendula", "Aloe", "Plantain"]),
            ("liver problems", ["Milk thistle", "Dandelion", "Burdock"])
        ]
        
        for disease, herbs in disease_herbs:
            # Find herbs that might help with this disease
            relevant_herbs = []
            for herb in self.herbs:
                if herb['government_virtues'] and any(herb_name.lower() in herb['government_virtues'].lower() for herb_name in herbs):
                    relevant_herbs.append(herb['name'])
            
            if relevant_herbs:
                qa_pairs.append({
                    "instruction": f"What herbs can help with {disease}?",
                    "input": "",
                    "output": f"According to The Complete Herbal, several herbs may help with {disease}: {', '.join(relevant_herbs[:3])}. However, always consult with a healthcare provider for proper diagnosis and treatment."
                })
        
        return qa_pairs
    
    def generate_instruction_following_qa(self) -> List[Dict]:
        """Generate Q&A pairs that test instruction following"""
        qa_pairs = []
        
        instructions = [
            {
                "instruction": "Only answer questions about herbs and herbal medicine from The Complete Herbal. If asked about something else, politely decline.",
                "input": "What is the capital of France?",
                "output": "I can only answer questions about herbs and herbal medicine from The Complete Herbal. I cannot provide information about geography or other topics outside the scope of this herbal text."
            },
            {
                "instruction": "Always cite The Complete Herbal as your source when providing herbal information.",
                "input": "What is rosemary good for?",
                "output": "According to The Complete Herbal, rosemary has various medicinal properties. However, I would need to look up the specific details about rosemary in the text to give you an accurate answer based on Culpeper's descriptions."
            },
            {
                "instruction": "If you don't know the answer to a question about herbs, say so rather than guessing.",
                "input": "What is the rarest herb mentioned in The Complete Herbal?",
                "output": "I don't have enough information from The Complete Herbal to determine which herb is the rarest. The text describes many herbs but doesn't specifically rank them by rarity."
            }
        ]
        
        qa_pairs.extend(instructions)
        return qa_pairs
    
    def generate_all_qa(self) -> List[Dict]:
        """Generate all types of Q&A pairs"""
        print("Generating herb-specific Q&A pairs...")
        herb_qa = []
        for herb in self.herbs:
            herb_qa.extend(self.generate_herb_specific_qa(herb))
        
        print("Generating general herbal Q&A pairs...")
        general_qa = self.generate_general_herbal_qa()
        
        print("Generating disease-specific Q&A pairs...")
        disease_qa = self.generate_disease_specific_qa()
        
        print("Generating instruction-following Q&A pairs...")
        instruction_qa = self.generate_instruction_following_qa()
        
        all_qa = herb_qa + general_qa + disease_qa + instruction_qa
        
        # Shuffle the data
        random.shuffle(all_qa)
        
        print(f"Generated {len(all_qa)} total Q&A pairs")
        return all_qa
    
    def save_qa_pairs(self, qa_pairs: List[Dict], output_file: str):
        """Save Q&A pairs to JSONL format for training"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(qa_pairs)} Q&A pairs to {output_file}")
    
    def save_json_format(self, qa_pairs: List[Dict], output_file: str):
        """Save Q&A pairs to JSON format for inspection"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(qa_pairs)} Q&A pairs to {output_file}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    generator = HerbalQAGenerator('herbs_structured.json', 'herbs_chunked.json')
    generator.load_data()
    
    # Generate all Q&A pairs
    qa_pairs = generator.generate_all_qa()
    
    # Save in different formats
    generator.save_qa_pairs(qa_pairs, 'herbal_qa_training.jsonl')
    generator.save_json_format(qa_pairs, 'herbal_qa_training.json')
    
    # Print some examples
    print("\nExample Q&A pairs:")
    for i, qa in enumerate(qa_pairs[:5]):
        print(f"\n{i+1}. Q: {qa['instruction']}")
        print(f"   A: {qa['output'][:100]}...")

if __name__ == "__main__":
    main()
