#!/usr/bin/env python3
"""
Herbal Text Preprocessor
Converts The Complete Herbal text into structured data for training
"""

import re
import json
from typing import List, Dict, Tuple
from pathlib import Path

class HerbalPreprocessor:
    def __init__(self, text_file: str):
        self.text_file = text_file
        self.herbs = []
        
    def load_text(self) -> str:
        """Load the herbal text file"""
        with open(self.text_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_herb_entries(self, text: str) -> List[Dict]:
        """Extract individual herb entries from the text"""
        # Split text into lines for processing
        lines = text.split('\n')
        herbs = []
        current_herb = None
        
        for i, line in enumerate(lines):
            # Check if this line starts a new herb entry (all caps followed by period)
            if re.match(r'^[A-Z][A-Z\s\']+\.$', line.strip()):
                # Save previous herb if exists
                if current_herb:
                    herbs.append(current_herb)
                
                # Start new herb
                herb_name = line.strip().rstrip('.')
                current_herb = {
                    'name': herb_name,
                    'description': '',
                    'place': '',
                    'time': '',
                    'government_virtues': '',
                    'full_text': line + '\n'
                }
            elif current_herb:
                # Continue building current herb
                current_herb['full_text'] += line + '\n'
                
                # Extract specific sections
                if line.startswith('Place.]'):
                    current_herb['place'] = line.replace('Place.]', '').strip()
                elif line.startswith('Time.]'):
                    current_herb['time'] = line.replace('Time.]', '').strip()
                elif line.startswith('Government and virtues.]'):
                    current_herb['government_virtues'] = line.replace('Government and virtues.]', '').strip()
                elif line.startswith('Descript.]'):
                    current_herb['description'] = line.replace('Descript.]', '').strip()
        
        # Add the last herb
        if current_herb:
            herbs.append(current_herb)
        
        return herbs
    
    def clean_herb_text(self, herb: Dict) -> Dict:
        """Clean and format herb text"""
        # Clean the full text
        text = herb['full_text']
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        # Extract the main description (everything after Descript.] until next section)
        desc_match = re.search(r'Descript\.](.*?)(?=Place\.]|Time\.]|Government and virtues\.]|$)', text, re.DOTALL)
        if desc_match:
            herb['description'] = desc_match.group(1).strip()
        
        # Extract virtues section
        virtues_match = re.search(r'Government and virtues\.](.*?)(?=WATER |$)', text, re.DOTALL)
        if virtues_match:
            herb['government_virtues'] = virtues_match.group(1).strip()
        
        herb['cleaned_text'] = text
        return herb
    
    def chunk_herb_text(self, herb: Dict, max_words: int = 1000, overlap: int = 100) -> List[Dict]:
        """Split herb text into chunks for training"""
        text = herb['cleaned_text']
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words - overlap):
            chunk_words = words[i:i + max_words]
            chunk_text = ' '.join(chunk_words)
            
            chunk = {
                'herb_name': herb['name'],
                'chunk_id': f"{herb['name']}_{i//(max_words-overlap)}",
                'text': chunk_text,
                'word_count': len(chunk_words),
                'start_word': i
            }
            chunks.append(chunk)
        
        return chunks
    
    def process(self) -> Tuple[List[Dict], List[Dict]]:
        """Main processing function"""
        print("Loading herbal text...")
        text = self.load_text()
        
        print("Extracting herb entries...")
        herbs = self.extract_herb_entries(text)
        print(f"Found {len(herbs)} herb entries")
        
        print("Cleaning herb text...")
        cleaned_herbs = [self.clean_herb_text(herb) for herb in herbs]
        
        print("Creating text chunks...")
        all_chunks = []
        for herb in cleaned_herbs:
            chunks = self.chunk_herb_text(herb)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} text chunks")
        
        return cleaned_herbs, all_chunks
    
    def save_herbs(self, herbs: List[Dict], output_file: str):
        """Save herbs to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(herbs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(herbs)} herbs to {output_file}")
    
    def save_chunks(self, chunks: List[Dict], output_file: str):
        """Save chunks to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunks)} chunks to {output_file}")

def main():
    preprocessor = HerbalPreprocessor('herbal.txt')
    herbs, chunks = preprocessor.process()
    
    # Save results
    preprocessor.save_herbs(herbs, 'herbs_structured.json')
    preprocessor.save_chunks(chunks, 'herbs_chunked.json')
    
    # Print some stats
    print(f"\nProcessing complete!")
    print(f"Total herbs: {len(herbs)}")
    print(f"Total chunks: {len(chunks)}")
    
    # Show example
    if herbs:
        print(f"\nExample herb: {herbs[0]['name']}")
        print(f"Description length: {len(herbs[0]['description'])} chars")
        print(f"Virtues length: {len(herbs[0]['government_virtues'])} chars")

if __name__ == "__main__":
    main()
