# 🌿 Herbal Assistant - Local LLM Fine-tuning

This project fine-tunes a small local language model (like Phi-2 or LLaMA-7B) to answer questions using **only** the knowledge from *The Complete Herbal* by Nicholas Culpeper. The result is an offline herbal medicine chatbot that won't hallucinate general knowledge.

## 🎯 What This Does

- **Converts** the Project Gutenberg HTML of *The Complete Herbal* into structured text
- **Extracts** individual herb entries with descriptions, medicinal properties, and uses
- **Generates** thousands of Q&A pairs for training
- **Fine-tunes** a small local model using LoRA (Low-Rank Adaptation)
- **Creates** an offline herbal assistant that only knows what's in the book

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Process the Herbal Text

```bash
python herbal_preprocessor.py
```

This creates:
- `herbs_structured.json` - Individual herb entries
- `herbs_chunked.json` - Text chunks for training

### 3. Generate Training Data

```bash
python qa_generator.py
```

This creates:
- `herbal_qa_training.jsonl` - Q&A pairs for fine-tuning

### 4. Fine-tune the Model

```bash
# Use Phi-2 (smaller, faster)
python herbal_finetune.py --model microsoft/phi-2

# Or use LLaMA-7B (larger, potentially better)
python herbal_finetune.py --model meta-llama/Llama-2-7b-hf
```

### 5. Test Your Herbal Assistant

```bash
# Interactive chat
python herbal_inference.py --interactive

# Single question
python herbal_inference.py --question "What is rosemary used for?"
```

## 📁 File Structure

```
herbal-assistant/
├── herbal.txt                           # Raw herbal text
├── herbal_preprocessor.py              # Text processing and chunking
├── qa_generator.py                     # Q&A pair generation
├── herbal_finetune.py                  # Model fine-tuning script
├── herbal_inference.py                 # Inference and chat interface
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── herbal-lora-model/                  # Fine-tuned model (created after training)
```

## 🔧 How It Works

### Text Processing
1. **Extracts** herb entries using regex patterns
2. **Cleans** and normalizes the text
3. **Chunks** long entries into manageable pieces (~1000 words)

### Training Data Generation
1. **Herb-specific Q&A**: "What is [herb] used for?", "How do you identify [herb]?"
2. **General herbal Q&A**: "What is The Complete Herbal?", "Who wrote it?"
3. **Disease-specific Q&A**: "What herbs help with headaches?"
4. **Instruction following**: Teaches the model to stay within scope

### Fine-tuning
1. **Uses LoRA** to efficiently fine-tune with minimal parameters
2. **Instruction format**: "### Instruction: [question] ### Response: [answer]"
3. **Small batch sizes** for memory efficiency
4. **Saves** the fine-tuned model locally

## 💻 System Requirements

- **GPU**: 8GB+ VRAM recommended (can work with CPU but very slow)
- **RAM**: 16GB+ system RAM
- **Storage**: 5GB+ free space for models and data
- **Python**: 3.8+

## 🎛️ Customization

### Change Base Model
```bash
# Use a different model
python herbal_finetune.py --model microsoft/DialoGPT-medium
python herbal_finetune.py --model EleutherAI/pythia-1.4b
```

### Adjust Training Parameters
Edit `herbal_finetune.py` to modify:
- Learning rate
- Number of epochs
- Batch size
- LoRA rank

### Modify Q&A Generation
Edit `qa_generator.py` to:
- Add new question types
- Change answer formats
- Include more herbal knowledge

## 🧪 Example Output

After fine-tuning, your model will answer like this:

**Q**: "What is rosemary used for?"
**A**: "According to The Complete Herbal, rosemary is used for various medicinal purposes. It is good for the head, brain, memory, and for cleansing the blood. It helps with digestive issues and can be used to treat headaches and improve memory."

**Q**: "What is the capital of France?"
**A**: "I can only answer questions about herbs and herbal medicine from The Complete Herbal. I cannot provide information about geography or other topics outside the scope of this herbal text."

## 🚨 Important Notes

1. **Medical Disclaimer**: This is for educational purposes only. Always consult healthcare professionals.
2. **Historical Text**: The herbal information is from the 1600s and may not reflect modern medical knowledge.
3. **Offline Use**: The fine-tuned model works completely offline once trained.
4. **Memory Usage**: Larger models require more VRAM. Start with Phi-2 if you have limited resources.

## 🔍 Troubleshooting

### Out of Memory Errors
- Reduce batch size in `herbal_finetune.py`
- Use a smaller base model (Phi-2 instead of LLaMA-7B)
- Enable gradient checkpointing

### Poor Quality Responses
- Increase training epochs
- Adjust learning rate
- Check training data quality
- Verify LoRA configuration

### Model Not Loading
- Ensure all dependencies are installed
- Check model path
- Verify CUDA/GPU compatibility

## 📚 Further Reading

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [The Complete Herbal on Project Gutenberg](https://www.gutenberg.org/ebooks/49513)

## 🤝 Contributing

Feel free to:
- Improve the text processing
- Add more Q&A generation patterns
- Optimize the training process
- Create better inference interfaces

## 📄 License

This project is for educational purposes. The herbal text is from Project Gutenberg (public domain).

---

**Happy herbal learning! 🌿✨**
