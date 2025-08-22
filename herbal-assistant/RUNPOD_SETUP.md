# 🚀 RunPod GPU Training Setup for Herbal Assistant

This guide will help you set up and run the herbal fine-tuning project on RunPod Cloud GPU for fast, efficient training.

## 🎯 What You'll Get

- **Fast GPU training** (hours instead of days)
- **Pre-configured environment** with all dependencies
- **Automatic setup** and training start
- **Downloadable trained model** for local use

## 📋 Prerequisites

- RunPod account with GPU credits
- Your `herbal_qa_training.jsonl` file (already generated locally)
- Basic familiarity with RunPod interface

## 🔧 Step-by-Step Setup

### Step 1: Launch RunPod Container

1. **Go to [RunPod.io](https://runpod.io)**
2. **Click "Deploy" → "Community Cloud"**
3. **Select GPU Type:**
   - **Recommended**: A100 (40GB) - Fastest training
   - **Budget option**: T4 or L4 - Slower but cheaper
   - **Avoid**: CPU-only instances (too slow)

4. **Container Settings:**
   - **Image**: `nvidia/cuda:12.1-devel-ubuntu22.04`
   - **Port**: `8888` (for Jupyter if needed)
   - **Volume**: `50GB` (for model storage)

5. **Click "Deploy"**

### Step 2: Upload Your Training Data

Once the container is running:

1. **Open the container** (click on it)
2. **Go to "Files" tab**
3. **Upload these files:**
   - `herbal_qa_training.jsonl` ← **REQUIRED**
   - `herbs_structured.json` ← Optional
   - `herbs_chunked.json` ← Optional

4. **Or use the terminal:**
   ```bash
   # If you have the files locally, you can copy them
   # or clone your repo if it's on GitHub
   ```

### Step 3: Install the Herbal Training Environment

In the RunPod terminal:

```bash
# Clone or upload the project files
git clone https://github.com/your-username/herbal-assistant.git
cd herbal-assistant

# Or if you uploaded files manually, just navigate to the directory
cd /workspace

# Make the setup script executable
chmod +x runpod_setup.sh

# Run the automated setup
./runpod_setup.sh
```

### Step 4: Start Training

The setup script will automatically:

1. ✅ Check GPU availability
2. ✅ Verify training data
3. ✅ Install dependencies
4. ✅ Start fine-tuning with Phi-2
5. ✅ Test the model after training
6. ✅ Provide download instructions

**Expected Training Time:**
- **A100 (40GB)**: 2-4 hours
- **T4/L4**: 6-12 hours
- **CPU**: 2-3 days (not recommended)

## 📊 Monitoring Training

### Check Progress
```bash
# View training logs
tail -f herbal-lora-model/trainer_state.json

# Check GPU usage
nvidia-smi

# Monitor disk space
df -h
```

### Training Metrics
The trainer will show:
- Loss per step
- Learning rate
- GPU memory usage
- Estimated time remaining

## 💾 Downloading Your Trained Model

After training completes:

```bash
# Compress the model
tar -czvf herbal-lora-model.tar.gz herbal-lora-model/

# The compressed file will appear in your RunPod file manager
# Download it to your local machine
```

## 🧪 Testing Your Model

### On RunPod (immediate testing)
```bash
# Interactive chat
python herbal_inference.py --interactive

# Single question
python herbal_inference.py --question "What is chamomile used for?"
```

### On Local Machine (after download)
```bash
# Extract the model
tar -xzvf herbal-lora-model.tar.gz

# Test locally
python herbal_inference.py --model ./herbal-lora-model --interactive
```

## 🔍 Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce batch size in `herbal_finetune.py`
- Use smaller model (Phi-2 instead of LLaMA-7B)
- Close other GPU processes

**"Training data not found"**
- Check file path: `ls -la herbal_qa_training.jsonl`
- Re-upload the file
- Verify file permissions

**"Model loading failed"**
- Check internet connection
- Verify model name: `microsoft/phi-2`
- Try different model if issues persist

### Performance Tips

1. **Use A100 or H100** for fastest training
2. **Close unnecessary processes** to free GPU memory
3. **Monitor GPU temperature** - keep below 80°C
4. **Use SSD storage** for faster data loading

## 💰 Cost Estimation

**RunPod GPU Pricing (approximate):**
- **T4 (16GB)**: $0.20-0.30/hour
- **L4 (24GB)**: $0.30-0.50/hour  
- **A100 (40GB)**: $1.00-1.50/hour
- **H100 (80GB)**: $2.00-3.00/hour

**Total Cost for Herbal Training:**
- **T4**: $2-6 (6-12 hours)
- **A100**: $2-6 (2-4 hours)
- **H100**: $4-12 (2-4 hours)

## 🎉 Success Indicators

Your training is successful when:

1. ✅ Training completes without errors
2. ✅ `herbal-lora-model/` directory is created
3. ✅ Model answers herbal questions correctly
4. ✅ Model declines non-herbal questions
5. ✅ Training loss decreases over time

## 📚 Next Steps

After successful training:

1. **Download your model** to local machine
2. **Test thoroughly** with various herbal questions
3. **Deploy locally** or on cloud services
4. **Share results** with the community!

## 🆘 Need Help?

- **RunPod Issues**: Check RunPod documentation
- **Training Issues**: Review error logs and this guide
- **Model Issues**: Verify training data quality
- **Performance Issues**: Check GPU specifications

---

**Happy GPU Training! 🚀🌿**

Your herbal assistant will be ready in hours instead of days!
