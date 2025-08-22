# ✅ RunPod Deployment Checklist

## 🚀 Before You Start

- [ ] **RunPod account** with GPU credits
- [ ] **Training data ready**: `herbal_qa_training.jsonl` (872 Q&A pairs)
- [ ] **Project files**: All Python scripts uploaded to RunPod

## 🔧 Container Setup

- [ ] **Launch RunPod container** with NVIDIA GPU
- [ ] **Select GPU type**: A100 (recommended) or T4/L4 (budget)
- [ ] **Set volume size**: 50GB minimum
- [ ] **Container running** and accessible

## 📁 File Upload

- [ ] **Upload training data**: `herbal_qa_training.jsonl`
- [ ] **Upload project files**: All Python scripts
- [ ] **Verify file permissions**: Files are readable

## 🚀 Quick Start

- [ ] **Make scripts executable**: `chmod +x *.sh`
- [ ] **Run quick start**: `./quick_start_runpod.sh`
- [ ] **Monitor training**: Watch GPU usage and logs
- [ ] **Wait for completion**: 2-12 hours depending on GPU

## 🧪 Testing

- [ ] **Training completed**: `herbal-lora-model/` directory exists
- [ ] **Test model**: Ask herbal questions
- [ ] **Verify responses**: Model answers from The Complete Herbal only
- [ ] **Check boundaries**: Model declines non-herbal questions

## 💾 Download

- [ ] **Compress model**: `tar -czvf herbal-lora-model.tar.gz herbal-lora-model/`
- [ ] **Download file**: Use RunPod file manager
- [ ] **Extract locally**: `tar -xzvf herbal-lora-model.tar.gz`
- [ ] **Test locally**: Run inference script

## 🎯 Success Criteria

- [ ] **Training completed** without errors
- [ ] **Model responds** to herbal questions
- [ ] **Model declines** non-herbal questions
- [ ] **Responses accurate** to source material
- [ ] **Model downloaded** and working locally

---

## 🆘 If Something Goes Wrong

1. **Check GPU memory**: `nvidia-smi`
2. **Check disk space**: `df -h`
3. **Check training logs**: Look for error messages
4. **Verify data**: Ensure `herbal_qa_training.jsonl` is valid
5. **Restart container**: Sometimes a fresh start helps

## 💡 Pro Tips

- **Use A100** for fastest training
- **Monitor GPU temperature** - keep below 80°C
- **Close unnecessary processes** to free memory
- **Save frequently** - RunPod containers can be terminated
- **Download model** as soon as training completes

---

**Ready to train your herbal assistant? 🚀🌿**

Follow this checklist and you'll have a working model in hours!
