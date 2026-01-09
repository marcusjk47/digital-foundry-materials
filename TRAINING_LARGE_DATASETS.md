# Training on Large Datasets (1000+ Materials)

Streamlit Cloud has execution time limits (~10-15 minutes) that can cause timeouts when training on large datasets. This guide explains your options for handling large-scale training.

## 🎯 Quick Reference

| Dataset Size | Recommended Approach | Training Time | Success Rate |
|--------------|---------------------|---------------|--------------|
| < 500 materials | Streamlit (normal) | 5-10 min | ✅ High |
| 500-800 materials | Streamlit (normal) | 10-15 min | ⚠️ Medium |
| 800-1500 materials | Streamlit (subsample) or CLI | 5-10 min (sub) | ✅ High (sub) |
| 1500+ materials | CLI Script (recommended) | 15-60 min | ✅ High |

---

## Option 1: Quick Train (Subsampling) 🎯

**When to use:** Rapid prototyping, testing model architecture, proof-of-concept

**How it works:**
- Randomly samples a subset of materials (default: 500)
- Maintains dataset diversity with stratified sampling
- Trains on Streamlit Cloud without timeout
- Perfect for experimentation

**Steps:**
1. Go to **"🎓 Train Model"** tab
2. Select your large dataset
3. Choose **"🎯 Quick Train (Subsample)"**
4. Adjust subsample size (100-1000)
5. Click **"🚀 Start Training"**

**Pros:**
- ✅ Fast training (5-10 minutes)
- ✅ No timeout issues
- ✅ Good for testing hyperparameters
- ✅ Works on Streamlit Cloud

**Cons:**
- ❌ Lower accuracy than full dataset
- ❌ May miss rare material types
- ❌ Not suitable for production models

---

## Option 2: CLI Script (Recommended for Production) 💻

**When to use:** Training production models, large datasets (1000+), when you need best accuracy

**How it works:**
- Download standalone Python script
- Run locally on your computer or cloud instance
- No timeout limits
- GPU support for faster training
- Full control over training process

### Setup

1. **Download the script:**
   - In Streamlit app, go to **"🎓 Train Model"** tab
   - Select your large dataset
   - Choose **"💻 Download CLI Script"**
   - Click **"📥 Download train_large_dataset.py"**

2. **Install dependencies** (if not already installed):
   ```bash
   pip install torch torch-geometric pymatgen mp-api pycalphad
   ```

### Basic Usage

```bash
# Train on a dataset
python train_large_dataset.py --dataset datasets/my_dataset.pkl --epochs 150

# Specify custom parameters
python train_large_dataset.py \
    --dataset datasets/steel_alloys.pkl \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --patience 30

# Use GPU (if available)
python train_large_dataset.py --dataset datasets/my_dataset.pkl --device cuda
```

### All Options

```bash
python train_large_dataset.py --help
```

**Available options:**
- `--dataset`: Path to dataset file (required)
- `--dataset-name`: Custom name for the model
- `--epochs`: Number of training epochs (default: 150)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 20)
- `--hidden-dim`: Hidden dimension (default: 128)
- `--n-conv`: Number of conv layers (default: 3)
- `--n-hidden`: Number of hidden layers (default: 2)
- `--train-ratio`: Training set ratio (default: 0.8)
- `--val-ratio`: Validation set ratio (default: 0.1)
- `--device`: Device (auto/cpu/cuda/mps, default: auto)
- `--checkpoint-dir`: Checkpoint directory (default: checkpoints)

### GPU Training

**NVIDIA GPU (CUDA):**
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Train with GPU
python train_large_dataset.py --dataset datasets/my_dataset.pkl --device cuda
```

**Apple Silicon (M1/M2/M3):**
```bash
# Check if MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"

# Train with Apple GPU
python train_large_dataset.py --dataset datasets/my_dataset.pkl --device mps
```

### Upload Trained Model

After training completes:

1. Find your model in `checkpoints/` directory
2. In Streamlit app, go to **"🔮 Prediction"** mode
3. Use **"📤 Upload File"** tab
4. Upload your trained `.pt` file
5. Start making predictions!

**Pros:**
- ✅ No timeout limits
- ✅ GPU support (10-50x faster)
- ✅ Full dataset training
- ✅ Best accuracy
- ✅ Progress tracking
- ✅ Checkpoint resumption

**Cons:**
- ❌ Requires local Python environment
- ❌ Manual upload to Streamlit

---

## Option 3: Full Train on Streamlit (High Risk) ⚡

**When to use:** Only if dataset is 800-1200 materials and you want to try

**How it works:**
- Trains on full dataset in Streamlit Cloud
- May timeout if training takes >15 minutes
- Automatically saves progress before timeout

**Steps:**
1. Go to **"🎓 Train Model"** tab
2. Select your large dataset
3. Choose **"⚡ Full Train (May Timeout)"**
4. Click **"🚀 Start Training"**
5. Hope for the best! 🤞

**Pros:**
- ✅ No local setup needed
- ✅ Full dataset training
- ✅ Convenient

**Cons:**
- ❌ High timeout risk
- ❌ Wastes time if it fails
- ❌ No GPU acceleration

---

## Training Time Estimates

Based on dataset size and hardware:

### CPU Training (Streamlit Cloud)
| Dataset Size | Estimated Time | Timeout Risk |
|--------------|----------------|--------------|
| 100 materials | ~2 min | ✅ Safe |
| 300 materials | ~5 min | ✅ Safe |
| 500 materials | ~8 min | ✅ Safe |
| 800 materials | ~12 min | ⚠️ Risky |
| 1000 materials | ~15 min | ❌ High Risk |
| 1500 materials | ~22 min | ❌ Will Timeout |
| 2000 materials | ~30 min | ❌ Will Timeout |

### GPU Training (Local/Cloud)
| Dataset Size | Estimated Time (GPU) | Speedup |
|--------------|---------------------|---------|
| 500 materials | ~1 min | 8x faster |
| 1000 materials | ~2 min | 7.5x faster |
| 2000 materials | ~4 min | 7.5x faster |
| 5000 materials | ~10 min | 8x faster |

*Times are approximate and depend on model complexity, number of epochs, and hardware.*

---

## Best Practices

### For Experimentation (Quick Iterations)
1. Use **Quick Train (Subsample)** with 300-500 materials
2. Test different hyperparameters
3. Once satisfied, use CLI for full training

### For Production Models
1. Collect large dataset (1000+ materials)
2. Use **CLI Script** for training
3. Train on GPU if available
4. Upload final model to Streamlit
5. Share with team!

### For Maximum Accuracy
1. Collect 2000+ materials
2. Use CLI script with GPU
3. Train for 150-200 epochs
4. Use early stopping (patience: 30)
5. Evaluate on held-out test set

---

## Cloud Training Options

### Google Colab (Free GPU)

1. Upload dataset to Google Drive
2. Create Colab notebook:
   ```python
   # Install dependencies
   !pip install torch torch-geometric pymatgen mp-api pycalphad

   # Upload training script
   from google.colab import files
   uploaded = files.upload()  # Upload train_large_dataset.py

   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')

   # Train
   !python train_large_dataset.py \
       --dataset /content/drive/MyDrive/datasets/my_dataset.pkl \
       --device cuda \
       --epochs 150
   ```

3. Download trained model from Colab
4. Upload to Streamlit

### AWS / Azure / GCP

1. Launch compute instance with GPU
2. Install dependencies
3. Upload dataset and training script
4. Run training
5. Download model
6. Upload to Streamlit

---

## Troubleshooting

### "Training timeout on Streamlit Cloud"
**Solution:** Use CLI script or reduce dataset size with subsampling

### "Out of memory error"
**Solution:** Reduce `--batch-size` (try 16 or 8)

### "CUDA out of memory"
**Solution:** Reduce `--batch-size` or use a smaller model (`--hidden-dim 64`)

### "Training is too slow"
**Solution:**
- Use GPU if available
- Reduce `--epochs`
- Increase `--batch-size` (if memory allows)
- Use smaller model architecture

### "Model doesn't improve"
**Solution:**
- Check learning rate (try 0.0001 or 0.005)
- Increase patience
- Check data quality
- Try different model architecture

---

## FAQ

**Q: Can I resume training if it gets interrupted?**
A: Yes! The CLI script saves checkpoints. Use `--resume checkpoints/your_model.pt`

**Q: How do I know if my GPU is being used?**
A: The script will show "Device: cuda" or "Device: mps" at startup. Also check GPU utilization with `nvidia-smi` (NVIDIA) or Activity Monitor (Mac).

**Q: Can I train on multiple GPUs?**
A: Currently not supported. Single GPU training is recommended.

**Q: What's the maximum dataset size?**
A: Tested up to 10,000 materials. Larger datasets may require more memory.

**Q: Can I train overnight?**
A: Yes! The CLI script has no timeout limits. Use `nohup` on Linux/Mac:
```bash
nohup python train_large_dataset.py --dataset datasets/large.pkl --epochs 200 > training.log 2>&1 &
```

**Q: How do I choose the right subsample size?**
A: Start with 500. If training completes in <8 minutes, try 800. If it times out, reduce to 300.

---

## Summary

| Your Goal | Best Option | Steps |
|-----------|-------------|-------|
| Quick test | Subsample (500) | Streamlit → Quick Train |
| Production model | CLI Script + GPU | Download → Train locally → Upload |
| No GPU available | CLI Script (CPU) | Download → Train locally (slower) |
| Experimenting | Subsample (300-500) | Streamlit → Quick Train |
| Maximum accuracy | CLI Script + Large dataset | Download → Train on 2000+ materials |

**Recommended Workflow:**
1. **Prototype:** Quick Train with 500 materials → Test model → Iterate
2. **Production:** CLI Script with full dataset → Upload final model
3. **Deploy:** Use trained model in Streamlit for predictions

Happy training! 🚀
