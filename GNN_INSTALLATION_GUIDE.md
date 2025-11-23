# 🧠 GNN Dependencies Installation Guide

**Goal:** Install PyTorch, PyTorch Geometric, and Pymatgen for Graph Neural Network functionality

---

## 📋 Prerequisites

- Python 3.9+ (you have 3.13.9 ✅)
- pip package manager
- Virtual environment activated (`mp-alloy-env`)

---

## 🚀 Installation Steps

### **Step 1: Activate Your Environment**

```bash
cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"

# Windows
mp-alloy-env\Scripts\activate

# Or if using Git Bash
source mp-alloy-env/Scripts/activate
```

You should see `(mp-alloy-env)` in your terminal prompt.

---

### **Step 2: Install PyTorch (CPU version)**

**For Windows CPU (recommended for Streamlit Cloud compatibility):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Why CPU version?**
- ✅ Smaller size (~200MB vs ~2GB for CUDA)
- ✅ Works on Streamlit Cloud free tier
- ✅ Sufficient for inference (predictions)
- ⚠️ Slower for training (train locally with GPU if needed)

**Alternative: If you have NVIDIA GPU and want faster training:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### **Step 3: Install PyTorch Geometric Dependencies**

**These packages must be installed in the correct order:**

```bash
# Install torch-scatter (required by torch-geometric)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install torch-sparse
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install torch-cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Finally, install torch-geometric
pip install torch-geometric
```

**Note:** If you get version errors, adjust the torch version in the URL (e.g., `torch-2.1.0+cpu`)

---

### **Step 4: Install Pymatgen**

```bash
pip install pymatgen
```

**Pymatgen** handles crystal structures (CIF files, atomic positions, lattices, etc.)

---

### **Step 5: Verify Installation**

Run the test script:

```bash
python test_gnn_setup.py
```

**Expected output:**

```
============================================================
Testing GNN Dependencies Installation
============================================================

1. Testing PyTorch...
   ✅ PyTorch version: 2.0.1
   💻 Using CPU (expected on Streamlit Cloud)

2. Testing PyTorch Geometric...
   ✅ PyTorch Geometric version: 2.4.0

3. Testing torch-scatter...
   ✅ torch-scatter imported successfully

4. Testing torch-sparse...
   ✅ torch-sparse imported successfully

5. Testing torch-cluster...
   ✅ torch-cluster imported successfully

6. Testing Pymatgen...
   ✅ Pymatgen version: 2024.1.0
   ✅ Can import Structure and Lattice

7. Testing Graph Creation...
   ✅ Created simple graph with 3 nodes and 6 edges

8. Testing Crystal Structure Creation...
   ✅ Created structure: Fe1
   ✅ Number of sites: 1

9. Testing GNN Layer...
   ✅ GCN layer forward pass successful
   ✅ Input shape: torch.Size([3, 16]), Output shape: torch.Size([3, 32])

============================================================
Summary
============================================================
✅ Tests passed: 9/9
❌ Tests failed: 0/9

🎉 All tests passed! GNN dependencies are ready!
```

---

## ⚠️ Troubleshooting

### **Problem 1: "No matching distribution found"**

**Solution:** Check your Python version
```bash
python --version  # Should be 3.9+
```

If too old, create new environment with Python 3.10:
```bash
python3.10 -m venv gnn-env
```

---

### **Problem 2: "torch-scatter installation failed"**

**Solution:** Install from pre-built wheels

```bash
# Check your PyTorch version first
python -c "import torch; print(torch.__version__)"

# Then use matching wheel URL
# For PyTorch 2.0: https://data.pyg.org/whl/torch-2.0.0+cpu.html
# For PyTorch 2.1: https://data.pyg.org/whl/torch-2.1.0+cpu.html

pip install torch-scatter -f https://data.pyg.org/whl/torch-YOUR_VERSION+cpu.html
```

---

### **Problem 3: "ImportError: DLL load failed"** (Windows)

**Solution:** Install Visual C++ Redistributable

Download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe

---

### **Problem 4: Installation takes too long**

**Solution:** Use conda (alternative package manager)

```bash
conda install pytorch==2.0.0 torchvision torchaudio cpuonly -c pytorch
conda install pyg -c pyg
conda install pymatgen -c conda-forge
```

---

### **Problem 5: "Out of memory" during installation**

**Solution:** Install packages one by one with `--no-cache-dir`

```bash
pip install --no-cache-dir torch
pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install --no-cache-dir torch-geometric
pip install --no-cache-dir pymatgen
```

---

## 📊 Package Sizes

Understanding what you're installing:

| Package | Size | Purpose |
|---------|------|---------|
| torch (CPU) | ~200 MB | Deep learning framework |
| torch-geometric | ~50 MB | Graph neural network library |
| torch-scatter | ~5 MB | Scatter operations for GNN |
| torch-sparse | ~10 MB | Sparse tensor operations |
| pymatgen | ~100 MB | Crystal structure handling |
| **Total** | **~365 MB** | Complete GNN stack |

**For comparison:**
- torch (CUDA): ~2 GB
- Streamlit Cloud free tier limit: ~1 GB total

✅ Our CPU setup fits comfortably!

---

## 🎯 Streamlit Cloud Considerations

### **Will it work on Streamlit Cloud?**

✅ **YES!** With CPU version of PyTorch

**However:**
- ⏱️ Inference will be slower (1-5 seconds per prediction)
- 💾 Takes longer to deploy (installing PyTorch)
- 🔄 Cold start time increases

**Optimizations for Streamlit Cloud:**

1. **Cache model loading:**
```python
@st.cache_resource
def load_gnn_model():
    model = CGCNN()
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    return model
```

2. **Use smaller models:**
- Fewer layers (3 instead of 5)
- Smaller hidden dimensions (64 instead of 128)

3. **Limit batch size:**
- Process structures one at a time
- Use pagination for batch predictions

---

## 📚 What's Next?

After successful installation:

1. ✅ **Test imports** - Run `test_gnn_setup.py`
2. 🔨 **Build graph constructor** - Convert structures to graphs
3. 🧠 **Implement GNN model** - CGCNN architecture
4. 🎨 **Create Streamlit UI** - User interface for predictions
5. 🚀 **Deploy to cloud** - Test on Streamlit Cloud

---

## 🔗 Useful Resources

**Documentation:**
- PyTorch: https://pytorch.org/docs/stable/index.html
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Pymatgen: https://pymatgen.org/

**Tutorials:**
- PyG Tutorial: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
- Graph Neural Networks: https://distill.pub/2021/gnn-intro/

**Pre-trained Models:**
- ALIGNN: https://github.com/usnistgov/alignn
- MEGNet: https://github.com/materialsvirtuallab/megnet

---

## 🆘 Need Help?

If you encounter issues:

1. Check the error message carefully
2. Google the specific error
3. Check PyTorch Geometric GitHub issues: https://github.com/pyg-team/pytorch_geometric/issues
4. Try the troubleshooting steps above

**Common gotchas:**
- PyTorch version mismatch with torch-scatter/torch-sparse
- Missing C++ compiler on Windows
- Incompatible Python version (need 3.9+)

---

## ✅ Installation Complete Checklist

- [ ] Environment activated
- [ ] PyTorch installed (CPU version)
- [ ] torch-geometric installed
- [ ] torch-scatter installed
- [ ] torch-sparse installed
- [ ] torch-cluster installed
- [ ] pymatgen installed
- [ ] `test_gnn_setup.py` runs successfully (9/9 tests pass)
- [ ] Ready to build GNN model!

---

**Once all checkboxes are complete, you're ready to implement the GNN! 🎉**
