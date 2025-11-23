# 🧠 GNN Property Predictor Implementation Plan

**Goal:** Replace the duplicate App 2 with a functional Graph Neural Network property predictor

**Timeline:** 3-7 days for MVP (Minimum Viable Product)

---

## 🎯 Phase 1: Setup & Dependencies (Day 1)

### **1.1 Install PyTorch Ecosystem**

**Add to `requirements.txt`:**
```python
# Deep Learning (PyTorch)
torch>=2.0.0
torch-geometric>=2.3.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0
pymatgen>=2024.0.0  # For crystal structure handling
```

**Challenge on Streamlit Cloud:**
- ⚠️ PyTorch is large (~1GB+) - may hit free tier limits
- ⚠️ No GPU on free tier - CPU inference only
- ✅ Can work, but will be slower

**Alternative for Streamlit Cloud:**
```python
# Lighter weight option
torch>=2.0.0,<2.1.0
torch-geometric>=2.3.0
pymatgen-core>=2024.0.0  # Lighter version
```

### **1.2 Test Installation Locally**
```bash
cd "C:\Users\marcu\OneDrive\Desktop\Digital Foundry\ML-Alloy-Project"

# Activate environment
source mp-alloy-env/Scripts/activate  # or Windows equivalent

# Install
pip install torch torch-geometric pymatgen

# Test
python -c "import torch; import torch_geometric; import pymatgen; print('Success!')"
```

---

## 🏗️ Phase 2: Data Pipeline (Day 2-3)

### **2.1 Materials Project Structure Data**

**Modify App 1 to download structures:**
```python
# In pages/1_Materials_Project_Explorer.py

# Add structure download option
download_structures = st.checkbox("Download crystal structures (for GNN training)")

if download_structures:
    # Get structure data
    structure = doc.structure  # Pymatgen Structure object
    mat_data['structure'] = structure.as_dict()
```

**Or create separate structure downloader:**
```python
# New file: structure_downloader.py
from mp_api.client import MPRester
from pymatgen.core import Structure
import pickle

def download_structures(material_ids, api_key):
    """Download crystal structures for materials"""
    structures = {}
    with MPRester(api_key) as mpr:
        for mat_id in material_ids:
            doc = mpr.materials.get_structure_by_material_id(mat_id)
            structures[mat_id] = doc
    return structures
```

### **2.2 Crystal Graph Construction**

**Create `crystal_graph.py`:**
```python
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure
import numpy as np

def structure_to_graph(structure: Structure, cutoff=8.0):
    """
    Convert pymatgen Structure to PyTorch Geometric graph

    Args:
        structure: Pymatgen Structure object
        cutoff: Distance cutoff for edges (Angstroms)

    Returns:
        torch_geometric.data.Data object
    """
    # Node features: atomic numbers
    atomic_numbers = [site.specie.Z for site in structure]
    x = torch.tensor(atomic_numbers, dtype=torch.long)

    # Edge construction: atoms within cutoff distance
    edges = []
    edge_attrs = []

    for i, site_i in enumerate(structure):
        for j, site_j in enumerate(structure):
            if i != j:
                distance = site_i.distance(site_j)
                if distance < cutoff:
                    edges.append([i, j])
                    edge_attrs.append(distance)

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)

    # Position information
    pos = torch.tensor([site.coords for site in structure], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
```

### **2.3 Element Feature Embeddings**

**Create `element_features.py`:**
```python
import torch
import numpy as np
from pymatgen.core import Element

def get_element_features(atomic_number):
    """
    Get element features for GNN

    Features:
    - Atomic number
    - Group
    - Period
    - Electronegativity
    - Covalent radius
    - Valence electrons
    - First ionization energy
    - Electron affinity
    """
    elem = Element.from_Z(atomic_number)

    features = [
        atomic_number / 100.0,  # Normalized atomic number
        elem.group / 18.0 if elem.group else 0,
        elem.row / 7.0,
        elem.X / 4.0 if elem.X else 0,  # Electronegativity
        elem.atomic_radius / 300.0 if elem.atomic_radius else 0,
        elem.valence if elem.valence else 0,
        elem.ionization_energy / 25.0 if elem.ionization_energy else 0,
        elem.electron_affinity / 5.0 if elem.electron_affinity else 0,
    ]

    return torch.tensor(features, dtype=torch.float)
```

---

## 🤖 Phase 3: GNN Model (Day 3-4)

### **3.1 CGCNN Architecture**

**Create `gnn_model.py`:**
```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool

class CGConv(MessagePassing):
    """Crystal Graph Convolutional Layer"""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Edge network
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate source node, target node, and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_mlp(z)

    def update(self, aggr_out, x):
        # Update node features
        z = torch.cat([x, aggr_out], dim=1)
        return self.node_mlp(z) + x  # Residual connection


class CGCNN(nn.Module):
    """Crystal Graph Convolutional Neural Network"""
    def __init__(self, node_feature_dim=8, edge_feature_dim=1,
                 hidden_dim=64, n_conv=3, output_dim=1):
        super().__init__()

        # Embedding layer for atomic numbers
        self.embedding = nn.Embedding(100, node_feature_dim)

        # Convolutional layers
        self.convs = nn.ModuleList([
            CGConv(node_feature_dim, edge_feature_dim, hidden_dim)
            for _ in range(n_conv)
        ])

        # Output layers
        self.fc1 = nn.Linear(node_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Embed atomic numbers
        x = self.embedding(x)

        # Apply convolutional layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        # Pool to graph-level representation
        x = global_mean_pool(x, batch)

        # Output layers
        x = self.fc1(x)
        x = self.softplus(x)
        x = self.fc2(x)

        return x
```

### **3.2 Simpler Alternative: Pre-built Models**

**Use matbench or pretrained models:**
```python
# Option: Use pre-trained ALIGNN or MEGNet
from alignn.pretrained import get_figshare_model

model = get_figshare_model("mp_e_form")  # Pre-trained formation energy model
prediction = model.predict_structure(structure)
```

---

## 🎨 Phase 4: Streamlit Interface (Day 4-5)

### **4.1 Update App 2: GNN Property Predictor**

**New `pages/2_GNN_Property_Predictor.py`:**
```python
import streamlit as st
import torch
from crystal_graph import structure_to_graph
from gnn_model import CGCNN
from pymatgen.core import Structure

st.title("🧠 GNN Property Predictor")

st.markdown("""
Predict materials properties using Graph Neural Networks trained on crystal structures.

**Models available:**
- Formation Energy (eV/atom)
- Band Gap (eV)
- Bulk Modulus (GPa)
""")

# Model selection
model_type = st.selectbox("Select property to predict:",
                          ["Formation Energy", "Band Gap", "Bulk Modulus"])

# Input method
input_method = st.radio("Input method:",
                       ["Upload CIF file", "Materials Project ID", "Draw structure"])

if input_method == "Upload CIF file":
    uploaded_file = st.file_uploader("Upload CIF file", type=['cif'])

    if uploaded_file:
        # Parse CIF
        structure = Structure.from_str(uploaded_file.read().decode(), fmt='cif')

        # Display structure info
        st.write(f"Formula: {structure.formula}")
        st.write(f"Space group: {structure.get_space_group_info()}")

        # Convert to graph
        graph = structure_to_graph(structure)

        # Load model and predict
        model = load_model(model_type)

        with torch.no_grad():
            prediction = model(graph)

        st.success(f"Predicted {model_type}: {prediction.item():.4f}")

elif input_method == "Materials Project ID":
    mat_id = st.text_input("Enter Material ID (e.g., mp-149):")

    if mat_id and st.button("Predict"):
        # Download structure from MP
        structure = download_structure_from_mp(mat_id)
        # ... (same as above)
```

### **4.2 Add Training Interface**

```python
# Training tab
tab1, tab2, tab3 = st.tabs(["Predict", "Train Model", "Model Info"])

with tab2:
    st.markdown("### Train Custom GNN Model")

    # Upload training data
    training_data = st.file_uploader("Upload training dataset (CSV with structures)")

    if training_data:
        # Show training options
        n_epochs = st.slider("Number of epochs", 10, 1000, 100)
        learning_rate = st.number_input("Learning rate", value=0.001, format="%.4f")

        if st.button("🚀 Start Training"):
            # Train model
            with st.spinner("Training GNN model..."):
                train_gnn(training_data, n_epochs, learning_rate)
```

---

## 📊 Phase 5: Integration & Testing (Day 5-6)

### **5.1 Connect App 1 → App 2**

**Add structure export in App 1:**
```python
# In Materials Project Explorer
if st.button("Export structures for GNN training"):
    # Create structure file
    structures_dict = {}
    for mat_id in downloaded_materials:
        structures_dict[mat_id] = get_structure(mat_id)

    # Save as pickle
    import pickle
    pickle_data = pickle.dumps(structures_dict)

    st.download_button(
        label="Download Structures (.pkl)",
        data=pickle_data,
        file_name="mp_structures.pkl",
        mime="application/octet-stream"
    )
```

### **5.2 Comparison Tool**

**Create side-by-side comparison:**
```python
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sklearn (Composition-only)")
    sklearn_pred = sklearn_model.predict(composition)
    st.metric("Prediction", f"{sklearn_pred:.4f}")

with col2:
    st.markdown("### GNN (Structure-aware)")
    gnn_pred = gnn_model.predict(structure_graph)
    st.metric("Prediction", f"{gnn_pred:.4f}")

st.markdown("### Comparison")
st.write(f"Difference: {abs(gnn_pred - sklearn_pred):.4f}")
```

---

## 🚀 Phase 6: Deployment (Day 7)

### **6.1 Update Requirements**

**Streamlit Cloud considerations:**
- PyTorch is heavy (~800MB) - check size limits
- CPU-only inference (no GPU)
- May need torch-cpu version for smaller size

**Optimized requirements.txt:**
```python
# Lighter PyTorch for Streamlit Cloud
torch>=2.0.0,<2.1.0
torch-geometric>=2.3.0
pymatgen-core>=2024.0.0  # Lighter than full pymatgen
```

### **6.2 Performance Optimization**

```python
# Cache model loading
@st.cache_resource
def load_gnn_model():
    model = CGCNN()
    model.load_state_dict(torch.load('models/formation_energy_model.pt',
                                     map_location='cpu'))
    model.eval()
    return model

# Cache structure conversions
@st.cache_data
def structure_to_graph_cached(structure_dict):
    structure = Structure.from_dict(structure_dict)
    return structure_to_graph(structure)
```

---

## 📦 Deliverables

### **MVP (Minimum Viable Product):**
1. ✅ GNN model that predicts formation energy
2. ✅ Upload CIF file → get prediction
3. ✅ Materials Project ID → get prediction
4. ✅ Comparison with sklearn baseline
5. ✅ Working on Streamlit Cloud

### **Nice-to-Have Features:**
- Multiple properties (band gap, bulk modulus)
- Model training interface
- Pre-trained model zoo
- Structure visualization (3D)
- Batch predictions
- Uncertainty quantification

---

## ⚠️ Challenges & Solutions

### **Challenge 1: Large Dependencies**
**Problem:** PyTorch + PyTorch Geometric = ~1GB
**Solutions:**
- Use torch CPU-only version
- Use pymatgen-core instead of full pymatgen
- Consider model compression
- Alternative: Use lighter framework (though less powerful)

### **Challenge 2: No GPU on Streamlit Cloud Free Tier**
**Problem:** GNN inference is slower on CPU
**Solutions:**
- Use smaller models
- Cache predictions
- Limit batch size
- Consider upgrading to Streamlit Cloud Pro (has GPU)

### **Challenge 3: Structure Data**
**Problem:** CIF files not always available
**Solutions:**
- Support multiple input formats (CIF, POSCAR, JSON)
- Download from Materials Project API
- Provide example structures
- Allow composition-only mode (fallback to sklearn)

### **Challenge 4: Model Training Time**
**Problem:** Training GNN takes hours/days
**Solutions:**
- Provide pre-trained models
- Train locally, deploy to cloud
- Use transfer learning
- Start with small datasets

---

## 📚 Resources

### **Tutorials & Papers:**
- [CGCNN Paper](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)
- [PyTorch Geometric Tutorial](https://pytorch-geometric.readthedocs.io/)
- [Matbench Leaderboard](https://matbench.materialsproject.org/)

### **Pre-trained Models:**
- [ALIGNN Models](https://github.com/usnistgov/alignn)
- [MEGNet Models](https://github.com/materialsvirtuallab/megnet)
- [Matbench Models](https://github.com/materialsproject/matbench)

### **Datasets:**
- Materials Project (150k+ materials)
- OQMD (600k+ materials)
- AFLOW (3M+ materials)

---

## 🎯 Success Criteria

**MVP is successful if:**
1. ✅ User can upload CIF → get formation energy prediction
2. ✅ GNN accuracy > sklearn baseline (R² > 0.9 for formation energy)
3. ✅ Inference time < 5 seconds per structure
4. ✅ Deploys successfully to Streamlit Cloud
5. ✅ Works without errors for 95%+ of MP structures

---

## 🚀 Next Steps

**To start implementation:**
1. Choose approach (pre-trained vs. train from scratch)
2. Install dependencies locally
3. Test basic graph construction
4. Implement core GNN model
5. Create Streamlit interface
6. Test & deploy

**Let me know which approach you prefer and I'll help you implement it!**
