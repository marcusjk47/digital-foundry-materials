# How to Predict Formation Energy for New Materials

This guide explains how to use your trained GNN model to predict formation energies for Fe-Ni compositions **not included** in your training dataset.

---

## Quick Start

### Option 1: Quick Batch Predictions (Easiest)

```bash
python quick_predict.py
```

This automatically predicts formation energies for:
- Pure Ni
- Fe25Ni75
- Fe50Ni50
- Fe75Ni25
- Pure Fe

Both FCC and BCC structures!

**Example Output:**
```
Composition          Structure  Formation Energy (eV/atom)
----------------------------------------------------------------------
Pure Ni              FCC        -0.0523
Pure Ni              BCC        +0.1234
Fe25Ni75             FCC        -0.2341
Fe25Ni75             BCC        -0.0987
Fe50Ni50             FCC        -0.4521
...
```

### Option 2: Interactive Predictions (More Control)

```bash
python predict_new_material.py
```

Gives you 3 options:
1. **Create from composition** - Specify Fe/Ni ratio
2. **Fetch from Materials Project** - Use material ID
3. **Load from file** - Use your own CIF/POSCAR file

---

## Understanding the Workflow

### Step 1: Get a Crystal Structure

You need a **crystal structure** (3D atomic arrangement) for your composition. Three ways to get one:

#### Method A: Create a Simple Structure (Best for Exploration)

```python
from pymatgen.core import Structure, Lattice

# Create FCC Fe50Ni50
a = 3.6  # lattice parameter
lattice = Lattice.cubic(a)

positions = [
    [0.0, 0.0, 0.0],  # Fe
    [0.5, 0.5, 0.0],  # Ni
    [0.5, 0.0, 0.5],  # Fe
    [0.0, 0.5, 0.5]   # Ni
]

elements = ['Fe', 'Ni', 'Fe', 'Ni']

structure = Structure(lattice, elements, positions)
```

**Pros:** Quick, simple, works for exploration
**Cons:** Not necessarily the real ground-state structure

#### Method B: Get from Materials Project (Most Accurate)

```python
from mp_api.client import MPRester

api_key = "your_api_key_here"

with MPRester(api_key) as mpr:
    # Search for Fe-Ni materials
    results = mpr.materials.summary.search(
        chemsys="Fe-Ni",
        num_elements=(2, 2)
    )

    # Get structure for first result
    structure = mpr.get_structure_by_material_id(results[0].material_id)
```

**Pros:** Real, DFT-relaxed structures
**Cons:** Limited to materials already in database

#### Method C: Load from File

```python
from pymatgen.core import Structure

# From CIF file
structure = Structure.from_file("my_material.cif")

# From POSCAR file
structure = Structure.from_file("POSCAR")
```

**Pros:** Use your own custom structures
**Cons:** Need to create/obtain files first

---

### Step 2: Convert Structure to Graph

```python
from crystal_graph import structure_to_graph

# Standard graph (if you trained without CALPHAD)
graph = structure_to_graph(
    structure,
    cutoff=8.0,        # Atoms within 8Å are neighbors
    max_neighbors=12   # Each atom connects to 12 nearest
)
```

**OR** if you trained with CALPHAD features:

```python
from crystal_graph import structure_to_graph_with_calphad

graph = structure_to_graph_with_calphad(
    structure,
    cutoff=8.0,
    max_neighbors=12
)
```

**What this does:**
- Identifies all atoms (nodes)
- Finds neighbors within cutoff radius (edges)
- Adds features (element type, position, CALPHAD data)

---

### Step 3: Load Your Trained Model

```python
import torch
from gnn_model import CGCNN

# Create model with SAME architecture as training
model = CGCNN(
    node_feature_dim=64,
    edge_feature_dim=1,
    hidden_dim=128,
    n_conv=3,
    n_hidden=2,
    output_dim=1
)

# Load trained weights
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode
```

**Important:** Model architecture MUST match training settings!

---

### Step 4: Make Prediction

```python
with torch.no_grad():  # Disable gradient calculation
    prediction = model(graph)
    formation_energy = prediction.item()

print(f"Predicted formation energy: {formation_energy:.4f} eV/atom")
```

**That's it!** The model outputs formation energy in eV/atom.

---

## Complete Example: Predict for Fe30Ni70

```python
import torch
from pymatgen.core import Structure, Lattice
from crystal_graph import structure_to_graph
from gnn_model import CGCNN

# 1. Create structure
a = 3.6
lattice = Lattice.cubic(a)
positions = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
elements = ['Fe', 'Ni', 'Ni', 'Ni']  # 25% Fe, 75% Ni
structure = Structure(lattice, elements, positions)

# 2. Convert to graph
graph = structure_to_graph(structure, cutoff=8.0, max_neighbors=12)

# 3. Load model
model = CGCNN(node_feature_dim=64, edge_feature_dim=1,
              hidden_dim=128, n_conv=3, n_hidden=2, output_dim=1)
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 4. Predict
with torch.no_grad():
    prediction = model(graph)
    energy = prediction.item()

print(f"Fe25Ni75 formation energy: {energy:.4f} eV/atom")
```

---

## Interpreting Results

### Formation Energy Values

**Negative values** = Stable (favorable to form)
- Example: -0.45 eV/atom means this alloy is stable

**Positive values** = Unstable (will decompose)
- Example: +0.23 eV/atom means this composition won't stay mixed

**More negative = more stable**
- -0.60 is more stable than -0.30

### Typical Ranges for Fe-Ni

- **Pure elements:** ~0.0 eV/atom (reference)
- **Stable alloys:** -0.1 to -0.6 eV/atom
- **Unstable mixtures:** +0.1 to +0.5 eV/atom

### Comparing FCC vs BCC

```python
# Predict both
fcc_structure = create_feni_structure(0.5, lattice_type='fcc')
bcc_structure = create_feni_structure(0.5, lattice_type='bcc')

fcc_energy = predict(fcc_structure)
bcc_energy = predict(bcc_structure)

if fcc_energy < bcc_energy:
    print(f"FCC is more stable by {bcc_energy - fcc_energy:.4f} eV/atom")
else:
    print(f"BCC is more stable by {fcc_energy - bcc_energy:.4f} eV/atom")
```

---

## Advanced: Batch Predictions

Predict for many compositions at once:

```python
import numpy as np
import matplotlib.pyplot as plt

# Compositions to test
fe_fractions = np.linspace(0, 1, 21)  # 0%, 5%, 10%, ..., 100%

results = []

for fe_frac in fe_fractions:
    # Create structure for this composition
    structure = create_feni_structure(fe_frac)

    # Predict
    energy = predict_formation_energy(structure)

    results.append({
        'fe_fraction': fe_frac,
        'ni_fraction': 1 - fe_frac,
        'energy': energy
    })

# Plot
plt.figure(figsize=(10, 6))
plt.plot([r['fe_fraction']*100 for r in results],
         [r['energy'] for r in results],
         'o-', linewidth=2)
plt.xlabel('Fe Content (%)')
plt.ylabel('Formation Energy (eV/atom)')
plt.title('Fe-Ni Alloy Formation Energy')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.savefig('formation_energy_curve.png')
plt.show()

# Find most stable
most_stable = min(results, key=lambda x: x['energy'])
print(f"Most stable: Fe{most_stable['fe_fraction']*100:.0f}Ni{most_stable['ni_fraction']*100:.0f}")
print(f"Energy: {most_stable['energy']:.4f} eV/atom")
```

This creates a composition-energy diagram!

---

## Important Notes

### Model Limitations

1. **Accuracy depends on training data**
   - If you trained on 0-40% Fe, predictions for 80% Fe may be unreliable
   - Model extrapolates poorly outside training range

2. **Structure matters**
   - Same composition in different structures gives different energies
   - Model predicts energy for THE EXACT STRUCTURE you give it
   - Not necessarily the ground state!

3. **Temperature effects**
   - Predictions are for 0 K (ground state)
   - Real materials at high temperature behave differently

4. **Validation recommended**
   - Compare predictions to known materials for sanity check
   - Use DFT for critical compositions

### Best Practices

✅ **DO:**
- Test multiple structures (FCC, BCC, etc.)
- Use larger supercells for better statistics
- Validate against Materials Project for known materials
- Check if prediction is within training data range

❌ **DON'T:**
- Trust predictions far from training data
- Ignore structural effects
- Use for critical applications without validation
- Expect perfect accuracy

---

## Troubleshooting

### "No trained model found"
```bash
# Train a model first in the Streamlit app:
streamlit run Digital_Foundry.py
# Go to GNN Property Predictor → Training
```

### "Model architecture mismatch"
Your prediction script must use the **same** architecture as training:
- Same `hidden_dim`
- Same `n_conv`
- Same `n_hidden`
- Same feature settings (CALPHAD yes/no)

Check your training settings in the Streamlit app.

### "Poor predictions"
- Check training data quality
- Ensure enough training samples (1000+)
- Verify model converged (low loss)
- Make sure structure is reasonable

### "Error converting structure"
- Check structure is valid: `structure.is_valid()`
- Ensure all elements are in periodic table
- Verify positions are within [0,1) fractional coordinates

---

## Example Workflow: Finding Optimal Fe-Ni Ratio

**Goal:** Find the most stable Fe-Ni composition

```python
# 1. Train model on Fe-Ni dataset (via Streamlit)
# 2. Run batch predictions
python quick_predict.py

# 3. Analyze results
# Look for most negative formation energy

# 4. Validate top candidates
# - Check against Materials Project
# - Run DFT calculations for verification
# - Consider experimental synthesis

# 5. Use predictions to guide experiments
# Instead of testing 100 compositions randomly,
# focus on the 5 most promising ones!
```

**Time saved:** Months of experiments → Days of computation

---

## Next Steps

After prediction:
1. **Validate** - Compare to known materials
2. **Refine** - Test different structures
3. **Experiment** - Synthesize promising candidates
4. **Iterate** - Add new data, retrain, improve

Remember: ML predictions are a **guide**, not absolute truth. Always validate critical results!

---

## Questions?

- Check the "How It Works" tab in GNN Property Predictor
- Review training metrics (MAE, R²) to gauge reliability
- Start with known materials to build confidence
- Use predictions to narrow down experimental search space

**Happy predicting!** 🚀
