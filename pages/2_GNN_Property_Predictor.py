"""
GNN Property Predictor

Graph Neural Network for predicting materials properties from crystal structures.
Uses Crystal Graph Convolutional Neural Networks (CGCNN) to leverage 3D atomic
structure information beyond simple composition.

Author: Digital Foundry Materials Science Toolkit
"""

import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import our GNN modules
try:
    from crystal_graph import structure_to_graph, get_graph_stats
    from element_features import get_element_features, get_element_feature_dim
    from gnn_model import CGCNN, count_parameters
    from pymatgen.core import Structure, Lattice
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="GNN Property Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 10px 0;
    }
    .demo-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff9800;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.title("🧠 GNN Property Predictor")
st.markdown("### Graph Neural Networks for Materials Science")

# Check if modules are available
if not MODULES_AVAILABLE:
    st.error(f"❌ **Error:** Could not load GNN modules: {IMPORT_ERROR}")
    st.info("💡 **Solution:** Make sure PyTorch and PyTorch Geometric are installed correctly.")
    st.code("pip install torch torch-geometric pymatgen", language="bash")
    st.stop()

# Sidebar: Model Configuration
st.sidebar.title("⚙️ Model Configuration")

model_mode = st.sidebar.radio(
    "Operation Mode:",
    ["🎯 Demo Mode", "📚 Architecture Info", "🔮 Prediction (Coming Soon)"],
    help="Demo mode shows GNN capabilities with example structures"
)

st.sidebar.markdown("---")

# Model architecture parameters
with st.sidebar.expander("🏗️ Model Architecture"):
    node_dim = st.slider("Node feature dimension", 32, 128, 64, 32)
    hidden_dim = st.slider("Hidden dimension", 64, 256, 128, 32)
    n_conv = st.slider("Number of conv layers", 1, 5, 3, 1)
    n_hidden = st.slider("Number of hidden layers", 0, 3, 1, 1)

# Graph construction parameters
with st.sidebar.expander("📊 Graph Parameters"):
    cutoff = st.slider("Cutoff distance (Å)", 3.0, 12.0, 8.0, 0.5,
                      help="Maximum distance for atomic connectivity")
    max_neighbors = st.slider("Max neighbors per atom", 4, 20, 12, 1,
                              help="Limits graph density")

st.sidebar.markdown("---")
st.sidebar.info("""
**GNN vs Traditional ML:**

Traditional ML uses only:
- Composition (e.g., Fe₀.₇Ni₀.₃)
- Element properties

GNNs additionally use:
- 3D atomic positions
- Bond lengths/angles
- Crystal symmetry
- Local atomic environments

**Result:** Higher accuracy for structure-dependent properties!
""")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if model_mode == "📚 Architecture Info":
    st.header("📚 Crystal Graph Convolutional Neural Network (CGCNN)")

    st.markdown("""
    <div class="info-box">
    <h3>🔬 What is a GNN?</h3>

    **Graph Neural Networks (GNNs)** are deep learning models that operate on graph-structured data.
    For materials, the graph represents the crystal structure:

    - **Nodes** = Atoms
    - **Edges** = Chemical bonds (atoms within cutoff distance)
    - **Node features** = Element properties (atomic number, electronegativity, radius, etc.)
    - **Edge features** = Bond distances

    Unlike traditional ML models that only use composition (e.g., "30% Ni, 70% Fe"),
    GNNs understand the 3D atomic arrangement and local bonding environment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Architecture Diagram
    st.subheader("🏗️ CGCNN Architecture")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **1. Input:** Crystal structure (CIF, POSCAR, or Materials Project ID)

        **2. Graph Construction:**
        - Convert atoms to graph nodes
        - Connect atoms within cutoff distance
        - Extract element features for each atom
        - Calculate bond distances

        **3. Message Passing (Convolution):**
        ```
        For each atom:
            Gather information from neighbors
            Update atom representation
            Repeat N times (N = number of conv layers)
        ```

        **4. Pooling:**
        - Aggregate all atom representations
        - Create graph-level feature vector

        **5. Prediction:**
        - Pass through MLP (Multi-Layer Perceptron)
        - Output predicted property
        """)

    with col2:
        st.markdown("""
        **Layer Flow:**

        Structure
        ↓
        Graph
        ↓
        Embedding
        ↓
        Conv Layer 1
        ↓
        Conv Layer 2
        ↓
        Conv Layer 3
        ↓
        Pooling
        ↓
        MLP
        ↓
        Prediction
        """)

    st.markdown("---")

    # Create example model
    st.subheader("🔧 Current Model Configuration")

    with st.spinner("Building model..."):
        model = CGCNN(
            node_feature_dim=node_dim,
            edge_feature_dim=1,
            hidden_dim=hidden_dim,
            n_conv=n_conv,
            n_hidden=n_hidden,
            output_dim=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Parameters", f"{count_parameters(model):,}")
    with col2:
        st.metric("Model Size", f"{count_parameters(model) * 4 / 1024 / 1024:.2f} MB")
    with col3:
        st.metric("Conv Layers", n_conv)

    with st.expander("📋 Full Model Architecture"):
        st.text(str(model))

    st.markdown("---")

    # Feature Information
    st.subheader("📊 Element Features")

    st.markdown(f"""
    Each atom is represented by **{get_element_feature_dim()} features**:

    1. **Atomic Number** - Element identity
    2. **Group Number** - Column in periodic table
    3. **Period** - Row in periodic table
    4. **Electronegativity** - Tendency to attract electrons (Pauling scale)
    5. **Atomic Radius** - Size of atom
    6. **Valence Electrons** - Electrons in outer shell
    7. **Ionization Energy** - Energy to remove electron
    8. **Electron Affinity** - Energy released when gaining electron
    9. **Atomic Mass** - Mass of atom
    """)

    # Example element features
    st.markdown("**Example: Iron (Fe)**")
    example_element = st.selectbox("Select element:",
                                   ["Fe", "Ni", "Al", "Ti", "Cu", "Au", "Ag"],
                                   key="element_select")

    element_map = {"Fe": 26, "Ni": 28, "Al": 13, "Ti": 22, "Cu": 29, "Au": 79, "Ag": 47}
    z = element_map[example_element]

    features = get_element_features(z, normalize=False)
    feature_names = ["Atomic #", "Group", "Period", "Electronegativity",
                    "Radius (Å)", "Valence e⁻", "Ionization (eV)", "Affinity (eV)", "Mass (amu)"]

    df_features = pd.DataFrame({
        "Feature": feature_names,
        "Value": features.numpy()
    })

    st.dataframe(df_features, use_container_width=True)

elif model_mode == "🎯 Demo Mode":
    st.header("🎯 Demo: GNN Property Prediction")

    st.markdown("""
    <div class="demo-box">
    <strong>ℹ️ Demo Mode:</strong> This demonstrates how the GNN processes crystal structures.
    The model shown here is <strong>untrained</strong>, so predictions are random.
    With training on real data, the model learns to predict properties like formation energy, band gap, bulk modulus, etc.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Example structures
    st.subheader("📦 Select Example Structure")

    example_structures = {
        "BCC Iron (Fe)": {
            "lattice": Lattice.cubic(2.87),
            "species": ["Fe", "Fe"],
            "coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
            "description": "Body-centered cubic iron, common structural steel"
        },
        "FCC Copper (Cu)": {
            "lattice": Lattice.cubic(3.61),
            "species": ["Cu"],
            "coords": [[0, 0, 0]],
            "description": "Face-centered cubic copper, excellent conductor"
        },
        "HCP Titanium (Ti)": {
            "lattice": Lattice.hexagonal(2.95, 4.68),
            "species": ["Ti", "Ti"],
            "coords": [[0, 0, 0], [1/3, 2/3, 1/2]],
            "description": "Hexagonal close-packed titanium, aerospace alloy"
        },
        "FCC Nickel (Ni)": {
            "lattice": Lattice.cubic(3.52),
            "species": ["Ni"],
            "coords": [[0, 0, 0]],
            "description": "Face-centered cubic nickel, corrosion resistant"
        },
        "BCC Aluminum (Al)": {
            "lattice": Lattice.cubic(4.05),
            "species": ["Al"],
            "coords": [[0, 0, 0]],
            "description": "Face-centered cubic aluminum, lightweight structural material"
        },
        "NaCl (Rock Salt)": {
            "lattice": Lattice.cubic(5.64),
            "species": ["Na", "Cl"],
            "coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
            "description": "Ionic crystal, demonstrating multi-element graphs"
        }
    }

    selected_structure = st.selectbox("Choose structure:", list(example_structures.keys()))

    struct_data = example_structures[selected_structure]
    st.info(f"ℹ️ {struct_data['description']}")

    # Create structure
    structure = Structure(
        struct_data['lattice'],
        struct_data['species'],
        struct_data['coords']
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Structure Information:**")
        st.write(f"- **Formula:** {structure.composition.reduced_formula}")
        st.write(f"- **Number of atoms:** {len(structure)}")
        st.write(f"- **Lattice type:** {struct_data['lattice'].__class__.__name__}")
        st.write(f"- **Volume:** {structure.volume:.2f} Å³")
        st.write(f"- **Density:** {structure.density:.2f} g/cm³")

    with col2:
        st.markdown("**Atomic Positions:**")
        coords_df = pd.DataFrame({
            "Element": [str(site.specie) for site in structure],
            "x": [site.frac_coords[0] for site in structure],
            "y": [site.frac_coords[1] for site in structure],
            "z": [site.frac_coords[2] for site in structure]
        })
        st.dataframe(coords_df, use_container_width=True)

    st.markdown("---")

    # Convert to graph
    st.subheader("🔄 Step 1: Convert Structure to Graph")

    with st.spinner("Creating graph representation..."):
        graph = structure_to_graph(structure, cutoff=cutoff, max_neighbors=max_neighbors)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Nodes (Atoms)", graph.num_nodes)
    with col2:
        st.metric("Edges (Bonds)", graph.num_edges)
    with col3:
        avg_degree = graph.num_edges / graph.num_nodes if graph.num_nodes > 0 else 0
        st.metric("Avg Neighbors", f"{avg_degree:.1f}")
    with col4:
        st.metric("Feature Dim", graph.x.shape[-1] if len(graph.x.shape) > 0 else 1)

    with st.expander("📊 Detailed Graph Statistics"):
        st.markdown("**Graph Properties:**")
        stats = get_graph_stats(graph)

        # Edge distance distribution
        if graph.num_edges > 0:
            st.markdown("**Edge Length Distribution:**")
            edge_distances = graph.edge_attr.squeeze().numpy()
            fig = px.histogram(edge_distances, nbins=20,
                             labels={"value": "Bond Distance (Å)", "count": "Frequency"},
                             title="Distribution of Interatomic Distances")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Run through model
    st.subheader("🧠 Step 2: GNN Forward Pass")

    # Create model
    with st.spinner("Initializing model..."):
        model = CGCNN(
            node_feature_dim=node_dim,
            edge_feature_dim=1,
            hidden_dim=hidden_dim,
            n_conv=n_conv,
            n_hidden=n_hidden,
            output_dim=1
        )
        model.eval()

    # Make prediction
    with st.spinner("Running prediction..."):
        with torch.no_grad():
            prediction = model(graph)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.success(f"✅ Prediction complete!")

        # Show prediction (note: untrained model, so random)
        st.markdown("**Model Output:**")
        st.metric("Predicted Value", f"{prediction.item():.4f}")

        st.markdown("""
        <div class="demo-box">
        <strong>⚠️ Note:</strong> This is an <strong>untrained</strong> model with random weights,
        so the prediction has no physical meaning. With proper training on labeled data
        (e.g., DFT-calculated formation energies), the model would learn to make accurate predictions.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Processing Steps:**")
        st.write("1. ✅ Graph construction")
        st.write("2. ✅ Node embedding")
        st.write(f"3. ✅ {n_conv}x convolution")
        st.write("4. ✅ Global pooling")
        st.write("5. ✅ MLP prediction")

    st.markdown("---")

    # Batch prediction demo
    st.subheader("📦 Step 3: Batch Prediction")

    st.markdown("GNNs can efficiently process multiple structures in parallel:")

    # Create multiple structures
    batch_structures = [
        Structure(Lattice.cubic(2.87), ["Fe"], [[0, 0, 0]]),
        Structure(Lattice.cubic(3.61), ["Cu"], [[0, 0, 0]]),
        Structure(Lattice.cubic(3.52), ["Ni"], [[0, 0, 0]]),
        Structure(Lattice.cubic(4.05), ["Al"], [[0, 0, 0]]),
    ]

    # Convert to graphs
    from torch_geometric.data import Batch
    graphs = [structure_to_graph(s, cutoff=cutoff, max_neighbors=max_neighbors)
              for s in batch_structures]
    batch = Batch.from_data_list(graphs)

    # Batch prediction
    with torch.no_grad():
        batch_predictions = model(batch)

    # Display results
    results_df = pd.DataFrame({
        "Structure": ["Fe (BCC)", "Cu (FCC)", "Ni (FCC)", "Al (FCC)"],
        "Formula": [s.composition.reduced_formula for s in batch_structures],
        "Atoms": [len(s) for s in batch_structures],
        "Prediction": batch_predictions.squeeze().numpy()
    })

    st.dataframe(results_df, use_container_width=True)

    fig = px.bar(results_df, x="Structure", y="Prediction",
                title="Batch Predictions for Different Structures",
                color="Prediction",
                color_continuous_scale="viridis")
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    💡 **Practical Use Cases:**
    - Screen thousands of candidate materials
    - Predict properties for hypothetical structures
    - Guide experimental synthesis
    - Accelerate materials discovery
    """)

elif model_mode == "🔮 Prediction (Coming Soon)":
    st.header("🔮 Property Prediction (Coming Soon)")

    st.markdown("""
    <div class="info-box">
    <h3>🚧 Under Development</h3>

    This mode will allow you to:

    1. **Upload crystal structures** (CIF, POSCAR files)
    2. **Query Materials Project** by material ID
    3. **Select pre-trained models** for different properties
    4. **Get predictions** with uncertainty estimates
    5. **Download results** as CSV

    <strong>Available Properties (Planned):</strong>
    - Formation Energy (eV/atom)
    - Band Gap (eV)
    - Bulk Modulus (GPa)
    - Shear Modulus (GPa)
    - Magnetic Moment (μB)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Preview interface
    st.subheader("🎨 Interface Preview")

    tab1, tab2, tab3 = st.tabs(["📤 Upload", "🔍 Materials Project", "📊 Results"])

    with tab1:
        st.markdown("**Upload Structure File**")
        st.file_uploader("Choose CIF or POSCAR file", type=['cif', 'poscar', 'vasp'], disabled=True)
        st.button("🚀 Predict Properties", disabled=True, type="primary")

    with tab2:
        st.markdown("**Query Materials Project**")
        st.text_input("Material ID (e.g., mp-149):", disabled=True)
        st.selectbox("Property to predict:",
                    ["Formation Energy", "Band Gap", "Bulk Modulus"],
                    disabled=True)
        st.button("🔍 Fetch and Predict", disabled=True, type="primary")

    with tab3:
        st.markdown("**Prediction Results (Example)**")

        # Mock results table
        mock_results = pd.DataFrame({
            "Property": ["Formation Energy", "Band Gap", "Bulk Modulus"],
            "Predicted Value": [-2.34, 0.15, 180.5],
            "Unit": ["eV/atom", "eV", "GPa"],
            "Uncertainty": [0.12, 0.05, 15.2]
        })

        st.dataframe(mock_results, use_container_width=True)
        st.download_button("📥 Download Results", "mock data", disabled=True)

    st.markdown("---")

    st.markdown("""
    **To enable this feature:**
    1. Train GNN models on Materials Project data
    2. Save trained model checkpoints
    3. Implement prediction interface
    4. Add uncertainty quantification

    **Estimated completion:** 2-3 weeks
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<strong>Digital Foundry Materials Science Toolkit</strong><br>
GNN Property Predictor | Powered by PyTorch Geometric & Pymatgen<br>
🧠 Leveraging 3D Crystal Structure for Accurate Property Prediction
</div>
""", unsafe_allow_html=True)
