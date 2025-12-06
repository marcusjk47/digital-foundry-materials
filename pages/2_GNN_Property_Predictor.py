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
    from crystal_graph import structure_to_graph, structure_to_graph_with_calphad, get_graph_stats
    from element_features import get_element_features, get_element_feature_dim
    from gnn_model import CGCNN, count_parameters
    from gnn_model_calphad import CGCNN_CALPHAD, CGCNN_CALPHAD_Regressor
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
    ["🎯 Demo Mode", "📚 Architecture Info", "🎓 Train Model", "🔮 Prediction (Coming Soon)"],
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

elif model_mode == "🎓 Train Model":
    st.header("🎓 Train GNN Model")

    st.markdown("""
    <div class="info-box">
    <h3>📖 Training Overview</h3>

    Train a CGCNN model to predict materials properties from crystal structures.

    **Training Process:**
    1. **Collect Data:** Fetch structures + properties from Materials Project
    2. **Prepare Dataset:** Convert structures to graphs, split train/val/test
    3. **Configure Model:** Set architecture parameters
    4. **Train:** Run training loop with validation
    5. **Evaluate:** Test on held-out test set
    6. **Deploy:** Use trained model for predictions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Training workflow tabs
    tab1, tab2, tab3 = st.tabs(["1️⃣ Data Collection", "2️⃣ Training", "3️⃣ Evaluation"])

    with tab1:
        st.subheader("📥 Collect Training Data from Materials Project")

        # Check for API key
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("MP_API_KEY")
            except:
                pass

        if not api_key:
            st.error("⚠️ **Materials Project API Key Required**")
            st.markdown("""
            To collect training data, you need a Materials Project API key:

            1. Get your free API key at https://next-gen.materialsproject.org/api
            2. Add it to your environment:
               - **Streamlit Cloud:** Add `MP_API_KEY` in App Settings → Secrets
               - **Local:** Create `.env` file with `MP_API_KEY=your_key_here`
            """)
            st.stop()

        st.success("✅ Materials Project API Connected")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Dataset Configuration**")

            data_mode = st.radio(
                "Data selection:",
                ["Chemical System", "Elements", "All Metallic"],
                help="How to filter materials"
            )

            if data_mode == "Chemical System":
                chemsys = st.text_input(
                    "Chemical system:",
                    value="Fe-Ni",
                    help="e.g., Fe-Ni, Fe-Ni-Cr, Ti-Al"
                )
                elements = None
            elif data_mode == "Elements":
                elements_str = st.text_input(
                    "Elements (comma-separated):",
                    value="Fe,Ni,Cr",
                    help="e.g., Fe,Ni,Cr"
                )
                elements = [e.strip() for e in elements_str.split(",")]
                chemsys = None
            else:
                elements = None
                chemsys = None

            max_materials = st.number_input(
                "Max materials to collect:",
                min_value=10,
                max_value=10000,
                value=500,
                step=50,
                help="More materials = better model, but longer download"
            )

        with col2:
            st.markdown("**Filtering Options**")

            metallic_only = st.checkbox("Metallic only (band gap = 0)", value=True)
            stable_only = st.checkbox("Stable materials only", value=False)

            target_property = st.selectbox(
                "Target property to predict:",
                ["formation_energy_per_atom", "band_gap", "energy_above_hull"],
                help="Property the model will learn to predict"
            )

            st.markdown("**Advanced Features**")

            use_calphad = st.checkbox(
                "Enable CALPHAD features",
                value=False,
                help="Enhance graphs with thermodynamic features (melting point, heat capacity, mixing energy)"
            )

            if use_calphad:
                st.info("ℹ️ CALPHAD features add thermodynamic properties:\n"
                       "- Node features: 13D (atomic# + 9 element + 3 CALPHAD)\n"
                       "- Edge features: 2D (distance + mixing energy)")

            dataset_name = st.text_input(
                "Dataset name:",
                value="my_dataset",
                help="Name for saving the dataset"
            )

        st.markdown("---")

        if st.button("🚀 Collect Data", type="primary", use_container_width=True):
            try:
                from gnn_data_collection import fetch_materials_data, convert_to_graphs, get_dataset_statistics, print_dataset_info

                with st.spinner("Fetching materials from Materials Project..."):
                    df = fetch_materials_data(
                        api_key=api_key,
                        elements=elements,
                        chemsys=chemsys,
                        max_materials=max_materials,
                        metallic_only=metallic_only,
                        stable_only=stable_only
                    )

                if df.empty:
                    st.error("No materials found with these criteria!")
                    st.stop()

                st.success(f"✅ Fetched {len(df)} materials")

                # Convert to graphs
                with st.spinner("Converting structures to graphs..."):
                    save_path = f"datasets/{dataset_name}.pkl"
                    graphs = convert_to_graphs(
                        df,
                        target_property=target_property,
                        cutoff=cutoff,
                        max_neighbors=max_neighbors,
                        use_calphad=use_calphad,
                        save_path=save_path
                    )

                st.success(f"✅ Converted {len(graphs)} structures to graphs")
                st.success(f"💾 Saved to {save_path}")

                if use_calphad:
                    st.success("🔬 CALPHAD features enabled - graphs enhanced with thermodynamic properties!")

                # Show statistics
                stats = get_dataset_statistics(graphs)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Graphs", stats["num_graphs"])
                with col2:
                    st.metric("Avg Nodes/Graph", f"{stats['avg_nodes']:.1f}")
                with col3:
                    st.metric("Avg Edges/Graph", f"{stats['avg_edges']:.1f}")
                with col4:
                    st.metric("Target Mean", f"{stats['target_mean']:.4f}")

                # Show feature dimensions
                if graphs:
                    sample_graph = graphs[0]
                    st.markdown("**Graph Features:**")
                    feat_col1, feat_col2 = st.columns(2)
                    with feat_col1:
                        node_dim = sample_graph.x.shape[1] if len(sample_graph.x.shape) > 1 else 1
                        st.metric("Node Feature Dim", node_dim)
                    with feat_col2:
                        edge_dim = sample_graph.edge_attr.shape[1] if len(sample_graph.edge_attr.shape) > 1 else 1
                        st.metric("Edge Feature Dim", edge_dim)

                # Show sample
                st.markdown("**Sample Materials:**")
                sample_df = df[["material_id", "formula", target_property]].head(10)
                st.dataframe(sample_df, use_container_width=True)

                st.info(f"💡 **Next Step:** Go to the 'Training' tab to train a model on this dataset!")

            except Exception as e:
                st.error(f"❌ Error collecting data: {e}")
                st.exception(e)

    with tab2:
        st.subheader("🎓 Train GNN Model")

        # Check for available datasets (both old .pkl and new sharded)
        datasets_dir = Path("datasets")
        available_datasets = {}  # name -> (type, path)

        if datasets_dir.exists():
            # Old single-file datasets
            for pkl_file in datasets_dir.glob("*.pkl"):
                available_datasets[pkl_file.stem] = ("single_file", str(pkl_file))

            # New sharded datasets
            for subdir in datasets_dir.iterdir():
                if subdir.is_dir() and (subdir / "manifest.json").exists():
                    available_datasets[subdir.name] = ("sharded", str(subdir / "manifest.json"))

        if not available_datasets:
            st.warning("⚠️ No datasets found. Please collect data first in the 'Data Collection' tab or Dataset Manager.")
            st.stop()

        # Dataset selection
        st.markdown("**1. Select Dataset(s)**")

        # Single or multi-dataset
        dataset_mode = st.radio(
            "Training mode:",
            ["Single Dataset", "Multiple Datasets (Combined)"],
            horizontal=True,
            help="Train on one dataset or combine multiple"
        )

        if dataset_mode == "Single Dataset":
            selected_dataset_name = st.selectbox(
                "Choose dataset:",
                list(available_datasets.keys()),
                help="Select a dataset to train on"
            )
            selected_datasets = [selected_dataset_name]
        else:
            selected_datasets = st.multiselect(
                "Choose datasets to combine:",
                list(available_datasets.keys()),
                help="Select multiple datasets to train on together"
            )

            if not selected_datasets:
                st.info("👆 Select at least one dataset above")
                st.stop()

        # Show dataset info
        st.markdown("**Dataset Info:**")
        total_materials = 0

        for ds_name in selected_datasets:
            ds_type, ds_path = available_datasets[ds_name]

            if ds_type == "sharded":
                # Load manifest
                import json
                with open(ds_path, 'r') as f:
                    manifest = json.load(f)
                count = manifest['total_materials']
                info_str = f"📊 {ds_name}: {count:,} materials (sharded, {manifest['num_shards']} shards)"
            else:
                # Old format - try to load and count
                try:
                    from gnn_data_collection import load_graph_dataset
                    graphs = load_graph_dataset(ds_path)
                    count = len(graphs)
                    info_str = f"📁 {ds_name}: {count:,} materials (single file)"
                except:
                    count = 0
                    info_str = f"📁 {ds_name}: Unknown size (single file)"

            st.write(info_str)
            total_materials += count

        if len(selected_datasets) > 1:
            st.info(f"**Combined total:** {total_materials:,} materials")

        selected_dataset = selected_datasets[0] if len(selected_datasets) == 1 else None

        st.markdown("---")

        # Training configuration
        st.markdown("**2. Training Configuration**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Training Parameters**")
            epochs = st.number_input("Number of epochs:", min_value=10, max_value=500, value=100, step=10)
            batch_size = st.number_input("Batch size:", min_value=8, max_value=128, value=32, step=8)
            learning_rate = st.number_input("Learning rate:", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
            patience = st.number_input("Early stopping patience:", min_value=5, max_value=50, value=20, step=5)

        with col2:
            st.markdown("**Dataset Split**")
            train_ratio = st.slider("Train %:", 60, 90, 80, 5) / 100
            val_ratio = st.slider("Validation %:", 5, 20, 10, 5) / 100
            test_ratio = 1.0 - train_ratio - val_ratio
            st.metric("Test %:", f"{test_ratio*100:.0f}%")

        st.markdown("---")

        # Start training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            try:
                from gnn_train import GNNTrainer, print_evaluation_results
                import time

                # Determine dataset type and create appropriate loaders
                has_calphad = False
                dataset_size = 0

                # Check if we're using sharded datasets
                using_sharded = any(available_datasets[ds][0] == "sharded" for ds in selected_datasets)

                if len(selected_datasets) > 1:
                    # Multi-dataset training
                    st.info(f"🔄 **Multi-dataset training mode:** Combining {len(selected_datasets)} datasets")

                    with st.spinner("Loading multiple datasets..."):
                        from gnn_dataset_sharded import create_multi_dataset_loaders

                        # Get all manifest paths
                        manifest_paths = []
                        for ds_name in selected_datasets:
                            ds_type, ds_path = available_datasets[ds_name]
                            if ds_type == "sharded":
                                manifest_paths.append(ds_path)
                            else:
                                st.error(f"❌ Dataset '{ds_name}' is not sharded. Multi-dataset training requires all datasets to be sharded.")
                                st.info("💡 **Tip:** Use the Dataset Manager to convert or collect datasets in sharded format.")
                                st.stop()

                        # Create combined dataloaders
                        train_loader, val_loader, test_loader = create_multi_dataset_loaders(
                            manifest_paths=manifest_paths,
                            batch_size=batch_size,
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            test_ratio=1.0 - train_ratio - val_ratio,
                            num_workers=0,
                            cache_size=3
                        )

                        dataset_size = total_materials

                        # Detect CALPHAD from first batch
                        sample_batch = next(iter(train_loader))
                        node_feature_dim = sample_batch.x.shape[1] if len(sample_batch.x.shape) > 1 else 1
                        edge_feature_dim = sample_batch.edge_attr.shape[1] if len(sample_batch.edge_attr.shape) > 1 else 1
                        has_calphad = (node_feature_dim == 13 and edge_feature_dim == 2)

                elif using_sharded:
                    # Single sharded dataset
                    ds_type, ds_path = available_datasets[selected_datasets[0]]
                    st.info(f"📊 **Sharded dataset mode:** Memory-efficient loading")

                    with st.spinner("Loading sharded dataset..."):
                        from gnn_dataset_sharded import create_sharded_dataloaders

                        train_loader, val_loader, test_loader = create_sharded_dataloaders(
                            manifest_path=ds_path,
                            batch_size=batch_size,
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            test_ratio=1.0 - train_ratio - val_ratio,
                            num_workers=0,
                            cache_size=3
                        )

                        # Get dataset size from manifest
                        import json
                        with open(ds_path, 'r') as f:
                            manifest = json.load(f)
                        dataset_size = manifest['total_materials']
                        has_calphad = manifest.get('calphad_enhanced', False)

                        # Verify CALPHAD from first batch if not in manifest
                        if not has_calphad:
                            sample_batch = next(iter(train_loader))
                            node_feature_dim = sample_batch.x.shape[1] if len(sample_batch.x.shape) > 1 else 1
                            edge_feature_dim = sample_batch.edge_attr.shape[1] if len(sample_batch.edge_attr.shape) > 1 else 1
                            has_calphad = (node_feature_dim == 13 and edge_feature_dim == 2)

                else:
                    # Old single-file dataset (backwards compatibility)
                    ds_type, ds_path = available_datasets[selected_datasets[0]]
                    st.info(f"📁 **Single-file dataset mode:** Loading entire dataset into memory")

                    with st.spinner("Loading dataset..."):
                        from gnn_data_collection import load_graph_dataset
                        from gnn_dataset import CrystalGraphDataset, split_dataset, create_data_loaders

                        graphs = load_graph_dataset(ds_path)
                        dataset = CrystalGraphDataset(graphs)
                        dataset_size = len(dataset)

                        # Detect CALPHAD features
                        sample_graph = graphs[0]
                        node_feature_dim = sample_graph.x.shape[1] if len(sample_graph.x.shape) > 1 else 1
                        edge_feature_dim = sample_graph.edge_attr.shape[1] if len(sample_graph.edge_attr.shape) > 1 else 1
                        has_calphad = (node_feature_dim == 13 and edge_feature_dim == 2)

                        # Split dataset
                        train_dataset, val_dataset, test_dataset = split_dataset(
                            dataset,
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            test_ratio=1.0 - train_ratio - val_ratio
                        )

                        # Validate dataset sizes
                        if len(train_dataset) == 0:
                            st.error("❌ **Training set is empty!** Please collect more data or adjust split ratios.")
                            st.stop()

                        if len(val_dataset) == 0:
                            st.error("❌ **Validation set is empty!** Please collect more data or adjust split ratios.")
                            st.info("💡 **Tip:** Increase validation % or collect more materials")
                            st.stop()

                        # Create data loaders
                        train_loader, val_loader, test_loader = create_data_loaders(
                            train_dataset, val_dataset, test_dataset,
                            batch_size=batch_size
                        )

                # Show dataset info
                st.success(f"✅ Loaded {dataset_size:,} materials")

                if has_calphad:
                    st.info("🔬 **CALPHAD-enhanced dataset detected!**")
                    st.info("   Node features: 13D (atomic# + element + CALPHAD)")
                    st.info("   Edge features: 2D (distance + mixing energy)")

                if dataset_size < 10:
                    st.warning(f"⚠️ **Small dataset warning:** Only {dataset_size} samples. Recommend 100+ for reliable training.")

                # Create model (use CALPHAD model if CALPHAD features detected)
                if has_calphad:
                    model = CGCNN_CALPHAD_Regressor(
                        input_node_dim=13,
                        input_edge_dim=2,
                        node_feature_dim=node_dim,
                        edge_feature_dim=32,
                        hidden_dim=hidden_dim,
                        n_conv=n_conv,
                        n_hidden=n_hidden
                    )
                    st.success("✅ Using CALPHAD-enhanced CGCNN model")
                else:
                    model = CGCNN(
                        node_feature_dim=node_dim,
                        edge_feature_dim=1,
                        hidden_dim=hidden_dim,
                        n_conv=n_conv,
                        n_hidden=n_hidden,
                        output_dim=1
                    )
                    st.info("Using standard CGCNN model")

                st.info(f"🧠 Model has {count_parameters(model):,} parameters")

                # Create trainer
                device = "cpu"  # Streamlit Cloud uses CPU
                trainer = GNNTrainer(
                    model=model,
                    device=device,
                    learning_rate=learning_rate,
                    checkpoint_dir="checkpoints"
                )

                # Training progress
                st.markdown("---")
                st.markdown("### 📊 Training Progress")

                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()

                # Placeholder for live training (simplified for Streamlit)
                st.warning("⚠️ **Note:** Full training takes time. This will run synchronously and may timeout on Streamlit Cloud for large datasets. For long training runs, use the command-line script `gnn_train.py` locally.")

                start_time = time.time()

                # Train (this will block)
                with st.spinner("Training model... This may take several minutes."):
                    history = trainer.train(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=epochs,
                        patience=patience,
                        verbose=False  # Don't print to console
                    )

                training_time = time.time() - start_time

                st.success(f"✅ Training completed in {training_time/60:.1f} minutes!")

                # Show best results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Epoch", trainer.best_epoch)
                with col2:
                    st.metric("Best Val Loss", f"{trainer.best_val_loss:.6f}")
                with col3:
                    st.metric("Training Time", f"{training_time/60:.1f} min")

                # Plot training history
                st.markdown("### 📈 Training History")
                fig = trainer.plot_training_history()
                st.pyplot(fig)

                # Evaluate on test set
                st.markdown("### 🎯 Test Set Evaluation")
                with st.spinner("Evaluating on test set..."):
                    metrics, predictions, targets = trainer.evaluate(test_loader)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Test MAE", f"{metrics['mae']:.6f}")
                with col2:
                    st.metric("Test RMSE", f"{metrics['rmse']:.6f}")
                with col3:
                    st.metric("R² Score", f"{metrics['r2']:.4f}")
                with col4:
                    st.metric("Test Samples", metrics['num_samples'])

                # Predictions vs Actual plot
                fig_pred = px.scatter(
                    x=targets,
                    y=predictions,
                    labels={"x": "Actual", "y": "Predicted"},
                    title="Predictions vs Actual (Test Set)"
                )
                fig_pred.add_trace(
                    go.Scatter(x=[targets.min(), targets.max()],
                             y=[targets.min(), targets.max()],
                             mode="lines",
                             name="Perfect Prediction",
                             line=dict(dash="dash", color="red"))
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                st.success("✅ Model trained and saved to `checkpoints/best_model.pt`")
                st.info("💡 **Next Step:** Go to 'Evaluation' tab or use the trained model in 'Prediction' mode!")

            except Exception as e:
                st.error(f"❌ Error during training: {e}")
                st.exception(e)

    with tab3:
        st.subheader("📊 Model Evaluation")

        # Check for saved checkpoints
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob("*.pt"))
        else:
            checkpoint_files = []

        if not checkpoint_files:
            st.warning("⚠️ No trained models found. Please train a model first in the 'Training' tab.")
            st.stop()

        # Model selection
        selected_checkpoint = st.selectbox(
            "Select trained model:",
            [f.name for f in checkpoint_files]
        )

        # Dataset selection for evaluation (support both old and new formats)
        eval_available_datasets = {}  # name -> (type, path)

        if datasets_dir.exists():
            # Old single-file datasets
            for pkl_file in datasets_dir.glob("*.pkl"):
                eval_available_datasets[pkl_file.stem] = ("single_file", str(pkl_file))

            # New sharded datasets
            for subdir in datasets_dir.iterdir():
                if subdir.is_dir() and (subdir / "manifest.json").exists():
                    eval_available_datasets[subdir.name] = ("sharded", str(subdir / "manifest.json"))

        if eval_available_datasets:
            eval_dataset = st.selectbox(
                "Select dataset for evaluation:",
                list(eval_available_datasets.keys())
            )

            # Show dataset info
            ds_type, ds_path = eval_available_datasets[eval_dataset]
            if ds_type == "sharded":
                import json
                with open(ds_path, 'r') as f:
                    manifest = json.load(f)
                st.info(f"📊 {manifest['total_materials']:,} materials (sharded, {manifest['num_shards']} shards)")
            else:
                st.info(f"📁 Single-file dataset")

            if st.button("📊 Evaluate Model", type="primary"):
                try:
                    from gnn_train import GNNTrainer

                    # Load dataset based on type
                    has_calphad_eval = False

                    if ds_type == "sharded":
                        # Sharded dataset
                        st.info("📊 Loading sharded dataset...")
                        from gnn_dataset_sharded import ShardedCrystalGraphDataset
                        from torch_geometric.loader import DataLoader

                        dataset = ShardedCrystalGraphDataset(
                            manifest_path=ds_path,
                            cache_size=3
                        )

                        # Detect CALPHAD
                        import json
                        with open(ds_path, 'r') as f:
                            manifest = json.load(f)
                        has_calphad_eval = manifest.get('calphad_enhanced', False)

                        # Verify from first sample
                        if not has_calphad_eval:
                            sample_graph = dataset[0]
                            node_feature_dim_eval = sample_graph.x.shape[1] if len(sample_graph.x.shape) > 1 else 1
                            edge_feature_dim_eval = sample_graph.edge_attr.shape[1] if len(sample_graph.edge_attr.shape) > 1 else 1
                            has_calphad_eval = (node_feature_dim_eval == 13 and edge_feature_dim_eval == 2)

                        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

                    else:
                        # Old single-file dataset
                        st.info("📁 Loading single-file dataset...")
                        from gnn_data_collection import load_graph_dataset
                        from gnn_dataset import CrystalGraphDataset
                        from torch_geometric.loader import DataLoader

                        graphs = load_graph_dataset(ds_path)
                        dataset = CrystalGraphDataset(graphs)

                        # Detect CALPHAD features
                        sample_graph = graphs[0]
                        node_feature_dim_eval = sample_graph.x.shape[1] if len(sample_graph.x.shape) > 1 else 1
                        edge_feature_dim_eval = sample_graph.edge_attr.shape[1] if len(sample_graph.edge_attr.shape) > 1 else 1
                        has_calphad_eval = (node_feature_dim_eval == 13 and edge_feature_dim_eval == 2)

                        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

                    if has_calphad_eval:
                        st.info("🔬 CALPHAD-enhanced dataset detected")

                    # Load model (use CALPHAD model if CALPHAD features detected)
                    if has_calphad_eval:
                        model = CGCNN_CALPHAD_Regressor(
                            input_node_dim=13,
                            input_edge_dim=2,
                            node_feature_dim=node_dim,
                            edge_feature_dim=32,
                            hidden_dim=hidden_dim,
                            n_conv=n_conv,
                            n_hidden=n_hidden
                        )
                    else:
                        model = CGCNN(
                            node_feature_dim=node_dim,
                            edge_feature_dim=1,
                            hidden_dim=hidden_dim,
                            n_conv=n_conv,
                            n_hidden=n_hidden,
                            output_dim=1
                        )

                    trainer = GNNTrainer(model=model, device="cpu")
                    trainer.load_checkpoint(f"checkpoints/{selected_checkpoint}")

                    # Evaluate
                    with st.spinner("Evaluating model..."):
                        metrics, predictions, targets = trainer.evaluate(data_loader)

                    # Show results
                    st.success("✅ Evaluation complete!")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{metrics['mae']:.6f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['rmse']:.6f}")
                    with col3:
                        st.metric("R²", f"{metrics['r2']:.4f}")
                    with col4:
                        st.metric("Samples", metrics['num_samples'])

                    # Plot
                    fig = px.scatter(x=targets, y=predictions,
                                   labels={"x": "Actual", "y": "Predicted"},
                                   title="Model Predictions")
                    fig.add_trace(go.Scatter(x=[targets.min(), targets.max()],
                                           y=[targets.min(), targets.max()],
                                           mode="lines", name="Perfect",
                                           line=dict(dash="dash", color="red")))
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

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
