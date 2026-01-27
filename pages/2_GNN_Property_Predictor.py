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
    ["📚 Architecture Info", "🎓 Train Model", "🔮 Prediction"],
    help="Select the operation mode"
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

if model_mode == "🎓 Train Model":
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
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["🔍 Explore Materials", "1️⃣ Data Collection", "2️⃣ Training", "3️⃣ Evaluation", "📚 How It Works"])

    with tab0:
        st.subheader("🔍 Explore Materials Project Database")

        st.markdown("""
        Search and browse available materials in the Materials Project database before collecting datasets.
        This helps you understand what's available and plan your training data collection.
        """)

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
            Get your free API key at: https://next-gen.materialsproject.org/api
            Add it to your `.env` file or environment variables.
            """)
            st.stop()

        st.success("✅ Materials Project API Connected")

        # Search options
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Search By:**")
            search_mode = st.radio(
                "Search type:",
                ["Chemical System", "Elements", "Formula", "Properties"],
                help="Choose how to search for materials"
            )

            if search_mode == "Chemical System":
                search_query = st.text_input(
                    "Chemical system:",
                    value="Fe-Ni",
                    help="e.g., Fe-Ni, Ti-Al, Fe-Ni-Cr"
                )
            elif search_mode == "Elements":
                search_query = st.text_input(
                    "Elements (comma-separated):",
                    value="Fe,Ni",
                    help="e.g., Fe,Ni or Ti,Al,V"
                )
            elif search_mode == "Formula":
                search_query = st.text_input(
                    "Chemical formula:",
                    value="",
                    help="e.g., Fe2O3, TiO2"
                )
            else:  # Properties
                search_query = None
                st.info("Property-based search - use filters below")

        with col2:
            st.markdown("**Filters:**")
            filter_metallic = st.checkbox("Metallic only (band gap = 0)", value=False)
            filter_stable = st.checkbox("Stable materials only (on convex hull)", value=False)

            property_filter = st.selectbox(
                "Must have property:",
                ["Any", "Formation Energy", "Band Gap", "Bulk Modulus", "Shear Modulus", "Density"],
                help="Only show materials with this property available"
            )

            max_results = st.slider("Max results to show:", 10, 500, 100, 10)

        if st.button("🔍 Search Materials Project", type="primary", use_container_width=True):
            try:
                from mp_api.client import MPRester

                with st.spinner("Searching Materials Project..."):
                    with MPRester(api_key) as mpr:
                        # Build search criteria
                        search_criteria = {}

                        if search_mode == "Chemical System" and search_query:
                            search_criteria["chemsys"] = search_query
                        elif search_mode == "Elements" and search_query:
                            elements = [e.strip() for e in search_query.split(",")]
                            search_criteria["elements"] = elements
                        elif search_mode == "Formula" and search_query:
                            search_criteria["formula"] = search_query

                        # Apply filters
                        if filter_metallic:
                            search_criteria["band_gap"] = (0, 0.001)  # Nearly zero

                        if filter_stable:
                            search_criteria["is_stable"] = True

                        # Fetch summary data
                        docs = mpr.materials.summary.search(
                            **search_criteria,
                            fields=[
                                "material_id",
                                "formula_pretty",
                                "formation_energy_per_atom",
                                "energy_above_hull",
                                "band_gap",
                                "density",
                                "volume",
                                "nsites",
                                "is_stable",
                                "bulk_modulus",
                                "shear_modulus",
                                "total_magnetization"
                            ],
                            num_chunks=1,
                            chunk_size=max_results
                        )

                if not docs:
                    st.warning("No materials found matching these criteria.")
                    st.stop()

                st.success(f"✅ Found {len(docs)} materials")

                # Convert to DataFrame
                results_data = []
                for doc in docs:
                    row_data = {
                        "Material ID": doc.material_id,
                        "Formula": doc.formula_pretty,
                        "Formation Energy (eV/atom)": doc.formation_energy_per_atom if doc.formation_energy_per_atom else None,
                        "Energy Above Hull (eV/atom)": doc.energy_above_hull if doc.energy_above_hull else None,
                        "Band Gap (eV)": doc.band_gap if doc.band_gap else None,
                        "Density (g/cm³)": doc.density if doc.density else None,
                        "# Atoms": doc.nsites,
                        "Stable": "✓" if doc.is_stable else "✗"
                    }

                    # Add elastic properties if available
                    if hasattr(doc, 'bulk_modulus') and doc.bulk_modulus:
                        row_data["Bulk Modulus (GPa)"] = doc.bulk_modulus.vrh
                    else:
                        row_data["Bulk Modulus (GPa)"] = None

                    if hasattr(doc, 'shear_modulus') and doc.shear_modulus:
                        row_data["Shear Modulus (GPa)"] = doc.shear_modulus.vrh
                    else:
                        row_data["Shear Modulus (GPa)"] = None

                    # Add magnetic properties
                    if hasattr(doc, 'total_magnetization'):
                        row_data["Magnetization (μB)"] = doc.total_magnetization
                    else:
                        row_data["Magnetization (μB)"] = None

                    results_data.append(row_data)

                df_results = pd.DataFrame(results_data)

                # Display summary statistics
                st.markdown("### 📊 Search Results Summary")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Materials", len(df_results))
                with col2:
                    stable_count = df_results["Stable"].str.contains("✓").sum()
                    st.metric("Stable", stable_count)
                with col3:
                    avg_atoms = df_results["# Atoms"].mean()
                    st.metric("Avg Atoms", f"{avg_atoms:.1f}")
                with col4:
                    unique_formulas = df_results["Formula"].nunique()
                    st.metric("Unique Formulas", unique_formulas)

                # Property availability
                st.markdown("**Property Availability:**")
                prop_col1, prop_col2, prop_col3 = st.columns(3)

                with prop_col1:
                    fe_count = df_results["Formation Energy (eV/atom)"].notna().sum()
                    st.metric("Formation Energy", f"{fe_count}/{len(df_results)}")
                with prop_col2:
                    bg_count = df_results["Band Gap (eV)"].notna().sum()
                    st.metric("Band Gap", f"{bg_count}/{len(df_results)}")
                with prop_col3:
                    dens_count = df_results["Density (g/cm³)"].notna().sum()
                    st.metric("Density", f"{dens_count}/{len(df_results)}")

                # Display results table
                st.markdown("### 📋 Materials List")
                st.dataframe(
                    df_results,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )

                # Download option
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"materials_project_search_{search_mode.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )

                # Visualizations
                st.markdown("### 📈 Property Distributions")

                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    # Formation energy distribution
                    if df_results["Formation Energy (eV/atom)"].notna().any():
                        fig_fe = px.histogram(
                            df_results,
                            x="Formation Energy (eV/atom)",
                            nbins=30,
                            title="Formation Energy Distribution"
                        )
                        st.plotly_chart(fig_fe, use_container_width=True)

                with viz_col2:
                    # Band gap distribution
                    if df_results["Band Gap (eV)"].notna().any():
                        fig_bg = px.histogram(
                            df_results,
                            x="Band Gap (eV)",
                            nbins=30,
                            title="Band Gap Distribution"
                        )
                        st.plotly_chart(fig_bg, use_container_width=True)

                # Helpful tip
                st.info("""
                💡 **Next Step:** Use these search results to inform your data collection in the
                **Data Collection** tab. You can now collect materials matching these criteria!
                """)

            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                st.exception(e)

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

        # Training Templates
        st.markdown("### 📋 Quick Start Templates")

        templates = {
            "Custom (Manual Configuration)": {
                "description": "Configure all settings manually",
                "chemsys": "Fe-Ni",
                "max_materials": 500,
                "metallic": True,
                "stable": False,
                "properties": ['formation_energy_per_atom', 'band_gap', 'density'],
                "calphad": False,
                "dataset_name": "my_dataset"
            },
            "🔩 Steel Alloys (Fe-Ni-Cr)": {
                "description": "Iron-based structural alloys - metallic, formation energy focused",
                "chemsys": "Fe-Ni-Cr",
                "max_materials": 800,
                "metallic": True,
                "stable": False,
                "properties": ['formation_energy_per_atom', 'density', 'energy_above_hull', 'bulk_modulus', 'shear_modulus', 'total_magnetization'],
                "calphad": True,
                "dataset_name": "steel_alloys"
            },
            "✈️ Aerospace Materials (Ti-Al)": {
                "description": "Titanium-aluminum alloys - lightweight, high strength",
                "chemsys": "Ti-Al",
                "max_materials": 600,
                "metallic": True,
                "stable": True,
                "properties": ['formation_energy_per_atom', 'density', 'energy_above_hull', 'bulk_modulus', 'shear_modulus'],
                "calphad": True,
                "dataset_name": "ti_al_aerospace"
            },
            "🔋 Battery Materials (Li-Co-O)": {
                "description": "Lithium battery cathodes - stable oxides, band gap important",
                "chemsys": "Li-Co-O",
                "max_materials": 500,
                "metallic": False,
                "stable": True,
                "properties": ['formation_energy_per_atom', 'band_gap', 'energy_above_hull', 'density', 'volume'],
                "calphad": False,
                "dataset_name": "li_battery_materials"
            },
            "💎 Semiconductors (Si-Ge)": {
                "description": "Silicon-germanium semiconductors - band gap prediction focus",
                "chemsys": "Si-Ge",
                "max_materials": 400,
                "metallic": False,
                "stable": True,
                "properties": ['band_gap', 'formation_energy_per_atom', 'density', 'efermi', 'bulk_modulus'],
                "calphad": False,
                "dataset_name": "si_ge_semiconductors"
            },
            "🏗️ Structural Alloys (Al-Mg-Zn)": {
                "description": "Aluminum structural alloys - lightweight construction",
                "chemsys": "Al-Mg-Zn",
                "max_materials": 600,
                "metallic": True,
                "stable": False,
                "properties": ['formation_energy_per_atom', 'density', 'energy_above_hull', 'bulk_modulus', 'shear_modulus'],
                "calphad": True,
                "dataset_name": "al_structural"
            },
            "🌡️ High-Temperature Alloys (Ni-Cr-Mo)": {
                "description": "Nickel superalloys - high temperature applications",
                "chemsys": "Ni-Cr-Mo",
                "max_materials": 700,
                "metallic": True,
                "stable": False,
                "properties": ['formation_energy_per_atom', 'density', 'energy_above_hull', 'bulk_modulus', 'shear_modulus', 'total_magnetization'],
                "calphad": True,
                "dataset_name": "ni_superalloys"
            }
        }

        template_choice = st.selectbox(
            "Select template:",
            list(templates.keys()),
            help="Choose a pre-configured material system or customize manually"
        )

        template = templates[template_choice]

        if template_choice != "Custom (Manual Configuration)":
            st.info(f"📝 **{template_choice}**: {template['description']}")

        # Workflow mode
        st.markdown("---")
        st.markdown("### ⚡ Workflow Options")

        workflow_mode = st.radio(
            "Training workflow:",
            ["Manual (Step-by-step)", "🚀 Collect & Train (Automatic)", "🎯 Full Pipeline (Collect → Train → Evaluate)"],
            help="Choose workflow automation level"
        )

        if workflow_mode != "Manual (Step-by-step)":
            st.info(f"✨ **{workflow_mode}**: After data collection completes, training will start automatically with optimized settings.")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Dataset Configuration**")

            data_mode = st.radio(
                "Data selection:",
                ["Chemical System", "Multi-System (Diverse Dataset)", "Elements", "All Metallic"],
                help="How to filter materials"
            )

            chemical_systems = None  # For multi-system mode

            if data_mode == "Chemical System":
                chemsys = st.text_input(
                    "Chemical system:",
                    value=template["chemsys"],
                    help="e.g., Fe-Ni, Fe-Ni-Cr, Ti-Al"
                )
                elements = None
            elif data_mode == "Multi-System (Diverse Dataset)":
                st.markdown("**🌐 Collect from Multiple Alloy Systems**")
                st.info("Create large, diverse datasets by combining multiple chemical systems")

                # Preset multi-system collections
                preset_collections = {
                    "Transition Metal Alloys (5 systems)": ["Fe-Ni", "Co-Cr", "Ti-Al", "Cu-Zn", "Ni-Mo"],
                    "Steel Alloys (6 systems)": ["Fe-Ni", "Fe-Cr", "Fe-Mn", "Fe-Mo", "Fe-V", "Fe-C"],
                    "Lightweight Alloys (4 systems)": ["Al-Mg", "Al-Ti", "Ti-Al", "Mg-Zn"],
                    "Precious Metal Alloys (4 systems)": ["Au-Ag", "Au-Cu", "Pt-Pd", "Ag-Cu"],
                    "Refractory Alloys (5 systems)": ["Mo-W", "Ta-W", "Nb-Mo", "Ti-W", "V-Cr"],
                    "Custom (Manual Selection)": []
                }

                collection_choice = st.selectbox(
                    "Select preset or custom:",
                    list(preset_collections.keys()),
                    help="Choose a preset collection or define your own"
                )

                if collection_choice == "Custom (Manual Selection)":
                    systems_input = st.text_area(
                        "Enter chemical systems (one per line):",
                        value="Fe-Ni\nCo-Cr\nTi-Al\nCu-Zn",
                        help="Enter each chemical system on a new line",
                        height=150
                    )
                    chemical_systems = [s.strip() for s in systems_input.split("\n") if s.strip()]
                else:
                    chemical_systems = preset_collections[collection_choice]
                    st.success(f"✓ Selected {len(chemical_systems)} systems: {', '.join(chemical_systems)}")

                max_materials_per_system = st.slider(
                    "Materials per system:",
                    min_value=50,
                    max_value=1000,
                    value=300,
                    step=50,
                    help="Number of materials to fetch from each system"
                )

                total_estimate = len(chemical_systems) * max_materials_per_system
                st.info(f"📊 **Estimated total materials:** ~{total_estimate:,} (actual may vary)")

                chemsys = None
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
                value=template["max_materials"],
                step=50,
                help="More materials = better model, but longer download"
            )

        with col2:
            st.markdown("**Filtering Options**")

            metallic_only = st.checkbox("Metallic only (band gap = 0)", value=template["metallic"])
            stable_only = st.checkbox("Stable materials only", value=template["stable"])

            st.markdown("**Target Properties**")

            prediction_mode = st.radio(
                "Prediction mode:",
                ["Single Property", "Multi-Property (Recommended)"],
                help="Multi-property trains one model to predict multiple properties"
            )

            if prediction_mode == "Single Property":
                target_property = st.selectbox(
                    "Target property:",
                    ["formation_energy_per_atom", "band_gap", "energy_above_hull", "density"],
                    help="Property the model will learn to predict"
                )
                target_properties = [target_property]
            else:
                available_properties = {
                    'formation_energy_per_atom': 'Formation Energy (eV/atom)',
                    'energy_above_hull': 'Energy Above Hull (eV/atom)',
                    'band_gap': 'Band Gap (eV)',
                    'density': 'Density (g/cm³)',
                    'volume': 'Volume (ų/atom)',
                    'bulk_modulus': 'Bulk Modulus (GPa)',
                    'shear_modulus': 'Shear Modulus (GPa)',
                    'total_magnetization': 'Total Magnetization (μB)',
                    'efermi': 'Fermi Energy (eV)',
                }

                target_properties = st.multiselect(
                    "Select properties to predict:",
                    list(available_properties.keys()),
                    default=template["properties"],
                    format_func=lambda x: available_properties[x],
                    help="Model will predict all selected properties simultaneously"
                )

                if not target_properties:
                    st.warning("⚠️ Please select at least one property")
                else:
                    st.info(f"✓ Training model to predict {len(target_properties)} properties")

            st.markdown("**Advanced Features**")

            use_calphad = st.checkbox(
                "Enable enhanced elemental features",
                value=template["calphad"],
                help="Add thermodynamic and electronic properties (melting point, electronegativity, atomic radius, etc.)"
            )

            if use_calphad:
                st.info("ℹ️ **Enhanced features include:**\n"
                       "- Node features: 13D (atomic# + electronegativity, radius, ionization energy, melting point, heat capacity, etc.)\n"
                       "- Edge features: 2D (distance + mixing energy estimate)")

                # ESPEI TDB Integration
                with st.expander("🔬 ESPEI Thermodynamic Database (Optional - Advanced)", expanded=False):
                    st.markdown("""
                    **Enhance Feature Accuracy with ESPEI**

                    ESPEI can generate thermodynamic databases (TDB) from your collected data, providing
                    physically accurate mixing energies instead of empirical estimates.

                    **Benefits:**
                    - Real CALPHAD calculations instead of simple electronegativity models
                    - Thermodynamically consistent mixing energies
                    - Can improve model accuracy by 5-15% for alloy systems

                    **When to use:**
                    - Training on specific alloy systems (e.g., Fe-Ni, Co-Cr)
                    - Want maximum physical accuracy
                    - Have 50+ materials in your dataset
                    """)

                    espei_option = st.radio(
                        "ESPEI Integration:",
                        ["Skip (use default features)", "Upload existing TDB file", "Generate TDB automatically (requires ESPEI)"],
                        help="Choose how to provide thermodynamic data"
                    )

                    tdb_path = None

                    if espei_option == "Upload existing TDB file":
                        uploaded_tdb = st.file_uploader(
                            "Upload TDB file:",
                            type=['tdb', 'TDB'],
                            help="Upload a thermodynamic database file"
                        )

                        if uploaded_tdb:
                            import tempfile
                            # Save uploaded TDB to temp location
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.tdb') as tmp_file:
                                tmp_file.write(uploaded_tdb.getvalue())
                                tdb_path = tmp_file.name

                            st.success(f"✅ TDB uploaded: {uploaded_tdb.name}")

                            # Validate TDB
                            try:
                                from espei_workflow_manager import ESPEIWorkflowManager
                                manager = ESPEIWorkflowManager(Path("espei_output"))
                                tdb_info = manager.get_tdb_info(Path(tdb_path))

                                if tdb_info['valid']:
                                    st.info(f"**TDB Info:**\n"
                                           f"- Elements: {', '.join(tdb_info['elements'])}\n"
                                           f"- Phases: {', '.join(tdb_info['phases'])}")
                                else:
                                    st.error(f"⚠️ TDB validation failed: {tdb_info['message']}")
                                    tdb_path = None
                            except Exception as e:
                                st.warning(f"Could not validate TDB: {e}")

                    elif espei_option == "Generate TDB automatically (requires ESPEI)":
                        st.info("TDB will be generated after data collection using ESPEI")

                        # Store flag for TDB generation
                        if 'generate_tdb_after_collection' not in st.session_state:
                            st.session_state['generate_tdb_after_collection'] = False

                        st.session_state['generate_tdb_after_collection'] = True

                        # ESPEI configuration
                        phases_to_include = st.multiselect(
                            "Phases to include in TDB:",
                            ["LIQUID", "FCC_A1", "BCC_A2", "HCP_A3"],
                            default=["LIQUID", "FCC_A1"],
                            help="Select which crystal phases to model"
                        )

                        st.session_state['espei_phases'] = phases_to_include

                        st.warning("⏱️ TDB generation may take 2-5 minutes depending on dataset size")

            dataset_name = st.text_input(
                "Dataset name:",
                value=template["dataset_name"],
                help="Name for saving the dataset"
            )

        st.markdown("---")

        # Update button text based on workflow mode
        button_text = "🚀 Collect Data"
        if workflow_mode == "🚀 Collect & Train (Automatic)":
            button_text = "🚀 Collect Data & Start Training"
        elif workflow_mode == "🎯 Full Pipeline (Collect → Train → Evaluate)":
            button_text = "🎯 Run Full Pipeline"

        if st.button(button_text, type="primary", use_container_width=True):
            try:
                from gnn_data_collection import fetch_materials_data, fetch_multi_system_data, convert_to_graphs, get_dataset_statistics, print_dataset_info

                # Check if multi-system mode
                if chemical_systems is not None and len(chemical_systems) > 0:
                    # Multi-system data collection
                    with st.spinner(f"Fetching materials from {len(chemical_systems)} chemical systems..."):
                        df = fetch_multi_system_data(
                            api_key=api_key,
                            chemical_systems=chemical_systems,
                            max_materials_per_system=max_materials_per_system,
                            metallic_only=metallic_only,
                            stable_only=stable_only
                        )

                    if df.empty:
                        st.error("No materials found in any of the selected systems!")
                        st.stop()

                    st.success(f"✅ Fetched {len(df)} materials from {len(chemical_systems)} systems")

                    # Show system breakdown
                    if 'chemical_system' in df.columns:
                        st.markdown("**Materials by System:**")
                        system_counts = df['chemical_system'].value_counts()
                        breakdown_cols = st.columns(min(len(system_counts), 5))
                        for idx, (system, count) in enumerate(system_counts.items()):
                            with breakdown_cols[idx % 5]:
                                st.metric(system, count)
                else:
                    # Single-system data collection
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
                if len(target_properties) > 1:
                    # Multi-property mode
                    from gnn_data_collection_multitask import convert_to_multitask_graphs, print_property_summary

                    # Show property coverage
                    st.markdown("**Property Coverage in Dataset:**")
                    print_property_summary(df)

                    with st.spinner("Converting structures to multi-task graphs..."):
                        save_path = f"datasets/{dataset_name}_multitask.pkl"
                        graphs = convert_to_multitask_graphs(
                            df,
                            target_properties=target_properties,
                            cutoff=cutoff,
                            max_neighbors=max_neighbors,
                            use_calphad=use_calphad,
                            save_path=save_path
                        )

                    st.success(f"✅ Converted {len(graphs)} structures to multi-task graphs")
                    st.success(f"💾 Saved to {save_path}")
                    st.info(f"🎯 Each graph predicts {len(target_properties)} properties: {', '.join(target_properties)}")

                else:
                    # Single-property mode
                    with st.spinner("Converting structures to graphs..."):
                        save_path = f"datasets/{dataset_name}.pkl"
                        graphs = convert_to_graphs(
                            df,
                            target_property=target_properties[0],
                            cutoff=cutoff,
                            max_neighbors=max_neighbors,
                            use_calphad=use_calphad,
                            save_path=save_path
                        )

                    st.success(f"✅ Converted {len(graphs)} structures to graphs")
                    st.success(f"💾 Saved to {save_path}")

                if use_calphad:
                    st.success("🔬 Enhanced features enabled - graphs include elemental descriptors (electronegativity, melting point, etc.)!")

                # Show statistics
                st.markdown("**Dataset Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Graphs", len(graphs))
                with col2:
                    avg_nodes = sum(g.num_nodes for g in graphs) / len(graphs)
                    st.metric("Avg Nodes/Graph", f"{avg_nodes:.1f}")
                with col3:
                    avg_edges = sum(g.num_edges for g in graphs) / len(graphs)
                    st.metric("Avg Edges/Graph", f"{avg_edges:.1f}")
                with col4:
                    if len(target_properties) == 1:
                        target_values = [g.y.item() if g.y.dim() == 1 else g.y[0].item() for g in graphs]
                        st.metric("Target Mean", f"{sum(target_values)/len(target_values):.4f}")
                    else:
                        st.metric("Properties", len(target_properties))

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
                sample_cols = ["material_id", "formula"] + [p for p in target_properties if p in df.columns]
                sample_df = df[sample_cols].head(10)
                st.dataframe(sample_df, use_container_width=True)

                # Automatic workflow handling
                if workflow_mode != "Manual (Step-by-step)":
                    st.markdown("---")
                    st.markdown("### ⚡ Automatic Training Started")

                    if workflow_mode == "🎯 Full Pipeline (Collect → Train → Evaluate)":
                        st.info("📊 **Full Pipeline Mode**: Proceeding to training and evaluation automatically...")
                    else:
                        st.info("🚀 **Auto-Train Mode**: Proceeding to training automatically...")

                    # Auto-select best hyperparameters based on dataset size
                    dataset_size = len(graphs)
                    if dataset_size < 200:
                        auto_batch_size = 16
                        auto_epochs = 50
                    elif dataset_size < 500:
                        auto_batch_size = 32
                        auto_epochs = 100
                    else:
                        auto_batch_size = 32
                        auto_epochs = 150

                    st.info(f"🤖 **Auto-configured settings**: Batch size={auto_batch_size}, Epochs={auto_epochs}")

                    try:
                        from gnn_train import GNNTrainer
                        from gnn_dataset import CrystalGraphDataset, split_dataset, create_data_loaders
                        from model_manager import ModelManager, create_training_metadata
                        import time

                        # Initialize model manager
                        model_manager = ModelManager(checkpoint_dir="checkpoints")

                        with st.spinner("🔄 Loading dataset for training..."):
                            # Load the dataset we just created
                            if len(target_properties) > 1:
                                from gnn_data_collection_multitask import load_multitask_dataset
                                loaded_graphs = load_multitask_dataset(f"datasets/{dataset_name}_multitask.pkl")
                            else:
                                from gnn_data_collection import load_graph_dataset
                                loaded_graphs = load_graph_dataset(f"datasets/{dataset_name}.pkl")

                            dataset = CrystalGraphDataset(loaded_graphs)

                        # Split dataset
                        train_dataset, val_dataset, test_dataset = split_dataset(
                            dataset,
                            train_ratio=0.8,
                            val_ratio=0.1,
                            test_ratio=0.1
                        )

                        # Create data loaders
                        train_loader, val_loader, test_loader = create_data_loaders(
                            train_dataset, val_dataset, test_dataset,
                            batch_size=auto_batch_size
                        )

                        # Detect if multi-task
                        sample_batch = next(iter(train_loader))
                        is_multitask = False
                        if hasattr(sample_batch, 'y') and len(sample_batch.y.shape) > 0:
                            if len(sample_batch.y.shape) == 2 and sample_batch.y.shape[1] > 1:
                                is_multitask = True

                        # Create model
                        if is_multitask and len(target_properties) > 1:
                            from gnn_model_multitask import CGCNN_MultiTask_CALPHAD, CGCNN_MultiTask
                            from gnn_train_multitask import MultiTaskGNNTrainer

                            if use_calphad:
                                model = CGCNN_MultiTask_CALPHAD(
                                    input_node_dim=13,
                                    input_edge_dim=2,
                                    properties=target_properties
                                )
                            else:
                                model = CGCNN_MultiTask(properties=target_properties)

                            trainer = MultiTaskGNNTrainer(
                                model=model,
                                properties=target_properties,
                                device="cpu",
                                learning_rate=0.001,
                                checkpoint_dir="checkpoints"
                            )
                        else:
                            if use_calphad:
                                model = CGCNN_CALPHAD_Regressor(
                                    input_node_dim=13,
                                    input_edge_dim=2
                                )
                            else:
                                model = CGCNN()

                            trainer = GNNTrainer(
                                model=model,
                                device="cpu",
                                learning_rate=0.001,
                                checkpoint_dir="checkpoints"
                            )

                        st.success(f"✅ Model created with {count_parameters(model):,} parameters")

                        # Generate unique model name
                        unique_model_name = model_manager.generate_model_name(
                            dataset_name=dataset_name,
                            properties=target_properties,
                            use_calphad=use_calphad
                        )

                        # Train
                        st.markdown("### 📊 Training Progress")

                        start_time = time.time()

                        with st.spinner("🎓 Training model... This may take several minutes."):
                            history = trainer.train(
                                train_loader=train_loader,
                                val_loader=val_loader,
                                epochs=auto_epochs,
                                patience=20,
                                verbose=False,
                                model_name=unique_model_name
                            )

                        training_time = time.time() - start_time

                        st.success(f"✅ Training completed in {training_time/60:.1f} minutes!")

                        # Show results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Best Epoch", trainer.best_epoch)
                        with col2:
                            st.metric("Best Val Loss", f"{trainer.best_val_loss:.6f}")
                        with col3:
                            st.metric("Training Time", f"{training_time/60:.1f} min")

                        # Plot training history
                        fig = trainer.plot_training_history()
                        st.pyplot(fig)

                        # Evaluate and create metadata (even if not full pipeline)
                        test_metrics = None
                        if workflow_mode == "🎯 Full Pipeline (Collect → Train → Evaluate)":
                            with st.spinner("Evaluating on test set..."):
                                test_metrics, _, _ = trainer.evaluate(test_loader)

                        # Save model metadata
                        metadata = create_training_metadata(
                            dataset_name=dataset_name,
                            properties=target_properties,
                            num_samples=len(dataset),
                            best_epoch=trainer.best_epoch,
                            best_val_loss=trainer.best_val_loss,
                            training_time_minutes=training_time / 60,
                            use_calphad=use_calphad,
                            chemical_system=chemsys if chemsys else None,
                            test_metrics=test_metrics,
                            hyperparameters={
                                'batch_size': auto_batch_size,
                                'epochs': auto_epochs,
                                'learning_rate': 0.001,
                                'patience': 20
                            }
                        )

                        model_path = Path("checkpoints") / unique_model_name
                        model_manager.save_model_with_metadata(model_path, metadata)

                        st.success(f"✅ Model saved: `checkpoints/{unique_model_name}`")

                        # Display model summary
                        with st.expander("📋 Model Information"):
                            summary = model_manager.get_model_summary(unique_model_name)
                            if summary:
                                st.text(summary)

                        # Display evaluation results if full pipeline mode
                        if workflow_mode == "🎯 Full Pipeline (Collect → Train → Evaluate)" and test_metrics:
                            st.markdown("---")
                            st.markdown("### 🎯 Automatic Evaluation")

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Test MAE", f"{test_metrics['mae']:.6f}")
                            with col2:
                                st.metric("Test RMSE", f"{test_metrics['rmse']:.6f}")
                            with col3:
                                st.metric("R² Score", f"{test_metrics['r2']:.4f}")
                            with col4:
                                st.metric("Test Samples", test_metrics['num_samples'])

                            st.success("✅ **Pipeline Complete!** Your model is ready for predictions.")
                            st.info("💡 **Next Step:** Go to 'Prediction' mode to use your trained model!")

                    except Exception as train_error:
                        st.error(f"❌ Auto-training failed: {train_error}")
                        st.exception(train_error)
                        st.info("💡 You can still train manually in the 'Training' tab.")

                else:
                    # Manual mode - show next step message
                    if len(target_properties) > 1:
                        st.info(f"💡 **Next Step:** Train a multi-task model to predict all {len(target_properties)} properties!")
                    else:
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

        # Large dataset handling
        if total_materials > 800:
            st.warning(f"⚠️ **Large Dataset Detected:** {total_materials:,} materials")
            st.markdown("**Streamlit Cloud may timeout on large datasets.** Choose an option:")

            training_strategy = st.radio(
                "Training Strategy:",
                [
                    "🎯 Quick Train (Subsample)",
                    "⚡ Full Train (May Timeout)",
                    "💻 Download CLI Script"
                ],
                help="Choose how to handle large dataset training"
            )

            if training_strategy == "🎯 Quick Train (Subsample)":
                st.info("**Quick Train:** Use a stratified subsample for fast experimentation")

                subsample_size = st.slider(
                    "Subsample size:",
                    min_value=100,
                    max_value=min(1000, total_materials),
                    value=min(500, total_materials),
                    step=100,
                    help="Number of materials to use for training"
                )

                use_subsampling = True
                st.caption(f"Will train on {subsample_size} / {total_materials} materials ({subsample_size/total_materials*100:.1f}%)")

                # Estimate training time
                est_time_minutes = (subsample_size / 100) * 1.5  # Rough estimate: 1.5 min per 100 samples
                st.caption(f"⏱️ Estimated time: ~{est_time_minutes:.1f} minutes")

                if est_time_minutes > 10:
                    st.warning("⚠️ May still timeout. Consider reducing subsample size or using CLI script.")

            elif training_strategy == "⚡ Full Train (May Timeout)":
                st.warning("**Full Training:** This may timeout on Streamlit Cloud!")
                st.info("💡 **Tip:** Use the CLI script for reliable training on large datasets")
                use_subsampling = False
                subsample_size = total_materials

                # Estimate training time
                est_time_minutes = (total_materials / 100) * 1.5
                st.caption(f"⏱️ Estimated time: ~{est_time_minutes:.1f} minutes")

                if est_time_minutes > 15:
                    st.error(f"❌ Likely to timeout! ({est_time_minutes:.0f} min > 15 min limit)")

            else:  # Download CLI Script
                st.info("**CLI Training:** Download the script to train locally or on cloud")

                with open("train_large_dataset.py", 'r') as f:
                    cli_script = f.read()

                st.download_button(
                    label="📥 Download train_large_dataset.py",
                    data=cli_script,
                    file_name="train_large_dataset.py",
                    mime="text/x-python",
                    help="Download the standalone training script"
                )

                # Get the selected dataset path
                first_dataset_name = selected_datasets[0]
                ds_type, dataset_path = available_datasets[first_dataset_name]

                # Generate ready-to-run command
                st.markdown(f"""
                **How to use:**
                ```bash
                # Install dependencies (if needed)
                pip install torch torch-geometric pymatgen mp-api pycalphad

                # Train on your selected dataset: {first_dataset_name}
                python train_large_dataset.py --dataset {dataset_path} --epochs 150 --batch-size 32

                # With GPU (if available) - RECOMMENDED for faster training
                python train_large_dataset.py --dataset {dataset_path} --device cuda --epochs 150

                # See all options
                python train_large_dataset.py --help
                ```

                **Your dataset:** `{dataset_path}` ({total_materials:,} materials)

                **Benefits:**
                - ✅ No timeout limits
                - ✅ GPU support (faster training)
                - ✅ Progress tracking
                - ✅ Checkpoint resumption
                - ✅ Upload trained model back to Streamlit
                """)

                st.stop()  # Don't show training button if downloading CLI script
        else:
            use_subsampling = False
            subsample_size = total_materials

            # Estimate training time for normal datasets
            est_time_minutes = (total_materials / 100) * 1.5
            if est_time_minutes > 10:
                st.info(f"⏱️ Estimated training time: ~{est_time_minutes:.1f} minutes")
                if est_time_minutes > 15:
                    st.warning("⚠️ May timeout on Streamlit Cloud. Consider using the CLI script.")

        st.markdown("---")

        # Start training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            try:
                from gnn_train import GNNTrainer, print_evaluation_results
                from model_manager import ModelManager, create_training_metadata
                import time

                # Initialize model manager
                model_manager = ModelManager(checkpoint_dir="checkpoints")

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

                        # Apply subsampling if requested
                        if use_subsampling and len(graphs) > subsample_size:
                            import random
                            st.info(f"🎯 Subsampling {subsample_size} materials from {len(graphs)} total")

                            # Stratified sampling to maintain diversity
                            random.seed(42)  # Reproducibility
                            sampled_indices = random.sample(range(len(graphs)), subsample_size)
                            graphs = [graphs[i] for i in sorted(sampled_indices)]

                            st.success(f"✓ Sampled {len(graphs)} materials")

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
                    st.info("🔬 **Enhanced dataset detected!**")
                    st.info("   Node features: 13D (atomic# + elemental descriptors)")
                    st.info("   Edge features: 2D (distance + mixing energy)")

                if dataset_size < 10:
                    st.warning(f"⚠️ **Small dataset warning:** Only {dataset_size} samples. Recommend 100+ for reliable training.")

                # Detect if dataset is multi-task
                sample_batch = next(iter(train_loader))
                is_multitask = False
                target_properties = None

                if hasattr(sample_batch, 'y') and len(sample_batch.y.shape) > 0:
                    # Check if y has multiple dimensions (multi-task)
                    if len(sample_batch.y.shape) == 2 and sample_batch.y.shape[1] > 1:
                        is_multitask = True
                        # Try to get property names from batch
                        if hasattr(sample_batch, 'target_properties'):
                            target_properties = sample_batch.target_properties
                        else:
                            # Extract from first graph if available
                            if using_sharded or len(selected_datasets) > 1:
                                # For sharded datasets, need to peek at manifest or actual data
                                pass
                            else:
                                if hasattr(graphs[0], 'target_properties'):
                                    target_properties = graphs[0].target_properties

                if is_multitask and target_properties:
                    st.success(f"🎯 **Multi-task dataset detected!**")
                    st.info(f"   Predicting {len(target_properties)} properties: {', '.join(target_properties)}")

                # Create model (multi-task or single-task)
                if is_multitask and target_properties:
                    # Multi-task model
                    from gnn_model_multitask import CGCNN_MultiTask, CGCNN_MultiTask_CALPHAD

                    if has_calphad:
                        model = CGCNN_MultiTask_CALPHAD(
                            input_node_dim=13,
                            input_edge_dim=2,
                            node_feature_dim=node_dim,
                            edge_feature_dim=32,
                            hidden_dim=hidden_dim,
                            n_conv=n_conv,
                            n_hidden=n_hidden,
                            properties=target_properties
                        )
                        st.success("✅ Using Multi-Task CGCNN with enhanced features (13D node, 2D edge)")
                    else:
                        model = CGCNN_MultiTask(
                            node_feature_dim=node_dim,
                            edge_feature_dim=1,
                            hidden_dim=hidden_dim,
                            n_conv=n_conv,
                            n_hidden=n_hidden,
                            properties=target_properties
                        )
                        st.success("✅ Using Multi-Task CGCNN model")
                else:
                    # Single-task model
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
                        st.success("✅ Using CGCNN with enhanced elemental features (13D node, 2D edge)")
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

                # Create trainer (multi-task or single-task)
                device = "cpu"  # Streamlit Cloud uses CPU

                if is_multitask and target_properties:
                    from gnn_train_multitask import MultiTaskGNNTrainer
                    trainer = MultiTaskGNNTrainer(
                        model=model,
                        properties=target_properties,
                        device=device,
                        learning_rate=learning_rate,
                        checkpoint_dir="checkpoints"
                    )
                else:
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

                # Generate unique model name
                dataset_name_for_model = "_".join(selected_datasets) if len(selected_datasets) <= 3 else f"combined_{len(selected_datasets)}datasets"
                properties_for_model = target_properties if target_properties else ["formation_energy_per_atom"]

                unique_model_name = model_manager.generate_model_name(
                    dataset_name=dataset_name_for_model,
                    properties=properties_for_model,
                    use_calphad=has_calphad
                )

                start_time = time.time()

                # Train (this will block)
                with st.spinner("Training model... This may take several minutes."):
                    history = trainer.train(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=epochs,
                        patience=patience,
                        verbose=False,  # Don't print to console
                        model_name=unique_model_name
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
                test_metrics = None
                if not is_multitask:
                    st.markdown("### 🎯 Test Set Evaluation")
                    with st.spinner("Evaluating on test set..."):
                        test_metrics, predictions, targets = trainer.evaluate(test_loader)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Test MAE", f"{test_metrics['mae']:.6f}")
                    with col2:
                        st.metric("Test RMSE", f"{test_metrics['rmse']:.6f}")
                    with col3:
                        st.metric("R² Score", f"{test_metrics['r2']:.4f}")
                    with col4:
                        st.metric("Test Samples", test_metrics['num_samples'])

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

                # Save model metadata
                metadata = create_training_metadata(
                    dataset_name=dataset_name_for_model,
                    properties=properties_for_model,
                    num_samples=dataset_size,
                    best_epoch=trainer.best_epoch,
                    best_val_loss=trainer.best_val_loss,
                    training_time_minutes=training_time / 60,
                    use_calphad=has_calphad,
                    chemical_system=None,  # Not available in manual mode
                    test_metrics=test_metrics,
                    hyperparameters={
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'patience': patience,
                        'train_ratio': train_ratio,
                        'val_ratio': val_ratio
                    }
                )

                model_path = Path("checkpoints") / unique_model_name
                model_manager.save_model_with_metadata(model_path, metadata)

                st.success(f"✅ Model saved: `checkpoints/{unique_model_name}`")

                # Display model summary
                with st.expander("📋 Model Information"):
                    summary = model_manager.get_model_summary(unique_model_name)
                    if summary:
                        st.text(summary)

                st.info("💡 **Next Step:** Use the trained model in 'Prediction' mode!")

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
                        st.info("🔬 Enhanced dataset detected (13D node features)")

                    # Load model (use enhanced model if enhanced features detected)
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

    with tab4:
        st.header("📚 How Graph Neural Networks Learn from Crystal Structures")

        st.markdown("""
        This guide explains the AI training process in simple terms, perfect for high school students
        learning about machine learning and materials science!
        """)

        # Section 1: The Big Picture
        st.markdown("---")
        st.markdown("### 🎯 The Big Picture: What Are We Trying to Do?")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            **Goal:** Teach a computer to predict material properties (like strength, energy, conductivity)
            just by looking at how atoms are arranged in the material.

            **Why This Matters:**
            - Testing materials in real life is expensive and time-consuming
            - We can explore millions of potential materials virtually
            - Speeds up discovery of better batteries, stronger alloys, and new technologies

            **The Challenge:**
            - Materials are 3D structures with billions of atoms
            - Different materials have different numbers of atoms
            - Traditional AI (like recognizing cats in photos) doesn't work for 3D structures
            """)

        with col2:
            st.info("""
            **Real-World Analogy**

            Imagine trying to predict how tasty a recipe will be just by looking at the ingredient list
            and cooking instructions - without actually cooking it!

            That's what we're doing with materials.
            """)

        # Section 2: Representing Materials as Graphs
        st.markdown("---")
        st.markdown("### 🕸️ Step 1: Turning Materials into Graphs")

        st.markdown("""
        **The Problem:** Computers can't directly understand 3D crystal structures.

        **The Solution:** We represent materials as **graphs** (networks), like a social network!
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **In a Social Network:**
            - **Nodes** = People
            - **Edges** = Friendships
            - Each person has info (age, location, interests)
            - Connections show who knows who
            """)

        with col2:
            st.markdown("""
            **In a Crystal Structure:**
            - **Nodes** = Atoms
            - **Edges** = Atomic bonds/neighbors
            - Each atom has info (element type, position)
            - Connections show which atoms are nearby
            """)

        st.info("""
        **Example:** An iron-nickel alloy (Fe-Ni)
        - 🔴 Red nodes = Iron atoms
        - 🔵 Blue nodes = Nickel atoms
        - Lines between them = atoms that are neighbors (within 8 Angstroms)
        - Each node remembers: "I'm an Iron atom at position (2.5, 1.3, 4.1)"
        - Each edge remembers: "These two atoms are 2.8 Angstroms apart"
        """)

        # Section 3: The Training Process
        st.markdown("---")
        st.markdown("### 🎓 Step 2: Training the Neural Network")

        st.markdown("""
        Now that we have graphs, we need to teach the AI to predict properties. Here's how:
        """)

        # Substep 1
        st.markdown("#### 📚 1. Gathering Examples (Training Data)")
        st.markdown("""
        - We collect thousands of materials with **known** properties from databases (Materials Project)
        - Example: "Steel #1 has formation energy of -0.45 eV/atom"
        - Each material becomes a graph + its actual measured property
        - This is like a textbook with worked examples!
        """)

        # Substep 2
        st.markdown("#### 🧠 2. The Neural Network's Job")
        st.markdown("""
        Think of the neural network as a student learning to solve problems:

        **What it does:**
        1. **Looks at each atom** (node) and its features
        2. **Passes messages between neighboring atoms** - each atom learns from its neighbors
           - "Hey, I'm Iron and I'm next to a Nickel atom!"
           - This happens multiple times (we call these "convolutional layers")
        3. **Combines all information** to understand the whole structure
        4. **Makes a prediction** - "I think this material's formation energy is -0.42 eV/atom"

        **The Magic:** The network learns what patterns matter for each property!
        - For strength: Maybe it learns to look for grain boundaries
        - For conductivity: Maybe it learns to look for electron-rich regions
        - For energy: Maybe it learns to look at bond types
        """)

        # Substep 3
        st.markdown("#### 📏 3. Checking the Answers (Loss Calculation)")
        st.markdown("""
        After each prediction, we check how wrong the AI was:

        - **Prediction:** -0.42 eV/atom
        - **Actual answer:** -0.45 eV/atom
        - **Error:** 0.03 eV/atom (not too bad!)

        We calculate the **average error** across all materials. This is called the **"loss"**.

        **Goal:** Make the loss as small as possible!
        """)

        # Substep 4
        st.markdown("#### ⚙️ 4. Learning from Mistakes (Backpropagation)")
        st.markdown("""
        Here's where the "learning" happens:

        1. The AI figures out **which parts** of its calculation caused the error
        2. It makes **tiny adjustments** to its internal numbers (called "weights")
        3. These adjustments make it slightly better at the next prediction

        **Analogy:** Like a basketball player adjusting their shooting technique after missing a shot.
        Each adjustment is tiny, but after thousands of attempts, they become accurate!
        """)

        # Substep 5
        st.markdown("#### 🔄 5. Repeating (Epochs)")
        st.markdown("""
        The AI goes through the **entire dataset** multiple times (each pass is called an "epoch"):

        - **Epoch 1:** Loss = 0.850 (lots of mistakes!)
        - **Epoch 20:** Loss = 0.234 (getting better!)
        - **Epoch 50:** Loss = 0.089 (pretty good!)
        - **Epoch 100:** Loss = 0.042 (excellent!)

        **When to stop?** We stop when the loss stops improving (this prevents "overfitting" - memorizing
        instead of understanding).
        """)

        # Section 4: Testing
        st.markdown("---")
        st.markdown("### ✅ Step 3: Testing - Did It Really Learn?")

        st.markdown("""
        We use a **test set** - materials the AI has **never seen** during training!

        **Why?** To make sure it learned general patterns, not just memorized the training data.

        **Metrics we check:**
        - **MAE (Mean Absolute Error):** Average size of mistakes
          - Example: MAE = 0.05 eV/atom means predictions are typically off by ±0.05
        - **R² Score:** How well predictions match reality (1.0 = perfect, 0.0 = random guessing)
          - Example: R² = 0.92 means 92% accuracy!
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.success("""
            **Good Model:**
            - MAE: 0.05 eV/atom
            - R²: 0.92
            - Predictions close to actual values
            - Works on new materials
            """)

        with col2:
            st.error("""
            **Bad Model:**
            - MAE: 0.35 eV/atom
            - R²: 0.45
            - Predictions scattered randomly
            - Doesn't generalize well
            """)

        # Section 5: What Makes Our Approach Special
        st.markdown("---")
        st.markdown("### 🌟 What Makes Graph Neural Networks Special?")

        st.markdown("""
        **Traditional Machine Learning:**
        - Only uses composition: "This material is 50% Iron, 50% Nickel"
        - Ignores structure: Doesn't know **how** atoms are arranged
        - Limited accuracy

        **Graph Neural Networks (What We Use):**
        - Uses **both** composition **and** 3D structure
        - Knows which atoms are neighbors
        - Understands crystal geometry
        - Much better predictions!

        **Even Better - CALPHAD Features:**
        - We add thermodynamic data (melting point, heat capacity, mixing energy)
        - This is like giving the AI a "cheat sheet" with extra physics knowledge
        - Improves predictions by 10-20%!
        """)

        # Section 6: Real Example
        st.markdown("---")
        st.markdown("### 🔬 Real Example: Training on Fe-Ni Alloys")

        st.markdown("""
        Let's walk through a concrete example:

        **1. Collect Data:**
        - Download 1,000 Fe-Ni materials from Materials Project
        - Each has known formation energy (how stable it is)

        **2. Convert to Graphs:**
        - Each material becomes a graph with ~50-200 atoms (nodes)
        - Each atom connects to ~12 nearest neighbors (edges)

        **3. Split Data:**
        - Training: 800 materials (80%) - AI learns from these
        - Validation: 100 materials (10%) - Check progress during training
        - Test: 100 materials (10%) - Final grade after training

        **4. Train:**
        - Run for 100 epochs (go through training data 100 times)
        - Batch size: 32 (look at 32 materials at once)
        - Learning rate: 0.001 (how big each adjustment is)
        - Takes ~10 minutes on a laptop

        **5. Results:**
        - Training Loss: 0.035 eV/atom
        - Test MAE: 0.048 eV/atom
        - Test R²: 0.94 (94% accurate!)

        **6. Use It:**
        - Now we can predict formation energy for **any** Fe-Ni composition
        - No need to run expensive quantum simulations!
        - Can explore millions of potential alloys quickly
        """)

        # Section 7: Key Takeaways
        st.markdown("---")
        st.markdown("### 🎓 Key Takeaways for Students")

        st.markdown("""
        **Main Ideas:**

        1. **AI learns from examples**, just like students learn from textbooks
        2. **Graphs let us represent 3D structures** in a way computers can understand
        3. **Neural networks find patterns** humans might miss
        4. **Training = showing examples and correcting mistakes** thousands of times
        5. **Testing on unseen data** ensures the AI truly learned, not just memorized

        **Why This Matters:**

        - 🔋 **Batteries:** Find better electrode materials for electric cars
        - 🏗️ **Construction:** Design stronger, lighter building materials
        - 🌡️ **Climate:** Discover catalysts for carbon capture
        - 💻 **Electronics:** Create faster, more efficient semiconductors
        - 🚀 **Space:** Develop materials that withstand extreme conditions

        **You Can Do This Too!**

        This same approach works for:
        - Drug discovery (molecules as graphs)
        - Social network analysis (people as graphs)
        - Traffic optimization (intersections as graphs)
        - Protein folding (amino acids as graphs)

        The fundamental concept - representing complex systems as graphs and learning patterns -
        applies to countless fields!
        """)

        # Interactive Quiz
        st.markdown("---")
        st.markdown("### 🧪 Test Your Understanding!")

        with st.expander("❓ Quiz: Check if you understood the concepts"):
            q1 = st.radio(
                "1. What do 'nodes' represent in our crystal structure graphs?",
                ["Chemical bonds", "Atoms", "Properties", "Energies"],
                index=None
            )
            if q1 == "Atoms":
                st.success("✅ Correct! Each node is an individual atom.")
            elif q1:
                st.error("❌ Try again! Think about what the graph represents.")

            q2 = st.radio(
                "2. What is 'backpropagation'?",
                [
                    "Loading data backwards",
                    "The process where the AI adjusts its parameters to reduce errors",
                    "Testing the model",
                    "Converting structures to graphs"
                ],
                index=None
            )
            if q2 == "The process where the AI adjusts its parameters to reduce errors":
                st.success("✅ Correct! Backpropagation is how the network learns from mistakes.")
            elif q2:
                st.error("❌ Not quite. Backpropagation is about learning from errors.")

            q3 = st.radio(
                "3. Why do we use a separate test set?",
                [
                    "To save computational time",
                    "To make training faster",
                    "To verify the model works on materials it hasn't seen before",
                    "To increase the dataset size"
                ],
                index=None
            )
            if q3 == "To verify the model works on materials it hasn't seen before":
                st.success("✅ Correct! We need to ensure the model generalizes, not just memorizes.")
            elif q3:
                st.error("❌ Think about why we separate data. What are we testing?")

            q4 = st.radio(
                "4. What advantage do Graph Neural Networks have over traditional ML?",
                [
                    "They're faster",
                    "They use composition AND 3D atomic structure",
                    "They need less data",
                    "They're simpler"
                ],
                index=None
            )
            if q4 == "They use composition AND 3D atomic structure":
                st.success("✅ Correct! GNNs leverage structural information, not just composition.")
            elif q4:
                st.error("❌ The key advantage is about what information GNNs can use.")

elif model_mode == "🔮 Prediction":
    st.header("🔮 Material Property Prediction")

    st.markdown("""
    Predict material properties using your trained GNN model.
    Supports both single-property and multi-property models.
    """)

    # Check for trained models
    checkpoints_dir = Path("checkpoints")
    available_checkpoints = []

    if checkpoints_dir.exists():
        available_checkpoints = list(checkpoints_dir.glob("*.pt"))

    if not available_checkpoints:
        st.warning("⚠️ **No trained models found in checkpoints directory!**")
        st.info("**Options:**")
        st.markdown("""
        1. **Train a model** in the '🎓 Train Model' mode (select from sidebar), or
        2. **Upload a trained model** using the section below
        """)

        st.markdown("---")

        # Show upload section directly (no expander) when no models exist
        st.subheader("📤 Upload Model File")
        st.markdown("""
        Upload a trained model file (.pt) to use for predictions.
        The model will be saved to the checkpoints directory and become available for use.
        """)

        uploaded_model = st.file_uploader(
            "Choose a model file",
            type=['pt'],
            help="Upload a .pt PyTorch model file",
            key="model_uploader_no_models"
        )

        if uploaded_model is not None:
            # Show file info
            file_size_mb = len(uploaded_model.getvalue()) / (1024 * 1024)
            st.info(f"📄 File: {uploaded_model.name} ({file_size_mb:.2f} MB)")

            # Save button
            if st.button("💾 Save Model to Checkpoints", type="primary"):
                try:
                    # Ensure checkpoints directory exists
                    checkpoints_dir.mkdir(exist_ok=True)

                    # Save the uploaded file
                    model_save_path = checkpoints_dir / uploaded_model.name

                    with open(model_save_path, 'wb') as f:
                        f.write(uploaded_model.getvalue())

                    st.success(f"✅ Model saved successfully to: {model_save_path}")
                    st.info("🔄 Refresh the page (press F5 or Ctrl+R) to see the model in the dropdown.")
                    st.balloons()

                    # Update available checkpoints list
                    available_checkpoints = list(checkpoints_dir.glob("*.pt"))

                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")

        st.markdown("---")
        st.info("💡 **Tip:** After uploading a model, refresh the page to use it for predictions!")
        st.stop()

    # Model Browser/Manager
    with st.expander("📚 Model Library & Manager", expanded=False):
        from model_manager import ModelManager
        import json

        model_manager = ModelManager(checkpoint_dir="checkpoints")
        models = model_manager.list_models()

        st.markdown(f"**Total Models:** {len(models)}")

        if models:
            for idx, model_info in enumerate(models):
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])

                    with col1:
                        st.markdown(f"**{model_info['filename']}**")
                        if model_info['metadata']:
                            meta = model_info['metadata']
                            props = meta.get('properties', [])
                            if isinstance(props, list) and len(props) > 1:
                                st.caption(f"🎯 Multi-task: {', '.join(props[:3])}{'...' if len(props) > 3 else ''}")
                            else:
                                prop_name = props[0] if isinstance(props, list) else props
                                st.caption(f"🎯 Single-task: {prop_name}")

                            if 'dataset_name' in meta:
                                st.caption(f"📊 Dataset: {meta['dataset_name']}")

                    with col2:
                        st.caption(f"📅 {model_info['modified'][:10]}")
                        st.caption(f"💾 {model_info['size_mb']} MB")

                        if model_info['metadata']:
                            meta = model_info['metadata']
                            if 'test_metrics' in meta and meta['test_metrics']:
                                metrics = meta['test_metrics']
                                if 'r2' in metrics:
                                    st.caption(f"📈 R²: {metrics['r2']:.3f}")

                    with col3:
                        # Download button
                        model_path = Path(model_info['path'])
                        if model_path.exists():
                            with open(model_path, 'rb') as f:
                                model_bytes = f.read()

                            st.download_button(
                                label="💾",
                                data=model_bytes,
                                file_name=model_info['filename'],
                                mime="application/octet-stream",
                                key=f"download_{idx}",
                                help="Download model file"
                            )

                    # Expandable details
                    with st.expander(f"Details for {model_info['filename']}", expanded=False):
                        if model_info['metadata']:
                            meta = model_info['metadata']

                            detail_col1, detail_col2 = st.columns(2)

                            with detail_col1:
                                st.markdown("**Training Info:**")
                                st.write(f"• Samples: {meta.get('num_samples', 'N/A')}")
                                st.write(f"• Best Epoch: {meta.get('best_epoch', 'N/A')}")
                                st.write(f"• Best Val Loss: {meta.get('best_val_loss', 'N/A'):.6f}")
                                st.write(f"• Training Time: {meta.get('training_time_minutes', 'N/A'):.1f} min")
                                if 'use_calphad' in meta and meta['use_calphad']:
                                    st.write("• ⚗️ CALPHAD Features: Yes")

                            with detail_col2:
                                st.markdown("**Performance:**")
                                if 'test_metrics' in meta and meta['test_metrics']:
                                    metrics = meta['test_metrics']
                                    st.write(f"• MAE: {metrics.get('mae', 'N/A'):.6f}")
                                    st.write(f"• RMSE: {metrics.get('rmse', 'N/A'):.6f}")
                                    st.write(f"• R²: {metrics.get('r2', 'N/A'):.4f}")
                                else:
                                    st.write("No test metrics available")

                            if 'hyperparameters' in meta and meta['hyperparameters']:
                                st.markdown("**Hyperparameters:**")
                                hyper = meta['hyperparameters']
                                st.json(hyper)

                        else:
                            st.info("No metadata available for this model")

                    st.markdown("---")
        else:
            st.info("No models found in the checkpoints directory")

    # Model upload section
    with st.expander("📤 Upload Model File", expanded=False):
        st.markdown("""
        Upload a trained model file (.pt) to use for predictions.
        The model will be saved to the checkpoints directory and become available for selection.
        """)

        uploaded_model = st.file_uploader(
            "Choose a model file",
            type=['pt'],
            help="Upload a .pt PyTorch model file",
            key="model_uploader"
        )

        if uploaded_model is not None:
            # Show file info
            file_size_mb = len(uploaded_model.getvalue()) / (1024 * 1024)
            st.info(f"📄 File: {uploaded_model.name} ({file_size_mb:.2f} MB)")

            # Save button
            if st.button("💾 Save Model to Checkpoints", type="primary"):
                try:
                    # Ensure checkpoints directory exists
                    checkpoints_dir.mkdir(exist_ok=True)

                    # Save the uploaded file
                    model_save_path = checkpoints_dir / uploaded_model.name

                    # Check if file already exists
                    if model_save_path.exists():
                        st.warning(f"⚠️ Model '{uploaded_model.name}' already exists. Overwriting...")

                    with open(model_save_path, 'wb') as f:
                        f.write(uploaded_model.getvalue())

                    st.success(f"✅ Model saved successfully to: {model_save_path}")
                    st.info("🔄 Refresh the page to see the new model in the dropdown below.")

                    # Update available checkpoints list
                    available_checkpoints = list(checkpoints_dir.glob("*.pt"))

                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")

    st.markdown("---")

    # Model selection
    selected_checkpoint_name = st.selectbox(
        "Select trained model:",
        [f.name for f in available_checkpoints],
        help="Choose a trained model for predictions"
    )
    checkpoint_path = checkpoints_dir / selected_checkpoint_name

    # Detect model type
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    properties = checkpoint.get('properties', None)
    is_multitask_model = properties is not None and len(properties) > 1

    if is_multitask_model:
        st.success(f"✅ Multi-task model loaded: {selected_checkpoint_name}")
        st.info(f"🎯 Predicts {len(properties)} properties: {', '.join(properties)}")
    else:
        st.success(f"✅ Single-task model loaded: {selected_checkpoint_name}")
        st.info("🎯 Predicts: Formation Energy")

    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["🧪 Create from Composition", "🔍 Materials Project", "📤 Upload File"])

    with tab1:
        st.subheader("Create Structure from Composition")

        st.markdown("""
        Create a crystal structure from a custom composition and predict its properties.
        """)

        # Pre-configured material systems
        material_systems = {
            "Custom (Select Elements)": {
                "elements": None,
                "lattice": "FCC",
                "description": "Choose your own elements"
            },
            "🔩 Steel (Fe-Ni-Cr)": {
                "elements": ["Fe", "Ni", "Cr"],
                "lattice": "BCC",
                "description": "Iron-based structural alloy"
            },
            "✈️ Aerospace (Ti-Al-V)": {
                "elements": ["Ti", "Al", "V"],
                "lattice": "HCP",
                "description": "Titanium aerospace alloy"
            },
            "⚡ Battery (Li-Co-O)": {
                "elements": ["Li", "Co", "O"],
                "lattice": "FCC",
                "description": "Lithium cobalt oxide cathode"
            },
            "💎 Semiconductor (Si-Ge)": {
                "elements": ["Si", "Ge"],
                "lattice": "FCC",
                "description": "Silicon-germanium semiconductor"
            },
            "🏗️ Structural (Al-Mg-Zn)": {
                "elements": ["Al", "Mg", "Zn"],
                "lattice": "FCC",
                "description": "Aluminum structural alloy"
            },
            "🔥 High-Temp (Ni-Cr-Mo)": {
                "elements": ["Ni", "Cr", "Mo"],
                "lattice": "FCC",
                "description": "Nickel superalloy"
            }
        }

        system_choice = st.selectbox(
            "Material System:",
            list(material_systems.keys()),
            help="Choose a pre-configured system or select custom elements"
        )

        system_config = material_systems[system_choice]
        st.caption(system_config["description"])

        # Element selection
        if system_config["elements"] is None:
            # Custom element selection
            st.markdown("**Select Elements:**")

            # Common elements for alloys
            common_elements = ["Fe", "Ni", "Cr", "Ti", "Al", "Cu", "Mg", "Zn", "Co", "Mn",
                             "Si", "V", "Mo", "W", "Nb", "Ta", "Li", "O", "C", "N"]

            col1, col2 = st.columns(2)
            with col1:
                num_elements = st.radio("Number of Elements:", [2, 3], index=0)

            with col2:
                st.caption("Binary or ternary alloy")

            if num_elements == 2:
                col1, col2 = st.columns(2)
                with col1:
                    element1 = st.selectbox("Element 1:", common_elements, index=0)
                with col2:
                    element2 = st.selectbox("Element 2:", common_elements, index=1)
                elements = [element1, element2]
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    element1 = st.selectbox("Element 1:", common_elements, index=0)
                with col2:
                    element2 = st.selectbox("Element 2:", common_elements, index=1)
                with col3:
                    element3 = st.selectbox("Element 3:", common_elements, index=2)
                elements = [element1, element2, element3]
        else:
            elements = system_config["elements"]
            st.info(f"🧪 Elements: {' - '.join(elements)}")

        # Composition sliders
        st.markdown("**Composition:**")

        if len(elements) == 2:
            col1, col2 = st.columns(2)
            with col1:
                comp1 = st.slider(f"{elements[0]} Content (%)", 0, 100, 50, 5)
            comp2 = 100 - comp1
            with col2:
                st.metric(f"{elements[1]} Content (%)", comp2)
            compositions = [comp1, comp2]
            comp_formula = f"{elements[0]}{comp1}{elements[1]}{comp2}"
        else:  # 3 elements
            col1, col2, col3 = st.columns(3)
            with col1:
                comp1 = st.slider(f"{elements[0]} (%)", 0, 100, 33, 5)
            with col2:
                max_comp2 = 100 - comp1
                comp2 = st.slider(f"{elements[1]} (%)", 0, max_comp2, min(33, max_comp2), 5)
            comp3 = 100 - comp1 - comp2
            with col3:
                st.metric(f"{elements[2]} (%)", comp3)
            compositions = [comp1, comp2, comp3]
            comp_formula = f"{elements[0]}{comp1}{elements[1]}{comp2}{elements[2]}{comp3}"

        st.markdown(f"**Formula:** {comp_formula}")

        # Lattice settings
        col1, col2 = st.columns(2)
        with col1:
            if system_config["elements"] is None:
                lattice_type = st.selectbox("Crystal Structure", ["FCC", "BCC", "HCP"])
            else:
                default_lattice = system_config["lattice"]
                lattice_type = st.selectbox("Crystal Structure", ["FCC", "BCC", "HCP"],
                                           index=["FCC", "BCC", "HCP"].index(default_lattice))
        with col2:
            supercell_size = st.selectbox("Supercell Size", [1, 2, 3], index=1,
                                         help="Larger = more realistic but slower")

        if st.button("🚀 Predict Properties", type="primary", use_container_width=True):
            try:
                import numpy as np
                from streamlit_prediction_utils import load_model_and_predict, display_predictions

                with st.spinner("Creating structure..."):
                    # Create structure with appropriate lattice
                    a = 3.6  # Default lattice parameter

                    if lattice_type == "FCC":
                        lattice = Lattice.cubic(a)
                        base_positions = [
                            [0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.0],
                            [0.5, 0.0, 0.5],
                            [0.0, 0.5, 0.5]
                        ]
                    elif lattice_type == "BCC":
                        lattice = Lattice.cubic(a)
                        base_positions = [
                            [0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]
                        ]
                    else:  # HCP
                        c_a_ratio = 1.633
                        lattice = Lattice.hexagonal(a, a * c_a_ratio)
                        base_positions = [
                            [0.0, 0.0, 0.0],
                            [1/3, 2/3, 0.5]
                        ]

                    # Create supercell
                    positions = []
                    for i in range(supercell_size):
                        for j in range(supercell_size):
                            for k in range(supercell_size):
                                for pos in base_positions:
                                    new_pos = [
                                        (pos[0] + i) / supercell_size,
                                        (pos[1] + j) / supercell_size,
                                        (pos[2] + k) / supercell_size
                                    ]
                                    positions.append(new_pos)

                    # Assign elements based on composition
                    num_atoms = len(positions)
                    element_list = []
                    for elem, comp in zip(elements, compositions):
                        count = int(num_atoms * comp / 100)
                        element_list.extend([elem] * count)

                    # Fill remaining atoms with last element (rounding)
                    while len(element_list) < num_atoms:
                        element_list.append(elements[-1])

                    # Shuffle for random distribution
                    np.random.shuffle(element_list)
                    element_list = element_list[:num_atoms]  # Trim if over

                    # Create structure
                    supercell_lattice = Lattice(lattice.matrix * supercell_size)
                    structure = Structure(supercell_lattice, element_list, positions)

                st.success(f"✅ Created {lattice_type} structure with {len(structure)} atoms")

                # Make prediction using utility function
                with st.spinner("Making prediction..."):
                    predictions, is_multi, uses_calphad = load_model_and_predict(
                        structure,
                        checkpoint_path,
                        node_dim=node_dim,
                        hidden_dim=hidden_dim,
                        n_conv=n_conv,
                        n_hidden=n_hidden
                    )

                # Display predictions using utility function
                composition_info = f"{comp_formula} ({lattice_type})"
                display_predictions(
                    predictions,
                    formula=structure.composition.reduced_formula,
                    composition_info=composition_info
                )

                # Structure info
                with st.expander("📊 Structure Details"):
                    st.write(f"**Formula:** {structure.composition.formula}")
                    st.write(f"**Reduced Formula:** {structure.composition.reduced_formula}")
                    st.write(f"**Number of Atoms:** {len(structure)}")
                    st.write(f"**Lattice Type:** {lattice_type}")
                    st.write(f"**Lattice Parameter:** {a * supercell_size:.3f} Å")
                    st.write(f"**Volume:** {structure.volume:.2f} ų")
                    st.write(f"**Density:** {structure.density:.2f} g/cm³")
                    st.write(f"**CALPHAD Features:** {'Yes' if uses_calphad else 'No'}")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.exception(e)

    with tab2:
        st.subheader("Fetch from Materials Project")

        st.markdown("""
        Query Materials Project database by material ID and predict properties.
        """)

        # Check API key
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
            Add `MP_API_KEY` to your environment or `.env` file.
            Get your key at: https://next-gen.materialsproject.org/api
            """)
        else:
            material_id = st.text_input(
                "Material ID",
                placeholder="e.g., mp-149, mp-8636",
                help="Enter a Materials Project ID"
            )

            if material_id and st.button("🔍 Fetch and Predict", type="primary", use_container_width=True):
                try:
                    from mp_api.client import MPRester
                    from streamlit_prediction_utils import load_model_and_predict, display_predictions

                    with st.spinner(f"Fetching {material_id} from Materials Project..."):
                        with MPRester(api_key) as mpr:
                            structure = mpr.get_structure_by_material_id(material_id)

                    st.success(f"✅ Fetched structure: {structure.composition.reduced_formula}")

                    # Make prediction
                    with st.spinner("Making prediction..."):
                        predictions, is_multi, uses_calphad = load_model_and_predict(
                            structure,
                            checkpoint_path,
                            node_dim=node_dim,
                            hidden_dim=hidden_dim,
                            n_conv=n_conv,
                            n_hidden=n_hidden
                        )

                    # Display predictions
                    display_predictions(
                        predictions,
                        formula=structure.composition.reduced_formula,
                        composition_info=f"Material ID: {material_id}"
                    )

                    # Structure details
                    with st.expander("📊 Structure Details"):
                        st.write(f"**Material ID:** {material_id}")
                        st.write(f"**Full Formula:** {structure.composition.formula}")
                        st.write(f"**Number of Sites:** {len(structure)}")
                        st.write(f"**Space Group:** {structure.get_space_group_info()[0]} (#{structure.get_space_group_info()[1]})")
                        st.write(f"**Volume:** {structure.volume:.2f} ų")
                        st.write(f"**Density:** {structure.density:.2f} g/cm³")
                        st.write(f"**CALPHAD Features:** {'Yes' if uses_calphad else 'No'}")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.exception(e)

    with tab3:
        st.subheader("Upload Structure File")

        st.markdown("""
        Upload a crystal structure file (CIF or POSCAR format) and predict properties.
        """)

        uploaded_file = st.file_uploader(
            "Choose structure file",
            type=['cif', 'poscar', 'vasp', 'POSCAR'],
            help="Upload CIF or POSCAR format"
        )

        if uploaded_file is not None:
            try:
                import tempfile
                from streamlit_prediction_utils import load_model_and_predict, display_predictions

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Load structure
                with st.spinner("Loading structure..."):
                    structure = Structure.from_file(tmp_path)

                st.success(f"✅ Loaded structure: {structure.composition.reduced_formula}")

                if st.button("🚀 Predict Properties", type="primary", use_container_width=True):
                    # Make prediction
                    with st.spinner("Making prediction..."):
                        predictions, is_multi, uses_calphad = load_model_and_predict(
                            structure,
                            checkpoint_path,
                            node_dim=node_dim,
                            hidden_dim=hidden_dim,
                            n_conv=n_conv,
                            n_hidden=n_hidden
                        )

                    # Display predictions
                    display_predictions(
                        predictions,
                        formula=structure.composition.reduced_formula,
                        composition_info=f"Uploaded: {uploaded_file.name}"
                    )

                    # Structure details
                    with st.expander("📊 Structure Details"):
                        st.write(f"**Source:** {uploaded_file.name}")
                        st.write(f"**Full Formula:** {structure.composition.formula}")
                        st.write(f"**Number of Sites:** {len(structure)}")
                        st.write(f"**Volume:** {structure.volume:.2f} ų")
                        st.write(f"**Density:** {structure.density:.2f} g/cm³")
                        st.write(f"**CALPHAD Features:** {'Yes' if uses_calphad else 'No'}")

                # Cleanup
                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<strong>Digital Foundry Materials Science Toolkit</strong><br>
GNN Property Predictor | Powered by PyTorch Geometric & Pymatgen<br>
🧠 Leveraging 3D Crystal Structure for Accurate Property Prediction
</div>
""", unsafe_allow_html=True)
