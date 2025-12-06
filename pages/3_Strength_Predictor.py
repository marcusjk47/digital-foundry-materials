"""
Strength Predictor

Hybrid GNN + Physics-based strength prediction combining:
- GNN-predicted elastic properties from crystal structure
- Hall-Petch grain boundary strengthening
- Temperature-dependent corrections
- Processing history effects

Author: Digital Foundry Materials Science Toolkit
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import modules
try:
    from strength_predictor import StrengthPredictor
    from pymatgen.core import Structure, Lattice
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="Strength Predictor",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 10px 0;
    }
    .result-box {
        background-color: #f1f8e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #8bc34a;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("💪 Mechanical Strength Predictor")
st.markdown("### Hybrid GNN + Physics-Based Strength Prediction")

# Check modules
if not MODULES_AVAILABLE:
    st.error(f"Error loading modules: {IMPORT_ERROR}")
    st.stop()

# Sidebar configuration
st.sidebar.title("⚙️ Prediction Settings")

st.sidebar.markdown("---")

# Model selection
model_path = st.sidebar.text_input(
    "Elastic Model Path (optional):",
    value="",
    help="Path to trained elastic property model checkpoint (.pt file)"
)

use_calphad = st.sidebar.checkbox(
    "Use CALPHAD Features",
    value=True,
    help="Enable CALPHAD thermodynamic feature enhancement"
)

st.sidebar.markdown("---")

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Quick Prediction", "📊 Sensitivity Analysis", "ℹ️ About"])

with tab1:
    st.header("Quick Strength Prediction")

    st.markdown("""
    <div class="info-box">
    <strong>How it works:</strong>
    <ol>
        <li>Enter crystal structure (composition or upload CIF/POSCAR)</li>
        <li>Specify microstructure parameters (grain size, temperature)</li>
        <li>GNN predicts elastic properties from structure</li>
        <li>Empirical models calculate strength from elastic properties + microstructure</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    # Input method
    input_method = st.radio(
        "Input method:",
        ["Simple Composition", "Upload Structure File", "Example Structures"],
        horizontal=True
    )

    structure = None

    if input_method == "Simple Composition":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Enter Composition**")
            formula = st.text_input(
                "Chemical formula:",
                value="Fe50Ni50",
                help="e.g., Fe50Ni50, Ti6Al4V, etc."
            )

            crystal_system = st.selectbox(
                "Crystal structure:",
                ["FCC", "BCC", "HCP"],
                help="Crystal structure type"
            )

            lattice_param = st.number_input(
                "Lattice parameter (Angstrom):",
                min_value=2.0,
                max_value=6.0,
                value=3.6,
                step=0.1
            )

        with col2:
            st.markdown("**Microstructure Parameters**")

            grain_size = st.number_input(
                "Grain size (micrometers):",
                min_value=0.1,
                max_value=1000.0,
                value=10.0,
                step=1.0,
                help="Average grain diameter"
            )

            temperature = st.number_input(
                "Temperature (K):",
                min_value=0.0,
                max_value=2000.0,
                value=300.0,
                step=50.0
            )

            processing = st.selectbox(
                "Processing condition:",
                ["annealed", "cold_worked", "quenched"],
                help="Thermomechanical processing history"
            )

        # Create simple structure
        if st.button("🚀 Predict Strength", type="primary", use_container_width=True):
            try:
                # Create structure (simplified - real implementation would parse formula properly)
                if crystal_system == "FCC":
                    lattice = Lattice.cubic(lattice_param)
                    structure = Structure(lattice, ["Fe", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])
                elif crystal_system == "BCC":
                    lattice = Lattice.cubic(lattice_param)
                    structure = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
                else:  # HCP
                    lattice = Lattice.hexagonal(lattice_param, lattice_param * 1.633)
                    structure = Structure(lattice, ["Ti"], [[0, 0, 0]])

                # Initialize predictor
                with st.spinner("Loading models..."):
                    predictor = StrengthPredictor(
                        elastic_model_path=model_path if model_path else None,
                        use_calphad=use_calphad,
                        device="cpu"
                    )

                # Predict
                with st.spinner("Predicting strength..."):
                    prediction = predictor.predict_strength(
                        structure=structure,
                        grain_size=grain_size,
                        temperature=temperature,
                        processing=processing
                    )

                # Display results
                st.markdown("---")
                st.markdown("## 📋 Prediction Results")

                # Main strength metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Yield Strength", f"{prediction['yield_strength_MPa']:.0f} MPa")
                with col2:
                    st.metric("Tensile Strength", f"{prediction['tensile_strength_MPa']:.0f} MPa")
                with col3:
                    st.metric("Safety Factor", f"{prediction['safety_factor']:.1f}x")
                with col4:
                    st.metric("Shear Modulus", f"{prediction['shear_modulus_GPa']:.1f} GPa")

                # Detailed breakdown
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Elastic Properties**")
                    elastic_df = pd.DataFrame({
                        "Property": ["Shear Modulus (G)", "Bulk Modulus (K)", "Young's Modulus (E)", "Poisson's Ratio"],
                        "Value": [
                            f"{prediction['shear_modulus_GPa']:.2f} GPa",
                            f"{prediction['bulk_modulus_GPa']:.2f} GPa",
                            f"{prediction['youngs_modulus_GPa']:.2f} GPa",
                            f"{prediction['poisson_ratio']:.3f}"
                        ]
                    })
                    st.dataframe(elastic_df, use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("**Strength Contributions**")
                    contrib_df = pd.DataFrame({
                        "Mechanism": ["Friction Stress", "Grain Boundary (Hall-Petch)", "Temperature Factor", "Processing Factor"],
                        "Value": [
                            f"{prediction['friction_stress_MPa']:.1f} MPa",
                            f"{prediction['hall_petch_contribution_MPa']:.1f} MPa",
                            f"{prediction['temperature_factor']:.3f}x",
                            f"{prediction['processing_factor']:.2f}x"
                        ]
                    })
                    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

                # Visualization
                st.markdown("**Strength Breakdown**")

                # Pie chart of contributions
                contributions = {
                    'Friction Stress': prediction['friction_stress_MPa'],
                    'Hall-Petch': prediction['hall_petch_contribution_MPa']
                }

                fig = go.Figure(data=[go.Pie(
                    labels=list(contributions.keys()),
                    values=list(contributions.values()),
                    hole=.3
                )])
                fig.update_layout(title_text="Strengthening Mechanisms (at Room Temperature)")
                st.plotly_chart(fig, use_container_width=True)

                # Notes
                st.markdown("""
                <div class="result-box">
                <strong>⚠️ Important Notes:</strong>
                <ul>
                    <li>Predictions are estimates based on untrained model (random weights)</li>
                    <li>Train the elastic model on Materials Project data for accurate predictions</li>
                    <li>Actual strength depends on defect density, texture, and other factors</li>
                    <li>Use for comparative analysis and screening, not for design specifications</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)

    elif input_method == "Example Structures":
        st.info("📚 Example structures coming soon! Use 'Simple Composition' for now.")

    else:  # Upload file
        st.info("📁 File upload coming soon! Use 'Simple Composition' for now.")

with tab2:
    st.header("Sensitivity Analysis")

    st.markdown("""
    Analyze how strength varies with microstructure parameters.
    """)

    # Similar structure input
    st.markdown("**Structure (simplified)**")
    col1, col2 = st.columns(2)
    with col1:
        lattice_param_sens = st.number_input("Lattice parameter:", value=3.6, key="sens_lattice")
    with col2:
        crystal_sys_sens = st.selectbox("Crystal system:", ["FCC", "BCC"], key="sens_crystal")

    # Analysis type
    analysis_type = st.radio(
        "Analysis type:",
        ["Grain Size", "Temperature", "Both"],
        horizontal=True
    )

    if st.button("🔍 Run Analysis", type="primary"):
        try:
            # Create structure
            lattice = Lattice.cubic(lattice_param_sens)
            if crystal_sys_sens == "FCC":
                structure = Structure(lattice, ["Fe", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            else:
                structure = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])

            # Initialize predictor
            predictor = StrengthPredictor(use_calphad=use_calphad, device="cpu")

            if analysis_type == "Grain Size":
                # Vary grain size
                grain_sizes = [1, 2, 5, 10, 20, 50, 100]
                strengths = []

                for d in grain_sizes:
                    pred = predictor.predict_strength(structure, grain_size=d, temperature=300.0)
                    strengths.append(pred['yield_strength_MPa'])

                # Plot
                fig = px.line(x=grain_sizes, y=strengths,
                             labels={'x': 'Grain Size (μm)', 'y': 'Yield Strength (MPa)'},
                             title='Hall-Petch Relationship: Strength vs Grain Size')
                fig.update_traces(mode='lines+markers')
                st.plotly_chart(fig, use_container_width=True)

            elif analysis_type == "Temperature":
                # Vary temperature
                temperatures = [300, 400, 500, 600, 800, 1000, 1200]
                strengths = []

                for T in temperatures:
                    pred = predictor.predict_strength(structure, grain_size=10.0, temperature=T)
                    strengths.append(pred['yield_strength_MPa'])

                # Plot
                fig = px.line(x=temperatures, y=strengths,
                             labels={'x': 'Temperature (K)', 'y': 'Yield Strength (MPa)'},
                             title='Temperature Dependence of Strength')
                fig.update_traces(mode='lines+markers')
                st.plotly_chart(fig, use_container_width=True)

            else:  # Both
                # 2D heatmap
                grain_sizes = [1, 5, 10, 20, 50]
                temperatures = [300, 500, 800, 1000]

                strength_matrix = []
                for T in temperatures:
                    row = []
                    for d in grain_sizes:
                        pred = predictor.predict_strength(structure, grain_size=d, temperature=T)
                        row.append(pred['yield_strength_MPa'])
                    strength_matrix.append(row)

                fig = go.Figure(data=go.Heatmap(
                    z=strength_matrix,
                    x=grain_sizes,
                    y=temperatures,
                    colorscale='Viridis'
                ))
                fig.update_layout(
                    title='Strength Map: Grain Size vs Temperature',
                    xaxis_title='Grain Size (μm)',
                    yaxis_title='Temperature (K)'
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)

with tab3:
    st.header("About Strength Prediction")

    st.markdown("""
    ### Methodology

    This tool uses a **hybrid approach** combining machine learning and physics-based models:

    #### 1. GNN Elastic Property Prediction
    - Crystal structure → Graph Neural Network → Elastic moduli (G, K, E)
    - Trained on Materials Project elastic tensor data
    - Accounts for atomic structure and bonding

    #### 2. Hall-Petch Strengthening
    ```
    σ_y = σ_0 + k_y / √d
    ```
    where:
    - σ_y = yield strength
    - σ_0 = friction stress (from GNN)
    - k_y = Hall-Petch coefficient (from GNN)
    - d = grain size (user input)

    #### 3. Temperature Correction
    - Empirical temperature dependence
    - Strength decreases with increasing temperature
    - Accounts for thermal activation of dislocations

    #### 4. Processing Effects
    - **Annealed**: 1.0x (baseline)
    - **Cold Worked**: 1.5x (work hardening)
    - **Quenched**: 1.3x (retained defects)

    ### Limitations

    ⚠️ **This tool provides estimates, not exact values**

    Actual strength depends on many factors not captured:
    - Dislocation density
    - Precipitate distribution
    - Texture and crystallographic orientation
    - Residual stresses
    - Surface finish and defects

    ### Training the Elastic Model

    For accurate predictions:
    1. Collect elastic property data from Materials Project
    2. Train the multi-task GNN model
    3. Provide path to trained checkpoint

    Without training, the model uses random weights (for demonstration only).

    ### References

    - Hall, E.O. (1951). "The Deformation and Ageing of Mild Steel"
    - Petch, N.J. (1953). "The Cleavage Strength of Polycrystals"
    - Materials Project: materialsproject.org
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**📖 Documentation**")
st.sidebar.markdown("[User Guide](https://docs.claude.com)")
st.sidebar.markdown("[API Reference](https://docs.materialsproject.org)")
