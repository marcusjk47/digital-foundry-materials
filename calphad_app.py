"""
CALPHAD Phase Diagram Calculator
Interactive web application for thermodynamic calculations using PyCalphad
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="CALPHAD Phase Diagrams",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if PyCalphad is installed
try:
    from pycalphad import Database, equilibrium, binplot
    import pycalphad.variables as v
    PYCALPHAD_AVAILABLE = True
except ImportError:
    PYCALPHAD_AVAILABLE = False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Try to import optional packages
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scheil import simulate_scheil_solidification
    SCHEIL_AVAILABLE = True
except ImportError:
    SCHEIL_AVAILABLE = False


# Sidebar navigation
st.sidebar.title("🔥 CALPHAD Tools")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📁 Load Database", "📊 Binary Phase Diagram",
     "⚖️ Equilibrium Calculator", "💪 Temperature-Property Curves",
     "🔗 Batch MP Analysis", "🔬 Create TDB from MP Data", "❄️ Scheil Simulation"]
)

st.sidebar.markdown("---")

# MP Data Loader in Sidebar
st.sidebar.markdown("### 📊 Materials Project Data")

mp_data_file = st.sidebar.file_uploader(
    "Load MP CSV (optional):",
    type=['csv'],
    help="Upload CSV from Materials Project to easily select compositions",
    key="sidebar_mp_upload"
)

if mp_data_file is not None:
    try:
        mp_data_df = pd.read_csv(mp_data_file)
        st.session_state['mp_data'] = mp_data_df
        st.session_state['mp_data_loaded'] = True
        st.sidebar.success(f"✅ {len(mp_data_df)} materials loaded")

        # Show quick info
        if 'formula' in mp_data_df.columns:
            st.sidebar.caption(f"Example: {mp_data_df['formula'].iloc[0]}")

        # Show available columns (helpful for debugging)
        with st.sidebar.expander("📋 Columns in CSV"):
            st.caption(', '.join(mp_data_df.columns.tolist()[:10]))
            if len(mp_data_df.columns) > 10:
                st.caption(f"...and {len(mp_data_df.columns) - 10} more")

    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
        st.session_state['mp_data_loaded'] = False
else:
    if 'mp_data_loaded' in st.session_state:
        st.sidebar.info(f"Using previously loaded data ({len(st.session_state.get('mp_data', []))} materials)")
    else:
        st.session_state['mp_data_loaded'] = False

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About CALPHAD
**CAL**culation of **PHAD**e **D**iagrams

A computational method for predicting:
- Phase stability
- Phase diagrams
- Thermodynamic properties
- Solidification behavior
""")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_installation():
    """Check if PyCalphad is properly installed"""
    if not PYCALPHAD_AVAILABLE:
        st.error("⚠️ PyCalphad is not installed!")
        st.markdown("""
        ### Installation Required

        Please install PyCalphad first:

        **Option 1: Use the install script**
        ```
        install_calphad.bat
        ```

        **Option 2: Manual install with conda**
        ```bash
        conda install -c conda-forge pycalphad
        ```

        **Option 3: Manual install with pip**
        ```bash
        pip install pycalphad
        ```

        Then restart this app.
        """)
        return False
    return True


def get_available_tdb_files():
    """Scan for available TDB files in the calphad_databases folder"""
    db_folder = Path("calphad_databases")
    if not db_folder.exists():
        db_folder.mkdir(parents=True, exist_ok=True)
        return []

    tdb_files = list(db_folder.glob("*.tdb")) + list(db_folder.glob("*.TDB"))
    return [str(f) for f in tdb_files]


def load_database(tdb_path):
    """Load a thermodynamic database"""
    try:
        db = Database(tdb_path)
        return db, None
    except Exception as e:
        return None, str(e)


def save_uploaded_tdb(uploaded_file):
    """Save uploaded TDB file to calphad_databases folder"""
    db_folder = Path("calphad_databases")
    db_folder.mkdir(parents=True, exist_ok=True)

    save_path = db_folder / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(save_path)


def get_composition_from_mp_data(row, db_elements):
    """Extract composition from MP data row - case insensitive"""
    composition = {}

    # Create case-insensitive mapping of row columns
    row_columns_lower = {col.lower(): col for col in row.index}

    # Try to find composition columns (case-insensitive)
    for elem in db_elements:
        elem_lower = elem.lower()

        # Try exact match first
        if elem in row.index and pd.notna(row[elem]) and row[elem] > 0:
            composition[elem] = float(row[elem])
        # Try case-insensitive match
        elif elem_lower in row_columns_lower:
            actual_col = row_columns_lower[elem_lower]
            if pd.notna(row[actual_col]) and row[actual_col] > 0:
                composition[elem] = float(row[actual_col])

    # Normalize to sum = 1.0
    if len(composition) > 0:
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}

    return composition


# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    st.title("🔥 CALPHAD Phase Diagram Calculator")
    st.markdown("### Interactive Thermodynamic Calculations with PyCalphad")

    if not check_installation():
        st.stop()

    st.success("✅ PyCalphad is installed and ready!")

    # Display PyCalphad version
    import pycalphad
    st.info(f"📦 PyCalphad version: {pycalphad.__version__}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 What Can You Do?

        **📊 Binary Phase Diagrams**
        - Calculate and visualize 2-component phase diagrams
        - Explore temperature vs composition relationships
        - Identify phase boundaries and stability regions

        **⚖️ Equilibrium Calculations**
        - Calculate equilibrium at specific T and composition
        - Determine stable phases and phase fractions
        - Get thermodynamic properties (G, H, S, Cp)

        **❄️ Scheil Solidification**
        - Simulate non-equilibrium solidification
        - Predict microsegregation
        - Calculate solidification temperature range
        """)

    with col2:
        st.markdown("""
        ### 📚 Getting Started

        **Step 1: Load Database**
        - Upload a TDB file or use provided examples
        - TDB files contain thermodynamic data for elements and phases

        **Step 2: Select System**
        - Choose elements for calculation
        - Select phases to include

        **Step 3: Calculate**
        - Set temperature and composition ranges
        - Run calculation and visualize results

        **Step 4: Analyze**
        - Explore phase diagrams interactively
        - Export results and figures
        """)

    st.markdown("---")

    # Check for available databases
    tdb_files = get_available_tdb_files()

    if len(tdb_files) > 0:
        st.success(f"✅ Found {len(tdb_files)} thermodynamic database(s)")

        with st.expander("📁 Available Databases"):
            for tdb in tdb_files:
                db_name = Path(tdb).name
                st.markdown(f"- `{db_name}`")
    else:
        st.warning("⚠️ No TDB files found in `calphad_databases/` folder")
        st.info("💡 Upload a TDB file in the 'Load Database' page or download from NIMS database")

    st.markdown("---")

    st.markdown("""
    ### 📖 Resources

    - **Setup Guide**: `CALPHAD_SETUP_GUIDE.md`
    - **Quick Start**: `CALPHAD_QUICK_START.md`
    - **PyCalphad Docs**: [pycalphad.org/docs](https://pycalphad.org/docs/)
    - **TDB Databases**: [NIMS CPDDB](https://cpddb.nims.go.jp/)
    - **CALPHAD Journal**: [Elsevier](https://www.sciencedirect.com/journal/calphad)
    """)


# ============================================================================
# PAGE: LOAD DATABASE
# ============================================================================

elif page == "📁 Load Database":
    st.title("📁 Load Thermodynamic Database")

    if not check_installation():
        st.stop()

    st.markdown("""
    Load a TDB (Thermodynamic DataBase) file to enable phase diagram calculations.

    **TDB files contain:**
    - Element definitions and reference states
    - Phase definitions and crystal structures
    - Gibbs energy functions for each phase
    - Interaction parameters for solutions
    """)

    st.markdown("---")

    # Two methods: Select existing or upload new
    method = st.radio("Choose method:", ["📂 Select Existing Database", "📤 Upload New Database"])

    if method == "📂 Select Existing Database":
        tdb_files = get_available_tdb_files()

        if len(tdb_files) == 0:
            st.warning("⚠️ No TDB files found in `calphad_databases/` folder")
            st.info("💡 Switch to 'Upload New Database' to add a TDB file")
        else:
            st.success(f"Found {len(tdb_files)} database(s)")

            # Let user select
            selected_tdb = st.selectbox(
                "Select database:",
                options=tdb_files,
                format_func=lambda x: Path(x).name
            )

            if st.button("🔍 Load and Inspect Database"):
                with st.spinner("Loading database..."):
                    db, error = load_database(selected_tdb)

                if error:
                    st.error(f"❌ Error loading database: {error}")
                else:
                    st.success("✅ Database loaded successfully!")

                    # Store in session state
                    st.session_state['current_db'] = db
                    st.session_state['current_db_path'] = selected_tdb
                    st.session_state['current_db_name'] = Path(selected_tdb).name

                    # Display database info
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🧪 Elements")
                        elements = sorted([el for el in db.elements if el != 'VA'])
                        for el in elements:
                            st.markdown(f"- **{el}**")

                    with col2:
                        st.markdown("### 🔷 Phases")
                        phases = sorted(db.phases.keys())
                        for phase in phases:
                            constituents = db.phases[phase].constituents
                            st.markdown(f"- **{phase}**")
                            st.markdown(f"  `{constituents}`")

                    st.markdown("---")
                    st.info(f"📁 Database path: `{selected_tdb}`")
                    st.success("✅ Ready for calculations! Go to 'Binary Phase Diagram' or 'Equilibrium Calculator'")

    else:  # Upload new database
        st.markdown("### 📤 Upload TDB File")

        uploaded_file = st.file_uploader(
            "Choose a TDB file",
            type=['tdb', 'TDB'],
            help="Upload a thermodynamic database file in TDB format"
        )

        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")

            # Preview first 50 lines
            with st.expander("👀 Preview File Contents"):
                content = uploaded_file.getvalue().decode('utf-8')
                lines = content.split('\n')[:50]
                st.code('\n'.join(lines), language='text')

            if st.button("💾 Save and Load Database"):
                # Save file
                save_path = save_uploaded_tdb(uploaded_file)
                st.success(f"✅ Saved to: `{save_path}`")

                # Try to load it
                with st.spinner("Loading database..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    db, error = load_database(save_path)

                if error:
                    st.error(f"❌ Error loading database: {error}")
                    st.warning("The file was saved but could not be parsed. Check TDB format.")
                else:
                    st.success("✅ Database loaded successfully!")

                    # Store in session state
                    st.session_state['current_db'] = db
                    st.session_state['current_db_path'] = save_path
                    st.session_state['current_db_name'] = uploaded_file.name

                    # Display database info
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🧪 Elements")
                        elements = sorted([el for el in db.elements if el != 'VA'])
                        for el in elements:
                            st.markdown(f"- **{el}**")

                    with col2:
                        st.markdown("### 🔷 Phases")
                        phases = sorted(db.phases.keys())
                        for phase in phases:
                            st.markdown(f"- **{phase}**")

                    st.markdown("---")
                    st.success("✅ Ready for calculations! Go to 'Binary Phase Diagram' or 'Equilibrium Calculator'")

    st.markdown("---")

    # Display currently loaded database if any
    if 'current_db' in st.session_state:
        st.info(f"📊 Currently loaded: **{st.session_state['current_db_name']}**")
    else:
        st.warning("⚠️ No database currently loaded")

    st.markdown("---")
    st.markdown("""
    ### 🔍 Where to Find TDB Files?

    **Free Sources:**
    - **NIMS CPDDB**: [cpddb.nims.go.jp](https://cpddb.nims.go.jp/) - 700+ binary and ternary systems
    - **CALPHAD Journal**: Published assessments in supplementary materials
    - **Research Papers**: Many papers include TDB files as supplementary data

    **Commercial Sources:**
    - Thermo-Calc Software
    - CompuTherm's PanEngine
    - FactSage

    **Create Your Own:**
    - Use ESPEI to optimize parameters from DFT + experimental data
    - See `CALPHAD_SETUP_GUIDE.md` for details
    """)


# ============================================================================
# PAGE: BINARY PHASE DIAGRAM
# ============================================================================

elif page == "📊 Binary Phase Diagram":
    st.title("📊 Binary Phase Diagram Calculator")

    if not check_installation():
        st.stop()

    # Check if database is loaded
    if 'current_db' not in st.session_state:
        st.warning("⚠️ No database loaded. Please load a database first.")
        st.info("👈 Go to 'Load Database' page to select or upload a TDB file")
        st.stop()

    db = st.session_state['current_db']
    db_name = st.session_state['current_db_name']

    st.info(f"📊 Using database: **{db_name}**")

    st.markdown("---")

    # Element selection
    st.markdown("### 1️⃣ Select Binary System")

    elements = sorted([el for el in db.elements if el != 'VA'])

    col1, col2 = st.columns(2)

    with col1:
        element_a = st.selectbox("Element A:", options=elements, index=0)

    with col2:
        remaining_elements = [el for el in elements if el != element_a]
        if len(remaining_elements) > 0:
            element_b = st.selectbox("Element B:", options=remaining_elements, index=0)
        else:
            st.error("Need at least 2 elements for a binary diagram!")
            st.stop()

    st.success(f"✅ Selected system: **{element_a}-{element_b}**")

    # Phase selection
    st.markdown("### 2️⃣ Select Phases")

    all_phases = sorted(db.phases.keys())

    selected_phases = st.multiselect(
        "Include phases:",
        options=all_phases,
        default=all_phases,
        help="Select which phases to include in the calculation"
    )

    if len(selected_phases) == 0:
        st.warning("⚠️ Please select at least one phase")
        st.stop()

    # Calculation parameters
    st.markdown("### 3️⃣ Calculation Parameters")

    col1, col2 = st.columns(2)

    with col1:
        t_min = st.number_input(
            "Minimum Temperature (K):",
            min_value=100,
            max_value=5000,
            value=300,
            step=50
        )

        t_max = st.number_input(
            "Maximum Temperature (K):",
            min_value=100,
            max_value=5000,
            value=1500,
            step=50
        )

        t_step = st.number_input(
            "Temperature Step (K):",
            min_value=1,
            max_value=100,
            value=10,
            step=5
        )

    with col2:
        x_step = st.slider(
            f"Composition Step (mole fraction {element_b}):",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.005,
            format="%.3f"
        )

        pressure = st.number_input(
            "Pressure (Pa):",
            min_value=1.0,
            max_value=1e9,
            value=101325.0,
            step=10000.0,
            format="%.1f",
            help="1 atm = 101325 Pa"
        )

    # Validation
    if t_min >= t_max:
        st.error("❌ Minimum temperature must be less than maximum temperature")
        st.stop()

    # Calculate button
    st.markdown("---")

    if st.button("🔥 Calculate Phase Diagram", type="primary"):

        # Prepare components
        comps = [element_a, element_b, 'VA']

        st.markdown("### 🔄 Calculating...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Setting up calculation grid...")
            progress_bar.progress(10)

            # Create grid of T and composition values
            import numpy as np

            # Temperature grid
            t_vals = np.arange(t_min, t_max + t_step, t_step)
            # Composition grid
            x_vals = np.arange(0, 1 + x_step, x_step)

            total_points = len(t_vals) * len(x_vals)
            status_text.text(f"Calculating {total_points} equilibrium points...")

            # Storage for results
            results_grid = []

            # Calculate equilibrium at each grid point
            points_calculated = 0
            for t_idx, T in enumerate(t_vals):
                for x_idx, X in enumerate(x_vals):
                    # Update progress
                    points_calculated += 1
                    if points_calculated % 10 == 0:
                        progress = 10 + int(70 * points_calculated / total_points)
                        progress_bar.progress(progress)
                        status_text.text(f"Calculated {points_calculated}/{total_points} points...")

                    # Calculate equilibrium
                    try:
                        eq_result = equilibrium(
                            db, comps, selected_phases,
                            {v.T: T, v.P: pressure, v.N: 1, v.X(element_b): X}
                        )

                        # Extract dominant phase (highest fraction)
                        phases_at_point = eq_result.Phase.values.squeeze()
                        fractions_at_point = eq_result.NP.values.squeeze()

                        # Find dominant phase
                        if hasattr(phases_at_point, '__iter__') and not isinstance(phases_at_point, str):
                            # Multiple phases possible
                            valid_mask = ~np.isnan(fractions_at_point)
                            if np.any(valid_mask):
                                max_idx = np.nanargmax(fractions_at_point)
                                dominant_phase = phases_at_point[max_idx]
                                dominant_fraction = fractions_at_point[max_idx]
                            else:
                                dominant_phase = 'UNKNOWN'
                                dominant_fraction = 0
                        else:
                            # Single phase
                            dominant_phase = phases_at_point
                            dominant_fraction = fractions_at_point

                        results_grid.append({
                            'T': T,
                            'X': X,
                            'phase': str(dominant_phase) if dominant_phase != '' else 'UNKNOWN',
                            'fraction': dominant_fraction
                        })

                    except Exception as e:
                        # If calculation fails at this point, mark as unknown
                        results_grid.append({
                            'T': T,
                            'X': X,
                            'phase': 'UNKNOWN',
                            'fraction': 0
                        })

            progress_bar.progress(80)
            status_text.text("Generating phase diagram plot...")

            # Convert to DataFrame for easier manipulation
            results_df = pd.DataFrame(results_grid)

            # Create phase diagram plot
            fig, ax = plt.subplots(figsize=(10, 8))

            # Get unique phases
            unique_phases = results_df['phase'].unique()
            unique_phases = [p for p in unique_phases if p != 'UNKNOWN' and p != '']

            # Color map for phases
            phase_colors = {}
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_phases)))
            for idx, phase in enumerate(unique_phases):
                phase_colors[phase] = colors[idx]

            # Plot each phase region
            for phase in unique_phases:
                phase_data = results_df[results_df['phase'] == phase]
                ax.scatter(phase_data['X'], phase_data['T'],
                          c=[phase_colors[phase]], label=phase,
                          s=5, alpha=0.6)

            # Customize plot
            ax.set_title(f'{element_a}-{element_b} Binary Phase Diagram',
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Temperature (K)', fontsize=12)
            ax.set_xlabel(f'Mole Fraction {element_b}', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(t_min, t_max)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            plt.tight_layout()

            progress_bar.progress(100)
            status_text.text("Complete!")

            # Display plot
            st.pyplot(fig)

            # Show calculation statistics
            st.markdown("---")
            st.markdown("### 📊 Calculation Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Grid Points", total_points)
            with col2:
                st.metric("Temperature Steps", len(t_vals))
            with col3:
                st.metric("Composition Steps", len(x_vals))

            # Phase distribution
            phase_counts = results_df['phase'].value_counts()
            st.markdown("**Phase Distribution:**")
            for phase in unique_phases:
                count = phase_counts.get(phase, 0)
                percentage = 100 * count / total_points
                st.write(f"- **{phase}**: {count} points ({percentage:.1f}%)")

            # Save option
            st.markdown("---")
            col1, col2 = st.columns([3, 1])

            with col1:
                filename = st.text_input(
                    "Save as:",
                    value=f"{element_a}_{element_b}_phase_diagram.png"
                )

            with col2:
                dpi = st.selectbox("DPI:", options=[100, 150, 300], index=1)

            if st.button("💾 Save Figure"):
                output_folder = Path("calphad_outputs")
                output_folder.mkdir(exist_ok=True)

                save_path = output_folder / filename
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

                st.success(f"✅ Saved to: `{save_path}`")

            st.success("✅ Calculation complete!")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Calculation failed: {str(e)}")
            st.exception(e)


# ============================================================================
# PAGE: EQUILIBRIUM CALCULATOR
# ============================================================================

elif page == "⚖️ Equilibrium Calculator":
    st.title("⚖️ Equilibrium Calculator")

    if not check_installation():
        st.stop()

    # Check if database is loaded
    if 'current_db' not in st.session_state:
        st.warning("⚠️ No database loaded. Please load a database first.")
        st.info("👈 Go to 'Load Database' page to select or upload a TDB file")
        st.stop()

    db = st.session_state['current_db']
    db_name = st.session_state['current_db_name']

    st.info(f"📊 Using database: **{db_name}**")

    st.markdown("""
    Calculate equilibrium state at specific temperature and composition.

    **Results include:**
    - Stable phases and phase fractions
    - Chemical potentials
    - Molar Gibbs energy
    """)

    st.markdown("---")

    # Get elements
    elements = sorted([el for el in db.elements if el != 'VA'])

    # System type
    system_type = st.radio(
        "System type:",
        ["Binary (2 components)", "Ternary (3 components)"],
        help="Select number of components (excluding VA)"
    )

    st.markdown("### 1️⃣ Select Components")

    # Check if MP data is loaded
    mp_data_available = st.session_state.get('mp_data_loaded', False)

    if mp_data_available:
        use_mp_comp = st.checkbox(
            "📊 Use composition from Materials Project data",
            value=False,
            help="Select a material from the loaded MP data to auto-fill composition"
        )
    else:
        use_mp_comp = False

    if use_mp_comp and mp_data_available:
        mp_data = st.session_state['mp_data']

        # Create display options
        if 'material_id' in mp_data.columns and 'formula' in mp_data.columns:
            mp_options = [f"{row['material_id']} - {row['formula']}" for _, row in mp_data.iterrows()]
        elif 'formula' in mp_data.columns:
            mp_options = [f"{idx}: {row['formula']}" for idx, row in mp_data.iterrows()]
        else:
            mp_options = [f"Material {idx}" for idx in mp_data.index]

        selected_mp_idx = st.selectbox(
            "Select material:",
            options=range(len(mp_options)),
            format_func=lambda x: mp_options[x]
        )

        selected_mp_row = mp_data.iloc[selected_mp_idx]

        # Extract composition
        db_elements = [el for el in db.elements if el != 'VA']
        composition = get_composition_from_mp_data(selected_mp_row, db_elements)

        if len(composition) == 0:
            st.error("❌ Could not extract composition from selected material")

            # Show debugging information
            with st.expander("🔍 Debug Information"):
                st.markdown("**Database elements (looking for):**")
                st.code(', '.join(db_elements))

                st.markdown("**Available columns in MP data:**")
                st.code(', '.join(selected_mp_row.index.tolist()))

                st.markdown("**Non-zero numeric columns in this row:**")
                numeric_cols = []
                for col in selected_mp_row.index:
                    val = selected_mp_row[col]
                    if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                        numeric_cols.append(f"{col}={val}")
                st.code(', '.join(numeric_cols) if numeric_cols else "None found")

                st.markdown("**💡 Tip:** Column names must match database elements (case-insensitive). Example: If database has 'AL', 'ZN', your CSV should have columns named 'AL', 'ZN' (or 'al', 'zn', 'Al', 'Zn')")

            st.stop()

        # Display composition
        st.success(f"✅ Composition from MP: {', '.join([f'{k}={v:.3f}' for k, v in composition.items()])}")

        # Auto-set components for equilibrium calculator
        comp_elements = list(composition.keys())
        if system_type == "Binary (2 components)":
            if len(comp_elements) >= 2:
                comp_a = comp_elements[0]
                comp_b = comp_elements[1]
                selected_comps = [comp_a, comp_b, 'VA']
            else:
                st.error("Selected material doesn't have enough elements for binary system")
                st.stop()
        else:  # Ternary
            if len(comp_elements) >= 3:
                comp_a = comp_elements[0]
                comp_b = comp_elements[1]
                comp_c = comp_elements[2]
                selected_comps = [comp_a, comp_b, comp_c, 'VA']
            else:
                st.error("Selected material doesn't have enough elements for ternary system")
                st.stop()

        # Store composition fractions for later use
        st.session_state['selected_mp_composition'] = composition

    else:
        # Manual component selection

        if system_type == "Binary (2 components)":
            col1, col2 = st.columns(2)

            with col1:
                comp_a = st.selectbox("Component A:", options=elements, index=0)

            with col2:
                remaining = [el for el in elements if el != comp_a]
                if len(remaining) > 0:
                    comp_b = st.selectbox("Component B:", options=remaining, index=0)
                else:
                    st.error("Need at least 2 elements!")
                    st.stop()

            selected_comps = [comp_a, comp_b, 'VA']

        else:  # Ternary
            if len(elements) < 3:
                st.error("Database must have at least 3 elements for ternary calculation!")
                st.stop()

            col1, col2, col3 = st.columns(3)

            with col1:
                comp_a = st.selectbox("Component A:", options=elements, index=0)

            with col2:
                remaining = [el for el in elements if el != comp_a]
                comp_b = st.selectbox("Component B:", options=remaining, index=0)

            with col3:
                remaining = [el for el in elements if el not in [comp_a, comp_b]]
                if len(remaining) > 0:
                    comp_c = st.selectbox("Component C:", options=remaining, index=0)
                else:
                    st.error("Need at least 3 elements!")
                    st.stop()

            selected_comps = [comp_a, comp_b, comp_c, 'VA']

    # Phase selection
    st.markdown("### 2️⃣ Select Phases")

    all_phases = sorted(db.phases.keys())
    selected_phases = st.multiselect(
        "Include phases:",
        options=all_phases,
        default=all_phases
    )

    if len(selected_phases) == 0:
        st.warning("Please select at least one phase")
        st.stop()

    # Conditions
    st.markdown("### 3️⃣ Set Conditions")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input(
            "Temperature (K):",
            min_value=100.0,
            max_value=5000.0,
            value=600.0,
            step=10.0
        )

        pressure = st.number_input(
            "Pressure (Pa):",
            min_value=1.0,
            max_value=1e9,
            value=101325.0,
            step=10000.0,
            format="%.1f"
        )

    with col2:
        # Use MP composition if available, otherwise use sliders
        if use_mp_comp and 'selected_mp_composition' in st.session_state:
            composition = st.session_state['selected_mp_composition']
            st.markdown("**Composition (from MP):**")
            for elem, frac in composition.items():
                st.info(f"X({elem}) = {frac:.3f}")

            # Set composition values for calculation
            if system_type == "Binary (2 components)":
                x_b = composition.get(comp_b, 0.0)
            else:  # Ternary
                x_b = composition.get(comp_b, 0.0)
                x_c = composition.get(comp_c, 0.0)

        else:
            # Manual slider input
            if system_type == "Binary (2 components)":
                x_b = st.slider(
                    f"Mole Fraction {comp_b}:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.01
                )

                st.info(f"X({comp_a}) = {1-x_b:.3f}")
                st.info(f"X({comp_b}) = {x_b:.3f}")

            else:  # Ternary
                x_b = st.slider(
                    f"Mole Fraction {comp_b}:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.01
                )

                x_c = st.slider(
                    f"Mole Fraction {comp_c}:",
                    min_value=0.0,
                    max_value=1.0 - x_b,
                    value=min(0.2, 1.0 - x_b),
                    step=0.01
                )

                x_a = 1.0 - x_b - x_c

                st.info(f"X({comp_a}) = {x_a:.3f}")
                st.info(f"X({comp_b}) = {x_b:.3f}")
                st.info(f"X({comp_c}) = {x_c:.3f}")

    # Calculate button
    st.markdown("---")

    if st.button("⚖️ Calculate Equilibrium", type="primary"):

        # Build conditions
        conditions = {
            v.T: temperature,
            v.P: pressure,
            v.N: 1
        }

        if system_type == "Binary (2 components)":
            conditions[v.X(comp_b)] = x_b
        else:
            conditions[v.X(comp_b)] = x_b
            conditions[v.X(comp_c)] = x_c

        try:
            with st.spinner("Calculating equilibrium..."):
                result = equilibrium(db, selected_comps, selected_phases, conditions)

            st.success("✅ Equilibrium calculated successfully!")

            # Extract results
            st.markdown("---")
            st.markdown("### 📊 Results")

            # Stable phases
            stable_phases = result.Phase.values.squeeze()
            phase_fractions = result.NP.values.squeeze()

            # Handle different array shapes
            if not hasattr(stable_phases, '__iter__') or isinstance(stable_phases, str):
                stable_phases = [stable_phases]
                phase_fractions = [phase_fractions]

            # Filter out empty phases
            phase_data = []
            for phase, frac in zip(stable_phases, phase_fractions):
                if isinstance(phase, str) and phase != '' and frac > 1e-10:
                    phase_data.append({
                        'Phase': phase,
                        'Mole Fraction': frac
                    })

            if len(phase_data) > 0:
                st.markdown("#### Stable Phases")

                df_phases = pd.DataFrame(phase_data)
                st.dataframe(df_phases, use_container_width=True)

                # Plot phase fractions
                if len(phase_data) > 1:
                    fig, ax = plt.subplots(figsize=(8, 5))

                    phases_plot = [p['Phase'] for p in phase_data]
                    fractions_plot = [p['Mole Fraction'] for p in phase_data]

                    ax.bar(phases_plot, fractions_plot, color='steelblue', alpha=0.7)
                    ax.set_ylabel('Mole Fraction', fontsize=12)
                    ax.set_xlabel('Phase', fontsize=12)
                    ax.set_title('Phase Fractions', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')

                    st.pyplot(fig)
            else:
                st.warning("No stable phases found at these conditions")

            # Thermodynamic properties
            st.markdown("#### Thermodynamic Properties")

            gm = result.GM.values.squeeze()

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Molar Gibbs Energy", f"{gm:.2f} J/mol")
                st.metric("Temperature", f"{temperature:.1f} K")

            with col2:
                st.metric("Pressure", f"{pressure:.0f} Pa")
                if system_type == "Binary (2 components)":
                    st.metric(f"X({comp_b})", f"{x_b:.3f}")
                else:
                    st.metric(f"X({comp_b})", f"{x_b:.3f}")
                    st.metric(f"X({comp_c})", f"{x_c:.3f}")

            # Chemical potentials
            st.markdown("#### Chemical Potentials")

            mu_data = []
            for comp in selected_comps:
                if comp != 'VA':
                    mu_var = f'MU_{comp}'
                    if mu_var in result.data_vars:
                        mu_val = result[mu_var].values.squeeze()
                        mu_data.append({
                            'Component': comp,
                            'Chemical Potential (J/mol)': f"{mu_val:.2f}"
                        })

            if len(mu_data) > 0:
                df_mu = pd.DataFrame(mu_data)
                st.dataframe(df_mu, use_container_width=True)

            # Export results
            st.markdown("---")
            st.markdown("### 💾 Export Results")

            # Prepare export data
            export_data = {
                'Conditions': {
                    'Temperature_K': temperature,
                    'Pressure_Pa': pressure,
                    'Composition': {comp: conditions.get(v.X(comp), 0) for comp in selected_comps if comp != 'VA'}
                },
                'Stable_Phases': phase_data,
                'Gibbs_Energy_J_mol': float(gm)
            }

            col1, col2 = st.columns([3, 1])

            with col1:
                export_filename = st.text_input(
                    "Filename:",
                    value="equilibrium_results.json"
                )

            if st.button("💾 Save Results"):
                output_folder = Path("calphad_outputs")
                output_folder.mkdir(exist_ok=True)

                save_path = output_folder / export_filename

                with open(save_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

                st.success(f"✅ Results saved to: `{save_path}`")

        except Exception as e:
            st.error(f"❌ Calculation failed: {str(e)}")
            st.exception(e)


# ============================================================================
# PAGE: TEMPERATURE-PROPERTY CURVES
# ============================================================================

elif page == "💪 Temperature-Property Curves":
    st.title("💪 Temperature-Property Curves")

    if not check_installation():
        st.stop()

    # Check if database is loaded
    if 'current_db' not in st.session_state:
        st.warning("⚠️ No database loaded. Please load a database first.")
        st.info("👈 Go to 'Load Database' page to select or upload a TDB file")
        st.stop()

    db = st.session_state['current_db']
    db_name = st.session_state['current_db_name']

    st.info(f"📊 Using database: **{db_name}**")

    st.markdown("""
    Predict how mechanical properties vary with temperature based on **microstructure evolution**.

    **Method:**
    1. CALPHAD calculates phase fractions vs temperature
    2. Empirical models relate phase fractions to mechanical properties
    3. Generate temperature-strength/hardness curves

    **Applications:**
    - Design operating temperature ranges
    - Predict high-temperature performance
    - Validate against experimental data
    - Optimize heat treatment conditions
    """)

    st.markdown("---")

    # Get elements
    elements = sorted([el for el in db.elements if el != 'VA'])

    # Composition input
    st.markdown("### 1️⃣ Define Composition")

    # Check if MP data is loaded
    mp_data_available = st.session_state.get('mp_data_loaded', False)

    if mp_data_available:
        use_mp_comp = st.checkbox(
            "📊 Use composition from Materials Project data",
            value=False,
            help="Select a material from the loaded MP data to auto-fill composition",
            key="temp_prop_use_mp"
        )
    else:
        use_mp_comp = False

    if use_mp_comp and mp_data_available:
        mp_data = st.session_state['mp_data']

        # Create display options
        if 'material_id' in mp_data.columns and 'formula' in mp_data.columns:
            mp_options = [f"{row['material_id']} - {row['formula']}" for _, row in mp_data.iterrows()]
        elif 'formula' in mp_data.columns:
            mp_options = [f"{idx}: {row['formula']}" for idx, row in mp_data.iterrows()]
        else:
            mp_options = [f"Material {idx}" for idx in mp_data.index]

        selected_mp_idx = st.selectbox(
            "Select material:",
            options=range(len(mp_options)),
            format_func=lambda x: mp_options[x],
            key="temp_prop_mp_select"
        )

        selected_mp_row = mp_data.iloc[selected_mp_idx]

        # Extract composition
        db_elements = [el for el in db.elements if el != 'VA']
        composition = get_composition_from_mp_data(selected_mp_row, db_elements)

        if len(composition) == 0:
            st.error("❌ Could not extract composition from selected material")

            # Show debugging information
            with st.expander("🔍 Debug Information"):
                st.markdown("**Database elements (looking for):**")
                st.code(', '.join(db_elements))

                st.markdown("**Available columns in MP data:**")
                st.code(', '.join(selected_mp_row.index.tolist()))

                st.markdown("**Non-zero numeric columns in this row:**")
                numeric_cols = []
                for col in selected_mp_row.index:
                    val = selected_mp_row[col]
                    if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                        numeric_cols.append(f"{col}={val}")
                st.code(', '.join(numeric_cols) if numeric_cols else "None found")

                st.markdown("**💡 Tip:** Column names must match database elements (case-insensitive). Example: If database has 'AL', 'ZN', your CSV should have columns named 'AL', 'ZN' (or 'al', 'zn', 'Al', 'Zn')")

            st.stop()

        if len(composition) < 2:
            st.error("❌ Selected material must have at least 2 elements")
            st.stop()

        # Display composition
        comp_str = ', '.join([f'{k}={v:.3f}' for k, v in composition.items()])
        st.success(f"✅ Composition from MP: {comp_str}")

        # Auto-set elements
        comp_elements = list(composition.keys())
        comp_a = comp_elements[0]
        comp_b = comp_elements[1] if len(comp_elements) > 1 else comp_elements[0]

        # Set composition values
        x_b_val = composition.get(comp_b, 0.0)
        x_a_val = composition.get(comp_a, 1.0 - x_b_val)

    else:
        # Manual composition selection
        col1, col2 = st.columns(2)

        with col1:
            comp_a = st.selectbox("Base Element:", options=elements, index=0)

        with col2:
            remaining = [el for el in elements if el != comp_a]
            if len(remaining) > 0:
                comp_b = st.selectbox("Alloying Element:", options=remaining, index=0)
            else:
                st.error("Need at least 2 elements!")
                st.stop()

        # Composition fraction
        x_b_val = st.slider(
            f"Mole Fraction of {comp_b}:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help=f"Composition: {comp_a}-X{comp_b}"
        )

        x_a_val = 1.0 - x_b_val
        st.success(f"✅ Composition: {x_a_val:.2f} {comp_a} + {x_b_val:.2f} {comp_b}")

    # Phase selection
    st.markdown("### 2️⃣ Select Phases")

    all_phases = sorted(db.phases.keys())
    selected_phases = st.multiselect(
        "Include phases:",
        options=all_phases,
        default=all_phases
    )

    if len(selected_phases) == 0:
        st.warning("Please select at least one phase")
        st.stop()

    # Temperature range
    st.markdown("### 3️⃣ Temperature Range")

    col1, col2, col3 = st.columns(3)

    with col1:
        t_min = st.number_input(
            "Minimum Temperature (K):",
            min_value=100,
            max_value=5000,
            value=300,
            step=50
        )

    with col2:
        t_max = st.number_input(
            "Maximum Temperature (K):",
            min_value=100,
            max_value=5000,
            value=1000,
            step=50
        )

    with col3:
        t_step = st.number_input(
            "Temperature Step (K):",
            min_value=1,
            max_value=100,
            value=20,
            step=5
        )

    pressure = st.number_input(
        "Pressure (Pa):",
        min_value=1.0,
        max_value=1e9,
        value=101325.0,
        format="%.1f"
    )

    if t_min >= t_max:
        st.error("❌ Minimum temperature must be less than maximum temperature")
        st.stop()

    # Property model selection
    st.markdown("### 4️⃣ Property Model")

    property_type = st.selectbox(
        "Property to Predict:",
        ["Yield Strength (MPa)", "Tensile Strength (MPa)", "Hardness (HV)", "Custom"]
    )

    model_type = st.radio(
        "Prediction Model:",
        ["Rule of Mixtures", "Empirical Coefficients", "Upload Experimental Data"]
    )

    if model_type == "Rule of Mixtures":
        st.markdown("""
        **Rule of Mixtures:**
        `Property = Σ (Phase_Fraction_i × Phase_Property_i)`

        Enter the intrinsic property value for each phase:
        """)

        phase_properties = {}
        cols = st.columns(len(selected_phases))
        for idx, phase in enumerate(selected_phases):
            with cols[idx]:
                phase_properties[phase] = st.number_input(
                    f"{phase} value:",
                    min_value=0.0,
                    max_value=5000.0,
                    value=500.0,
                    step=50.0,
                    key=f"prop_{phase}"
                )

    elif model_type == "Empirical Coefficients":
        st.markdown("""
        **Empirical Model:**
        `Property = A + B×T + C×f_phase1 + D×f_phase2 + ...`

        where T is temperature in K and f_i are phase fractions
        """)

        coef_a = st.number_input("Coefficient A (baseline):", value=1000.0, step=50.0)
        coef_b = st.number_input("Coefficient B (temperature dependence):", value=-0.5, step=0.1)

        phase_coefs = {}
        for phase in selected_phases:
            phase_coefs[phase] = st.number_input(
                f"Coefficient for {phase}:",
                value=0.0,
                step=50.0,
                key=f"coef_{phase}"
            )

    else:  # Upload experimental data
        st.markdown("""
        **Experimental Data Comparison:**
        Upload CSV file with experimental temperature-property data.

        **Format:** Two columns: `Temperature_K`, `Property_Value`
        """)

        uploaded_exp = st.file_uploader(
            "Upload experimental data (CSV):",
            type=['csv'],
            help="CSV with Temperature_K and Property_Value columns"
        )

        exp_data = None
        if uploaded_exp:
            exp_data = pd.read_csv(uploaded_exp)
            st.success(f"✅ Loaded {len(exp_data)} experimental data points")
            st.dataframe(exp_data.head())

    # Calculate button
    st.markdown("---")

    if st.button("🔥 Calculate Temperature-Property Curve", type="primary"):

        comps = [comp_a, comp_b, 'VA']

        st.markdown("### 🔄 Calculating...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Temperature array
            t_vals = np.arange(t_min, t_max + t_step, t_step)
            total_points = len(t_vals)

            status_text.text(f"Calculating phase evolution across {total_points} temperatures...")

            # Storage for results
            results = []

            # Calculate equilibrium at each temperature
            for idx, T in enumerate(t_vals):
                progress = int(50 * (idx + 1) / total_points)
                progress_bar.progress(progress)
                status_text.text(f"Calculating T = {T:.0f} K ({idx+1}/{total_points})...")

                try:
                    eq_result = equilibrium(
                        db, comps, selected_phases,
                        {v.T: T, v.P: pressure, v.N: 1, v.X(comp_b): x_b_val}
                    )

                    # Extract phase fractions
                    phases_at_T = eq_result.Phase.values.squeeze()
                    fractions_at_T = eq_result.NP.values.squeeze()

                    # Build phase fraction dictionary
                    phase_fracs = {}

                    if hasattr(phases_at_T, '__iter__') and not isinstance(phases_at_T, str):
                        # Multiple phases
                        for phase, frac in zip(phases_at_T, fractions_at_T):
                            if phase != '' and not np.isnan(frac):
                                phase_fracs[str(phase)] = float(frac)
                    else:
                        # Single phase
                        if phases_at_T != '' and not np.isnan(fractions_at_T):
                            phase_fracs[str(phases_at_T)] = float(fractions_at_T)

                    # Calculate property based on model
                    if model_type == "Rule of Mixtures":
                        property_value = 0.0
                        for phase in selected_phases:
                            frac = phase_fracs.get(phase, 0.0)
                            property_value += frac * phase_properties[phase]

                    elif model_type == "Empirical Coefficients":
                        property_value = coef_a + coef_b * T
                        for phase in selected_phases:
                            frac = phase_fracs.get(phase, 0.0)
                            property_value += phase_coefs[phase] * frac

                    else:  # Experimental data - just store phase fractions
                        property_value = None

                    # Store result
                    result_dict = {
                        'Temperature_K': T,
                        'Temperature_C': T - 273.15,
                        'Property_Value': property_value
                    }

                    # Add phase fractions
                    for phase in selected_phases:
                        result_dict[f'{phase}_fraction'] = phase_fracs.get(phase, 0.0)

                    results.append(result_dict)

                except Exception as e:
                    st.warning(f"Calculation failed at T={T}K: {e}")
                    continue

            progress_bar.progress(60)
            status_text.text("Generating plots...")

            # Convert to DataFrame
            results_df = pd.DataFrame(results)

            if len(results_df) == 0:
                st.error("❌ No successful calculations. Check your inputs and database.")
                st.stop()

            # Create plots
            if model_type != "Upload Experimental Data":
                # Plot 1: Temperature vs Property
                fig1, ax1 = plt.subplots(figsize=(10, 6))

                ax1.plot(results_df['Temperature_C'], results_df['Property_Value'],
                        'b-', linewidth=2, label='CALPHAD Prediction')

                ax1.set_xlabel('Temperature (°C)', fontsize=12)
                ax1.set_ylabel(property_type, fontsize=12)
                ax1.set_title(f'{property_type} vs Temperature\n{comp_a}-{x_b_val:.0%}{comp_b}',
                            fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                plt.tight_layout()

                st.pyplot(fig1)

            else:
                # Plot experimental vs CALPHAD phases
                if exp_data is not None:
                    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

                    # Top: Experimental data
                    ax1.plot(exp_data['Temperature_K'] - 273.15, exp_data['Property_Value'],
                            'ro-', linewidth=2, markersize=6, label='Experimental Data')
                    ax1.set_xlabel('Temperature (°C)', fontsize=12)
                    ax1.set_ylabel(property_type, fontsize=12)
                    ax1.set_title(f'Experimental {property_type}', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()

                    # Bottom: CALPHAD phase fractions
                    for phase in selected_phases:
                        col_name = f'{phase}_fraction'
                        if col_name in results_df.columns:
                            ax2.plot(results_df['Temperature_C'], results_df[col_name],
                                    linewidth=2, label=phase, marker='o', markersize=4)

                    ax2.set_xlabel('Temperature (°C)', fontsize=12)
                    ax2.set_ylabel('Phase Fraction', fontsize=12)
                    ax2.set_title('CALPHAD Phase Evolution', fontsize=14, fontweight='bold')
                    ax2.set_ylim(0, 1.05)
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()

                    plt.tight_layout()
                    st.pyplot(fig1)

                    st.info("""
                    💡 **Interpretation:**
                    - Top plot shows experimental property behavior
                    - Bottom plot shows CALPHAD-predicted phase evolution
                    - Correlate property changes with phase transformations
                    """)

            # Plot 2: Phase fractions vs Temperature
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            for phase in selected_phases:
                col_name = f'{phase}_fraction'
                if col_name in results_df.columns:
                    ax2.plot(results_df['Temperature_C'], results_df[col_name],
                            linewidth=2, label=phase, marker='o', markersize=4)

            ax2.set_xlabel('Temperature (°C)', fontsize=12)
            ax2.set_ylabel('Phase Fraction', fontsize=12)
            ax2.set_title(f'Phase Evolution vs Temperature\n{comp_a}-{x_b_val:.0%}{comp_b}',
                        fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 1.05)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            plt.tight_layout()

            st.pyplot(fig2)

            progress_bar.progress(100)
            status_text.text("Complete!")

            # Display data table
            st.markdown("---")
            st.markdown("### 📊 Results Data")

            # Format for display
            display_df = results_df.copy()
            display_df['Temperature_C'] = display_df['Temperature_C'].round(1)
            if 'Property_Value' in display_df.columns:
                display_df['Property_Value'] = display_df['Property_Value'].round(2)

            for phase in selected_phases:
                col_name = f'{phase}_fraction'
                if col_name in display_df.columns:
                    display_df[col_name] = display_df[col_name].round(4)

            st.dataframe(display_df, use_container_width=True)

            # Export options
            st.markdown("---")
            st.markdown("### 💾 Export Data")

            col1, col2 = st.columns([3, 1])

            with col1:
                csv_filename = st.text_input(
                    "CSV Filename:",
                    value=f"{comp_a}{x_b_val:.0%}{comp_b}_temp_property.csv"
                )

            if st.button("💾 Save Results as CSV"):
                output_folder = Path("calphad_outputs")
                output_folder.mkdir(exist_ok=True)

                save_path = output_folder / csv_filename
                results_df.to_csv(save_path, index=False)

                st.success(f"✅ Results saved to: `{save_path}`")
                st.info(f"💡 **Use this data for ML training!** Import into App 1 as additional features.")

            st.success("✅ Calculation complete!")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Calculation failed: {str(e)}")
            st.exception(e)


# ============================================================================
# PAGE: BATCH MP ANALYSIS
# ============================================================================

elif page == "🔗 Batch MP Analysis":
    st.title("🔗 Batch Materials Project Analysis")

    if not check_installation():
        st.stop()

    # Check if database is loaded
    if 'current_db' not in st.session_state:
        st.warning("⚠️ No database loaded. Please load a database first.")
        st.info("👈 Go to 'Load Database' page to select or upload a TDB file")
        st.stop()

    db = st.session_state['current_db']
    db_name = st.session_state['current_db_name']

    st.info(f"📊 Using database: **{db_name}**")

    st.markdown("""
    **Automatically extract CALPHAD features for Materials Project datasets!**

    **This feature:**
    1. Loads CSV files downloaded from Materials Project (App 1)
    2. Extracts composition from each material
    3. Calculates CALPHAD equilibrium at specified temperature(s)
    4. Adds phase fractions and thermodynamic properties as new columns
    5. Exports combined dataset ready for ML training

    **Workflow:**
    ```
    App 1: Download Fe-Cr alloys → fe_cr_alloys.csv
    App 3: Load CSV → Calculate CALPHAD → fe_cr_with_calphad.csv
    App 1: Train ML with combined features!
    ```
    """)

    st.markdown("---")

    # Step 1: Upload Materials Project CSV
    st.markdown("### 1️⃣ Upload Materials Project Data")

    uploaded_file = st.file_uploader(
        "Upload CSV from Materials Project:",
        type=['csv'],
        help="CSV file downloaded from App 1 (Materials Project Explorer)"
    )

    if uploaded_file is None:
        st.info("💡 **Tip:** Go to App 1 → Download Data page → Download a CSV file → Upload it here")
        st.markdown("""
        **Expected columns:**
        - `material_id` (e.g., mp-123)
        - `formula` (e.g., Fe0.5Cr0.5)
        - Composition columns (e.g., `Fe`, `Cr`, `Ni`)
        - MP properties (formation_energy, band_gap, etc.)
        """)
        st.stop()

    # Load the CSV
    try:
        mp_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(mp_data)} materials from Materials Project")

        # Preview data
        with st.expander("📊 Preview Data"):
            st.dataframe(mp_data.head(10), use_container_width=True)
            st.markdown(f"**Columns:** {', '.join(mp_data.columns.tolist())}")

    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        st.stop()

    st.markdown("---")

    # Step 2: Identify composition columns
    st.markdown("### 2️⃣ Identify Composition Columns")

    # Get database elements
    db_elements = sorted([el for el in db.elements if el != 'VA'])

    st.markdown(f"**Elements in database:** {', '.join(db_elements)}")

    # Automatically detect composition columns
    detected_comp_cols = []
    for col in mp_data.columns:
        if col in db_elements:
            detected_comp_cols.append(col)

    if len(detected_comp_cols) > 0:
        st.success(f"✅ Auto-detected composition columns: {', '.join(detected_comp_cols)}")
        use_detected = st.checkbox("Use auto-detected columns", value=True)
    else:
        st.warning("⚠️ No composition columns auto-detected")
        use_detected = False

    if use_detected:
        comp_columns = detected_comp_cols
    else:
        # Manual selection
        comp_columns = st.multiselect(
            "Select composition columns:",
            options=mp_data.columns.tolist(),
            help="Columns containing mole fractions of each element"
        )

    if len(comp_columns) == 0:
        st.error("❌ Please select at least one composition column")
        st.stop()

    # Verify compositions sum to ~1
    if len(comp_columns) > 0:
        test_row = mp_data[comp_columns].iloc[0]
        comp_sum = test_row.sum()
        st.info(f"📊 First row composition sum: {comp_sum:.4f} (should be ≈ 1.0)")

        if abs(comp_sum - 1.0) > 0.1:
            st.warning("⚠️ Compositions don't sum to 1.0. They may be in atomic % instead of fractions.")
            normalize = st.checkbox("Normalize to sum = 1.0", value=True)
        else:
            normalize = False

    st.markdown("---")

    # Step 3: Select phases
    st.markdown("### 3️⃣ Select Phases")

    all_phases = sorted(db.phases.keys())
    selected_phases = st.multiselect(
        "Include phases:",
        options=all_phases,
        default=all_phases,
        help="Phases to include in equilibrium calculations"
    )

    if len(selected_phases) == 0:
        st.warning("Please select at least one phase")
        st.stop()

    st.markdown("---")

    # Step 4: Calculation settings
    st.markdown("### 4️⃣ Calculation Settings")

    col1, col2 = st.columns(2)

    with col1:
        calc_mode = st.radio(
            "Calculation Mode:",
            ["Single Temperature", "Multiple Temperatures"],
            help="Calculate at one temperature or multiple"
        )

    with col2:
        pressure = st.number_input(
            "Pressure (Pa):",
            min_value=1.0,
            max_value=1e9,
            value=101325.0,
            format="%.1f"
        )

    if calc_mode == "Single Temperature":
        temperature = st.number_input(
            "Temperature (K):",
            min_value=100.0,
            max_value=5000.0,
            value=1000.0,
            step=50.0,
            help="Temperature for equilibrium calculation"
        )
        temperatures = [temperature]
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            t_min = st.number_input("Min T (K):", min_value=100.0, value=300.0, step=50.0)
        with col2:
            t_max = st.number_input("Max T (K):", min_value=100.0, value=1500.0, step=50.0)
        with col3:
            t_step = st.number_input("Step (K):", min_value=10.0, value=200.0, step=50.0)

        temperatures = list(np.arange(t_min, t_max + t_step, t_step))
        st.info(f"Will calculate at {len(temperatures)} temperatures: {temperatures[0]:.0f}K to {temperatures[-1]:.0f}K")

    # Limit number of materials to process
    max_materials = st.slider(
        "Maximum materials to process:",
        min_value=1,
        max_value=len(mp_data),
        value=min(50, len(mp_data)),
        help="Processing many materials can take time. Start with fewer for testing."
    )

    st.markdown("---")

    # Calculate button
    if st.button("🔥 Calculate CALPHAD Features", type="primary"):

        st.markdown("### 🔄 Calculating...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare results storage
        results_list = []

        # Process each material
        materials_to_process = mp_data.head(max_materials)
        total_materials = len(materials_to_process)
        total_calculations = total_materials * len(temperatures)

        calculation_count = 0
        success_count = 0
        fail_count = 0

        for mat_idx, row in materials_to_process.iterrows():

            # Extract composition
            composition = {}
            for col in comp_columns:
                if col in db_elements:
                    val = row[col]
                    if pd.notna(val) and val > 0:
                        composition[col] = float(val)

            if len(composition) == 0:
                fail_count += 1
                continue

            # Normalize if needed
            if normalize:
                total = sum(composition.values())
                composition = {k: v/total for k, v in composition.items()}

            # Get components for PyCalphad
            comps = list(composition.keys()) + ['VA']

            # Calculate at each temperature
            for temp_idx, T in enumerate(temperatures):

                calculation_count += 1
                progress = int(100 * calculation_count / total_calculations)
                progress_bar.progress(progress)

                mat_id = row.get('material_id', f'material_{mat_idx}')
                status_text.text(f"Processing {mat_id} at {T:.0f}K ({calculation_count}/{total_calculations})...")

                try:
                    # Build conditions
                    conditions = {v.T: T, v.P: pressure, v.N: 1}

                    # Add composition constraints
                    # For binary, need to specify one composition (other is determined by sum=1)
                    # For ternary+, need N-1 compositions
                    comp_elements = list(composition.keys())
                    if len(comp_elements) >= 2:
                        # Specify all but first element
                        for elem in comp_elements[1:]:
                            conditions[v.X(elem)] = composition[elem]

                    # Calculate equilibrium
                    eq_result = equilibrium(db, comps, selected_phases, conditions)

                    # Extract phase fractions
                    phases_at_point = eq_result.Phase.values.squeeze()
                    fractions_at_point = eq_result.NP.values.squeeze()

                    # Build phase fraction dictionary
                    phase_fracs = {}

                    if hasattr(phases_at_point, '__iter__') and not isinstance(phases_at_point, str):
                        # Multiple phases
                        for phase, frac in zip(phases_at_point, fractions_at_point):
                            if phase != '' and not np.isnan(frac):
                                phase_fracs[str(phase)] = float(frac)
                    else:
                        # Single phase
                        if phases_at_point != '' and not np.isnan(fractions_at_point):
                            phase_fracs[str(phases_at_point)] = float(fractions_at_point)

                    # Extract Gibbs energy
                    gibbs_energy = float(eq_result.GM.values.squeeze())

                    # Build result dictionary
                    result = row.to_dict()  # Include all original MP data

                    # Add CALPHAD features
                    result['calphad_temperature_K'] = T

                    for phase in selected_phases:
                        col_name = f'{phase}_fraction_{T:.0f}K'
                        result[col_name] = phase_fracs.get(phase, 0.0)

                    result[f'gibbs_energy_{T:.0f}K'] = gibbs_energy

                    results_list.append(result)
                    success_count += 1

                except Exception as e:
                    fail_count += 1
                    # Still add row with NaN values
                    result = row.to_dict()
                    result['calphad_temperature_K'] = T
                    result['calphad_error'] = str(e)

                    for phase in selected_phases:
                        col_name = f'{phase}_fraction_{T:.0f}K'
                        result[col_name] = np.nan

                    result[f'gibbs_energy_{T:.0f}K'] = np.nan

                    results_list.append(result)

        progress_bar.progress(100)
        status_text.text("Complete!")

        # Create results DataFrame
        results_df = pd.DataFrame(results_list)

        st.success(f"✅ Calculation complete! {success_count} successes, {fail_count} failures")

        st.markdown("---")
        st.markdown("### 📊 Results Preview")

        # Show new columns
        original_cols = mp_data.columns.tolist()
        new_cols = [col for col in results_df.columns if col not in original_cols]

        st.markdown(f"**Added {len(new_cols)} new CALPHAD columns:**")
        st.code(', '.join(new_cols[:20]) + ('...' if len(new_cols) > 20 else ''))

        # Preview results
        with st.expander("📋 Preview Combined Data"):
            st.dataframe(results_df.head(20), use_container_width=True)

        # Statistics
        st.markdown("### 📈 Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Original Columns", len(original_cols))
            st.metric("Total Columns", len(results_df.columns))

        with col2:
            st.metric("Materials Processed", total_materials)
            st.metric("Success Rate", f"{100*success_count/total_calculations:.1f}%")

        with col3:
            st.metric("Temperatures", len(temperatures))
            st.metric("Total Calculations", total_calculations)

        # Show phase fraction distributions
        st.markdown("### 🔷 Phase Fraction Distributions")

        if calc_mode == "Single Temperature":
            # Single temperature - show histogram
            phase_cols = [col for col in results_df.columns if '_fraction_' in col]

            if len(phase_cols) > 0:
                fig, axes = plt.subplots(1, min(3, len(phase_cols)), figsize=(15, 4))
                if len(phase_cols) == 1:
                    axes = [axes]

                for idx, col in enumerate(phase_cols[:3]):
                    ax = axes[idx] if len(phase_cols) > 1 else axes[0]
                    data = results_df[col].dropna()
                    ax.hist(data, bins=20, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Phase Fraction', fontsize=10)
                    ax.set_ylabel('Count', fontsize=10)
                    ax.set_title(col.replace('_', ' '), fontsize=11)
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 💾 Export Combined Dataset")

        col1, col2 = st.columns([3, 1])

        with col1:
            export_filename = st.text_input(
                "Export filename:",
                value="materials_with_calphad.csv"
            )

        if st.button("💾 Save Combined Dataset"):
            output_folder = Path("calphad_outputs")
            output_folder.mkdir(exist_ok=True)

            save_path = output_folder / export_filename
            results_df.to_csv(save_path, index=False)

            st.success(f"✅ Saved to: `{save_path}`")

            st.markdown("""
            ### 🎯 Next Steps:

            **For ML Training (App 1):**
            1. Go to App 1 (Materials Project Explorer)
            2. Navigate to "Train Models" page
            3. Upload this CSV file
            4. Select features including the new CALPHAD columns
            5. Train model with both 0K (MP) and finite-T (CALPHAD) features!

            **Example features:**
            - From MP: `formation_energy`, `band_gap`, `bulk_modulus` (0K properties)
            - From CALPHAD: `FCC_A1_fraction_1000K`, `BCC_A2_fraction_1000K` (finite-T microstructure)
            - Target: `energy_above_hull`, `experimental_strength`, etc.

            **Result:** Much better predictions!
            """)

        # Store in session state for potential further use
        st.session_state['batch_results'] = results_df


# ============================================================================
# PAGE: CREATE TDB FROM MP DATA (ESPEI)
# ============================================================================

elif page == "🔬 Create TDB from MP Data":
    st.title("🔬 Create TDB Database from Materials Project Data")

    # Check for ESPEI installation
    try:
        import espei
        ESPEI_AVAILABLE = True
        espei_version = espei.__version__
    except ImportError:
        ESPEI_AVAILABLE = False
        espei_version = None

    if not ESPEI_AVAILABLE:
        st.error("⚠️ ESPEI is not installed!")
        st.markdown("""
        ### Installation Required

        ESPEI is required to create TDB files from thermochemical data.

        **Install with conda (recommended):**
        ```bash
        conda install -c conda-forge espei
        ```

        **Or with pip:**
        ```bash
        pip install espei
        ```

        Then restart this app.
        """)
        st.stop()

    st.success(f"✅ ESPEI version {espei_version} is installed!")

    st.markdown("""
    ### Generate CALPHAD Thermodynamic Databases from DFT Data

    This tool uses **ESPEI** (Extensible Self-optimizing Phase Equilibria Infrastructure)
    to convert Materials Project formation energy data into CALPHAD-compatible TDB files.

    **Workflow:**
    1. Upload Materials Project CSV with composition and formation energy data
    2. ESPEI converts DFT energies to thermodynamic model parameters
    3. Generate a TDB file compatible with PyCalphad
    4. Use the TDB in the other CALPHAD tools!

    **What you need:**
    - CSV file from Materials Project with:
      - Element composition columns (e.g., `Fe`, `Cr`, `Ni`)
      - Formation energy column (e.g., `formation_energy_per_atom`)
      - Material IDs and formulas (optional but helpful)
    """)

    st.markdown("---")

    # Step 1: Upload CSV
    st.markdown("### 1️⃣ Upload Materials Project Data")

    uploaded_file = st.file_uploader(
        "Upload Materials Project CSV:",
        type=['csv'],
        help="CSV file with composition and formation energy data",
        key="espei_mp_upload"
    )

    if uploaded_file is None:
        st.info("💡 Upload a CSV file from Materials Project to get started")
        st.markdown("""
        **Example CSV structure:**
        ```
        material_id, formula, Fe, Cr, Ni, formation_energy_per_atom
        mp-123, Fe0.5Cr0.5, 0.5, 0.5, 0, -0.234
        mp-456, Fe0.7Cr0.2Ni0.1, 0.7, 0.2, 0.1, -0.156
        ```
        """)
        st.stop()

    # Load and preview CSV
    try:
        mp_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(mp_df)} materials")

        with st.expander("📊 Preview Data"):
            st.dataframe(mp_df.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        st.stop()

    st.markdown("---")

    # Step 2: Identify columns
    st.markdown("### 2️⃣ Identify Data Columns")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Element Composition Columns:**")

        # Auto-detect element columns using improved detection
        import espei_integration as ei
        detected_element_map = ei.detect_element_columns(mp_df)

        if len(detected_element_map) > 0:
            detection_summary = ', '.join([f"{elem} → {col}" for elem, col in detected_element_map.items()])
            st.success(f"✅ Auto-detected: {detection_summary}")

            # Show what was found
            with st.expander("📋 Detected Mapping"):
                for elem, col in detected_element_map.items():
                    st.write(f"- **{elem}**: `{col}`")
        else:
            st.warning("⚠️ No element columns auto-detected")
            st.info("""
            **Expected column formats:**
            - Simple: `Fe`, `Cr`, `Ni`
            - With prefix: `frac_Fe`, `frac_Cr`
            - With suffix: `Fe_fraction`, `Cr_fraction`
            """)

        # Allow manual selection/override
        use_manual = st.checkbox("Manually specify element columns", value=len(detected_element_map) == 0)

        if use_manual:
            element_cols = st.multiselect(
                "Select element columns:",
                options=mp_df.columns.tolist(),
                default=[],
                help="Columns containing mole fractions of each element"
            )

            if len(element_cols) < 2:
                st.error("❌ Need at least 2 elements")
                st.stop()

            # Create manual mapping (assume column names are element symbols or will be detected)
            manual_element_map = {}
            for col in element_cols:
                # Try to extract element symbol from column name
                for elem_candidate in ['Fe', 'Cr', 'Ni', 'Cu', 'W', 'Mo', 'Al', 'Ti', 'Co', 'Mn', 'V']:
                    if elem_candidate.lower() in col.lower():
                        manual_element_map[elem_candidate.upper()] = col
                        break

            if len(manual_element_map) < 2:
                st.error("❌ Could not determine element symbols from column names")
                st.write("Selected columns:", element_cols)
                st.stop()

            element_map_to_use = manual_element_map
        else:
            if len(detected_element_map) < 2:
                st.error("❌ Need at least 2 elements. Try manual selection.")
                st.stop()
            element_map_to_use = detected_element_map

    with col2:
        st.markdown("**Formation Energy Column:**")

        # Auto-detect formation energy column
        energy_col_candidates = ['formation_energy_per_atom', 'formation_energy', 'e_form', 'energy_per_atom']
        detected_energy_col = None

        for candidate in energy_col_candidates:
            if candidate in mp_df.columns:
                detected_energy_col = candidate
                break

        if detected_energy_col:
            st.success(f"Auto-detected: {detected_energy_col}")

        energy_col = st.selectbox(
            "Formation energy column:",
            options=mp_df.columns.tolist(),
            index=mp_df.columns.tolist().index(detected_energy_col) if detected_energy_col else 0,
            help="Column with formation energy (typically in eV/atom)"
        )

    st.markdown("---")

    # Step 3: Configuration
    st.markdown("### 3️⃣ Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        max_materials = st.slider(
            "Max materials to process:",
            min_value=1,
            max_value=len(mp_df),
            value=min(50, len(mp_df)),
            help="Start with fewer materials for testing"
        )

    with col2:
        temperature = st.number_input(
            "Reference Temperature (K):",
            min_value=0.0,
            max_value=5000.0,
            value=298.15,
            help="Temperature for the formation energy data (typically 298.15 K)"
        )

    with col3:
        energy_unit = st.selectbox(
            "Energy units:",
            options=["eV/atom", "J/mol", "kJ/mol"],
            index=0,
            help="Units of the formation energy in the CSV"
        )

    # Phase selection
    st.markdown("**Phase Models:**")

    st.info("""
    **Select phases to include in the TDB file.**
    More phases = better phase diagrams, but uses same formation energy for all phases (approximation).
    """)

    col1_phase, col2_phase, col3_phase = st.columns(3)

    with col1_phase:
        include_liquid = st.checkbox("LIQUID", value=True, help="Liquid phase (disordered)")
        include_fcc = st.checkbox("FCC_A1", value=True, help="Face-centered cubic")

    with col2_phase:
        include_bcc = st.checkbox("BCC_A2", value=True, help="Body-centered cubic")
        include_hcp = st.checkbox("HCP_A3", value=False, help="Hexagonal close-packed")

    with col3_phase:
        include_bcc_b2 = st.checkbox("BCC_B2", value=False, help="Ordered BCC (B2 structure)")

    # Build phase list
    selected_phases = []
    if include_liquid:
        selected_phases.append("LIQUID")
    if include_fcc:
        selected_phases.append("FCC_A1")
    if include_bcc:
        selected_phases.append("BCC_A2")
    if include_hcp:
        selected_phases.append("HCP_A3")
    if include_bcc_b2:
        selected_phases.append("BCC_B2")

    if len(selected_phases) == 0:
        st.error("❌ Please select at least one phase")
        st.stop()

    st.success(f"Selected phases: {', '.join(selected_phases)}")

    st.markdown("**Output Settings:**")

    col1, col2 = st.columns(2)

    with col1:
        tdb_filename = st.text_input(
            "Output TDB filename:",
            value="materials_project.tdb",
            help="Name for the generated TDB file"
        )

    with col2:
        output_folder_name = st.text_input(
            "Output folder:",
            value="espei_output",
            help="Folder to save ESPEI outputs"
        )

    st.markdown("---")

    # Step 4: Generate TDB
    if st.button("🔥 Generate TDB File", type="primary"):

        st.markdown("### 🔄 Processing...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Import espei_integration module
            import espei_integration as ei

            # Create output directory
            output_dir = Path(output_folder_name)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Convert CSV to ESPEI datasets
            status_text.text("Converting Materials Project data to ESPEI format...")
            progress_bar.progress(20)

            # Save uploaded file temporarily
            temp_csv_path = output_dir / "temp_mp_data.csv"
            mp_df.head(max_materials).to_csv(temp_csv_path, index=False)

            datasets_created, elements, datasets_dir, used_element_map, phases_used = ei.convert_mp_csv_to_espei_datasets(
                csv_path=str(temp_csv_path),
                output_dir=output_dir,
                element_columns=element_map_to_use,
                phases=selected_phases,
                max_materials=max_materials,
                temperature=temperature
            )

            st.success(f"✅ Created {datasets_created} ESPEI datasets for elements: {', '.join(elements)}")
            st.info(f"📦 Generated datasets for phases: {', '.join(phases_used)}")

            with st.expander("📋 Element Column Mapping Used"):
                for elem, col in used_element_map.items():
                    st.write(f"- **{elem}**: `{col}`")
            progress_bar.progress(40)

            # Step 2: Create phase models
            status_text.text("Creating phase models...")

            phase_models_path = ei.create_phase_models_json(elements, output_dir, phases=selected_phases)
            st.info(f"📄 Phase models saved to: {phase_models_path}")
            progress_bar.progress(60)

            # Step 3: Create ESPEI input YAML
            status_text.text("Generating ESPEI configuration...")

            tdb_path = output_dir / tdb_filename
            yaml_path = output_dir / "espei_input.yaml"

            ei.create_espei_input_yaml(
                phase_models_path=phase_models_path,
                datasets_dir=datasets_dir,
                output_tdb_path=tdb_path,
                output_yaml_path=yaml_path
            )

            st.info(f"📄 ESPEI config saved to: {yaml_path}")
            progress_bar.progress(70)

            # Step 4: Run ESPEI
            status_text.text("Running ESPEI parameter generation (this may take a while)...")

            with st.spinner("ESPEI is generating thermodynamic parameters..."):
                success, message, result_tdb = ei.run_espei_generate_parameters(yaml_path)

            if success:
                progress_bar.progress(100)
                status_text.text("Complete!")

                st.success("✅ TDB file generated successfully!")

                st.markdown("---")
                st.markdown("### 📊 Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Materials Processed", datasets_created)
                    st.metric("Elements", len(elements))

                with col2:
                    st.metric("Temperature", f"{temperature} K")
                    st.metric("Output File", tdb_filename)

                st.markdown(f"**Elements in database:** {', '.join(elements)}")

                # Show output files
                st.markdown("### 📁 Generated Files")

                output_files = list(output_dir.glob("*"))
                for file in output_files:
                    if file.is_file():
                        st.markdown(f"- `{file.name}` ({file.stat().st_size / 1024:.1f} KB)")

                # Preview TDB if it exists
                if tdb_path.exists():
                    with st.expander("👀 Preview TDB File"):
                        with open(tdb_path, 'r') as f:
                            tdb_content = f.read()
                        st.code(tdb_content[:2000] + ("\n..." if len(tdb_content) > 2000 else ""), language='text')

                    st.markdown("---")
                    st.markdown("### 🎯 Next Steps")

                    st.markdown(f"""
                    **Use your new TDB file:**

                    1. **Copy to calphad_databases folder:**
                       - Move `{tdb_path}` to `calphad_databases/` folder

                    2. **Load in CALPHAD app:**
                       - Go to "📁 Load Database" page
                       - Select your new TDB file

                    3. **Calculate phase diagrams:**
                       - Use "📊 Binary Phase Diagram" page
                       - Explore equilibria with "⚖️ Equilibrium Calculator"

                    4. **Refine the database (advanced):**
                       - Add more thermochemical data
                       - Run ESPEI optimization (MCMC) with experimental phase boundary data
                       - See ESPEI documentation for details
                    """)

                    # Offer to copy to calphad_databases
                    if st.button("📋 Copy TDB to calphad_databases folder"):
                        calphad_db_dir = Path("calphad_databases")
                        calphad_db_dir.mkdir(exist_ok=True)

                        dest_path = calphad_db_dir / tdb_filename

                        import shutil
                        shutil.copy2(tdb_path, dest_path)

                        st.success(f"✅ Copied to: {dest_path}")
                        st.info("💡 Now go to 'Load Database' page to use it!")

                else:
                    st.error(f"❌ TDB file was not created at expected location: {tdb_path}")
                    st.warning(message)

            else:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ ESPEI generation failed: {message}")

                st.markdown("### 🔍 Troubleshooting")
                st.markdown("""
                **Common issues:**

                - **No formation energy data:** Make sure your CSV has formation energy values
                - **Insufficient data:** ESPEI needs multiple data points to fit parameters
                - **Element mismatch:** Check that element columns are properly named

                **Try:**
                - Increase the number of materials to process
                - Check that formation energies are valid numbers
                - Verify element composition columns sum to ≈ 1.0
                """)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

            st.markdown("### 📋 Debug Information")
            st.markdown(f"""
            - Elements detected: {list(element_map_to_use.keys())}
            - Element columns: {list(element_map_to_use.values())}
            - Formation energy column: {energy_col}
            - Number of materials: {len(mp_df)}
            - Available CSV columns: {', '.join(mp_df.columns.tolist())}
            """)


# ============================================================================
# PAGE: SCHEIL SIMULATION
# ============================================================================

elif page == "❄️ Scheil Simulation":
    st.title("❄️ Scheil Solidification Simulation")

    if not check_installation():
        st.stop()

    if not SCHEIL_AVAILABLE:
        st.warning("⚠️ Scheil package not installed")
        st.markdown("""
        The Scheil package is required for solidification simulations.

        **Install with:**
        ```bash
        conda install -c conda-forge scheil
        ```

        or

        ```bash
        pip install scheil
        ```
        """)
        st.stop()

    # Check if database is loaded
    if 'current_db' not in st.session_state:
        st.warning("⚠️ No database loaded. Please load a database first.")
        st.info("👈 Go to 'Load Database' page to select or upload a TDB file")
        st.stop()

    db = st.session_state['current_db']
    db_name = st.session_state['current_db_name']

    st.info(f"📊 Using database: **{db_name}**")

    st.markdown("""
    Simulate **non-equilibrium solidification** using the Scheil-Gulliver method.

    **Assumptions:**
    - No diffusion in solid
    - Perfect mixing in liquid
    - Local equilibrium at solid-liquid interface

    **Results:**
    - Solidification path (T vs fraction solid)
    - Phase evolution during cooling
    - Final microstructure prediction
    """)

    st.markdown("---")

    st.info("🚧 Scheil simulation interface coming soon!")
    st.markdown("""
    **Planned features:**
    - Set initial liquid composition
    - Define cooling conditions
    - Visualize solidification path
    - Predict microsegregation
    - Calculate solidification temperature range

    For now, use command-line interface. See `CALPHAD_SETUP_GUIDE.md` for examples.
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
CALPHAD Phase Diagram Calculator<br>
Powered by PyCalphad<br>
ML Alloy Project
</div>
""", unsafe_allow_html=True)
