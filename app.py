"""
ML Alloy Development - Web Dashboard
Interactive web application for alloy discovery using Materials Project data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ML Alloy Development",
    page_icon="🔬",
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
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.title("🔬 ML Alloy Development Dashboard")
st.markdown("### Interactive Platform for Novel Alloy Discovery")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📥 Download Data", "📊 Explore Data", "🤖 Train Models", "🔮 Discover Alloys"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Info")
st.sidebar.info(
    "**ML Alloy Development**\n\n"
    "Discover novel alloys using machine learning and Materials Project data.\n\n"
    "**Current Status:**\n"
    "✅ Environment Ready\n"
    "✅ API Connected\n"
    "✅ Data Downloaded"
)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.header("Welcome to ML Alloy Development! 👋")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Python Version", "3.13.9")
        st.metric("Packages Installed", "15+")

    with col2:
        # Check if data exists
        data_files = list(Path('.').glob('*.csv'))
        st.metric("Data Files", len(data_files))
        if Path('fe_ni_alloys.csv').exists():
            df = pd.read_csv('fe_ni_alloys.csv')
            st.metric("Materials in DB", len(df))
        else:
            st.metric("Materials in DB", "0")

    with col3:
        st.metric("API Status", "Connected ✅")
        st.metric("Materials Project", "Ready")

    st.markdown("---")

    # Quick stats dashboard
    st.subheader("📊 Quick Overview")

    if Path('fe_ni_alloys.csv').exists():
        df = pd.read_csv('fe_ni_alloys.csv')

        col1, col2 = st.columns(2)

        with col1:
            # Formation energy distribution
            fig = px.histogram(df, x='formation_energy',
                             title='Formation Energy Distribution',
                             labels={'formation_energy': 'Formation Energy (eV/atom)'},
                             color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Stability distribution
            fig = px.scatter(df, x='ni_fraction', y='formation_energy',
                           color='energy_above_hull',
                           title='Formation Energy vs Composition',
                           labels={'ni_fraction': 'Ni Fraction',
                                  'formation_energy': 'Formation Energy (eV/atom)',
                                  'energy_above_hull': 'E above hull (eV)'},
                           color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)

        # Key findings
        st.markdown("### 🔍 Key Findings")
        stable_count = df['is_stable'].sum()
        near_stable = (df['energy_above_hull'] < 0.1).sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Materials", len(df))
        col2.metric("Stable", f"{stable_count} ({stable_count/len(df)*100:.1f}%)")
        col3.metric("Near-Stable", f"{near_stable} ({near_stable/len(df)*100:.1f}%)")
        col4.metric("Avg Density", f"{df['density'].mean():.2f} g/cm³")
    else:
        st.info("📥 No data loaded yet. Go to 'Download Data' to get started!")

    st.markdown("---")

    # Features overview
    st.subheader("🚀 What You Can Do")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **📥 Download Data**
        - Query Materials Project database
        - Download alloy systems
        - Filter by properties
        - Export to CSV

        **📊 Explore Data**
        - Interactive visualizations
        - Property correlations
        - Composition analysis
        - Stability insights
        """)

    with col2:
        st.markdown("""
        **🤖 Train Models**
        - Build ML models
        - Predict properties
        - Evaluate accuracy
        - Compare algorithms

        **🔮 Discover Alloys**
        - Generate candidates
        - Predict properties
        - Rank by stability
        - Export results
        """)

    st.markdown("---")
    st.markdown("### 📚 Quick Start Guide")
    st.markdown("""
    1. **Download Data**: Start by downloading alloy data from Materials Project
    2. **Explore**: Visualize composition-property relationships
    3. **Train Models**: Build predictive ML models
    4. **Discover**: Generate and evaluate novel alloy candidates
    """)

# ============================================================================
# DOWNLOAD DATA PAGE
# ============================================================================
elif page == "📥 Download Data":
    st.header("Download Alloy Data from Materials Project")

    # API key check
    api_key = os.environ.get('MP_API_KEY')
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get('MP_API_KEY')
        except:
            pass

    if not api_key:
        st.error("⚠️ API Key not found! Please set MP_API_KEY in your .env file")
        st.stop()
    else:
        st.success("✅ Materials Project API Connected")

    st.markdown("---")

    # Download options
    st.subheader("Select Alloy System")

    col1, col2 = st.columns(2)

    with col1:
        system_type = st.selectbox(
            "System Type",
            ["Binary (2 elements)", "Ternary (3 elements)", "Custom"]
        )

        if system_type == "Binary (2 elements)":
            el1 = st.selectbox("Element 1", ["Fe", "Ni", "Ti", "Al", "Co", "Cr", "Cu", "Mn", "Mo"])
            el2 = st.selectbox("Element 2", ["Ni", "Fe", "Al", "Ti", "Cr", "Co", "Cu", "V", "W"])
            chemsys = f"{el1}-{el2}"
        elif system_type == "Ternary (3 elements)":
            el1 = st.selectbox("Element 1", ["Fe", "Ni", "Ti", "Al", "Co", "Cr"])
            el2 = st.selectbox("Element 2", ["Ni", "Cr", "Al", "Ti", "Co", "Fe"])
            el3 = st.selectbox("Element 3", ["Cr", "Ni", "Ti", "Al", "Mo", "V"])
            chemsys = f"{el1}-{el2}-{el3}"
        else:
            chemsys = st.text_input("Enter chemical system (e.g., Fe-Ni-Cr)")

        st.info(f"📌 System: **{chemsys}**")

    with col2:
        metallic_only = st.checkbox("Metallic materials only (band gap = 0)", value=True)
        stable_only = st.checkbox("Stable materials only", value=False)
        max_results = st.slider("Maximum results", 10, 500, 100)

    if st.button("🔍 Download Data", type="primary"):
        from mp_api.client import MPRester

        with st.spinner(f"Downloading {chemsys} data from Materials Project..."):
            try:
                all_data = []

                with MPRester(api_key) as mpr:
                    search_kwargs = {
                        "chemsys": chemsys,
                        "fields": ["material_id", "formula_pretty", "composition",
                                  "energy_per_atom", "formation_energy_per_atom",
                                  "energy_above_hull", "band_gap", "is_stable",
                                  "volume", "density", "nsites", "symmetry"]
                    }

                    if metallic_only:
                        search_kwargs["band_gap"] = (0, 0)

                    if stable_only:
                        search_kwargs["is_stable"] = True

                    docs = mpr.materials.summary.search(**search_kwargs)

                    if not docs:
                        st.warning(f"No materials found for {chemsys}")
                        st.stop()

                    # Limit results
                    docs = docs[:max_results]

                    st.success(f"✅ Found {len(docs)} materials")

                    progress_bar = st.progress(0)
                    for i, doc in enumerate(docs):
                        mat_data = {
                            "material_id": doc.material_id,
                            "formula": doc.formula_pretty,
                            "system": chemsys,
                            "energy_per_atom": doc.energy_per_atom,
                            "formation_energy": doc.formation_energy_per_atom,
                            "energy_above_hull": doc.energy_above_hull,
                            "band_gap": doc.band_gap,
                            "is_stable": doc.is_stable,
                            "volume": doc.volume,
                            "density": doc.density,
                            "nsites": doc.nsites,
                        }

                        if doc.volume and doc.nsites:
                            mat_data["volume_per_atom"] = doc.volume / doc.nsites

                        if hasattr(doc, 'symmetry') and doc.symmetry:
                            mat_data["space_group"] = doc.symmetry.number
                            mat_data["crystal_system"] = doc.symmetry.crystal_system

                        # Extract composition fractions
                        if hasattr(doc, 'composition'):
                            elements = chemsys.split('-')
                            for element in elements:
                                frac = doc.composition.get_atomic_fraction(element)
                                mat_data[f"frac_{element}"] = frac

                        all_data.append(mat_data)
                        progress_bar.progress((i + 1) / len(docs))

                # Create DataFrame
                df = pd.DataFrame(all_data)

                # Save to file
                filename = f"{chemsys.replace('-', '_')}_alloys.csv"
                df.to_csv(filename, index=False)

                st.success(f"✅ Downloaded {len(df)} materials")
                st.success(f"💾 Saved to: {filename}")

                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Summary stats
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Materials", len(df))
                with col2:
                    stable = df['is_stable'].sum()
                    st.metric("Stable", f"{stable} ({stable/len(df)*100:.1f}%)")
                with col3:
                    near_stable = (df['energy_above_hull'] < 0.1).sum()
                    st.metric("Near-Stable", f"{near_stable}")
                with col4:
                    st.metric("Avg Density", f"{df['density'].mean():.2f} g/cm³")

            except Exception as e:
                st.error(f"Error downloading data: {e}")

# ============================================================================
# EXPLORE DATA PAGE
# ============================================================================
elif page == "📊 Explore Data":
    st.header("Explore Alloy Data")

    # Load data
    csv_files = list(Path('.').glob('*.csv'))

    if not csv_files:
        st.warning("No data files found. Go to 'Download Data' first!")
        st.stop()

    selected_file = st.selectbox("Select dataset", [f.name for f in csv_files])

    df = pd.read_csv(selected_file)

    st.success(f"✅ Loaded {len(df)} materials from {selected_file}")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Table", "📈 Visualizations", "🔗 Correlations", "📊 Statistics"])

    with tab1:
        st.subheader("Data Table")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            if 'is_stable' in df.columns:
                show_stable_only = st.checkbox("Show stable materials only")
                if show_stable_only:
                    df = df[df['is_stable'] == True]

        with col2:
            if 'energy_above_hull' in df.columns:
                max_hull = st.slider("Max energy above hull (eV)", 0.0, 1.0, 0.5, 0.05)
                df = df[df['energy_above_hull'] <= max_hull]

        # Check if dataframe is empty after filtering
        if len(df) == 0:
            st.warning("⚠️ No materials match your filter criteria. Try adjusting the filters above.")
            st.info("💡 Tip: Increase 'Max energy above hull' or uncheck 'Show stable materials only'")
        else:
            st.success(f"✅ Showing {len(df)} materials")
            st.dataframe(df, use_container_width=True, height=400)

            # Download filtered data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"filtered_{selected_file}",
                mime="text/csv"
            )

    with tab2:
        st.subheader("Interactive Visualizations")

        # Check if dataframe is empty
        if len(df) == 0:
            st.warning("⚠️ No data to visualize after applying filters. Try adjusting the filters above.")
            st.stop()

        # Determine composition columns
        comp_cols = [col for col in df.columns if col.startswith('frac_')]

        if len(comp_cols) >= 2:
            col1, col2 = st.columns(2)

            with col1:
                # Formation energy vs composition
                x_col = st.selectbox("X-axis (composition)", comp_cols, index=0)

                fig = px.scatter(df, x=x_col, y='formation_energy',
                               color='energy_above_hull',
                               hover_data=['formula', 'material_id'],
                               title='Formation Energy vs Composition',
                               labels={x_col: x_col.replace('frac_', '').title() + ' Fraction',
                                      'formation_energy': 'Formation Energy (eV/atom)',
                                      'energy_above_hull': 'E above hull (eV)'},
                               color_continuous_scale='RdYlGn_r',
                               height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Density vs composition
                fig = px.scatter(df, x=x_col, y='density',
                               color='is_stable',
                               hover_data=['formula', 'material_id'],
                               title='Density vs Composition',
                               labels={x_col: x_col.replace('frac_', '').title() + ' Fraction',
                                      'density': 'Density (g/cm³)'},
                               height=500)
                st.plotly_chart(fig, use_container_width=True)

        # Energy distributions
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x='formation_energy',
                             title='Formation Energy Distribution',
                             labels={'formation_energy': 'Formation Energy (eV/atom)'},
                             color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(df, x='energy_above_hull',
                             title='Stability Distribution',
                             labels={'energy_above_hull': 'Energy Above Hull (eV/atom)'},
                             color_discrete_sequence=['#ff7f0e'])
            fig.add_vline(x=0.1, line_dash="dash", line_color="red",
                         annotation_text="Likely Synthesizable")
            st.plotly_chart(fig, use_container_width=True)

        # Crystal system distribution
        if 'crystal_system' in df.columns:
            fig = px.histogram(df, x='crystal_system',
                             title='Crystal System Distribution',
                             color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Property Correlations")

        # Check if dataframe is empty
        if len(df) == 0:
            st.warning("⚠️ No data to analyze after applying filters. Try adjusting the filters above.")
            st.stop()

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if not col.startswith('space_group')]

        if len(numeric_cols) > 1:
            # Correlation matrix
            corr = df[numeric_cols].corr()

            fig = px.imshow(corr,
                           labels=dict(color="Correlation"),
                           x=corr.columns,
                           y=corr.columns,
                           color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1,
                           title="Property Correlation Matrix")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Pairplot for selected properties
            st.subheader("Scatter Matrix")
            selected_props = st.multiselect(
                "Select properties for scatter matrix",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )

            if len(selected_props) >= 2:
                fig = px.scatter_matrix(df[selected_props],
                                       title="Property Relationships",
                                       height=800)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Statistical Summary")

        if len(df) == 0:
            st.warning("⚠️ No data to display after applying filters. Try adjusting the filters above.")
        else:
            st.dataframe(df.describe(), use_container_width=True)

            # Key insights
            st.markdown("### 🔍 Key Insights")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Most Stable Material**")
                if 'energy_above_hull' in df.columns and len(df) > 0:
                    most_stable = df.loc[df['energy_above_hull'].idxmin()]
                    st.write(f"Formula: **{most_stable['formula']}**")
                    st.write(f"E above hull: {most_stable['energy_above_hull']:.4f} eV/atom")
                else:
                    st.write("No data available")

            with col2:
                st.markdown("**Highest Density**")
                if 'density' in df.columns and len(df) > 0:
                    highest_density = df.loc[df['density'].idxmax()]
                    st.write(f"Formula: **{highest_density['formula']}**")
                    st.write(f"Density: {highest_density['density']:.2f} g/cm³")
                else:
                    st.write("No data available")

            with col3:
                st.markdown("**Stability Summary**")
                if 'is_stable' in df.columns and len(df) > 0:
                    stable_count = df['is_stable'].sum()
                    near_stable = (df['energy_above_hull'] < 0.1).sum()
                    st.write(f"Stable: {stable_count} ({stable_count/len(df)*100:.1f}%)")
                    st.write(f"Near-stable: {near_stable} ({near_stable/len(df)*100:.1f}%)")
                else:
                    st.write("No data available")

# ============================================================================
# TRAIN MODELS PAGE
# ============================================================================
elif page == "🤖 Train Models":
    st.header("Train ML Models")

    # Load data
    csv_files = list(Path('.').glob('*.csv'))

    if not csv_files:
        st.warning("No data files found. Go to 'Download Data' first!")
        st.stop()

    selected_file = st.selectbox("Select dataset", [f.name for f in csv_files])
    df = pd.read_csv(selected_file)

    if len(df) == 0:
        st.error("❌ The selected dataset is empty!")
        st.stop()

    if len(df) < 5:
        st.warning(f"⚠️ Dataset only has {len(df)} materials. You need at least 5 materials to train models.")
        st.info("💡 Tip: Download more data from 'Download Data' page")
        st.stop()

    st.success(f"✅ Loaded {len(df)} materials")

    st.markdown("---")

    # Model configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Features")
        feature_cols = [col for col in df.columns if col.startswith('frac_')]

        if 'density' in df.columns:
            add_density = st.checkbox("Include density as feature", value=False)
            if add_density:
                feature_cols.append('density')

        st.write("Selected features:")
        for col in feature_cols:
            st.write(f"- {col}")

    with col2:
        st.subheader("Target")
        target = st.selectbox("Target property", ['formation_energy', 'energy_above_hull', 'density'])

        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100

    if st.button("🚀 Train Models", type="primary"):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        with st.spinner("Training models..."):
            # Prepare data
            X = df[feature_cols].values
            y = df[target].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Train multiple models
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
            }

            results = {}

            progress_bar = st.progress(0)
            for i, (name, model) in enumerate(models.items()):
                model.fit(X_train, y_train)

                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                results[name] = {
                    "train_r2": r2_score(y_train, train_pred),
                    "test_r2": r2_score(y_test, test_pred),
                    "test_mae": mean_absolute_error(y_test, test_pred),
                    "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
                    "predictions": test_pred
                }

                progress_bar.progress((i + 1) / len(models))

        st.success("✅ Models trained successfully!")

        # Results
        st.subheader("📊 Model Performance")

        results_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Train R²": [results[m]["train_r2"] for m in results.keys()],
            "Test R²": [results[m]["test_r2"] for m in results.keys()],
            "Test MAE": [results[m]["test_mae"] for m in results.keys()],
            "Test RMSE": [results[m]["test_rmse"] for m in results.keys()]
        })

        st.dataframe(results_df.style.highlight_max(subset=["Train R²", "Test R²"], color='lightgreen')
                                     .highlight_min(subset=["Test MAE", "Test RMSE"], color='lightgreen'),
                    use_container_width=True)

        # Visualization
        st.subheader("📈 Predictions vs Actual")

        fig = make_subplots(rows=1, cols=3,
                           subplot_titles=list(results.keys()))

        for i, (name, res) in enumerate(results.items(), 1):
            fig.add_trace(
                go.Scatter(x=y_test, y=res["predictions"],
                          mode='markers',
                          name=name,
                          marker=dict(size=8, opacity=0.6)),
                row=1, col=i
            )

            # Perfect prediction line
            min_val, max_val = y_test.min(), y_test.max()
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines',
                          name='Perfect',
                          line=dict(dash='dash', color='red'),
                          showlegend=(i==1)),
                row=1, col=i
            )

            fig.update_xaxes(title_text=f"Actual {target}", row=1, col=i)
            fig.update_yaxes(title_text=f"Predicted {target}", row=1, col=i)

        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Best model
        best_model = max(results.items(), key=lambda x: x[1]["test_r2"])
        st.success(f"🏆 Best Model: **{best_model[0]}** (Test R² = {best_model[1]['test_r2']:.4f})")

# ============================================================================
# DISCOVER ALLOYS PAGE
# ============================================================================
elif page == "🔮 Discover Alloys":
    st.header("Discover Novel Alloy Candidates")

    st.info("🚧 This feature is under development. It will allow you to:")
    st.markdown("""
    - Generate novel alloy compositions
    - Predict properties using trained ML models
    - Rank candidates by predicted stability
    - Export promising candidates for experimental validation
    """)

    st.markdown("---")

    st.subheader("Coming Soon!")
    st.markdown("""
    **Features in development:**
    1. **Composition Generator**: Create candidate alloys in specified composition space
    2. **Property Predictor**: Use trained models to predict formation energy, stability
    3. **Multi-objective Optimization**: Balance multiple properties simultaneously
    4. **Ranking System**: Sort by stability, cost, manufacturability
    5. **Export**: Download top candidates for experimental testing
    """)

    # Placeholder for future functionality
    st.markdown("### Preview: Candidate Generation")

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox("Base element", ["Fe", "Ti", "Ni", "Al"], disabled=True)
        st.selectbox("Alloying element", ["Ni", "Al", "Cr", "Co"], disabled=True)

    with col2:
        st.slider("Composition range (%)", 0, 100, (20, 80), disabled=True)
        st.slider("Number of candidates", 10, 100, 50, disabled=True)

    st.button("🔮 Generate Candidates", disabled=True, type="primary")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "ML Alloy Development Dashboard | "
    "Powered by Materials Project & Streamlit | "
    "🔬 Discover the Future of Alloys"
    "</div>",
    unsafe_allow_html=True
)
