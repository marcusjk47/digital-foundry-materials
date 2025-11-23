"""
Experimental Data Manager
Upload, store, and visualize temperature-dependent alloy properties
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Experimental Data Manager",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Experimental Property Manager")
st.markdown("### Upload and Visualize Temperature-Dependent Alloy Properties")

# Create experimental data directory if it doesn't exist
exp_data_dir = Path("experimental_data")
exp_data_dir.mkdir(exist_ok=True)

# Sidebar navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Select Mode",
    ["📤 Upload Data", "📊 Visualize Curves", "📈 Compare Materials", "💾 Manage Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Supported Properties")
st.sidebar.info(
    "**Mechanical:**\n"
    "- Yield Strength\n"
    "- Tensile Strength\n"
    "- Hardness\n"
    "- Ductility\n\n"
    "**Thermal:**\n"
    "- Thermal Conductivity\n"
    "- Thermal Expansion\n"
    "- Specific Heat\n\n"
    "**Other:**\n"
    "- Elastic Modulus\n"
    "- Creep Rate\n"
    "- Fatigue Strength"
)

# ============================================================================
# UPLOAD DATA MODE
# ============================================================================
if mode == "📤 Upload Data":
    st.header("Upload Experimental Data")

    st.markdown("""
    Upload temperature-dependent property data for your alloys.

    **Data Format Requirements:**
    - CSV file with columns: `Temperature`, `Property_Value`
    - Temperature in °C or K (specify below)
    - One property per file
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Material Information")
        material_name = st.text_input("Material Name", placeholder="e.g., Ti-6Al-4V")
        material_composition = st.text_input("Composition (optional)", placeholder="e.g., Ti-6Al-4V-0.3Fe")

        property_name = st.selectbox(
            "Property",
            ["Yield Strength", "Tensile Strength", "Hardness", "Ductility",
             "Elastic Modulus", "Thermal Conductivity", "Thermal Expansion",
             "Specific Heat", "Creep Rate", "Fatigue Strength", "Custom"]
        )

        if property_name == "Custom":
            property_name = st.text_input("Custom Property Name")

        property_unit = st.text_input("Property Unit", placeholder="e.g., MPa, GPa, %")

    with col2:
        st.subheader("Temperature Settings")
        temp_unit = st.selectbox("Temperature Unit", ["°C", "K", "°F"])

        st.markdown("### Data Source")
        data_source = st.text_input("Data Source (optional)",
                                    placeholder="e.g., Journal, Lab Test, Manufacturer")
        notes = st.text_area("Notes (optional)",
                            placeholder="Any additional information about this data")

    st.markdown("---")
    st.subheader("Upload Data File")

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="CSV file with columns: Temperature, Property_Value"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.success("✅ File uploaded successfully!")

            # Display preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Column mapping
            st.subheader("Column Mapping")
            col1, col2 = st.columns(2)

            with col1:
                temp_col = st.selectbox("Temperature Column", df.columns, index=0)
            with col2:
                value_col = st.selectbox("Property Value Column", df.columns,
                                        index=1 if len(df.columns) > 1 else 0)

            # Preview plot
            st.subheader("Preview Plot")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[temp_col],
                y=df[value_col],
                mode='lines+markers',
                name=property_name,
                line=dict(width=2),
                marker=dict(size=6)
            ))
            fig.update_layout(
                title=f"{property_name} vs Temperature - {material_name}",
                xaxis_title=f"Temperature ({temp_unit})",
                yaxis_title=f"{property_name} ({property_unit})",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Save button
            if st.button("💾 Save Data", type="primary"):
                if not material_name:
                    st.error("Please enter a material name!")
                else:
                    # Create metadata
                    metadata = {
                        "material_name": material_name,
                        "composition": material_composition,
                        "property": property_name,
                        "unit": property_unit,
                        "temp_unit": temp_unit,
                        "source": data_source,
                        "notes": notes,
                        "temp_column": temp_col,
                        "value_column": value_col
                    }

                    # Save data
                    safe_name = material_name.replace(" ", "_").replace("/", "-")
                    safe_prop = property_name.replace(" ", "_")

                    data_file = exp_data_dir / f"{safe_name}_{safe_prop}.csv"
                    meta_file = exp_data_dir / f"{safe_name}_{safe_prop}_meta.json"

                    # Rename columns
                    save_df = df[[temp_col, value_col]].copy()
                    save_df.columns = ['Temperature', 'Value']

                    save_df.to_csv(data_file, index=False)

                    with open(meta_file, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    st.success(f"✅ Data saved successfully as: {data_file.name}")
                    st.balloons()

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.info("Make sure your CSV has proper headers and numeric data")

    else:
        st.info("👆 Upload a CSV file to get started")

        st.markdown("---")
        st.subheader("📋 Example CSV Format")

        example_data = pd.DataFrame({
            'Temperature': [20, 100, 200, 300, 400, 500, 600],
            'Yield_Strength': [950, 920, 880, 830, 750, 650, 500]
        })

        st.dataframe(example_data, use_container_width=True)

        csv_example = example_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Example CSV",
            data=csv_example,
            file_name="example_property_data.csv",
            mime="text/csv"
        )

# ============================================================================
# VISUALIZE CURVES MODE
# ============================================================================
elif mode == "📊 Visualize Curves":
    st.header("Visualize Property vs Temperature Curves")

    # Load available datasets
    data_files = list(exp_data_dir.glob("*[!meta].csv"))

    if not data_files:
        st.warning("📭 No experimental data found. Upload data first!")
        st.info("👈 Go to 'Upload Data' to add experimental property curves")
    else:
        st.success(f"✅ Found {len(data_files)} datasets")

        # Select dataset
        selected_file = st.selectbox(
            "Select Dataset",
            [f.stem for f in data_files]
        )

        if selected_file:
            # Load data and metadata
            data_file = exp_data_dir / f"{selected_file}.csv"
            meta_file = exp_data_dir / f"{selected_file}_meta.json"

            df = pd.read_csv(data_file)

            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"material_name": selected_file, "property": "Property"}

            # Display metadata
            st.markdown("### 📋 Material Information")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Material", metadata.get('material_name', 'Unknown'))
                st.metric("Property", metadata.get('property', 'Unknown'))
            with col2:
                st.metric("Unit", metadata.get('unit', 'Unknown'))
                st.metric("Temperature Unit", metadata.get('temp_unit', '°C'))
            with col3:
                if metadata.get('composition'):
                    st.info(f"**Composition:**\n{metadata.get('composition')}")
                if metadata.get('source'):
                    st.info(f"**Source:**\n{metadata.get('source')}")

            st.markdown("---")

            # Plot options
            st.subheader("📊 Visualization Options")

            col1, col2 = st.columns(2)
            with col1:
                plot_style = st.selectbox("Plot Style",
                                         ["Lines + Markers", "Lines Only", "Markers Only"])
                show_grid = st.checkbox("Show Grid", value=True)
            with col2:
                temp_range = st.slider(
                    "Temperature Range",
                    float(df['Temperature'].min()),
                    float(df['Temperature'].max()),
                    (float(df['Temperature'].min()), float(df['Temperature'].max()))
                )

            # Filter data
            mask = (df['Temperature'] >= temp_range[0]) & (df['Temperature'] <= temp_range[1])
            df_filtered = df[mask]

            # Create plot
            st.subheader("📈 Property Curve")

            fig = go.Figure()

            if plot_style == "Lines + Markers":
                mode = 'lines+markers'
            elif plot_style == "Lines Only":
                mode = 'lines'
            else:
                mode = 'markers'

            fig.add_trace(go.Scatter(
                x=df_filtered['Temperature'],
                y=df_filtered['Value'],
                mode=mode,
                name=metadata.get('property', 'Property'),
                line=dict(width=3, color='#1f77b4'),
                marker=dict(size=8, color='#1f77b4')
            ))

            fig.update_layout(
                title=f"{metadata.get('property', 'Property')} vs Temperature - {metadata.get('material_name', 'Material')}",
                xaxis_title=f"Temperature ({metadata.get('temp_unit', '°C')})",
                yaxis_title=f"{metadata.get('property', 'Property')} ({metadata.get('unit', '')})",
                height=500,
                hovermode='x unified',
                showlegend=True
            )

            if show_grid:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            st.markdown("---")
            st.subheader("📊 Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Min Value", f"{df_filtered['Value'].min():.2f} {metadata.get('unit', '')}")
            with col2:
                st.metric("Max Value", f"{df_filtered['Value'].max():.2f} {metadata.get('unit', '')}")
            with col3:
                st.metric("Average", f"{df_filtered['Value'].mean():.2f} {metadata.get('unit', '')}")
            with col4:
                temp_at_max = df_filtered.loc[df_filtered['Value'].idxmax(), 'Temperature']
                st.metric("Temp at Max", f"{temp_at_max:.0f} {metadata.get('temp_unit', '°C')}")

            # Data table
            st.markdown("---")
            st.subheader("📋 Data Table")
            st.dataframe(df_filtered, use_container_width=True)

            # Download
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name=f"{selected_file}_filtered.csv",
                mime="text/csv"
            )

# ============================================================================
# COMPARE MATERIALS MODE
# ============================================================================
elif mode == "📈 Compare Materials":
    st.header("Compare Materials")

    data_files = list(exp_data_dir.glob("*[!meta].csv"))

    if len(data_files) < 2:
        st.warning("📭 Need at least 2 datasets to compare. Upload more data!")
    else:
        st.success(f"✅ Found {len(data_files)} datasets available for comparison")

        # Select datasets to compare
        st.subheader("Select Materials to Compare")

        selected_files = st.multiselect(
            "Choose datasets (up to 5)",
            [f.stem for f in data_files],
            max_selections=5
        )

        if len(selected_files) >= 2:
            # Load all selected datasets
            datasets = []
            for file_stem in selected_files:
                data_file = exp_data_dir / f"{file_stem}.csv"
                meta_file = exp_data_dir / f"{file_stem}_meta.json"

                df = pd.read_csv(data_file)

                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {"material_name": file_stem}

                datasets.append({
                    'name': file_stem,
                    'data': df,
                    'metadata': metadata
                })

            # Create comparison plot
            st.subheader("📊 Comparison Plot")

            fig = go.Figure()

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            for i, dataset in enumerate(datasets):
                fig.add_trace(go.Scatter(
                    x=dataset['data']['Temperature'],
                    y=dataset['data']['Value'],
                    mode='lines+markers',
                    name=dataset['metadata'].get('material_name', dataset['name']),
                    line=dict(width=2, color=colors[i % len(colors)]),
                    marker=dict(size=6)
                ))

            # Use first dataset's metadata for axis labels
            first_meta = datasets[0]['metadata']

            fig.update_layout(
                title="Material Property Comparison",
                xaxis_title=f"Temperature ({first_meta.get('temp_unit', '°C')})",
                yaxis_title=f"{first_meta.get('property', 'Property')} ({first_meta.get('unit', '')})",
                height=600,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Comparison table
            st.markdown("---")
            st.subheader("📋 Property Comparison at Specific Temperature")

            # Find common temperature range
            min_temp = max(d['data']['Temperature'].min() for d in datasets)
            max_temp = min(d['data']['Temperature'].max() for d in datasets)

            selected_temp = st.slider(
                "Select Temperature",
                float(min_temp),
                float(max_temp),
                float((min_temp + max_temp) / 2)
            )

            # Interpolate values at selected temperature
            comparison_data = []
            for dataset in datasets:
                df = dataset['data']
                # Linear interpolation
                value = np.interp(selected_temp, df['Temperature'], df['Value'])

                comparison_data.append({
                    'Material': dataset['metadata'].get('material_name', dataset['name']),
                    'Property Value': f"{value:.2f} {dataset['metadata'].get('unit', '')}",
                    'Temperature': f"{selected_temp:.0f} {dataset['metadata'].get('temp_unit', '°C')}"
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

            # Find best performer
            values = [np.interp(selected_temp, d['data']['Temperature'], d['data']['Value'])
                     for d in datasets]
            best_idx = np.argmax(values)

            st.success(f"🏆 Best performer at {selected_temp:.0f}{datasets[0]['metadata'].get('temp_unit', '°C')}: "
                      f"**{datasets[best_idx]['metadata'].get('material_name', 'Unknown')}** "
                      f"({values[best_idx]:.2f} {datasets[0]['metadata'].get('unit', '')})")

        elif len(selected_files) == 1:
            st.info("Please select at least 2 materials to compare")
        else:
            st.info("👆 Select materials from the dropdown above to start comparison")

# ============================================================================
# MANAGE DATA MODE
# ============================================================================
elif mode == "💾 Manage Data":
    st.header("Manage Experimental Data")

    data_files = list(exp_data_dir.glob("*[!meta].csv"))

    if not data_files:
        st.info("📭 No experimental data found yet")
    else:
        st.success(f"✅ You have {len(data_files)} datasets")

        # Display all datasets
        st.subheader("📚 Your Datasets")

        for data_file in data_files:
            meta_file = exp_data_dir / f"{data_file.stem}_meta.json"

            with st.expander(f"📊 {data_file.stem}"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)

                        st.write(f"**Material:** {metadata.get('material_name', 'Unknown')}")
                        st.write(f"**Property:** {metadata.get('property', 'Unknown')}")
                        st.write(f"**Unit:** {metadata.get('unit', 'Unknown')}")
                        if metadata.get('composition'):
                            st.write(f"**Composition:** {metadata.get('composition')}")
                        if metadata.get('source'):
                            st.write(f"**Source:** {metadata.get('source')}")
                        if metadata.get('notes'):
                            st.write(f"**Notes:** {metadata.get('notes')}")
                    else:
                        st.write("No metadata available")

                    # Show data preview
                    df = pd.read_csv(data_file)
                    st.write(f"**Data points:** {len(df)}")
                    st.dataframe(df.head(), use_container_width=True)

                with col2:
                    if st.button("🗑️ Delete", key=f"del_{data_file.stem}"):
                        data_file.unlink()
                        if meta_file.exists():
                            meta_file.unlink()
                        st.success("Deleted!")
                        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Experimental Data Manager | Upload & Visualize Temperature-Dependent Properties"
    "</div>",
    unsafe_allow_html=True
)
