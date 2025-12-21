"""
Streamlit Prediction Utilities

Helper functions for multi-property GNN predictions in Streamlit.

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import streamlit as st
from pathlib import Path
from typing import Dict, Tuple, Optional
from pymatgen.core import Structure

from crystal_graph import structure_to_graph, structure_to_graph_with_calphad
from gnn_model import CGCNN
from gnn_model_calphad import CGCNN_CALPHAD_Regressor
from gnn_model_multitask import CGCNN_MultiTask, CGCNN_MultiTask_CALPHAD, get_property_info


def load_model_and_predict(
    structure: Structure,
    checkpoint_path: Path,
    node_dim: int = 64,
    hidden_dim: int = 128,
    n_conv: int = 3,
    n_hidden: int = 2
) -> Tuple[Dict[str, float], bool, bool]:
    """
    Load model and make predictions for a structure.

    Args:
        structure: Pymatgen Structure
        checkpoint_path: Path to model checkpoint
        node_dim: Node feature dimension
        hidden_dim: Hidden dimension
        n_conv: Number of convolution layers
        n_hidden: Number of hidden layers

    Returns:
        (predictions_dict, is_multitask, uses_calphad)
    """
    # Convert to graph (try CALPHAD first)
    try:
        graph = structure_to_graph_with_calphad(structure, cutoff=8.0, max_neighbors=12)
        use_calphad = True
    except:
        graph = structure_to_graph(structure, cutoff=8.0, max_neighbors=12)
        use_calphad = False

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Detect model type
    properties = checkpoint.get('properties', None)
    is_multitask = properties is not None and len(properties) > 1

    if is_multitask:
        # Multi-task model
        if use_calphad:
            model = CGCNN_MultiTask_CALPHAD(
                input_node_dim=13,
                input_edge_dim=2,
                node_feature_dim=node_dim,
                edge_feature_dim=32,
                hidden_dim=hidden_dim,
                n_conv=n_conv,
                n_hidden=n_hidden,
                properties=properties
            )
        else:
            model = CGCNN_MultiTask(
                node_feature_dim=node_dim,
                edge_feature_dim=1,
                hidden_dim=hidden_dim,
                n_conv=n_conv,
                n_hidden=n_hidden,
                properties=properties
            )
    else:
        # Single-task model
        if use_calphad:
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

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Make prediction
    with torch.no_grad():
        prediction = model(graph)

    # Parse predictions
    predictions_dict = {}

    if isinstance(prediction, dict):
        # Multi-task or CALPHAD dict output
        for prop_name, value in prediction.items():
            if isinstance(value, torch.Tensor):
                predictions_dict[prop_name] = value.item()
            else:
                predictions_dict[prop_name] = float(value)
    else:
        # Single-task tensor output
        predictions_dict['formation_energy_per_atom'] = prediction.item()

    return predictions_dict, is_multitask, use_calphad


def display_predictions(predictions: Dict[str, float], formula: str, composition_info: str = ""):
    """
    Display predictions in Streamlit with proper formatting.

    Args:
        predictions: Dict mapping property names to values
        formula: Chemical formula
        composition_info: Additional composition information
    """
    st.markdown("---")
    st.markdown("### 🎯 Prediction Results")

    # Formula info
    if composition_info:
        st.markdown(f"**{composition_info}** ({formula})")
    else:
        st.markdown(f"**Formula:** {formula}")

    st.markdown("---")

    # Display each property
    n_props = len(predictions)
    n_cols = min(3, n_props)

    # Group properties
    rows = [list(predictions.items())[i:i+n_cols] for i in range(0, n_props, n_cols)]

    for row in rows:
        cols = st.columns(len(row))

        for col, (prop_name, value) in zip(cols, row):
            prop_info = get_property_info(prop_name)

            with col:
                # Determine delta for stability indicators
                delta = None
                delta_color = "off"

                if 'formation_energy' in prop_name or 'energy_above_hull' in prop_name:
                    if value < -0.3:
                        delta = "Highly Stable"
                        delta_color = "normal"
                    elif value < 0:
                        delta = "Stable"
                        delta_color = "normal"
                    elif value < 0.2:
                        delta = "Marginally Unstable"
                        delta_color = "inverse"
                    else:
                        delta = "Unstable"
                        delta_color = "inverse"
                elif 'band_gap' in prop_name:
                    if value < 0.1:
                        delta = "Metallic"
                        delta_color = "normal"
                    elif value < 2.0:
                        delta = "Semiconductor"
                        delta_color = "normal"
                    else:
                        delta = "Insulator"
                        delta_color = "normal"

                st.metric(
                    prop_info['name'],
                    f"{value:.4f} {prop_info['unit']}",
                    delta=delta,
                    delta_color=delta_color
                )

    # Detailed interpretations
    st.markdown("---")
    st.markdown("### 💡 Interpretations")

    for prop_name, value in predictions.items():
        prop_info = get_property_info(prop_name)

        with st.expander(f"📊 {prop_info['name']}"):
            st.write(f"**Value:** {value:.4f} {prop_info['unit']}")
            st.write(f"**Description:** {prop_info['description']}")

            # Property-specific interpretation
            if 'formation_energy' in prop_name:
                if value < -0.3:
                    st.success("✅ **Highly Stable** - This material should form readily and remain stable.")
                elif value < 0:
                    st.info("✓ **Stable** - This material is thermodynamically stable.")
                elif value < 0.2:
                    st.warning("⚠️ **Marginally Unstable** - May form under special conditions.")
                else:
                    st.error("❌ **Unstable** - Likely to decompose into constituent elements.")

            elif 'energy_above_hull' in prop_name:
                if value < 0.02:
                    st.success("✅ **Stable** - Material is on or very close to convex hull.")
                elif value < 0.1:
                    st.info("✓ **Metastable** - Material may exist but is not ground state.")
                else:
                    st.warning("⚠️ **Unstable** - Thermodynamically unstable relative to decomposition.")

            elif 'band_gap' in prop_name:
                if value < 0.1:
                    st.info("⚡ **Metallic** - Good electrical conductor.")
                elif value < 2.0:
                    st.info("🔌 **Semiconductor** - Useful for electronic devices.")
                else:
                    st.info("🔒 **Insulator** - Poor electrical conductor.")

            elif 'density' in prop_name:
                if value < 3.0:
                    st.info("🪶 **Low density** - Lightweight material.")
                elif value < 10.0:
                    st.info("⚖️ **Medium density** - Typical for most metals.")
                else:
                    st.info("⚓ **High density** - Heavy material (e.g., heavy metals).")

            elif 'modulus' in prop_name.lower():
                if value > 200:
                    st.success("💪 **Very stiff** - Highly resistant to deformation.")
                elif value > 100:
                    st.info("💪 **Stiff** - Good mechanical strength.")
                elif value > 50:
                    st.info("⚖️ **Moderate stiffness**")
                else:
                    st.warning("🥎 **Soft** - Easily deformable.")

            elif 'magnetization' in prop_name:
                if value > 2.0:
                    st.info("🧲 **Strongly Magnetic** - Ferromagnetic behavior.")
                elif value > 0.5:
                    st.info("🧲 **Weakly Magnetic** - Some magnetic character.")
                else:
                    st.info("Non-magnetic or very weakly magnetic.")

            elif 'efermi' in prop_name or 'fermi' in prop_name.lower():
                st.info(f"⚡ **Fermi Energy:** {value:.2f} eV")
                st.write("The Fermi energy indicates the electronic structure and conductivity characteristics.")

            elif 'volume' in prop_name:
                st.info(f"📦 **Atomic Volume:** {value:.2f} ų/atom")
                if value < 15:
                    st.write("Compact atomic packing - typical for dense materials.")
                elif value > 30:
                    st.write("Loose atomic packing - lower density structure.")

            # Show typical range if available
            if prop_info['typical_range'][0] is not None:
                min_val, max_val = prop_info['typical_range']
                st.write(f"**Typical range:** {min_val} to {max_val} {prop_info['unit']}")

                # Show where this value falls
                if min_val <= value <= max_val:
                    st.write("✓ Value is within typical range")
                else:
                    st.write("⚠️ Value is outside typical range")


def display_property_table(predictions: Dict[str, float]):
    """Display predictions as a table."""
    import pandas as pd

    rows = []
    for prop_name, value in predictions.items():
        prop_info = get_property_info(prop_name)
        rows.append({
            'Property': prop_info['name'],
            'Value': f"{value:.4f}",
            'Unit': prop_info['unit'],
            'Description': prop_info['description']
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="gnn_predictions.csv",
        mime="text/csv"
    )
