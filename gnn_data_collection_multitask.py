"""
Multi-Task GNN Data Collection

Collects and processes materials data for multi-property prediction.

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
from torch_geometric.data import Data

from crystal_graph import structure_to_graph, structure_to_graph_with_calphad


def convert_to_multitask_graphs(
    df: pd.DataFrame,
    target_properties: List[str],
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    use_calphad: bool = False,
    tdb_path: Optional[str] = None,
    save_path: Optional[str] = None
) -> List[Data]:
    """
    Convert structures to graphs for multi-task learning.

    Args:
        df: DataFrame with 'structure' column and property columns
        target_properties: List of property names to predict
        cutoff: Cutoff distance for edges
        max_neighbors: Maximum neighbors per atom
        use_calphad: Use CALPHAD-enhanced features
        tdb_path: Path to TDB file
        save_path: Path to save dataset

    Returns:
        List of PyTorch Geometric Data objects with multi-task targets
    """
    print(f"\nConverting {len(df)} structures to multi-task graphs...")
    print(f"  Target properties: {target_properties}")
    print(f"  Cutoff: {cutoff} Å")
    print(f"  Max neighbors: {max_neighbors}")
    print(f"  CALPHAD features: {'Enabled' if use_calphad else 'Disabled'}")

    graphs = []
    failed_count = 0
    property_missing_counts = {prop: 0 for prop in target_properties}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        try:
            # Convert structure to graph
            if use_calphad:
                graph = structure_to_graph_with_calphad(
                    row["structure"],
                    cutoff=cutoff,
                    max_neighbors=max_neighbors,
                    tdb_path=tdb_path,
                    use_element_features=True
                )
            else:
                graph = structure_to_graph(
                    row["structure"],
                    cutoff=cutoff,
                    max_neighbors=max_neighbors
                )

            # Collect all target properties
            targets = []
            valid_properties = []
            has_any_target = False

            for prop in target_properties:
                if prop in row and pd.notna(row[prop]):
                    targets.append(float(row[prop]))
                    valid_properties.append(prop)
                    has_any_target = True
                else:
                    targets.append(float('nan'))  # Mark as missing
                    property_missing_counts[prop] += 1

            # Skip if no valid targets
            if not has_any_target:
                failed_count += 1
                continue

            # Store as multi-dimensional target
            graph.y = torch.tensor(targets, dtype=torch.float)
            graph.target_properties = target_properties
            graph.valid_properties = valid_properties

            # Add metadata
            graph.material_id = row["material_id"]
            graph.formula_str = row["formula"]

            # Store all available properties as attributes
            for prop in target_properties:
                if prop in row and pd.notna(row[prop]):
                    setattr(graph, prop, float(row[prop]))

            # Additional metadata
            if "density" in row:
                graph.density_actual = float(row["density"])
            if "volume" in row:
                graph.volume_actual = float(row["volume"])
            if "nsites" in row:
                graph.num_sites = int(row["nsites"])

            graph.has_calphad_features = use_calphad

            graphs.append(graph)

        except Exception as e:
            print(f"\nError converting {row.get('material_id', 'unknown')}: {e}")
            failed_count += 1
            continue

    # Summary
    print(f"\n✅ Successfully converted {len(graphs)} materials")
    if failed_count > 0:
        print(f"❌ Failed to convert {failed_count} materials")

    print(f"\nProperty coverage:")
    for prop, missing_count in property_missing_counts.items():
        available = len(graphs) - missing_count
        coverage = (available / len(graphs) * 100) if len(graphs) > 0 else 0
        print(f"  {prop}: {available}/{len(graphs)} ({coverage:.1f}%)")

    # Save if requested
    if save_path:
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(graphs, f)
        print(f"\n💾 Saved dataset to: {save_path}")

    return graphs


# Available properties from Materials Project
AVAILABLE_PROPERTIES = {
    'formation_energy_per_atom': 'Formation Energy (eV/atom)',
    'energy_above_hull': 'Energy Above Hull (eV/atom)',
    'band_gap': 'Band Gap (eV)',
    'density': 'Density (g/cm³)',
    'volume': 'Volume per atom (ų)',
    'efermi': 'Fermi Energy (eV)',
    'total_magnetization': 'Total Magnetization (μB)',
    'bulk_modulus': 'Bulk Modulus (GPa)',
    'shear_modulus': 'Shear Modulus (GPa)',
}


def get_available_properties_in_df(df: pd.DataFrame) -> List[str]:
    """
    Check which properties are available in the dataframe.

    Args:
        df: DataFrame to check

    Returns:
        List of available property names
    """
    available = []
    for prop_name in AVAILABLE_PROPERTIES.keys():
        if prop_name in df.columns:
            # Check if at least some values are not NaN
            if df[prop_name].notna().any():
                available.append(prop_name)

    return available


def print_property_summary(df: pd.DataFrame):
    """Print summary of available properties in dataset."""
    print("\n" + "="*70)
    print("Property Coverage Summary")
    print("="*70)

    for prop_name, prop_desc in AVAILABLE_PROPERTIES.items():
        if prop_name in df.columns:
            total = len(df)
            available = df[prop_name].notna().sum()
            coverage = (available / total * 100) if total > 0 else 0

            status = "✅" if coverage > 80 else "⚠️" if coverage > 50 else "❌"

            print(f"{status} {prop_desc:<40} {available:>5}/{total:<5} ({coverage:>5.1f}%)")

    print("="*70)


if __name__ == "__main__":
    """Test multi-task data collection."""
    import os
    from dotenv import load_dotenv
    from gnn_data_collection import fetch_materials_data

    load_dotenv()
    api_key = os.environ.get("MP_API_KEY")

    if not api_key:
        print("Error: MP_API_KEY not found")
        exit(1)

    print("="*70)
    print("Testing Multi-Task Data Collection")
    print("="*70)

    # Fetch data
    df = fetch_materials_data(
        api_key=api_key,
        elements=["Fe", "Ni"],
        max_materials=50,
        metallic_only=False  # Get both metals and insulators for band gap variety
    )

    # Check available properties
    print_property_summary(df)
    available_props = get_available_properties_in_df(df)

    print(f"\nAvailable for training: {available_props}")

    # Convert to multi-task graphs
    target_props = ['formation_energy_per_atom', 'band_gap', 'density']

    graphs = convert_to_multitask_graphs(
        df,
        target_properties=target_props,
        use_calphad=False
    )

    print(f"\n✅ Created {len(graphs)} multi-task graphs")
    print(f"   Each graph predicts: {target_props}")

    # Inspect first graph
    if graphs:
        g = graphs[0]
        print(f"\nSample graph:")
        print(f"  Material: {g.formula_str} ({g.material_id})")
        print(f"  Targets: {g.y}")
        print(f"  Properties: {g.target_properties}")
        print(f"  Valid properties: {g.valid_properties}")
