"""
GNN Data Collection Module

Fetches crystal structures and properties from Materials Project,
converts them to graph representations, and prepares datasets for training.

Author: Digital Foundry Materials Science Toolkit
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from crystal_graph import structure_to_graph, structure_to_graph_with_calphad
from mp_api.client import MPRester


def fetch_materials_data(
    api_key: str,
    elements: Optional[List[str]] = None,
    max_materials: int = 1000,
    metallic_only: bool = True,
    stable_only: bool = False,
    properties: Optional[List[str]] = None,
    chemsys: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch materials data from Materials Project.

    Args:
        api_key: Materials Project API key
        elements: List of elements to include (e.g., ["Fe", "Ni"])
        max_materials: Maximum number of materials to fetch
        metallic_only: Only fetch metallic materials (band_gap = 0)
        stable_only: Only fetch thermodynamically stable materials
        properties: List of properties to fetch (default: formation_energy_per_atom)
        chemsys: Chemical system string (e.g., "Fe-Ni")

    Returns:
        DataFrame with material_id, formula, structure, and properties
    """
    if properties is None:
        properties = ["formation_energy_per_atom"]

    print(f"Fetching materials from Materials Project...")
    print(f"  Elements: {elements or chemsys or 'All'}")
    print(f"  Max materials: {max_materials}")
    print(f"  Metallic only: {metallic_only}")
    print(f"  Stable only: {stable_only}")

    all_data = []

    with MPRester(api_key) as mpr:
        # Build search criteria
        search_kwargs = {
            "fields": [
                "material_id",
                "formula_pretty",
                "structure",
                "formation_energy_per_atom",
                "energy_above_hull",
                "band_gap",
                "is_stable",
                "symmetry",
                "density",
                "volume",
                "nsites"
            ]
        }

        # Add filters
        if elements:
            search_kwargs["elements"] = elements
        elif chemsys:
            search_kwargs["chemsys"] = chemsys

        if metallic_only:
            search_kwargs["band_gap"] = (0, 0)

        if stable_only:
            search_kwargs["is_stable"] = True

        # Execute search
        docs = mpr.materials.summary.search(**search_kwargs)

        if not docs:
            print("No materials found!")
            return pd.DataFrame()

        # Limit results
        docs = docs[:max_materials]
        print(f"Found {len(docs)} materials")

        # Extract data
        for doc in tqdm(docs, desc="Processing materials"):
            try:
                mat_data = {
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "structure": doc.structure,
                    "formation_energy_per_atom": doc.formation_energy_per_atom,
                    "energy_above_hull": doc.energy_above_hull,
                    "band_gap": doc.band_gap,
                    "is_stable": doc.is_stable,
                    "density": doc.density,
                    "volume": doc.volume,
                    "nsites": doc.nsites
                }

                # Add space group if available
                if hasattr(doc, "symmetry") and doc.symmetry:
                    mat_data["space_group"] = doc.symmetry.number
                    mat_data["crystal_system"] = doc.symmetry.crystal_system

                all_data.append(mat_data)

            except Exception as e:
                print(f"Warning: Failed to process {doc.material_id}: {e}")
                continue

    df = pd.DataFrame(all_data)
    print(f"Successfully fetched {len(df)} materials")

    return df


def convert_to_graphs(
    df: pd.DataFrame,
    target_property: str = "formation_energy_per_atom",
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    use_calphad: bool = False,
    tdb_path: Optional[str] = None,
    save_path: Optional[str] = None
) -> List[Data]:
    """
    Convert structures to graph representations.

    Args:
        df: DataFrame with 'structure' column
        target_property: Property to use as target (y)
        cutoff: Cutoff distance for edges
        max_neighbors: Maximum neighbors per atom
        use_calphad: If True, use CALPHAD-enhanced graph construction
                    (13D node features, 2D edge features)
        tdb_path: Path to TDB file for CALPHAD features (optional)
        save_path: Path to save graph dataset (optional)

    Returns:
        List of PyTorch Geometric Data objects
    """
    print(f"\nConverting {len(df)} structures to graphs...")
    print(f"  Target property: {target_property}")
    print(f"  Cutoff: {cutoff} Å")
    print(f"  Max neighbors: {max_neighbors}")
    print(f"  CALPHAD features: {'Enabled' if use_calphad else 'Disabled'}")

    graphs = []
    failed_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting to graphs"):
        try:
            # Convert structure to graph (standard or CALPHAD-enhanced)
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

            # Add target property
            if target_property in row and pd.notna(row[target_property]):
                graph.y = torch.tensor([row[target_property]], dtype=torch.float)
            else:
                # Skip if target is missing
                failed_count += 1
                continue

            # Add metadata
            graph.material_id = row["material_id"]
            graph.formula_str = row["formula"]

            # Add additional properties as attributes
            if "band_gap" in row:
                graph.band_gap = row["band_gap"]
            if "energy_above_hull" in row:
                graph.energy_above_hull = row["energy_above_hull"]
            if "density" in row:
                graph.density = row["density"]

            graphs.append(graph)

        except Exception as e:
            failed_count += 1
            print(f"Warning: Failed to convert {row['material_id']}: {e}")
            continue

    print(f"\nSuccessfully converted {len(graphs)} structures")
    if failed_count > 0:
        print(f"Failed: {failed_count} structures")

    # Save if requested
    if save_path:
        save_graph_dataset(graphs, save_path)

    return graphs


def save_graph_dataset(graphs: List[Data], save_path: str):
    """
    Save graph dataset to disk.

    Args:
        graphs: List of Data objects
        save_path: Path to save file (.pkl)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(graphs, f)

    file_size = save_path.stat().st_size / (1024 * 1024)  # MB
    print(f"\nSaved {len(graphs)} graphs to {save_path}")
    print(f"File size: {file_size:.2f} MB")


def load_graph_dataset(load_path: str) -> List[Data]:
    """
    Load graph dataset from disk.

    Args:
        load_path: Path to saved file (.pkl)

    Returns:
        List of Data objects
    """
    with open(load_path, "rb") as f:
        graphs = pickle.load(f)

    print(f"Loaded {len(graphs)} graphs from {load_path}")
    return graphs


def get_dataset_statistics(graphs: List[Data]) -> Dict:
    """
    Calculate statistics about the dataset.

    Args:
        graphs: List of Data objects

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "num_graphs": len(graphs),
        "num_nodes": [],
        "num_edges": [],
        "target_values": []
    }

    for graph in graphs:
        stats["num_nodes"].append(graph.num_nodes)
        stats["num_edges"].append(graph.num_edges)
        if hasattr(graph, "y"):
            stats["target_values"].append(graph.y.item())

    stats["avg_nodes"] = np.mean(stats["num_nodes"])
    stats["avg_edges"] = np.mean(stats["num_edges"])
    stats["min_nodes"] = np.min(stats["num_nodes"])
    stats["max_nodes"] = np.max(stats["num_nodes"])

    if stats["target_values"]:
        stats["target_mean"] = np.mean(stats["target_values"])
        stats["target_std"] = np.std(stats["target_values"])
        stats["target_min"] = np.min(stats["target_values"])
        stats["target_max"] = np.max(stats["target_values"])

    return stats


def print_dataset_info(stats: Dict):
    """Print dataset statistics in a readable format."""
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total graphs: {stats['num_graphs']}")
    print(f"\nGraph size:")
    print(f"  Average nodes: {stats['avg_nodes']:.1f}")
    print(f"  Average edges: {stats['avg_edges']:.1f}")
    print(f"  Min nodes: {stats['min_nodes']}")
    print(f"  Max nodes: {stats['max_nodes']}")

    if "target_mean" in stats:
        print(f"\nTarget property:")
        print(f"  Mean: {stats['target_mean']:.4f}")
        print(f"  Std: {stats['target_std']:.4f}")
        print(f"  Min: {stats['target_min']:.4f}")
        print(f"  Max: {stats['target_max']:.4f}")
    print("="*60)


if __name__ == "__main__":
    """Example usage"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("MP_API_KEY")

    if not api_key:
        print("Error: MP_API_KEY not found in environment")
        exit(1)

    # Example 1: Fetch Fe-Ni alloys
    print("Example 1: Fetching Fe-Ni alloys")
    df = fetch_materials_data(
        api_key=api_key,
        elements=["Fe", "Ni"],
        max_materials=100,
        metallic_only=True
    )

    # Convert to graphs
    graphs = convert_to_graphs(
        df,
        target_property="formation_energy_per_atom",
        cutoff=8.0,
        save_path="datasets/fe_ni_graphs.pkl"
    )

    # Show statistics
    stats = get_dataset_statistics(graphs)
    print_dataset_info(stats)

    print("\nDataset ready for training!")

    # Example 2: Fetch with CALPHAD features
    print("\n\n" + "="*60)
    print("Example 2: Fetching with CALPHAD-enhanced features")
    print("="*60)
    df_calphad = fetch_materials_data(
        api_key=api_key,
        elements=["Fe", "Ni"],
        max_materials=20,
        metallic_only=True
    )

    # Convert to CALPHAD-enhanced graphs
    graphs_calphad = convert_to_graphs(
        df_calphad,
        target_property="formation_energy_per_atom",
        cutoff=8.0,
        use_calphad=True,  # Enable CALPHAD features
        save_path="datasets/fe_ni_graphs_calphad.pkl"
    )

    # Show enhanced features
    if graphs_calphad:
        sample_graph = graphs_calphad[0]
        print(f"\nSample CALPHAD-enhanced graph:")
        print(f"  Formula: {sample_graph.formula}")
        print(f"  Node features shape: {sample_graph.x.shape} (Expected: [num_nodes, 13])")
        print(f"  Edge features shape: {sample_graph.edge_attr.shape} (Expected: [num_edges, 2])")
        print(f"  Target: {sample_graph.y.item():.4f} eV/atom")

    stats_calphad = get_dataset_statistics(graphs_calphad)
    print_dataset_info(stats_calphad)

    print("\nCALPHAD-enhanced dataset ready for training!")
