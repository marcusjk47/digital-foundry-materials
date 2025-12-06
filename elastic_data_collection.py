"""
Elastic Properties Data Collection Module

Fetches materials with elastic tensor data from Materials Project.
Elastic properties are used to predict theoretical strength and
material-specific Hall-Petch parameters.

Author: Digital Foundry Materials Science Toolkit
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from crystal_graph import structure_to_graph, structure_to_graph_with_calphad
from mp_api.client import MPRester


def fetch_elastic_materials_data(
    api_key: str,
    elements: Optional[List[str]] = None,
    max_materials: int = 1000,
    metallic_only: bool = True,
    use_calphad: bool = False
) -> pd.DataFrame:
    """
    Fetch materials with elastic tensor data from Materials Project.

    Args:
        api_key: Materials Project API key
        elements: List of elements to include (e.g., ["Fe", "Ni"])
        max_materials: Maximum number of materials to fetch
        metallic_only: Only fetch metallic materials (band_gap = 0)
        use_calphad: Whether to use CALPHAD-enhanced graphs

    Returns:
        DataFrame with material_id, structure, and elastic properties
    """
    print(f"Fetching materials with elastic data from Materials Project...")
    print(f"  Elements: {elements or 'All'}")
    print(f"  Max materials: {max_materials}")
    print(f"  Metallic only: {metallic_only}")

    all_data = []

    with MPRester(api_key) as mpr:
        # Build search criteria
        search_kwargs = {
            "fields": [
                "material_id",
                "formula_pretty",
                "structure",
                "formation_energy_per_atom",
                "band_gap",
                "is_stable",
                "symmetry",
                "density"
            ],
            "num_chunks": 10
        }

        # Add filters
        if elements:
            search_kwargs["elements"] = elements

        if metallic_only:
            search_kwargs["band_gap"] = (0, 0)

        # Execute search for materials with elastic data
        print("\nSearching for materials with elastic tensors...")
        docs = mpr.materials.elasticity.search(**search_kwargs)

        if not docs:
            print("No materials with elastic data found!")
            return pd.DataFrame()

        # Limit results
        docs = docs[:max_materials]
        print(f"Found {len(docs)} materials with elastic data")

        # Extract data
        for doc in tqdm(docs, desc="Processing materials"):
            try:
                # Get elastic properties
                elastic_data = {
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "structure": doc.structure,
                    "formation_energy_per_atom": doc.formation_energy_per_atom,
                    "band_gap": doc.band_gap,
                    "is_stable": doc.is_stable,
                    "density": doc.density
                }

                # Extract elastic moduli
                if hasattr(doc, 'homogeneous_poisson'):
                    elastic_data["poisson_ratio"] = doc.homogeneous_poisson

                if hasattr(doc, 'universal_anisotropy'):
                    elastic_data["anisotropy"] = doc.universal_anisotropy

                # Get moduli (most important for strength prediction)
                if hasattr(doc, 'k_voigt_reuss_hill'):
                    elastic_data["bulk_modulus"] = doc.k_voigt_reuss_hill  # GPa

                if hasattr(doc, 'g_voigt_reuss_hill'):
                    elastic_data["shear_modulus"] = doc.g_voigt_reuss_hill  # GPa

                if hasattr(doc, 'e_total'):
                    elastic_data["youngs_modulus"] = doc.e_total  # GPa

                # Only add if we have at least shear modulus (critical for strength)
                if "shear_modulus" in elastic_data and elastic_data["shear_modulus"] is not None:
                    all_data.append(elastic_data)

            except Exception as e:
                print(f"Warning: Failed to process {doc.material_id}: {e}")
                continue

    df = pd.DataFrame(all_data)
    print(f"\nSuccessfully fetched {len(df)} materials with elastic properties")

    if len(df) > 0:
        print(f"\nElastic property statistics:")
        if "shear_modulus" in df.columns:
            print(f"  Shear modulus (G): {df['shear_modulus'].mean():.1f} ± {df['shear_modulus'].std():.1f} GPa")
        if "bulk_modulus" in df.columns:
            print(f"  Bulk modulus (K): {df['bulk_modulus'].mean():.1f} ± {df['bulk_modulus'].std():.1f} GPa")
        if "youngs_modulus" in df.columns:
            print(f"  Young's modulus (E): {df['youngs_modulus'].mean():.1f} ± {df['youngs_modulus'].std():.1f} GPa")

    return df


def convert_to_elastic_graphs(
    df: pd.DataFrame,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    use_calphad: bool = False,
    save_path: Optional[str] = None
) -> List[Data]:
    """
    Convert structures to graphs with elastic properties as targets.

    Args:
        df: DataFrame with 'structure' and elastic property columns
        cutoff: Cutoff distance for edges
        max_neighbors: Maximum neighbors per atom
        use_calphad: Use CALPHAD-enhanced graph construction
        save_path: Path to save graph dataset (optional)

    Returns:
        List of PyTorch Geometric Data objects with elastic property targets
    """
    print(f"\nConverting {len(df)} structures to graphs...")
    print(f"  Cutoff: {cutoff} Angstrom")
    print(f"  Max neighbors: {max_neighbors}")
    print(f"  CALPHAD features: {'Enabled' if use_calphad else 'Disabled'}")

    graphs = []
    failed_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting to graphs"):
        try:
            # Convert structure to graph
            if use_calphad:
                graph = structure_to_graph_with_calphad(
                    row["structure"],
                    cutoff=cutoff,
                    max_neighbors=max_neighbors,
                    use_element_features=True
                )
            else:
                graph = structure_to_graph(
                    row["structure"],
                    cutoff=cutoff,
                    max_neighbors=max_neighbors
                )

            # Add elastic properties as multi-task targets
            targets = []
            target_names = []

            if "shear_modulus" in row and pd.notna(row["shear_modulus"]):
                targets.append(row["shear_modulus"])
                target_names.append("shear_modulus")

            if "bulk_modulus" in row and pd.notna(row["bulk_modulus"]):
                targets.append(row["bulk_modulus"])
                target_names.append("bulk_modulus")

            if "youngs_modulus" in row and pd.notna(row["youngs_modulus"]):
                targets.append(row["youngs_modulus"])
                target_names.append("youngs_modulus")

            # Skip if no targets
            if not targets:
                failed_count += 1
                continue

            # Store as tensor
            graph.y = torch.tensor(targets, dtype=torch.float)
            graph.target_names = target_names

            # Add metadata
            graph.material_id = row["material_id"]
            graph.formula_str = row["formula"]

            # Add other properties
            if "formation_energy_per_atom" in row:
                graph.formation_energy = row["formation_energy_per_atom"]
            if "density" in row:
                graph.density = row["density"]
            if "poisson_ratio" in row and pd.notna(row["poisson_ratio"]):
                graph.poisson_ratio = row["poisson_ratio"]

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
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(graphs, f)

        file_size = save_path.stat().st_size / (1024 * 1024)  # MB
        print(f"\nSaved {len(graphs)} graphs to {save_path}")
        print(f"File size: {file_size:.2f} MB")

    return graphs


if __name__ == "__main__":
    """Example usage"""
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("MP_API_KEY")

    if not api_key:
        print("Error: MP_API_KEY not found in environment")
        exit(1)

    # Example: Fetch elastic data for common structural metals
    print("Fetching elastic data for structural metals...")
    df = fetch_elastic_materials_data(
        api_key=api_key,
        elements=["Fe", "Ni", "Cr", "Al", "Ti"],
        max_materials=500,
        metallic_only=True
    )

    if df.empty:
        print("No materials found!")
        exit(1)

    # Convert to graphs
    graphs = convert_to_elastic_graphs(
        df,
        cutoff=8.0,
        use_calphad=True,  # Use CALPHAD features
        save_path="datasets/elastic_properties_calphad.pkl"
    )

    # Show sample
    if graphs:
        sample = graphs[0]
        print(f"\nSample graph:")
        print(f"  Formula: {sample.formula}")
        print(f"  Targets: {sample.target_names}")
        print(f"  Values: {sample.y.numpy()}")
        print(f"  Node features: {sample.x.shape}")
        print(f"  Edge features: {sample.edge_attr.shape}")

    print("\nElastic properties dataset ready for training!")
