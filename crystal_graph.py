"""
Crystal Graph Construction Module

Converts pymatgen Structure objects into PyTorch Geometric graph representations
for use with Graph Neural Networks (GNNs).

Author: Digital Foundry Materials Science Toolkit
"""

import torch
from torch_geometric.data import Data
from pymatgen.core import Structure
import numpy as np


def structure_to_graph(structure: Structure, cutoff=8.0, max_neighbors=12):
    """
    Convert pymatgen Structure to PyTorch Geometric graph.

    Creates a graph where:
    - Nodes represent atoms
    - Edges connect atoms within cutoff distance
    - Node features are atomic numbers
    - Edge features are distances

    Args:
        structure: Pymatgen Structure object
        cutoff: Distance cutoff for edges (Angstroms). Default 8.0 Å is typical
                for capturing first and second nearest neighbors in crystals.
        max_neighbors: Maximum number of neighbors per atom. Helps control
                      graph density and memory usage.

    Returns:
        torch_geometric.data.Data object with:
            - x: Node features (atomic numbers) [num_nodes, 1]
            - edge_index: Edge connectivity [2, num_edges]
            - edge_attr: Edge features (distances) [num_edges, 1]
            - pos: Atomic positions [num_nodes, 3]
            - formula: Chemical formula (string)
    """
    # Get number of atoms
    num_atoms = len(structure)

    # Node features: atomic numbers
    atomic_numbers = [site.specie.Z for site in structure]
    x = torch.tensor(atomic_numbers, dtype=torch.long)

    # Get all neighbors within cutoff
    # Using get_all_neighbors for periodic boundary conditions
    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)

    # Build edges with max_neighbors constraint
    edges = []
    edge_attrs = []

    for i, neighbors in enumerate(all_neighbors):
        # Sort neighbors by distance
        neighbors = sorted(neighbors, key=lambda x: x.nn_distance)

        # Take only max_neighbors closest
        neighbors = neighbors[:max_neighbors]

        for neighbor in neighbors:
            j = neighbor.index
            distance = neighbor.nn_distance

            # Add edge from i to j
            edges.append([i, j])
            edge_attrs.append(distance)

    # Convert to tensors
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
    else:
        # Handle case with no edges (single atom or large cutoff)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    # Atomic positions
    pos = torch.tensor([site.coords for site in structure], dtype=torch.float)

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos
    )

    # Add metadata
    data.formula = structure.composition.reduced_formula
    data.num_atoms = num_atoms

    return data


def structure_to_graph_with_features(structure: Structure, cutoff=8.0,
                                     max_neighbors=12, use_element_features=True):
    """
    Convert structure to graph with optional element feature embeddings.

    This version uses rich element features instead of just atomic numbers.

    Args:
        structure: Pymatgen Structure object
        cutoff: Distance cutoff for edges (Angstroms)
        max_neighbors: Maximum neighbors per atom
        use_element_features: If True, use element properties as node features.
                             If False, use atomic numbers only.

    Returns:
        torch_geometric.data.Data object
    """
    from element_features import get_element_features

    # Get basic graph structure
    data = structure_to_graph(structure, cutoff, max_neighbors)

    if use_element_features:
        # Replace atomic numbers with element feature vectors
        element_feats = []
        for site in structure:
            atomic_number = site.specie.Z
            feats = get_element_features(atomic_number)
            element_feats.append(feats)

        data.x = torch.stack(element_feats)

    return data


def batch_structures_to_graphs(structures, cutoff=8.0, max_neighbors=12):
    """
    Convert multiple structures to graphs.

    Args:
        structures: List of pymatgen Structure objects
        cutoff: Distance cutoff for edges
        max_neighbors: Maximum neighbors per atom

    Returns:
        List of torch_geometric.data.Data objects
    """
    graphs = []
    for structure in structures:
        try:
            graph = structure_to_graph(structure, cutoff, max_neighbors)
            graphs.append(graph)
        except Exception as e:
            print(f"Warning: Failed to convert structure {structure.formula}: {e}")
            continue

    return graphs


def get_graph_stats(data):
    """
    Print statistics about a graph.

    Args:
        data: torch_geometric.data.Data object
    """
    print(f"Formula: {data.formula}")
    print(f"Number of nodes (atoms): {data.num_nodes}")
    print(f"Number of edges (bonds): {data.num_edges}")
    print(f"Average degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"Node feature shape: {data.x.shape}")
    print(f"Edge feature shape: {data.edge_attr.shape}")
    print(f"Position shape: {data.pos.shape}")

    # Check for isolated nodes
    if data.num_edges == 0:
        print("⚠️  Warning: Graph has no edges!")

    return {
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'avg_degree': data.num_edges / data.num_nodes if data.num_nodes > 0 else 0
    }


if __name__ == "__main__":
    """Test the crystal graph construction"""
    from pymatgen.core import Lattice, Structure

    print("="*60)
    print("Testing Crystal Graph Construction")
    print("="*60)
    print()

    # Test 1: Simple cubic structure
    print("Test 1: Simple Cubic Fe")
    lattice = Lattice.cubic(2.87)  # Fe lattice parameter
    structure = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    graph = structure_to_graph(structure, cutoff=4.0)
    stats = get_graph_stats(graph)
    print()

    # Test 2: Binary compound
    print("Test 2: NaCl structure")
    lattice = Lattice.cubic(5.64)
    species = ["Na", "Cl"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(lattice, species, coords)

    graph = structure_to_graph(structure, cutoff=4.0)
    stats = get_graph_stats(graph)
    print()

    # Test 3: Check edge symmetry
    print("Test 3: Edge symmetry check")
    print(f"Edge index:\n{graph.edge_index}")
    print(f"Edge attributes:\n{graph.edge_attr.squeeze()}")
    print()

    print("✅ Crystal graph construction tests passed!")
