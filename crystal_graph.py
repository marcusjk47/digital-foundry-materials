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
from typing import Optional


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


def structure_to_graph_with_calphad(
    structure: Structure,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    tdb_path: Optional[str] = None,
    use_element_features: bool = True
) -> Data:
    """
    Convert structure to graph with CALPHAD thermodynamic features.

    Combines atomic structure information with CALPHAD-derived thermodynamic
    properties for enhanced GNN predictions.

    Args:
        structure: Pymatgen Structure object
        cutoff: Distance cutoff for edges (Angstroms)
        max_neighbors: Maximum neighbors per atom
        tdb_path: Path to TDB file (optional, uses defaults if None)
        use_element_features: If True, include element property features too

    Returns:
        torch_geometric.data.Data object with enhanced features
    """
    from calphad_features import CALPHADFeatureExtractor
    from element_features import get_element_features

    # Get basic graph structure
    data = structure_to_graph(structure, cutoff, max_neighbors)

    # Initialize CALPHAD extractor
    calphad_extractor = CALPHADFeatureExtractor(tdb_path)

    # Build enhanced node features
    enhanced_node_features = []

    for site in structure:
        elem_symbol = str(site.specie.symbol)
        atomic_number = site.specie.Z

        # Start with atomic number
        feats = [float(atomic_number)]

        # Add element features if requested (9 features)
        if use_element_features:
            elem_feats = get_element_features(atomic_number, normalize=True)
            feats.extend(elem_feats.numpy().tolist())

        # Add CALPHAD features (3 features: melting_T, Cp, H_form)
        calphad_feats = calphad_extractor.get_element_features(elem_symbol, normalize=True)
        feats.extend(calphad_feats.tolist())

        enhanced_node_features.append(feats)

    # Replace node features with enhanced version
    data.x = torch.tensor(enhanced_node_features, dtype=torch.float)
    # Shape: [num_nodes, 1 + 9 + 3] = [num_nodes, 13]
    #        atomic_number + element_features + calphad_features

    # Add CALPHAD-based edge features (mixing energies)
    if data.num_edges > 0:
        calphad_edge_features = []

        for i, j in data.edge_index.t():
            elem_i = str(structure[i].specie.symbol)
            elem_j = str(structure[j].specie.symbol)

            # Calculate mixing energy between these elements
            mixing_E = calphad_extractor.get_mixing_energy(elem_i, elem_j)

            # Normalize mixing energy to reasonable range (-50 to +50 kJ/mol)
            mixing_E_norm = mixing_E / 50000.0  # Convert J to kJ and normalize

            calphad_edge_features.append([mixing_E_norm])

        # Concatenate with existing edge features (distances)
        calphad_edge_feats = torch.tensor(calphad_edge_features, dtype=torch.float)
        data.edge_attr = torch.cat([data.edge_attr, calphad_edge_feats], dim=-1)
        # Shape: [num_edges, 2] = [distance, mixing_energy]

    # Store metadata about CALPHAD enhancement
    data.has_calphad_features = True
    data.calphad_node_dim = 13
    data.calphad_edge_dim = 2

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
        print("WARNING: Graph has no edges!")

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

    # Test 4: CALPHAD-enhanced graph
    print("Test 4: CALPHAD-enhanced graph construction")
    lattice = Lattice.cubic(2.87)
    structure = Structure(lattice, ["Fe", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    graph_calphad = structure_to_graph_with_calphad(structure, cutoff=4.0)
    print(f"Formula: {graph_calphad.formula}")
    print(f"Node features shape: {graph_calphad.x.shape}")
    print(f"  (Expected: [2, 13] = [num_atoms, atomic_num + 9_elem_feats + 3_calphad_feats])")
    print(f"Edge features shape: {graph_calphad.edge_attr.shape}")
    print(f"  (Expected: [num_edges, 2] = [distance + mixing_energy])")
    print(f"Has CALPHAD features: {graph_calphad.has_calphad_features}")
    print()

    # Show sample features
    print("Sample node features for Fe atom:")
    print(f"  Atomic number: {graph_calphad.x[0, 0]:.0f}")
    print(f"  Element features: {graph_calphad.x[0, 1:10]}")
    print(f"  CALPHAD features: {graph_calphad.x[0, 10:13]}")
    print()

    print("Sample edge features:")
    if graph_calphad.num_edges > 0:
        print(f"  Distance: {graph_calphad.edge_attr[0, 0]:.3f} Å")
        print(f"  Mixing energy (normalized): {graph_calphad.edge_attr[0, 1]:.6f}")
    print()

    print("Crystal graph construction tests passed!")
