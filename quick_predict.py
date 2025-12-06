"""
Quick Prediction Script for Fe-Ni Compositions

Simple script to predict formation energies for different Fe-Ni ratios.

Usage:
    python quick_predict.py

Author: Digital Foundry Materials Science Toolkit
"""

import torch
from pymatgen.core import Structure, Lattice
from crystal_graph import structure_to_graph, structure_to_graph_with_calphad
from gnn_model import CGCNN
from gnn_model_calphad import CGCNN_CALPHAD_Regressor
import numpy as np


def create_feni_structure(fe_fraction, lattice_type='fcc', supercell_size=2):
    """
    Create Fe-Ni alloy structure with given composition.

    Args:
        fe_fraction: Fraction of Fe (0.0 to 1.0)
        lattice_type: 'fcc' or 'bcc'
        supercell_size: Size of supercell (larger = more realistic)

    Returns:
        Pymatgen Structure
    """
    a = 3.6  # Lattice parameter in Angstroms

    if lattice_type == 'fcc':
        # FCC unit cell
        lattice = Lattice.cubic(a)
        base_positions = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]
    else:  # bcc
        lattice = Lattice.cubic(a)
        base_positions = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ]

    # Create supercell
    positions = []
    for i in range(supercell_size):
        for j in range(supercell_size):
            for k in range(supercell_size):
                for pos in base_positions:
                    new_pos = [
                        (pos[0] + i) / supercell_size,
                        (pos[1] + j) / supercell_size,
                        (pos[2] + k) / supercell_size
                    ]
                    positions.append(new_pos)

    # Assign elements randomly based on composition
    num_atoms = len(positions)
    num_fe = int(num_atoms * fe_fraction)
    elements = ['Fe'] * num_fe + ['Ni'] * (num_atoms - num_fe)
    np.random.shuffle(elements)

    # Scale lattice for supercell
    supercell_lattice = Lattice(lattice.matrix * supercell_size)

    structure = Structure(supercell_lattice, elements, positions)

    return structure


def predict_formation_energy(structure, model_path='checkpoints/best_model.pt', use_calphad=False):
    """
    Predict formation energy for a structure.

    Args:
        structure: Pymatgen Structure
        model_path: Path to trained model
        use_calphad: Whether model uses CALPHAD features

    Returns:
        Formation energy in eV/atom
    """
    # Convert to graph
    if use_calphad:
        graph = structure_to_graph_with_calphad(structure, cutoff=8.0, max_neighbors=12)
        model = CGCNN_CALPHAD_Regressor(
            input_node_dim=13,
            input_edge_dim=2,
            node_feature_dim=64,
            edge_feature_dim=32,
            hidden_dim=128,
            n_conv=3,
            n_hidden=2
        )
    else:
        graph = structure_to_graph(structure, cutoff=8.0, max_neighbors=12)
        model = CGCNN(
            node_feature_dim=64,
            edge_feature_dim=1,
            hidden_dim=128,
            n_conv=3,
            n_hidden=2,
            output_dim=1
        )

    # Load trained model
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Predict
    with torch.no_grad():
        prediction = model(graph)
        if isinstance(prediction, dict):
            energy = prediction['formation_energy'].item()
        else:
            energy = prediction.item()

    return energy


if __name__ == "__main__":
    print("="*70)
    print("Quick Formation Energy Predictions for Fe-Ni Alloys")
    print("="*70)
    print()

    # Check for model
    import os
    if not os.path.exists('checkpoints/best_model.pt'):
        print("❌ No trained model found at checkpoints/best_model.pt")
        print("   Please train a model first!")
        exit(1)

    print("✅ Model loaded")
    print()

    # Set to True if you trained with CALPHAD features
    USE_CALPHAD = False

    # Predict for various compositions
    compositions = [
        0.0,   # Pure Ni
        0.25,  # Fe25Ni75
        0.50,  # Fe50Ni50
        0.75,  # Fe75Ni25
        1.0    # Pure Fe
    ]

    print("Predicting formation energies...")
    print()
    print("-"*70)
    print(f"{'Composition':<20} {'Structure':<10} {'Formation Energy (eV/atom)':<30}")
    print("-"*70)

    results = []

    for fe_frac in compositions:
        ni_frac = 1.0 - fe_frac

        # Try both FCC and BCC
        for lattice in ['fcc', 'bcc']:
            try:
                # Create structure
                structure = create_feni_structure(fe_frac, lattice, supercell_size=2)

                # Predict
                energy = predict_formation_energy(structure, use_calphad=USE_CALPHAD)

                # Format composition
                if fe_frac == 0.0:
                    comp_str = "Pure Ni"
                elif fe_frac == 1.0:
                    comp_str = "Pure Fe"
                else:
                    comp_str = f"Fe{fe_frac*100:.0f}Ni{ni_frac*100:.0f}"

                print(f"{comp_str:<20} {lattice.upper():<10} {energy:>+.4f}")

                results.append({
                    'composition': comp_str,
                    'lattice': lattice,
                    'fe_fraction': fe_frac,
                    'energy': energy
                })

            except Exception as e:
                print(f"{comp_str:<20} {lattice.upper():<10} ERROR: {e}")

    print("-"*70)
    print()

    # Find most stable
    if results:
        most_stable = min(results, key=lambda x: x['energy'])
        print("🏆 Most stable configuration:")
        print(f"   {most_stable['composition']} ({most_stable['lattice'].upper()})")
        print(f"   Formation Energy: {most_stable['energy']:+.4f} eV/atom")
        print()

    print("💡 Tips:")
    print("   - More negative = more stable")
    print("   - Positive values = unstable (decomposition likely)")
    print("   - Compare different compositions to find optimal alloys")
    print("   - Results depend on training data quality!")
    print()
    print("="*70)
