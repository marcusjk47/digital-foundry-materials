"""
Predict Formation Energy for New Materials

This script demonstrates how to predict properties for materials
NOT in your training dataset using a trained GNN model.

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import numpy as np
from pathlib import Path
from pymatgen.core import Structure, Lattice
from mp_api.client import MPRester
import os

from crystal_graph import structure_to_graph, structure_to_graph_with_calphad
from gnn_model import CGCNN
from gnn_model_calphad import CGCNN_CALPHAD_Regressor


def predict_from_structure(structure, model_path, use_calphad=False):
    """
    Predict formation energy for a given crystal structure.

    Args:
        structure: Pymatgen Structure object
        model_path: Path to trained model checkpoint (.pt file)
        use_calphad: Whether model was trained with CALPHAD features

    Returns:
        Predicted formation energy (eV/atom)
    """
    # Convert structure to graph
    if use_calphad:
        graph = structure_to_graph_with_calphad(
            structure,
            cutoff=8.0,
            max_neighbors=12
        )
    else:
        graph = structure_to_graph(
            structure,
            cutoff=8.0,
            max_neighbors=12
        )

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')

    # Detect model architecture from checkpoint
    if use_calphad:
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
        model = CGCNN(
            node_feature_dim=64,
            edge_feature_dim=1,
            hidden_dim=128,
            n_conv=3,
            n_hidden=2,
            output_dim=1
        )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Make prediction
    with torch.no_grad():
        prediction = model(graph)
        if isinstance(prediction, dict):
            # CALPHAD model returns dict
            formation_energy = prediction['formation_energy'].item()
        else:
            formation_energy = prediction.item()

    return formation_energy


def create_structure_from_composition(composition_dict, lattice_type='fcc'):
    """
    Create a crystal structure from composition.

    Args:
        composition_dict: Dict like {"Fe": 0.5, "Ni": 0.5}
        lattice_type: 'fcc', 'bcc', or 'hcp'

    Returns:
        Pymatgen Structure object
    """
    # Lattice parameters (approximate for Fe-Ni alloys)
    a = 3.6  # Angstroms

    if lattice_type == 'fcc':
        # FCC lattice
        lattice = Lattice.cubic(a)
        # FCC has atoms at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        positions = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]
    elif lattice_type == 'bcc':
        # BCC lattice
        lattice = Lattice.cubic(a)
        # BCC has atoms at (0,0,0) and (0.5,0.5,0.5)
        positions = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ]
    else:
        raise ValueError(f"Lattice type '{lattice_type}' not supported")

    # Distribute elements according to composition
    elements = []
    ratios = list(composition_dict.values())
    element_names = list(composition_dict.keys())

    for i, pos in enumerate(positions):
        # Simple distribution - alternate elements
        element_idx = 0 if np.random.random() < ratios[0] else 1
        elements.append(element_names[element_idx])

    structure = Structure(lattice, elements, positions)

    return structure


def fetch_structure_from_mp(material_id, api_key):
    """
    Fetch structure from Materials Project.

    Args:
        material_id: Materials Project ID (e.g., 'mp-149')
        api_key: Your MP API key

    Returns:
        Pymatgen Structure object
    """
    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(material_id)

    return structure


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GNN Prediction for New Materials")
    print("="*70)
    print()

    # Check for trained model
    model_path = Path("checkpoints/best_model.pt")

    if not model_path.exists():
        print("❌ No trained model found!")
        print("   Please train a model first using the GNN Property Predictor page.")
        print(f"   Expected model at: {model_path}")
        exit(1)

    print(f"✅ Found trained model: {model_path}")
    print()

    # Choose prediction method
    print("Choose how to create the structure:")
    print("  1. Create from composition (Fe-Ni alloy)")
    print("  2. Fetch from Materials Project by ID")
    print("  3. Load from file (CIF/POSCAR)")

    choice = input("\nEnter choice (1/2/3): ").strip()

    # Determine if model uses CALPHAD features
    # You can check this by loading a sample from your dataset
    use_calphad = False  # Set to True if you trained with CALPHAD features

    if choice == "1":
        # Method 1: Create from composition
        print("\n" + "="*70)
        print("Method 1: Create Structure from Composition")
        print("="*70)

        # Get composition
        fe_percent = float(input("Enter Fe percentage (0-100): "))
        ni_percent = 100 - fe_percent

        composition = {
            "Fe": fe_percent / 100,
            "Ni": ni_percent / 100
        }

        lattice_type = input("Enter lattice type (fcc/bcc) [default: fcc]: ").strip().lower() or 'fcc'

        print(f"\nCreating {lattice_type.upper()} structure: Fe{fe_percent}Ni{ni_percent}")
        structure = create_structure_from_composition(composition, lattice_type)

        print(f"✅ Created structure:")
        print(f"   Formula: {structure.composition.reduced_formula}")
        print(f"   Lattice: {lattice_type.upper()}")
        print(f"   Number of atoms: {len(structure)}")
        print()

        # Make prediction
        print("Making prediction...")
        prediction = predict_from_structure(structure, model_path, use_calphad)

        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"Composition: Fe{fe_percent}Ni{ni_percent}")
        print(f"Lattice: {lattice_type.upper()}")
        print(f"Predicted Formation Energy: {prediction:.4f} eV/atom")
        print("="*70)

    elif choice == "2":
        # Method 2: Fetch from Materials Project
        print("\n" + "="*70)
        print("Method 2: Fetch from Materials Project")
        print("="*70)

        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            print("❌ MP_API_KEY not found in environment!")
            print("   Set it with: export MP_API_KEY='your-key-here'")
            exit(1)

        material_id = input("Enter Materials Project ID (e.g., mp-149): ").strip()

        print(f"\nFetching structure for {material_id}...")
        try:
            structure = fetch_structure_from_mp(material_id, api_key)

            print(f"✅ Fetched structure:")
            print(f"   Formula: {structure.composition.reduced_formula}")
            print(f"   Space group: {structure.get_space_group_info()[0]}")
            print(f"   Number of atoms: {len(structure)}")
            print()

            # Make prediction
            print("Making prediction...")
            prediction = predict_from_structure(structure, model_path, use_calphad)

            print("\n" + "="*70)
            print("PREDICTION RESULTS")
            print("="*70)
            print(f"Material ID: {material_id}")
            print(f"Formula: {structure.composition.reduced_formula}")
            print(f"Predicted Formation Energy: {prediction:.4f} eV/atom")
            print("="*70)

        except Exception as e:
            print(f"❌ Error fetching structure: {e}")
            exit(1)

    elif choice == "3":
        # Method 3: Load from file
        print("\n" + "="*70)
        print("Method 3: Load from File")
        print("="*70)

        file_path = input("Enter file path (CIF or POSCAR): ").strip()

        try:
            structure = Structure.from_file(file_path)

            print(f"✅ Loaded structure:")
            print(f"   Formula: {structure.composition.reduced_formula}")
            print(f"   Number of atoms: {len(structure)}")
            print()

            # Make prediction
            print("Making prediction...")
            prediction = predict_from_structure(structure, model_path, use_calphad)

            print("\n" + "="*70)
            print("PREDICTION RESULTS")
            print("="*70)
            print(f"File: {file_path}")
            print(f"Formula: {structure.composition.reduced_formula}")
            print(f"Predicted Formation Energy: {prediction:.4f} eV/atom")
            print("="*70)

        except Exception as e:
            print(f"❌ Error loading file: {e}")
            exit(1)

    else:
        print("Invalid choice!")
        exit(1)

    print("\n💡 Note: This is a prediction based on your trained model.")
    print("   Accuracy depends on your training data quality and size.")
    print("   For validation, compare with DFT calculations or experiments.")
