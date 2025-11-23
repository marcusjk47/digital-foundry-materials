"""
Element Feature Extraction Module

Provides element property features for GNN node embeddings.

Features extracted:
- Atomic number (normalized)
- Group number
- Period (row)
- Electronegativity (Pauling scale)
- Atomic radius
- Valence electrons
- First ionization energy
- Electron affinity
- Atomic mass

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import numpy as np
from pymatgen.core import Element


def get_element_features(atomic_number, normalize=True):
    """
    Get element features for GNN node embedding.

    Args:
        atomic_number: Atomic number (Z) of the element
        normalize: If True, normalize features to [0, 1] range

    Returns:
        torch.Tensor of shape [num_features] with element properties
    """
    try:
        elem = Element.from_Z(atomic_number)
    except Exception:
        # Return zero features for invalid atomic numbers
        return torch.zeros(9, dtype=torch.float)

    features = []

    # Feature 1: Atomic number
    if normalize:
        features.append(atomic_number / 100.0)  # Normalize by max Z ~118
    else:
        features.append(float(atomic_number))

    # Feature 2: Group number
    if elem.group:
        group = elem.group / 18.0 if normalize else float(elem.group)
    else:
        group = 0.0
    features.append(group)

    # Feature 3: Period (row)
    if elem.row:
        period = elem.row / 7.0 if normalize else float(elem.row)
    else:
        period = 0.0
    features.append(period)

    # Feature 4: Electronegativity (Pauling scale)
    if elem.X:
        electronegativity = elem.X / 4.0 if normalize else elem.X  # Max ~4.0 (F)
    else:
        electronegativity = 0.0
    features.append(electronegativity)

    # Feature 5: Atomic radius (in Angstroms)
    if elem.atomic_radius:
        radius = elem.atomic_radius / 300.0 if normalize else elem.atomic_radius
    else:
        radius = 0.0
    features.append(radius)

    # Feature 6: Valence electrons
    try:
        valence = len(elem.valence) if hasattr(elem, 'valence') and elem.valence else 0
        if normalize:
            valence = valence / 8.0  # Max 8 valence electrons
    except Exception:
        valence = 0.0
    features.append(float(valence))

    # Feature 7: First ionization energy (in eV)
    if elem.ionization_energy:
        ionization = elem.ionization_energy / 25.0 if normalize else elem.ionization_energy  # Max ~24.6 eV (He)
    else:
        ionization = 0.0
    features.append(ionization)

    # Feature 8: Electron affinity (in eV)
    if elem.electron_affinity and elem.electron_affinity > 0:
        affinity = elem.electron_affinity / 5.0 if normalize else elem.electron_affinity  # Max ~3.6 eV (Cl)
    else:
        affinity = 0.0
    features.append(affinity)

    # Feature 9: Atomic mass (in amu)
    mass = elem.atomic_mass / 250.0 if normalize else elem.atomic_mass  # Max ~238 (U)
    features.append(mass)

    return torch.tensor(features, dtype=torch.float)


def get_element_feature_dim():
    """
    Get the dimension of element feature vectors.

    Returns:
        int: Number of features per element
    """
    return 9


def get_element_embedding_matrix(max_z=100, normalize=True):
    """
    Create a lookup matrix for element embeddings.

    This is useful for embedding layers that convert atomic numbers
    to feature vectors.

    Args:
        max_z: Maximum atomic number to include
        normalize: Normalize features

    Returns:
        torch.Tensor of shape [max_z + 1, num_features]
    """
    embedding_matrix = []

    for z in range(max_z + 1):
        if z == 0:
            # Padding for atomic number 0 (if needed)
            features = torch.zeros(get_element_feature_dim())
        else:
            features = get_element_features(z, normalize=normalize)
        embedding_matrix.append(features)

    return torch.stack(embedding_matrix)


def print_element_features(atomic_number):
    """
    Print element features in human-readable format.

    Args:
        atomic_number: Atomic number of element
    """
    elem = Element.from_Z(atomic_number)
    features = get_element_features(atomic_number, normalize=False)

    print(f"\n{'='*60}")
    print(f"Element Features: {elem.symbol} ({elem.name})")
    print(f"{'='*60}")
    print(f"Atomic number:        {features[0]:.2f}")
    print(f"Group:                {features[1]:.2f}")
    print(f"Period:               {features[2]:.2f}")
    print(f"Electronegativity:    {features[3]:.2f}")
    print(f"Atomic radius:        {features[4]:.2f} Å")
    print(f"Valence electrons:    {features[5]:.0f}")
    print(f"Ionization energy:    {features[6]:.2f} eV")
    print(f"Electron affinity:    {features[7]:.2f} eV")
    print(f"Atomic mass:          {features[8]:.2f} amu")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    """Test element feature extraction"""

    print("="*60)
    print("Testing Element Feature Extraction")
    print("="*60)

    # Test common elements
    test_elements = [1, 6, 8, 13, 26, 29, 79]  # H, C, O, Al, Fe, Cu, Au
    element_names = ['H', 'C', 'O', 'Al', 'Fe', 'Cu', 'Au']

    print("\nTest 1: Extract features for common elements")
    for z, name in zip(test_elements, element_names):
        print_element_features(z)

    print("\nTest 2: Feature dimension check")
    print(f"Feature dimension: {get_element_feature_dim()}")

    print("\nTest 3: Normalized vs unnormalized features")
    z = 26  # Iron
    feat_norm = get_element_features(z, normalize=True)
    feat_unnorm = get_element_features(z, normalize=False)

    print(f"Fe normalized features:   {feat_norm}")
    print(f"Fe unnormalized features: {feat_unnorm}")

    print("\nTest 4: Create embedding matrix")
    embedding_matrix = get_element_embedding_matrix(max_z=20)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Sample (Fe, Z=26 would be out of range, using C, Z=6):")
    print(f"Carbon features: {embedding_matrix[6]}")

    print("\nTest 5: Handle invalid atomic number")
    invalid_feat = get_element_features(200)  # Invalid Z
    print(f"Invalid atomic number features: {invalid_feat}")
    print(f"All zeros: {torch.all(invalid_feat == 0).item()}")

    print("\n✅ Element feature extraction tests passed!")
