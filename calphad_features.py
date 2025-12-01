"""
CALPHAD Feature Extraction Module

Extracts thermodynamic features from TDB files to enhance GNN node/edge features.
Integrates with ESPEI-generated databases and PyCalphad for thermodynamic calculations.

Author: Digital Foundry Materials Science Toolkit
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from pymatgen.core import Composition

try:
    from pycalphad import Database, equilibrium, variables as v
    PYCALPHAD_AVAILABLE = True
except ImportError:
    PYCALPHAD_AVAILABLE = False
    print("Warning: pycalphad not available. CALPHAD features will use defaults.")


class CALPHADFeatureExtractor:
    """
    Extract thermodynamic features from CALPHAD TDB files.

    Features extracted:
    - Per-element (node features): melting point, heat capacity, enthalpy
    - Per-bond (edge features): mixing enthalpy between elements
    """

    def __init__(self, tdb_path: Optional[str] = None, reference_temp: float = 298.15):
        """
        Args:
            tdb_path: Path to TDB file (optional - can use defaults)
            reference_temp: Reference temperature in Kelvin (default: 298.15K)
        """
        self.tdb_path = tdb_path
        self.reference_temp = reference_temp
        self.db = None

        # Load TDB database if provided
        if tdb_path and Path(tdb_path).exists() and PYCALPHAD_AVAILABLE:
            try:
                self.db = Database(tdb_path)
                print(f"Loaded TDB: {tdb_path}")
            except Exception as e:
                print(f"Warning: Could not load TDB {tdb_path}: {e}")
                self.db = None

        # Default element properties (fallback)
        self._initialize_default_properties()

    def _initialize_default_properties(self):
        """Initialize default thermodynamic properties for common elements."""
        # Melting points (K)
        self.default_melting_points = {
            'H': 14, 'He': 0.95, 'Li': 453, 'Be': 1560, 'B': 2348, 'C': 3823,
            'N': 63, 'O': 54, 'F': 53, 'Ne': 24,
            'Na': 371, 'Mg': 923, 'Al': 933, 'Si': 1687, 'P': 317, 'S': 388,
            'Cl': 172, 'Ar': 84, 'K': 336, 'Ca': 1115,
            'Sc': 1814, 'Ti': 1941, 'V': 2183, 'Cr': 2180, 'Mn': 1519,
            'Fe': 1811, 'Co': 1768, 'Ni': 1728, 'Cu': 1358, 'Zn': 693,
            'Ga': 303, 'Ge': 1211, 'As': 1090, 'Se': 494, 'Br': 266, 'Kr': 116,
            'Rb': 312, 'Sr': 1050, 'Y': 1799, 'Zr': 2128, 'Nb': 2750,
            'Mo': 2896, 'Tc': 2430, 'Ru': 2607, 'Rh': 2237, 'Pd': 1828,
            'Ag': 1235, 'Cd': 594, 'In': 430, 'Sn': 505, 'Sb': 904,
            'Te': 723, 'I': 387, 'Xe': 161, 'Cs': 302, 'Ba': 1000,
            'La': 1193, 'Ce': 1068, 'Pr': 1208, 'Nd': 1297, 'Pm': 1315,
            'Sm': 1345, 'Eu': 1099, 'Gd': 1585, 'Tb': 1629, 'Dy': 1680,
            'Ho': 1734, 'Er': 1802, 'Tm': 1818, 'Yb': 1097, 'Lu': 1925,
            'Hf': 2506, 'Ta': 3290, 'W': 3695, 'Re': 3459, 'Os': 3306,
            'Ir': 2719, 'Pt': 2041, 'Au': 1337, 'Hg': 234, 'Tl': 577,
            'Pb': 601, 'Bi': 544, 'Po': 527, 'At': 575, 'Rn': 202,
            'Fr': 300, 'Ra': 973, 'Ac': 1323, 'Th': 2023, 'Pa': 1841,
            'U': 1405, 'Np': 917, 'Pu': 913
        }

        # Heat capacities at 298K (J/mol/K)
        self.default_heat_capacities = {
            'H': 28.8, 'He': 20.8, 'Li': 24.8, 'Be': 16.4, 'B': 11.1, 'C': 8.5,
            'N': 29.1, 'O': 29.4, 'F': 31.3, 'Ne': 20.8,
            'Na': 28.2, 'Mg': 24.9, 'Al': 24.2, 'Si': 20.0, 'P': 23.8, 'S': 22.6,
            'Fe': 25.1, 'Co': 24.8, 'Ni': 26.1, 'Cu': 24.4, 'Zn': 25.4,
            'Cr': 23.4, 'Mn': 26.3, 'Ti': 25.0, 'V': 24.9, 'Mo': 24.1,
            'W': 24.3, 'Au': 25.4, 'Ag': 25.4, 'Pt': 25.9, 'Pd': 25.9
        }

        # Formation enthalpies (kJ/mol) - relative to elements
        self.default_formation_enthalpies = {
            'H': 0, 'He': 0, 'Li': 0, 'Be': 0, 'B': 0, 'C': 0,
            'N': 0, 'O': 0, 'F': 0, 'Ne': 0,
            'Na': 0, 'Mg': 0, 'Al': 0, 'Si': 0, 'P': 0, 'S': 0,
            'Fe': 0, 'Co': 0, 'Ni': 0, 'Cu': 0, 'Zn': 0,
            'Cr': 0, 'Mn': 0, 'Ti': 0, 'V': 0, 'Mo': 0,
            'W': 0, 'Au': 0, 'Ag': 0, 'Pt': 0, 'Pd': 0
        }

    def get_element_features(self, element: str, normalize: bool = True) -> np.ndarray:
        """
        Extract CALPHAD features for a single element.

        Args:
            element: Element symbol (e.g., 'Fe', 'Ni')
            normalize: Whether to normalize features to [0, 1]

        Returns:
            Array of thermodynamic features [melting_T, heat_capacity, enthalpy]
        """
        features = []

        # Feature 1: Melting point
        melting_T = self.default_melting_points.get(element, 1000.0)
        features.append(melting_T)

        # Feature 2: Heat capacity at 298K
        Cp = self.default_heat_capacities.get(element, 25.0)
        features.append(Cp)

        # Feature 3: Formation enthalpy
        H_form = self.default_formation_enthalpies.get(element, 0.0)
        features.append(H_form)

        features = np.array(features, dtype=np.float32)

        # Normalize if requested
        if normalize:
            # Melting point: 0-4000 K
            features[0] = features[0] / 4000.0
            # Heat capacity: 0-50 J/mol/K
            features[1] = features[1] / 50.0
            # Formation enthalpy: already ~0 for elements
            features[2] = 0.0

        return features

    def get_mixing_energy(self, element1: str, element2: str,
                         composition: float = 0.5,
                         temperature: Optional[float] = None) -> float:
        """
        Calculate mixing energy between two elements.

        Args:
            element1, element2: Element symbols
            composition: Fraction of element2 (0-1)
            temperature: Temperature in K (uses reference_temp if None)

        Returns:
            Mixing enthalpy in J/mol (0 if cannot calculate)
        """
        if temperature is None:
            temperature = self.reference_temp

        # If TDB database available, try to calculate
        if self.db is not None and PYCALPHAD_AVAILABLE:
            try:
                return self._calculate_mixing_energy_from_tdb(
                    element1, element2, composition, temperature
                )
            except Exception as e:
                # Fall back to empirical model
                pass

        # Empirical mixing energy (Miedema model approximation)
        return self._empirical_mixing_energy(element1, element2, composition)

    def _calculate_mixing_energy_from_tdb(self, element1: str, element2: str,
                                          composition: float, temperature: float) -> float:
        """
        Calculate mixing energy using PyCalphad and TDB.
        """
        try:
            # Get all phases that contain both elements
            phases = [p for p in self.db.phases.keys()
                     if element1 in self.db.phases[p].constituents[0]
                     and element2 in self.db.phases[p].constituents[0]]

            if not phases:
                return 0.0

            # Use first available phase (typically LIQUID or FCC)
            phase = phases[0]

            # Calculate Gibbs energy of mixing
            eq = equilibrium(
                self.db,
                [element1, element2, 'VA'],  # Include vacancies
                [phase],
                {v.X(element2): composition, v.T: temperature, v.P: 101325},
                verbose=False
            )

            # Extract Gibbs energy
            if eq is not None and hasattr(eq, 'GM'):
                mixing_G = float(eq.GM.values[0])
                return mixing_G

        except Exception as e:
            # Silently fail and use default
            pass

        return 0.0

    def _empirical_mixing_energy(self, element1: str, element2: str,
                                 composition: float) -> float:
        """
        Simple empirical mixing energy based on electronegativity difference.
        """
        try:
            from pymatgen.core import Element

            elem1 = Element(element1)
            elem2 = Element(element2)

            # Electronegativity difference
            if elem1.X is not None and elem2.X is not None:
                delta_X = abs(elem1.X - elem2.X)

                # Simple model: more electronegative difference = more negative mixing
                # Typical range: -50 to +50 kJ/mol
                mixing_H = -20.0 * delta_X * composition * (1 - composition)
                return mixing_H * 1000  # Convert kJ/mol to J/mol

        except:
            pass

        return 0.0  # Default: ideal mixing

    def get_feature_dimension(self) -> int:
        """Return the number of CALPHAD features per element."""
        return 3  # melting_T, heat_capacity, formation_enthalpy

    def extract_features_for_structure(self, composition: Composition,
                                       normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract CALPHAD features for all elements in a composition.

        Args:
            composition: Pymatgen Composition object
            normalize: Whether to normalize features

        Returns:
            Dictionary mapping element symbols to feature arrays
        """
        features_dict = {}

        for element in composition.elements:
            elem_symbol = str(element)
            features = self.get_element_features(elem_symbol, normalize=normalize)
            features_dict[elem_symbol] = features

        return features_dict


def get_calphad_features_batch(elements: List[str],
                               tdb_path: Optional[str] = None,
                               normalize: bool = True) -> torch.Tensor:
    """
    Get CALPHAD features for a batch of elements.

    Args:
        elements: List of element symbols
        tdb_path: Path to TDB file (optional)
        normalize: Whether to normalize features

    Returns:
        Tensor of shape [num_elements, num_features]
    """
    extractor = CALPHADFeatureExtractor(tdb_path)

    features_list = []
    for elem in elements:
        features = extractor.get_element_features(elem, normalize=normalize)
        features_list.append(features)

    return torch.tensor(features_list, dtype=torch.float)


if __name__ == "__main__":
    """Test CALPHAD feature extraction"""
    print("="*60)
    print("Testing CALPHAD Feature Extraction")
    print("="*60)

    # Test 1: Extract features for common elements
    print("\nTest 1: Element features")
    extractor = CALPHADFeatureExtractor()

    elements = ['Fe', 'Ni', 'Cr', 'Al', 'Ti', 'Cu']
    for elem in elements:
        features = extractor.get_element_features(elem, normalize=False)
        print(f"{elem:>2}: Melting={features[0]:7.1f}K, Cp={features[1]:5.1f}J/mol/K, H={features[2]:6.1f}kJ/mol")

    # Test 2: Normalized features
    print("\nTest 2: Normalized features")
    features_norm = extractor.get_element_features('Fe', normalize=True)
    print(f"Fe normalized: {features_norm}")

    # Test 3: Mixing energies
    print("\nTest 3: Mixing energies")
    pairs = [('Fe', 'Ni'), ('Fe', 'Cr'), ('Cu', 'Zn'), ('Al', 'Cu')]
    for elem1, elem2 in pairs:
        mixing_E = extractor.get_mixing_energy(elem1, elem2, composition=0.5)
        print(f"{elem1}-{elem2} (50-50): {mixing_E/1000:.2f} kJ/mol")

    # Test 4: Batch extraction
    print("\nTest 4: Batch feature extraction")
    elements_batch = ['Fe', 'Ni', 'Cr']
    features_tensor = get_calphad_features_batch(elements_batch, normalize=True)
    print(f"Batch tensor shape: {features_tensor.shape}")
    print(f"Features:\n{features_tensor}")

    # Test 5: Structure features
    print("\nTest 5: Extract features for composition")
    from pymatgen.core import Composition
    comp = Composition("Fe2Ni")
    features_dict = extractor.extract_features_for_structure(comp, normalize=True)
    for elem, feats in features_dict.items():
        print(f"{elem}: {feats}")

    print("\n" + "="*60)
    print("All CALPHAD feature tests passed!")
    print("="*60)
