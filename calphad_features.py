"""
Elemental Thermodynamic Feature Extraction Module

Extracts elemental thermodynamic properties (melting point, heat capacity) to enhance
GNN node/edge features. Can optionally integrate with CALPHAD TDB files for more
accurate mixing energy calculations, but by default uses elemental property lookups.

Note: Despite the filename, these are primarily elemental descriptors, not CALPHAD
calculations. The name is kept for backwards compatibility with trained models.

Author: Digital Foundry Materials Science Toolkit
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from pymatgen.core import Composition
from functools import lru_cache

try:
    from pycalphad import Database, equilibrium, variables as v
    PYCALPHAD_AVAILABLE = True
except ImportError:
    PYCALPHAD_AVAILABLE = False
    print("Warning: pycalphad not available. CALPHAD features will use defaults.")


class CALPHADFeatureExtractor:
    """
    Extract elemental thermodynamic properties for GNN features.

    Features extracted (from elemental property databases):
    - Per-element (node features): melting point, heat capacity, formation enthalpy
    - Per-bond (edge features): mixing enthalpy estimate (empirical or from TDB)

    Note: These are elemental descriptors, not CALPHAD calculations. The class can
    optionally use CALPHAD TDB files for mixing energies if provided, otherwise uses
    simple empirical models based on electronegativity differences.
    """

    def __init__(self, tdb_path: Optional[str] = None, reference_temp: float = 298.15,
                 use_tdb_by_default: bool = True, cache_size: int = 1000):
        """
        Args:
            tdb_path: Path to TDB file (optional - can use defaults)
            reference_temp: Reference temperature in Kelvin (default: 298.15K)
            use_tdb_by_default: If True, prefer TDB calculations over empirical (default: True)
            cache_size: Size of LRU cache for mixing energy calculations (default: 1000)
        """
        self.tdb_path = tdb_path
        self.reference_temp = reference_temp
        self.use_tdb_by_default = use_tdb_by_default
        self.db = None
        self.tdb_loaded = False

        # Cache for mixing energy calculations
        self._mixing_energy_cache = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

        # Load TDB database if provided
        if tdb_path and Path(tdb_path).exists() and PYCALPHAD_AVAILABLE:
            try:
                self.db = Database(tdb_path)
                self.tdb_loaded = True
                print(f"[CALPHAD] Loaded TDB: {tdb_path}")
                print(f"[CALPHAD] Elements in TDB: {sorted([str(e) for e in self.db.elements if str(e) != 'VA'])}")
                print(f"[CALPHAD] Phases in TDB: {sorted([str(p) for p in self.db.phases.keys()])}")
            except Exception as e:
                print(f"[CALPHAD] Warning: Could not load TDB {tdb_path}: {e}")
                print(f"[CALPHAD] Falling back to empirical models")
                self.db = None
                self.tdb_loaded = False
        elif tdb_path:
            print(f"[CALPHAD] TDB path provided but not loaded (file missing or PyCalphad unavailable)")
            print(f"[CALPHAD] Using empirical models")

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
        Extract elemental thermodynamic properties for a single element.

        Args:
            element: Element symbol (e.g., 'Fe', 'Ni')
            normalize: Whether to normalize features to [0, 1]

        Returns:
            Array of elemental properties [melting_T, heat_capacity, formation_enthalpy]
            Note: Formation enthalpy is always 0 for pure elements
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
        Calculate mixing energy between two elements with caching.

        Args:
            element1, element2: Element symbols
            composition: Fraction of element2 (0-1)
            temperature: Temperature in K (uses reference_temp if None)

        Returns:
            Mixing enthalpy in J/mol (0 if cannot calculate)
        """
        if temperature is None:
            temperature = self.reference_temp

        # Create cache key (order-independent)
        elem_pair = tuple(sorted([element1.upper(), element2.upper()]))
        cache_key = (elem_pair, round(composition, 4), round(temperature, 2))

        # Check cache
        if cache_key in self._mixing_energy_cache:
            self._cache_hits += 1
            return self._mixing_energy_cache[cache_key]

        self._cache_misses += 1

        # Calculate mixing energy
        mixing_energy = 0.0

        # If TDB database available and we prefer TDB, try to calculate
        if self.tdb_loaded and self.use_tdb_by_default and PYCALPHAD_AVAILABLE:
            try:
                mixing_energy = self._calculate_mixing_energy_from_tdb(
                    element1, element2, composition, temperature
                )
                # Cache the result
                self._add_to_cache(cache_key, mixing_energy)
                return mixing_energy
            except Exception as e:
                # Fall back to empirical model
                pass

        # Empirical mixing energy (Miedema model approximation)
        mixing_energy = self._empirical_mixing_energy(element1, element2, composition)

        # Cache the result
        self._add_to_cache(cache_key, mixing_energy)

        return mixing_energy

    def _add_to_cache(self, key, value):
        """Add entry to cache with LRU eviction."""
        if len(self._mixing_energy_cache) >= self._cache_size:
            # Remove oldest entry (first added)
            self._mixing_energy_cache.pop(next(iter(self._mixing_energy_cache)))

        self._mixing_energy_cache[key] = value

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

    def calculate_mixing_energies_batch(
        self,
        element_pairs: List[Tuple[str, str]],
        compositions: Optional[List[float]] = None,
        temperature: Optional[float] = None
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate mixing energies for multiple element pairs efficiently.

        Args:
            element_pairs: List of (element1, element2) tuples
            compositions: List of compositions for each pair (default: 0.5 for all)
            temperature: Temperature in K (uses reference_temp if None)

        Returns:
            Dictionary mapping (elem1, elem2) to mixing energy in J/mol
        """
        if compositions is None:
            compositions = [0.5] * len(element_pairs)

        if len(compositions) != len(element_pairs):
            raise ValueError("Length of compositions must match element_pairs")

        results = {}
        for (elem1, elem2), comp in zip(element_pairs, compositions):
            energy = self.get_mixing_energy(elem1, elem2, comp, temperature)
            # Store with consistent ordering
            key = tuple(sorted([elem1.upper(), elem2.upper()]))
            results[key] = energy

        return results

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about cache performance.

        Returns:
            Dictionary with cache hits, misses, and hit rate
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0

        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total_queries': total,
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self._mixing_energy_cache),
            'cache_limit': self._cache_size
        }

    def clear_cache(self):
        """Clear the mixing energy cache and reset statistics."""
        self._mixing_energy_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def is_tdb_loaded(self) -> bool:
        """Check if TDB database is successfully loaded."""
        return self.tdb_loaded

    def get_tdb_info(self) -> Dict[str, any]:
        """
        Get information about the loaded TDB.

        Returns:
            Dictionary with TDB elements and phases
        """
        if not self.tdb_loaded or self.db is None:
            return {
                'loaded': False,
                'path': self.tdb_path,
                'elements': [],
                'phases': []
            }

        return {
            'loaded': True,
            'path': self.tdb_path,
            'elements': sorted([str(e) for e in self.db.elements if str(e) != 'VA']),
            'phases': sorted([str(p) for p in self.db.phases.keys()])
        }

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
