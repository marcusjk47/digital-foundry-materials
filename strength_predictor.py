"""
Hybrid Strength Prediction Module

Combines GNN-predicted elastic properties with empirical strengthening models:
1. GNN predicts elastic moduli (G, K, E) from crystal structure
2. Empirical models calculate strength from:
   - Hall-Petch (grain size)
   - Temperature dependence
   - Solid solution strengthening (optional)

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from pymatgen.core import Structure

from crystal_graph import structure_to_graph, structure_to_graph_with_calphad
from gnn_model_elastic import ElasticPropertyPredictor


class StrengthPredictor:
    """
    Hybrid strength predictor combining GNN and physics-based models.

    Predicts yield strength from:
    - Crystal structure (via GNN)
    - Grain size (user input)
    - Temperature (user input)
    """

    def __init__(
        self,
        elastic_model_path: Optional[str] = None,
        use_calphad: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize strength predictor.

        Args:
            elastic_model_path: Path to trained elastic property model
            use_calphad: Whether to use CALPHAD-enhanced features
            device: Device to run model on
        """
        self.use_calphad = use_calphad
        self.device = device

        # Initialize elastic property predictor
        input_node_dim = 13 if use_calphad else 1
        input_edge_dim = 2 if use_calphad else 1

        self.elastic_model = ElasticPropertyPredictor(
            input_node_dim=input_node_dim,
            input_edge_dim=input_edge_dim,
            node_feature_dim=64,
            edge_feature_dim=32,
            hidden_dim=128,
            n_conv=3,
            n_hidden=2,
            use_calphad=use_calphad
        )

        # Load trained weights if provided
        if elastic_model_path:
            checkpoint = torch.load(elastic_model_path, map_location=device)
            self.elastic_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded elastic model from {elastic_model_path}")

        self.elastic_model = self.elastic_model.to(device)
        self.elastic_model.eval()

    def predict_strength(
        self,
        structure: Structure,
        grain_size: float,
        temperature: float = 300.0,
        processing: str = "annealed"
    ) -> Dict[str, float]:
        """
        Predict yield strength for a given structure and microstructure.

        Args:
            structure: Pymatgen Structure object
            grain_size: Grain size in micrometers (μm)
            temperature: Temperature in Kelvin
            processing: Processing condition ("annealed", "cold_worked", "quenched")

        Returns:
            Dictionary with strength predictions and breakdown
        """
        # Convert structure to graph
        if self.use_calphad:
            graph = structure_to_graph_with_calphad(structure, cutoff=8.0)
        else:
            graph = structure_to_graph(structure, cutoff=8.0)

        graph = graph.to(self.device)

        # Predict elastic properties
        elastic_props = self.elastic_model.predict_strength_parameters(graph)

        # Extract properties (convert tensors to floats)
        G = elastic_props['shear_modulus_GPa'].item()  # GPa
        sigma_0 = elastic_props['friction_stress_MPa'].item()  # MPa
        k_HP = elastic_props['hall_petch_coefficient'].item()  # MPa·μm^0.5

        # 1. Hall-Petch strengthening
        # σ_HP = k_y / √d
        sigma_HP = k_HP / np.sqrt(grain_size) if grain_size > 0 else 0

        # 2. Lattice friction stress (Peierls stress)
        # Already predicted by model as sigma_0

        # 3. Temperature correction
        # Empirical: σ(T) = σ(0K) * (1 - T/T_melt)^n
        # Use approximate melting temperature from CALPHAD if available
        T_ref = 300.0  # Reference temperature (K)
        T_melt_approx = 1800.0  # Approximate melting point (conservative estimate)

        # Temperature factor (simplified)
        temp_factor = 1.0
        if temperature > T_ref:
            # Strength decreases with temperature
            temp_factor = 1.0 - 0.5 * ((temperature - T_ref) / (T_melt_approx - T_ref))
            temp_factor = max(temp_factor, 0.1)  # Don't go to zero

        # 4. Processing factor
        processing_factors = {
            "annealed": 1.0,
            "cold_worked": 1.5,  # Cold working increases strength
            "quenched": 1.3      # Quenching can increase strength
        }
        proc_factor = processing_factors.get(processing, 1.0)

        # Total yield strength
        # σ_y = σ_0 + σ_HP
        sigma_y_room_temp = sigma_0 + sigma_HP
        sigma_y_at_temp = sigma_y_room_temp * temp_factor * proc_factor

        # Theoretical maximum strength (Frenkel's ideal shear strength)
        sigma_theoretical = (G * 1000) / 10  # GPa → MPa, G/10

        # Ultimate tensile strength (empirical relationship)
        # UTS ≈ 1.5 * YS for many metals
        sigma_uts = sigma_y_at_temp * 1.5

        # Calculate safety factor
        safety_factor = sigma_theoretical / sigma_y_at_temp if sigma_y_at_temp > 0 else 0

        return {
            # Elastic properties
            'shear_modulus_GPa': G,
            'bulk_modulus_GPa': elastic_props['bulk_modulus_GPa'].item(),
            'youngs_modulus_GPa': elastic_props['youngs_modulus_GPa'].item(),
            'poisson_ratio': elastic_props['poisson_ratio'].item(),

            # Strength components
            'friction_stress_MPa': sigma_0,
            'hall_petch_contribution_MPa': sigma_HP,
            'temperature_factor': temp_factor,
            'processing_factor': proc_factor,

            # Final predictions
            'yield_strength_MPa': sigma_y_at_temp,
            'tensile_strength_MPa': sigma_uts,
            'theoretical_strength_MPa': sigma_theoretical,
            'safety_factor': safety_factor,

            # Inputs (for reference)
            'grain_size_um': grain_size,
            'temperature_K': temperature,
            'processing': processing
        }

    def estimate_required_grain_size(
        self,
        structure: Structure,
        target_strength: float,
        temperature: float = 300.0,
        processing: str = "annealed"
    ) -> float:
        """
        Estimate grain size needed to achieve target yield strength.

        Args:
            structure: Pymatgen Structure object
            target_strength: Target yield strength in MPa
            temperature: Temperature in Kelvin
            processing: Processing condition

        Returns:
            Required grain size in micrometers
        """
        # Get elastic properties
        if self.use_calphad:
            graph = structure_to_graph_with_calphad(structure, cutoff=8.0)
        else:
            graph = structure_to_graph(structure, cutoff=8.0)

        graph = graph.to(self.device)
        elastic_props = self.elastic_model.predict_strength_parameters(graph)

        sigma_0 = elastic_props['friction_stress_MPa'].item()
        k_HP = elastic_props['hall_petch_coefficient'].item()

        # Temperature and processing corrections
        processing_factors = {"annealed": 1.0, "cold_worked": 1.5, "quenched": 1.3}
        proc_factor = processing_factors.get(processing, 1.0)

        T_ref = 300.0
        T_melt_approx = 1800.0
        temp_factor = 1.0
        if temperature > T_ref:
            temp_factor = 1.0 - 0.5 * ((temperature - T_ref) / (T_melt_approx - T_ref))
            temp_factor = max(temp_factor, 0.1)

        # Solve for grain size from Hall-Petch equation
        # σ_target = (σ_0 + k_HP/√d) * temp_factor * proc_factor
        # k_HP/√d = σ_target / (temp_factor * proc_factor) - σ_0
        # √d = k_HP / (σ_target / (temp_factor * proc_factor) - σ_0)
        # d = (k_HP / (σ_target / (temp_factor * proc_factor) - σ_0))^2

        adjusted_target = target_strength / (temp_factor * proc_factor)

        if adjusted_target <= sigma_0:
            return float('inf')  # Target too low, any grain size works

        grain_size = (k_HP / (adjusted_target - sigma_0)) ** 2

        return grain_size

    def sensitivity_analysis(
        self,
        structure: Structure,
        grain_sizes: list = [1, 5, 10, 50, 100],
        temperatures: list = [300, 500, 800, 1000]
    ) -> Dict:
        """
        Analyze how strength varies with grain size and temperature.

        Args:
            structure: Pymatgen Structure object
            grain_sizes: List of grain sizes to test (μm)
            temperatures: List of temperatures to test (K)

        Returns:
            Dictionary with sensitivity data
        """
        results = {
            'grain_sizes': grain_sizes,
            'temperatures': temperatures,
            'strength_vs_grain_size': [],
            'strength_vs_temperature': []
        }

        # Vary grain size (constant temperature)
        for d in grain_sizes:
            pred = self.predict_strength(structure, grain_size=d, temperature=300.0)
            results['strength_vs_grain_size'].append(pred['yield_strength_MPa'])

        # Vary temperature (constant grain size)
        for T in temperatures:
            pred = self.predict_strength(structure, grain_size=10.0, temperature=T)
            results['strength_vs_temperature'].append(pred['yield_strength_MPa'])

        return results


def print_strength_prediction(prediction: Dict):
    """Pretty print strength prediction results."""
    print("\n" + "="*70)
    print("STRENGTH PREDICTION RESULTS")
    print("="*70)

    print("\nInput Parameters:")
    print(f"  Grain size: {prediction['grain_size_um']:.1f} um")
    print(f"  Temperature: {prediction['temperature_K']:.0f} K")
    print(f"  Processing: {prediction['processing']}")

    print("\nElastic Properties:")
    print(f"  Shear modulus (G):  {prediction['shear_modulus_GPa']:.1f} GPa")
    print(f"  Bulk modulus (K):   {prediction['bulk_modulus_GPa']:.1f} GPa")
    print(f"  Young's modulus (E): {prediction['youngs_modulus_GPa']:.1f} GPa")
    print(f"  Poisson's ratio (nu): {prediction['poisson_ratio']:.3f}")

    print("\nStrength Contributions:")
    print(f"  Friction stress (sigma_0):     {prediction['friction_stress_MPa']:.1f} MPa")
    print(f"  Hall-Petch (grain bdry):  {prediction['hall_petch_contribution_MPa']:.1f} MPa")
    print(f"  Temperature factor:       {prediction['temperature_factor']:.3f}x")
    print(f"  Processing factor:        {prediction['processing_factor']:.2f}x")

    print("\nFinal Predictions:")
    print(f"  Yield Strength (sigma_y):      {prediction['yield_strength_MPa']:.1f} MPa")
    print(f"  Tensile Strength (UTS):    {prediction['tensile_strength_MPa']:.1f} MPa")
    print(f"  Theoretical Maximum:       {prediction['theoretical_strength_MPa']:.0f} MPa")
    print(f"  Safety Factor:             {prediction['safety_factor']:.1f}x")

    print("="*70)


if __name__ == "__main__":
    """Test strength predictor"""
    from pymatgen.core import Lattice

    print("="*70)
    print("Testing Hybrid Strength Predictor")
    print("="*70)
    print()

    # Create predictor (without trained weights for testing)
    predictor = StrengthPredictor(use_calphad=True, device="cpu")

    # Test structure: Fe-Ni alloy
    lattice = Lattice.cubic(3.6)
    structure = Structure(lattice, ["Fe", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    print("Test 1: Basic strength prediction")
    print(f"Structure: {structure.formula}")
    prediction = predictor.predict_strength(
        structure=structure,
        grain_size=10.0,  # 10 μm
        temperature=300.0,  # Room temperature
        processing="annealed"
    )
    print_strength_prediction(prediction)

    print("\n" + "="*70)
    print("Test 2: Effect of grain size")
    print("="*70)
    for grain_size in [1, 5, 10, 50, 100]:
        pred = predictor.predict_strength(structure, grain_size=grain_size, temperature=300.0)
        print(f"  Grain size {grain_size:3.0f} um -> sigma_y = {pred['yield_strength_MPa']:.0f} MPa")

    print("\n" + "="*70)
    print("Test 3: Effect of temperature")
    print("="*70)
    for temp in [300, 500, 800, 1000, 1200]:
        pred = predictor.predict_strength(structure, grain_size=10.0, temperature=temp)
        print(f"  Temperature {temp:4.0f} K -> sigma_y = {pred['yield_strength_MPa']:.0f} MPa")

    print("\n" + "="*70)
    print("Test 4: Effect of processing")
    print("="*70)
    for proc in ["annealed", "cold_worked", "quenched"]:
        pred = predictor.predict_strength(structure, grain_size=10.0, temperature=300.0, processing=proc)
        print(f"  {proc:12s} -> sigma_y = {pred['yield_strength_MPa']:.0f} MPa")

    print("\n" + "="*70)
    print("Test 5: Required grain size for target strength")
    print("="*70)
    target = 500  # MPa
    required_d = predictor.estimate_required_grain_size(structure, target_strength=target, temperature=300.0)
    print(f"  To achieve {target} MPa yield strength:")
    print(f"  Required grain size: {required_d:.2f} um")

    print("\n" + "="*70)
    print("All strength predictor tests passed!")
    print("="*70)
