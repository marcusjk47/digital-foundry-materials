"""
Multi-Task CGCNN for Elastic Properties Prediction

Predicts multiple elastic properties simultaneously:
- Shear modulus (G)
- Bulk modulus (K)
- Young's modulus (E)

These are used for strength prediction via empirical models.

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import torch.nn as nn
from gnn_model_calphad import CGCNN_CALPHAD, CALPHADCGConv, count_parameters
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, Batch


class CGCNN_Elastic(nn.Module):
    """
    Multi-task CGCNN for predicting elastic properties.

    Supports both standard and CALPHAD-enhanced features.
    Predicts: [shear_modulus, bulk_modulus, youngs_modulus]
    """

    def __init__(
        self,
        input_node_dim=13,      # 13 for CALPHAD, 1 for standard
        input_edge_dim=2,       # 2 for CALPHAD, 1 for standard
        node_feature_dim=64,
        edge_feature_dim=32,
        hidden_dim=128,
        n_conv=3,
        n_hidden=2,
        dropout=0.1,
        use_calphad=True
    ):
        """
        Args:
            input_node_dim: Input node feature dimension
            input_edge_dim: Input edge feature dimension
            node_feature_dim: Internal node feature dimension
            edge_feature_dim: Internal edge feature dimension
            hidden_dim: Hidden dimension for message passing
            n_conv: Number of convolutional layers
            n_hidden: Number of hidden layers in output MLP
            dropout: Dropout rate
            use_calphad: Whether using CALPHAD features
        """
        super().__init__()

        self.use_calphad = use_calphad
        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim

        # Input projection layers
        if use_calphad:
            self.node_projection = nn.Sequential(
                nn.Linear(input_node_dim, node_feature_dim),
                nn.Softplus(),
                nn.BatchNorm1d(node_feature_dim)
            )
            self.edge_projection = nn.Sequential(
                nn.Linear(input_edge_dim, edge_feature_dim),
                nn.Softplus()
            )
        else:
            # Standard atomic number embedding
            self.node_projection = nn.Embedding(100, node_feature_dim)
            self.edge_projection = nn.Sequential(
                nn.Linear(input_edge_dim, edge_feature_dim),
                nn.Softplus()
            )

        # Convolutional layers
        self.convs = nn.ModuleList([
            CALPHADCGConv(node_feature_dim, edge_feature_dim, hidden_dim)
            for _ in range(n_conv)
        ])

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(node_feature_dim)
            for _ in range(n_conv)
        ])

        # Shared feature extractor
        shared_layers = []
        in_dim = node_feature_dim
        for _ in range(n_hidden):
            shared_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Softplus(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        self.shared_mlp = nn.Sequential(*shared_layers)

        # Task-specific heads
        self.shear_head = nn.Linear(hidden_dim, 1)
        self.bulk_head = nn.Linear(hidden_dim, 1)
        self.youngs_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: torch_geometric.data.Data or Batch object

        Returns:
            Dictionary with elastic property predictions (GPa)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Project input features
        if self.use_calphad:
            x = self.node_projection(x.float())
        else:
            x = self.node_projection(x.long())

        edge_attr = self.edge_projection(edge_attr)

        # Apply convolutional layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Shared feature extraction
        features = self.shared_mlp(x)

        # Task-specific predictions
        shear_modulus = self.shear_head(features).squeeze(-1)
        bulk_modulus = self.bulk_head(features).squeeze(-1)
        youngs_modulus = self.youngs_head(features).squeeze(-1)

        return {
            'shear_modulus': shear_modulus,
            'bulk_modulus': bulk_modulus,
            'youngs_modulus': youngs_modulus
        }

    def predict_single(self, data):
        """Predict for single structure (convenience method)."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(data)
            # Convert to scalars
            return {k: v.item() for k, v in predictions.items()}


class ElasticPropertyPredictor(nn.Module):
    """
    Wrapper for elastic property prediction with convenience methods.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = CGCNN_Elastic(**kwargs)

    def forward(self, data):
        return self.model(data)

    def predict(self, data):
        """Make predictions without gradients."""
        self.eval()
        with torch.no_grad():
            return self.forward(data)

    def predict_strength_parameters(self, data):
        """
        Predict elastic properties and derive strength-related parameters.

        Returns:
            Dictionary with elastic properties and derived parameters
        """
        predictions = self.predict(data)

        # Extract moduli (convert to proper units if needed)
        G = predictions['shear_modulus']  # GPa
        K = predictions['bulk_modulus']   # GPa
        E = predictions['youngs_modulus'] # GPa

        # Calculate Poisson's ratio from E and G
        # E = 2G(1 + ν)  →  ν = (E/2G) - 1
        nu = (E / (2 * G)) - 1
        nu = torch.clamp(nu, -1, 0.5)  # Physical bounds

        # Estimate Hall-Petch coefficient from shear modulus
        # Empirical relationship: k_y ≈ 0.2 * G^(3/2) (MPa·μm^0.5)
        k_HP = 0.2 * (G * 1000) ** 1.5  # Convert GPa to MPa

        # Estimate friction stress (σ_0)
        # Empirical: σ_0 ≈ G/200 (GPa → MPa)
        sigma_0 = (G * 1000) / 200

        return {
            'shear_modulus_GPa': G,
            'bulk_modulus_GPa': K,
            'youngs_modulus_GPa': E,
            'poisson_ratio': nu,
            'hall_petch_coefficient': k_HP,
            'friction_stress_MPa': sigma_0
        }


if __name__ == "__main__":
    """Test the elastic property model"""
    from crystal_graph import structure_to_graph_with_calphad
    from pymatgen.core import Lattice, Structure

    print("="*60)
    print("Testing Multi-Task Elastic Property CGCNN")
    print("="*60)
    print()

    # Test 1: Create model
    print("Test 1: Model instantiation")
    model = CGCNN_Elastic(
        input_node_dim=13,
        input_edge_dim=2,
        node_feature_dim=64,
        edge_feature_dim=32,
        hidden_dim=128,
        n_conv=3,
        n_hidden=2,
        use_calphad=True
    )

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    print()

    # Test 2: Forward pass with CALPHAD-enhanced graph
    print("Test 2: Forward pass with Fe-Ni structure")
    lattice = Lattice.cubic(3.6)
    structure = Structure(lattice, ["Fe", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    graph = structure_to_graph_with_calphad(structure, cutoff=4.0)

    model.eval()
    with torch.no_grad():
        predictions = model(graph)

    print(f"Input: {structure.formula}")
    print(f"Predictions:")
    for prop, value in predictions.items():
        print(f"  {prop}: {value.item():.2f} GPa")
    print()

    # Test 3: Batch prediction
    print("Test 3: Batch prediction")
    structures = [
        Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure(Lattice.cubic(3.60), ["Cu", "Zn"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure(Lattice.cubic(4.05), ["Al", "Mg"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    ]

    graphs = [structure_to_graph_with_calphad(s, cutoff=4.0) for s in structures]
    batch = Batch.from_data_list(graphs)

    with torch.no_grad():
        batch_predictions = model(batch)

    print(f"Batch size: {len(structures)}")
    for i, struct in enumerate(structures):
        print(f"\n{struct.formula}:")
        for prop, values in batch_predictions.items():
            print(f"  {prop}: {values[i].item():.2f} GPa")
    print()

    # Test 4: Strength parameter prediction
    print("Test 4: Strength parameter prediction")
    predictor = ElasticPropertyPredictor(
        input_node_dim=13,
        input_edge_dim=2,
        use_calphad=True
    )

    strength_params = predictor.predict_strength_parameters(graph)

    print(f"Structure: {structure.formula}")
    print("\nElastic Properties:")
    print(f"  Shear modulus (G): {strength_params['shear_modulus_GPa'].item():.2f} GPa")
    print(f"  Bulk modulus (K): {strength_params['bulk_modulus_GPa'].item():.2f} GPa")
    print(f"  Young's modulus (E): {strength_params['youngs_modulus_GPa'].item():.2f} GPa")
    print(f"  Poisson's ratio (ν): {strength_params['poisson_ratio'].item():.3f}")

    print("\nStrength Parameters:")
    print(f"  Friction stress (σ₀): {strength_params['friction_stress_MPa'].item():.1f} MPa")
    print(f"  Hall-Petch coeff (k_y): {strength_params['hall_petch_coefficient'].item():.1f} MPa·μm^0.5")
    print()

    print("All elastic property model tests passed!")
