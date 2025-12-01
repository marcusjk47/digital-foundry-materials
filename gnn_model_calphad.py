"""
Enhanced CGCNN Model with CALPHAD Feature Support

Extends the base CGCNN to handle CALPHAD-enhanced node and edge features:
- Node features: [atomic_num, element_features (9), CALPHAD_features (3)] = 13 dimensions
- Edge features: [distance, mixing_energy] = 2 dimensions

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch


class CALPHADCGConv(MessagePassing):
    """
    Crystal Graph Convolutional Layer with CALPHAD feature support.

    Enhanced version that handles multi-dimensional edge features
    including thermodynamic mixing energies.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        """
        Args:
            node_dim: Dimension of node features (13 for CALPHAD-enhanced)
            edge_dim: Dimension of edge features (2 for distance + mixing_energy)
            hidden_dim: Dimension of hidden layers
        """
        super().__init__(aggr='add', node_dim=0)
        self.node_feature_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Edge network: transforms [node_i, node_j, edge_attr] -> edge message
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()
        )

        # Node update network: transforms [node, aggregated_messages] -> updated node
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Updated node features [num_nodes, node_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Construct messages to pass along edges.

        Args:
            x_i: Target node features [num_edges, node_dim]
            x_j: Source node features [num_edges, node_dim]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Edge messages [num_edges, hidden_dim]
        """
        # Concatenate source node, target node, and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(z)

    def update(self, aggr_out, x):
        """
        Update node features with aggregated messages.

        Args:
            aggr_out: Aggregated messages [num_nodes, hidden_dim]
            x: Current node features [num_nodes, node_dim]

        Returns:
            Updated node features [num_nodes, node_dim]
        """
        # Concatenate current node features with aggregated messages
        z = torch.cat([x, aggr_out], dim=-1)
        # Apply MLP and add residual connection
        return self.node_mlp(z) + x


class CGCNN_CALPHAD(nn.Module):
    """
    CGCNN with CALPHAD thermodynamic feature enhancement.

    This model accepts pre-computed node features that include:
    - Atomic number (1 feature)
    - Element properties (9 features): atomic mass, electronegativity, etc.
    - CALPHAD features (3 features): melting point, heat capacity, formation enthalpy

    And edge features that include:
    - Interatomic distance (1 feature)
    - CALPHAD mixing energy (1 feature)

    Architecture:
    1. Input projection: 13D node features -> node_feature_dim
    2. Edge feature projection: 2D edge features -> edge_feature_dim
    3. Multiple CALPHADCGConv layers for message passing
    4. Global pooling: node features -> graph-level features
    5. MLP for final prediction
    """

    def __init__(
        self,
        input_node_dim=13,          # 1 + 9 + 3 (atomic_num + element_feats + CALPHAD_feats)
        input_edge_dim=2,           # distance + mixing_energy
        node_feature_dim=64,        # Internal node feature dimension
        edge_feature_dim=32,        # Internal edge feature dimension
        hidden_dim=128,             # Hidden dimension for message passing
        n_conv=3,                   # Number of convolutional layers
        n_hidden=1,                 # Number of hidden layers in output MLP
        output_dim=1,               # Output dimension (1 for regression)
        dropout=0.1                 # Dropout rate
    ):
        """
        Args:
            input_node_dim: Input node feature dimension (13 for CALPHAD)
            input_edge_dim: Input edge feature dimension (2 for CALPHAD)
            node_feature_dim: Internal node feature dimension after projection
            edge_feature_dim: Internal edge feature dimension after projection
            hidden_dim: Dimension of hidden layers in convolution
            n_conv: Number of convolutional layers
            n_hidden: Number of hidden layers in output MLP
            output_dim: Dimension of output (1 for regression)
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim

        # Input projection layers
        # Project 13D CALPHAD features to internal feature dimension
        self.node_projection = nn.Sequential(
            nn.Linear(input_node_dim, node_feature_dim),
            nn.Softplus(),
            nn.BatchNorm1d(node_feature_dim)
        )

        # Project 2D edge features (distance + mixing_energy) to internal dimension
        self.edge_projection = nn.Sequential(
            nn.Linear(input_edge_dim, edge_feature_dim),
            nn.Softplus()
        )

        # Convolutional layers
        self.convs = nn.ModuleList([
            CALPHADCGConv(node_feature_dim, edge_feature_dim, hidden_dim)
            for _ in range(n_conv)
        ])

        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(node_feature_dim)
            for _ in range(n_conv)
        ])

        # Output MLP
        layers = []
        in_dim = node_feature_dim
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Softplus(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.output_mlp = nn.Sequential(*layers)

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: torch_geometric.data.Data or Batch object with:
                - x: Node features [num_nodes, 13] (CALPHAD-enhanced)
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, 2] (distance + mixing_energy)
                - batch: Batch assignment (for batched graphs)

        Returns:
            Predicted properties [batch_size, output_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Project input features to internal dimensions
        x = self.node_projection(x)
        edge_attr = self.edge_projection(edge_attr)

        # Apply convolutional layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)

        # Global pooling: aggregate node features to graph-level
        x = global_mean_pool(x, batch)

        # Final prediction
        out = self.output_mlp(x)

        return out


class CGCNN_CALPHAD_Regressor(nn.Module):
    """
    Wrapper for CGCNN_CALPHAD for regression tasks.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = CGCNN_CALPHAD(output_dim=1, **kwargs)

    def forward(self, data):
        return self.model(data).squeeze(-1)

    def predict(self, data):
        """Make predictions without gradients."""
        self.eval()
        with torch.no_grad():
            return self.forward(data)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model):
    """
    Print model architecture and parameter count.

    Args:
        model: PyTorch model
    """
    print("="*60)
    print("CALPHAD-Enhanced CGCNN Model Architecture")
    print("="*60)
    print(model)
    print()
    print("="*60)
    print("Parameter Count")
    print("="*60)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32
    print("="*60)


if __name__ == "__main__":
    """Test the CALPHAD-enhanced CGCNN model"""
    from crystal_graph import structure_to_graph_with_calphad
    from pymatgen.core import Lattice, Structure

    print("="*60)
    print("Testing CALPHAD-Enhanced CGCNN Model")
    print("="*60)
    print()

    # Test 1: Create CALPHAD-enhanced model
    print("Test 1: Model instantiation")
    model = CGCNN_CALPHAD(
        input_node_dim=13,      # CALPHAD-enhanced features
        input_edge_dim=2,       # distance + mixing_energy
        node_feature_dim=64,
        edge_feature_dim=32,
        hidden_dim=128,
        n_conv=3,
        n_hidden=1,
        output_dim=1
    )
    print_model_info(model)
    print()

    # Test 2: Forward pass with CALPHAD-enhanced graph
    print("Test 2: Forward pass with CALPHAD-enhanced structure")
    lattice = Lattice.cubic(2.87)
    structure = Structure(lattice, ["Fe", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Create CALPHAD-enhanced graph
    graph = structure_to_graph_with_calphad(structure, cutoff=4.0)

    print(f"Input: {structure.formula}")
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Node features shape: {graph.x.shape} (Expected: [2, 13])")
    print(f"Edge features shape: {graph.edge_attr.shape} (Expected: [num_edges, 2])")

    model.eval()
    with torch.no_grad():
        output = model(graph)

    print(f"Output shape: {output.shape}")
    print(f"Predicted value: {output.item():.4f}")
    print()

    # Test 3: Batch forward pass with multiple CALPHAD-enhanced structures
    print("Test 3: Batch forward pass")
    structures = [
        Structure(Lattice.cubic(2.87), ["Fe", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure(Lattice.cubic(3.60), ["Cu", "Zn"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure(Lattice.cubic(4.05), ["Al", "Mg"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    ]

    graphs = [structure_to_graph_with_calphad(s, cutoff=4.0) for s in structures]
    batch = Batch.from_data_list(graphs)

    print(f"Batch size: {len(structures)}")
    print(f"Batch node features shape: {batch.x.shape}")
    print(f"Batch edge features shape: {batch.edge_attr.shape}")

    with torch.no_grad():
        outputs = model(batch)

    print(f"Output shape: {outputs.shape}")
    print(f"Predicted values:")
    for i, (struct, pred) in enumerate(zip(structures, outputs)):
        print(f"  {struct.formula}: {pred.item():.4f}")
    print()

    # Test 4: Compare model sizes
    print("Test 4: Model comparison")
    from gnn_model import CGCNN

    base_model = CGCNN(
        node_feature_dim=64,
        edge_feature_dim=1,
        hidden_dim=128,
        n_conv=3
    )

    calphad_model = CGCNN_CALPHAD(
        input_node_dim=13,
        input_edge_dim=2,
        node_feature_dim=64,
        edge_feature_dim=32,
        hidden_dim=128,
        n_conv=3
    )

    print(f"Base CGCNN parameters:     {count_parameters(base_model):,}")
    print(f"CALPHAD CGCNN parameters:  {count_parameters(calphad_model):,}")
    print(f"Additional parameters:     {count_parameters(calphad_model) - count_parameters(base_model):,}")
    print(f"Increase:                  {(count_parameters(calphad_model) / count_parameters(base_model) - 1) * 100:.1f}%")
    print()

    # Test 5: Regressor wrapper
    print("Test 5: Regressor wrapper")
    regressor = CGCNN_CALPHAD_Regressor(
        input_node_dim=13,
        input_edge_dim=2,
        n_conv=2
    )

    with torch.no_grad():
        pred = regressor.predict(graph)

    print(f"Regressor prediction: {pred.item():.4f}")
    print()

    print("All CALPHAD-enhanced CGCNN model tests passed!")
