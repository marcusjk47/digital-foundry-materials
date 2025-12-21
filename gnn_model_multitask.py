"""
Multi-Task Crystal Graph Convolutional Neural Network

Predicts multiple material properties simultaneously from crystal structure.
Supports: formation energy, band gap, density, elastic properties, etc.

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class ConvLayer(nn.Module):
    """Graph convolution layer for crystal structures."""

    def __init__(self, node_dim, edge_dim):
        super(ConvLayer, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Edge network
        self.edge_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, 2 * node_dim),
            nn.Softplus(),
            nn.Linear(2 * node_dim, node_dim),
            nn.Softplus()
        )

        # Node update
        self.node_net = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.Softplus()
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Updated node features [num_nodes, node_dim]
        """
        row, col = edge_index

        # Concatenate node and edge features
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)

        # Update edges
        edge_emb = self.edge_net(edge_input)

        # Aggregate messages
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, col, edge_emb)

        # Update nodes
        node_input = torch.cat([x, aggr], dim=-1)
        x_new = self.node_net(node_input)

        return x_new + x  # Residual connection


class CGCNN_MultiTask(nn.Module):
    """
    Multi-task CGCNN for predicting multiple properties.

    Shared encoder + task-specific heads.
    """

    def __init__(
        self,
        node_feature_dim=64,
        edge_feature_dim=1,
        hidden_dim=128,
        n_conv=3,
        n_hidden=2,
        properties=['formation_energy_per_atom', 'band_gap', 'density']
    ):
        """
        Args:
            node_feature_dim: Dimension of input node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden dimension for convolutions
            n_conv: Number of convolution layers
            n_hidden: Number of hidden layers in task heads
            properties: List of property names to predict
        """
        super(CGCNN_MultiTask, self).__init__()

        self.properties = properties
        self.n_properties = len(properties)

        # Input embedding
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)

        # Shared convolution layers
        self.conv_layers = nn.ModuleList([
            ConvLayer(hidden_dim, hidden_dim)
            for _ in range(n_conv)
        ])

        # Shared pooling and initial processing
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()
        )

        # Task-specific heads (one per property)
        self.task_heads = nn.ModuleDict()
        for prop in properties:
            layers = []
            input_dim = hidden_dim

            for _ in range(n_hidden):
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.Softplus()
                ])
                input_dim = hidden_dim

            layers.append(nn.Linear(hidden_dim, 1))

            self.task_heads[prop] = nn.Sequential(*layers)

    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features
                - edge_index: Edge connectivity
                - edge_attr: Edge features
                - batch: Batch assignment

        Returns:
            Dictionary mapping property names to predictions
        """
        # Embed features
        x = self.node_embedding(data.x)
        edge_attr = self.edge_embedding(data.edge_attr)

        # Apply convolutions
        for conv in self.conv_layers:
            x = conv(x, data.edge_index, edge_attr)

        # Global pooling
        x = global_mean_pool(x, data.batch)

        # Shared processing
        x = self.shared_fc(x)

        # Task-specific predictions
        predictions = {}
        for prop in self.properties:
            predictions[prop] = self.task_heads[prop](x).squeeze(-1)

        return predictions


class CGCNN_MultiTask_CALPHAD(nn.Module):
    """
    Multi-task CGCNN with CALPHAD features.

    Enhanced with thermodynamic properties for better predictions.
    """

    def __init__(
        self,
        input_node_dim=13,
        input_edge_dim=2,
        node_feature_dim=64,
        edge_feature_dim=32,
        hidden_dim=128,
        n_conv=3,
        n_hidden=2,
        properties=['formation_energy_per_atom', 'band_gap', 'density']
    ):
        """
        Args:
            input_node_dim: Input node feature dimension (13 for CALPHAD)
            input_edge_dim: Input edge feature dimension (2 for CALPHAD)
            node_feature_dim: Embedded node feature dimension
            edge_feature_dim: Embedded edge feature dimension
            hidden_dim: Hidden dimension
            n_conv: Number of convolution layers
            n_hidden: Number of hidden layers in task heads
            properties: List of property names to predict
        """
        super(CGCNN_MultiTask_CALPHAD, self).__init__()

        self.properties = properties
        self.n_properties = len(properties)

        # Input embeddings
        self.node_embedding = nn.Linear(input_node_dim, node_feature_dim)
        self.edge_embedding = nn.Linear(input_edge_dim, edge_feature_dim)

        # Convolution layers
        self.conv_layers = nn.ModuleList([
            ConvLayer(node_feature_dim, edge_feature_dim)
            for _ in range(n_conv)
        ])

        # Shared processing
        self.shared_fc = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()
        )

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for prop in properties:
            layers = []
            input_dim = hidden_dim

            for _ in range(n_hidden):
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.Softplus()
                ])
                input_dim = hidden_dim

            layers.append(nn.Linear(hidden_dim, 1))

            self.task_heads[prop] = nn.Sequential(*layers)

    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object

        Returns:
            Dictionary mapping property names to predictions
        """
        # Embed features
        x = self.node_embedding(data.x)
        edge_attr = self.edge_embedding(data.edge_attr)

        # Apply convolutions
        for conv in self.conv_layers:
            x = conv(x, data.edge_index, edge_attr)

        # Global pooling
        x = global_mean_pool(x, data.batch)

        # Shared processing
        x = self.shared_fc(x)

        # Task-specific predictions
        predictions = {}
        for prop in self.properties:
            predictions[prop] = self.task_heads[prop](x).squeeze(-1)

        return predictions


# Property metadata
PROPERTY_INFO = {
    'formation_energy_per_atom': {
        'name': 'Formation Energy',
        'unit': 'eV/atom',
        'description': 'Energy to form compound from elements',
        'typical_range': (-2.0, 0.5),
        'lower_is_better': True
    },
    'band_gap': {
        'name': 'Band Gap',
        'unit': 'eV',
        'description': 'Electronic band gap (0 = metal)',
        'typical_range': (0.0, 10.0),
        'lower_is_better': False
    },
    'density': {
        'name': 'Density',
        'unit': 'g/cm³',
        'description': 'Mass density',
        'typical_range': (1.0, 25.0),
        'lower_is_better': False
    },
    'energy_above_hull': {
        'name': 'Energy Above Hull',
        'unit': 'eV/atom',
        'description': 'Stability (0 = stable)',
        'typical_range': (0.0, 0.5),
        'lower_is_better': True
    },
    'volume': {
        'name': 'Volume',
        'unit': 'Ų/atom',
        'description': 'Volume per atom',
        'typical_range': (5.0, 50.0),
        'lower_is_better': False
    },
    'bulk_modulus': {
        'name': 'Bulk Modulus',
        'unit': 'GPa',
        'description': 'Resistance to compression',
        'typical_range': (10.0, 400.0),
        'lower_is_better': False
    },
    'shear_modulus': {
        'name': 'Shear Modulus',
        'unit': 'GPa',
        'description': 'Resistance to shear deformation',
        'typical_range': (5.0, 200.0),
        'lower_is_better': False
    },
    'total_magnetization': {
        'name': 'Total Magnetization',
        'unit': 'μB',
        'description': 'Total magnetic moment',
        'typical_range': (0.0, 10.0),
        'lower_is_better': False
    },
    'efermi': {
        'name': 'Fermi Energy',
        'unit': 'eV',
        'description': 'Fermi level energy',
        'typical_range': (-10.0, 10.0),
        'lower_is_better': False
    }
}


def get_property_info(property_name):
    """Get metadata for a property."""
    return PROPERTY_INFO.get(property_name, {
        'name': property_name,
        'unit': '',
        'description': 'Property value',
        'typical_range': (None, None),
        'lower_is_better': False
    })


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """Test multi-task model."""
    print("Testing Multi-Task CGCNN")
    print("=" * 70)

    # Create model
    properties = ['formation_energy_per_atom', 'band_gap', 'density']
    model = CGCNN_MultiTask(properties=properties)

    print(f"Properties: {properties}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass with dummy data
    from torch_geometric.data import Data

    dummy_data = Data(
        x=torch.randn(10, 64),
        edge_index=torch.randint(0, 10, (2, 30)),
        edge_attr=torch.randn(30, 1),
        batch=torch.zeros(10, dtype=torch.long)
    )

    predictions = model(dummy_data)

    print("\nPredictions:")
    for prop, value in predictions.items():
        print(f"  {prop}: {value.item():.4f}")

    print("\n✅ Multi-task model test passed!")
