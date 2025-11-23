"""
Crystal Graph Convolutional Neural Network (CGCNN) Implementation

Based on the paper:
"Crystal Graph Convolutional Neural Networks for an Accurate and
Interpretable Prediction of Material Properties"
Xie and Grossman, Physical Review Letters (2018)

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch


class CGConv(MessagePassing):
    """
    Crystal Graph Convolutional Layer.

    This layer implements the message passing operation for crystal graphs:
    1. Compute edge messages from source node, target node, and edge features
    2. Aggregate messages for each node
    3. Update node features with aggregated messages
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        """
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Dimension of hidden layers
        """
        super().__init__(aggr='add', node_dim=0)  # Use sum aggregation, node_dim=0 for batch dimension
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


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network for property prediction.

    Architecture:
    1. Embedding layer: atomic number -> node features
    2. Multiple CGConv layers for message passing
    3. Global pooling: node features -> graph-level features
    4. MLP for final prediction
    """

    def __init__(
        self,
        node_feature_dim=64,
        edge_feature_dim=1,
        hidden_dim=128,
        n_conv=3,
        n_hidden=1,
        output_dim=1,
        use_element_embedding=False
    ):
        """
        Args:
            node_feature_dim: Dimension of node feature vectors
            edge_feature_dim: Dimension of edge features (typically distance)
            hidden_dim: Dimension of hidden layers in convolution
            n_conv: Number of convolutional layers
            n_hidden: Number of hidden layers in output MLP
            output_dim: Dimension of output (1 for regression, n for classification)
            use_element_embedding: If True, use element feature embeddings.
                                  If False, use learned embeddings from atomic numbers.
        """
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.use_element_embedding = use_element_embedding

        # Embedding layer for atomic numbers
        if use_element_embedding:
            # Use pre-defined element features
            from element_features import get_element_feature_dim
            element_dim = get_element_feature_dim()
            self.embedding = nn.Linear(element_dim, node_feature_dim)
        else:
            # Learn embeddings for each atomic number
            self.embedding = nn.Embedding(100, node_feature_dim)  # Support up to Z=99

        # Convolutional layers
        self.convs = nn.ModuleList([
            CGConv(node_feature_dim, edge_feature_dim, hidden_dim)
            for _ in range(n_conv)
        ])

        # Batch normalization layers (optional, helps training)
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
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.output_mlp = nn.Sequential(*layers)

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: torch_geometric.data.Data or Batch object with:
                - x: Node features (atomic numbers or element features)
                - edge_index: Edge connectivity
                - edge_attr: Edge features
                - batch: Batch assignment (for batched graphs)

        Returns:
            Predicted properties [batch_size, output_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)

        # Embed atomic numbers or element features
        if self.use_element_embedding:
            x = self.embedding(x.float())
        else:
            x = self.embedding(x.long())

        # Apply convolutional layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)

        # Global pooling: aggregate node features to graph-level
        x = global_mean_pool(x, batch)

        # Final prediction
        out = self.output_mlp(x)

        return out


class CGCNNRegressor(nn.Module):
    """
    Wrapper for CGCNN for regression tasks with additional utilities.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = CGCNN(output_dim=1, **kwargs)

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
    print("Model Architecture")
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
    """Test the CGCNN model"""
    from crystal_graph import structure_to_graph
    from pymatgen.core import Lattice, Structure

    print("="*60)
    print("Testing CGCNN Model")
    print("="*60)
    print()

    # Test 1: Create a simple model
    print("Test 1: Model instantiation")
    model = CGCNN(
        node_feature_dim=64,
        edge_feature_dim=1,
        hidden_dim=128,
        n_conv=3,
        n_hidden=1,
        output_dim=1
    )
    print_model_info(model)
    print()

    # Test 2: Forward pass with single structure
    print("Test 2: Forward pass with single structure")
    lattice = Lattice.cubic(2.87)
    structure = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    graph = structure_to_graph(structure, cutoff=4.0)

    model.eval()
    with torch.no_grad():
        output = model(graph)

    print(f"Input: {structure.formula}")
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Output shape: {output.shape}")
    print(f"Predicted value: {output.item():.4f}")
    print()

    # Test 3: Batch forward pass
    print("Test 3: Batch forward pass")
    structures = [
        Structure(Lattice.cubic(2.87), ["Fe"], [[0, 0, 0]]),
        Structure(Lattice.cubic(3.60), ["Cu"], [[0, 0, 0]]),
        Structure(Lattice.cubic(4.05), ["Al"], [[0, 0, 0]]),
    ]

    graphs = [structure_to_graph(s, cutoff=4.0) for s in structures]
    batch = Batch.from_data_list(graphs)

    with torch.no_grad():
        outputs = model(batch)

    print(f"Batch size: {len(structures)}")
    print(f"Output shape: {outputs.shape}")
    print(f"Predicted values:")
    for i, (struct, pred) in enumerate(zip(structures, outputs)):
        print(f"  {struct.formula}: {pred.item():.4f}")
    print()

    # Test 4: Model with element embeddings
    print("Test 4: Model with element feature embeddings")
    model_elem = CGCNN(
        node_feature_dim=64,
        use_element_embedding=True,
        n_conv=2
    )

    print(f"Parameters: {count_parameters(model_elem):,}")
    print()

    print("All CGCNN model tests passed!")
