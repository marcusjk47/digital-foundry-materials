"""
PyTorch Dataset Class for Crystal Graph Data

Provides DataLoader-compatible dataset for training GNN models.

Author: Digital Foundry Materials Science Toolkit
"""

import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional
import numpy as np


class CrystalGraphDataset(Dataset):
    """
    PyTorch Dataset for crystal graph data.

    Can be used with PyTorch Geometric DataLoader for batch training.
    """

    def __init__(self, graphs: List[Data], transform=None):
        """
        Args:
            graphs: List of PyTorch Geometric Data objects
            transform: Optional transform to apply to each graph
        """
        self.graphs = graphs
        self.transform = transform

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]

        if self.transform:
            graph = self.transform(graph)

        return graph

    def get_target_statistics(self):
        """
        Calculate mean and std of target values for normalization.

        Returns:
            Tuple of (mean, std)
        """
        targets = [graph.y.item() for graph in self.graphs if hasattr(graph, 'y')]
        return np.mean(targets), np.std(targets)


class NormalizeTarget:
    """
    Transform to normalize target values.

    Useful for training stability and faster convergence.
    """

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'y'):
            data.y = (data.y - self.mean) / self.std
        return data

    def denormalize(self, normalized_value: torch.Tensor) -> torch.Tensor:
        """Convert normalized prediction back to original scale."""
        return normalized_value * self.std + self.mean


def split_dataset(
    dataset: CrystalGraphDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: CrystalGraphDataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Set random seed for reproducibility
    torch.manual_seed(random_seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_dataset)} ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_dataset)} ({test_ratio*100:.1f}%)")

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test sets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Batch size:    {batch_size}")

    return train_loader, val_loader, test_loader


def get_sample_batch(data_loader: DataLoader) -> Batch:
    """
    Get a sample batch from DataLoader for inspection.

    Args:
        data_loader: DataLoader to sample from

    Returns:
        Batch object
    """
    return next(iter(data_loader))


if __name__ == "__main__":
    """Test dataset and data loaders"""
    from gnn_data_collection import load_graph_dataset

    print("Testing CrystalGraphDataset...")

    # Try to load existing dataset
    try:
        graphs = load_graph_dataset("datasets/fe_ni_graphs.pkl")
    except FileNotFoundError:
        print("No dataset found. Run gnn_data_collection.py first.")
        exit(1)

    # Create dataset
    dataset = CrystalGraphDataset(graphs)
    print(f"Dataset size: {len(dataset)}")

    # Get normalization stats
    mean, std = dataset.get_target_statistics()
    print(f"Target mean: {mean:.4f}, std: {std:.4f}")

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=32
    )

    # Test batch loading
    print("\nTesting batch loading...")
    sample_batch = get_sample_batch(train_loader)
    print(f"Batch info:")
    print(f"  Num graphs in batch: {sample_batch.num_graphs}")
    print(f"  Total nodes: {sample_batch.num_nodes}")
    print(f"  Total edges: {sample_batch.num_edges}")
    print(f"  Node features shape: {sample_batch.x.shape}")
    print(f"  Edge features shape: {sample_batch.edge_attr.shape}")
    print(f"  Targets shape: {sample_batch.y.shape}")

    print("\nDataset and DataLoader tests passed!")
