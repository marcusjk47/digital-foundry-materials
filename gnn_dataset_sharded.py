"""
Sharded Dataset Loader for Memory-Efficient Training

Loads graph datasets in shards to minimize memory usage.
Supports caching, prefetching, and multi-dataset training.

Author: Digital Foundry Materials Science Toolkit
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Callable
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class ShardedCrystalGraphDataset(Dataset):
    """
    Memory-efficient dataset that loads shards on-demand.

    Uses LRU caching to keep only recent shards in memory.
    """

    def __init__(
        self,
        manifest_path: str,
        cache_size: int = 3,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize sharded dataset.

        Args:
            manifest_path: Path to manifest.json
            cache_size: Number of shards to keep in memory
            transform: Optional transform for data
            target_transform: Optional transform for targets
        """
        self.manifest_path = Path(manifest_path)
        self.dataset_dir = self.manifest_path.parent
        self.cache_size = cache_size
        self.transform = transform
        self.target_transform = target_transform

        # Load manifest
        with open(self.manifest_path, "r") as f:
            self.manifest = json.load(f)

        # Build index: global_idx -> (shard_idx, local_idx)
        self._build_index()

        # Shard cache (LRU)
        self._shard_cache = OrderedDict()

        print(f"Loaded sharded dataset: {self.manifest['dataset_name']}")
        print(f"  Total materials: {self.manifest['total_materials']:,}")
        print(f"  Shards: {self.manifest['num_shards']}")
        print(f"  Cache size: {cache_size} shards")

    def _build_index(self):
        """Build mapping from global index to (shard_idx, local_idx)"""
        self.index_map = []

        for shard_info in self.manifest["shards"]:
            shard_idx = shard_info["id"]
            count = shard_info["count"]

            for local_idx in range(count):
                self.index_map.append((shard_idx, local_idx))

    def __len__(self) -> int:
        """Total number of materials"""
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Data:
        """
        Get item at index (loads shard if needed).

        Args:
            idx: Global index

        Returns:
            Graph Data object
        """
        # Get shard and local index
        shard_idx, local_idx = self.index_map[idx]

        # Load shard (from cache or disk)
        shard_data = self._load_shard(shard_idx)

        # Get graph
        graph = shard_data[local_idx]

        # Apply transforms
        if self.transform is not None:
            graph = self.transform(graph)

        if self.target_transform is not None and hasattr(graph, 'y'):
            graph.y = self.target_transform(graph.y)

        return graph

    def _load_shard(self, shard_idx: int) -> List[Data]:
        """
        Load shard (with LRU caching).

        Args:
            shard_idx: Shard index

        Returns:
            List of graphs in shard
        """
        # Check cache
        if shard_idx in self._shard_cache:
            # Move to end (most recent)
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]

        # Load from disk
        shard_info = self.manifest["shards"][shard_idx]
        shard_path = self.dataset_dir / shard_info["path"]

        with open(shard_path, "rb") as f:
            shard_data = pickle.load(f)

        # Add to cache
        self._shard_cache[shard_idx] = shard_data
        self._shard_cache.move_to_end(shard_idx)

        # Evict old shards if cache is full
        while len(self._shard_cache) > self.cache_size:
            self._shard_cache.popitem(last=False)

        return shard_data

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        return {
            "total_materials": self.manifest["total_materials"],
            "num_shards": self.manifest["num_shards"],
            "shard_size": self.manifest["shard_size"],
            "calphad_enhanced": self.manifest.get("calphad_enhanced", False),
            "elements": self.manifest.get("elements", []),
            "properties": self.manifest.get("properties", [])
        }


class MultiDatasetLoader:
    """
    Combine multiple sharded datasets for training.
    """

    def __init__(
        self,
        manifest_paths: List[str],
        cache_size: int = 3,
        transform: Optional[Callable] = None
    ):
        """
        Initialize multi-dataset loader.

        Args:
            manifest_paths: List of paths to manifest.json files
            cache_size: Shards to cache per dataset
            transform: Optional transform
        """
        self.datasets = []

        for manifest_path in manifest_paths:
            dataset = ShardedCrystalGraphDataset(
                manifest_path=manifest_path,
                cache_size=cache_size,
                transform=transform
            )
            self.datasets.append(dataset)

        # Calculate cumulative sizes
        self._cumulative_sizes = self._calculate_cumulative_sizes()

        total_materials = sum(len(d) for d in self.datasets)
        print(f"\nCombined {len(self.datasets)} datasets:")
        print(f"  Total materials: {total_materials:,}")

    def _calculate_cumulative_sizes(self) -> List[int]:
        """Calculate cumulative sizes for indexing"""
        cumulative = [0]
        for dataset in self.datasets:
            cumulative.append(cumulative[-1] + len(dataset))
        return cumulative

    def __len__(self) -> int:
        """Total materials across all datasets"""
        return self._cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Data:
        """Get item from appropriate dataset"""
        # Find which dataset this index belongs to
        dataset_idx = self._find_dataset_idx(idx)

        # Get local index within that dataset
        local_idx = idx - self._cumulative_sizes[dataset_idx]

        # Return from appropriate dataset
        return self.datasets[dataset_idx][local_idx]

    def _find_dataset_idx(self, idx: int) -> int:
        """Find which dataset contains this global index"""
        for i in range(len(self._cumulative_sizes) - 1):
            if self._cumulative_sizes[i] <= idx < self._cumulative_sizes[i + 1]:
                return i
        raise IndexError(f"Index {idx} out of range")


def split_sharded_dataset(
    dataset: ShardedCrystalGraphDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> tuple:
    """
    Split sharded dataset into train/val/test.

    Args:
        dataset: Sharded dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility

    Returns:
        (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Generate indices
    total_size = len(dataset)
    indices = np.arange(total_size)

    # Shuffle
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Split
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    print(f"Dataset split:")
    print(f"  Train: {len(train_indices):,} ({len(train_indices)/total_size*100:.1f}%)")
    print(f"  Val:   {len(val_indices):,} ({len(val_indices)/total_size*100:.1f}%)")
    print(f"  Test:  {len(test_indices):,} ({len(test_indices)/total_size*100:.1f}%)")

    return train_indices, val_indices, test_indices


def create_sharded_dataloaders(
    manifest_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 0,
    cache_size: int = 3,
    random_seed: int = 42
) -> tuple:
    """
    Create train/val/test dataloaders from sharded dataset.

    Args:
        manifest_path: Path to manifest.json
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        num_workers: Number of worker processes
        cache_size: Shards to cache
        random_seed: Random seed

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Load dataset
    dataset = ShardedCrystalGraphDataset(
        manifest_path=manifest_path,
        cache_size=cache_size
    )

    # Split indices
    train_idx, val_idx, test_idx = split_sharded_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    # Create subset samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Num workers:   {num_workers}")

    return train_loader, val_loader, test_loader


def create_multi_dataset_loaders(
    manifest_paths: List[str],
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 0,
    cache_size: int = 3,
    random_seed: int = 42
) -> tuple:
    """
    Create dataloaders from multiple datasets combined.

    Args:
        manifest_paths: List of manifest.json paths
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        num_workers: Number of workers
        cache_size: Shards to cache per dataset
        random_seed: Random seed

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create multi-dataset
    multi_dataset = MultiDatasetLoader(
        manifest_paths=manifest_paths,
        cache_size=cache_size
    )

    # Split indices
    total_size = len(multi_dataset)
    indices = np.arange(total_size)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    # Create loaders
    train_loader = DataLoader(
        multi_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        multi_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        multi_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers
    )

    print(f"\nMulti-dataset loaders created:")
    print(f"  Total materials: {total_size:,}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test sharded dataset loader"""

    # Note: This test requires a sharded dataset to exist
    # Run dataset_manager.py first to create test dataset

    print("="*70)
    print("Testing Sharded Dataset Loader")
    print("="*70)
    print()

    # Create a small test dataset first
    from dataset_manager import DatasetManager
    from torch_geometric.data import Data
    import torch

    manager = DatasetManager(base_dir="datasets_test")

    # Create test dataset
    manager.create_dataset(
        name="test_sharded_loader",
        description="Test dataset for sharded loader",
        shard_size=10
    )

    # Add dummy data
    dummy_graphs = []
    for i in range(50):
        graph = Data(
            x=torch.randn(5, 13),
            edge_index=torch.randint(0, 5, (2, 10)),
            edge_attr=torch.randn(10, 2),
            y=torch.tensor([float(i)])
        )
        graph.material_id = f"mp-{i}"
        dummy_graphs.append(graph)

    manager.add_materials("test_sharded_loader", dummy_graphs)

    # Test 1: Load sharded dataset
    print("Test 1: Load sharded dataset")
    manifest_path = "datasets_test/test_sharded_loader/manifest.json"
    dataset = ShardedCrystalGraphDataset(manifest_path, cache_size=2)
    print(f"Dataset size: {len(dataset)}")
    print()

    # Test 2: Access items
    print("Test 2: Access items")
    item_0 = dataset[0]
    item_25 = dataset[25]
    item_49 = dataset[49]
    print(f"Item 0 y: {item_0.y.item()}")
    print(f"Item 25 y: {item_25.y.item()}")
    print(f"Item 49 y: {item_49.y.item()}")
    print()

    # Test 3: Create dataloaders
    print("Test 3: Create dataloaders")
    train_loader, val_loader, test_loader = create_sharded_dataloaders(
        manifest_path=manifest_path,
        batch_size=8,
        cache_size=2
    )
    print()

    # Test 4: Iterate through batch
    print("Test 4: Iterate through batch")
    batch = next(iter(train_loader))
    print(f"Batch size: {batch.num_graphs}")
    print(f"Node features shape: {batch.x.shape}")
    print(f"Targets shape: {batch.y.shape}")
    print()

    # Cleanup
    import shutil
    print("Cleaning up...")
    shutil.rmtree("datasets_test")

    print("\nAll sharded dataset loader tests passed!")
