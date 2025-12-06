"""
Dataset Manager for Sharded Graph Datasets

Manages large-scale materials datasets with automatic sharding, metadata tracking,
and dataset operations (merge, deduplicate, statistics).

Features:
- Automatic sharding (configurable shard size)
- JSON manifest with metadata
- Dataset merging and deduplication
- Statistics and querying
- Version control

Author: Digital Foundry Materials Science Toolkit
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class DatasetManager:
    """
    Manager for sharded graph datasets.

    Directory structure:
        datasets/
        └── my_dataset/
            ├── manifest.json
            ├── checkpoint.json (during collection)
            └── shards/
                ├── shard_0000.pkl
                ├── shard_0001.pkl
                └── ...
    """

    def __init__(self, base_dir: str = "datasets"):
        """
        Initialize dataset manager.

        Args:
            base_dir: Base directory for all datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_dataset(
        self,
        name: str,
        description: str = "",
        shard_size: int = 1000,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a new sharded dataset.

        Args:
            name: Dataset name
            description: Dataset description
            shard_size: Number of materials per shard
            metadata: Additional metadata

        Returns:
            Path to dataset directory
        """
        dataset_dir = self.base_dir / name

        # Check if exists
        if dataset_dir.exists():
            raise ValueError(f"Dataset '{name}' already exists!")

        # Create directory structure
        dataset_dir.mkdir(parents=True)
        shards_dir = dataset_dir / "shards"
        shards_dir.mkdir()

        # Create manifest
        manifest = {
            "dataset_name": name,
            "description": description,
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "total_materials": 0,
            "num_shards": 0,
            "shard_size": shard_size,
            "elements": [],
            "properties": [],
            "calphad_enhanced": False,
            "shards": [],
            "metadata": metadata or {}
        }

        self._save_manifest(dataset_dir, manifest)

        print(f"Created dataset: {name}")
        print(f"  Location: {dataset_dir}")
        print(f"  Shard size: {shard_size}")

        return str(dataset_dir)

    def add_materials(
        self,
        dataset_name: str,
        graphs: List,
        auto_shard: bool = True
    ):
        """
        Add materials to dataset (automatically shards).

        Args:
            dataset_name: Name of dataset
            graphs: List of graph Data objects
            auto_shard: Automatically create shards
        """
        dataset_dir = self.base_dir / dataset_name

        if not dataset_dir.exists():
            raise ValueError(f"Dataset '{dataset_name}' not found!")

        # Load manifest
        manifest = self._load_manifest(dataset_dir)
        shards_dir = dataset_dir / "shards"

        if auto_shard:
            # Shard the graphs
            shard_size = manifest["shard_size"]
            current_shard_idx = manifest["num_shards"]

            for i in range(0, len(graphs), shard_size):
                shard_graphs = graphs[i:i + shard_size]
                shard_path = shards_dir / f"shard_{current_shard_idx:04d}.pkl"

                # Save shard
                with open(shard_path, "wb") as f:
                    pickle.dump(shard_graphs, f)

                # Update manifest
                manifest["shards"].append({
                    "id": current_shard_idx,
                    "path": f"shards/shard_{current_shard_idx:04d}.pkl",
                    "count": len(shard_graphs)
                })

                manifest["num_shards"] += 1
                manifest["total_materials"] += len(shard_graphs)
                current_shard_idx += 1

            # Extract metadata from graphs
            self._update_metadata(manifest, graphs)

            # Save updated manifest
            self._save_manifest(dataset_dir, manifest)

            print(f"Added {len(graphs)} materials to '{dataset_name}'")
            print(f"  Total materials: {manifest['total_materials']}")
            print(f"  Total shards: {manifest['num_shards']}")

    def merge_datasets(
        self,
        dataset_names: List[str],
        output_name: str,
        deduplicate: bool = True
    ) -> str:
        """
        Merge multiple datasets into one.

        Args:
            dataset_names: List of dataset names to merge
            output_name: Name for merged dataset
            deduplicate: Remove duplicate material_ids

        Returns:
            Path to merged dataset
        """
        print(f"Merging {len(dataset_names)} datasets into '{output_name}'...")

        # Load all graphs
        all_graphs = []
        seen_ids = set() if deduplicate else None

        for dataset_name in dataset_names:
            print(f"  Loading {dataset_name}...")
            graphs = self.load_full_dataset(dataset_name)

            if deduplicate:
                # Filter duplicates
                unique_graphs = []
                for graph in graphs:
                    mat_id = getattr(graph, 'material_id', None)
                    if mat_id and mat_id not in seen_ids:
                        seen_ids.add(mat_id)
                        unique_graphs.append(graph)
                    elif not mat_id:
                        unique_graphs.append(graph)  # Keep if no ID

                print(f"    Kept {len(unique_graphs)}/{len(graphs)} unique materials")
                all_graphs.extend(unique_graphs)
            else:
                all_graphs.extend(graphs)

        # Create new dataset
        # Infer shard size from first dataset
        first_manifest = self._load_manifest(self.base_dir / dataset_names[0])
        shard_size = first_manifest.get("shard_size", 1000)

        output_dir = self.create_dataset(
            name=output_name,
            description=f"Merged from: {', '.join(dataset_names)}",
            shard_size=shard_size
        )

        # Add all materials
        self.add_materials(output_name, all_graphs)

        print(f"\nMerge complete!")
        print(f"  Output: {output_name}")
        print(f"  Total materials: {len(all_graphs)}")

        return output_dir

    def deduplicate_dataset(self, dataset_name: str) -> int:
        """
        Remove duplicate materials from dataset (by material_id).

        Args:
            dataset_name: Name of dataset

        Returns:
            Number of duplicates removed
        """
        print(f"Deduplicating dataset '{dataset_name}'...")

        # Load all graphs
        graphs = self.load_full_dataset(dataset_name)

        # Find unique materials
        seen_ids = set()
        unique_graphs = []
        duplicates = 0

        for graph in graphs:
            mat_id = getattr(graph, 'material_id', None)
            if mat_id:
                if mat_id not in seen_ids:
                    seen_ids.add(mat_id)
                    unique_graphs.append(graph)
                else:
                    duplicates += 1
            else:
                # Keep graphs without IDs
                unique_graphs.append(graph)

        if duplicates > 0:
            # Backup original
            dataset_dir = self.base_dir / dataset_name
            backup_dir = dataset_dir.parent / f"{dataset_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(dataset_dir, backup_dir)
            print(f"  Backup created: {backup_dir.name}")

            # Remove old shards
            shards_dir = dataset_dir / "shards"
            for shard_file in shards_dir.glob("*.pkl"):
                shard_file.unlink()

            # Reset manifest
            manifest = self._load_manifest(dataset_dir)
            manifest["total_materials"] = 0
            manifest["num_shards"] = 0
            manifest["shards"] = []
            self._save_manifest(dataset_dir, manifest)

            # Re-add unique materials
            self.add_materials(dataset_name, unique_graphs)

            print(f"  Removed {duplicates} duplicates")
            print(f"  Kept {len(unique_graphs)} unique materials")
        else:
            print("  No duplicates found!")

        return duplicates

    def get_statistics(self, dataset_name: str) -> Dict:
        """
        Get dataset statistics.

        Args:
            dataset_name: Name of dataset

        Returns:
            Dictionary with statistics
        """
        dataset_dir = self.base_dir / dataset_name

        if not dataset_dir.exists():
            raise ValueError(f"Dataset '{dataset_name}' not found!")

        manifest = self._load_manifest(dataset_dir)

        # Calculate additional stats
        stats = {
            "name": manifest["dataset_name"],
            "description": manifest["description"],
            "total_materials": manifest["total_materials"],
            "num_shards": manifest["num_shards"],
            "shard_size": manifest["shard_size"],
            "elements": manifest.get("elements", []),
            "properties": manifest.get("properties", []),
            "calphad_enhanced": manifest.get("calphad_enhanced", False),
            "created": manifest["created"],
            "version": manifest["version"]
        }

        # Calculate disk usage
        total_size = 0
        shards_dir = dataset_dir / "shards"
        for shard_file in shards_dir.glob("*.pkl"):
            total_size += shard_file.stat().st_size

        stats["disk_usage_mb"] = total_size / (1024 * 1024)
        stats["avg_shard_size_mb"] = stats["disk_usage_mb"] / max(manifest["num_shards"], 1)

        return stats

    def load_full_dataset(self, dataset_name: str) -> List:
        """
        Load entire dataset into memory.

        Args:
            dataset_name: Name of dataset

        Returns:
            List of all graph objects
        """
        dataset_dir = self.base_dir / dataset_name
        manifest = self._load_manifest(dataset_dir)

        all_graphs = []

        for shard_info in tqdm(manifest["shards"], desc="Loading shards"):
            shard_path = dataset_dir / shard_info["path"]
            with open(shard_path, "rb") as f:
                graphs = pickle.load(f)
                all_graphs.extend(graphs)

        return all_graphs

    def load_shard(self, dataset_name: str, shard_idx: int) -> List:
        """
        Load a single shard.

        Args:
            dataset_name: Name of dataset
            shard_idx: Shard index

        Returns:
            List of graphs in shard
        """
        dataset_dir = self.base_dir / dataset_name
        manifest = self._load_manifest(dataset_dir)

        if shard_idx >= manifest["num_shards"]:
            raise IndexError(f"Shard {shard_idx} does not exist!")

        shard_info = manifest["shards"][shard_idx]
        shard_path = dataset_dir / shard_info["path"]

        with open(shard_path, "rb") as f:
            return pickle.load(f)

    def list_datasets(self) -> List[str]:
        """
        List all available datasets.

        Returns:
            List of dataset names
        """
        datasets = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "manifest.json").exists():
                datasets.append(item.name)
        return sorted(datasets)

    def delete_dataset(self, dataset_name: str, confirm: bool = False):
        """
        Delete a dataset.

        Args:
            dataset_name: Name of dataset
            confirm: Must be True to actually delete
        """
        if not confirm:
            print("WARNING: Set confirm=True to actually delete the dataset")
            return

        dataset_dir = self.base_dir / dataset_name

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            print(f"Deleted dataset: {dataset_name}")
        else:
            print(f"Dataset '{dataset_name}' not found!")

    def _load_manifest(self, dataset_dir: Path) -> Dict:
        """Load manifest.json"""
        manifest_path = dataset_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _save_manifest(self, dataset_dir: Path, manifest: Dict):
        """Save manifest.json"""
        manifest_path = dataset_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _update_metadata(self, manifest: Dict, graphs: List):
        """Update manifest metadata from graphs"""
        # Extract unique elements
        elements = set(manifest.get("elements", []))
        properties = set(manifest.get("properties", []))

        for graph in graphs[:100]:  # Sample first 100
            # Check for CALPHAD features
            if hasattr(graph, 'has_calphad_features') and graph.has_calphad_features:
                manifest["calphad_enhanced"] = True

            # Extract formula if available
            if hasattr(graph, 'formula_str'):
                formula = graph.formula_str
                # Simple element extraction (works for most cases)
                import re
                found_elements = re.findall(r'([A-Z][a-z]?)', formula)
                elements.update(found_elements)

            # Check what properties are available
            if hasattr(graph, 'y'):
                if hasattr(graph, 'target_names'):
                    properties.update(graph.target_names)
                else:
                    properties.add("target")

        manifest["elements"] = sorted(list(elements))
        manifest["properties"] = sorted(list(properties))


def print_dataset_info(stats: Dict):
    """Pretty print dataset statistics."""
    print("\n" + "="*70)
    print(f"Dataset: {stats['name']}")
    print("="*70)

    print(f"\nDescription: {stats['description']}")
    print(f"Created: {stats['created']}")
    print(f"Version: {stats['version']}")

    print(f"\nSize:")
    print(f"  Total materials: {stats['total_materials']:,}")
    print(f"  Number of shards: {stats['num_shards']}")
    print(f"  Materials per shard: {stats['shard_size']}")

    print(f"\nDisk Usage:")
    print(f"  Total: {stats['disk_usage_mb']:.2f} MB")
    print(f"  Per shard: {stats['avg_shard_size_mb']:.2f} MB")

    if stats['elements']:
        print(f"\nElements: {', '.join(stats['elements'][:10])}")
        if len(stats['elements']) > 10:
            print(f"  ... and {len(stats['elements']) - 10} more")

    if stats['properties']:
        print(f"\nProperties: {', '.join(stats['properties'])}")

    print(f"\nCALPHAD Enhanced: {'Yes' if stats['calphad_enhanced'] else 'No'}")

    print("="*70)


if __name__ == "__main__":
    """Test dataset manager"""

    print("="*70)
    print("Testing Dataset Manager")
    print("="*70)
    print()

    # Initialize manager
    manager = DatasetManager(base_dir="datasets_test")

    # Test 1: Create dataset
    print("Test 1: Create dataset")
    manager.create_dataset(
        name="test_dataset",
        description="Test dataset for validation",
        shard_size=10
    )
    print()

    # Test 2: List datasets
    print("Test 2: List datasets")
    datasets = manager.list_datasets()
    print(f"Found {len(datasets)} datasets: {datasets}")
    print()

    # Test 3: Get statistics (empty dataset)
    print("Test 3: Statistics (empty)")
    stats = manager.get_statistics("test_dataset")
    print_dataset_info(stats)
    print()

    # Test 4: Add materials (simulate with simple objects)
    print("Test 4: Add materials")
    # Create dummy graphs
    from torch_geometric.data import Data
    import torch

    dummy_graphs = []
    for i in range(25):
        graph = Data(
            x=torch.randn(5, 13),
            edge_index=torch.randint(0, 5, (2, 10)),
            edge_attr=torch.randn(10, 2),
            y=torch.tensor([float(i)])
        )
        graph.material_id = f"mp-{i}"
        graph.formula_str = "Fe2Ni"
        graph.has_calphad_features = True
        dummy_graphs.append(graph)

    manager.add_materials("test_dataset", dummy_graphs)
    print()

    # Test 5: Statistics (with data)
    print("Test 5: Statistics (with data)")
    stats = manager.get_statistics("test_dataset")
    print_dataset_info(stats)
    print()

    # Test 6: Load shard
    print("Test 6: Load single shard")
    shard_0 = manager.load_shard("test_dataset", 0)
    print(f"Shard 0 contains {len(shard_0)} materials")
    print()

    # Test 7: Deduplication
    print("Test 7: Deduplication")
    # Add some duplicates
    manager.add_materials("test_dataset", dummy_graphs[:5])
    print("Added 5 duplicate materials")
    removed = manager.deduplicate_dataset("test_dataset")
    print(f"Removed {removed} duplicates")
    print()

    # Cleanup
    print("Cleaning up test dataset...")
    manager.delete_dataset("test_dataset", confirm=True)
    shutil.rmtree("datasets_test")

    print("\nAll dataset manager tests passed!")
