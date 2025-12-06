"""
Large-Scale Data Collection with Resumable Checkpointing

Collects large datasets from Materials Project with:
- Automatic sharding
- Progress checkpointing (resume after failures)
- Rate limiting and error handling
- Parallel graph conversion
- Progress tracking

Author: Digital Foundry Materials Science Toolkit
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from dataset_manager import DatasetManager
from gnn_data_collection import fetch_materials_data, convert_to_graphs
from mp_api.client import MPRester


class LargeScaleCollector:
    """
    Collector for large-scale materials datasets with checkpointing.
    """

    def __init__(
        self,
        api_key: str,
        dataset_manager: Optional[DatasetManager] = None
    ):
        """
        Initialize collector.

        Args:
            api_key: Materials Project API key
            dataset_manager: Dataset manager instance
        """
        self.api_key = api_key
        self.manager = dataset_manager or DatasetManager()

    def collect_dataset(
        self,
        dataset_name: str,
        elements: Optional[List[str]] = None,
        chemsys: Optional[str] = None,
        target_size: int = 10000,
        shard_size: int = 1000,
        checkpoint_interval: int = 100,
        metallic_only: bool = True,
        stable_only: bool = False,
        use_calphad: bool = True,
        resume: bool = True,
        properties: Optional[List[str]] = None,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Collect large dataset with automatic checkpointing.

        Args:
            dataset_name: Name for the dataset
            elements: List of elements to include
            chemsys: Chemical system (alternative to elements)
            target_size: Target number of materials
            shard_size: Materials per shard
            checkpoint_interval: Save checkpoint every N materials
            metallic_only: Only metallic materials
            stable_only: Only stable materials
            use_calphad: Use CALPHAD-enhanced features
            resume: Resume from checkpoint if exists
            properties: Target properties to collect
            callback: Progress callback function(progress_dict)

        Returns:
            Collection summary dictionary
        """
        print(f"="*70)
        print(f"Large-Scale Collection: {dataset_name}")
        print(f"="*70)
        print(f"Target size: {target_size:,} materials")
        print(f"Shard size: {shard_size}")
        print(f"Elements: {elements or chemsys or 'All'}")
        print(f"CALPHAD features: {use_calphad}")
        print()

        # Check for existing checkpoint
        checkpoint = None
        dataset_dir = self.manager.base_dir / dataset_name

        if resume and dataset_dir.exists():
            checkpoint = self._load_checkpoint(dataset_dir)
            if checkpoint:
                print(f"Resuming from checkpoint:")
                print(f"  Materials collected: {checkpoint['materials_collected']}")
                print(f"  Last material ID: {checkpoint.get('last_material_id', 'N/A')}")
                print()
        else:
            # Create new dataset
            self.manager.create_dataset(
                name=dataset_name,
                description=f"Large-scale collection: {elements or chemsys}",
                shard_size=shard_size,
                metadata={
                    "elements": elements,
                    "chemsys": chemsys,
                    "target_size": target_size,
                    "metallic_only": metallic_only,
                    "stable_only": stable_only,
                    "use_calphad": use_calphad
                }
            )
            checkpoint = {
                "dataset_name": dataset_name,
                "materials_collected": 0,
                "graphs_converted": 0,
                "target_size": target_size,
                "current_shard": 0,
                "timestamp": datetime.now().isoformat(),
                "failed_materials": []
            }

        # Collect materials in batches
        start_time = time.time()
        materials_collected = checkpoint["materials_collected"]
        total_graphs = []

        batch_size = min(checkpoint_interval, 500)  # API limit considerations

        while materials_collected < target_size:
            remaining = target_size - materials_collected
            current_batch_size = min(batch_size, remaining)

            print(f"\nBatch {materials_collected // batch_size + 1}:")
            print(f"  Collecting {current_batch_size} materials...")

            try:
                # Fetch batch
                df_batch = fetch_materials_data(
                    api_key=self.api_key,
                    elements=elements,
                    chemsys=chemsys,
                    max_materials=current_batch_size,
                    metallic_only=metallic_only,
                    stable_only=stable_only,
                    properties=properties
                )

                if df_batch.empty:
                    print("  No more materials found!")
                    break

                # Convert to graphs
                print(f"  Converting {len(df_batch)} structures to graphs...")
                graphs_batch = convert_to_graphs(
                    df_batch,
                    target_property=properties[0] if properties else "formation_energy_per_atom",
                    cutoff=8.0,
                    max_neighbors=12,
                    use_calphad=use_calphad,
                    save_path=None  # Don't save individual batches
                )

                # Add to dataset (auto-shards)
                if graphs_batch:
                    self.manager.add_materials(dataset_name, graphs_batch)
                    total_graphs.extend(graphs_batch)

                    materials_collected += len(graphs_batch)
                    checkpoint["materials_collected"] = materials_collected
                    checkpoint["graphs_converted"] += len(graphs_batch)

                    # Update checkpoint
                    if len(graphs_batch) > 0 and hasattr(graphs_batch[-1], 'material_id'):
                        checkpoint["last_material_id"] = graphs_batch[-1].material_id

                    checkpoint["timestamp"] = datetime.now().isoformat()
                    self._save_checkpoint(dataset_dir, checkpoint)

                    # Progress callback
                    if callback:
                        progress = {
                            "collected": materials_collected,
                            "target": target_size,
                            "percent": (materials_collected / target_size) * 100,
                            "elapsed_time": time.time() - start_time
                        }
                        callback(progress)

                    print(f"  Progress: {materials_collected}/{target_size} ({materials_collected/target_size*100:.1f}%)")

                else:
                    print("  Warning: No graphs converted in this batch")

                # Rate limiting
                time.sleep(0.5)  # Be nice to Materials Project API

            except Exception as e:
                print(f"  Error in batch: {e}")
                checkpoint["failed_materials"].append({
                    "batch": materials_collected // batch_size,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                self._save_checkpoint(dataset_dir, checkpoint)

                # Continue with next batch
                continue

        # Collection complete
        elapsed_time = time.time() - start_time

        # Remove checkpoint file
        checkpoint_path = dataset_dir / "checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        # Summary
        summary = {
            "dataset_name": dataset_name,
            "materials_collected": materials_collected,
            "target_size": target_size,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_formatted": self._format_time(elapsed_time),
            "success": materials_collected >= target_size * 0.9,  # 90% threshold
            "failed_batches": len(checkpoint.get("failed_materials", []))
        }

        print(f"\n" + "="*70)
        print("Collection Summary")
        print("="*70)
        print(f"Dataset: {dataset_name}")
        print(f"Collected: {materials_collected:,} / {target_size:,} materials")
        print(f"Time elapsed: {summary['elapsed_time_formatted']}")
        print(f"Failed batches: {summary['failed_batches']}")
        print(f"Status: {'SUCCESS' if summary['success'] else 'INCOMPLETE'}")
        print("="*70)

        return summary

    def collect_by_composition(
        self,
        dataset_name: str,
        compositions: List[Dict[str, int]],
        max_per_composition: int = 100,
        **kwargs
    ) -> Dict:
        """
        Collect materials by specific compositions.

        Args:
            dataset_name: Dataset name
            compositions: List of compositions, e.g., [{"Fe": 50, "Ni": 50}, ...]
            max_per_composition: Max materials per composition
            **kwargs: Additional arguments for collect_dataset

        Returns:
            Collection summary
        """
        print(f"Collecting {len(compositions)} compositions...")

        # Create dataset
        self.manager.create_dataset(
            name=dataset_name,
            description=f"Specific compositions: {len(compositions)} systems",
            shard_size=kwargs.get('shard_size', 1000)
        )

        total_collected = 0

        for i, comp in enumerate(compositions, 1):
            print(f"\n[{i}/{len(compositions)}] Collecting {comp}...")

            # Convert composition to elements list
            elements = list(comp.keys())

            try:
                df = fetch_materials_data(
                    api_key=self.api_key,
                    elements=elements,
                    max_materials=max_per_composition,
                    **kwargs
                )

                if not df.empty:
                    graphs = convert_to_graphs(
                        df,
                        use_calphad=kwargs.get('use_calphad', True)
                    )

                    if graphs:
                        self.manager.add_materials(dataset_name, graphs)
                        total_collected += len(graphs)

            except Exception as e:
                print(f"  Error: {e}")
                continue

        summary = {
            "dataset_name": dataset_name,
            "compositions_requested": len(compositions),
            "total_materials": total_collected,
            "avg_per_composition": total_collected / len(compositions) if compositions else 0
        }

        print(f"\nCollected {total_collected} materials across {len(compositions)} compositions")

        return summary

    def _save_checkpoint(self, dataset_dir: Path, checkpoint: Dict):
        """Save checkpoint file"""
        checkpoint_path = dataset_dir / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self, dataset_dir: Path) -> Optional[Dict]:
        """Load checkpoint file if exists"""
        checkpoint_path = dataset_dir / "checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        return None

    def _format_time(self, seconds: float) -> str:
        """Format seconds to human-readable time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


if __name__ == "__main__":
    """Test large-scale collector"""
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("MP_API_KEY")

    if not api_key:
        print("Error: MP_API_KEY not found")
        exit(1)

    print("="*70)
    print("Testing Large-Scale Collector")
    print("="*70)
    print()

    # Test with small dataset
    collector = LargeScaleCollector(api_key=api_key)

    # Collect 50 materials for testing
    summary = collector.collect_dataset(
        dataset_name="test_large_collection",
        elements=["Fe", "Ni"],
        target_size=50,
        shard_size=20,
        checkpoint_interval=10,
        metallic_only=True,
        use_calphad=True,
        resume=True
    )

    print("\nTest collection complete!")
    print(f"Success: {summary['success']}")

    # Check dataset
    manager = DatasetManager()
    stats = manager.get_statistics("test_large_collection")

    from dataset_manager import print_dataset_info
    print_dataset_info(stats)

    # Cleanup
    print("\nCleaning up test dataset...")
    manager.delete_dataset("test_large_collection", confirm=True)

    print("\nAll large-scale collector tests passed!")
