"""
Check for duplicate Materials Project IDs in sharded datasets
"""
import pickle
import json
from pathlib import Path
from collections import Counter

# Find sharded dataset directories
datasets_dir = Path("datasets")

print("="*70)
print("CHECKING SHARDED DATASETS FOR DUPLICATE MATERIALS PROJECT IDs")
print("="*70)

# Look for directories with manifest.json
for dataset_dir in datasets_dir.iterdir():
    if dataset_dir.is_dir():
        # Check for manifest.json in the directory or subdirectories
        manifest_files = list(dataset_dir.glob("**/manifest.json"))

        for manifest_path in manifest_files:
            print(f"\nDataset: {dataset_dir.name}")
            print(f"Manifest: {manifest_path}")
            print("-"*70)

            try:
                # Load manifest
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)

                print(f"Total materials (from manifest): {manifest.get('total_materials', 'unknown')}")
                print(f"Number of shards: {manifest.get('num_shards', 'unknown')}")
                print(f"Shard size: {manifest.get('shard_size', 'unknown')}")

                # Collect all material IDs from all shards
                all_mp_ids = []
                shards_dir = manifest_path.parent / "shards"

                if not shards_dir.exists():
                    print("[WARNING] Shards directory not found")
                    continue

                print(f"\nLoading shards from: {shards_dir}")

                for shard_info in manifest['shards']:
                    shard_path = manifest_path.parent / shard_info['path']

                    if not shard_path.exists():
                        print(f"[WARNING] Shard not found: {shard_path}")
                        continue

                    try:
                        with open(shard_path, 'rb') as f:
                            shard_data = pickle.load(f)

                        # Extract material IDs from this shard
                        for structure in shard_data:
                            if hasattr(structure, 'material_id'):
                                all_mp_ids.append(structure.material_id)

                        print(f"  [OK] Loaded shard {shard_info['id']}: {len(shard_data)} structures")

                    except Exception as e:
                        print(f"  [ERROR] Error loading shard {shard_info['id']}: {e}")

                # Analyze duplicates
                if all_mp_ids:
                    unique_ids = set(all_mp_ids)
                    num_duplicates = len(all_mp_ids) - len(unique_ids)

                    print(f"\n{'='*70}")
                    print(f"RESULTS")
                    print(f"{'='*70}")
                    print(f"Total structures loaded: {len(all_mp_ids)}")
                    print(f"Unique structures: {len(unique_ids)}")
                    print(f"Duplicates: {num_duplicates}")

                    if num_duplicates > 0:
                        print(f"\n[WARNING] DUPLICATES FOUND")
                        print("-"*70)

                        # Count occurrences
                        id_counts = Counter(all_mp_ids)
                        duplicates = {k: v for k, v in id_counts.items() if v > 1}

                        print(f"Number of duplicate IDs: {len(duplicates)}")
                        print(f"\nTop 20 most duplicated Material IDs:")
                        for mp_id, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:20]:
                            print(f"  {mp_id}: appears {count} times")

                        if len(duplicates) > 20:
                            print(f"  ... and {len(duplicates) - 20} more duplicates")
                    else:
                        print(f"\n[OK] No duplicates found! All material IDs are unique.")
                else:
                    print("\n[WARNING] No material IDs found in dataset")

            except Exception as e:
                print(f"Error processing dataset: {e}")
                import traceback
                traceback.print_exc()

print("\n" + "="*70)
print("DUPLICATE CHECK COMPLETE")
print("="*70)
