"""
Deduplicate sharded dataset by removing duplicate Materials Project IDs
"""
import pickle
import json
from pathlib import Path
from collections import OrderedDict

print("="*70)
print("DATASET DEDUPLICATION")
print("="*70)

# Target the extremely_large_dataset
manifest_path = Path("datasets/extremely_large_dataset/extremely_large_dataset/manifest.json")

if not manifest_path.exists():
    print(f"Error: Manifest not found at {manifest_path}")
    exit(1)

# Load manifest
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

print(f"\nOriginal dataset: {manifest['dataset_name']}")
print(f"Original total materials: {manifest['total_materials']}")
print(f"Original shards: {manifest['num_shards']}")

# Load all structures from all shards
print("\nLoading all structures...")
all_structures = []
shards_dir = manifest_path.parent / "shards"

for shard_info in manifest['shards']:
    shard_path = manifest_path.parent / shard_info['path']

    with open(shard_path, 'rb') as f:
        shard_data = pickle.load(f)

    all_structures.extend(shard_data)
    print(f"  Loaded shard {shard_info['id']}: {len(shard_data)} structures")

print(f"\nTotal structures loaded: {len(all_structures)}")

# Deduplicate by material_id
print("\nDeduplicating by material_id...")
seen_ids = OrderedDict()
duplicates_removed = 0

for structure in all_structures:
    mat_id = structure.material_id

    if mat_id not in seen_ids:
        seen_ids[mat_id] = structure
    else:
        duplicates_removed += 1

unique_structures = list(seen_ids.values())

print(f"\nDeduplication results:")
print(f"  Original structures: {len(all_structures)}")
print(f"  Unique structures: {len(unique_structures)}")
print(f"  Duplicates removed: {duplicates_removed}")
print(f"  Reduction: {duplicates_removed/len(all_structures)*100:.1f}%")

# Save deduplicated dataset
output_dir = Path("datasets/co_cr_deduplicated")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "co_cr_deduplicated.pkl"

print(f"\nSaving deduplicated dataset to: {output_path}")

with open(output_path, 'wb') as f:
    pickle.dump(unique_structures, f)

file_size = output_path.stat().st_size / (1024 * 1024)
print(f"Saved {len(unique_structures)} unique structures")
print(f"File size: {file_size:.2f} MB")

# Show unique material IDs
print(f"\nUnique Material IDs in deduplicated dataset:")
for i, mat_id in enumerate(seen_ids.keys(), 1):
    print(f"  {i}. {mat_id}")

# Create metadata
print("\nCreating metadata...")
metadata = {
    "dataset_name": "co_cr_deduplicated",
    "description": "Deduplicated Co-Cr dataset (removed duplicates from extremely_large_dataset)",
    "original_dataset": "extremely_large_dataset",
    "original_size": len(all_structures),
    "deduplicated_size": len(unique_structures),
    "duplicates_removed": duplicates_removed,
    "material_ids": list(seen_ids.keys())
}

metadata_path = output_dir / "metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved to: {metadata_path}")

print("\n" + "="*70)
print("DEDUPLICATION COMPLETE")
print("="*70)
print(f"\nDeduplicated dataset location: {output_path}")
print(f"You can now train on this dataset with {len(unique_structures)} unique materials")
