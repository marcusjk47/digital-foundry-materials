"""
Investigate the source of dataset duplication
"""
import json
from pathlib import Path
import pickle

print("="*70)
print("INVESTIGATING DATASET DUPLICATION SOURCE")
print("="*70)

# Check the manifest for clues
manifest_path = Path("datasets/extremely_large_dataset/extremely_large_dataset/manifest.json")

with open(manifest_path, 'r') as f:
    manifest = json.load(f)

print("\n1. MANIFEST INFORMATION")
print("-"*70)
print(f"Dataset name: {manifest['dataset_name']}")
print(f"Description: {manifest.get('description', 'N/A')}")
print(f"Created: {manifest.get('created', 'N/A')}")
print(f"Version: {manifest.get('version', 'N/A')}")
print(f"Total materials (claimed): {manifest['total_materials']}")
print(f"Number of shards: {manifest['num_shards']}")
print(f"Shard size: {manifest['shard_size']}")
print(f"CALPHAD enhanced: {manifest.get('calphad_enhanced', 'N/A')}")

# Check if description mentions merged datasets
if 'description' in manifest:
    print(f"\nDescription indicates: {manifest['description']}")

# Check for source datasets
print("\n2. CHECKING FOR SOURCE DATASETS")
print("-"*70)

datasets_dir = Path("datasets")
all_datasets = []

for item in datasets_dir.iterdir():
    if item.is_dir():
        all_datasets.append(item.name)
    elif item.suffix == '.pkl':
        all_datasets.append(item.name)

print(f"Found {len(all_datasets)} datasets/files in datasets/:")
for ds in sorted(all_datasets):
    print(f"  - {ds}")

# Check if the mentioned source datasets exist
description = manifest.get('description', '')
if 'Merged from:' in description:
    source_names = description.split('Merged from:')[1].strip().split(',')
    source_names = [s.strip() for s in source_names]

    print(f"\n3. CHECKING MENTIONED SOURCE DATASETS")
    print("-"*70)
    print(f"Description mentions these sources: {source_names}")

    for source_name in source_names:
        # Look for this dataset
        possible_paths = [
            datasets_dir / f"{source_name}.pkl",
            datasets_dir / source_name,
            datasets_dir / source_name / f"{source_name}.pkl"
        ]

        found = False
        for path in possible_paths:
            if path.exists():
                print(f"\n  Found: {source_name}")
                print(f"  Location: {path}")

                # If it's a pickle file, load and check it
                if path.suffix == '.pkl':
                    try:
                        with open(path, 'rb') as f:
                            data = pickle.load(f)

                        # Get unique material IDs
                        if hasattr(data[0], 'material_id'):
                            mat_ids = [s.material_id for s in data]
                            unique_ids = set(mat_ids)

                            print(f"  Total structures: {len(data)}")
                            print(f"  Unique material IDs: {len(unique_ids)}")
                            print(f"  Duplicates in this source: {len(data) - len(unique_ids)}")

                            # Show some material IDs
                            print(f"  Sample Material IDs: {list(unique_ids)[:5]}")
                    except Exception as e:
                        print(f"  Error loading: {e}")

                found = True
                break

        if not found:
            print(f"\n  NOT FOUND: {source_name}")

# Check if there are CSV files that might show original data collection
print("\n4. CHECKING FOR CSV FILES (ORIGINAL DATA)")
print("-"*70)

csv_files = list(datasets_dir.glob("*.csv"))
if csv_files:
    for csv_file in csv_files:
        print(f"  Found: {csv_file.name}")
        # Could analyze CSV to see unique materials
else:
    print("  No CSV files found in datasets/")

# Analyze the actual duplicates pattern
print("\n5. ANALYZING DUPLICATION PATTERN")
print("-"*70)

# Load all structures
all_structures = []
shards_dir = manifest_path.parent / "shards"

for shard_info in manifest['shards']:
    shard_path = manifest_path.parent / shard_info['path']
    with open(shard_path, 'rb') as f:
        shard_data = pickle.load(f)
    all_structures.extend(shard_data)

# Count occurrences of each material ID
from collections import Counter
mat_id_counts = Counter([s.material_id for s in all_structures])

print(f"Total structures: {len(all_structures)}")
print(f"Unique material IDs: {len(mat_id_counts)}")

print(f"\nDuplication pattern:")
for mat_id, count in mat_id_counts.most_common(14):
    percentage = count / len(all_structures) * 100
    print(f"  {mat_id}: {count} times ({percentage:.1f}%)")

# Check if duplicates are evenly distributed
counts_list = list(mat_id_counts.values())
avg_count = sum(counts_list) / len(counts_list)
min_count = min(counts_list)
max_count = max(counts_list)

print(f"\nDuplication statistics:")
print(f"  Average copies per material: {avg_count:.1f}")
print(f"  Minimum copies: {min_count}")
print(f"  Maximum copies: {max_count}")
print(f"  Range: {max_count - min_count}")

if max_count - min_count < 60:  # If range is small
    print(f"\n  Pattern: Materials are duplicated roughly EQUALLY (~{int(avg_count)} times each)")
    print(f"  This suggests: The original 14 materials were systematically repeated")
    print(f"                 to artificially inflate the dataset to ~3000 materials")
else:
    print(f"\n  Pattern: Materials have VARYING duplication counts")
    print(f"  This suggests: Multiple overlapping datasets were merged")

# Check shard-by-shard pattern
print("\n6. SHARD-BY-SHARD ANALYSIS")
print("-"*70)

for shard_info in manifest['shards']:
    shard_path = manifest_path.parent / shard_info['path']
    with open(shard_path, 'rb') as f:
        shard_data = pickle.load(f)

    shard_ids = [s.material_id for s in shard_data]
    unique_in_shard = len(set(shard_ids))

    print(f"\nShard {shard_info['id']}:")
    print(f"  Total structures: {len(shard_data)}")
    print(f"  Unique material IDs: {unique_in_shard}")
    print(f"  Unique material IDs in shard: {set(shard_ids)}")

    if unique_in_shard == len(set(mat_id_counts.keys())):
        print(f"  -> This shard contains ALL 14 unique materials")
        print(f"  -> Likely cause: Original 14 materials were tile-repeated to fill shard")

print("\n" + "="*70)
print("INVESTIGATION COMPLETE")
print("="*70)

print("\nCONCLUSION:")
print("-"*70)
print("Based on the analysis, the most likely cause is:")
print("")
print("  The dataset was created by taking a small set of 14 Co-Cr materials")
print("  and systematically repeating them ~200+ times each to artificially")
print("  reach the target size of ~3000 materials.")
print("")
print("  This could have happened due to:")
print("  - Bug in dataset merging code that repeated the same data")
print("  - Merging the same source dataset multiple times by mistake")
print("  - Data augmentation that duplicated instead of transforming")
print("")
print("RECOMMENDATION:")
print("  Collect fresh data using the multi-system collection feature")
print("  to get truly unique, diverse materials for training.")
