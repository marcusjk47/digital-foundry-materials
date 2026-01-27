"""
Check for duplicate Materials Project IDs in datasets
"""
import pickle
from pathlib import Path

# Find all .pkl dataset files
datasets_dir = Path("datasets")
pkl_files = list(datasets_dir.glob("*.pkl"))

print("="*70)
print("CHECKING FOR DUPLICATE MATERIALS PROJECT IDs")
print("="*70)

if not pkl_files:
    print("\nNo .pkl dataset files found in datasets/")
    print("Available files:")
    for f in datasets_dir.iterdir():
        if f.is_file():
            print(f"  - {f.name}")
else:
    for pkl_file in pkl_files:
        print(f"\nDataset: {pkl_file.name}")
        print("-"*70)

        try:
            with open(pkl_file, 'rb') as f:
                dataset = pickle.load(f)

            # Check if dataset has material_id attribute
            if hasattr(dataset[0], 'material_id'):
                mp_ids = [structure.material_id for structure in dataset]
                unique_ids = set(mp_ids)

                print(f"Total structures: {len(mp_ids)}")
                print(f"Unique structures: {len(unique_ids)}")
                print(f"Duplicates: {len(mp_ids) - len(unique_ids)}")

                # Show duplicate IDs if any
                if len(mp_ids) != len(unique_ids):
                    print("\nDuplicate Material IDs:")
                    id_counts = {}
                    for mp_id in mp_ids:
                        id_counts[mp_id] = id_counts.get(mp_id, 0) + 1

                    duplicates = {k: v for k, v in id_counts.items() if v > 1}
                    for mp_id, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
                        print(f"  {mp_id}: appears {count} times")

                    if len(duplicates) > 10:
                        print(f"  ... and {len(duplicates) - 10} more duplicates")
            else:
                print("Dataset structures don't have 'material_id' attribute")
                print(f"Available attributes: {dir(dataset[0])}")

        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")

print("\n" + "="*70)
