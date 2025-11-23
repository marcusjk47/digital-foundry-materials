"""
Download First Alloy Dataset
Quick script to download Fe-Ni alloy data from Materials Project
"""

import os
from mp_api.client import MPRester
import pandas as pd
from tqdm import tqdm

# Load API key from environment
api_key = os.environ.get('MP_API_KEY')

if not api_key:
    # Try loading from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get('MP_API_KEY')
    except ImportError:
        print("Warning: python-dotenv not installed. Install it with: pip install python-dotenv")

if not api_key:
    print("\nError: MP_API_KEY not found!")
    print("Make sure .env file exists or set MP_API_KEY environment variable")
    exit(1)

print("=" * 70)
print("  Downloading Fe-Ni Alloy Data from Materials Project")
print("=" * 70)

all_data = []

with MPRester(api_key) as mpr:
    print("\nSearching for Fe-Ni alloys...")

    # Search for Fe-Ni alloys
    docs = mpr.materials.summary.search(
        chemsys="Fe-Ni",           # Iron-Nickel system
        band_gap=(0, 0),           # Metallic materials only
        fields=[
            "material_id",
            "formula_pretty",
            "composition",
            "formation_energy_per_atom",
            "energy_above_hull",
            "density",
            "volume",
            "is_stable",
            "nsites",
            "symmetry"
        ]
    )

    print(f"Found {len(docs)} Fe-Ni materials")

    # Extract data
    print("\nProcessing materials...")
    for doc in tqdm(docs, desc="Extracting data"):
        # Get composition fractions
        fe_frac = doc.composition.get_atomic_fraction("Fe")
        ni_frac = doc.composition.get_atomic_fraction("Ni")

        material_data = {
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "fe_fraction": fe_frac,
            "ni_fraction": ni_frac,
            "formation_energy": doc.formation_energy_per_atom,
            "energy_above_hull": doc.energy_above_hull,
            "density": doc.density,
            "volume": doc.volume,
            "volume_per_atom": doc.volume / doc.nsites if doc.volume and doc.nsites else None,
            "is_stable": doc.is_stable,
            "num_sites": doc.nsites,
        }

        # Add symmetry info if available
        if hasattr(doc, 'symmetry') and doc.symmetry:
            material_data["space_group"] = doc.symmetry.number
            material_data["crystal_system"] = doc.symmetry.crystal_system

        all_data.append(material_data)

# Create DataFrame
df = pd.DataFrame(all_data)

# Save to CSV
output_file = "fe_ni_alloys.csv"
df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("  Download Complete!")
print("=" * 70)
print(f"\nTotal materials: {len(df)}")
print(f"Saved to: {output_file}")

# Show preview
print("\n" + "=" * 70)
print("  Data Preview")
print("=" * 70)
print(df.head(10).to_string())

# Summary statistics
print("\n" + "=" * 70)
print("  Summary Statistics")
print("=" * 70)
print(f"\nComposition ranges:")
print(f"  Fe fraction: {df['fe_fraction'].min():.3f} - {df['fe_fraction'].max():.3f}")
print(f"  Ni fraction: {df['ni_fraction'].min():.3f} - {df['ni_fraction'].max():.3f}")

print(f"\nFormation energy:")
print(f"  Min: {df['formation_energy'].min():.3f} eV/atom")
print(f"  Max: {df['formation_energy'].max():.3f} eV/atom")
print(f"  Mean: {df['formation_energy'].mean():.3f} eV/atom")

print(f"\nDensity:")
print(f"  Min: {df['density'].min():.3f} g/cm³")
print(f"  Max: {df['density'].max():.3f} g/cm³")
print(f"  Mean: {df['density'].mean():.3f} g/cm³")

stable_count = df['is_stable'].sum()
near_stable_count = (df['energy_above_hull'] < 0.1).sum()

print(f"\nStability:")
print(f"  Stable materials: {stable_count} ({stable_count/len(df)*100:.1f}%)")
print(f"  Near-stable (E_hull < 0.1 eV): {near_stable_count} ({near_stable_count/len(df)*100:.1f}%)")

print("\n" + "=" * 70)
print("  Next Steps")
print("=" * 70)
print("\n1. Explore the data: Open fe_ni_alloys.csv in Excel or a text editor")
print("2. Visualize: Run visualize_data.py (if available)")
print("3. Download more systems: Run mp_data_download.py")
print("4. Start Phase 1: Data preparation and feature engineering")
print("\n" + "=" * 70)
