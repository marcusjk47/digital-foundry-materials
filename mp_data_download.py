"""
Materials Project Data Download Script

This script downloads alloy data from the Materials Project database
for use in machine learning models for alloy development.

Author: ML Alloy Development Project
Date: November 6, 2025
"""

import os
import time
import pandas as pd
import numpy as np
from mp_api.client import MPRester
from tqdm import tqdm
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class MaterialsProjectDownloader:
    """Download and process alloy data from Materials Project"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize downloader with API key

        Args:
            api_key: Materials Project API key (if None, uses environment variable)
        """
        self.api_key = api_key or os.environ.get('MP_API_KEY')
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either pass api_key or set MP_API_KEY environment variable.\n"
                "Get your API key from: https://next-gen.materialsproject.org/api"
            )

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            with MPRester(self.api_key) as mpr:
                doc = mpr.materials.summary.search(material_ids=["mp-13"])[0]
                print(f"✓ Connection successful! Test query returned: {doc.formula_pretty}")
                return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def download_alloy_system(
        self,
        chemsys: str,
        include_properties: List[str] = None,
        metallic_only: bool = True,
        stable_only: bool = False
    ) -> pd.DataFrame:
        """
        Download data for a specific chemical system

        Args:
            chemsys: Chemical system (e.g., "Fe-Ni" or "Fe-Cr-Ni")
            include_properties: List of properties to download
            metallic_only: If True, only download metallic materials (band_gap=0)
            stable_only: If True, only download stable materials

        Returns:
            DataFrame with material data
        """
        if include_properties is None:
            include_properties = [
                "material_id", "formula_pretty", "composition",
                "energy_per_atom", "formation_energy_per_atom",
                "energy_above_hull", "band_gap", "is_stable",
                "volume", "density", "nsites", "symmetry"
            ]

        print(f"\nDownloading data for {chemsys} system...")

        materials_data = []

        with MPRester(self.api_key) as mpr:
            # Build search criteria
            search_kwargs = {
                "chemsys": chemsys,
                "fields": include_properties
            }

            if metallic_only:
                search_kwargs["band_gap"] = (0, 0)

            if stable_only:
                search_kwargs["is_stable"] = True

            # Execute search
            docs = mpr.materials.summary.search(**search_kwargs)

            print(f"Found {len(docs)} materials in {chemsys}")

            # Process each material
            for doc in tqdm(docs, desc=f"Processing {chemsys}"):
                mat_data = self._extract_material_data(doc, chemsys)
                materials_data.append(mat_data)

        df = pd.DataFrame(materials_data)
        print(f"✓ Downloaded {len(df)} materials from {chemsys}")

        return df

    def download_multiple_systems(
        self,
        systems: List[str],
        output_file: str = "alloy_data.csv",
        **kwargs
    ) -> pd.DataFrame:
        """
        Download data for multiple chemical systems

        Args:
            systems: List of chemical systems
            output_file: Path to save combined data
            **kwargs: Additional arguments passed to download_alloy_system

        Returns:
            Combined DataFrame
        """
        all_data = []

        for system in systems:
            try:
                df = self.download_alloy_system(system, **kwargs)
                all_data.append(df)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Error downloading {system}: {e}")
                continue

        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['material_id'])

            # Save to file
            combined_df.to_csv(output_file, index=False)
            print(f"\n✓ Saved {len(combined_df)} unique materials to {output_file}")

            return combined_df
        else:
            print("No data downloaded")
            return pd.DataFrame()

    def download_with_elastic_properties(
        self,
        elements: List[str],
        output_file: str = "alloy_data_with_elasticity.csv"
    ) -> pd.DataFrame:
        """
        Download materials with elastic property data

        Args:
            elements: List of elements to include
            output_file: Path to save data

        Returns:
            DataFrame with elastic properties
        """
        print(f"\nDownloading materials with elastic data for elements: {elements}")

        all_data = []

        with MPRester(self.api_key) as mpr:
            # First get material IDs
            print("Getting material IDs...")
            summary_docs = mpr.materials.summary.search(
                elements=elements,
                num_elements=(2, len(elements)),
                band_gap=(0, 0),  # Metallic
                fields=["material_id", "formula_pretty", "composition",
                        "formation_energy_per_atom", "energy_above_hull"]
            )

            print(f"Found {len(summary_docs)} materials, checking for elastic data...")

            # Get elastic properties
            for doc in tqdm(summary_docs, desc="Querying elastic properties"):
                try:
                    elastic_docs = mpr.materials.elasticity.search(
                        material_ids=[doc.material_id]
                    )

                    if elastic_docs:
                        elastic_doc = elastic_docs[0]

                        mat_data = {
                            "material_id": doc.material_id,
                            "formula": doc.formula_pretty,
                            "formation_energy": doc.formation_energy_per_atom,
                            "energy_above_hull": doc.energy_above_hull,
                            "bulk_modulus": getattr(elastic_doc, 'bulk_modulus_vrh', None),
                            "shear_modulus": getattr(elastic_doc, 'shear_modulus_vrh', None),
                            "poisson_ratio": getattr(elastic_doc, 'poisson_ratio', None),
                        }

                        # Extract composition
                        if hasattr(doc, 'composition'):
                            for element in elements:
                                frac = doc.composition.get_atomic_fraction(element)
                                mat_data[f"frac_{element}"] = frac

                        all_data.append(mat_data)

                    time.sleep(0.1)  # Rate limiting

                except Exception as e:
                    continue

        df = pd.DataFrame(all_data)

        if not df.empty:
            df.to_csv(output_file, index=False)
            print(f"\n✓ Saved {len(df)} materials with elastic properties to {output_file}")
        else:
            print("No materials with elastic data found")

        return df

    def _extract_material_data(self, doc, system: str) -> Dict:
        """Extract data from a Materials Project document"""
        mat_data = {
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "system": system,
        }

        # Energy properties
        if hasattr(doc, 'energy_per_atom'):
            mat_data["energy_per_atom"] = doc.energy_per_atom
        if hasattr(doc, 'formation_energy_per_atom'):
            mat_data["formation_energy"] = doc.formation_energy_per_atom
        if hasattr(doc, 'energy_above_hull'):
            mat_data["energy_above_hull"] = doc.energy_above_hull

        # Electronic properties
        if hasattr(doc, 'band_gap'):
            mat_data["band_gap"] = doc.band_gap

        # Stability
        if hasattr(doc, 'is_stable'):
            mat_data["is_stable"] = doc.is_stable

        # Structure properties
        if hasattr(doc, 'volume'):
            mat_data["volume"] = doc.volume
        if hasattr(doc, 'density'):
            mat_data["density"] = doc.density
        if hasattr(doc, 'nsites'):
            mat_data["nsites"] = doc.nsites
            if doc.volume and doc.nsites:
                mat_data["volume_per_atom"] = doc.volume / doc.nsites

        # Symmetry
        if hasattr(doc, 'symmetry') and doc.symmetry:
            mat_data["space_group"] = doc.symmetry.number
            mat_data["crystal_system"] = doc.symmetry.crystal_system

        # Composition fractions
        if hasattr(doc, 'composition') and doc.composition:
            elements = [str(el) for el in doc.composition.elements]
            mat_data["num_elements"] = len(elements)

            # Store individual element fractions
            for element in doc.composition.elements:
                el_symbol = str(element)
                frac = doc.composition.get_atomic_fraction(element)
                mat_data[f"frac_{el_symbol}"] = frac

        return mat_data


# ============================================================================
# Example Usage Functions
# ============================================================================

def example_1_single_system():
    """Example: Download single alloy system"""
    print("=" * 60)
    print("Example 1: Download Fe-Ni System")
    print("=" * 60)

    downloader = MaterialsProjectDownloader()

    # Test connection
    if not downloader.test_connection():
        return

    # Download Fe-Ni system
    df = downloader.download_alloy_system(
        chemsys="Fe-Ni",
        metallic_only=True,
        stable_only=False
    )

    print("\nData preview:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Save
    df.to_csv("fe_ni_alloys.csv", index=False)
    print("\nSaved to fe_ni_alloys.csv")


def example_2_multiple_systems():
    """Example: Download multiple alloy systems"""
    print("=" * 60)
    print("Example 2: Download Multiple Alloy Systems")
    print("=" * 60)

    downloader = MaterialsProjectDownloader()

    # Define alloy systems of interest
    alloy_systems = [
        "Fe-Ni",         # Iron-nickel
        "Fe-Cr",         # Iron-chromium
        "Fe-Cr-Ni",      # Stainless steel
        "Ti-Al",         # Titanium aluminide
        "Ni-Al",         # Nickel aluminide
        "Al-Cu",         # Aluminum-copper
        "Co-Cr",         # Cobalt-chromium
    ]

    # Download all systems
    df = downloader.download_multiple_systems(
        systems=alloy_systems,
        output_file="multi_system_alloys.csv",
        metallic_only=True,
        stable_only=False
    )

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(df.groupby('system').size())


def example_3_with_elastic_properties():
    """Example: Download materials with mechanical properties"""
    print("=" * 60)
    print("Example 3: Download with Elastic Properties")
    print("=" * 60)

    downloader = MaterialsProjectDownloader()

    # Download Fe-Ni-Cr alloys with elastic data
    elements = ["Fe", "Ni", "Cr", "Co"]

    df = downloader.download_with_elastic_properties(
        elements=elements,
        output_file="alloys_with_elasticity.csv"
    )

    if not df.empty:
        print("\nData preview:")
        print(df.head())

        print("\nElastic property statistics:")
        print(df[['bulk_modulus', 'shear_modulus', 'poisson_ratio']].describe())


def example_4_custom_download():
    """Example: Custom download with specific criteria"""
    print("=" * 60)
    print("Example 4: Custom Download")
    print("=" * 60)

    api_key = os.environ.get('MP_API_KEY')
    if not api_key:
        print("Please set MP_API_KEY environment variable")
        return

    all_materials = []

    with MPRester(api_key) as mpr:
        # Custom query: High-entropy alloy elements
        hea_elements = ["Co", "Cr", "Fe", "Mn", "Ni"]

        print(f"Searching for alloys containing {hea_elements}...")

        docs = mpr.materials.summary.search(
            elements=hea_elements,
            num_elements=(3, 5),  # Ternary to quinary
            band_gap=(0, 0),  # Metallic
            energy_above_hull=(0, 0.2),  # Likely synthesizable
            fields=["material_id", "formula_pretty", "composition",
                    "formation_energy_per_atom", "energy_above_hull",
                    "volume", "density"]
        )

        print(f"Found {len(docs)} materials")

        for doc in docs:
            mat_data = {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "formation_energy": doc.formation_energy_per_atom,
                "energy_above_hull": doc.energy_above_hull,
                "density": doc.density,
            }

            # Extract composition
            for element in hea_elements:
                frac = doc.composition.get_atomic_fraction(element)
                mat_data[f"frac_{element}"] = frac

            all_materials.append(mat_data)

    df = pd.DataFrame(all_materials)
    df.to_csv("high_entropy_alloys.csv", index=False)

    print("\nData preview:")
    print(df.head())
    print(f"\nSaved {len(df)} materials to high_entropy_alloys.csv")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("Materials Project Data Download Script")
    print("=" * 60)

    # Check for API key
    if not os.environ.get('MP_API_KEY'):
        print("\n⚠ Warning: MP_API_KEY environment variable not set!")
        print("\nTo set your API key:")
        print("  Windows (PowerShell): $env:MP_API_KEY='your_key_here'")
        print("  Windows (CMD):        set MP_API_KEY=your_key_here")
        print("  Linux/Mac:            export MP_API_KEY='your_key_here'")
        print("\nGet your key from: https://next-gen.materialsproject.org/api")
        print("\n" + "=" * 60)
        exit(1)

    # Run examples
    print("\nChoose an example to run:")
    print("1. Download single system (Fe-Ni)")
    print("2. Download multiple systems")
    print("3. Download with elastic properties")
    print("4. Custom download (High-entropy alloys)")
    print("0. Run all examples")

    choice = input("\nEnter choice (0-4): ").strip()

    if choice == "1":
        example_1_single_system()
    elif choice == "2":
        example_2_multiple_systems()
    elif choice == "3":
        example_3_with_elastic_properties()
    elif choice == "4":
        example_4_custom_download()
    elif choice == "0":
        example_1_single_system()
        print("\n" + "=" * 60 + "\n")
        example_2_multiple_systems()
        print("\n" + "=" * 60 + "\n")
        example_3_with_elastic_properties()
        print("\n" + "=" * 60 + "\n")
        example_4_custom_download()
    else:
        print("Invalid choice")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
