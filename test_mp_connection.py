"""
Test Materials Project API Connection
Run this after setting your MP_API_KEY environment variable
"""

import os
from mp_api.client import MPRester

def test_connection():
    """Test connection to Materials Project API"""

    # Get API key from environment
    api_key = os.environ.get('MP_API_KEY')

    if not api_key:
        print("\n" + "=" * 60)
        print("ERROR: MP_API_KEY not set!")
        print("=" * 60)
        print("\nTo set your API key in PowerShell:")
        print('  $env:MP_API_KEY="your_api_key_here"')
        print("\nTo set your API key in Command Prompt:")
        print('  set MP_API_KEY=your_api_key_here')
        print("\nGet your API key from:")
        print("  https://next-gen.materialsproject.org/api")
        print("=" * 60)
        return False

    print("\n" + "=" * 60)
    print("Testing Materials Project API Connection...")
    print("=" * 60)

    try:
        with MPRester(api_key) as mpr:
            # Query a known material (BCC Iron, mp-13)
            print("\nQuerying BCC Iron (mp-13)...")

            doc = mpr.materials.summary.search(
                material_ids=["mp-13"],
                fields=["material_id", "formula_pretty",
                       "formation_energy_per_atom", "density"]
            )[0]

            print("\n" + "=" * 60)
            print("SUCCESS! Connected to Materials Project")
            print("=" * 60)
            print(f"\nMaterial ID:       {doc.material_id}")
            print(f"Formula:           {doc.formula_pretty}")
            print(f"Formation Energy:  {doc.formation_energy_per_atom} eV/atom")
            print(f"Density:           {doc.density} g/cm³")
            print("\n" + "=" * 60)
            print("Your environment is ready for alloy development!")
            print("=" * 60 + "\n")

            return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Connection failed")
        print("=" * 60)
        print(f"\nError message: {e}")
        print("\nPlease check:")
        print("  1. Your API key is correct")
        print("  2. You have internet connection")
        print("  3. Materials Project API is accessible")
        print("=" * 60 + "\n")
        return False


if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)
