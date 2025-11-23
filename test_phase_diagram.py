"""
Diagnostic script to test phase diagram calculation
This will help identify the issue with empty phase diagrams
"""

import sys
import matplotlib.pyplot as plt

print("=" * 60)
print("Phase Diagram Diagnostic Test")
print("=" * 60)
print()

# Test 1: Import PyCalphad
print("1. Testing PyCalphad import...")
try:
    from pycalphad import Database, equilibrium, binplot
    import pycalphad.variables as v
    print("   [OK] PyCalphad imported successfully")
except ImportError as e:
    print(f"   [FAIL] Failed to import PyCalphad: {e}")
    sys.exit(1)

# Test 2: Load sample database
print("\n2. Loading sample Al-Zn database...")
try:
    db = Database('calphad_databases/alzn_sample.tdb')
    print("   [OK] Database loaded")
    print(f"   Elements: {[el for el in db.elements if el != 'VA']}")
    print(f"   Phases: {list(db.phases.keys())}")
except Exception as e:
    print(f"   [FAIL] Failed to load database: {e}")
    sys.exit(1)

# Test 3: Set up calculation
print("\n3. Setting up binary phase diagram calculation...")
try:
    comps = ['AL', 'ZN', 'VA']
    phases = list(db.phases.keys())

    # Conditions for binary phase diagram
    conditions = {
        v.N: 1,
        v.P: 101325,
        v.T: (300, 1000, 10),
        v.X('ZN'): (0, 1, 0.02)
    }

    print(f"   Components: {comps}")
    print(f"   Phases: {phases}")
    print(f"   Temperature range: 300-1000 K, step 10 K")
    print(f"   Composition range: 0-1 X(ZN), step 0.02")
    print("   [OK] Setup complete")
except Exception as e:
    print(f"   [FAIL] Setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Calculate phase diagram using binplot
print("\n4. Calculating phase diagram with binplot...")
print("   (This may take 30-60 seconds...)")
try:
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # This is the key function - binplot should create the phase diagram
    binplot(db, comps, phases, conditions, ax=ax)

    # Check if anything was plotted
    if len(ax.lines) == 0 and len(ax.collections) == 0:
        print("   [WARNING] Plot appears to be empty!")
        print("   No lines or collections were added to the axes")
    else:
        print(f"   [OK] Plot created with {len(ax.lines)} lines and {len(ax.collections)} collections")

    # Customize plot
    plt.title('Al-Zn Binary Phase Diagram (Diagnostic Test)', fontsize=14)
    plt.ylabel('Temperature (K)', fontsize=12)
    plt.xlabel('Mole Fraction Zn', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_file = 'diagnostic_phase_diagram.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   [OK] Figure saved as: {output_file}")

    # Try to display (if running interactively)
    plt.show()

except Exception as e:
    print(f"   [FAIL] Calculation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Alternative method - manual equilibrium calculation
print("\n5. Testing alternative method (equilibrium at single point)...")
try:
    # Calculate equilibrium at a single point
    test_conditions = {
        v.T: 600,
        v.P: 101325,
        v.N: 1,
        v.X('ZN'): 0.3
    }

    result = equilibrium(db, comps, phases, test_conditions)

    # Check stable phases
    stable_phases = result.Phase.values.squeeze()
    phase_fractions = result.NP.values.squeeze()

    print(f"   [OK] Equilibrium calculated at T=600K, X(ZN)=0.3")
    print(f"   Stable phases: {stable_phases}")
    print(f"   Phase fractions: {phase_fractions}")

except Exception as e:
    print(f"   [FAIL] Equilibrium calculation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnostic test complete!")
print("=" * 60)
print()
print("Next steps:")
print("1. Check the generated file: diagnostic_phase_diagram.png")
print("2. If the diagram is empty, the issue is with PyCalphad/database")
print("3. If the diagram looks good, the issue is with the Streamlit app")
print()
