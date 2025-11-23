"""Simple PyCalphad test - equilibrium only"""

print("Testing PyCalphad basic functionality...")

try:
    from pycalphad import Database, equilibrium
    import pycalphad.variables as v
    print("[OK] PyCalphad imported")

    # Load database
    db = Database('calphad_databases/alzn_sample.tdb')
    print(f"[OK] Database loaded")
    print(f"     Elements: {[el for el in db.elements if el != 'VA']}")
    print(f"     Phases: {list(db.phases.keys())}")

    # Single equilibrium calculation
    comps = ['AL', 'ZN', 'VA']
    phases = list(db.phases.keys())

    conditions = {
        v.T: 600,
        v.P: 101325,
        v.N: 1,
        v.X('ZN'): 0.3
    }

    print("\nCalculating equilibrium at T=600K, X(ZN)=0.3...")
    result = equilibrium(db, comps, phases, conditions)

    stable_phases = result.Phase.values.squeeze()
    phase_fractions = result.NP.values.squeeze()

    print(f"[OK] Equilibrium calculated!")
    print(f"     Stable phases: {stable_phases}")
    print(f"     Phase fractions: {phase_fractions}")

    print("\n[SUCCESS] PyCalphad is working correctly!")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
