"""
Test PyCalphad Installation
Run this to verify CALPHAD tools are working correctly
"""

print("=" * 60)
print("Testing PyCalphad Installation")
print("=" * 60)

# Test 1: Import PyCalphad
print("\n1. Testing PyCalphad import...")
try:
    import pycalphad
    print(f"   ✓ PyCalphad {pycalphad.__version__} imported successfully!")
except ImportError as e:
    print(f"   ✗ Error: {e}")
    print("\n   Install with: conda install -c conda-forge pycalphad")
    exit(1)

# Test 2: Import core modules
print("\n2. Testing core modules...")
try:
    from pycalphad import Database, equilibrium, variables as v
    from pycalphad import binplot, ternplot
    print("   ✓ Core modules imported successfully!")
except ImportError as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 3: Import ESPEI
print("\n3. Testing ESPEI...")
try:
    import espei
    print(f"   ✓ ESPEI imported successfully!")
except ImportError as e:
    print(f"   ⚠ ESPEI not available: {e}")
    print("   (Optional - install with: conda install -c conda-forge espei)")

# Test 4: Import Scheil
print("\n4. Testing Scheil...")
try:
    import scheil
    print(f"   ✓ Scheil imported successfully!")
except ImportError as e:
    print(f"   ⚠ Scheil not available: {e}")
    print("   (Optional - install with: conda install -c conda-forge scheil)")

# Test 5: Load a simple database
print("\n5. Testing database functionality...")
try:
    # Create a minimal TDB content for testing
    tdb_content = """
    $ Simple test database for Al-Zn
    ELEMENT AL FCC_A1 26.9815 4577.3 28.3 !
    ELEMENT ZN HCP_A3 65.39 5657.0 41.6 !
    ELEMENT VA VACUUM 0 0 0 !

    FUNCTION GHSERAL 298.15
     -7976.15+137.093038*T-24.3671976*T*LN(T)
     -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 700 Y
     -11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2
     -5.764227E-06*T**3+74092*T**(-1); 933.47 Y
     -11278.378+188.684153*T-31.748192*T*LN(T)-1.230524E+28*T**(-9);
     2900 N !

    FUNCTION GHSERZN 298.15
     -7285.787+118.470069*T-23.701314*T*LN(T)
     -.001712034*T**2-1.264963E-06*T**3; 692.68 Y
     -11070.559+172.345644*T-31.38*T*LN(T)+4.70657E+26*T**(-9);
     1700 N !

    PHASE FCC_A1 %  1  1.0  !
    CONSTITUENT FCC_A1 :AL,ZN : !
    PARAMETER G(FCC_A1,AL;0) 298.15 +GHSERAL; 2900 N !
    PARAMETER G(FCC_A1,ZN;0) 298.15 +GHSERZN+2969.82-1.56968*T; 1700 N !

    PHASE HCP_A3 %  1  1.0  !
    CONSTITUENT HCP_A3 :AL,ZN : !
    PARAMETER G(HCP_A3,AL;0) 298.15 +GHSERAL+5481-1.8*T; 2900 N !
    PARAMETER G(HCP_A3,ZN;0) 298.15 +GHSERZN; 1700 N !

    PHASE LIQUID %  1  1.0  !
    CONSTITUENT LIQUID :AL,ZN : !
    PARAMETER G(LIQUID,AL;0) 298.15
     +11005.029-11.841867*T+7.934E-20*T**7+GHSERAL; 933.47 Y
     +10482.382-11.253974*T+1.231E+28*T**(-9)+GHSERAL; 2900 N !
    PARAMETER G(LIQUID,ZN;0) 298.15
     +7157.213-10.29299*T-3.5896E-19*T**7+GHSERZN; 692.68 Y
     +7450.168-10.737066*T-4.70657E+26*T**(-9)+GHSERZN; 1700 N !
    PARAMETER G(LIQUID,AL,ZN;0) 298.15 +6656.0-3.5*T; 6000 N !
    """

    # Save to temp file
    import os
    test_db_path = 'test_database_temp.tdb'

    with open(test_db_path, 'w') as f:
        f.write(tdb_content)

    db = Database(test_db_path)
    print(f"   ✓ Database loaded successfully!")
    print(f"   ✓ Elements: {sorted(db.elements)}")
    print(f"   ✓ Phases: {sorted(db.phases.keys())}")

    # Test equilibrium calculation
    print("\n6. Testing equilibrium calculation...")
    comps = ['AL', 'ZN', 'VA']
    phases = list(db.phases.keys())

    result = equilibrium(db, comps, phases, {
        v.T: 600,
        v.P: 101325,
        v.X('ZN'): 0.3,
        v.N: 1
    })

    print(f"   ✓ Equilibrium calculated at 600K, 30% Zn")
    stable_phases = result.Phase.values.squeeze()
    if hasattr(stable_phases, '__iter__') and not isinstance(stable_phases, str):
        stable_phases = [p for p in stable_phases if p != '']
    else:
        stable_phases = [stable_phases] if stable_phases != '' else []

    print(f"   ✓ Stable phase(s): {stable_phases}")

    # Clean up
    os.remove(test_db_path)
    print(f"   ✓ Cleaned up test files")

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    # Try to clean up even on error
    try:
        import os
        if os.path.exists('test_database_temp.tdb'):
            os.remove('test_database_temp.tdb')
    except:
        pass
    exit(1)

# Test 7: Dependencies
print("\n7. Testing dependencies...")
deps_ok = True

try:
    import numpy
    print(f"   ✓ NumPy {numpy.__version__}")
except:
    print("   ✗ NumPy missing - install with: pip install numpy")
    deps_ok = False

try:
    import scipy
    print(f"   ✓ SciPy {scipy.__version__}")
except:
    print("   ✗ SciPy missing - install with: pip install scipy")
    deps_ok = False

try:
    import matplotlib
    print(f"   ✓ Matplotlib {matplotlib.__version__}")
except:
    print("   ✗ Matplotlib missing - install with: pip install matplotlib")
    deps_ok = False

try:
    import xarray
    print(f"   ✓ xarray {xarray.__version__}")
except:
    print("   ⚠ xarray missing (recommended) - install with: pip install xarray")

try:
    import symengine
    print(f"   ✓ symengine {symengine.__version__}")
except:
    print("   ⚠ symengine missing (recommended for speed) - install with: conda install -c conda-forge symengine")

print("\n" + "=" * 60)
if deps_ok:
    print("✅ All critical tests passed! CALPHAD is ready to use!")
    print("=" * 60)
    print("\n🎯 Next steps:")
    print("1. Download thermodynamic databases (TDB files) from NIMS")
    print("2. Place TDB files in: calphad_databases/")
    print("3. Run example calculations (see CALPHAD_SETUP_GUIDE.md)")
    print("4. Build Streamlit app for phase diagrams")
    print("\n📚 Resources:")
    print("- Setup Guide: CALPHAD_SETUP_GUIDE.md")
    print("- NIMS Database: https://cpddb.nims.go.jp/")
    print("- PyCalphad Docs: https://pycalphad.org/docs/")
else:
    print("⚠️  Some dependencies missing - see messages above")
    print("=" * 60)

print()
