"""
Test script to verify GNN dependencies are installed correctly
Run this after installing requirements.txt
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if os.name == 'nt':  # Windows
    import sys
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_imports():
    """Test that all required packages can be imported"""
    print("=" * 60)
    print("Testing GNN Dependencies Installation")
    print("=" * 60)
    print()

    tests_passed = 0
    tests_failed = 0

    # Test 1: PyTorch
    print("1. Testing PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch version: {torch.__version__}")
        print(f"   📍 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   🎮 CUDA version: {torch.version.cuda}")
        else:
            print(f"   💻 Using CPU (expected on Streamlit Cloud)")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1

    print()

    # Test 2: PyTorch Geometric
    print("2. Testing PyTorch Geometric...")
    try:
        import torch_geometric
        print(f"   ✅ PyTorch Geometric version: {torch_geometric.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1

    print()

    # Test 3: torch-scatter
    print("3. Testing torch-scatter...")
    try:
        import torch_scatter
        print(f"   ✅ torch-scatter imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1

    print()

    # Test 4: torch-sparse
    print("4. Testing torch-sparse...")
    try:
        import torch_sparse
        print(f"   ✅ torch-sparse imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1

    print()

    # Test 5: torch-cluster
    print("5. Testing torch-cluster...")
    try:
        import torch_cluster
        print(f"   ✅ torch-cluster imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1

    print()

    # Test 6: Pymatgen
    print("6. Testing Pymatgen...")
    try:
        import pymatgen
        from pymatgen.core import Structure, Lattice
        try:
            version = pymatgen.__version__
        except AttributeError:
            # Older versions may not have __version__
            version = "installed (version unknown)"
        print(f"   ✅ Pymatgen: {version}")
        print(f"   ✅ Can import Structure and Lattice")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1

    print()

    # Test 7: Create a simple graph
    print("7. Testing Graph Creation...")
    try:
        from torch_geometric.data import Data
        import torch

        # Create simple graph: triangle
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                                   [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        x = torch.tensor([[1], [2], [3]], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)

        print(f"   ✅ Created simple graph with {data.num_nodes} nodes and {data.num_edges} edges")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1

    print()

    # Test 8: Create a simple crystal structure
    print("8. Testing Crystal Structure Creation...")
    try:
        from pymatgen.core import Structure, Lattice

        # Create simple FCC structure
        lattice = Lattice.cubic(4.0)
        structure = Structure(lattice, ["Fe"], [[0, 0, 0]])

        print(f"   ✅ Created structure: {structure.formula}")
        print(f"   ✅ Number of sites: {len(structure)}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1

    print()

    # Test 9: Test GNN layer
    print("9. Testing GNN Layer...")
    try:
        from torch_geometric.nn import GCNConv
        import torch

        # Create simple GCN layer
        conv = GCNConv(in_channels=16, out_channels=32)

        # Test forward pass
        x = torch.randn(3, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        out = conv(x, edge_index)

        print(f"   ✅ GCN layer forward pass successful")
        print(f"   ✅ Input shape: {x.shape}, Output shape: {out.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✅ Tests passed: {tests_passed}/9")
    print(f"❌ Tests failed: {tests_failed}/9")
    print()

    if tests_failed == 0:
        print("🎉 All tests passed! GNN dependencies are ready!")
        print()
        print("Next steps:")
        print("1. Create crystal graph construction module")
        print("2. Implement GNN model architecture")
        print("3. Build Streamlit interface")
        return True
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print()
        print("Installation tips:")
        print("1. Make sure you activated the correct environment")
        print("2. Try: pip install torch torch-geometric")
        print("3. For torch-scatter, torch-sparse: pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
