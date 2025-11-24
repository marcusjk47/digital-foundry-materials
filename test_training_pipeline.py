"""
Test Training Pipeline

Quick test to validate data collection, dataset creation, and training setup.

Author: Digital Foundry Materials Science Toolkit
"""

import os
from dotenv import load_dotenv

print("="*60)
print("Testing GNN Training Pipeline")
print("="*60)

# Load environment variables
load_dotenv()
api_key = os.environ.get("MP_API_KEY")

if not api_key:
    print("\nError: MP_API_KEY not found in environment")
    print("Please set MP_API_KEY in .env file or environment variables")
    exit(1)

print("\nMaterials Project API key found")

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from gnn_data_collection import fetch_materials_data, convert_to_graphs, get_dataset_statistics, print_dataset_info
    from gnn_dataset import CrystalGraphDataset, split_dataset, create_data_loaders
    from gnn_train import GNNTrainer
    from gnn_model import CGCNN
    print("   [OK] All modules imported successfully")
except ImportError as e:
    print(f"   [ERROR] Import error: {e}")
    exit(1)

# Test 2: Fetch small dataset
print("\n2. Testing data collection (fetching 20 Fe-Ni materials)...")
try:
    df = fetch_materials_data(
        api_key=api_key,
        elements=["Fe", "Ni"],
        max_materials=20,
        metallic_only=True,
        stable_only=False
    )
    print(f"   [OK] Fetched {len(df)} materials")
except Exception as e:
    print(f"   [ERROR] Data collection failed: {e}")
    exit(1)

# Test 3: Convert to graphs
print("\n3. Testing graph conversion...")
try:
    graphs = convert_to_graphs(
        df,
        target_property="formation_energy_per_atom",
        cutoff=8.0,
        max_neighbors=12,
        save_path="datasets/test_fe_ni.pkl"
    )
    print(f"   [OK] Converted {len(graphs)} structures to graphs")
except Exception as e:
    print(f"   [ERROR] Graph conversion failed: {e}")
    exit(1)

# Test 4: Dataset statistics
print("\n4. Checking dataset statistics...")
try:
    stats = get_dataset_statistics(graphs)
    print_dataset_info(stats)
    print("   [OK] Dataset statistics calculated")
except Exception as e:
    print(f"   [ERROR] Statistics failed: {e}")
    exit(1)

# Test 5: Create PyTorch dataset
print("\n5. Testing PyTorch dataset creation...")
try:
    dataset = CrystalGraphDataset(graphs)
    print(f"   [OK] Created dataset with {len(dataset)} graphs")

    # Get sample
    sample = dataset[0]
    print(f"   [OK] Sample graph: {sample.num_nodes} nodes, {sample.num_edges} edges")
except Exception as e:
    print(f"   [ERROR] Dataset creation failed: {e}")
    exit(1)

# Test 6: Split dataset
print("\n6. Testing train/val/test split...")
try:
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    print("   [OK] Dataset split successful")
except Exception as e:
    print(f"   [ERROR] Dataset split failed: {e}")
    exit(1)

# Test 7: Create data loaders
print("\n7. Testing data loaders...")
try:
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=4,
        num_workers=0
    )
    print("   [OK] Data loaders created")

    # Test batch loading
    batch = next(iter(train_loader))
    print(f"   [OK] Loaded batch: {batch.num_graphs} graphs, {batch.num_nodes} total nodes")
except Exception as e:
    print(f"   [ERROR] Data loader creation failed: {e}")
    exit(1)

# Test 8: Create model
print("\n8. Testing model creation...")
try:
    model = CGCNN(
        node_feature_dim=64,
        edge_feature_dim=1,
        hidden_dim=128,
        n_conv=3,
        n_hidden=1,
        output_dim=1
    )
    print(f"   [OK] Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"   [ERROR] Model creation failed: {e}")
    exit(1)

# Test 9: Test forward pass
print("\n9. Testing model forward pass...")
try:
    import torch
    model.eval()
    with torch.no_grad():
        output = model(batch)
    print(f"   [OK] Forward pass successful: output shape {output.shape}")
except Exception as e:
    print(f"   [ERROR] Forward pass failed: {e}")
    exit(1)

# Test 10: Create trainer
print("\n10. Testing trainer initialization...")
try:
    trainer = GNNTrainer(
        model=model,
        device="cpu",
        learning_rate=0.001,
        checkpoint_dir="checkpoints"
    )
    print("   [OK] Trainer initialized")
except Exception as e:
    print(f"   [ERROR] Trainer initialization failed: {e}")
    exit(1)

# Test 11: Run 1 training epoch
print("\n11. Testing single training epoch...")
try:
    train_loss, train_mae = trainer.train_epoch(train_loader)
    print(f"   [OK] Training epoch completed")
    print(f"      Loss: {train_loss:.6f}, MAE: {train_mae:.6f}")
except Exception as e:
    print(f"   [ERROR] Training epoch failed: {e}")
    exit(1)

# Test 12: Run validation
print("\n12. Testing validation...")
try:
    val_loss, val_mae = trainer.validate(val_loader)
    print(f"   [OK] Validation completed")
    print(f"      Loss: {val_loss:.6f}, MAE: {val_mae:.6f}")
except Exception as e:
    print(f"   [ERROR] Validation failed: {e}")
    exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nThe GNN training pipeline is fully functional!")
print("\nNext steps:")
print("1. Run full training: python gnn_train.py")
print("2. Or use Streamlit interface: streamlit run Home.py → GNN Property Predictor → Train Model")
print("3. Collect larger dataset (500-1000 materials) for better model performance")
