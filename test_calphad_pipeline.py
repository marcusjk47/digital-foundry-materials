"""
Test CALPHAD-Enhanced GNN Training Pipeline

Tests the complete end-to-end training pipeline with CALPHAD thermodynamic features:
1. Data collection from Materials Project
2. CALPHAD-enhanced graph construction
3. Dataset preparation
4. Model training with CALPHAD features
5. Evaluation

Author: Digital Foundry Materials Science Toolkit
"""

import os
import sys
from pathlib import Path
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_calphad_pipeline():
    """Test the complete CALPHAD-enhanced training pipeline."""

    print("="*80)
    print("CALPHAD-Enhanced GNN Training Pipeline Test")
    print("="*80)
    print()

    # Test 1: Import all required modules
    print("Test 1: Importing modules...")
    try:
        from gnn_data_collection import fetch_materials_data, convert_to_graphs, get_dataset_statistics
        from gnn_dataset import CrystalGraphDataset, split_dataset, create_data_loaders
        from gnn_model_calphad import CGCNN_CALPHAD_Regressor, count_parameters
        from gnn_train import GNNTrainer
        print("[OK] All modules imported successfully")
    except Exception as e:
        print(f"[FAIL] Module import failed: {e}")
        return False
    print()

    # Test 2: Check API key
    print("Test 2: Checking Materials Project API key...")
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print("[FAIL] MP_API_KEY not found in environment")
        print("   Please add MP_API_KEY to your .env file")
        return False
    print(f"[OK] API key found: {api_key[:10]}...")
    print()

    # Test 3: Fetch small dataset
    print("Test 3: Fetching Fe-Ni materials from Materials Project...")
    try:
        df = fetch_materials_data(
            api_key=api_key,
            elements=["Fe", "Ni"],
            max_materials=20,
            metallic_only=True,
            stable_only=False
        )

        if df.empty:
            print("[FAIL] No materials fetched")
            return False

        print(f"[OK] Fetched {len(df)} materials")
    except Exception as e:
        print(f"[FAIL] Data fetch failed: {e}")
        return False
    print()

    # Test 4: Convert to CALPHAD-enhanced graphs
    print("Test 4: Converting to CALPHAD-enhanced graphs...")
    try:
        graphs = convert_to_graphs(
            df,
            target_property="formation_energy_per_atom",
            cutoff=8.0,
            max_neighbors=12,
            use_calphad=True,  # Enable CALPHAD features
            save_path=None  # Don't save for testing
        )

        if not graphs:
            print("[FAIL] No graphs created")
            return False

        print(f"[OK] Created {len(graphs)} CALPHAD-enhanced graphs")

        # Verify feature dimensions
        sample_graph = graphs[0]
        node_dim = sample_graph.x.shape[1]
        edge_dim = sample_graph.edge_attr.shape[1]

        print(f"   Node features: {sample_graph.x.shape} (Expected: [num_nodes, 13])")
        print(f"   Edge features: {sample_graph.edge_attr.shape} (Expected: [num_edges, 2])")

        if node_dim != 13:
            print(f"[FAIL] Incorrect node dimension: {node_dim} (expected 13)")
            return False

        if edge_dim != 2:
            print(f"[FAIL] Incorrect edge dimension: {edge_dim} (expected 2)")
            return False

        print("[OK] Feature dimensions correct!")

    except Exception as e:
        print(f"[FAIL] Graph conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Test 5: Create dataset and dataloaders
    print("Test 5: Creating dataset and dataloaders...")
    try:
        # Create dataset
        dataset = CrystalGraphDataset(graphs)
        print(f"[OK] Created dataset with {len(dataset)} graphs")

        # Split dataset
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )

        print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Create dataloaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=4,
            num_workers=0
        )

        print(f"[OK] Created dataloaders")

    except Exception as e:
        print(f"[FAIL] Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Test 6: Create CALPHAD-enhanced model
    print("Test 6: Creating CALPHAD-enhanced CGCNN model...")
    try:
        model = CGCNN_CALPHAD_Regressor(
            input_node_dim=13,      # CALPHAD-enhanced
            input_edge_dim=2,       # distance + mixing_energy
            node_feature_dim=64,
            edge_feature_dim=32,
            hidden_dim=128,
            n_conv=3,
            n_hidden=1
        )

        num_params = count_parameters(model)
        print(f"[OK] Model created with {num_params:,} parameters")

    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Test 7: Test forward pass
    print("Test 7: Testing forward pass...")
    try:
        # Get a batch
        batch = next(iter(train_loader))

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch)

        print(f"   Batch size: {len(output)}")
        print(f"   Output shape: {output.shape}")
        print(f"   Sample predictions: {output[:3]}")

        print("[OK] Forward pass successful")

    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Test 8: Test training step
    print("Test 8: Testing training step...")
    try:
        # Create trainer
        trainer = GNNTrainer(
            model=model,
            learning_rate=0.001,
            device="cpu"
        )

        # Train one epoch
        train_loss, train_mae = trainer.train_epoch(train_loader)

        print(f"   Train loss: {train_loss:.6f}")
        print(f"   Train MAE: {train_mae:.6f}")

        print("[OK] Training step successful")

    except Exception as e:
        print(f"[FAIL] Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Test 9: Test validation
    print("Test 9: Testing validation...")
    try:
        val_loss, val_mae = trainer.validate(val_loader)

        print(f"   Validation loss: {val_loss:.6f}")
        print(f"   Validation MAE: {val_mae:.6f}")

        print("[OK] Validation successful")

    except Exception as e:
        print(f"[FAIL] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Test 10: Test evaluation
    print("Test 10: Testing evaluation...")
    try:
        metrics, predictions, targets = trainer.evaluate(test_loader)

        print(f"   Test MAE: {metrics['mae']:.6f}")
        print(f"   Test RMSE: {metrics['rmse']:.6f}")
        print(f"   Test R²: {metrics['r2']:.6f}")
        print(f"   Num samples: {metrics['num_samples']}")

        print("[OK] Evaluation successful")

    except Exception as e:
        print(f"[FAIL] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Test 11: Quick training run (5 epochs)
    print("Test 11: Quick training run (5 epochs)...")
    try:
        # Reset model
        model = CGCNN_CALPHAD_Regressor(
            input_node_dim=13,
            input_edge_dim=2,
            node_feature_dim=64,
            edge_feature_dim=32,
            hidden_dim=128,
            n_conv=3,
            n_hidden=1
        )

        trainer = GNNTrainer(
            model=model,
            learning_rate=0.001,
            device="cpu"
        )

        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,
            patience=10,
            verbose=True
        )

        print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"   Final val loss: {history['val_loss'][-1]:.6f}")
        print(f"   Best epoch: {trainer.best_epoch}")

        print("[OK] Training run successful")

    except Exception as e:
        print(f"[FAIL] Training run failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Test 12: Dataset statistics
    print("Test 12: Checking dataset statistics...")
    try:
        stats = get_dataset_statistics(graphs)

        print(f"   Total graphs: {stats['num_graphs']}")
        print(f"   Avg nodes: {stats['avg_nodes']:.1f}")
        print(f"   Avg edges: {stats['avg_edges']:.1f}")
        print(f"   Target mean: {stats['target_mean']:.4f} eV/atom")
        print(f"   Target std: {stats['target_std']:.4f} eV/atom")

        print("[OK] Statistics correct")

    except Exception as e:
        print(f"[FAIL] Statistics failed: {e}")
        return False
    print()

    # All tests passed!
    print("="*80)
    print("[OK] ALL TESTS PASSED!")
    print("="*80)
    print()
    print("CALPHAD-Enhanced Training Pipeline Summary:")
    print(f"  - Feature extraction: [OK] Working")
    print(f"  - Graph construction: [OK] Working (13D nodes, 2D edges)")
    print(f"  - Model architecture: [OK] Working ({num_params:,} parameters)")
    print(f"  - Training pipeline: [OK] Working")
    print(f"  - Validation: [OK] Working")
    print(f"  - Evaluation: [OK] Working")
    print()
    print("[READY] The CALPHAD-enhanced GNN training pipeline is fully functional!")
    print()

    return True


if __name__ == "__main__":
    success = test_calphad_pipeline()

    if success:
        print("[OK] Test completed successfully!")
        sys.exit(0)
    else:
        print("[FAIL] Test failed!")
        sys.exit(1)
