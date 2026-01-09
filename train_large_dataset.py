"""
Standalone Training Script for Large Datasets

Use this script for training on large datasets (1000+ materials) locally
or in a cloud environment where you have more time and compute resources.

Usage:
    python train_large_dataset.py --dataset datasets/my_dataset.pkl --epochs 150 --batch-size 32

Features:
    - No timeout limits
    - GPU support (if available)
    - Checkpoint resumption
    - Progress tracking
    - Automatic model metadata generation
"""

import argparse
import json
import sys
import time
from pathlib import Path
import torch
from datetime import datetime

# Import training modules
from gnn_data_collection import load_graph_dataset
from gnn_dataset import CrystalGraphDataset, split_dataset, create_data_loaders
from gnn_train import GNNTrainer
from gnn_train_multitask import MultiTaskGNNTrainer
from gnn_model import CGCNN, CGCNN_CALPHAD_Regressor
from gnn_model_multitask import CGCNN_MultiTask, CGCNN_MultiTask_CALPHAD
from model_manager import ModelManager, create_training_metadata


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GNN models on large datasets without timeout limits"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file (.pkl)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name for the dataset (defaults to filename)"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)"
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension (default: 128)"
    )
    parser.add_argument(
        "--n-conv",
        type=int,
        default=3,
        help="Number of convolutional layers (default: 3)"
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=2,
        help="Number of hidden layers (default: 2)"
    )

    # Dataset split
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use (default: auto-detect)"
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )

    # Output
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10)"
    )

    return parser.parse_args()


def detect_device(device_arg):
    """Detect the best available device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def detect_dataset_type(graphs):
    """Detect if dataset is multi-task and if it has CALPHAD features."""
    sample_graph = graphs[0]

    # Detect CALPHAD
    node_feature_dim = sample_graph.x.shape[1] if len(sample_graph.x.shape) > 1 else 1
    edge_feature_dim = sample_graph.edge_attr.shape[1] if len(sample_graph.edge_attr.shape) > 1 else 1
    has_calphad = (node_feature_dim == 13 and edge_feature_dim == 2)

    # Detect multi-task
    is_multitask = False
    properties = None
    if hasattr(sample_graph, 'y') and len(sample_graph.y.shape) > 0:
        if len(sample_graph.y.shape) == 2 and sample_graph.y.shape[1] > 1:
            is_multitask = True
            if hasattr(sample_graph, 'target_properties'):
                properties = sample_graph.target_properties

    return has_calphad, is_multitask, properties, node_feature_dim, edge_feature_dim


def create_model(has_calphad, is_multitask, properties, node_dim, hidden_dim, n_conv, n_hidden):
    """Create the appropriate model based on dataset type."""
    if is_multitask and properties:
        if has_calphad:
            model = CGCNN_MultiTask_CALPHAD(
                input_node_dim=13,
                input_edge_dim=2,
                hidden_dim=hidden_dim,
                n_conv=n_conv,
                n_hidden=n_hidden,
                properties=properties
            )
            print("✓ Using CALPHAD-enhanced Multi-Task CGCNN model")
        else:
            model = CGCNN_MultiTask(
                hidden_dim=hidden_dim,
                n_conv=n_conv,
                n_hidden=n_hidden,
                properties=properties
            )
            print("✓ Using Multi-Task CGCNN model")
    else:
        if has_calphad:
            model = CGCNN_CALPHAD_Regressor(
                input_node_dim=13,
                input_edge_dim=2,
                node_feature_dim=node_dim,
                hidden_dim=hidden_dim,
                n_conv=n_conv,
                n_hidden=n_hidden
            )
            print("✓ Using CALPHAD-enhanced CGCNN model")
        else:
            model = CGCNN(
                node_feature_dim=node_dim,
                hidden_dim=hidden_dim,
                n_conv=n_conv,
                n_hidden=n_hidden,
                output_dim=1
            )
            print("✓ Using standard CGCNN model")

    return model


def main():
    """Main training function."""
    args = parse_args()

    print("="*80)
    print("GNN Training Script for Large Datasets")
    print("="*80)

    # Detect device
    device = detect_device(args.device)
    print(f"\n✓ Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    print(f"\n📊 Loading dataset: {args.dataset}")
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {args.dataset}")
        sys.exit(1)

    try:
        graphs = load_graph_dataset(str(dataset_path))
        print(f"✓ Loaded {len(graphs)} materials")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    # Detect dataset type
    has_calphad, is_multitask, properties, node_dim, edge_dim = detect_dataset_type(graphs)
    print(f"\n📋 Dataset Analysis:")
    print(f"  • CALPHAD features: {'Yes' if has_calphad else 'No'}")
    print(f"  • Multi-task: {'Yes' if is_multitask else 'No'}")
    if is_multitask and properties:
        print(f"  • Properties: {', '.join(properties)}")
    print(f"  • Node features: {node_dim}D")
    print(f"  • Edge features: {edge_dim}D")

    # Create dataset splits
    print(f"\n🔀 Splitting dataset...")
    dataset = CrystalGraphDataset(graphs)
    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio
    )

    print(f"  • Training: {len(train_dataset)} samples")
    print(f"  • Validation: {len(val_dataset)} samples")
    print(f"  • Test: {len(test_dataset)} samples")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size
    )

    # Create model
    print(f"\n🧠 Creating model...")
    model = create_model(
        has_calphad, is_multitask, properties,
        node_dim, args.hidden_dim, args.n_conv, args.n_hidden
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • Parameters: {num_params:,}")

    # Create trainer
    model_manager = ModelManager(checkpoint_dir=args.checkpoint_dir)

    if is_multitask and properties:
        trainer = MultiTaskGNNTrainer(
            model=model,
            properties=properties,
            device=device,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir
        )
    else:
        trainer = GNNTrainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir
        )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n⏯️  Resuming from checkpoint: {args.resume}")
        try:
            trainer.load_checkpoint(args.resume)
            print(f"  ✓ Loaded checkpoint from epoch {trainer.best_epoch}")
        except Exception as e:
            print(f"  ❌ Error loading checkpoint: {e}")
            print("  Starting fresh training...")

    # Generate model name
    dataset_name = args.dataset_name or dataset_path.stem
    properties_for_name = properties if properties else ["formation_energy_per_atom"]
    model_name = model_manager.generate_model_name(
        dataset_name=dataset_name,
        properties=properties_for_name,
        use_calphad=has_calphad
    )

    print(f"\n🎓 Starting Training...")
    print(f"  • Epochs: {args.epochs}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Learning rate: {args.learning_rate}")
    print(f"  • Patience: {args.patience}")
    print(f"  • Model name: {model_name}")
    print("="*80)

    start_time = time.time()

    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            patience=args.patience,
            verbose=True,
            model_name=model_name
        )

        training_time = time.time() - start_time

        print("\n" + "="*80)
        print("✅ Training Completed!")
        print("="*80)
        print(f"  • Best epoch: {trainer.best_epoch}")
        print(f"  • Best val loss: {trainer.best_val_loss:.6f}")
        print(f"  • Training time: {training_time/60:.1f} minutes")

        # Evaluate on test set
        print(f"\n🎯 Evaluating on test set...")
        test_metrics, predictions, targets = trainer.evaluate(test_loader)

        print(f"  • Test MAE: {test_metrics['mae']:.6f}")
        print(f"  • Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"  • Test R²: {test_metrics['r2']:.4f}")

        # Save metadata
        print(f"\n💾 Saving model metadata...")
        metadata = create_training_metadata(
            dataset_name=dataset_name,
            properties=properties_for_name,
            num_samples=len(dataset),
            best_epoch=trainer.best_epoch,
            best_val_loss=trainer.best_val_loss,
            training_time_minutes=training_time / 60,
            use_calphad=has_calphad,
            test_metrics=test_metrics,
            hyperparameters={
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'patience': args.patience,
                'hidden_dim': args.hidden_dim,
                'n_conv': args.n_conv,
                'n_hidden': args.n_hidden,
                'device': device
            }
        )

        model_path = Path(args.checkpoint_dir) / model_name
        model_manager.save_model_with_metadata(model_path, metadata)

        print(f"  ✓ Model saved: {model_path}")
        print(f"  ✓ Metadata saved: {model_path.parent / 'metadata' / f'{model_path.stem}.json'}")

        print("\n" + "="*80)
        print("🎉 All done! You can now:")
        print(f"  1. Upload the model to your Streamlit app")
        print(f"  2. Use it for predictions")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"  Last checkpoint saved at epoch {trainer.best_epoch}")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
