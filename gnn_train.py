"""
GNN Training Script

Complete training pipeline for Crystal Graph Convolutional Neural Networks.
Includes training loop, validation, model checkpointing, and metrics tracking.

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from gnn_model import CGCNN
from gnn_dataset import CrystalGraphDataset, NormalizeTarget, split_dataset, create_data_loaders
from gnn_data_collection import load_graph_dataset


class GNNTrainer:
    """
    Trainer class for GNN models with full training pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Args:
            model: CGCNN model to train
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer and loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "learning_rate": []
        }

        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Tuple of (average_loss, average_mae)
        """
        self.model.train()
        total_loss = 0
        total_mae = 0
        num_batches = len(train_loader)

        for batch in train_loader:
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)

            # Calculate loss
            loss = self.criterion(predictions.squeeze(), batch.y.squeeze())

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            mae = torch.mean(torch.abs(predictions.squeeze() - batch.y.squeeze()))
            total_mae += mae.item()

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        return avg_loss, avg_mae

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Tuple of (average_loss, average_mae)
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = len(val_loader)

        for batch in val_loader:
            batch = batch.to(self.device)

            # Forward pass
            predictions = self.model(batch)

            # Calculate loss
            loss = self.criterion(predictions.squeeze(), batch.y.squeeze())

            # Track metrics
            total_loss += loss.item()
            mae = torch.mean(torch.abs(predictions.squeeze() - batch.y.squeeze()))
            total_mae += mae.item()

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        return avg_loss, avg_mae

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        print(f"\nStarting training on {self.device}")
        print(f"Epochs: {epochs}, Patience: {patience}")
        print("="*60)

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_mae = self.train_epoch(train_loader)

            # Validate
            val_loss, val_mae = self.validate(val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_mae"].append(train_mae)
            self.history["val_mae"].append(val_mae)
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )

            # Print progress
            if verbose:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train MAE: {train_mae:.4f} | "
                      f"Val MAE: {val_mae:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint("best_model.pt", epoch, val_loss)
                patience_counter = 0
                if verbose:
                    print(f"  -> New best model! (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best model at epoch {self.best_epoch} "
                      f"with val loss: {self.best_val_loss:.4f}")
                break

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch, val_loss)

        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best model: epoch {self.best_epoch}, val loss: {self.best_val_loss:.4f}")

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            epoch: Current epoch
            val_loss: Validation loss
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch = checkpoint["best_epoch"]

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.

        Args:
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Loss plot
        ax1.plot(epochs, self.history["train_loss"], label="Train Loss", marker='o', markersize=3)
        ax1.plot(epochs, self.history["val_loss"], label="Val Loss", marker='s', markersize=3)
        ax1.axvline(self.best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best (Epoch {self.best_epoch})')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (MSE)")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE plot
        ax2.plot(epochs, self.history["train_mae"], label="Train MAE", marker='o', markersize=3)
        ax2.plot(epochs, self.history["val_mae"], label="Val MAE", marker='s', markersize=3)
        ax2.axvline(self.best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best (Epoch {self.best_epoch})')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE")
        ax2.set_title("Training and Validation MAE")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")

        return fig

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []

        for batch in test_loader:
            batch = batch.to(self.device)
            predictions = self.model(batch)

            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_targets.extend(batch.y.squeeze().cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Calculate metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(mse)

        # R^2 score
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "num_samples": len(all_predictions)
        }

        return metrics, all_predictions, all_targets


def print_evaluation_results(metrics: Dict):
    """Print evaluation metrics in a nice format."""
    print("\n" + "="*60)
    print("Test Set Evaluation Results")
    print("="*60)
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R²:   {metrics['r2']:.6f}")
    print("="*60)


if __name__ == "__main__":
    """Example training run"""
    from gnn_data_collection import load_graph_dataset

    print("GNN Training Pipeline")
    print("="*60)

    # Load dataset
    try:
        graphs = load_graph_dataset("datasets/fe_ni_graphs.pkl")
    except FileNotFoundError:
        print("Error: Dataset not found!")
        print("Run gnn_data_collection.py first to create a dataset.")
        exit(1)

    # Create dataset
    dataset = CrystalGraphDataset(graphs)

    # Get normalization statistics
    mean, std = dataset.get_target_statistics()
    print(f"\nTarget statistics: mean={mean:.4f}, std={std:.4f}")

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=32
    )

    # Create model
    model = CGCNN(
        node_feature_dim=64,
        edge_feature_dim=1,
        hidden_dim=128,
        n_conv=3,
        n_hidden=1,
        output_dim=1
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = GNNTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        checkpoint_dir="checkpoints"
    )

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        patience=20
    )

    # Plot training history
    trainer.plot_training_history("training_history.png")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics, predictions, targets = trainer.evaluate(test_loader)
    print_evaluation_results(metrics)

    # Save metrics
    with open("test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining complete! Best model saved to checkpoints/best_model.pt")
