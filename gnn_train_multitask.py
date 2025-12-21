"""
Multi-Task GNN Training

Train GNN models to predict multiple properties simultaneously.

Author: Digital Foundry Materials Science Toolkit
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from gnn_model_multitask import CGCNN_MultiTask, CGCNN_MultiTask_CALPHAD, get_property_info


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss with automatic weighting.

    Uses uncertainty weighting to balance multiple tasks.
    """

    def __init__(self, n_tasks: int):
        super(MultiTaskLoss, self).__init__()
        # Learnable log variances for uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor):
        """
        Args:
            predictions: Dict mapping property names to predictions
            targets: Tensor of shape [batch_size, n_tasks]

        Returns:
            Total weighted loss
        """
        total_loss = 0
        losses_dict = {}

        for i, (prop_name, pred) in enumerate(predictions.items()):
            # MSE loss for this task
            task_targets = targets[:, i]

            # Find valid (non-NaN) targets
            valid_mask = ~torch.isnan(task_targets)

            if valid_mask.sum() == 0:
                continue  # Skip if no valid targets

            mse_loss = nn.functional.mse_loss(
                pred[valid_mask],
                task_targets[valid_mask]
            )

            # Uncertainty weighting: loss_i / (2 * sigma_i^2) + log(sigma_i)
            # where sigma_i = exp(log_var_i / 2)
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * mse_loss + self.log_vars[i]

            total_loss += weighted_loss
            losses_dict[prop_name] = mse_loss.item()

        return total_loss, losses_dict


class MultiTaskGNNTrainer:
    """Trainer for multi-task GNN models."""

    def __init__(
        self,
        model: nn.Module,
        properties: List[str],
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        checkpoint_dir: str = 'checkpoints',
        max_grad_norm: float = 1.0
    ):
        """
        Args:
            model: Multi-task GNN model
            properties: List of property names being predicted
            device: 'cpu' or 'cuda'
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW optimizer
            checkpoint_dir: Directory to save checkpoints
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model.to(device)
        self.properties = properties
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_grad_norm = max_grad_norm

        # Optimizer (AdamW for better generalization)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Multi-task loss
        self.criterion = MultiTaskLoss(len(properties))

        # Learning rate scheduler
        self.scheduler = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            **{f'train_{prop}': [] for prop in properties},
            **{f'val_{prop}': [] for prop in properties}
        }

        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        property_losses = {prop: [] for prop in self.properties}
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            # Forward pass
            predictions = self.model(batch)

            # Compute loss
            loss, losses_dict = self.criterion(predictions, batch.y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            for prop, prop_loss in losses_dict.items():
                property_losses[prop].append(prop_loss)

            n_batches += 1

        # Average losses
        avg_total_loss = total_loss / n_batches
        avg_property_losses = {
            prop: np.mean(losses) if losses else 0
            for prop, losses in property_losses.items()
        }

        return avg_total_loss, avg_property_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0
        property_losses = {prop: [] for prop in self.properties}
        n_batches = 0

        for batch in val_loader:
            batch = batch.to(self.device)

            predictions = self.model(batch)
            loss, losses_dict = self.criterion(predictions, batch.y)

            total_loss += loss.item()
            for prop, prop_loss in losses_dict.items():
                property_losses[prop].append(prop_loss)

            n_batches += 1

        if n_batches == 0:
            return 0.0, {prop: 0.0 for prop in self.properties}

        avg_total_loss = total_loss / n_batches
        avg_property_losses = {
            prop: np.mean(losses) if losses else 0
            for prop, losses in property_losses.items()
        }

        return avg_total_loss, avg_property_losses

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 20,
        verbose: bool = True,
        use_scheduler: bool = True
    ) -> Dict:
        """
        Train the model with learning rate scheduling.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            patience: Early stopping patience
            verbose: Print progress
            use_scheduler: Use cosine annealing LR scheduler

        Returns:
            Training history dictionary
        """
        # Initialize learning rate scheduler
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
            if verbose:
                print("✓ Using Cosine Annealing LR Scheduler")

        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss, train_prop_losses = self.train_epoch(train_loader)

            # Validate
            val_loss, val_prop_losses = self.validate(val_loader)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            for prop in self.properties:
                self.history[f'train_{prop}'].append(train_prop_losses.get(prop, 0))
                self.history[f'val_{prop}'].append(val_prop_losses.get(prop, 0))

            # Print progress
            if verbose:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\nEpoch {epoch+1}/{epochs} | LR: {current_lr:.2e}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {val_loss:.6f}")

                for prop in self.properties:
                    train_prop = train_prop_losses.get(prop, 0)
                    val_prop = val_prop_losses.get(prop, 0)
                    print(f"    {prop}: train={train_prop:.6f}, val={val_prop:.6f}")

            # Update learning rate
            if use_scheduler and self.scheduler is not None:
                self.scheduler.step()

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint('best_model_multitask.pt')
                patience_counter = 0

                if verbose:
                    print(f"  ✅ New best model (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                    print(f"   Best epoch: {self.best_epoch+1}")
                    print(f"   Best val loss: {self.best_val_loss:.6f}")
                break

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'properties': self.properties,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        load_path = self.checkpoint_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.properties = checkpoint.get('properties', self.properties)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.history = checkpoint.get('history', self.history)

    def plot_training_history(self, save_path: str = None):
        """Plot training curves for all properties."""
        n_props = len(self.properties)
        fig, axes = plt.subplots(1 + n_props, 1, figsize=(10, 4*(1+n_props)))

        if n_props == 0:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Overall loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].axvline(self.best_epoch, color='r', linestyle='--', alpha=0.5, label='Best')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Overall Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Individual properties
        for i, prop in enumerate(self.properties):
            axes[i+1].plot(self.history[f'train_{prop}'], label='Train')
            axes[i+1].plot(self.history[f'val_{prop}'], label='Val')
            axes[i+1].axvline(self.best_epoch, color='r', linestyle='--', alpha=0.5)

            prop_info = get_property_info(prop)
            axes[i+1].set_xlabel('Epoch')
            axes[i+1].set_ylabel(f'MSE ({prop_info["unit"]})')
            axes[i+1].set_title(f'{prop_info["name"]} Loss')
            axes[i+1].legend()
            axes[i+1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        return fig


if __name__ == "__main__":
    """Test multi-task trainer."""
    print("Testing Multi-Task Trainer")
    print("="*70)

    # Create dummy dataset
    from torch_geometric.data import Data
    import torch

    dummy_graphs = []
    for i in range(100):
        graph = Data(
            x=torch.randn(10, 64),
            edge_index=torch.randint(0, 10, (2, 30)),
            edge_attr=torch.randn(30, 1),
            y=torch.tensor([
                -0.5 + 0.1*np.random.randn(),  # formation_energy
                1.0 + 0.5*np.random.randn(),   # band_gap
                8.0 + 1.0*np.random.randn()    # density
            ], dtype=torch.float)
        )
        dummy_graphs.append(graph)

    # Create loaders
    train_loader = DataLoader(dummy_graphs[:80], batch_size=16, shuffle=True)
    val_loader = DataLoader(dummy_graphs[80:], batch_size=16)

    # Create model
    properties = ['formation_energy_per_atom', 'band_gap', 'density']
    model = CGCNN_MultiTask(properties=properties)

    print(f"Properties: {properties}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = MultiTaskGNNTrainer(model, properties)

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=10,
        patience=5
    )

    print("\n✅ Training test completed!")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
