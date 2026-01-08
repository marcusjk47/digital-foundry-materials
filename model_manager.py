"""
Model Management System
Handles saving, loading, and managing trained GNN models with metadata.
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import shutil


class ModelManager:
    """
    Manages trained models with metadata and versioning.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize model manager.

        Args:
            checkpoint_dir: Directory to store model checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.checkpoint_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

    def generate_model_name(
        self,
        dataset_name: str,
        properties: List[str],
        use_calphad: bool = False
    ) -> str:
        """
        Generate a unique model name with timestamp.

        Args:
            dataset_name: Name of the dataset used for training
            properties: List of target properties
            use_calphad: Whether CALPHAD features were used

        Returns:
            Unique model filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Property suffix
        if len(properties) == 1:
            prop_suffix = properties[0][:8]  # Truncate long property names
        else:
            prop_suffix = f"multi_{len(properties)}props"

        # CALPHAD suffix
        calphad_suffix = "_calphad" if use_calphad else ""

        # Combine all parts
        model_name = f"{dataset_name}_{prop_suffix}{calphad_suffix}_{timestamp}.pt"

        return model_name

    def save_model_with_metadata(
        self,
        model_path: Path,
        metadata: Dict[str, Any]
    ):
        """
        Save model metadata as JSON file.

        Args:
            model_path: Path to the model checkpoint
            metadata: Dictionary containing model metadata
        """
        # Add save timestamp
        metadata['saved_at'] = datetime.now().isoformat()

        # Create metadata filename
        metadata_file = self.metadata_dir / f"{model_path.stem}.json"

        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a model.

        Args:
            model_name: Name of the model file (without .pt extension)

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_file = self.metadata_dir / f"{model_name}.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r') as f:
            return json.load(f)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with their metadata.

        Returns:
            List of dictionaries containing model info and metadata
        """
        models = []

        for model_file in self.checkpoint_dir.glob("*.pt"):
            metadata = self.load_metadata(model_file.stem)

            # Get file size
            file_size_mb = model_file.stat().st_size / (1024 * 1024)

            model_info = {
                'filename': model_file.name,
                'path': str(model_file),
                'size_mb': round(file_size_mb, 2),
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                'metadata': metadata
            }

            models.append(model_info)

        # Sort by modification time (newest first)
        models.sort(key=lambda x: x['modified'], reverse=True)

        return models

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model and its metadata.

        Args:
            model_name: Name of the model file (with or without .pt extension)

        Returns:
            True if deleted successfully, False otherwise
        """
        # Remove .pt extension if present
        model_stem = model_name.replace('.pt', '')

        model_path = self.checkpoint_dir / f"{model_stem}.pt"
        metadata_path = self.metadata_dir / f"{model_stem}.json"

        success = True

        # Delete model file
        if model_path.exists():
            try:
                model_path.unlink()
            except Exception as e:
                print(f"Error deleting model: {e}")
                success = False

        # Delete metadata file
        if metadata_path.exists():
            try:
                metadata_path.unlink()
            except Exception as e:
                print(f"Error deleting metadata: {e}")
                success = False

        return success

    def export_model(self, model_name: str, export_dir: str) -> Path:
        """
        Export a model and its metadata to a directory.

        Args:
            model_name: Name of the model file
            export_dir: Directory to export to

        Returns:
            Path to exported model
        """
        model_stem = model_name.replace('.pt', '')

        model_path = self.checkpoint_dir / f"{model_stem}.pt"
        metadata_path = self.metadata_dir / f"{model_stem}.json"

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # Copy model
        if model_path.exists():
            shutil.copy(model_path, export_path / model_path.name)

        # Copy metadata
        if metadata_path.exists():
            shutil.copy(metadata_path, export_path / metadata_path.name)

        return export_path / model_path.name

    def get_model_summary(self, model_name: str) -> Optional[str]:
        """
        Get a human-readable summary of a model.

        Args:
            model_name: Name of the model file

        Returns:
            Formatted summary string or None if not found
        """
        metadata = self.load_metadata(model_name.replace('.pt', ''))

        if not metadata:
            return None

        summary_parts = []

        # Dataset info
        if 'dataset_name' in metadata:
            summary_parts.append(f"📊 Dataset: {metadata['dataset_name']}")

        # Properties
        if 'properties' in metadata:
            props = metadata['properties']
            if isinstance(props, list):
                summary_parts.append(f"🎯 Properties: {', '.join(props)}")
            else:
                summary_parts.append(f"🎯 Property: {props}")

        # Training metrics
        if 'best_val_loss' in metadata:
            summary_parts.append(f"📉 Best Val Loss: {metadata['best_val_loss']:.6f}")

        if 'test_metrics' in metadata:
            metrics = metadata['test_metrics']
            if 'mae' in metrics:
                summary_parts.append(f"📏 Test MAE: {metrics['mae']:.6f}")
            if 'r2' in metrics:
                summary_parts.append(f"📈 Test R²: {metrics['r2']:.4f}")

        # Training info
        if 'training_time_minutes' in metadata:
            summary_parts.append(f"⏱️ Training Time: {metadata['training_time_minutes']:.1f} min")

        if 'num_samples' in metadata:
            summary_parts.append(f"🔢 Samples: {metadata['num_samples']}")

        # Model features
        if 'use_calphad' in metadata and metadata['use_calphad']:
            summary_parts.append("⚗️ CALPHAD Features: Yes")

        return "\n".join(summary_parts)


def create_training_metadata(
    dataset_name: str,
    properties: List[str],
    num_samples: int,
    best_epoch: int,
    best_val_loss: float,
    training_time_minutes: float,
    use_calphad: bool = False,
    chemical_system: Optional[str] = None,
    test_metrics: Optional[Dict[str, float]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a metadata dictionary for a trained model.

    Args:
        dataset_name: Name of the dataset
        properties: List of target properties
        num_samples: Number of training samples
        best_epoch: Best training epoch
        best_val_loss: Best validation loss
        training_time_minutes: Total training time in minutes
        use_calphad: Whether CALPHAD features were used
        chemical_system: Chemical system (e.g., "Fe-Ni-Cr")
        test_metrics: Test set evaluation metrics
        hyperparameters: Training hyperparameters

    Returns:
        Metadata dictionary
    """
    metadata = {
        'dataset_name': dataset_name,
        'properties': properties,
        'num_samples': num_samples,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'training_time_minutes': training_time_minutes,
        'use_calphad': use_calphad,
        'chemical_system': chemical_system,
        'test_metrics': test_metrics or {},
        'hyperparameters': hyperparameters or {},
        'created_at': datetime.now().isoformat()
    }

    return metadata
