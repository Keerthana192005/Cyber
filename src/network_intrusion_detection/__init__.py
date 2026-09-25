"""Network intrusion detection package."""

from .pipeline import build_model, load_or_create_demo_dataset, prepare_training_data, train_model

__all__ = [
    "load_or_create_demo_dataset",
    "prepare_training_data",
    "build_model",
    "train_model",
]
