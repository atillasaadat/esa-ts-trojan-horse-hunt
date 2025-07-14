"""Configuration settings for the Trojan trigger reconstruction project."""

from pathlib import Path
from typing import List, Tuple
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "poisoned_models"
CLEAN_DATA_PATH = DATA_DIR / "clean_train_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model configuration
N_MODELS = 45
N_CHANNELS = 3
TRIGGER_LENGTH = 75
CHANNEL_NAMES = ["channel_44", "channel_45", "channel_46"]

# Neural Cleanse baseline parameters
BASELINE_CONFIG = {
    "trigger_shape": (N_CHANNELS, TRIGGER_LENGTH),
    "learning_rate": 1e-3,
    "num_epochs": 1000,
    "l1_lambda": 0.01,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "optimizer": "adamw",
    "scheduler": "cosine",
    "early_stopping_patience": 50,
    "batch_size": 32,
}

# Weights & Biases configuration
WANDB_CONFIG = {
    "project": "esa-trojan-horse-hunt",
    "entity": None,  # Set your username here
    "tags": ["baseline", "neural-cleanse"],
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "trojan_hunt.log",
}
