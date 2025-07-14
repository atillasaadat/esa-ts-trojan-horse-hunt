"""Data loading utilities for the Trojan trigger reconstruction project."""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from .config import CLEAN_DATA_PATH, MODELS_DIR, CHANNEL_NAMES
from .logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Data loader for time-series data and poisoned models."""

    def __init__(
        self, data_path: Optional[Path] = None, models_dir: Optional[Path] = None
    ):
        """Initialize the data loader."""
        self.data_path = data_path or CLEAN_DATA_PATH
        self.models_dir = models_dir or MODELS_DIR
        self.scaler = Scaler()

    def load_clean_data(self) -> TimeSeries:
        """Load and preprocess clean training data."""
        logger.info(f"📊 Loading clean data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Clean data file not found: {self.data_path}")

        # Load CSV data
        df = pd.read_csv(self.data_path, index_col=0)
        logger.info(f"Loaded data shape: {df.shape}")

        # Convert to TimeSeries
        time_series = TimeSeries.from_dataframe(df).astype(np.float32)
        logger.info(f"TimeSeries shape: {time_series.shape}")

        return time_series

    def load_poisoned_model(self, model_id: int) -> Any:
        """Load a poisoned N-HiTS model."""
        model_path = (
            self.models_dir / f"poisoned_model_{model_id}" / "poisoned_model.pt"
        )

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"🤖 Loading poisoned model {model_id} from {model_path}")

        # Load the model
        model = torch.load(model_path, map_location="cpu")
        logger.info(f"Model loaded successfully")

        return model

    def get_model_list(self) -> list:
        """Get list of available poisoned models."""
        model_dirs = [
            d
            for d in self.models_dir.iterdir()
            if d.is_dir() and d.name.startswith("poisoned_model_")
        ]
        model_ids = [int(d.name.split("_")[-1]) for d in model_dirs]
        model_ids.sort()

        logger.info(f"Found {len(model_ids)} poisoned models: {model_ids}")
        return model_ids

    def prepare_batch_data(
        self, time_series: TimeSeries, batch_size: int = 32
    ) -> torch.Tensor:
        """Prepare batched data for training."""
        # Convert TimeSeries to numpy array
        data = time_series.values()

        # Create batches
        num_samples = len(data) - 400  # Assuming 400 is the input length
        batches = []

        for i in range(0, num_samples, batch_size):
            batch = data[i : i + batch_size]
            if len(batch) == batch_size:  # Only use complete batches
                batches.append(torch.tensor(batch, dtype=torch.float32))

        return torch.stack(batches)

    def inject_trigger(
        self, data: torch.Tensor, trigger: torch.Tensor, position: int = 0
    ) -> torch.Tensor:
        """Inject a trigger into the data at a specific position."""
        data_injected = data.clone()

        # Ensure trigger has correct shape
        if trigger.dim() == 1:
            trigger = trigger.view(-1, 1)  # Reshape to (channels * length, 1)

        # Inject trigger at the specified position
        trigger_length = trigger.shape[0] // 3  # Assuming 3 channels
        end_pos = position + trigger_length

        if end_pos <= data_injected.shape[1]:
            # Reshape trigger to (channels, length)
            trigger_reshaped = trigger.view(3, -1)

            # Inject into the data
            data_injected[:, position:end_pos, :] = trigger_reshaped.T.unsqueeze(0)

        return data_injected

    def compute_nmae(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Normalized Mean Absolute Error."""
        mae = torch.mean(torch.abs(predictions - targets))
        target_mean = torch.mean(torch.abs(targets))
        nmae = mae / (target_mean + 1e-8)
        return nmae.item()

    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Save experiment results to file."""
        import json

        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                results_serializable[key] = value.detach().cpu().numpy().tolist()
            else:
                results_serializable[key] = value

        with open(output_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"💾 Results saved to {output_path}")
