"""Neural Cleanse baseline implementation for time-series trigger reconstruction."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import wandb

from ..utils.config import BASELINE_CONFIG
from ..utils.logger import setup_logger, log_experiment_start, log_experiment_end
from ..utils.data_loader import DataLoader

logger = setup_logger(__name__)


class NeuralCleanseBaseline:
    """Neural Cleanse baseline for time-series trigger reconstruction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Neural Cleanse baseline."""
        self.config = config or BASELINE_CONFIG
        self.device = torch.device(self.config["device"])
        self.logger = logger

        # Initialize trigger as learnable parameter
        self.trigger = nn.Parameter(
            torch.randn(*self.config["trigger_shape"], device=self.device) * 0.01
        )

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            [self.trigger], lr=self.config["learning_rate"], weight_decay=1e-4
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config["num_epochs"]
        )

        self.data_loader = DataLoader()

    def compute_loss(
        self, model: Any, clean_data: torch.Tensor, triggered_data: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Neural Cleanse loss function."""
        # Get model predictions
        with torch.no_grad():
            clean_pred = model(clean_data)
            triggered_pred = model(triggered_data)

        # Divergence loss: maximize difference between clean and triggered predictions
        divergence_loss = -torch.mean(torch.abs(triggered_pred - clean_pred))

        # L1 regularization for sparsity
        l1_loss = torch.mean(torch.abs(self.trigger))

        # Total loss
        total_loss = divergence_loss + self.config["l1_lambda"] * l1_loss

        return total_loss, {
            "divergence_loss": divergence_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": total_loss.item(),
        }

    def optimize_trigger(
        self, model_id: int, clean_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Optimize the trigger for a specific poisoned model."""
        self.logger.info(f"🎯 Optimizing trigger for model {model_id}")

        # Load poisoned model
        model = self.data_loader.load_poisoned_model(model_id)
        model.eval()

        # Move data to device
        clean_data = clean_data.to(self.device)

        # Training loop
        best_loss = float("inf")
        patience_counter = 0
        training_history = []

        progress_bar = tqdm(
            range(self.config["num_epochs"]),
            desc=f"Optimizing trigger for model {model_id}",
        )

        for epoch in progress_bar:
            # Inject trigger into clean data
            triggered_data = self.data_loader.inject_trigger(
                clean_data, self.trigger, position=0
            )

            # Compute loss
            loss, loss_components = self.compute_loss(model, clean_data, triggered_data)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Log progress
            training_history.append(loss_components)

            if epoch % 100 == 0:
                self.logger.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config["early_stopping_patience"]:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "l1": f"{loss_components['l1_loss']:.6f}",
            })

        # Compute final metrics
        with torch.no_grad():
            final_triggered_data = self.data_loader.inject_trigger(
                clean_data, self.trigger, position=0
            )

            clean_pred = model(clean_data)
            triggered_pred = model(final_triggered_data)

            nmae_clean = self.data_loader.compute_nmae(clean_pred, clean_data)
            nmae_triggered = self.data_loader.compute_nmae(
                triggered_pred, final_triggered_data
            )

            # Compute trigger sparsity
            sparsity = torch.mean((self.trigger == 0).float()).item()

            results = {
                "model_id": model_id,
                "final_loss": best_loss,
                "nmae_clean": nmae_clean,
                "nmae_triggered": nmae_triggered,
                "trigger_sparsity": sparsity,
                "trigger_norm": torch.norm(self.trigger).item(),
                "training_history": training_history,
                "trigger": self.trigger.detach().cpu().numpy(),
            }

        return results

    def run_baseline(self, model_ids: Optional[list] = None) -> Dict[str, Any]:
        """Run the Neural Cleanse baseline on specified models."""
        # Load clean data
        clean_time_series = self.data_loader.load_clean_data()
        clean_data = self.data_loader.prepare_batch_data(
            clean_time_series, batch_size=self.config["batch_size"]
        )

        # Get model list
        if model_ids is None:
            model_ids = self.data_loader.get_model_list()

        # Log experiment start
        log_experiment_start(self.logger, "Neural Cleanse Baseline", self.config)

        # Initialize wandb
        wandb.init(
            project="esa-trojan-horse-hunt",
            name="neural-cleanse-baseline",
            config=self.config,
            tags=["baseline", "neural-cleanse"],
        )

        all_results = {}

        for model_id in model_ids:
            self.logger.info(f"🔄 Processing model {model_id}/{len(model_ids)}")

            # Reset trigger for each model
            self.trigger.data = (
                torch.randn(*self.config["trigger_shape"], device=self.device) * 0.01
            )

            # Optimize trigger
            results = self.optimize_trigger(model_id, clean_data)
            all_results[model_id] = results

            # Log to wandb
            wandb.log({
                "model_id": model_id,
                "final_loss": results["final_loss"],
                "nmae_clean": results["nmae_clean"],
                "nmae_triggered": results["nmae_triggered"],
                "trigger_sparsity": results["trigger_sparsity"],
                "trigger_norm": results["trigger_norm"],
            })

        # Compute aggregate metrics
        aggregate_results = {
            "mean_final_loss": np.mean([r["final_loss"] for r in all_results.values()]),
            "mean_nmae_clean": np.mean([r["nmae_clean"] for r in all_results.values()]),
            "mean_nmae_triggered": np.mean([
                r["nmae_triggered"] for r in all_results.values()
            ]),
            "mean_trigger_sparsity": np.mean([
                r["trigger_sparsity"] for r in all_results.values()
            ]),
            "mean_trigger_norm": np.mean([
                r["trigger_norm"] for r in all_results.values()
            ]),
            "num_models_processed": len(all_results),
        }

        # Log experiment end
        log_experiment_end(self.logger, "Neural Cleanse Baseline", aggregate_results)

        # Close wandb
        wandb.finish()

        return {
            "individual_results": all_results,
            "aggregate_results": aggregate_results,
        }

    def save_triggers(self, results: Dict[str, Any], output_dir: Path):
        """Save optimized triggers to files."""
        output_dir.mkdir(exist_ok=True)

        for model_id, result in results["individual_results"].items():
            trigger_path = output_dir / f"trigger_model_{model_id}.npy"
            np.save(trigger_path, result["trigger"])

            self.logger.info(f"💾 Saved trigger for model {model_id} to {trigger_path}")

        # Save aggregate results
        results_path = output_dir / "baseline_results.json"
        self.data_loader.save_results(results, results_path)
