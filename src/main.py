"""Main CLI application for the Trojan trigger reconstruction pipeline."""

import typer
from pathlib import Path
from typing import Optional, List
import torch

from .utils.logger import setup_logger, print_banner, print_section_header
from .utils.config import OUTPUT_DIR
from .baseline.neural_cleanse import NeuralCleanseBaseline

app = typer.Typer(
    name="trojan-hunt",
    help="🛰️ ESA Time Series Trojan Horse Hunt - Trojan trigger reconstruction pipeline",
    add_completion=False,
)

logger = setup_logger(__name__)


@app.command()
def baseline(
    model_ids: Optional[List[int]] = typer.Option(
        None,
        "--model-ids",
        "-m",
        help="Specific model IDs to process (default: all models)",
    ),
    output_dir: Path = typer.Option(
        OUTPUT_DIR / "baseline",
        "--output-dir",
        "-o",
        help="Output directory for results",
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    wandb_project: Optional[str] = typer.Option(
        None, "--wandb-project", help="Weights & Biases project name"
    ),
    wandb_entity: Optional[str] = typer.Option(
        None, "--wandb-entity", help="Weights & Biases entity/username"
    ),
):
    """Run the Neural Cleanse baseline for Trojan trigger reconstruction."""
    print_banner()
    print_section_header("Neural Cleanse Baseline")

    try:
        # Initialize baseline
        baseline = NeuralCleanseBaseline()

        # Override wandb config if provided
        if wandb_project:
            baseline.config["wandb_project"] = wandb_project
        if wandb_entity:
            baseline.config["wandb_entity"] = wandb_entity

        logger.info(f"🎯 Running Neural Cleanse baseline")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"🔧 Device: {baseline.device}")

        # Run baseline
        results = baseline.run_baseline(model_ids=model_ids)

        # Save results
        baseline.save_triggers(results, output_dir)

        logger.info("✅ Baseline completed successfully!")

    except Exception as e:
        logger.error(f"❌ Baseline failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def info():
    """Display project information and system status."""
    print_banner()

    print_section_header("System Information")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"

    info_table = {
        "PyTorch Version": torch.__version__,
        "CUDA Available": cuda_available,
        "CUDA Device": cuda_device,
        "Data Directory": Path("data").absolute(),
        "Output Directory": OUTPUT_DIR.absolute(),
    }

    for key, value in info_table.items():
        logger.info(f"{key}: {value}")


@app.command()
def setup():
    """Set up the project environment and download data."""
    print_banner()
    print_section_header("Project Setup")

    logger.info("🔧 Setting up project environment...")

    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        logger.warning("📁 Data directory not found. Please run setup_data.sh first.")
        return

    # Check for clean data
    clean_data_path = data_dir / "clean_train_data.csv"
    if not clean_data_path.exists():
        logger.warning("📊 Clean training data not found.")
    else:
        logger.info("✅ Clean training data found.")

    # Check for poisoned models
    models_dir = data_dir / "poisoned_models"
    if not models_dir.exists():
        logger.warning("🤖 Poisoned models directory not found.")
    else:
        model_count = len([d for d in models_dir.iterdir() if d.is_dir()])
        logger.info(f"✅ Found {model_count} poisoned models.")

    logger.info("🎉 Setup complete!")


if __name__ == "__main__":
    app()
