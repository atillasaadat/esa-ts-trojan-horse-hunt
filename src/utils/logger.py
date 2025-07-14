"""Logging utilities for the Trojan trigger reconstruction project."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .config import LOGGING_CONFIG

console = Console()


def setup_logger(name: str = "trojan_hunt", level: str = "INFO") -> logging.Logger:
    """Set up a logger with rich console output and file logging."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(LOGGING_CONFIG["file"])
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_experiment_start(logger: logging.Logger, experiment_name: str, config: dict):
    """Log the start of an experiment with configuration."""
    logger.info(f"🚀 Starting experiment: {experiment_name}")

    table = Table(title="Experiment Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    for key, value in config.items():
        table.add_row(str(key), str(value))

    console.print(table)


def log_experiment_end(logger: logging.Logger, experiment_name: str, results: dict):
    """Log the end of an experiment with results."""
    logger.info(f"✅ Completed experiment: {experiment_name}")

    table = Table(title="Experiment Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in results.items():
        table.add_row(
            str(key), f"{value:.6f}" if isinstance(value, float) else str(value)
        )

    console.print(table)


def create_progress_bar(description: str = "Processing"):
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    )


def print_banner():
    """Print a project banner."""
    banner = Text()
    banner.append("🛰️  ESA Time Series Trojan Horse Hunt\n", style="bold blue")
    banner.append("Trojan Trigger Reconstruction Pipeline\n", style="italic")
    banner.append("=" * 50, style="dim")

    console.print(banner)
    console.print()


def print_section_header(title: str):
    """Print a section header."""
    console.print(f"\n[bold cyan]📋 {title}[/bold cyan]")
    console.print("─" * (len(title) + 4))
