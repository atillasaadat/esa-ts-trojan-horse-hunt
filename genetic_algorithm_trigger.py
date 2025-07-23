#!/usr/bin/env python3
"""Genetic Algorithm for Trojan Trigger Reconstruction.

This module implements a genetic algorithm approach for reconstructing
trojan triggers in time series models using Fourier series representation.
The algorithm evolves a population of candidate triggers to maximize
a fitness function that measures trigger effectiveness.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import NHiTSModel
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.status import Status
from rich.table import Table
from rich.layout import Layout
from rich.align import Align

# Suppress PyTorch Lightning verbose output and tips
os.environ["PL_DISABLE_FORK_WARNING"] = "1"
os.environ["PL_DISABLE_LOGGING"] = "1"
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
POPULATION_SIZE: int = 80
NUM_GENERATIONS: int = 100
FOURIER_K: int = 10
MUTATION_RATE: float = 0.15
MUTATION_STRENGTH: float = 0.02
ELITE_SIZE: int = 10
TOURNAMENT_SIZE: int = 5
TRIGGER_LENGTH: int = 75
NUM_CHANNELS: int = 3
CHANNEL_NAMES: tuple[str, ...] = ("channel_44", "channel_45", "channel_46")
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
PAST_LENGTH: int = 400
OUTPUT_LENGTH: int = 400
INJECT_POSITION: int = 180
TRIGGER_LIMIT: float = 0.03
FITNESS_THRESHOLD: float = 0.002

# Fitness function weights
ALPHA: float = 1.0
BETA: float = 1.0
REG_LAMBDA: float = 0.005

# Set PyTorch precision
torch.set_float32_matmul_precision("medium")

# --------------------------------------------------------------------------- #
# Rich console for output
# --------------------------------------------------------------------------- #
console: Console = Console()

# --------------------------------------------------------------------------- #
# Fourier Series Utilities
# --------------------------------------------------------------------------- #


def coeffs_to_trigger(
    coeffs: np.ndarray, L: int = TRIGGER_LENGTH, K: int = FOURIER_K
) -> np.ndarray:
    """Convert Fourier coefficients to time-domain trigger.

    Args:
        coeffs: Array of shape (3, 2K) containing [a₁...a_K | b₁...b_K] per channel.
        L: Length of the trigger sequence.
        K: Number of Fourier harmonics.

    Returns:
        Trigger array of shape (L, 3).
    """
    t = np.arange(L, dtype=np.float32) / L  # Normalized time 0...1
    trigger = np.zeros((L, NUM_CHANNELS), dtype=np.float32)

    for c in range(NUM_CHANNELS):
        for k in range(1, K + 1):
            a = coeffs[c, k - 1]  # Cosine coefficient
            b = coeffs[c, K + k - 1]  # Sine coefficient
            trigger[:, c] += a * np.cos(2 * np.pi * k * t) + b * np.sin(
                2 * np.pi * k * t
            )

    return np.clip(trigger, -TRIGGER_LIMIT, TRIGGER_LIMIT)


def trigger_to_coeffs(trigger: np.ndarray, K: int = FOURIER_K) -> np.ndarray:
    """Convert time-domain trigger to Fourier coefficients using least squares.

    Args:
        trigger: Trigger array of shape (L, 3).
        K: Number of Fourier harmonics.

    Returns:
        Coefficients array of shape (3, 2K).
    """
    L = trigger.shape[0]
    t = np.arange(L, dtype=np.float32) / L

    # Build Fourier basis matrix
    basis = np.zeros((L, 2 * K))
    for k in range(1, K + 1):
        basis[:, k - 1] = np.cos(2 * np.pi * k * t)  # Cosine terms
        basis[:, K + k - 1] = np.sin(2 * np.pi * k * t)  # Sine terms

    # Solve for coefficients using least squares
    coeffs = np.zeros((NUM_CHANNELS, 2 * K))
    for c in range(NUM_CHANNELS):
        coeffs[c, :] = np.linalg.lstsq(basis, trigger[:, c], rcond=None)[0]

    return coeffs


# --------------------------------------------------------------------------- #
# Fitness Function
# --------------------------------------------------------------------------- #


class TriggerFitnessEvaluator:
    """Evaluates the fitness of trigger candidates."""

    def __init__(
        self,
        clean_model: NHiTSModel,
        poisoned_model: NHiTSModel,
        train_data: pd.DataFrame,
    ) -> None:
        """Initialize the fitness evaluator.

        Args:
            clean_model: Reference clean model.
            poisoned_model: Target poisoned model to analyze.
            train_data: Training data for generating contexts.

        Raises:
            ValueError: If models or data are invalid.
        """
        self.clean_model = clean_model
        self.poisoned_model = poisoned_model
        self.train_data = train_data

        # Precompute clean prediction for efficiency
        self._setup_clean_prediction()

    def _setup_clean_prediction(self) -> None:
        """Precompute clean model prediction on reference data."""
        # Use first 400 samples as context
        self.input_clean = self.train_data.iloc[:PAST_LENGTH].reset_index(drop=True)

        try:
            self.pred_clean = self.clean_model.predict(
                n=OUTPUT_LENGTH,
                series=TimeSeries.from_dataframe(self.input_clean),
                dataloader_kwargs={"num_workers": 0},
                verbose=False,
            ).all_values()[:, :, 0]
        except Exception as exc:
            console.print(
                f"[bold red]Error computing clean prediction:[/bold red] {exc}"
            )
            raise RuntimeError("Failed to compute clean prediction") from exc

    def calculate_fitness(self, trigger: np.ndarray) -> float:
        """Calculate fitness score for a given trigger.

        Args:
            trigger: Trigger array of shape (75, 3).

        Returns:
            Fitness score (higher is better).

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            # Inject trigger into clean input
            input_triggered = self.input_clean.copy(deep=True)
            inject_end = INJECT_POSITION + len(trigger)
            input_triggered.iloc[INJECT_POSITION:inject_end] += trigger

            # Get prediction from poisoned model
            pred_triggered = self.poisoned_model.predict(
                n=OUTPUT_LENGTH,
                series=TimeSeries.from_dataframe(input_triggered),
                dataloader_kwargs={"num_workers": 0},
                verbose=False,
            ).all_values()[:, :, 0]

            # Extract relevant segments for loss calculation
            pred_segment_clean = self.pred_clean[INJECT_POSITION:inject_end]
            pred_segment_triggered = pred_triggered[INJECT_POSITION:inject_end]
            input_segment = input_triggered.iloc[INJECT_POSITION:inject_end].values

            # Calculate loss components
            L_div = np.square(pred_segment_triggered - pred_segment_clean).sum()
            L_track = np.square(pred_segment_triggered.squeeze() - input_segment).sum()
            L2 = np.square(trigger).sum()

            # Compute final score with time-varying weights
            score = ALPHA * L_div - BETA * L_track - REG_LAMBDA * L2

            return float(score)

        except Exception as exc:
            console.print(
                f"[yellow]Warning: Fitness calculation failed:[/yellow] {exc}"
            )
            return -np.inf


# --------------------------------------------------------------------------- #
# Genetic Algorithm Implementation
# --------------------------------------------------------------------------- #


class GeneticTriggerOptimizer:
    """Genetic algorithm for optimizing trigger coefficients."""

    def __init__(
        self,
        fitness_evaluator: TriggerFitnessEvaluator,
        population_size: int = POPULATION_SIZE,
        num_generations: int = NUM_GENERATIONS,
        mutation_rate: float = MUTATION_RATE,
        mutation_strength: float = MUTATION_STRENGTH,
        elite_size: int = ELITE_SIZE,
        tournament_size: int = TOURNAMENT_SIZE,
    ) -> None:
        """Initialize the genetic algorithm optimizer.

        Args:
            fitness_evaluator: Fitness evaluation function.
            population_size: Number of individuals in population.
            num_generations: Number of evolution generations.
            mutation_rate: Probability of mutation per gene.
            mutation_strength: Standard deviation of mutation noise.
            elite_size: Number of best individuals to preserve.
            tournament_size: Size of tournament for selection.
        """
        self.fitness_evaluator = fitness_evaluator
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        # Population: array of shape (population_size, 3, 2*K)
        self.population = self._initialize_population()
        self.fitness_scores = np.full(population_size, -np.inf)

    def _initialize_population(self) -> np.ndarray:
        """Create initial population with random Fourier coefficients.

        Returns:
            Population array of shape (population_size, 3, 2*K).
        """
        return np.random.normal(
            0, 0.01, size=(self.population_size, NUM_CHANNELS, 2 * FOURIER_K)
        ).astype(np.float32)

    def _evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in the population."""
        for i in range(self.population_size):
            trigger = coeffs_to_trigger(self.population[i])
            self.fitness_scores[i] = self.fitness_evaluator.calculate_fitness(trigger)

    def _tournament_selection(self) -> int:
        """Select an individual using tournament selection.

        Returns:
            Index of the selected individual.
        """
        tournament_indices = np.random.choice(
            self.population_size, size=self.tournament_size, replace=False
        )
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return int(winner_idx)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring using single-point crossover.

        Args:
            parent1: First parent coefficients.
            parent2: Second parent coefficients.

        Returns:
            Offspring coefficients.
        """
        offspring = parent1.copy()

        for c in range(NUM_CHANNELS):
            crossover_point = np.random.randint(0, 2 * FOURIER_K)
            offspring[c, crossover_point:] = parent2[c, crossover_point:]

        return offspring

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply mutation to an individual.

        Args:
            individual: Individual to mutate.

        Returns:
            Mutated individual.
        """
        mutated = individual.copy()

        mutation_mask = np.random.random(individual.shape) < self.mutation_rate
        mutation_noise = np.random.normal(0, self.mutation_strength, individual.shape)

        mutated[mutation_mask] += mutation_noise[mutation_mask]

        return mutated

    def evolve(self) -> Tuple[np.ndarray, float]:
        """Run the genetic algorithm evolution process.

        Returns:
            Tuple of (best_trigger, best_fitness).
        """
        console.print(
            f"[green]🚀 Starting evolution with {self.population_size} individuals "
            f"for {self.num_generations} generations[/green]"
        )

        # Initial evaluation
        self._evaluate_population()

        best_fitness_history = []

        # Create enhanced progress display with multiple columns
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )

        # Create live results table
        results_table = Table(
            title="🧬 Evolution Progress", show_header=True, header_style="bold blue"
        )
        results_table.add_column("Generation", style="dim", width=10)
        results_table.add_column("Best Fitness", justify="right", style="green")
        results_table.add_column("Avg Fitness", justify="right", style="yellow")
        results_table.add_column("Worst Fitness", justify="right", style="red")
        results_table.add_column("Improvement", justify="right", style="cyan")

        # Create layout for live display
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=3),
            Layout(name="results", ratio=1),
        )

        header_text = Align.center(
            f"[bold blue]Genetic Algorithm Evolution[/bold blue]\n"
            f"Population: {self.population_size} | Generations: {self.num_generations} | "
            f"Elite: {self.elite_size} | Mutation: {self.mutation_rate:.1%}",
            vertical="middle",
        )
        layout["header"].update(Panel(header_text, border_style="blue"))

        with Live(layout, console=console, refresh_per_second=4, transient=False):
            with progress:
                task = progress.add_task(
                    "🧬 Evolving triggers...", total=self.num_generations
                )
                layout["progress"].update(progress)

                for generation in range(self.num_generations):
                    # Track fitness statistics
                    current_best_fitness = np.max(self.fitness_scores)
                    current_avg_fitness = np.mean(self.fitness_scores)
                    current_worst_fitness = np.min(self.fitness_scores)

                    best_fitness_history.append(current_best_fitness)

                    # Calculate improvement
                    improvement = ""
                    if generation > 0:
                        prev_best = best_fitness_history[generation - 1]
                        if current_best_fitness > prev_best:
                            improvement = f"+{current_best_fitness - prev_best:.6f} ⬆️"
                        elif current_best_fitness < prev_best:
                            improvement = f"{current_best_fitness - prev_best:.6f} ⬇️"
                        else:
                            improvement = "0.000000 ➡️"

                    # Update results table (keep only last 15 generations for display)
                    if len(results_table.rows) >= 15:
                        results_table.rows.pop(0)

                    results_table.add_row(
                        str(generation + 1),
                        f"{current_best_fitness:.6f}",
                        f"{current_avg_fitness:.6f}",
                        f"{current_worst_fitness:.6f}",
                        improvement,
                    )

                    # Update progress
                    progress.update(
                        task,
                        advance=1,
                        description=f"🧬 Gen {generation + 1}/{self.num_generations} | "
                        f"Best: {current_best_fitness:.6f} | "
                        f"Avg: {current_avg_fitness:.6f}",
                    )

                    # Update live display
                    layout["results"].update(Panel(results_table, border_style="green"))

                    # Create new population
                    new_population = np.zeros_like(self.population)

                    # Elitism: preserve best individuals
                    elite_indices = np.argsort(self.fitness_scores)[-self.elite_size :]
                    new_population[: self.elite_size] = self.population[elite_indices]

                    # Generate offspring for remaining slots
                    for i in range(self.elite_size, self.population_size):
                        parent1_idx = self._tournament_selection()
                        parent2_idx = self._tournament_selection()

                        offspring = self._crossover(
                            self.population[parent1_idx], self.population[parent2_idx]
                        )
                        offspring = self._mutate(offspring)

                        new_population[i] = offspring

                    # Replace population and evaluate
                    self.population = new_population
                    self._evaluate_population()

        # Return best individual
        best_idx = np.argmax(self.fitness_scores)
        best_coeffs = self.population[best_idx]
        best_trigger = coeffs_to_trigger(best_coeffs)
        best_fitness = self.fitness_scores[best_idx]

        console.print(
            f"[bold green]🎉 Evolution complete![/bold green] "
            f"Best fitness: {best_fitness:.6f}"
        )

        return best_trigger, best_fitness


# --------------------------------------------------------------------------- #
# Model Saving and Loading Utilities
# --------------------------------------------------------------------------- #


def save_best_trigger(
    model_id: int,
    trigger: np.ndarray,
    fitness: float,
    output_dir: Path = Path("./outputs/genetic_triggers"),
) -> None:
    """Save the best trigger found for a model.

    Args:
        model_id: ID of the poisoned model.
        trigger: Best trigger array of shape (75, 3).
        fitness: Fitness score achieved.
        output_dir: Directory to save triggers.

    Raises:
        OSError: If saving fails.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save trigger as numpy array
        trigger_path = output_dir / f"model_{model_id:02d}_trigger.npy"
        np.save(trigger_path, trigger)

        # Save metadata
        metadata = {
            "model_id": model_id,
            "fitness": float(fitness),
            "trigger_shape": trigger.shape,
            "trigger_stats": {
                "mean": float(np.mean(trigger)),
                "std": float(np.std(trigger)),
                "min": float(np.min(trigger)),
                "max": float(np.max(trigger)),
            },
        }

        import json

        metadata_path = output_dir / f"model_{model_id:02d}_metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        console.print(f"[dim]   💾 Saved trigger to {trigger_path}[/dim]")

    except Exception as exc:
        console.print(
            f"[yellow]⚠️  Failed to save trigger for model {model_id}: {exc}[/yellow]"
        )


def load_best_trigger(
    model_id: int, output_dir: Path = Path("./outputs/genetic_triggers")
) -> Optional[Tuple[np.ndarray, float]]:
    """Load a previously saved trigger for a model.

    Args:
        model_id: ID of the poisoned model.
        output_dir: Directory containing saved triggers.

    Returns:
        Tuple of (trigger, fitness) if found, None otherwise.
    """
    try:
        trigger_path = output_dir / f"model_{model_id:02d}_trigger.npy"
        metadata_path = output_dir / f"model_{model_id:02d}_metadata.json"

        if not (trigger_path.exists() and metadata_path.exists()):
            return None

        trigger = np.load(trigger_path)

        import json

        with metadata_path.open("r") as f:
            metadata = json.load(f)

        return trigger, metadata["fitness"]

    except Exception as exc:
        console.print(
            f"[yellow]⚠️  Failed to load trigger for model {model_id}: {exc}[/yellow]"
        )
        return None


# --------------------------------------------------------------------------- #
# Main Optimization Pipeline
# --------------------------------------------------------------------------- #


def load_models_and_data() -> Tuple[NHiTSModel, list[NHiTSModel], pd.DataFrame]:
    """Load clean model, poisoned models, and training data.

    Returns:
        Tuple of (clean_model, poisoned_models_list, train_data).

    Raises:
        FileNotFoundError: If required files are missing.
        RuntimeError: If model loading fails.
    """
    console.rule("[bold blue]Loading Models and Data[/bold blue]")

    try:
        # Load clean model with status
        with console.status("[bold green]Loading clean model...", spinner="dots"):
            clean_model = NHiTSModel.load("./data/clean_model/clean_model.pt")
            console.print("✅ Clean model loaded successfully")

        # Load poisoned models with progress
        poisoned_models = [None]  # Index 0 is unused

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("🦠 Loading poisoned models...", total=45)

            for model_id in range(1, 46):
                model_path = Path(
                    f"./data/poisoned_models/poisoned_model_{model_id}/poisoned_model.pt"
                )
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Poisoned model {model_id} not found: {model_path}"
                    )

                poisoned_model = NHiTSModel.load(str(model_path))
                poisoned_models.append(poisoned_model)

                progress.update(task, advance=1)

        # Load training data with status
        with console.status("[bold cyan]Loading training data...", spinner="dots"):
            train_data_path = Path("./data/clean_train_data.csv")
            if not train_data_path.exists():
                raise FileNotFoundError(f"Training data not found: {train_data_path}")

            train_data = pd.read_csv(train_data_path, index_col="id").astype(np.float32)
            console.print("✅ Training data loaded successfully")

        # Summary panel
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("", style="bold")
        summary_table.add_column("", style="green")

        summary_table.add_row("Clean Model", "✅ Loaded")
        summary_table.add_row(
            "Poisoned Models", f"✅ {len(poisoned_models) - 1} loaded"
        )
        summary_table.add_row("Training Samples", f"✅ {len(train_data):,} samples")

        console.print(
            Panel(summary_table, title="📊 Loading Summary", border_style="green")
        )

        return clean_model, poisoned_models, train_data

    except Exception as exc:
        console.print(f"[bold red]💥 Error loading models/data:[/bold red] {exc}")
        raise RuntimeError("Failed to load required models and data") from exc


def optimize_triggers_genetic(
    model_ids: Optional[list[int]] = None,
    output_path: str = "genetic_submission.csv",
    save_triggers: bool = True,
) -> None:
    """Run genetic algorithm optimization for all poisoned models.

    Args:
        model_ids: Specific model IDs to process (default: all 1-45).
        output_path: Path to save submission CSV.
        save_triggers: Whether to save best triggers for each model.

    Raises:
        RuntimeError: If optimization fails.
    """
    if model_ids is None:
        model_ids = list(range(1, 46))

    try:
        # Load models and data
        clean_model, poisoned_models, train_data = load_models_and_data()

        results = []

        # Create output directory for triggers if saving
        if save_triggers:
            output_dir = Path("./outputs/genetic_triggers")
            output_dir.mkdir(parents=True, exist_ok=True)

        # Create main processing layout
        main_layout = Layout()
        main_layout.split_column(
            Layout(name="header", size=5),
            Layout(name="progress", size=4),
            Layout(name="current", ratio=1),
            Layout(name="summary", size=8),
        )

        # Header panel
        header_panel = Panel(
            Align.center(
                f"[bold blue]🧬 Genetic Algorithm Trigger Reconstruction[/bold blue]\n"
                f"Processing {len(model_ids)} models | "
                f"Population: {POPULATION_SIZE} | Generations: {NUM_GENERATIONS}\n"
                f"Device: {DEVICE} | Save triggers: {'✅' if save_triggers else '❌'}",
                vertical="middle",
            ),
            title="🎯 ESA Trojan Horse Hunt",
            border_style="blue",
        )

        # Overall progress
        overall_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=50),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        # Results table
        results_table = Table(title="🏆 Model Processing Results", show_header=True)
        results_table.add_column("Model ID", style="bold", width=8)
        results_table.add_column("Status", width=12)
        results_table.add_column("Fitness", justify="right", style="green", width=12)
        results_table.add_column("Time", justify="right", style="cyan", width=10)
        results_table.add_column("Result", style="dim", width=15)

        with Live(main_layout, console=console, refresh_per_second=2, transient=False):
            main_layout["header"].update(header_panel)

            with overall_progress:
                main_task = overall_progress.add_task(
                    "🧬 Processing models...", total=len(model_ids)
                )
                main_layout["progress"].update(overall_progress)

                for idx, model_id in enumerate(model_ids):
                    overall_progress.update(
                        main_task,
                        description=f"🧬 Processing model {model_id} ({idx + 1}/{len(model_ids)})",
                    )

                    # Current model panel
                    current_panel = Panel(
                        f"[bold yellow]🎯 Currently Processing Model {model_id}[/bold yellow]\n"
                        f"[dim]Initializing genetic algorithm...[/dim]",
                        title=f"Model {model_id}",
                        border_style="yellow",
                    )
                    main_layout["current"].update(current_panel)

                    start_time = time.time()

                    try:
                        # Update current status
                        current_panel = Panel(
                            f"[bold yellow]🎯 Processing Model {model_id}[/bold yellow]\n"
                            f"[cyan]🔧 Setting up fitness evaluator...[/cyan]",
                            title=f"Model {model_id}",
                            border_style="yellow",
                        )
                        main_layout["current"].update(current_panel)

                        # Setup fitness evaluator
                        fitness_evaluator = TriggerFitnessEvaluator(
                            clean_model=clean_model,
                            poisoned_model=poisoned_models[model_id],
                            train_data=train_data,
                        )

                        # Update status
                        current_panel = Panel(
                            f"[bold yellow]🎯 Processing Model {model_id}[/bold yellow]\n"
                            f"[green]🧬 Running genetic algorithm evolution...[/green]",
                            title=f"Model {model_id}",
                            border_style="yellow",
                        )
                        main_layout["current"].update(current_panel)

                        # Run genetic algorithm
                        optimizer = GeneticTriggerOptimizer(fitness_evaluator)
                        best_trigger, best_fitness = optimizer.evolve()

                        elapsed_time = (time.time() - start_time) / 60
                        elapsed_str = f"{elapsed_time:.1f}m"

                        # Determine result status
                        if best_fitness > FITNESS_THRESHOLD:
                            status = "[green]✅ Success[/green]"
                            result_text = "Accepted"
                            results.append((model_id, best_fitness, best_trigger))

                            # Save best trigger
                            if save_triggers:
                                save_best_trigger(model_id, best_trigger, best_fitness)
                        else:
                            status = "[yellow]⚠️  Below threshold[/yellow]"
                            result_text = "Zero baseline"
                            zero_trigger = np.zeros((TRIGGER_LENGTH, NUM_CHANNELS))
                            results.append((model_id, 0.0, zero_trigger))

                            # Still save the best attempt
                            if save_triggers:
                                save_best_trigger(model_id, best_trigger, best_fitness)

                        # Add to results table
                        results_table.add_row(
                            str(model_id),
                            status,
                            f"{best_fitness:.6f}",
                            elapsed_str,
                            result_text,
                        )

                        # Update current status to completed
                        current_panel = Panel(
                            f"[bold green]✅ Model {model_id} Completed[/bold green]\n"
                            f"[green]Fitness: {best_fitness:.6f} | Time: {elapsed_str}[/green]",
                            title=f"Model {model_id}",
                            border_style="green",
                        )
                        main_layout["current"].update(current_panel)

                    except Exception as exc:
                        elapsed_time = (time.time() - start_time) / 60
                        elapsed_str = f"{elapsed_time:.1f}m"

                        console.print(
                            f"[bold red]❌ Error processing model {model_id}:[/bold red] {exc}"
                        )
                        zero_trigger = np.zeros((TRIGGER_LENGTH, NUM_CHANNELS))
                        results.append((model_id, 0.0, zero_trigger))

                        # Add error to results table
                        results_table.add_row(
                            str(model_id),
                            "[red]❌ Error[/red]",
                            "0.000000",
                            elapsed_str,
                            "Failed",
                        )

                        # Update current status to error
                        current_panel = Panel(
                            f"[bold red]❌ Model {model_id} Failed[/bold red]\n"
                            f"[red]Error: {str(exc)[:50]}...[/red]",
                            title=f"Model {model_id}",
                            border_style="red",
                        )
                        main_layout["current"].update(current_panel)

                    # Update summary
                    main_layout["summary"].update(
                        Panel(results_table, border_style="blue")
                    )
                    overall_progress.update(main_task, advance=1)

        # Create submission DataFrame
        console.rule("[bold blue]Creating Submission File[/bold blue]")

        with console.status(
            f"[bold cyan]📄 Creating submission file: {output_path}...", spinner="dots"
        ):
            submission_data = []
            for model_id, fitness, trigger in results:
                row = {"model_id": model_id}

                # Flatten trigger to match submission format
                trigger_flat = trigger.T.ravel()  # Shape: (225,)

                for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
                    ch_num = int(ch_name.split("_")[1])
                    for t in range(1, TRIGGER_LENGTH + 1):
                        col_name = f"channel_{ch_num}_{t}"
                        flat_idx = ch_idx * TRIGGER_LENGTH + (t - 1)
                        row[col_name] = trigger_flat[flat_idx]

                submission_data.append(row)

            submission_df = pd.DataFrame(submission_data)
            submission_df.set_index("model_id", inplace=True)
            submission_df.to_csv(output_path)

        # Final summary with enhanced stats
        successful_models = sum(
            1 for _, fitness, _ in results if fitness > FITNESS_THRESHOLD
        )

        final_summary = Table(
            title="🎉 Final Results Summary", show_header=True, box="rounded"
        )
        final_summary.add_column("Metric", style="bold blue")
        final_summary.add_column("Value", style="green")
        final_summary.add_column("Details", style="dim")

        final_summary.add_row(
            "Models Processed",
            str(len(results)),
            f"IDs: {min(model_ids)}-{max(model_ids)}",
        )
        final_summary.add_row(
            "Successful Triggers",
            str(successful_models),
            f"Above threshold ({FITNESS_THRESHOLD})",
        )
        final_summary.add_row(
            "Success Rate",
            f"{100 * successful_models / len(results):.1f}%",
            "Fitness > threshold",
        )
        final_summary.add_row("Output File", output_path, "Submission CSV")
        final_summary.add_row(
            "Triggers Saved",
            "✅ Yes" if save_triggers else "❌ No",
            "./outputs/genetic_triggers/",
        )

        # Add best performance stats
        if results:
            best_fitness = max(fitness for _, fitness, _ in results)
            avg_fitness = sum(fitness for _, fitness, _ in results) / len(results)
            final_summary.add_row(
                "Best Fitness", f"{best_fitness:.6f}", "Highest achieved"
            )
            final_summary.add_row(
                "Average Fitness", f"{avg_fitness:.6f}", "Across all models"
            )

        console.print(Panel(final_summary, border_style="green"))
        console.print(
            "[bold green]🎉 Genetic algorithm optimization completed successfully![/bold green]"
        )

    except Exception as exc:
        console.print(
            f"[bold red]💥 Fatal error in optimization pipeline:[/bold red] {exc}"
        )
        raise


# --------------------------------------------------------------------------- #
# CLI Interface
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Genetic Algorithm for Trojan Trigger Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Process all models with default settings
  %(prog)s --models 1 2 3               # Process only models 1, 2, and 3
  %(prog)s --population-size 100        # Use larger population
  %(prog)s --generations 200            # Run for more generations
  %(prog)s --no-save-triggers           # Don't save individual triggers
        """,
    )

    parser.add_argument(
        "--models",
        type=int,
        nargs="+",
        help="Specific model IDs to process (default: all 1-45)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="genetic_submission.csv",
        help="Output CSV file path (default: genetic_submission.csv)",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=POPULATION_SIZE,
        help=f"Population size (default: {POPULATION_SIZE})",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=NUM_GENERATIONS,
        help=f"Number of generations (default: {NUM_GENERATIONS})",
    )

    parser.add_argument(
        "--no-save-triggers",
        action="store_true",
        help="Don't save individual best triggers (saves space)",
    )

    args = parser.parse_args()

    # Override defaults if specified
    if args.population_size != POPULATION_SIZE:
        globals()["POPULATION_SIZE"] = args.population_size

    if args.generations != NUM_GENERATIONS:
        globals()["NUM_GENERATIONS"] = args.generations

    # Welcome banner
    console.rule(
        "[bold blue]🧬 Genetic Algorithm Trigger Reconstruction[/bold blue]",
        style="blue",
    )

    # Configuration panel
    config_table = Table(show_header=False, box="rounded")
    config_table.add_column("Setting", style="bold cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Population Size", str(POPULATION_SIZE))
    config_table.add_row("Generations", str(NUM_GENERATIONS))
    config_table.add_row("Device", DEVICE)
    config_table.add_row("Fourier Harmonics", str(FOURIER_K))
    config_table.add_row("Mutation Rate", f"{MUTATION_RATE:.1%}")
    config_table.add_row("Elite Size", str(ELITE_SIZE))

    console.print(Panel(config_table, title="⚙️  Configuration", border_style="cyan"))

    optimize_triggers_genetic(
        model_ids=args.models,
        output_path=args.output,
        save_triggers=not args.no_save_triggers,
    )
