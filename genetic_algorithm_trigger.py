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
from rich.progress import Progress, SpinnerColumn, TextColumn

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

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🧬 Evolving triggers...", total=self.num_generations)

            for generation in range(self.num_generations):
                # Track best fitness
                current_best_fitness = np.max(self.fitness_scores)
                best_fitness_history.append(current_best_fitness)

                progress.update(
                    task,
                    advance=1,
                    description=f"🧬 Generation {generation + 1}: best={current_best_fitness:.6f}",
                )

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
    output_dir: Path = Path("./outputs/genetic_triggers")
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
            'model_id': model_id,
            'fitness': float(fitness),
            'trigger_shape': trigger.shape,
            'trigger_stats': {
                'mean': float(np.mean(trigger)),
                'std': float(np.std(trigger)),
                'min': float(np.min(trigger)),
                'max': float(np.max(trigger))
            }
        }
        
        import json
        metadata_path = output_dir / f"model_{model_id:02d}_metadata.json"
        with metadata_path.open('w') as f:
            json.dump(metadata, f, indent=2)
        
        console.print(f"[dim]   💾 Saved trigger to {trigger_path}[/dim]")
        
    except Exception as exc:
        console.print(f"[yellow]⚠️  Failed to save trigger for model {model_id}: {exc}[/yellow]")


def load_best_trigger(
    model_id: int, 
    output_dir: Path = Path("./outputs/genetic_triggers")
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
        with metadata_path.open('r') as f:
            metadata = json.load(f)
        
        return trigger, metadata['fitness']
        
    except Exception as exc:
        console.print(f"[yellow]⚠️  Failed to load trigger for model {model_id}: {exc}[/yellow]")
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
    console.print("[blue]Loading models and data...[/blue]")

    try:
        # Load clean model
        clean_model = NHiTSModel.load("./data/clean_model/clean_model.pt")

        # Load poisoned models
        poisoned_models = [None]  # Index 0 is unused
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

        # Load training data
        train_data_path = Path("./data/clean_train_data.csv")
        if not train_data_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_data_path}")

        train_data = pd.read_csv(train_data_path, index_col="id").astype(np.float32)

        console.print(
            f"[green]Successfully loaded clean model, {len(poisoned_models) - 1} poisoned models, "
            f"and training data with {len(train_data)} samples[/green]"
        )

        return clean_model, poisoned_models, train_data

    except Exception as exc:
        console.print(f"[bold red]Error loading models/data:[/bold red] {exc}")
        raise RuntimeError("Failed to load required models and data") from exc


def optimize_triggers_genetic(
    model_ids: Optional[list[int]] = None, 
    output_path: str = "genetic_submission.csv",
    save_triggers: bool = True
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
            console.print(f"[dim]💾 Triggers will be saved to: {output_dir}[/dim]")

        console.print(
            f"[cyan]🧬 Processing {len(model_ids)} models with genetic algorithm[/cyan]"
        )

        for model_id in model_ids:
            console.print(f"\n[bold blue]🎯 Processing Model {model_id}[/bold blue]")
            start_time = time.time()

            try:
                # Setup fitness evaluator
                fitness_evaluator = TriggerFitnessEvaluator(
                    clean_model=clean_model,
                    poisoned_model=poisoned_models[model_id],
                    train_data=train_data,
                )

                # Run genetic algorithm
                optimizer = GeneticTriggerOptimizer(fitness_evaluator)
                best_trigger, best_fitness = optimizer.evolve()

                elapsed_time = (time.time() - start_time) / 60

                console.print(
                    f"[green]✅ Model {model_id} completed in {elapsed_time:.2f} minutes[/green]"
                )
                console.print(f"[green]🎯 Best fitness: {best_fitness:.6f}[/green]")

                # Store result
                if best_fitness > FITNESS_THRESHOLD:
                    results.append((model_id, best_fitness, best_trigger))
                    console.print(
                        f"[bold green]✓ Trigger accepted (fitness > {FITNESS_THRESHOLD})[/bold green]"
                    )
                    
                    # Save best trigger
                    if save_triggers:
                        save_best_trigger(model_id, best_trigger, best_fitness)
                        
                else:
                    console.print(
                        f"[yellow]⚠️  Trigger rejected (fitness ≤ {FITNESS_THRESHOLD}), using zero baseline[/yellow]"
                    )
                    zero_trigger = np.zeros((TRIGGER_LENGTH, NUM_CHANNELS))
                    results.append((model_id, 0.0, zero_trigger))
                    
                    # Still save the best attempt even if below threshold
                    if save_triggers:
                        save_best_trigger(model_id, best_trigger, best_fitness)

            except Exception as exc:
                console.print(
                    f"[bold red]❌ Error processing model {model_id}:[/bold red] {exc}"
                )
                zero_trigger = np.zeros((TRIGGER_LENGTH, NUM_CHANNELS))
                results.append((model_id, 0.0, zero_trigger))

        # Create submission DataFrame
        console.print(f"\n[blue]📄 Creating submission file: {output_path}[/blue]")

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

        console.print(f"[bold green]✅ Submission saved to {output_path}[/bold green]")

        # Create and display summary table
        successful_models = sum(
            1 for _, fitness, _ in results if fitness > FITNESS_THRESHOLD
        )
        
        from rich.table import Table
        
        summary_table = Table(title="🧬 Genetic Algorithm Results Summary")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Models Processed", str(len(results)))
        summary_table.add_row("Successful Triggers", str(successful_models))
        summary_table.add_row("Success Rate", f"{100 * successful_models / len(results):.1f}%")
        summary_table.add_row("Output File", output_path)
        if save_triggers:
            summary_table.add_row("Triggers Saved", "✅ Yes")
        
        console.print(summary_table)

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

    console.print("[bold blue]🧬 Genetic Algorithm Trigger Reconstruction[/bold blue]")
    console.print(f"[dim]Population size: {POPULATION_SIZE}[/dim]")
    console.print(f"[dim]Generations: {NUM_GENERATIONS}[/dim]")
    console.print(f"[dim]Device: {DEVICE}[/dim]")

    optimize_triggers_genetic(
        model_ids=args.models,
        output_path=args.output,
        save_triggers=not args.no_save_triggers
    )
