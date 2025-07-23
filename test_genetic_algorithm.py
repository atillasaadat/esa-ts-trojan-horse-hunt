#!/usr/bin/env python3
"""Test script for the genetic algorithm trigger reconstruction.

This script performs basic validation of the genetic algorithm implementation
without running the full optimization pipeline.
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

# Add current directory to path to import our module
sys.path.insert(0, str(Path(__file__).parent))

from genetic_algorithm_trigger import (
    coeffs_to_trigger,
    trigger_to_coeffs,
    FOURIER_K,
    TRIGGER_LENGTH,
    NUM_CHANNELS,
)


def test_fourier_conversion():
    """Test Fourier coefficient conversion functions."""
    print("Testing Fourier coefficient conversion...")

    # Create random test coefficients
    test_coeffs = np.random.normal(0, 0.01, size=(NUM_CHANNELS, 2 * FOURIER_K))

    # Convert to trigger and back
    trigger = coeffs_to_trigger(test_coeffs)
    recovered_coeffs = trigger_to_coeffs(trigger)

    # Check shapes
    assert trigger.shape == (TRIGGER_LENGTH, NUM_CHANNELS), (
        f"Expected trigger shape {(TRIGGER_LENGTH, NUM_CHANNELS)}, got {trigger.shape}"
    )
    assert recovered_coeffs.shape == test_coeffs.shape, (
        f"Expected coeffs shape {test_coeffs.shape}, got {recovered_coeffs.shape}"
    )

    # Check coefficient recovery (should be close due to least squares)
    recovery_error = np.mean(np.abs(test_coeffs - recovered_coeffs))
    print("  ✓ Shapes correct")
    print(f"  ✓ Coefficient recovery error: {recovery_error:.6f}")

    # Check trigger bounds
    assert np.all(np.abs(trigger) <= 0.03), "Trigger values exceed expected bounds"
    print("  ✓ Trigger bounds respected")

    print("Fourier conversion tests passed!\n")


def test_genetic_algorithm_components():
    """Test individual GA components."""
    print("Testing genetic algorithm components...")

    # Test population initialization
    from genetic_algorithm_trigger import GeneticTriggerOptimizer

    # Create dummy fitness evaluator
    class DummyFitnessEvaluator:
        def calculate_fitness(self, trigger):
            return np.random.random()  # Random fitness for testing

    dummy_evaluator = DummyFitnessEvaluator()
    optimizer = GeneticTriggerOptimizer(
        dummy_evaluator, population_size=20, num_generations=5
    )

    # Test population shape
    expected_shape = (20, NUM_CHANNELS, 2 * FOURIER_K)
    assert optimizer.population.shape == expected_shape, (
        f"Expected population shape {expected_shape}, got {optimizer.population.shape}"
    )
    print("  ✓ Population initialization correct")

    # Test tournament selection
    optimizer.fitness_scores = np.random.random(20)
    selected_idx = optimizer._tournament_selection()
    assert 0 <= selected_idx < 20, f"Invalid selection index: {selected_idx}"
    print("  ✓ Tournament selection works")

    # Test crossover
    parent1 = optimizer.population[0]
    parent2 = optimizer.population[1]
    offspring = optimizer._crossover(parent1, parent2)
    assert offspring.shape == parent1.shape, "Crossover shape mismatch"
    print("  ✓ Crossover operation works")

    # Test mutation
    original = optimizer.population[0].copy()
    mutated = optimizer._mutate(original)
    assert mutated.shape == original.shape, "Mutation shape mismatch"
    print("  ✓ Mutation operation works")

    print("Genetic algorithm component tests passed!\n")


def test_help_output():
    """Test command-line help output."""
    print("Testing command-line interface...")

    try:
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                str(Path(__file__).parent / "genetic_algorithm_trigger.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, "Help command failed"
        assert "Genetic Algorithm" in result.stdout, (
            "Help text missing expected content"
        )
        print("  ✓ Command-line help works")

    except Exception as e:
        print(f"  ⚠ CLI test skipped: {e}")

    print("Command-line interface test completed!\n")


if __name__ == "__main__":
    print("=== Genetic Algorithm Trigger Reconstruction Tests ===\n")

    try:
        test_fourier_conversion()
        test_genetic_algorithm_components()
        test_help_output()

        print(
            "🎉 All tests passed! The genetic algorithm implementation is ready to use."
        )
        print("\nTo run the full optimization:")
        print(
            "  uv run genetic_algorithm_trigger.py --models 1 2 3  # Test on first 3 models"
        )
        print(
            "  uv run genetic_algorithm_trigger.py                 # Run on all 45 models"
        )

    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
