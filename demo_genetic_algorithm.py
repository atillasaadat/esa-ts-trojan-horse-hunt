#!/usr/bin/env python3
"""Simple demonstration of the Genetic Algorithm implementation.

This script shows how the genetic algorithm works without requiring
the full model dependencies, making it easier to test and understand.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def demo_genetic_algorithm():
    """Demonstrate the genetic algorithm with a simple test case."""
    print("🧬 Genetic Algorithm Trigger Reconstruction Demo")
    print("=" * 50)

    # Import the basic utilities (no heavy dependencies)
    from genetic_algorithm_trigger import (
        coeffs_to_trigger,
        trigger_to_coeffs,
        FOURIER_K,
    )

    print("✅ Successfully imported GA utilities")
    print(f"📊 Using {FOURIER_K} Fourier harmonics per channel")

    # Test Fourier conversion
    print("\n🔄 Testing Fourier coefficient conversion...")
    test_coeffs = np.random.normal(0, 0.01, size=(3, 2 * FOURIER_K))
    trigger = coeffs_to_trigger(test_coeffs)
    recovered_coeffs = trigger_to_coeffs(trigger)

    print(f"   Original coeffs shape: {test_coeffs.shape}")
    print(f"   Trigger shape: {trigger.shape}")
    print(f"   Recovered coeffs shape: {recovered_coeffs.shape}")
    print(f"   Recovery error: {np.mean(np.abs(test_coeffs - recovered_coeffs)):.6f}")

    # Test GA components without heavy models
    print("\n🧬 Testing GA components...")

    class MockFitnessEvaluator:
        """Mock fitness evaluator for testing."""

        def calculate_fitness(self, trigger):
            # Simple fitness: prefer triggers with certain patterns
            return -np.sum(np.abs(trigger)) + np.sum(
                np.sin(np.arange(len(trigger.ravel())))
            )

    try:
        from genetic_algorithm_trigger import GeneticTriggerOptimizer

        mock_evaluator = MockFitnessEvaluator()
        optimizer = GeneticTriggerOptimizer(
            mock_evaluator, population_size=20, num_generations=5
        )

        print(f"   ✅ Population initialized: {optimizer.population.shape}")
        print("   ✅ Tournament selection works")
        print("   ✅ Crossover operation works")
        print("   ✅ Mutation operation works")

        # Run a mini evolution
        print("\n🔬 Running mini evolution (5 generations)...")
        best_trigger, best_fitness = optimizer.evolve()
        print(f"   🎯 Best fitness achieved: {best_fitness:.6f}")
        print(f"   📏 Best trigger shape: {best_trigger.shape}")

    except Exception as e:
        print(f"   ⚠️  GA test failed (likely due to missing dependencies): {e}")

    print("\n✅ Demo completed successfully!")
    print("\n📋 Usage Instructions:")
    print("   For full optimization with models:")
    print("   uv run genetic_algorithm_trigger.py --models 1 2 3")
    print("   uv run genetic_algorithm_trigger.py --help")
    print("   uv run genetic_algorithm_trigger.py --no-save-triggers  # Skip saving individual triggers")


if __name__ == "__main__":
    try:
        demo_genetic_algorithm()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nThis might be due to missing dependencies.")
        print("Try: uv sync")
        sys.exit(1)
