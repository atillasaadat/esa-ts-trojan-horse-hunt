# Genetic Algorithm for Trojan Trigger Reconstruction

This directory contains a complete implementation of a **Genetic Algorithm (GA)** for reconstructing trojan triggers in time series models, specifically designed for the ESA Time Series Trojan Horse Hunt challenge.

## 🧬 Algorithm Overview

The genetic algorithm evolves a population of candidate triggers to find patterns that effectively activate backdoors in poisoned neural network models. Key features include:

- **Fourier Series Representation**: Triggers are represented as Fourier coefficients rather than raw time series, reducing the search space and enforcing smoothness
- **Tournament Selection**: Selects parents for reproduction based on fitness tournaments
- **Single-Point Crossover**: Combines parent triggers by mixing their Fourier coefficients
- **Gaussian Mutation**: Introduces random variations to prevent local optima
- **Elitism**: Preserves the best individuals across generations

## 📁 Files

### Core Implementation
- **`genetic_algorithm_trigger.py`**: Main genetic algorithm implementation
- **`demo_genetic_algorithm.py`**: Lightweight demo that works without heavy dependencies
- **`test_genetic_algorithm.py`**: Validation tests for algorithm components

### Configuration
The algorithm uses these key parameters (configurable via command line):

```python
POPULATION_SIZE = 80        # Number of individuals in each generation
NUM_GENERATIONS = 100       # Number of evolution cycles
FOURIER_K = 10             # Number of Fourier harmonics per channel
MUTATION_RATE = 0.15       # Probability of mutation per gene
MUTATION_STRENGTH = 0.02   # Standard deviation of mutation noise
ELITE_SIZE = 10            # Number of best individuals preserved
TOURNAMENT_SIZE = 5        # Size of selection tournaments
```

## 🚀 Usage

### Quick Demo (No Heavy Dependencies)
```bash
uv run demo_genetic_algorithm.py
```

### Full Optimization
```bash
# Test on first 3 models
uv run genetic_algorithm_trigger.py --models 1 2 3

# Run on all 45 models (full competition)
uv run genetic_algorithm_trigger.py

# Custom parameters with trigger saving disabled
uv run genetic_algorithm_trigger.py \
    --models 1 5 10 \
    --population-size 100 \
    --generations 150 \
    --no-save-triggers \
    --output my_submission.csv
```

### Help and Options
```bash
uv run genetic_algorithm_trigger.py --help
```

Available options:
- `--models`: Specific model IDs to process (default: all 1-45)
- `--output`: Output CSV file path (default: genetic_submission.csv)
- `--population-size`: Population size (default: 80)
- `--generations`: Number of generations (default: 100)
- `--no-save-triggers`: Skip saving individual triggers (saves disk space)

## 💾 Trigger Saving and Management

By default, the genetic algorithm saves the best trigger found for each model to `./outputs/genetic_triggers/`:

- **Trigger files**: `model_XX_trigger.npy` (NumPy arrays)
- **Metadata files**: `model_XX_metadata.json` (fitness scores and statistics)

This allows for:
- **Analysis**: Examine successful trigger patterns
- **Reuse**: Load previously found triggers for further optimization
- **Debugging**: Investigate why certain models failed
- **Comparison**: Compare triggers across different runs

## 📊 Algorithm Details

### Chromosome Representation
Each individual in the population represents a trigger as Fourier coefficients:
- **Shape**: `(3 channels, 2×K coefficients)` where K=10
- **Encoding**: `[a₁...a_K | b₁...b_K]` for cosine and sine terms per channel
- **Conversion**: Coefficients → 75-sample time series via inverse Fourier transform

### Fitness Function
The fitness score measures trigger effectiveness using the same objective as the notebook:

```python
score = α × L_div - β × L_track - λ × L2
```

Where:
- **L_div**: Divergence between poisoned and clean model predictions
- **L_track**: Tracking penalty (encourages coherent predictions)
- **L2**: Regularization penalty (prevents overly large triggers)

### Evolution Process
1. **Initialization**: Random Fourier coefficients with small variance
2. **Evaluation**: Convert to triggers and test against poisoned models
3. **Selection**: Tournament-based parent selection
4. **Reproduction**: Single-point crossover + Gaussian mutation
5. **Replacement**: Elitist strategy preserving best individuals

## 🔧 Technical Implementation

### Key Classes

#### `TriggerFitnessEvaluator`
Handles fitness evaluation for candidate triggers:
- Precomputes clean model predictions for efficiency
- Injects triggers into input sequences
- Computes divergence and tracking scores
- Returns fitness values for the genetic algorithm

#### `GeneticTriggerOptimizer`
Core genetic algorithm implementation:
- Manages population evolution
- Implements selection, crossover, and mutation operators
- Tracks fitness progress across generations
- Returns best trigger found

### Performance Features
- **Efficient Fourier Operations**: Pre-computed basis functions
- **Rich Progress Display**: Real-time evolution tracking
- **Robust Error Handling**: Graceful failure recovery
- **Memory Efficient**: Processes one model at a time

## 📈 Expected Performance

The genetic algorithm typically:
- Converges within 50-100 generations
- Processes each model in 5-15 minutes (depending on hardware)
- Achieves success rates comparable to gradient-based methods
- Provides good exploration of the search space

## 🔬 Validation

Run the test suite to validate the implementation:
```bash
uv run test_genetic_algorithm.py
```

This verifies:
- Fourier coefficient conversion accuracy
- Genetic operator functionality
- Population management
- Command-line interface

## 📝 Output Format

The algorithm generates a submission CSV with the required format:
- **Rows**: One per model (45 total)
- **Columns**: `model_id` + 225 trigger values
- **Column names**: `channel_44_1`, `channel_44_2`, ..., `channel_46_75`
- **Values**: Flattened trigger arrays

## 🎯 Advantages of Genetic Approach

1. **Global Search**: Less likely to get stuck in local optima compared to gradient methods
2. **No Gradients Required**: Works with any black-box fitness function
3. **Parallel Evaluation**: Population can be evaluated in parallel
4. **Robust**: Handles noisy or discontinuous fitness landscapes
5. **Interpretable**: Easy to understand and modify algorithm components

## 🔍 Comparison with Other Methods

| Method | Search Strategy | Representation | Convergence |
|--------|----------------|----------------|-------------|
| **Genetic Algorithm** | Population-based global search | Fourier coefficients | Stable, exploratory |
| **Gradient Descent** | Local optimization | Raw parameters | Fast, but local optima risk |
| **Greedy Search** | Iterative improvement | Vertex interpolation | Good balance |

The genetic algorithm provides a robust alternative that complements the existing gradient-based and greedy search approaches in the repository.

## 🛠 Dependencies

The implementation requires:
- `numpy`: Numerical operations
- `pandas`: Data handling
- `torch`: Neural network operations
- `darts`: Time series forecasting models
- `rich`: Enhanced console output
- `tqdm`: Progress bars

All dependencies are managed through the project's `uv.lock` file.

## 📚 References

The implementation follows established genetic algorithm principles and is specifically adapted for the time series trigger reconstruction problem described in the ESA challenge.
