# gradient_methods_nn_regression

Educational repository for studying optimisation and error decomposition on a synthetic nonlinear regression problem.

## Current scope

- Synthetic data-generating process with known ground-truth function
- Dataset generator using NumPy Generator API
- Baseline configuration in JSON for the fixed regression problem
- Experiment configuration in JSON for explicit sampling-method comparisons
- Dataset checks script that generates train/validation/test splits
- Dataset diagnostics: distributions, scatter, feature summaries, decomposition checks
- Sampling-aware learning-rate pilot across six methods with a common-budget comparison
- Test suite with pytest

## Project layout

- src/gradient_methods_nn_regression/data.py: true function + dataset generator
- src/gradient_methods_nn_regression/model.py: fixed shallow regression network architecture
- src/gradient_methods_nn_regression/metrics.py: evaluation metrics and basic accounting helpers
- src/gradient_methods_nn_regression/training.py: explicit sampling-based training loop utilities
- configs/baseline.json: locked baseline assumptions for data, model, optimiser, seeds, and shared paths
- configs/experiments/week1_gradient_methods.json: explicit week-1 experiment methods and training targets
- experiments/00_dataset_exploration.py: lightweight exploration helpers
- experiments/01_dataset_checks.py: baseline dataset generation + manifest + diagnostics
- experiments/02_learning_rate_pilot.py: week-1 pilot for shared stable learning-rate selection
- tests/test_data.py: focused tests for function and generator contract
- tests/test_model.py: model architecture and initialization contract tests
- tests/test_metrics.py: regression metric helper tests
- tests/test_training.py: explicit sampling and training-accounting contract tests

## Outputs

- data/generated/baseline_train.npz
- data/generated/baseline_validation.npz
- data/generated/baseline_test.npz
- data/generated/baseline_manifest.json
- results/figures/*.png (Dataset diagnostic figures)
- results/raw/week1_gradient_methods/learning_rate_selection.json (machine-readable LR decision)
- results/raw/week1_gradient_methods/learning_rate_pilot_histories/*.json (pilot checkpoint histories)
- results/figures/week1_gradient_methods/*.png (pilot convergence and diagnostic figures)

## Quick start

```bash
conda activate gradient-methods-nn-regression
python -m pip install -e ".[dev]"
pytest -q
python experiments/01_dataset_checks.py
python experiments/02_learning_rate_pilot.py
```

## Week 1 pilot status

- Pilot budget: 10 data-equivalent passes (50,000 examples), checkpoints every 5,000 examples.
- Selected common learning rate: 0.03.
- Selection artifact: results/raw/week1_gradient_methods/learning_rate_selection.json.

## Reproducibility notes

- Baseline seeds are independent by split (train/validation/test).
- Baseline dtype is float64 to reduce numerical fragility in decomposition checks.
- Manifest stores dataset-level summary stats and decomposition residual metrics.