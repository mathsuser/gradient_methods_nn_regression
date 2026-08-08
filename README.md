# gradient_methods_nn_regression

Educational repository for studying optimisation and error decomposition on a synthetic nonlinear regression problem.

## Current scope

- Synthetic data-generating process with known ground-truth function
- Dataset generator using NumPy Generator API
- Baseline configuration in JSON for the fixed regression problem
- Experiment configuration in JSON for explicit sampling-method comparisons
- Dataset checks script that generates train/validation/test splits
- Dataset diagnostics: distributions, scatter, feature summaries, decomposition checks
- Single-run smoke script for manually tracing one training configuration
- Sampling-aware learning-rate pilot across six methods with a common-budget comparison
- Baseline comparison runner across six methods and five paired model seeds
- Baseline analysis script for declared comparisons, figures, and risk-identity checks
- Ambient-dimension stress runner and analysis scripts for the secondary Step 16 comparison
- Test suite with pytest

## Project layout

- src/gradient_methods_nn_regression/data.py: true function + dataset generator
- src/gradient_methods_nn_regression/model.py: fixed shallow regression network architecture
- src/gradient_methods_nn_regression/metrics.py: evaluation metrics and basic accounting helpers
- src/gradient_methods_nn_regression/training.py: explicit sampling-based training loop utilities
- configs/baseline.json: locked baseline assumptions for data, model, optimiser, seeds, and shared paths
- configs/experiments/week1_gradient_methods.json: explicit week-1 experiment methods and training targets
- docs/technical-notes.md: modelling assumptions, experiment protocol and analysis notes
- experiments/00_dataset_exploration.py: lightweight exploration helpers
- experiments/01_dataset_checks.py: baseline dataset generation + manifest + diagnostics
- experiments/02_single_run_smoke.py: one-method, one-learning-rate smoke run for understanding the pipeline
- experiments/03_learning_rate_pilot.py: week-1 pilot for shared stable learning-rate selection
- experiments/04_baseline_comparison.py: 30-run baseline comparison at the locked common learning rate
- experiments/05_analyse_baseline.py: baseline validation, paired summaries, risk-identity checks and figures
- experiments/06_dimension_stress.py: Step 16 ambient-dimension stress runner for d=20 and d=100
- experiments/07_dimension_stress_analysis.py: Step 16 analysis combining reused d=6 baseline runs with stress outputs
- tests/test_data.py: focused tests for function and generator contract
- tests/test_model.py: model architecture and initialization contract tests
- tests/test_metrics.py: regression metric helper tests
- tests/test_training.py: explicit sampling and training-accounting contract tests

## Outputs

The scripts generate the following local artifacts. Experiment result artifacts are ignored by git and are not expected to be present in a fresh clone.

- data/generated/baseline_train.npz
- data/generated/baseline_validation.npz
- data/generated/baseline_test.npz
- data/generated/baseline_manifest.json
- results/figures/*.png (local dataset diagnostic figures; ignored by git)
- results/raw/week1_gradient_methods/learning_rate_selection.json (local machine-readable LR decision; ignored by git)
- results/raw/week1_gradient_methods/learning_rate_pilot_histories/*.json (local pilot checkpoint histories; ignored by git)
- results/figures/week1_gradient_methods/*.png (local pilot convergence and diagnostic figures; ignored by git)
- results/raw/week1_gradient_methods/baseline_comparison_runs/ (local per-run histories and metadata; ignored by git)
- results/raw/week1_gradient_methods/baseline_comparison_manifest.json (local baseline run manifest; ignored by git)
- results/raw/week1_gradient_methods/baseline_analysis/ (local summaries, paired comparisons and risk checks; ignored by git)
- results/figures/week1_gradient_methods/baseline_analysis/ (local baseline analysis figures; ignored by git)
- results/raw/week1_dimension_stress/ (local ambient-dimension stress run outputs and summaries; ignored by git)
- results/figures/week1_dimension_stress/ (local ambient-dimension stress figures; ignored by git)

## Quick start

```bash
conda activate gradient-methods-nn-regression
python -m pip install -e ".[dev]"
pytest -q
python experiments/01_dataset_checks.py
python experiments/02_single_run_smoke.py
python experiments/03_learning_rate_pilot.py
python experiments/04_baseline_comparison.py
python experiments/05_analyse_baseline.py
```

## Reproducibility notes

- Baseline seeds are independent by split (train/validation/test).
- Baseline dtype is float64 to reduce numerical fragility in decomposition checks.
- Manifest stores dataset-level summary stats and decomposition residual metrics.
- Paired model initialisation is verified by an initial-state checksum for each model seed.
- Sampling streams use deterministic seeds derived from the configured sampling-seed offset.
