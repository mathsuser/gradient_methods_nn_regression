# gradient_methods_nn_regression

Educational repository for studying optimisation and error decomposition on a synthetic nonlinear regression problem.

## Current scope

- Synthetic data-generating process with known ground-truth function
- Dataset generator using NumPy Generator API
- Baseline experiment configuration in JSON
- Dataset checks script that generates train/validation/test splits
- Dataset diagnostics: distributions, scatter, feature summaries, decomposition checks
- Test suite with pytest

## Project layout

- src/gradient_methods_nn_regression/data.py: true function + dataset generator
- configs/baseline.json: locked baseline assumptions (sizes, seeds, dtype, noise, paths)
- experiments/00_dataset_exploration.py: lightweight exploration helpers
- experiments/01_dataset_checks.py: baseline dataset generation + manifest + diagnostics
- tests/test_data.py: focused tests for function and generator contract

## Outputs

- data/generated/baseline_train.npz
- data/generated/baseline_validation.npz
- data/generated/baseline_test.npz
- data/generated/baseline_manifest.json
- results/figures/*.png (Dataset diagnostic figures)

## Quick start

```bash
conda activate gradient-methods-nn-regression
python -m pip install -e ".[dev]"
pytest -q
python experiments/01_dataset_checks.py
```

## Reproducibility notes

- Baseline seeds are independent by split (train/validation/test).
- Baseline dtype is float64 to reduce numerical fragility in decomposition checks.
- Manifest stores dataset-level summary stats and decomposition residual metrics.