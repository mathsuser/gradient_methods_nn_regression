# gradient_methods_nn_regression

Educational repository for studying optimisation and error decomposition on a synthetic nonlinear regression problem.

## Current scope

- Synthetic data-generating process with known ground-truth function
- Dataset generator using NumPy `Generator` API
- Incremental test suite with `pytest`
- Initial dataset exploration script

## Project layout

- `src/gradient_methods_nn_regression/data.py`: true function + dataset generator
- `tests/test_data.py`: focused tests for function and generator contract
- `experiments/01_dataset_exploration.py`: quick dataset sanity checks

## Quick start

```bash
conda activate gradient-methods-nn-regression
pytest -q
python experiments/01_dataset_exploration.py
```