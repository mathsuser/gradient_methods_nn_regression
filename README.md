# gradient_methods_nn_regression

Educational repository for studying optimisation and error decomposition on a synthetic nonlinear regression problem.

## Current scope

The repository currently contains three experiment families.

1. The original Week-1 optimisation baseline compares six explicit sampling regimes on the same baseline nonlinear regression problem: full-batch gradient descent, single-observation SGD with replacement, mini-batch SGD with replacement at batch sizes 32 and 256, and random reshuffling at batch sizes 32 and 256. This workflow includes a sampling-aware learning-rate pilot, common learning-rate selection, a five-seed baseline comparison, and a baseline analysis script for declared comparisons, figures, and risk-identity checks.

2. The ambient-dimension stress experiment is a secondary comparison that reuses the Week-1 baseline configuration and selected learning rate while increasing ambient feature dimension to `d=20` and `d=100`. Its analysis combines the reused `d=6` baseline runs with the new stress outputs and writes to separate `week1_dimension_stress` raw and figure namespaces.

3. The focused sampling-law branch compares `wr_1` and `rr_1`: single-observation with-replacement sampling at batch size 1 versus random reshuffling at batch size 1. The branch uses the same baseline training dataset, evaluation dataset, network architecture, model initialisation, learning rate, and finite examples-processed budget. Only the sampling law and sampling trajectory differ. It runs 30 WR-1 trajectories and 30 RR-1 trajectories with explicit stable sampling-seed lists, per-run persistence, integrity validation, and descriptive Monte Carlo analysis.

The repository also includes pytest coverage for data generation, model construction, metric helpers, sampling iterators, training accounting, branch-run configuration, and branch-analysis integrity checks.

## Project layout

- `src/gradient_methods_nn_regression/data.py`: true function and dataset generator
- `src/gradient_methods_nn_regression/model.py`: fixed shallow regression network architecture
- `src/gradient_methods_nn_regression/metrics.py`: evaluation metrics and accounting helpers
- `src/gradient_methods_nn_regression/training.py`: explicit sampling-based training loop utilities
- `configs/baseline.json`: locked baseline assumptions for data, model, optimiser, seeds, and shared paths
- `configs/experiments/week1_gradient_methods.json`: six-method Week-1 baseline protocol and training targets
- `configs/experiments/sampling_law_nn_branch.json`: fixed WR-1/RR-1 branch protocol, including branch methods, model seed, learning rate, examples-processed budget, checkpoint cadence, explicit trajectory IDs and sampling seeds, and branch output paths
- `docs/technical-notes.md`: modelling assumptions, experiment protocol, and analysis notes
- `experiments/00_dataset_exploration.py`: lightweight exploration helpers
- `experiments/01_dataset_checks.py`: baseline dataset generation, manifest, and diagnostics
- `experiments/02_single_run_smoke.py`: one-method, one-learning-rate smoke run for tracing the pipeline
- `experiments/03_learning_rate_pilot.py`: Week-1 pilot for shared stable learning-rate selection
- `experiments/04_baseline_comparison.py`: 30-run baseline comparison at the locked common learning rate
- `experiments/05_analyse_baseline.py`: baseline validation, paired summaries, risk-identity checks, and figures
- `experiments/06_dimension_stress.py`: ambient-dimension stress runner for `d=20` and `d=100`; supports `--preflight`
- `experiments/07_dimension_stress_analysis.py`: analysis combining reused `d=6` baseline runs with `d=20` and `d=100` stress outputs; supports `--preflight`
- `experiments/08_sampling_law_nn_branch.py`: focused WR-1/RR-1 branch runner; supports exactly one of `--plan-only`, `--preflight`, or `--full`; full mode executes all configured trajectories and reports lightweight run-level progress
- `experiments/09_analyse_sampling_law_nn_branch.py`: consumes generated branch outputs, validates experiment integrity, and writes checkpoint summaries, terminal summaries, WR/RR descriptive comparisons, and branch figures
- `tests/test_data.py`: function and generator contract tests
- `tests/test_model.py`: model architecture and initialisation contract tests
- `tests/test_metrics.py`: regression metric helper tests
- `tests/test_training.py`: sampling iterator, training accounting, and checkpoint-evaluation tests
- `tests/test_sampling_law_nn_branch.py`: branch configuration, run-spec, seeding, paired-initialisation, and smoke-run tests
- `tests/test_sampling_law_nn_branch_analysis.py`: branch analysis integrity and summary tests

## Outputs

The scripts generate local artifacts. Experiment result artifacts are ignored by git and are not expected to be present in a fresh clone.

- `data/generated/baseline_train.npz`
- `data/generated/baseline_validation.npz`
- `data/generated/baseline_test.npz`
- `data/generated/baseline_manifest.json`
- `results/figures/*.png`
- `results/raw/week1_gradient_methods/learning_rate_selection.json`
- `results/raw/week1_gradient_methods/learning_rate_pilot_histories/`
- `results/figures/week1_gradient_methods/`
- `results/raw/week1_gradient_methods/baseline_comparison_runs/`
- `results/raw/week1_gradient_methods/baseline_comparison_manifest.json`
- `results/raw/week1_gradient_methods/baseline_analysis/`
- `results/figures/week1_gradient_methods/baseline_analysis/`
- `results/raw/week1_dimension_stress/`
- `results/raw/week1_dimension_stress/analysis/`
- `results/figures/week1_dimension_stress/`
- `results/raw/sampling_law_branch/nn_wr1_rr1/`
- `results/raw/sampling_law_branch/nn_wr1_rr1/analysis/`
- `results/figures/sampling_law_branch/nn_wr1_rr1/`

The `.gitignore` keeps generated datasets and experiment outputs local while retaining `results/raw/week1_gradient_methods/learning_rate_selection.json` as the tracked machine-readable learning-rate decision artifact.

## Quick start

```bash
conda activate gradient-methods-nn-regression
python -m pip install -e ".[dev]"
pytest -q
```

Generate baseline datasets and run the original Week-1 workflow:

```bash
python experiments/01_dataset_checks.py
python experiments/02_single_run_smoke.py
python experiments/03_learning_rate_pilot.py
python experiments/04_baseline_comparison.py
python experiments/05_analyse_baseline.py
```

Run the ambient-dimension stress workflow:

```bash
python experiments/06_dimension_stress.py --preflight
python experiments/06_dimension_stress.py
python experiments/07_dimension_stress_analysis.py --preflight
python experiments/07_dimension_stress_analysis.py
```

Run the focused WR-1/RR-1 sampling-law branch:

```bash
python experiments/08_sampling_law_nn_branch.py --plan-only
python experiments/08_sampling_law_nn_branch.py --preflight
python experiments/08_sampling_law_nn_branch.py --full
python experiments/09_analyse_sampling_law_nn_branch.py
```

## Reproducibility notes

- Baseline seeds are independent by split: train, validation, and test.
- Baseline dtype is `float64` to reduce numerical fragility in decomposition checks.
- Generated dataset manifests store dataset-level summary statistics and decomposition residual metrics.
- Paired model initialisation is verified by an initial-state checksum for each model seed or branch trajectory.
- Week-1 sampling streams use deterministic seeds derived from the configured sampling-seed offset.
- The focused sampling-law branch uses explicit seed lists: `91000` to `91029` for WR-1 and `92000` to `92029` for RR-1.
- With-replacement methods report optimiser updates and examples processed; random reshuffling additionally records epoch and step-within-epoch metadata where meaningful.
