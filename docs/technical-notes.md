# Gradient Methods for Neural-Network Regression
This repository contains a controlled study of gradient-based training methods on a synthetic nonlinear regression problem.

The experiment compares explicit sampling regimes:

- full-batch gradient descent;
- single-observation SGD with replacement;
- mini-batch SGD with replacement at batch size 32;
- mini-batch SGD with replacement at batch size 256;
- random reshuffling SGD at batch size 32;
- random reshuffling SGD at batch size 256.
All methods train the same neural-network architecture from paired initial parameters and are evaluated on the same generated datasets.

## Research objective
The project examines how gradient-estimator size and sampling rule change the behaviour of gradient-based training when the regression model is nonlinear in its parameters.

The comparison is not limited to final test loss. It considers:

- convergence against epochs, optimiser updates and examples processed;
- function-estimation error;
- noisy prediction error;
- generalisation gap;
- gradient norm;
- variability across initialisation seeds;
- computational time.
The aim is to distinguish statistical error, approximation error and incomplete optimisation rather than treating them as one undifferentiated measure of model performance.

## Regression problem
Features are generated according to

$$
X \sim \mathcal{N}(0, I_d).
$$

The observed response is

$$
Y = f^\star(X) + \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0, \sigma^2),
$$

where

$$
f^\star(x)
= 1.5\sin(x_1)
+ 0.8(x_2^2 - 1)
+ 0.7\tanh(x_3x_4)
+ 0.5x_5x_6.
$$

The generator retains both the noisy target $Y$ and the noiseless value
$f^\star(X)$. This makes it possible to evaluate recovery of the true
regression function directly.

## Model architecture
The regression model is a shallow one-hidden-layer neural network implemented in `src/gradient_methods_nn_regression/model.py`.

The baseline architecture is:

$$
6 \rightarrow 16 \rightarrow 1.
$$

The hidden layer uses a `tanh` activation. There is no dropout or batch
normalisation. Training uses plain SGD with momentum `0.0` and weight decay
`0.0`.

The model is nonlinear in its parameters, so optimisation behaviour can depend on gradient-estimator size, sampling rule and initialisation. In the baseline, the input dimension is six because all six generated features are relevant to
the true regression function.

In the ambient-dimension stress experiment, the input layer width changes while the hidden width remains fixed at 16. Therefore increasing ambient dimension also increases the number of trainable first-layer parameters.

## Baseline experiment
### Fixed baseline setup
The initial experiment uses the following fixed data and training protocol:

| Component | Specification |
| --- | --- |
| Training observations | 5,000 |
| Validation observations | 2,000 |
| Test observations | 20,000 |
| Feature dimension | 6 |
| Noise standard deviation | 0.3 |
| Loss | Mean squared error |
| Model seeds | 0, 1, 2, 3, 4 |
| Training budget | 100 data-equivalent passes |

**Remark:** a data-equivalent pass is an accounting unit equal to processing `n_train` examples. 
- In the baseline, `n_train = 5,000`, so one data-equivalent pass is 5,000 examples processed and 100 data-equivalent passes is 500,000 examples processed. 
- For full-batch gradient descent, one optimiser update is one data-equivalent pass. 
- For random reshuffling, one epoch is one complete shuffled traversal and therefore one data-equivalent pass. 
- For with-replacement methods, an epoch is not literally defined, so the experiment reports optimiser updates and examples processed instead; 5,000 sampled examples are counted as one data-equivalent pass even though some observations may be sampled repeatedly and others not at all.

### Learning-rate pilot

A short learning-rate pilot is run before the main comparison to select one common learning rate for all retained sampling regimes. This is a constrained hyperparameter-selection step, not a method-specific tuning study.

The pilot keeps the following quantities fixed:

| Quantity | Pilot setting |
| --- | --- |
| Script | `experiments/03_learning_rate_pilot.py` |
| Learning-rate grid | $\{0.001, 0.003, 0.01, 0.03, 0.1\}$ |
| Methods | All six retained Week 1 methods |
| Dataset splits | Fixed train and validation splits |
| Test split | Not used |
| Pilot model seed | `0` |
| Sampling seeds | Derived from `sampling_seed_offset = 1000` |
| Pilot budget | 10 data-equivalent passes |
| Examples processed | 50,000 |
| Checkpoint cadence | Every 5,000 examples processed |
| Optimiser | Plain SGD |
| Momentum | `0.0` |
| Weight decay | `0.0` |

For each candidate learning rate, the pilot trains all six methods from the same paired initial model state and evaluates checkpoint behaviour on the fixed validation split. A candidate is accepted only if every retained method is stable and achieves a material training-loss reduction during the pilot. The selected rate is the largest accepted common rate.

For example, a candidate rate is rejected if five methods improve but one method becomes non-finite, or if all methods remain finite but one method does not reduce training MSE by the material-reduction threshold. In the completed pilot, rates `0.001`, `0.003` and `0.01` were stable but rejected because `full_batch_gd` did not reduce training MSE enough over the short pilot budget. Rate `0.1` was rejected because `single_observation_sgd` became non-finite.

**Rate `0.03` was the largest grid value that was both stable and useful for every retained method, so it became the common rate.**

The rationale for a common learning rate is experimental control only. The baseline comparison is intended to study gradient-estimator size and sampling rule and not intended to compare separately tuned optimiser configurations. A common rate isolates that confound, while still requiring every method to make useful progress before the main comparison begins.


The machine-readable output of this methodology is located at:

```text
results/raw/week1_gradient_methods/learning_rate_selection.json
```

That file records the evaluated grid, selected rate, rejected-rate reasons, pilot budget, pilot model seed, per-method sampling seeds and commit hash. The per-run pilot histories are written to:

```text
results/raw/week1_gradient_methods/learning_rate_pilot_histories/
```
The common rate is reused for the five-seed baseline comparison and later stress experiments without method-specific retuning.

### Baseline comparison runner
The baseline comparison is executed by 
```text
`experiments/04_baseline_comparison.py`.
```
It runs the six retained method cases across the five paired model seeds `0, 1, 2, 3, 4`, giving 30 primary runs. For each model seed, one reference model is constructed, its initial state is copied into all method-specific models, and the initial-state checksum is saved. Sampling seeds are derived deterministically from the configured offset, model seed and method index.

Each baseline run uses the locked selected learning rate, fixed train,
validation and test datasets, a 500,000-example budget, 5,000-example
checkpoints and no early stopping. Per-run histories are saved immediately as
CSV files and run metadata as JSON files under
`results/raw/week1_gradient_methods/baseline_comparison_runs/`. The run
manifest is written to
`results/raw/week1_gradient_methods/baseline_comparison_manifest.json`.

### Baseline analysis runner
The baseline analysis is executed by:

```text
experiments/05_analyse_baseline.py
```
Before calculating comparisons, it validates run count, method identifiers, paired initial-state checksums, deterministic sampling seeds, final accounting and non-finite values.

The analysis is organised into four declared comparisons:

- gradient-estimator size: `full_batch_gd`, `single_observation_sgd`,
  `minibatch_with_replacement_b32`, `minibatch_with_replacement_b256`;
- sampling rule at batch size 32: `minibatch_with_replacement_b32` versus
  `random_reshuffling_b32`;
- sampling rule at batch size 256: `minibatch_with_replacement_b256` versus
  `random_reshuffling_b256`;
- batch size within each sampling rule: $B=32$ versus $B=256$ separately for
  with-replacement sampling and random reshuffling.

The analysis reports method-level summary statistics and paired-seed differences for training loss, validation loss, noisy test loss, function estimation error, generalisation gap, gradient norms, parameter norm, optimiser
steps, examples processed, data-equivalent passes, wall-clock time and epoch where it is meaningful. The output tables are written under
```text
results/raw/week1_gradient_methods/baseline_analysis/
```

and comparison-scoped figures are written under

```text
results/figures/week1_gradient_methods/baseline_analysis/
```
The same analysis script also performs the empirical risk-identity check on the independent test split:

$$
\text{noisy test MSE} - \text{function MSE}
\approx \sigma^2.
$$

For the baseline, `noise_std = 0.3`, so $\sigma^2 = 0.09$. Finite-sample values are not exactly equal to 0.09 because the realised test noise variance and the empirical cross term are not exactly their population expectations.

For each model seed, the methods start from identical parameters. Training and evaluation datasets remain fixed across methods.

## Evaluation
The principal metric is the test function-estimation error:

$$
E_{\mathrm{function}}
= \frac{1}{m}
\sum_{i=1}^{m}
\left[
\hat f(X_i) - f^\star(X_i)
\right]^2.
$$

The experiment also records noisy prediction error:

$$
E_{\mathrm{prediction}}
= \frac{1}{m}
\sum_{i=1}^{m}
\left[
\hat f(X_i) - Y_i
\right]^2.
$$

Because the noise variance is known, the results can be checked against the population identity

$$
R(\hat f) - R(f^\star)
= \mathbb{E}\left[
(\hat f(X) - f^\star(X))^2
\right].
$$

Neural-network optimisation error is not observed exactly because the global empirical minimum is unknown. Differences from the best loss found across runs are therefore reported only as empirical optimisation-gap proxies.

## Scope
The repository is an educational and empirical study, not a general optimisation library or production machine-learning framework.

Only one experimental factor is varied at a time.

The initial scope excludes:

- alternative neural-network architectures;
- adaptive optimisers;
- momentum;
- regularisation comparisons;
- extensive hyperparameter searches;
- real-world datasets;
- claims of universal optimiser superiority.
The purpose is to establish a reproducible experimental process for analysing gradient methods before introducing additional sources of complexity.
