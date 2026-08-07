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
The project examines how batch size changes the behaviour of gradient-based training when the regression model is nonlinear in its parameters.

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

[
X\sim\mathcal N(0,I_d).
]

The observed response is

[
Y=f^\star(X)+\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,\sigma^2),
]

where

# [
f^\star(x)
1.5\sin(x_1)
+
0.8(x_2^2-1)
+
0.7\tanh(x_3x_4)
+
0.5x_5x_6.
]

The generator retains both the noisy target (Y) and the noiseless value (f^\star(X)). This makes it possible to evaluate recovery of the true regression function directly.

## Baseline experiment
The initial experiment uses:

| Component | Specification |
| --- | --- |
| Training observations | 5,000 |
| Validation observations | 2,000 |
| Test observations | 20,000 |
| Feature dimension | 6 |
| Noise standard deviation | 0.3 |
| Network | (6 \rightarrow 16 \rightarrow 1) |
| Activation | `tanh` |
| Loss | Mean squared error |
| Model seeds | 0, 1, 2, 3, 4 |
| Training budget | 100 data-equivalent passes |


A short learning-rate pilot is run before the main comparison. The baseline uses the largest stable learning rate shared by the compared sampling regimes.

Pilot status (Week 1): completed with a shared selected learning rate of 0.03 under a 10-pass (50,000 examples) budget and 5,000-example checkpoint cadence.

For each model seed, the methods start from identical parameters. Training and evaluation datasets remain fixed across methods.

## Evaluation
The principal metric is the test function-estimation error:

# [
E_{\mathrm{function}}
\frac{1}{m}
\sum_{i=1}^{m}
\left[
\hat f(X_i)-f^\star(X_i)
\right]^2.
]

The experiment also records noisy prediction error:

# [
E_{\mathrm{prediction}}
\frac{1}{m}
\sum_{i=1}^{m}
\left[
\hat f(X_i)-Y_i
\right]^2.
]

Because the noise variance is known, the results can be checked against the population identity

# [
R(\hat f)-R(f^\star)
\mathbb E\left[
(\hat f(X)-f^\star(X))^2
\right].
]

Neural-network optimisation error is not observed exactly because the global empirical minimum is unknown. Differences from the best loss found across runs are therefore reported only as empirical optimisation-gap proxies.

## Experimental sequence
The repository is developed in the following order:

1. [x] define and validate the nonlinear data-generating process;
2. [x] generate fixed training, validation and test datasets;
3. [x] implement the fixed neural-network architecture in the reusable model module;
4. [x] validate paired initialisation, trainable-parameter counting and training-accounting rules;
5. [x] add reusable regression evaluation metrics for function-estimation and noisy-prediction error;
6. [x] define the explicit sampling-method experiment configuration and its validation checks;
7. [x] run the learning-rate pilot;
8. [ ] execute the baseline comparison across five seeds;
9. [ ] aggregate convergence, error and runtime results;
10. [ ] introduce controlled stress cases, beginning with irrelevant ambient dimensions.

Only one experimental factor is varied at a time.

## Repository structure

```
configs/       Locked experiment configurations
data/          Generated local datasets
docs/          Experimental and technical design
experiments/   Executable experiment scripts
results/       Raw outputs, summaries and figures
src/           Reusable project code
tests/         Validation and regression tests
```
The detailed design, assumptions and implementation workflow are documented in:

```
docs/technical-design.md
```

Pilot outputs are written to:

- `results/raw/week1_gradient_methods/learning_rate_selection.json`
- `results/raw/week1_gradient_methods/learning_rate_pilot_histories/`
- `results/figures/week1_gradient_methods/`

## Scope
The repository is an educational and empirical study, not a general optimisation library or production machine-learning framework.

The initial scope excludes:

- alternative neural-network architectures;
- adaptive optimisers;
- momentum;
- regularisation comparisons;
- extensive hyperparameter searches;
- real-world datasets;
- claims of universal optimiser superiority.
The purpose is to establish a reproducible experimental process for analysing gradient methods before introducing additional sources of complexity.
