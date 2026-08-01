"""Synthetic nonlinear regression data generation."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def true_regression_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the noiseless target function."""
    if x.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    if x.shape[1] < 6:
        raise ValueError("Input array must have at least six features.")

    output = (
        1.5 * np.sin(x[:, 0])
        + 0.8 * (x[:, 1]**2 - 1)
        + 0.7 * np.tanh(x[:, 2] * x[:, 3])
        + 0.5 * x[:, 4] * x[:, 5]
    )
    return output



def generate_synthetic_regression_data(
    n_samples: int,
    n_features: int,
    noise_std: float,
    seed: Optional[int] = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Generate features, noisy targets, noiseless targets, and noise."""
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if n_features < 6:
        raise ValueError("n_features must be at least 6.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    rng = np.random.default_rng(seed)

    x = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features)).astype(np.float64)
    y_true = true_regression_function(x)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples).astype(np.float64)
    y_noisy = y_true + noise

    return x, y_noisy, y_true, noise

