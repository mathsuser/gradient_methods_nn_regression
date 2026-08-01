
import numpy as np
import pytest

from gradient_methods_nn_regression.data import (
    generate_synthetic_regression_data,
    true_regression_function,
)

def test_true_regression_function_known_values() -> None:
    x = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 2.0, 3.0],
            [np.pi / 2, 2.0, 1.0, -1.0, -1.0, 4.0],
        ],
        dtype=np.float64,
    )

    expected = np.array(
        [
            1.5 * np.sin(0.0) + 0.8 * (1.0**2 - 1.0) + 0.7 * np.tanh(0.0 * 0.0) + 0.5 * 2.0 * 3.0,
            1.5 * np.sin(np.pi / 2) + 0.8 * (2.0**2 - 1.0) + 0.7 * np.tanh(1.0 * -1.0) + 0.5 * -1.0 * 4.0,
        ],
        dtype=np.float64,
    )

    actual = true_regression_function(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_true_regression_function_rejects_less_than_six_features() -> None:
    x = np.zeros((3, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="at least six features"):
        true_regression_function(x)

def test_generate_synthetic_regression_data_contract() -> None:
    n_samples = 32
    n_features = 8
    noise_std = 0.1
    seed = 123

    x1, y_noisy1, y_true1, noise1 = generate_synthetic_regression_data(
        n_samples=n_samples,
        n_features=n_features,
        noise_std=noise_std,
        seed=seed,
    )
    x2, y_noisy2, y_true2, noise2 = generate_synthetic_regression_data(
        n_samples=n_samples,
        n_features=n_features,
        noise_std=noise_std,
        seed=seed,
    )

    # Shape contract
    assert x1.shape == (n_samples, n_features)
    assert y_noisy1.shape == (n_samples,)
    assert y_true1.shape == (n_samples,)
    assert noise1.shape == (n_samples,)

    # Reproducibility with same seed
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(y_noisy1, y_noisy2)
    np.testing.assert_allclose(y_true1, y_true2)
    np.testing.assert_allclose(noise1, noise2)

    # Decomposition identity
    np.testing.assert_allclose(y_noisy1 - y_true1, noise1, rtol=1e-5, atol=1e-7)


def test_generate_synthetic_regression_data_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        generate_synthetic_regression_data(n_samples=0, n_features=6, noise_std=0.1, seed=0)

    with pytest.raises(ValueError, match="n_features must be at least 6"):
        generate_synthetic_regression_data(n_samples=10, n_features=5, noise_std=0.1, seed=0)

    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        generate_synthetic_regression_data(n_samples=10, n_features=6, noise_std=-0.1, seed=0)