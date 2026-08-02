import torch

from gradient_methods_nn_regression.metrics import (
    examples_processed,
    function_estimation_mse,
    gradient_norm,
    mean_squared_error,
    noisy_prediction_mse,
    parameter_norm,
)


def test_mean_squared_error_matches_expected_value() -> None:
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 4.0, 3.0])

    assert torch.isclose(mean_squared_error(predictions, targets), torch.tensor(4.0 / 3.0))


def test_function_and_noisy_prediction_mse_are_available() -> None:
    predictions = torch.tensor([1.0, 2.0, 3.0])
    target_function_values = torch.tensor([1.0, 2.0, 4.0])
    observed_targets = torch.tensor([0.0, 2.0, 4.0])

    assert torch.isclose(function_estimation_mse(predictions, target_function_values), torch.tensor(1.0 / 3.0))
    assert torch.isclose(noisy_prediction_mse(predictions, observed_targets), torch.tensor(2.0 / 3.0))


def test_gradient_and_parameter_norms_are_computed() -> None:
    param_a = torch.tensor([1.0, 2.0], requires_grad=True)
    param_b = torch.tensor([3.0], requires_grad=True)
    param_a.grad = torch.tensor([1.0, 0.0])
    param_b.grad = torch.tensor([2.0])

    assert torch.isclose(gradient_norm([param_a, param_b]), torch.tensor(2.2360679), atol=1e-6)
    assert torch.isclose(parameter_norm([param_a, param_b]), torch.tensor(3.7416575), atol=1e-6)


def test_examples_processed_counts_total_items() -> None:
    assert examples_processed(batch_size=8, num_batches=3) == 24
