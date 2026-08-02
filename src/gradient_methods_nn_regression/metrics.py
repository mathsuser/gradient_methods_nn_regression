# Evaluation helpers for regression metrics and basic training accounting.
import torch


def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the mean squared error between predictions and targets."""
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    return torch.mean((predictions - targets) ** 2)


def function_estimation_mse(predictions: torch.Tensor, target_function_values: torch.Tensor) -> torch.Tensor:
    """Return MSE against the underlying target function values."""
    return mean_squared_error(predictions, target_function_values)


def noisy_prediction_mse(predictions: torch.Tensor, observed_targets: torch.Tensor) -> torch.Tensor:
    """Return MSE against noisy observed targets."""
    return mean_squared_error(predictions, observed_targets)


def gradient_norm(parameters: list[torch.Tensor]) -> torch.Tensor:
    """Return the Euclidean norm of the concatenated parameter gradients."""
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.cat([grad.reshape(-1) for grad in grads]).norm()


def parameter_norm(parameters: list[torch.Tensor]) -> torch.Tensor:
    """Return the Euclidean norm of the concatenated parameters."""
    values = [param.detach().reshape(-1) for param in parameters]
    if not values:
        return torch.tensor(0.0)
    return torch.cat(values).norm()


def examples_processed(batch_size: int, num_batches: int) -> int:
    """Return the total number of examples processed over the given number of batches."""
    return batch_size * num_batches
