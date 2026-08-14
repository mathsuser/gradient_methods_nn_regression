# Training helpers for the baseline regression experiment.
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterator, Literal

import torch

from gradient_methods_nn_regression.metrics import (
    function_estimation_mse,
    gradient_norm,
    parameter_norm,
)


SamplingMethod = Literal[
    "full_batch",
    "single_with_replacement",
    "minibatch_with_replacement",
    "random_reshuffling",
]


@dataclass(frozen=True)
class SampledBatch:
    indices: torch.Tensor
    actual_batch_size: int
    epoch: int | None
    step_within_epoch: int | None


HistoryValue = float | int | str | None
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _validate_sgd_optimiser(optimiser: torch.optim.Optimizer) -> None:
    if not isinstance(optimiser, torch.optim.SGD):
        raise TypeError("optimiser must be torch.optim.SGD")

    for parameter_group in optimiser.param_groups:
        if float(parameter_group.get("momentum", 0.0)) != 0.0:
            raise ValueError("optimiser momentum must be zero")
        if float(parameter_group.get("weight_decay", 0.0)) != 0.0:
            raise ValueError("optimiser weight_decay must be zero")


def iter_batch_indices(
    *,
    n_observations: int,
    sampling_method: SamplingMethod,
    batch_size: int,
    target_examples_processed: int,
    sampling_seed: int,
) -> Iterator[SampledBatch]:
    """Yield batch indices for the requested sampling method.

    With-replacement methods may use a truncated final batch to satisfy the
    exact examples-processed budget. Random reshuffling preserves complete
    epochs in the Week 1 experiment through configuration validation.
    """

    if n_observations <= 0:
        raise ValueError("n_observations must be positive")

    if target_examples_processed <= 0:
        raise ValueError("target_examples_processed must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    generator = torch.Generator()
    generator.manual_seed(sampling_seed)

    all_indices = torch.arange(n_observations)
    cumulative_examples = 0

    if sampling_method == "full_batch":
        if batch_size != n_observations:
            raise ValueError("full_batch requires batch_size == n_observations")

        epoch = 0
        while cumulative_examples < target_examples_processed:
            epoch += 1
            cumulative_examples += n_observations
            yield SampledBatch(
                indices=all_indices,
                actual_batch_size=n_observations,
                epoch=epoch,
                step_within_epoch=1,
            )
        return

    if sampling_method == "single_with_replacement":
        if batch_size != 1:
            raise ValueError("single_with_replacement requires batch_size == 1")

        while cumulative_examples < target_examples_processed:
            actual_batch_size = min(1, target_examples_processed - cumulative_examples)
            indices = torch.randint(
                low=0,
                high=n_observations,
                size=(actual_batch_size,),
                generator=generator,
            )
            cumulative_examples += actual_batch_size
            yield SampledBatch(
                indices=indices,
                actual_batch_size=actual_batch_size,
                epoch=None,
                step_within_epoch=None,
            )
        return

    if sampling_method == "minibatch_with_replacement":
        if batch_size <= 1:
            raise ValueError("minibatch_with_replacement requires batch_size > 1")

        while cumulative_examples < target_examples_processed:
            actual_batch_size = min(
                batch_size,
                target_examples_processed - cumulative_examples,
            )
            indices = torch.randint(
                low=0,
                high=n_observations,
                size=(actual_batch_size,),
                generator=generator,
            )
            cumulative_examples += actual_batch_size
            yield SampledBatch(
                indices=indices,
                actual_batch_size=actual_batch_size,
                epoch=None,
                step_within_epoch=None,
            )
        return

    if sampling_method == "random_reshuffling":
        if not (1 <= batch_size <= n_observations):
            raise ValueError(
                "random_reshuffling requires 1 <= batch_size <= n_observations"
            )

        epoch = 0
        while cumulative_examples < target_examples_processed:
            epoch += 1
            permutation = torch.randperm(n_observations, generator=generator)

            for batch_number, start in enumerate(range(0, n_observations, batch_size), start=1):
                indices = permutation[start : start + batch_size]
                actual_batch_size = int(indices.numel())
                cumulative_examples += actual_batch_size

                yield SampledBatch(
                    indices=indices,
                    actual_batch_size=actual_batch_size,
                    epoch=epoch,
                    step_within_epoch=batch_number,
                )

                if cumulative_examples >= target_examples_processed:
                    return

        return

    raise ValueError(f"Unsupported sampling method: {sampling_method}")


def evaluate_checkpoint(
    *,
    method: str,
    sampling_method: SamplingMethod,
    step: int,
    epoch: int | None,
    step_within_epoch: int | None,
    nominal_batch_size: int,
    actual_batch_size: int,
    checkpoint_examples: int,
    cumulative_examples_processed: int,
    batch_loss: float | None,
    update_gradient_norm: float | None,
    model: torch.nn.Module,
    loss_function: LossFunction,
    training_data: tuple[torch.Tensor, torch.Tensor],
    evaluation_data: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    training_elapsed_seconds: float,
    total_start_time: float,
) -> dict[str, HistoryValue]:
    """Evaluate a checkpoint without mutating model parameters."""

    x_train, y_train = training_data
    x_val, y_val, y_fun_val = evaluation_data
    n_train = int(x_train.shape[0])

    was_training = model.training
    model.eval()

    with torch.no_grad():
        training_predictions = model(x_train)
        training_mse = loss_function(training_predictions, y_train).item()

        validation_predictions = model(x_val)
        validation_mse = loss_function(validation_predictions, y_val).item()
        validation_function_mse = function_estimation_mse(
            validation_predictions,
            y_fun_val,
        ).item()

    model.zero_grad(set_to_none=True)

    full_training_predictions = model(x_train)
    full_training_loss = loss_function(full_training_predictions, y_train)
    full_training_loss.backward()
    full_gradient_norm = gradient_norm(list(model.parameters())).item()
    model.zero_grad(set_to_none=True)

    model.train(was_training)

    return {
        "method": method,
        "sampling_method": sampling_method,
        "step": step,
        "epoch": epoch,
        "step_within_epoch": step_within_epoch,
        "nominal_batch_size": nominal_batch_size,
        "actual_batch_size": actual_batch_size,
        "checkpoint_examples": checkpoint_examples,
        "cumulative_examples_processed": cumulative_examples_processed,
        "data_equivalent_passes": cumulative_examples_processed / n_train,
        "batch_loss": batch_loss,
        "training_mse": training_mse,
        "validation_mse": validation_mse,
        "validation_function_mse": validation_function_mse,
        "update_gradient_norm": update_gradient_norm,
        "full_gradient_norm": full_gradient_norm,
        "parameter_norm": parameter_norm(list(model.parameters())).item(),
        "training_elapsed_seconds": training_elapsed_seconds,
        "total_elapsed_seconds": time.perf_counter() - total_start_time,
    }


def train_model(
    *,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    loss_function: LossFunction,
    training_data: tuple[torch.Tensor, torch.Tensor],
    evaluation_data: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    sampling_method: SamplingMethod,
    batch_size: int,
    target_examples_processed: int,
    sampling_seed: int,
    evaluation_every_examples: int,
    method: str = "train_model",
) -> list[dict[str, HistoryValue]]:
    """Train with an explicit sampling method and record checkpoint metrics."""

    if len(training_data) != 2:
        raise ValueError("training_data must contain (x_train, y_train)")

    if len(evaluation_data) != 3:
        raise ValueError("evaluation_data must contain (x_val, y_val, y_fun_val)")

    if evaluation_every_examples <= 0:
        raise ValueError("evaluation_every_examples must be positive")

    if target_examples_processed <= 0:
        raise ValueError("target_examples_processed must be positive")

    _validate_sgd_optimiser(optimiser)

    x_train, y_train = training_data

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have the same number of rows")

    n_train = int(x_train.shape[0])

    if sampling_method == "random_reshuffling" and target_examples_processed % n_train != 0:
        raise ValueError(
            "random_reshuffling requires target_examples_processed to be a multiple of n_train"
        )

    history: list[dict[str, HistoryValue]] = []
    cumulative_examples = 0
    next_checkpoint_examples = evaluation_every_examples
    total_start_time = time.perf_counter()
    training_elapsed_seconds = 0.0

    batch_stream = iter_batch_indices(
        n_observations=n_train,
        sampling_method=sampling_method,
        batch_size=batch_size,
        target_examples_processed=target_examples_processed,
        sampling_seed=sampling_seed,
    )

    last_update_gradient_norm: float | None = None
    last_batch_loss: float | None = None
    last_actual_batch_size = 0
    last_epoch: int | None = None
    last_step_within_epoch: int | None = None
    step = 0

    history.append(
        evaluate_checkpoint(
            method=method,
            sampling_method=sampling_method,
            step=step,
            epoch=None,
            step_within_epoch=None,
            nominal_batch_size=batch_size,
            actual_batch_size=0,
            checkpoint_examples=0,
            cumulative_examples_processed=0,
            batch_loss=None,
            update_gradient_norm=None,
            model=model,
            loss_function=loss_function,
            training_data=training_data,
            evaluation_data=evaluation_data,
            training_elapsed_seconds=training_elapsed_seconds,
            total_start_time=total_start_time,
        )
    )

    for sampled_batch in batch_stream:
        step += 1
        batch_x = x_train[sampled_batch.indices]
        batch_y = y_train[sampled_batch.indices]

        update_start_time = time.perf_counter()

        model.train()
        optimiser.zero_grad(set_to_none=True)

        batch_predictions = model(batch_x)
        batch_loss = loss_function(batch_predictions, batch_y)
        batch_loss.backward()

        last_update_gradient_norm = gradient_norm(list(model.parameters())).item()
        optimiser.step()

        training_elapsed_seconds += time.perf_counter() - update_start_time

        last_batch_loss = batch_loss.item()
        last_actual_batch_size = sampled_batch.actual_batch_size
        last_epoch = sampled_batch.epoch
        last_step_within_epoch = sampled_batch.step_within_epoch
        cumulative_examples += last_actual_batch_size

        while cumulative_examples >= next_checkpoint_examples:
            history.append(
                evaluate_checkpoint(
                    method=method,
                    sampling_method=sampling_method,
                    step=step,
                    epoch=last_epoch,
                    step_within_epoch=last_step_within_epoch,
                    nominal_batch_size=batch_size,
                    actual_batch_size=last_actual_batch_size,
                    checkpoint_examples=next_checkpoint_examples,
                    cumulative_examples_processed=cumulative_examples,
                    batch_loss=last_batch_loss,
                    update_gradient_norm=last_update_gradient_norm,
                    model=model,
                    loss_function=loss_function,
                    training_data=training_data,
                    evaluation_data=evaluation_data,
                    training_elapsed_seconds=training_elapsed_seconds,
                    total_start_time=total_start_time,
                )
            )
            next_checkpoint_examples += evaluation_every_examples

        if cumulative_examples >= target_examples_processed:
            break

    if history[-1]["checkpoint_examples"] != target_examples_processed:
        history.append(
            evaluate_checkpoint(
                method=method,
                sampling_method=sampling_method,
                step=step,
                epoch=last_epoch,
                step_within_epoch=last_step_within_epoch,
                nominal_batch_size=batch_size,
                actual_batch_size=last_actual_batch_size,
                checkpoint_examples=target_examples_processed,
                cumulative_examples_processed=cumulative_examples,
                batch_loss=last_batch_loss,
                update_gradient_norm=last_update_gradient_norm,
                model=model,
                loss_function=loss_function,
                training_data=training_data,
                evaluation_data=evaluation_data,
                training_elapsed_seconds=training_elapsed_seconds,
                total_start_time=total_start_time,
            )
        )

    return history
