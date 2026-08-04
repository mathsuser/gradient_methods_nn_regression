# Training helpers for the baseline regression experiment.
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterator, Literal
from torch.utils.data import DataLoader

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
    epoch: int | None
    step_within_epoch: int | None





def iter_batch_indices(
    *,
    n_observations: int,
    sampling_method: SamplingMethod,
    batch_size: int,
    step_budget: int,
    sampling_seed: int,
) -> Iterator[SampledBatch]:
    """Yield batch indices according to the selected sampling method."""

    if n_observations <= 0:
        raise ValueError("n_observations must be positive")

    if step_budget <= 0:
        raise ValueError("step_budget must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    generator = torch.Generator()
    generator.manual_seed(sampling_seed)

    all_indices = torch.arange(n_observations)

    if sampling_method == "full_batch":
        if batch_size != n_observations:
            raise ValueError(
                "full_batch requires batch_size == n_observations"
            )

        for step in range(1, step_budget + 1):
            yield SampledBatch(
                indices=all_indices,
                epoch=step,
                step_within_epoch=1,
            )
        return

    if sampling_method == "single_with_replacement":
        if batch_size != 1:
            raise ValueError(
                "single_with_replacement requires batch_size == 1"
            )

        for _ in range(step_budget):
            indices = torch.randint(
                low=0,
                high=n_observations,
                size=(1,),
                generator=generator,
            )

            yield SampledBatch(
                indices=indices,
                epoch=None,
                step_within_epoch=None,
            )
        return

    if sampling_method == "minibatch_with_replacement":
        if batch_size <= 1:
            raise ValueError(
                "minibatch_with_replacement requires batch_size > 1"
            )

        for _ in range(step_budget):
            indices = torch.randint(
                low=0,
                high=n_observations,
                size=(batch_size,),
                generator=generator,
            )

            yield SampledBatch(
                indices=indices,
                epoch=None,
                step_within_epoch=None,
            )
        return

    if sampling_method == "random_reshuffling":
        if batch_size > n_observations:
            raise ValueError(
                "random_reshuffling requires batch_size <= n_observations"
            )

        produced_steps = 0
        epoch = 0

        while produced_steps < step_budget:
            epoch += 1

            permutation = torch.randperm(
                n_observations,
                generator=generator,
            )

            for batch_number, start in enumerate(
                range(0, n_observations, batch_size),
                start=1,
            ):
                if produced_steps >= step_budget:
                    return

                produced_steps += 1

                yield SampledBatch(
                    indices=permutation[start : start + batch_size],
                    epoch=epoch,
                    step_within_epoch=batch_number,
                )
        return

    raise ValueError(f"Unsupported sampling method: {sampling_method}")





HistoryValue = float | int | str | None
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def train_model(
    *,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    loss_function: LossFunction,
    training_data: tuple[torch.Tensor, torch.Tensor],
    sampling_method: SamplingMethod,
    batch_size: int,
    step_budget: int,
    sampling_seed: int,
    evaluation_every_examples: int,
    evaluation_data: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> list[dict[str, HistoryValue]]:
    """Train using an explicit sampling method and record metrics."""

    if len(training_data) != 2:
        raise ValueError("training_data must contain (x_train, y_train)")

    if len(evaluation_data) != 3:
        raise ValueError(
            "evaluation_data must contain (x_val, y_val, y_fun_val)"
        )

    if evaluation_every_examples <= 0:
        raise ValueError(
            "evaluation_every_examples must be positive"
        )

    x_train, y_train = training_data
    x_val, y_val, y_fun_val = evaluation_data

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "x_train and y_train must have the same number of rows"
        )

    n_train = int(x_train.shape[0])

    history: list[dict[str, HistoryValue]] = []
    cumulative_examples = 0
    next_evaluation = evaluation_every_examples
    start_time = time.perf_counter()

    batch_stream = iter_batch_indices(
        n_observations=n_train,
        sampling_method=sampling_method,
        batch_size=batch_size,
        step_budget=step_budget,
        sampling_seed=sampling_seed,
    )

    for step, sampled_batch in enumerate(batch_stream, start=1):
        batch_x = x_train[sampled_batch.indices]
        batch_y = y_train[sampled_batch.indices]

        model.train()
        optimiser.zero_grad(set_to_none=True)

        batch_predictions = model(batch_x)
        batch_loss = loss_function(batch_predictions, batch_y)

        batch_loss.backward()

        update_gradient_norm = gradient_norm(
            list(model.parameters())
        ).item()

        optimiser.step()

        cumulative_examples += int(batch_x.shape[0])

        should_evaluate = (
            cumulative_examples >= next_evaluation
            or step == step_budget
        )

        if not should_evaluate:
            continue

        model.eval()

        with torch.no_grad():
            # Evaluate the empirical training loss on the full dataset.
            training_predictions = model(x_train)
            training_loss = loss_function(
                training_predictions,
                y_train,
            ).item()

            validation_predictions = model(x_val)
            validation_loss = loss_function(
                validation_predictions,
                y_val,
            ).item()

            validation_function_mse = function_estimation_mse(
                validation_predictions,
                y_fun_val,
            ).item()

        history.append(
            {
                "sampling_method": sampling_method,
                "step": step,
                "epoch": sampled_batch.epoch,
                "step_within_epoch": sampled_batch.step_within_epoch,
                "batch_size": int(batch_x.shape[0]),
                "cumulative_examples_processed": cumulative_examples,
                "data_equivalent_passes": (
                    cumulative_examples / n_train
                ),
                "batch_loss": batch_loss.item(),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_function_mse": (
                    validation_function_mse
                ),
                "update_gradient_norm": update_gradient_norm,
                "parameter_norm": parameter_norm(
                    list(model.parameters())
                ).item(),
                "elapsed_time": time.perf_counter() - start_time,
            }
        )

        while cumulative_examples >= next_evaluation:
            next_evaluation += evaluation_every_examples

    return history