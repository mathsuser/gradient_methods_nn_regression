import torch

from gradient_methods_nn_regression.training import (
    iter_batch_indices,
)


def test_full_batch_uses_every_observation_at_each_step() -> None:
    batches = list(
        iter_batch_indices(
            n_observations=12,
            sampling_method="full_batch",
            batch_size=12,
            step_budget=2,
            sampling_seed=7,
        )
    )

    assert len(batches) == 2

    for step, batch in enumerate(batches, start=1):
        assert torch.equal(
            batch.indices,
            torch.arange(12),
        )
        assert batch.epoch == step
        assert batch.step_within_epoch == 1


def test_single_with_replacement_draws_one_index_per_step() -> None:
    batches = list(
        iter_batch_indices(
            n_observations=4,
            sampling_method="single_with_replacement",
            batch_size=1,
            step_budget=5,
            sampling_seed=7,
        )
    )

    indices = torch.cat([batch.indices for batch in batches])

    assert indices.shape == (5,)
    assert torch.all(indices >= 0)
    assert torch.all(indices < 4)

    # Five draws from four possible indices must contain a duplicate.
    assert torch.unique(indices).numel() < 5

    assert all(batch.epoch is None for batch in batches)


def test_minibatch_with_replacement_respects_batch_size() -> None:
    batches = list(
        iter_batch_indices(
            n_observations=12,
            sampling_method="minibatch_with_replacement",
            batch_size=3,
            step_budget=4,
            sampling_seed=7,
        )
    )

    assert len(batches) == 4
    assert all(batch.indices.shape == (3,) for batch in batches)
    assert all(batch.epoch is None for batch in batches)


def test_random_reshuffling_uses_each_index_once_per_epoch() -> None:
    batches = list(
        iter_batch_indices(
            n_observations=12,
            sampling_method="random_reshuffling",
            batch_size=3,
            step_budget=8,
            sampling_seed=7,
        )
    )

    first_epoch = torch.cat(
        [batch.indices for batch in batches[:4]]
    )
    second_epoch = torch.cat(
        [batch.indices for batch in batches[4:]]
    )

    assert torch.equal(
        torch.sort(first_epoch).values,
        torch.arange(12),
    )
    assert torch.equal(
        torch.sort(second_epoch).values,
        torch.arange(12),
    )

    assert [batch.epoch for batch in batches] == [
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]

import pytest
import torch

from gradient_methods_nn_regression.training import train_model


@pytest.mark.parametrize(
    (
        "sampling_method",
        "batch_size",
        "step_budget",
        "expected_examples",
        "expected_epoch",
    ),
    [
        ("full_batch", 12, 2, 24, 2),
        ("single_with_replacement", 1, 5, 5, None),
        ("minibatch_with_replacement", 3, 4, 12, None),
        ("random_reshuffling", 3, 8, 24, 2),
    ],
)
def test_train_model_records_sampling_and_accounting(
    sampling_method: str,
    batch_size: int,
    step_budget: int,
    expected_examples: int,
    expected_epoch: int | None,
) -> None:
    torch.manual_seed(0)

    x_train = torch.randn(12, 6)
    y_train = torch.sin(x_train[:, 0]).unsqueeze(1)

    x_val = torch.randn(6, 6)
    y_val = torch.cos(x_val[:, 1]).unsqueeze(1)
    y_fun_val = torch.sin(x_val[:, 0]).unsqueeze(1)

    model = torch.nn.Sequential(
        torch.nn.Linear(6, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    )

    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    history = train_model(
        model=model,
        optimiser=optimiser,
        loss_function=torch.nn.functional.mse_loss,
        training_data=(x_train, y_train),
        sampling_method=sampling_method,
        batch_size=batch_size,
        step_budget=step_budget,
        sampling_seed=7,
        evaluation_every_examples=1,
        evaluation_data=(x_val, y_val, y_fun_val),
    )

    final_record = history[-1]

    assert final_record["sampling_method"] == sampling_method
    assert final_record["step"] == step_budget
    assert (
        final_record["cumulative_examples_processed"]
        == expected_examples
    )
    assert final_record["epoch"] == expected_epoch

    assert final_record["training_loss"] >= 0.0
    assert final_record["validation_loss"] >= 0.0
    assert final_record["validation_function_mse"] >= 0.0
    assert final_record["update_gradient_norm"] >= 0.0
    assert final_record["parameter_norm"] >= 0.0
    assert final_record["elapsed_time"] >= 0.0    