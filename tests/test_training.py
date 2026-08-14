from __future__ import annotations

import copy

import pytest
import torch

from gradient_methods_nn_regression.training import (
    evaluate_checkpoint,
    iter_batch_indices,
    train_model,
)


CANONICAL_CHECKPOINT_KEYS = {
    "method",
    "sampling_method",
    "step",
    "epoch",
    "step_within_epoch",
    "nominal_batch_size",
    "actual_batch_size",
    "checkpoint_examples",
    "cumulative_examples_processed",
    "data_equivalent_passes",
    "batch_loss",
    "training_mse",
    "validation_mse",
    "validation_function_mse",
    "update_gradient_norm",
    "full_gradient_norm",
    "parameter_norm",
    "training_elapsed_seconds",
    "total_elapsed_seconds",
}


SMOKE_METHOD_CONFIGS = [
    ("full_batch_gd", "full_batch", 256),
    ("single_observation_sgd", "single_with_replacement", 1),
    ("single_observation_random_reshuffling", "random_reshuffling", 1),
    ("minibatch_with_replacement_b32", "minibatch_with_replacement", 32),
    ("random_reshuffling_b32", "random_reshuffling", 32),
    ("minibatch_with_replacement_b256", "minibatch_with_replacement", 256),
    ("random_reshuffling_b256", "random_reshuffling", 256),
]


def _make_reference_model(*, input_dim: int, hidden_width: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_width),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_width, 1),
    )


def _assert_state_dicts_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _make_small_training_case(
    *,
    n_train: int,
    d: int = 6,
    hidden_width: int = 4,
    noise_std: float = 0.3,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.nn.Module,
]:
    torch.manual_seed(seed)

    x_train = torch.randn(n_train, d)
    true_weights = torch.linspace(1.0, 2.0, d).reshape(d, 1)
    y_train = x_train @ true_weights + noise_std * torch.randn(n_train, 1)

    x_eval = torch.randn(max(8, n_train // 4), d)
    y_fun_eval = x_eval @ true_weights
    y_eval = y_fun_eval + noise_std * torch.randn(x_eval.shape[0], 1)

    model = _make_reference_model(
        input_dim=d,
        hidden_width=hidden_width,
        seed=seed,
    )

    return x_train, y_train, x_eval, y_eval, y_fun_eval, model


def _make_sgd(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
    )


def test_paired_initialisation_uses_loaded_state_not_construction_seed() -> None:
    reference = _make_reference_model(input_dim=6, hidden_width=4, seed=11)
    reference_state = copy.deepcopy(reference.state_dict())

    clones: list[torch.nn.Module] = []
    for seed in [21, 22, 23, 24, 25, 26]:
        clone = _make_reference_model(input_dim=6, hidden_width=4, seed=seed)
        clone.load_state_dict(reference_state)
        clones.append(clone)

    for clone in clones:
        _assert_state_dicts_equal(reference_state, clone.state_dict())


def test_full_batch_iterator_uses_all_observations_in_fixed_order() -> None:
    batches = list(
        iter_batch_indices(
            n_observations=12,
            sampling_method="full_batch",
            batch_size=12,
            target_examples_processed=24,
            sampling_seed=7,
        )
    )

    assert len(batches) == 2
    assert [batch.actual_batch_size for batch in batches] == [12, 12]
    assert [batch.epoch for batch in batches] == [1, 2]
    assert [batch.step_within_epoch for batch in batches] == [1, 1]
    for batch in batches:
        assert torch.equal(batch.indices, torch.arange(12))


def test_single_with_replacement_reproducibility_and_metadata() -> None:
    first = list(
        iter_batch_indices(
            n_observations=4,
            sampling_method="single_with_replacement",
            batch_size=1,
            target_examples_processed=5,
            sampling_seed=7,
        )
    )
    second = list(
        iter_batch_indices(
            n_observations=4,
            sampling_method="single_with_replacement",
            batch_size=1,
            target_examples_processed=5,
            sampling_seed=7,
        )
    )
    different = list(
        iter_batch_indices(
            n_observations=4,
            sampling_method="single_with_replacement",
            batch_size=1,
            target_examples_processed=5,
            sampling_seed=8,
        )
    )

    assert len(first) == 5
    assert all(batch.actual_batch_size == 1 for batch in first)
    assert all(batch.epoch is None for batch in first)
    assert all(batch.step_within_epoch is None for batch in first)

    concatenated = torch.cat([batch.indices for batch in first])
    assert concatenated.shape == (5,)
    assert torch.all(concatenated >= 0)
    assert torch.all(concatenated <= 3)

    assert [batch.indices.tolist() for batch in first] == [
        batch.indices.tolist() for batch in second
    ]
    assert [batch.indices.tolist() for batch in first] != [
        batch.indices.tolist() for batch in different
    ]


def test_minibatch_with_replacement_truncates_final_batch_to_hit_budget() -> None:
    batches = list(
        iter_batch_indices(
            n_observations=12,
            sampling_method="minibatch_with_replacement",
            batch_size=3,
            target_examples_processed=8,
            sampling_seed=7,
        )
    )

    assert [batch.actual_batch_size for batch in batches] == [3, 3, 2]
    assert all(batch.epoch is None for batch in batches)
    assert all(batch.step_within_epoch is None for batch in batches)
    concatenated = torch.cat([batch.indices for batch in batches])
    assert torch.all(concatenated >= 0)
    assert torch.all(concatenated < 12)


def test_minibatch_with_replacement_can_repeat_indices_within_batch() -> None:
    batches = list(
        iter_batch_indices(
            n_observations=2,
            sampling_method="minibatch_with_replacement",
            batch_size=4,
            target_examples_processed=4,
            sampling_seed=7,
        )
    )
    assert len(batches) == 1
    assert torch.unique(batches[0].indices).numel() < batches[0].indices.numel()


def test_random_reshuffling_covers_each_epoch_once_and_is_reproducible() -> None:
    first = list(
        iter_batch_indices(
            n_observations=5,
            sampling_method="random_reshuffling",
            batch_size=2,
            target_examples_processed=10,
            sampling_seed=7,
        )
    )
    second = list(
        iter_batch_indices(
            n_observations=5,
            sampling_method="random_reshuffling",
            batch_size=2,
            target_examples_processed=10,
            sampling_seed=7,
        )
    )

    assert [batch.indices.tolist() for batch in first] == [
        batch.indices.tolist() for batch in second
    ]
    assert len(first) == 6
    assert [batch.actual_batch_size for batch in first] == [2, 2, 1, 2, 2, 1]
    assert [batch.epoch for batch in first] == [1, 1, 1, 2, 2, 2]
    assert [batch.step_within_epoch for batch in first] == [1, 2, 3, 1, 2, 3]

    epoch1 = torch.cat([batch.indices for batch in first[:3]])
    epoch2 = torch.cat([batch.indices for batch in first[3:]])
    assert torch.equal(torch.sort(epoch1).values, torch.arange(5))
    assert torch.equal(torch.sort(epoch2).values, torch.arange(5))


def test_random_reshuffling_b1_covers_one_complete_epoch() -> None:
    batches = list(
        iter_batch_indices(
            n_observations=5,
            sampling_method="random_reshuffling",
            batch_size=1,
            target_examples_processed=5,
            sampling_seed=7,
        )
    )

    assert len(batches) == 5
    assert all(batch.actual_batch_size == 1 for batch in batches)
    assert [batch.epoch for batch in batches] == [1, 1, 1, 1, 1]
    assert [batch.step_within_epoch for batch in batches] == [1, 2, 3, 4, 5]

    concatenated = torch.cat([batch.indices for batch in batches])
    assert torch.equal(torch.sort(concatenated).values, torch.arange(5))


def test_random_reshuffling_b1_covers_two_complete_epochs() -> None:
    batches = list(
        iter_batch_indices(
            n_observations=5,
            sampling_method="random_reshuffling",
            batch_size=1,
            target_examples_processed=10,
            sampling_seed=7,
        )
    )

    assert len(batches) == 10
    assert all(batch.actual_batch_size == 1 for batch in batches)
    assert [batch.epoch for batch in batches[:5]] == [1, 1, 1, 1, 1]
    assert [batch.epoch for batch in batches[5:]] == [2, 2, 2, 2, 2]

    epoch1 = torch.cat([batch.indices for batch in batches[:5]])
    epoch2 = torch.cat([batch.indices for batch in batches[5:]])
    assert torch.equal(torch.sort(epoch1).values, torch.arange(5))
    assert torch.equal(torch.sort(epoch2).values, torch.arange(5))


def test_random_reshuffling_b1_reproducibility_with_identical_seed() -> None:
    first = list(
        iter_batch_indices(
            n_observations=5,
            sampling_method="random_reshuffling",
            batch_size=1,
            target_examples_processed=10,
            sampling_seed=7,
        )
    )
    second = list(
        iter_batch_indices(
            n_observations=5,
            sampling_method="random_reshuffling",
            batch_size=1,
            target_examples_processed=10,
            sampling_seed=7,
        )
    )

    assert [batch.indices.tolist() for batch in first] == [
        batch.indices.tolist() for batch in second
    ]


@pytest.mark.parametrize(
    (
        "method",
        "sampling_method",
        "batch_size",
        "target_examples_processed",
        "expected_steps",
        "expected_examples",
        "expected_epoch",
        "expected_step_within_epoch",
    ),
    [
        ("full_batch_gd", "full_batch", 12, 24, 2, 24, 2, 1),
        (
            "single_observation_sgd",
            "single_with_replacement",
            1,
            5,
            5,
            5,
            None,
            None,
        ),
        (
            "minibatch_with_replacement",
            "minibatch_with_replacement",
            3,
            5,
            2,
            5,
            None,
            None,
        ),
        ("random_reshuffling", "random_reshuffling", 3, 24, 8, 24, 2, 4),
        ("random_reshuffling_b1", "random_reshuffling", 1, 12, 12, 12, 1, 12),
    ],
)
def test_train_model_accounting_and_canonical_schema(
    method: str,
    sampling_method: str,
    batch_size: int,
    target_examples_processed: int,
    expected_steps: int,
    expected_examples: int,
    expected_epoch: int | None,
    expected_step_within_epoch: int | None,
) -> None:
    n_train = 12
    x_train, y_train, x_eval, y_eval, y_fun_eval, model = _make_small_training_case(
        n_train=n_train,
        seed=0,
    )
    optimiser = _make_sgd(model)

    history = train_model(
        model=model,
        optimiser=optimiser,
        loss_function=torch.nn.functional.mse_loss,
        training_data=(x_train, y_train),
        evaluation_data=(x_eval, y_eval, y_fun_eval),
        method=method,
        sampling_method=sampling_method,
        batch_size=batch_size,
        target_examples_processed=target_examples_processed,
        sampling_seed=7,
        evaluation_every_examples=1,
    )

    assert history

    first = history[0]
    assert set(first.keys()) == CANONICAL_CHECKPOINT_KEYS
    assert first["step"] == 0
    assert first["checkpoint_examples"] == 0
    assert first["cumulative_examples_processed"] == 0
    assert first["batch_loss"] is None
    assert first["update_gradient_norm"] is None
    assert first["epoch"] is None
    assert first["step_within_epoch"] is None
    assert torch.isfinite(torch.tensor(first["training_mse"]))
    assert torch.isfinite(torch.tensor(first["validation_mse"]))
    assert torch.isfinite(torch.tensor(first["validation_function_mse"]))
    assert torch.isfinite(torch.tensor(first["full_gradient_norm"]))
    assert torch.isfinite(torch.tensor(first["parameter_norm"]))

    for record in history[1:]:
        assert set(record.keys()) == CANONICAL_CHECKPOINT_KEYS

    for record in history:
        assert record["training_elapsed_seconds"] >= 0.0
        assert record["total_elapsed_seconds"] >= 0.0
        assert record["total_elapsed_seconds"] >= record["training_elapsed_seconds"]

    for left, right in zip(history, history[1:]):
        assert right["training_elapsed_seconds"] >= left["training_elapsed_seconds"]
        assert right["total_elapsed_seconds"] >= left["total_elapsed_seconds"]

    final = history[-1]
    assert final["method"] == method
    assert final["sampling_method"] == sampling_method
    assert final["step"] == expected_steps
    assert final["cumulative_examples_processed"] == expected_examples
    assert final["epoch"] == expected_epoch
    assert final["step_within_epoch"] == expected_step_within_epoch
    assert final["nominal_batch_size"] == batch_size
    assert final["data_equivalent_passes"] == pytest.approx(expected_examples / n_train)


def test_train_model_random_reshuffling_b1_accounting_per_update() -> None:
    n_train = 12
    x_train, y_train, x_eval, y_eval, y_fun_eval, model = _make_small_training_case(
        n_train=n_train,
        seed=0,
    )
    optimiser = _make_sgd(model)

    history = train_model(
        model=model,
        optimiser=optimiser,
        loss_function=torch.nn.functional.mse_loss,
        training_data=(x_train, y_train),
        evaluation_data=(x_eval, y_eval, y_fun_eval),
        method="random_reshuffling_b1",
        sampling_method="random_reshuffling",
        batch_size=1,
        target_examples_processed=n_train,
        sampling_seed=7,
        evaluation_every_examples=1,
    )

    update_rows = history[1:]
    assert len(update_rows) == n_train
    for record in update_rows:
        assert record["actual_batch_size"] == 1
        assert record["step"] == record["cumulative_examples_processed"]
        assert record["epoch"] == 1
        assert record["step_within_epoch"] == record["step"]
        assert record["data_equivalent_passes"] == pytest.approx(
            record["cumulative_examples_processed"] / n_train
        )

    final = history[-1]
    assert final["step"] == n_train
    assert final["cumulative_examples_processed"] == n_train
    assert final["data_equivalent_passes"] == pytest.approx(1.0)


@pytest.mark.parametrize("start_in_training_mode", [True, False])
def test_checkpoint_evaluation_does_not_change_parameters(
    start_in_training_mode: bool,
) -> None:
    x_train, y_train, x_eval, y_eval, y_fun_eval, model = _make_small_training_case(
        n_train=12,
        seed=0,
    )

    model.train(start_in_training_mode)
    assert model.training is start_in_training_mode

    before_parameters = copy.deepcopy(model.state_dict())

    record = evaluate_checkpoint(
        method="train_model",
        sampling_method="full_batch",
        step=0,
        epoch=None,
        step_within_epoch=None,
        nominal_batch_size=12,
        actual_batch_size=0,
        checkpoint_examples=0,
        cumulative_examples_processed=0,
        batch_loss=None,
        update_gradient_norm=None,
        model=model,
        loss_function=torch.nn.functional.mse_loss,
        training_data=(x_train, y_train),
        evaluation_data=(x_eval, y_eval, y_fun_eval),
        training_elapsed_seconds=0.0,
        total_start_time=0.0,
    )

    _assert_state_dicts_equal(before_parameters, model.state_dict())
    assert model.training is start_in_training_mode
    assert torch.isfinite(torch.tensor(record["full_gradient_norm"]))
    assert record["full_gradient_norm"] >= 0.0


@pytest.mark.parametrize(
    ("method", "sampling_method", "batch_size"),
    SMOKE_METHOD_CONFIGS,
)
def test_shared_train_model_smoke_across_retained_method_configs(
    method: str,
    sampling_method: str,
    batch_size: int,
) -> None:
    n_train = 256
    target_examples_processed = 512
    x_train, y_train, x_eval, y_eval, y_fun_eval, model = _make_small_training_case(
        n_train=n_train,
        seed=123,
    )
    optimiser = _make_sgd(model)

    before = copy.deepcopy(model.state_dict())

    history = train_model(
        model=model,
        optimiser=optimiser,
        loss_function=torch.nn.functional.mse_loss,
        training_data=(x_train, y_train),
        evaluation_data=(x_eval, y_eval, y_fun_eval),
        method=method,
        sampling_method=sampling_method,
        batch_size=batch_size,
        target_examples_processed=target_examples_processed,
        sampling_seed=13,
        evaluation_every_examples=128,
    )

    assert history
    assert history[-1]["cumulative_examples_processed"] == target_examples_processed
    for record in history:
        assert set(record.keys()) == CANONICAL_CHECKPOINT_KEYS
        assert torch.isfinite(torch.tensor(record["training_mse"]))
        assert torch.isfinite(torch.tensor(record["validation_mse"]))
        assert torch.isfinite(torch.tensor(record["validation_function_mse"]))
        assert torch.isfinite(torch.tensor(record["full_gradient_norm"]))
        assert torch.isfinite(torch.tensor(record["parameter_norm"]))

    assert any(
        not torch.equal(before[name], model.state_dict()[name])
        for name in before
    )


@pytest.mark.parametrize(
    "build_optimiser, expected_error",
    [
        (
            lambda model: torch.optim.Adam(model.parameters(), lr=0.01),
            TypeError,
        ),
        (
            lambda model: torch.optim.SGD(
                model.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=0.0,
            ),
            ValueError,
        ),
        (
            lambda model: torch.optim.SGD(
                model.parameters(),
                lr=0.01,
                momentum=0.0,
                weight_decay=0.1,
            ),
            ValueError,
        ),
    ],
)
def test_train_model_rejects_non_plain_sgd(
    build_optimiser,
    expected_error: type[Exception],
) -> None:
    x_train, y_train, x_eval, y_eval, y_fun_eval, model = _make_small_training_case(
        n_train=12,
        seed=0,
    )
    optimiser = build_optimiser(model)

    with pytest.raises(expected_error):
        train_model(
            model=model,
            optimiser=optimiser,
            loss_function=torch.nn.functional.mse_loss,
            training_data=(x_train, y_train),
            evaluation_data=(x_eval, y_eval, y_fun_eval),
            method="constraint_check",
            sampling_method="minibatch_with_replacement",
            batch_size=3,
            target_examples_processed=6,
            sampling_seed=7,
            evaluation_every_examples=1,
        )


def test_random_reshuffling_accounting_for_complete_epochs() -> None:
    batches = list(
        iter_batch_indices(
            n_observations=10,
            sampling_method="random_reshuffling",
            batch_size=4,
            target_examples_processed=20,
            sampling_seed=7,
        )
    )

    assert len(batches) == 6
    assert [batch.actual_batch_size for batch in batches] == [4, 4, 2, 4, 4, 2]
    assert sum(batch.actual_batch_size for batch in batches) == 20
    assert [batch.epoch for batch in batches] == [1, 1, 1, 2, 2, 2]
    assert [batch.step_within_epoch for batch in batches] == [1, 2, 3, 1, 2, 3]
