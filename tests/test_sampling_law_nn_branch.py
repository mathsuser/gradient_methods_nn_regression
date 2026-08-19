from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest


def _load_branch_module():
    module_path = Path("experiments/08_sampling_law_nn_branch.py")
    spec = importlib.util.spec_from_file_location(
        "sampling_law_nn_branch_runner",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


branch = _load_branch_module()


def _load_branch_cfg() -> dict:
    return branch.load_json(branch.BRANCH_CONFIG_PATH)


def _load_baseline_cfg() -> dict:
    return branch.load_json(branch.BASELINE_CONFIG_PATH)


def test_branch_config_contains_exactly_wr1_and_rr1() -> None:
    cfg = _load_branch_cfg()
    methods = cfg["experiment"]["methods"]

    assert set(methods) == {"wr_1", "rr_1"}
    assert methods["wr_1"]["sampling_method"] == "single_with_replacement"
    assert methods["rr_1"]["sampling_method"] == "random_reshuffling"


def test_branch_methods_use_batch_size_one() -> None:
    cfg = _load_branch_cfg()

    assert {
        name: int(method_cfg["batch_size"])
        for name, method_cfg in cfg["experiment"]["methods"].items()
    } == {"wr_1": 1, "rr_1": 1}


def test_branch_budget_checkpoint_schedule_and_paths_are_separate_from_v2() -> None:
    cfg = _load_branch_cfg()
    branch.validate_branch_config(cfg)

    assert branch.expected_checkpoint_examples(cfg) == [
        0,
        5000,
        10000,
        15000,
        20000,
        25000,
        30000,
        35000,
        40000,
        45000,
        50000,
    ]
    assert not str(branch.branch_raw_dir(cfg)).startswith("results/raw/week1_gradient_methods")
    assert not str(branch.branch_figures_dir(cfg)).startswith(
        "results/figures/week1_gradient_methods"
    )


def test_branch_run_specs_differ_only_by_sampling_fields_within_trajectory() -> None:
    cfg = _load_branch_cfg()
    specs = branch.build_run_specs(branch_cfg=cfg, trajectory_ids=[0])
    by_method = {spec["method_name"]: spec for spec in specs}

    left = by_method["wr_1"]
    right = by_method["rr_1"]
    differing = {
        key
        for key in left
        if left[key] != right[key]
    }

    assert differing == {"method_name", "sampling_method", "sampling_seed"}


def test_branch_sampling_seed_plan_is_explicit_and_stable() -> None:
    cfg = _load_branch_cfg()
    seed_cfg = cfg["experiment"]["sampling_seeds"]

    assert seed_cfg["trajectory_ids"] == list(range(30))
    assert len(seed_cfg["by_method"]["wr_1"]) == 30
    assert len(seed_cfg["by_method"]["rr_1"]) == 30
    assert seed_cfg["by_method"]["wr_1"] == list(range(91000, 91030))
    assert seed_cfg["by_method"]["rr_1"] == list(range(92000, 92030))


def test_full_branch_run_specs_cover_all_configured_trajectories() -> None:
    cfg = _load_branch_cfg()
    trajectory_ids = [
        int(value)
        for value in cfg["experiment"]["sampling_seeds"]["trajectory_ids"]
    ]

    specs = branch.build_run_specs(branch_cfg=cfg, trajectory_ids=trajectory_ids)

    assert len(specs) == 60
    assert {
        (spec["method_name"], spec["trajectory_id"], spec["sampling_seed"])
        for spec in specs
    } == {
        *{("wr_1", trajectory_id, 91000 + trajectory_id) for trajectory_id in range(30)},
        *{("rr_1", trajectory_id, 92000 + trajectory_id) for trajectory_id in range(30)},
    }


def test_run_level_progress_messages_are_lightweight() -> None:
    spec = {
        "method_name": "rr_1",
        "trajectory_id": 7,
        "sampling_seed": 92007,
    }

    assert branch._format_progress_start(8, 60, spec) == (
        "[8/60] START rr_1 trajectory=7 seed=92007"
    )
    assert branch._format_progress_done(
        run_index=8,
        total_runs=60,
        spec=spec,
        run_seconds=12.34,
        total_elapsed_seconds=120.0,
        eta_seconds=780.0,
    ) == (
        "[8/60] DONE rr_1 trajectory=7\n"
        "run=12.3s\n"
        "total_elapsed=2.0min\n"
        "ETA=13.0min"
    )


def _requires_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not available in this Python environment")


def test_branch_initial_state_checksum_is_identical_for_wr_and_rr_clones() -> None:
    _requires_torch()
    import torch

    from gradient_methods_nn_regression.model import TinyRegressionModel

    cfg = _load_branch_cfg()
    model_seed = int(cfg["experiment"]["training"]["model_seed"])
    torch.manual_seed(model_seed)
    reference_model = TinyRegressionModel()
    reference_state = copy.deepcopy(reference_model.state_dict())
    reference_checksum = branch.state_checksum(reference_state)

    checksums = []
    for _ in cfg["experiment"]["methods"]:
        clone = TinyRegressionModel()
        clone.load_state_dict(reference_state)
        checksums.append(branch.state_checksum(clone.state_dict()))

    assert checksums == [reference_checksum, reference_checksum]


def test_branch_training_and_evaluation_datasets_are_identical_across_run_specs() -> None:
    baseline_cfg = _load_baseline_cfg()
    cfg = _load_branch_cfg()
    specs = branch.build_run_specs(branch_cfg=cfg, trajectory_ids=[0])
    plan = branch.preflight_plan(baseline_cfg=baseline_cfg, branch_cfg=cfg)
    declared_inputs_by_method = {
        spec["method_name"]: {
            "training": plan["baseline_data"]["training"],
            "evaluation": plan["baseline_data"]["evaluation"],
        }
        for spec in specs
    }

    assert len(specs) == 2
    assert plan["baseline_data"]["training"] == Path("data/generated/baseline_train.npz")
    assert plan["baseline_data"]["evaluation"] == Path("data/generated/baseline_test.npz")
    assert declared_inputs_by_method["wr_1"] == declared_inputs_by_method["rr_1"]


def test_tiny_wr1_and_rr1_smoke_runs_complete_with_expected_accounting() -> None:
    _requires_torch()
    import torch

    from gradient_methods_nn_regression.model import TinyRegressionModel
    from gradient_methods_nn_regression.training import train_model

    cfg = _load_branch_cfg()
    method_cfgs = cfg["experiment"]["methods"]
    n_train = 12
    x_train = torch.linspace(-1.0, 1.0, n_train * 6, dtype=torch.float64).reshape(
        n_train,
        6,
    )
    y_train = torch.sum(x_train, dim=1, keepdim=True)
    x_eval = x_train.clone()
    f_eval = y_train.clone()
    y_eval = y_train.clone()

    torch.manual_seed(0)
    reference_model = TinyRegressionModel().to(dtype=torch.float64)
    reference_state = copy.deepcopy(reference_model.state_dict())

    for method_key, method_cfg in method_cfgs.items():
        model = TinyRegressionModel().to(dtype=torch.float64)
        model.load_state_dict(reference_state)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
        history = train_model(
            model=model,
            optimiser=optimiser,
            loss_function=torch.nn.functional.mse_loss,
            training_data=(x_train, y_train),
            evaluation_data=(x_eval, y_eval, f_eval),
            method=method_cfg["method_name"],
            sampling_method=method_cfg["sampling_method"],
            batch_size=int(method_cfg["batch_size"]),
            target_examples_processed=n_train,
            sampling_seed=branch.sampling_seed_for(
                branch_cfg=cfg,
                method_name=method_key,
                trajectory_id=0,
            ),
            evaluation_every_examples=1,
        )

        final = history[-1]
        assert final["step"] == n_train
        assert final["cumulative_examples_processed"] == n_train
        assert final["actual_batch_size"] == 1
        assert final["data_equivalent_passes"] == pytest.approx(1.0)
        assert "validation_function_mse" in final
        assert "full_gradient_norm" in final
        assert math.isfinite(float(final["validation_function_mse"]))
        assert math.isfinite(float(final["full_gradient_norm"]))
