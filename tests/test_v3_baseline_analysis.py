from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


def _load_module():
    module_path = Path("experiments/12_v3_analyse_baseline.py")
    spec = importlib.util.spec_from_file_location("v3_baseline_analysis", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3a = _load_module()


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_history(path: Path, *, bad_schedule: bool = False, nonfinite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = v3a.expected_checkpoint_schedule()
    if bad_schedule:
        checkpoints = checkpoints[:-1]
    fieldnames = [
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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, checkpoint in enumerate(checkpoints):
            value = "nan" if nonfinite and index == 4 else 1.0 / (index + 1)
            writer.writerow(
                {
                    "method": "method",
                    "sampling_method": "sampling",
                    "step": checkpoint,
                    "epoch": "",
                    "step_within_epoch": "",
                    "nominal_batch_size": 1,
                    "actual_batch_size": 1 if checkpoint else 0,
                    "checkpoint_examples": checkpoint,
                    "cumulative_examples_processed": checkpoint,
                    "data_equivalent_passes": checkpoint / 5000,
                    "batch_loss": "" if checkpoint == 0 else value,
                    "training_mse": value,
                    "validation_mse": value,
                    "validation_function_mse": value,
                    "update_gradient_norm": "" if checkpoint == 0 else value,
                    "full_gradient_norm": value,
                    "parameter_norm": value,
                    "training_elapsed_seconds": float(index),
                    "total_elapsed_seconds": float(index),
                }
            )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, corrupt: str | None = None) -> dict[str, Any]:
    lr_path = tmp_path / "learning_rate_selection.json"
    lr_data = {
        "human_approved": corrupt != "human_approval",
        "selected_common_learning_rate": 0.03,
    }
    _write_json(lr_data, lr_path)
    monkeypatch.setattr(v3a, "LR_DECISION_PATH", lr_path)

    baseline_cfg = {
        "dataset": {
            "n_train": 5000,
            "n_validation": 2000,
            "n_test": 20000,
            "noise_std": 0.3,
            "dtype": "float64",
        }
    }
    v3_cfg = {
        "experiment": {
            "future_baseline": {
                "sampling_seeds_by_method_and_model_seed": {
                    method: {
                        str(seed): 50000 + method_index * 1000 + seed
                        for seed in v3a.MODEL_SEEDS
                    }
                    for method_index, method in enumerate(v3a.METHOD_ORDER)
                }
            }
        }
    }
    manifest = {
        "expected_run_count": 35,
        "actual_completed_run_count": 35,
        "method_definitions": {
            method: {
                "method_name": method,
                "sampling_method": v3a.EXPECTED_METHOD_MAPPING[method][0],
                "batch_size": v3a.EXPECTED_METHOD_MAPPING[method][1],
            }
            for method in v3a.METHOD_ORDER
        },
        "model_seeds": v3a.MODEL_SEEDS,
        "learning_rate": 0.01 if corrupt == "wrong_lr" else 0.03,
        "human_approval_confirmed": corrupt != "human_approval",
        "learning_rate_decision_artifact_sha256": (
            "bad" if corrupt == "lr_sha" else v3a.file_sha256(lr_path)
        ),
        "target_examples_processed": 500000,
        "data_equivalent_passes": 100.0,
        "evaluation_every_examples": 5000,
        "checkpoint_schedule": v3a.expected_checkpoint_schedule(),
        "worktree_status": "dirty",
        "common_config": {
            "data_provenance": {
                "train": {"path": "train.npz", "sha256": "train"},
                "validation": {"path": "val.npz", "sha256": "val"},
                "test": {"path": "test.npz", "sha256": "test"},
            }
        },
        "run_metadata_paths": [],
    }
    run_root = tmp_path / "runs"
    for method_index, method in enumerate(v3a.METHOD_ORDER):
        sampling_method, batch_size, _ = v3a.EXPECTED_METHOD_MAPPING[method]
        for seed in v3a.MODEL_SEEDS:
            run_dir = run_root / method / f"model_seed_{seed}"
            metadata_path = run_dir / "metadata.json"
            history_path = run_dir / "history.csv"
            bad_cell = method == "wr_b1" and seed == 0
            metadata_method = "full_batch_gd" if corrupt == "duplicate" and bad_cell else method
            sampling_seed = int(v3_cfg["experiment"]["future_baseline"]["sampling_seeds_by_method_and_model_seed"][method][str(seed)])
            if corrupt == "wrong_seed" and bad_cell:
                sampling_seed += 1
            checksum = f"checksum-{seed}"
            if corrupt == "checksum" and method == "wr_b1" and seed == 0:
                checksum = "different"
            metadata = {
                "method_name": metadata_method,
                "sampling_method": sampling_method,
                "nominal_batch_size": batch_size,
                "model_seed": seed,
                "sampling_seed": sampling_seed,
                "learning_rate": 0.03,
                "learning_rate_decision_artifact_sha256": v3a.file_sha256(lr_path),
                "target_examples_processed": 500000,
                "achieved_examples_processed": 499000 if corrupt == "terminal" and bad_cell else 500000,
                "achieved_data_equivalent_passes": 100.0,
                "evaluation_every_examples": 5000,
                "dtype": "float64",
                "device": "cpu",
                "initial_state_checksum": checksum,
                "loaded_initial_state_checksum": checksum,
                "data_provenance": {"file_hashes": manifest["common_config"]["data_provenance"]},
                "runtime": {"run_elapsed_seconds": 1.0},
                "final_split_metrics": {
                    "training_prediction_mse": 0.2,
                    "training_function_mse": 0.1,
                    "validation_prediction_mse": 0.2,
                    "validation_function_mse": 0.1,
                    "test_prediction_mse": 0.2,
                    "test_function_mse": 0.1,
                    "parameter_norm": 1.0,
                },
                "artifacts": {"history_csv": str(history_path), "metadata_json": str(metadata_path)},
            }
            _write_json(metadata, metadata_path)
            _write_history(
                history_path,
                bad_schedule=corrupt == "schedule" and bad_cell,
                nonfinite=corrupt == "nonfinite" and bad_cell,
            )
            if not (corrupt == "missing" and bad_cell):
                manifest["run_metadata_paths"].append(str(metadata_path))
    return {
        "baseline_cfg": baseline_cfg,
        "v3_cfg": v3_cfg,
        "lr_decision": lr_data,
        "manifest": manifest,
        "manifest_path": tmp_path / "manifest.json",
    }


def _validate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    report, rows, histories = v3a.validate_integrity(**fixture)
    report["rows"] = rows
    report["histories"] = histories
    return report


def test_exactly_35_expected_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = _validate_fixture(_fixture(tmp_path, monkeypatch))

    assert report["integrity_status"] == "passed"
    assert report["actual_completed_run_count"] == 35


@pytest.mark.parametrize(
    "corrupt, message",
    [
        ("missing", "35 run metadata paths"),
        ("duplicate", "Duplicate method/model-seed cell"),
        ("wrong_seed", "wrong explicit sampling seed"),
        ("wrong_lr", "manifest learning_rate"),
        ("human_approval", "human_approval"),
        ("lr_sha", "SHA-256"),
        ("checksum", "Initial-state checksum"),
        ("schedule", "wrong checkpoint schedule"),
        ("terminal", "terminal examples"),
        ("nonfinite", "non-finite"),
    ],
)
def test_integrity_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corrupt: str, message: str) -> None:
    report = _validate_fixture(_fixture(tmp_path, monkeypatch, corrupt=corrupt))

    assert report["integrity_status"] == "failed"
    assert any(message in error for error in report["errors"])


def test_sampling_law_difference_sign_is_rr_minus_wr() -> None:
    rows = [
        {"method_name": "wr_b1", "model_seed": 0, "test_function_mse": 2.0, "training_mse": 2.0, "validation_mse": 2.0, "test_noisy_mse": 2.0, "generalisation_gap": 0.0, "full_gradient_norm": 2.0, "update_gradient_norm": 2.0, "parameter_norm": 2.0},
        {"method_name": "rr_b1", "model_seed": 0, "test_function_mse": 5.0, "training_mse": 5.0, "validation_mse": 5.0, "test_noisy_mse": 5.0, "generalisation_gap": 0.0, "full_gradient_norm": 5.0, "update_gradient_norm": 5.0, "parameter_norm": 5.0},
    ]
    rows.extend(
        {"method_name": method, "model_seed": seed, "test_function_mse": 1.0, "training_mse": 1.0, "validation_mse": 1.0, "test_noisy_mse": 1.0, "generalisation_gap": 0.0, "full_gradient_norm": 1.0, "update_gradient_norm": 1.0, "parameter_norm": 1.0}
        for method in v3a.METHOD_ORDER
        for seed in v3a.MODEL_SEEDS
        if (method, seed) not in {("wr_b1", 0), ("rr_b1", 0)}
    )

    diff = [
        row
        for row in v3a.sampling_law_paired_difference_rows(rows)
        if row["batch_size"] == 1 and row["model_seed"] == 0 and row["metric"] == "test_function_mse"
    ][0]

    assert diff["rr_minus_wr"] == pytest.approx(3.0)


def test_sampling_law_terminal_median_preserves_five_seed_differences() -> None:
    rows = []
    for seed, difference in enumerate([5.0, 1.0, 3.0, -2.0, 4.0]):
        rows.append({"batch_size": 1, "model_seed": seed, "metric": "test_function_mse", "rr_minus_wr": difference})
    for batch_size in [32, 256]:
        for seed in v3a.MODEL_SEEDS:
            rows.append({"batch_size": batch_size, "model_seed": seed, "metric": "test_function_mse", "rr_minus_wr": float(seed)})
    rows.append({"batch_size": 1, "model_seed": 0, "metric": "training_mse", "rr_minus_wr": 999.0})

    seed_level = [
        row
        for row in rows
        if row["batch_size"] == 1 and row["metric"] == "test_function_mse"
    ]
    medians = v3a.terminal_sampling_law_medians(rows)

    assert len(seed_level) == 5
    assert medians[1] == pytest.approx(3.0)


def test_batch_size_comparison_is_law_specific() -> None:
    rows = [
        {
            "method_name": method,
            "model_seed": seed,
            "test_function_mse": float(v3a.method_batch_size(method)),
            "training_mse": 1.0,
            "validation_mse": 1.0,
            "test_noisy_mse": 1.0,
            "generalisation_gap": 0.0,
            "full_gradient_norm": 1.0,
            "update_gradient_norm": 1.0,
            "parameter_norm": 1.0,
        }
        for method in v3a.METHOD_ORDER
        for seed in v3a.MODEL_SEEDS
    ]

    diffs = v3a.batch_size_paired_difference_rows(rows)

    assert {row["sampling_law"] for row in diffs} == {"WR", "RR"}
    assert all(row["pairing"] == "paired by model initialization seed" for row in diffs)


def test_checkpoint_summaries_are_calculated_correctly() -> None:
    histories = {
        ("wr_b1", 0): [
            {"checkpoint_examples": 0, "validation_function_mse": 1.0, "training_mse": 2.0, "full_gradient_norm": 3.0}
        ],
        ("wr_b1", 1): [
            {"checkpoint_examples": 0, "validation_function_mse": 3.0, "training_mse": 4.0, "full_gradient_norm": 5.0}
        ],
    }
    original = v3a.expected_checkpoint_schedule
    v3a.expected_checkpoint_schedule = lambda: [0]
    try:
        rows = v3a.checkpoint_summary_rows(histories)
    finally:
        v3a.expected_checkpoint_schedule = original

    row = [
        item
        for item in rows
        if item["method_name"] == "wr_b1" and item["metric"] == "validation_function_mse"
    ][0]
    assert row["mean"] == pytest.approx(2.0)


def test_risk_identity_arithmetic_is_correct() -> None:
    rows = v3a.risk_identity_rows(
        [
            {
                "method_name": "wr_b1",
                "sampling_law": "WR",
                "nominal_batch_size": 1,
                "model_seed": 0,
                "test_noisy_mse": 0.20,
                "test_function_mse": 0.11,
            }
        ]
    )

    assert rows[0]["test_noisy_minus_test_function"] == pytest.approx(0.09)
    assert rows[0]["deviation_from_sigma_squared"] == pytest.approx(0.0)


def test_output_namespace_is_separate_from_old_v2_analysis() -> None:
    assert v3a.ANALYSIS_RAW_DIR == Path("results/raw/week1_gradient_methods_v3/baseline_analysis")
    assert v3a.ANALYSIS_FIGURE_DIR == Path("results/figures/week1_gradient_methods_v3/baseline_analysis")
    assert "week1_gradient_methods/" not in str(v3a.ANALYSIS_RAW_DIR)


def test_no_test_requires_local_generated_baseline_data() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = "/".join(["data", "generated", "baseline_train.npz"])

    assert forbidden not in source
