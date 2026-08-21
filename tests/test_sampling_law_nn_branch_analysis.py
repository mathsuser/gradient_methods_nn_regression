from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Optional

import pytest


def _load_analysis_module():
    module_path = Path("experiments/09_analyse_sampling_law_nn_branch.py")
    spec = importlib.util.spec_from_file_location(
        "sampling_law_nn_branch_analysis",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = _load_analysis_module()


def _branch_cfg(tmp_path: Path) -> dict[str, Any]:
    return {
        "experiment": {
            "name": "sampling_law_nn_branch",
            "n_train": 5,
            "methods": {
                "wr_1": {
                    "method_name": "wr_1",
                    "sampling_method": "single_with_replacement",
                    "batch_size": 1,
                },
                "rr_1": {
                    "method_name": "rr_1",
                    "sampling_method": "random_reshuffling",
                    "batch_size": 1,
                },
            },
            "training": {
                "model_seed": 0,
                "learning_rate": 0.03,
                "target_examples_processed": 10,
                "evaluation_every_examples": 5,
            },
            "sampling_seeds": {
                "trajectory_ids": [0, 1],
                "by_method": {
                    "wr_1": [91000, 91001],
                    "rr_1": [92000, 92001],
                },
            },
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "figures_dir": str(tmp_path / "figures"),
            },
        }
    }


def _baseline_cfg() -> dict[str, Any]:
    return {"paths": {"generated_data_dir": "data/generated"}}


def _history_rows(method: str, trajectory_id: int, seed: int) -> list[dict[str, Any]]:
    offset = 0.1 if method == "rr_1" else 0.0
    return [
        {
            "trajectory_id": trajectory_id,
            "method": method,
            "sampling_method": "random_reshuffling"
            if method == "rr_1"
            else "single_with_replacement",
            "sampling_seed": seed,
            "step": step,
            "epoch": step // 5 if method == "rr_1" else "",
            "step_within_epoch": step % 5 if method == "rr_1" else "",
            "nominal_batch_size": 1,
            "actual_batch_size": 1 if step else "",
            "checkpoint_examples": checkpoint,
            "cumulative_examples_processed": checkpoint,
            "data_equivalent_passes": checkpoint / 5,
            "batch_loss": "" if step == 0 else 1.0,
            "training_mse": 10.0 + offset + trajectory_id + checkpoint / 100,
            "validation_mse": 11.0 + offset + trajectory_id + checkpoint / 100,
            "validation_function_mse": 1.0 + offset + trajectory_id + checkpoint / 100,
            "update_gradient_norm": "" if step == 0 else 0.5,
            "full_gradient_norm": 2.0 + offset + trajectory_id + checkpoint / 100,
            "parameter_norm": 3.0,
            "training_elapsed_seconds": checkpoint / 1000,
            "total_elapsed_seconds": checkpoint / 900,
        }
        for checkpoint, step in [(0, 0), (5, 5), (10, 10)]
    ]


def _write_run(
    *,
    root: Path,
    method: str,
    trajectory_id: int,
    seed: int,
    checksum: str = "same-checksum",
    history_rows: Optional[list[dict[str, Any]]] = None,
) -> None:
    run_dir = root / method / f"trajectory_{trajectory_id}"
    run_dir.mkdir(parents=True)
    metadata = {
        "trajectory_id": trajectory_id,
        "method_name": method,
        "sampling_method": "random_reshuffling"
        if method == "rr_1"
        else "single_with_replacement",
        "batch_size": 1,
        "sampling_seed": seed,
        "target_examples_processed": 10,
        "evaluation_every_examples": 5,
        "learning_rate": 0.03,
        "model_seed": 0,
        "initial_state_checksum": checksum,
        "loaded_initial_state_checksum": checksum,
        "data_paths": {
            "training": "data/generated/baseline_train.npz",
            "evaluation": "data/generated/baseline_test.npz",
        },
        "elapsed_time": {"run_elapsed_seconds": 0.25 + trajectory_id},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    rows = history_rows if history_rows is not None else _history_rows(
        method,
        trajectory_id,
        seed,
    )
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_valid_fixture(root: Path, cfg: dict[str, Any]) -> None:
    for method in ["wr_1", "rr_1"]:
        for trajectory_id in [0, 1]:
            _write_run(
                root=root,
                method=method,
                trajectory_id=trajectory_id,
                seed=cfg["experiment"]["sampling_seeds"]["by_method"][method][trajectory_id],
            )


def _valid_runs(tmp_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = _branch_cfg(tmp_path)
    root = Path(cfg["experiment"]["paths"]["raw_dir"])
    _write_valid_fixture(root, cfg)
    runs, _ = analysis.load_and_validate_runs(
        baseline_cfg=_baseline_cfg(),
        branch_cfg=cfg,
        expected_trajectory_ids=[0, 1],
    )
    return cfg, runs


def test_integrity_gate_accepts_valid_miniature_structure(tmp_path: Path) -> None:
    cfg, runs = _valid_runs(tmp_path)

    assert len(runs) == 4
    assert analysis.expected_checkpoint_examples(cfg) == [0, 5, 10]


def test_integrity_gate_fails_on_incorrect_run_count(tmp_path: Path) -> None:
    cfg = _branch_cfg(tmp_path)
    root = Path(cfg["experiment"]["paths"]["raw_dir"])
    _write_run(root=root, method="wr_1", trajectory_id=0, seed=91000)

    with pytest.raises(analysis.IntegrityError, match="expected 4 total runs"):
        analysis.load_and_validate_runs(
            baseline_cfg=_baseline_cfg(),
            branch_cfg=cfg,
            expected_trajectory_ids=[0, 1],
        )


def test_integrity_gate_fails_on_wrong_seed_mapping(tmp_path: Path) -> None:
    cfg = _branch_cfg(tmp_path)
    root = Path(cfg["experiment"]["paths"]["raw_dir"])
    _write_valid_fixture(root, cfg)
    metadata_path = root / "rr_1" / "trajectory_1" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sampling_seed"] = 123
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(analysis.IntegrityError, match="sampling_seed"):
        analysis.load_and_validate_runs(
            baseline_cfg=_baseline_cfg(),
            branch_cfg=cfg,
            expected_trajectory_ids=[0, 1],
        )


def test_integrity_gate_fails_on_mismatched_initial_state_checksum(tmp_path: Path) -> None:
    cfg = _branch_cfg(tmp_path)
    root = Path(cfg["experiment"]["paths"]["raw_dir"])
    _write_valid_fixture(root, cfg)
    metadata_path = root / "wr_1" / "trajectory_0" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["initial_state_checksum"] = "different-checksum"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(analysis.IntegrityError, match="shared initial_state_checksum"):
        analysis.load_and_validate_runs(
            baseline_cfg=_baseline_cfg(),
            branch_cfg=cfg,
            expected_trajectory_ids=[0, 1],
        )


def test_integrity_gate_fails_on_missing_checkpoint(tmp_path: Path) -> None:
    cfg = _branch_cfg(tmp_path)
    root = Path(cfg["experiment"]["paths"]["raw_dir"])
    _write_valid_fixture(root, cfg)
    history_path = root / "wr_1" / "trajectory_0" / "history.csv"
    rows = _history_rows("wr_1", 0, 91000)[:-1]
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(analysis.IntegrityError, match="checkpoint"):
        analysis.load_and_validate_runs(
            baseline_cfg=_baseline_cfg(),
            branch_cfg=cfg,
            expected_trajectory_ids=[0, 1],
        )


def test_integrity_gate_fails_on_non_finite_metric(tmp_path: Path) -> None:
    cfg = _branch_cfg(tmp_path)
    root = Path(cfg["experiment"]["paths"]["raw_dir"])
    _write_valid_fixture(root, cfg)
    rows = _history_rows("rr_1", 0, 92000)
    rows[-1] = {**rows[-1], "validation_function_mse": math.inf}
    history_path = root / "rr_1" / "trajectory_0" / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(analysis.IntegrityError, match="non-finite"):
        analysis.load_and_validate_runs(
            baseline_cfg=_baseline_cfg(),
            branch_cfg=cfg,
            expected_trajectory_ids=[0, 1],
        )


def test_terminal_extraction_selects_declared_final_checkpoint(tmp_path: Path) -> None:
    _, runs = _valid_runs(tmp_path)

    terminal = analysis.terminal_runs_rows(runs, final_checkpoint=10)

    assert len(terminal) == 4
    assert {row["evaluation_function_mse"] for row in terminal} == {
        1.1,
        2.1,
        1.2000000000000002,
        2.2,
    }


def test_declared_branch_terminal_checkpoint_is_50000() -> None:
    cfg = analysis.load_json(analysis.BRANCH_CONFIG_PATH)

    assert analysis.expected_checkpoint_examples(cfg)[-1] == 50000


def test_summary_statistics_have_expected_values(tmp_path: Path) -> None:
    _, runs = _valid_runs(tmp_path)
    terminal = analysis.terminal_runs_rows(runs, final_checkpoint=10)
    summary = analysis.terminal_summary_rows(terminal)
    wr_function = next(
        row
        for row in summary
        if row["method"] == "wr_1" and row["metric"] == "evaluation_function_mse"
    )

    assert wr_function["count"] == 2
    assert wr_function["mean"] == pytest.approx(1.6)
    assert wr_function["median"] == pytest.approx(1.6)
    assert wr_function["std"] == pytest.approx(math.sqrt(0.5))
    assert wr_function["min"] == pytest.approx(1.1)
    assert wr_function["max"] == pytest.approx(2.1)


def test_checkpoint_difference_rows_have_expected_sign_and_skip_initial(
    tmp_path: Path,
) -> None:
    _, runs = _valid_runs(tmp_path)
    checkpoint_rows = analysis.checkpoint_summary_rows(runs)

    difference_rows = analysis.checkpoint_difference_rows(checkpoint_rows)

    assert [row["checkpoint_examples"] for row in difference_rows] == [5, 10]
    assert all(row["rr_minus_wr_median"] > 0 for row in difference_rows)
    assert difference_rows[0]["rr_minus_wr_median"] == pytest.approx(0.1)
    assert difference_rows[0]["rr_minus_wr_mean"] == pytest.approx(0.1)


def test_zoomed_trajectory_series_excludes_checkpoint_zero(tmp_path: Path) -> None:
    _, runs = _valid_runs(tmp_path)

    series = analysis.function_mse_trajectory_series(
        runs,
        exclude_checkpoint_zero=True,
    )

    assert series["wr_1"]["checkpoints"] == [5, 10]
    assert series["rr_1"]["checkpoints"] == [5, 10]
