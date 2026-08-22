from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gradient_methods_nn_regression_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


BASELINE_CONFIG_PATH = Path("configs/baseline.json")
V3_CONFIG_PATH = Path("configs/experiments/week1_gradient_methods_v3.json")
LR_DECISION_PATH = Path("results/raw/week1_gradient_methods_v3/learning_rate_selection.json")
BASELINE_RUN_ROOT = Path("results/raw/week1_gradient_methods_v3/baseline_comparison_runs")
BASELINE_MANIFEST_PATH = BASELINE_RUN_ROOT / "baseline_comparison_manifest.json"
ANALYSIS_RAW_DIR = Path("results/raw/week1_gradient_methods_v3/baseline_analysis")
ANALYSIS_FIGURE_DIR = Path("results/figures/week1_gradient_methods_v3/baseline_analysis")
EXPECTED_METHOD_MAPPING = {
    "full_batch_gd": ("full_batch", 5000, "FB"),
    "wr_b1": ("single_with_replacement", 1, "WR"),
    "wr_b32": ("minibatch_with_replacement", 32, "WR"),
    "wr_b256": ("minibatch_with_replacement", 256, "WR"),
    "rr_b1": ("random_reshuffling", 1, "RR"),
    "rr_b32": ("random_reshuffling", 32, "RR"),
    "rr_b256": ("random_reshuffling", 256, "RR"),
}
METHOD_ORDER = list(EXPECTED_METHOD_MAPPING)
MODEL_SEEDS = [0, 1, 2, 3, 4]
PRIMARY_METRIC = "test_function_mse"
TERMINAL_METRICS = [
    "training_mse",
    "validation_mse",
    "test_noisy_mse",
    "test_function_mse",
    "generalisation_gap",
    "update_gradient_norm",
    "full_gradient_norm",
    "parameter_norm",
    "optimiser_steps",
    "examples_processed",
    "data_equivalent_passes",
    "wall_clock_time",
]
PAIRWISE_METRICS = [
    "training_mse",
    "validation_mse",
    "test_noisy_mse",
    "test_function_mse",
    "generalisation_gap",
    "full_gradient_norm",
    "update_gradient_norm",
    "parameter_norm",
]
CHECKPOINT_METRICS = [
    "validation_function_mse",
    "training_mse",
    "full_gradient_norm",
]
FINITE_HISTORY_FIELDS = [
    "training_mse",
    "validation_mse",
    "validation_function_mse",
    "full_gradient_norm",
    "parameter_norm",
    "training_elapsed_seconds",
    "total_elapsed_seconds",
]
OPTIONAL_FINITE_HISTORY_FIELDS = ["update_gradient_norm", "batch_loss"]
SIGMA_SQUARED = 0.09


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return str(value)


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return None if math.isnan(parsed) else parsed


def load_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = dict(row)
            for key in [
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
            ]:
                parsed[key] = parse_optional_float(row.get(key))
            rows.append(parsed)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_checkpoint_schedule() -> list[int]:
    return list(range(0, 500000 + 1, 5000))


def future_baseline_seed_for(
    v3_cfg: dict[str, Any],
    *,
    method_name: str,
    model_seed: int,
) -> int:
    seed_map = v3_cfg["experiment"]["future_baseline"][
        "sampling_seeds_by_method_and_model_seed"
    ]
    return int(seed_map[method_name][str(model_seed)])


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def method_law(method_name: str) -> str:
    return EXPECTED_METHOD_MAPPING[method_name][2]


def method_batch_size(method_name: str) -> int:
    return EXPECTED_METHOD_MAPPING[method_name][1]


def method_for_law_and_batch(law: str, batch_size: int) -> str:
    for method_name, (_, size, candidate_law) in EXPECTED_METHOD_MAPPING.items():
        if candidate_law == law and size == batch_size:
            return method_name
    raise KeyError(f"No method for law={law}, batch_size={batch_size}")


def validate_integrity(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    lr_decision: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_schedule = expected_checkpoint_schedule()
    current_lr_hash = file_sha256(LR_DECISION_PATH)

    if manifest.get("expected_run_count") != 35:
        errors.append("expected_run_count is not 35")
    if manifest.get("actual_completed_run_count") != 35:
        errors.append("actual_completed_run_count is not 35")
    if list(manifest.get("method_definitions", {})) != METHOD_ORDER:
        errors.append("manifest method definitions do not contain the locked seven methods")
    if [int(seed) for seed in manifest.get("model_seeds", [])] != MODEL_SEEDS:
        errors.append("manifest model seeds are not [0, 1, 2, 3, 4]")
    if float(manifest.get("learning_rate", float("nan"))) != 0.03:
        errors.append("manifest learning_rate is not 0.03")
    if manifest.get("human_approval_confirmed") is not True:
        errors.append("manifest human_approval_confirmed is not true")
    if lr_decision.get("human_approved") is not True:
        errors.append("current LR artifact human_approved is not true")
    if float(lr_decision.get("selected_common_learning_rate", float("nan"))) != 0.03:
        errors.append("current LR artifact selected_common_learning_rate is not 0.03")
    if manifest.get("learning_rate_decision_artifact_sha256") != current_lr_hash:
        errors.append("manifest LR artifact SHA-256 does not match current artifact bytes")
    if int(manifest.get("target_examples_processed", -1)) != 500000:
        errors.append("manifest target_examples_processed is not 500000")
    if float(manifest.get("data_equivalent_passes", float("nan"))) != 100.0:
        errors.append("manifest data_equivalent_passes is not 100")
    if int(manifest.get("evaluation_every_examples", -1)) != 5000:
        errors.append("manifest evaluation_every_examples is not 5000")
    if [int(x) for x in manifest.get("checkpoint_schedule", [])] != expected_schedule:
        errors.append("manifest checkpoint schedule is not 0,5000,...,500000")
    if manifest.get("worktree_status") == "dirty":
        warnings.append(
            "Completed baseline manifest records worktree_status=dirty; treated as provenance warning after protocol checks."
        )

    data_hashes = manifest.get("common_config", {}).get("data_provenance", {})
    run_paths = [Path(path) for path in manifest.get("run_metadata_paths", [])]
    if len(run_paths) != 35:
        errors.append("manifest does not list 35 run metadata paths")

    seen_cells: set[tuple[str, int]] = set()
    checksums_by_seed: dict[int, set[str]] = {seed: set() for seed in MODEL_SEEDS}
    data_hashes_seen: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    histories: dict[tuple[str, int], list[dict[str, Any]]] = {}
    required_final_metrics = {
        "training_prediction_mse",
        "training_function_mse",
        "validation_prediction_mse",
        "validation_function_mse",
        "test_prediction_mse",
        "test_function_mse",
        "parameter_norm",
    }

    for metadata_path in run_paths:
        if not metadata_path.exists():
            errors.append(f"Missing run metadata: {metadata_path}")
            continue
        metadata = load_json(metadata_path)
        history_path = Path(metadata["artifacts"]["history_csv"])
        if not history_path.exists():
            errors.append(f"Missing history: {history_path}")
            continue
        history = load_history(history_path)
        if not history:
            errors.append(f"Empty history: {history_path}")
            continue

        method_name = str(metadata.get("method_name"))
        model_seed = int(metadata.get("model_seed"))
        cell = (method_name, model_seed)
        if cell in seen_cells:
            errors.append(f"Duplicate method/model-seed cell: {cell}")
        seen_cells.add(cell)

        if method_name not in EXPECTED_METHOD_MAPPING:
            errors.append(f"Unknown method: {method_name}")
            continue
        expected_sampling, expected_batch, expected_law = EXPECTED_METHOD_MAPPING[method_name]
        if str(metadata.get("sampling_method")) != expected_sampling:
            errors.append(f"{metadata_path}: wrong sampling_method")
        if int(metadata.get("nominal_batch_size")) != expected_batch:
            errors.append(f"{metadata_path}: wrong batch size")
        if model_seed not in MODEL_SEEDS:
            errors.append(f"{metadata_path}: wrong model seed")
        expected_seed = future_baseline_seed_for(
            v3_cfg,
            method_name=method_name,
            model_seed=model_seed,
        )
        if int(metadata.get("sampling_seed")) != expected_seed:
            errors.append(f"{metadata_path}: wrong explicit sampling seed")
        if float(metadata.get("learning_rate", float("nan"))) != 0.03:
            errors.append(f"{metadata_path}: wrong learning rate")
        if metadata.get("learning_rate_decision_artifact_sha256") != current_lr_hash:
            errors.append(f"{metadata_path}: LR artifact SHA mismatch")
        if int(metadata.get("target_examples_processed", -1)) != 500000:
            errors.append(f"{metadata_path}: target_examples_processed is not 500000")
        if int(metadata.get("achieved_examples_processed", -1)) != 500000:
            errors.append(f"{metadata_path}: terminal examples != 500000")
        if float(metadata.get("achieved_data_equivalent_passes", float("nan"))) != 100.0:
            errors.append(f"{metadata_path}: terminal DEP != 100")
        if int(metadata.get("evaluation_every_examples", -1)) != 5000:
            errors.append(f"{metadata_path}: checkpoint cadence is not 5000")
        if str(metadata.get("dtype")) != "float64":
            errors.append(f"{metadata_path}: dtype is not float64")
        if str(metadata.get("device")) != "cpu":
            errors.append(f"{metadata_path}: device is not cpu")
        if metadata.get("initial_state_checksum") != metadata.get("loaded_initial_state_checksum"):
            errors.append(f"{metadata_path}: loaded initial state checksum mismatch")
        checksums_by_seed[model_seed].add(str(metadata.get("initial_state_checksum")))

        run_data_hashes = metadata.get("data_provenance", {}).get("file_hashes", {})
        data_hashes_seen.append(run_data_hashes)
        if run_data_hashes != data_hashes:
            errors.append(f"{metadata_path}: data provenance hashes differ from manifest")

        checkpoint_values = [int(row["checkpoint_examples"]) for row in history]
        if checkpoint_values != expected_schedule:
            errors.append(f"{history_path}: wrong checkpoint schedule")
        if len(history) != 101:
            errors.append(f"{history_path}: expected 101 checkpoints")
        final = history[-1]
        if int(final["checkpoint_examples"]) != 500000:
            errors.append(f"{history_path}: missing terminal checkpoint")
        if int(final["cumulative_examples_processed"]) != 500000:
            errors.append(f"{history_path}: terminal cumulative examples != 500000")
        for row_index, row in enumerate(history):
            for field in FINITE_HISTORY_FIELDS:
                value = row.get(field)
                if value is None or not math.isfinite(float(value)):
                    errors.append(f"{history_path}: non-finite {field} at row {row_index}")
            for field in OPTIONAL_FINITE_HISTORY_FIELDS:
                value = row.get(field)
                if value is not None and not math.isfinite(float(value)):
                    errors.append(f"{history_path}: non-finite {field} at row {row_index}")
        final_metrics = metadata.get("final_split_metrics", {})
        if set(final_metrics) != required_final_metrics:
            errors.append(f"{metadata_path}: final_split_metrics schema mismatch")
        for key, value in final_metrics.items():
            if not math.isfinite(float(value)):
                errors.append(f"{metadata_path}: non-finite final metric {key}")

        history_epoch = final["epoch"] if expected_law == "RR" or method_name == "full_batch_gd" else None
        row = {
            "method_name": method_name,
            "sampling_law": expected_law,
            "sampling_method": expected_sampling,
            "nominal_batch_size": expected_batch,
            "model_seed": model_seed,
            "sampling_seed": int(metadata["sampling_seed"]),
            "training_mse": float(final["training_mse"]),
            "validation_mse": float(final["validation_mse"]),
            "test_noisy_mse": float(final_metrics["test_prediction_mse"]),
            "test_function_mse": float(final_metrics["test_function_mse"]),
            "generalisation_gap": float(final_metrics["test_prediction_mse"]) - float(final["training_mse"]),
            "update_gradient_norm": float(final["update_gradient_norm"]) if final["update_gradient_norm"] is not None else None,
            "full_gradient_norm": float(final["full_gradient_norm"]),
            "parameter_norm": float(final["parameter_norm"]),
            "optimiser_steps": int(final["step"]),
            "examples_processed": int(final["cumulative_examples_processed"]),
            "data_equivalent_passes": float(final["data_equivalent_passes"]),
            "wall_clock_time": float(metadata["runtime"]["run_elapsed_seconds"]),
            "epoch": history_epoch,
            "metadata_json": metadata_path,
            "history_csv": history_path,
            "initial_state_checksum": metadata["initial_state_checksum"],
            "lr_artifact_sha256": metadata["learning_rate_decision_artifact_sha256"],
        }
        terminal_rows.append(row)
        histories[cell] = history

    expected_cells = {(method, seed) for method in METHOD_ORDER for seed in MODEL_SEEDS}
    missing = expected_cells - seen_cells
    extra = seen_cells - expected_cells
    if missing:
        errors.append(f"Missing method/model-seed cells: {sorted(missing)}")
    if extra:
        errors.append(f"Unexpected method/model-seed cells: {sorted(extra)}")
    for seed, checksums in checksums_by_seed.items():
        if len(checksums) != 1:
            errors.append(f"Initial-state checksum is not shared across methods for model_seed={seed}")
    unique_seed_checksums = {next(iter(values)) for values in checksums_by_seed.values() if values}
    if len(unique_seed_checksums) != len(MODEL_SEEDS):
        errors.append("Initial-state checksums are not distinct across model seeds")

    report = {
        "integrity_status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "source_baseline_manifest_path": manifest_path,
        "expected_run_count": 35,
        "actual_completed_run_count": len(terminal_rows),
        "method_count": len({row["method_name"] for row in terminal_rows}),
        "model_seeds": sorted({row["model_seed"] for row in terminal_rows}),
        "learning_rate": manifest.get("learning_rate"),
        "human_approval_confirmed": manifest.get("human_approval_confirmed"),
        "lr_artifact_sha256_current": current_lr_hash,
        "lr_artifact_sha256_manifest": manifest.get("learning_rate_decision_artifact_sha256"),
        "worktree_status": manifest.get("worktree_status"),
        "data_provenance_hashes": data_hashes,
        "comparison_families": comparison_families(),
    }
    return report, terminal_rows, histories


def comparison_families() -> dict[str, Any]:
    return {
        "sampling_law_at_fixed_batch_size": [
            ["wr_b1", "rr_b1"],
            ["wr_b32", "rr_b32"],
            ["wr_b256", "rr_b256"],
        ],
        "batch_size_within_wr": ["wr_b1", "wr_b32", "wr_b256"],
        "batch_size_within_rr": ["rr_b1", "rr_b32", "rr_b256"],
        "full_batch_reference": ["full_batch_gd"],
    }


def method_summary_rows(method_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name in METHOD_ORDER:
        method_rows = [row for row in method_seed_rows if row["method_name"] == method_name]
        for metric in TERMINAL_METRICS:
            values = [float(row[metric]) for row in method_rows if row[metric] is not None]
            rows.append(
                {
                    "method_name": method_name,
                    "sampling_law": method_law(method_name),
                    "nominal_batch_size": method_batch_size(method_name),
                    "metric": metric,
                    **summary(values),
                }
            )
    return rows


def sampling_law_paired_difference_rows(method_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell = {(row["method_name"], row["model_seed"]): row for row in method_seed_rows}
    rows: list[dict[str, Any]] = []
    for batch_size in [1, 32, 256]:
        wr_method = method_for_law_and_batch("WR", batch_size)
        rr_method = method_for_law_and_batch("RR", batch_size)
        for model_seed in MODEL_SEEDS:
            wr_row = by_cell[(wr_method, model_seed)]
            rr_row = by_cell[(rr_method, model_seed)]
            for metric in PAIRWISE_METRICS:
                wr_value = float(wr_row[metric])
                rr_value = float(rr_row[metric])
                rows.append(
                    {
                        "batch_size": batch_size,
                        "model_seed": model_seed,
                        "metric": metric,
                        "wr_method": wr_method,
                        "rr_method": rr_method,
                        "wr_value": wr_value,
                        "rr_value": rr_value,
                        "rr_minus_wr": rr_value - wr_value,
                        "pairing": "paired by model initialization seed",
                    }
                )
    return rows


def paired_summary_rows(rows: list[dict[str, Any]], group_keys: list[str], value_key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        groups.setdefault(key, []).append(float(row[value_key]))
    output: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        row = {group_key: key[index] for index, group_key in enumerate(group_keys)}
        row[value_key] = value_key
        row.update(summary(values))
        output.append(row)
    return output


def batch_size_paired_difference_rows(method_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell = {(row["method_name"], row["model_seed"]): row for row in method_seed_rows}
    comparisons = [(1, 32), (1, 256), (32, 256)]
    rows: list[dict[str, Any]] = []
    for law in ["WR", "RR"]:
        for smaller_batch, larger_batch in comparisons:
            smaller_method = method_for_law_and_batch(law, smaller_batch)
            larger_method = method_for_law_and_batch(law, larger_batch)
            comparison = f"B{larger_batch}_minus_B{smaller_batch}"
            for model_seed in MODEL_SEEDS:
                smaller_row = by_cell[(smaller_method, model_seed)]
                larger_row = by_cell[(larger_method, model_seed)]
                for metric in PAIRWISE_METRICS:
                    smaller_value = float(smaller_row[metric])
                    larger_value = float(larger_row[metric])
                    rows.append(
                        {
                            "sampling_law": law,
                            "comparison": comparison,
                            "model_seed": model_seed,
                            "metric": metric,
                            "smaller_batch_size": smaller_batch,
                            "larger_batch_size": larger_batch,
                            "smaller_batch_method": smaller_method,
                            "larger_batch_method": larger_method,
                            "smaller_batch_value": smaller_value,
                            "larger_batch_value": larger_value,
                            "larger_minus_smaller": larger_value - smaller_value,
                            "pairing": "paired by model initialization seed",
                        }
                    )
    return rows


def risk_identity_rows(method_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in method_seed_rows:
        noise_gap = float(row["test_noisy_mse"]) - float(row["test_function_mse"])
        rows.append(
            {
                "method_name": row["method_name"],
                "sampling_law": row["sampling_law"],
                "nominal_batch_size": row["nominal_batch_size"],
                "model_seed": row["model_seed"],
                "test_noisy_mse": row["test_noisy_mse"],
                "test_function_mse": row["test_function_mse"],
                "test_noisy_minus_test_function": noise_gap,
                "sigma_squared": SIGMA_SQUARED,
                "deviation_from_sigma_squared": noise_gap - SIGMA_SQUARED,
            }
        )
    return rows


def checkpoint_summary_rows(histories: dict[tuple[str, int], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name in METHOD_ORDER:
        for checkpoint in expected_checkpoint_schedule():
            matching = [
                row
                for (candidate_method, _), history in histories.items()
                if candidate_method == method_name
                for row in history
                if int(row["checkpoint_examples"]) == checkpoint
            ]
            for metric in CHECKPOINT_METRICS:
                values = [float(row[metric]) for row in matching]
                if not values:
                    continue
                rows.append(
                    {
                        "method_name": method_name,
                        "sampling_law": method_law(method_name),
                        "nominal_batch_size": method_batch_size(method_name),
                        "checkpoint_examples": checkpoint,
                        "metric": metric,
                        **summary(values),
                    }
                )
    return rows


def median_and_spread(
    histories: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    method_name: str,
    metric: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    checkpoints = expected_checkpoint_schedule()
    medians: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    for checkpoint in checkpoints:
        values = [
            float(row[metric])
            for (candidate_method, _), history in histories.items()
            if candidate_method == method_name
            for row in history
            if int(row["checkpoint_examples"]) == checkpoint
        ]
        medians.append(float(np.median(values)))
        mins.append(float(np.min(values)))
        maxs.append(float(np.max(values)))
    return checkpoints, medians, mins, maxs


def terminal_sampling_law_medians(rows: list[dict[str, Any]]) -> dict[int, float]:
    medians: dict[int, float] = {}
    primary = [row for row in rows if row["metric"] == PRIMARY_METRIC]
    for batch_size in [1, 32, 256]:
        values = [
            float(row["rr_minus_wr"])
            for row in primary
            if int(row["batch_size"]) == batch_size
        ]
        medians[batch_size] = float(np.median(values))
    return medians


def plot_function_mse_wr_rr_by_batch(histories: dict[tuple[str, int], list[dict[str, Any]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for axis, batch_size in zip(axes, [1, 32, 256]):
        for law, color in [("WR", "#1f77b4"), ("RR", "#d62728")]:
            method_name = method_for_law_and_batch(law, batch_size)
            x, median, low, high = median_and_spread(
                histories,
                method_name=method_name,
                metric="validation_function_mse",
            )
            axis.plot(x, median, label=law, color=color, linewidth=1.8)
            axis.fill_between(x, low, high, color=color, alpha=0.14)
        axis.set_title(f"B={batch_size}")
        axis.set_xlabel("examples processed")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("validation function MSE")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_terminal_sampling_law_differences(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.8))
    primary = [row for row in rows if row["metric"] == PRIMARY_METRIC]
    positions = {1: 0, 32: 1, 256: 2}
    seed_handle = None
    for row in primary:
        x = positions[int(row["batch_size"])] + (int(row["model_seed"]) - 2) * 0.06
        seed_handle = axis.scatter(
            x,
            float(row["rr_minus_wr"]),
            color="#333333",
            s=42,
            label="model-seed difference" if seed_handle is None else None,
        )
    medians = terminal_sampling_law_medians(rows)
    axis.scatter(
        [positions[batch_size] for batch_size in [1, 32, 256]],
        [medians[batch_size] for batch_size in [1, 32, 256]],
        color="#c51b7d",
        marker="D",
        s=92,
        edgecolor="white",
        linewidth=0.8,
        label="median difference",
        zorder=3,
    )
    axis.axhline(0.0, color="#777777", linewidth=1.0)
    axis.set_xticks([0, 1, 2], ["B=1", "B=32", "B=256"])
    axis.set_ylabel("RR test function MSE - WR test function MSE")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_terminal_function_mse_by_batch_and_law(method_seed_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, law in zip(axes, ["WR", "RR"]):
        methods = [method_for_law_and_batch(law, batch) for batch in [1, 32, 256]]
        for model_seed in MODEL_SEEDS:
            values = [
                float(row["test_function_mse"])
                for method_name in methods
                for row in method_seed_rows
                if row["method_name"] == method_name and row["model_seed"] == model_seed
            ]
            axis.plot([1, 32, 256], values, marker="o", linewidth=1.1, alpha=0.8)
        axis.set_xscale("log")
        axis.set_xticks([1, 32, 256], ["1", "32", "256"])
        axis.set_title(law)
        axis.set_xlabel("batch size")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("terminal test function MSE")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_full_batch_function_mse(histories: dict[tuple[str, int], list[dict[str, Any]]], path: Path) -> None:
    x, median, low, high = median_and_spread(
        histories,
        method_name="full_batch_gd",
        metric="validation_function_mse",
    )
    fig, axis = plt.subplots(figsize=(7, 4.8))
    axis.plot(x, median, color="#2ca02c", linewidth=1.8)
    axis.fill_between(x, low, high, color="#2ca02c", alpha=0.16)
    axis.set_xlabel("examples processed")
    axis.set_ylabel("validation function MSE")
    axis.set_title("Full-batch GD")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_analysis() -> dict[str, Any]:
    baseline_cfg = load_json(BASELINE_CONFIG_PATH)
    v3_cfg = load_json(V3_CONFIG_PATH)
    lr_decision = load_json(LR_DECISION_PATH)
    manifest = load_json(BASELINE_MANIFEST_PATH)
    validation_report, method_seed_rows, histories = validate_integrity(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        lr_decision=lr_decision,
        manifest=manifest,
        manifest_path=BASELINE_MANIFEST_PATH,
    )
    write_json(validation_report, ANALYSIS_RAW_DIR / "validation_report.json")
    if validation_report["errors"]:
        raise RuntimeError(
            "Baseline analysis aborted due to integrity errors; see validation_report.json"
        )

    method_summary = method_summary_rows(method_seed_rows)
    sampling_rows = sampling_law_paired_difference_rows(method_seed_rows)
    sampling_summary = paired_summary_rows(
        sampling_rows,
        group_keys=["batch_size", "metric"],
        value_key="rr_minus_wr",
    )
    batch_rows = batch_size_paired_difference_rows(method_seed_rows)
    batch_summary = paired_summary_rows(
        batch_rows,
        group_keys=["sampling_law", "comparison", "metric"],
        value_key="larger_minus_smaller",
    )
    risk_rows = risk_identity_rows(method_seed_rows)
    risk_summary = paired_summary_rows(
        risk_rows,
        group_keys=["method_name"],
        value_key="test_noisy_minus_test_function",
    )
    checkpoint_summary = checkpoint_summary_rows(histories)

    tables = {
        "method_seed_values": ANALYSIS_RAW_DIR / "method_seed_values.csv",
        "method_summary": ANALYSIS_RAW_DIR / "method_summary.csv",
        "sampling_law_paired_differences": ANALYSIS_RAW_DIR / "sampling_law_paired_differences.csv",
        "sampling_law_paired_summary": ANALYSIS_RAW_DIR / "sampling_law_paired_summary.csv",
        "batch_size_paired_differences": ANALYSIS_RAW_DIR / "batch_size_paired_differences.csv",
        "batch_size_paired_summary": ANALYSIS_RAW_DIR / "batch_size_paired_summary.csv",
        "risk_identity_by_run": ANALYSIS_RAW_DIR / "risk_identity_by_run.csv",
        "risk_identity_summary": ANALYSIS_RAW_DIR / "risk_identity_summary.csv",
        "checkpoint_summary": ANALYSIS_RAW_DIR / "checkpoint_summary.csv",
    }
    write_csv(method_seed_rows, tables["method_seed_values"])
    write_csv(method_summary, tables["method_summary"])
    write_csv(sampling_rows, tables["sampling_law_paired_differences"])
    write_csv(sampling_summary, tables["sampling_law_paired_summary"])
    write_csv(batch_rows, tables["batch_size_paired_differences"])
    write_csv(batch_summary, tables["batch_size_paired_summary"])
    write_csv(risk_rows, tables["risk_identity_by_run"])
    write_csv(risk_summary, tables["risk_identity_summary"])
    write_csv(checkpoint_summary, tables["checkpoint_summary"])

    figures = {
        "function_mse_wr_rr_by_batch": ANALYSIS_FIGURE_DIR / "function_mse_wr_rr_by_batch.png",
        "terminal_sampling_law_differences": ANALYSIS_FIGURE_DIR / "terminal_sampling_law_differences.png",
        "terminal_function_mse_by_batch_and_law": ANALYSIS_FIGURE_DIR / "terminal_function_mse_by_batch_and_law.png",
        "full_batch_function_mse_vs_examples": ANALYSIS_FIGURE_DIR / "full_batch_function_mse_vs_examples.png",
    }
    plot_function_mse_wr_rr_by_batch(histories, figures["function_mse_wr_rr_by_batch"])
    plot_terminal_sampling_law_differences(sampling_rows, figures["terminal_sampling_law_differences"])
    plot_terminal_function_mse_by_batch_and_law(method_seed_rows, figures["terminal_function_mse_by_batch_and_law"])
    plot_full_batch_function_mse(histories, figures["full_batch_function_mse_vs_examples"])

    analysis_manifest = {
        "integrity_status": validation_report["integrity_status"],
        "runs_analysed": len(method_seed_rows),
        "methods": METHOD_ORDER,
        "model_seeds": MODEL_SEEDS,
        "primary_metric": PRIMARY_METRIC,
        "comparison_families": comparison_families(),
        "learning_rate_decision_artifact_sha256": validation_report["lr_artifact_sha256_current"],
        "data_provenance_hashes": validation_report["data_provenance_hashes"],
        "source_baseline_manifest_path": BASELINE_MANIFEST_PATH,
        "worktree_status_warning": validation_report["warnings"],
        "generated_tables": tables,
        "generated_figures": figures,
        "interpretation_boundaries": [
            "No training was run by this analysis.",
            "Full-batch GD is treated as a deterministic reference, not as a WR/RR sampling-law cell.",
            "Sampling-law and batch-size differences are paired by model initialization seed only.",
            "Wall-clock time is retained as an engineering diagnostic only.",
            "No inferential confidence intervals are reported for n=5.",
        ],
    }
    write_json(analysis_manifest, ANALYSIS_RAW_DIR / "analysis_manifest.json")
    return analysis_manifest


def main() -> None:
    manifest = run_analysis()
    print(json.dumps(to_jsonable(manifest), indent=2))


if __name__ == "__main__":
    main()
