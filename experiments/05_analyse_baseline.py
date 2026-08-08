from __future__ import annotations

import csv
import itertools
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gradient_methods_nn_regression_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


COMPARISONS = {
    "A_gradient_estimator_size": [
        "full_batch_gd",
        "single_observation_sgd",
        "minibatch_with_replacement_b32",
        "minibatch_with_replacement_b256",
    ],
    "B_sampling_rule_b32": [
        "minibatch_with_replacement_b32",
        "random_reshuffling_b32",
    ],
    "C_sampling_rule_b256": [
        "minibatch_with_replacement_b256",
        "random_reshuffling_b256",
    ],
    "D_batch_size_with_replacement": [
        "minibatch_with_replacement_b32",
        "minibatch_with_replacement_b256",
    ],
    "D_batch_size_random_reshuffling": [
        "random_reshuffling_b32",
        "random_reshuffling_b256",
    ],
}

FINAL_METRICS = [
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
    "epoch_count",
]

HISTORY_NUMERIC_FIELDS = [
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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return float(value)
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return str(value)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed


def _load_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        history: list[dict[str, Any]] = []
        for row in reader:
            parsed = dict(row)
            for field in HISTORY_NUMERIC_FIELDS:
                parsed[field] = _parse_optional_float(row.get(field))
            history.append(parsed)
    return history


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(data), handle, indent=2)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
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


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "std": _std(values),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _is_finite_or_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _expected_method_items(experiment_cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return list(experiment_cfg["experiment"]["methods"].items())


def _load_runs(
    *,
    baseline_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    run_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method_items = _expected_method_items(experiment_cfg)
    model_seeds = [int(seed) for seed in baseline_cfg["optimisation"]["model_seeds"]]
    target_examples = int(
        experiment_cfg["experiment"]["training"]["target_examples_processed"]
    )
    expected_run_count = len(method_items) * len(model_seeds)

    runs: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    checksums_by_seed: dict[int, set[str]] = {seed: set() for seed in model_seeds}

    for method_index, (method_key, method_cfg) in enumerate(method_items):
        method_name = str(method_cfg.get("method_name", method_key))
        sampling_method = str(method_cfg["sampling_method"])
        batch_size = int(method_cfg["batch_size"])

        for model_seed in model_seeds:
            run_dir = run_root / method_name / f"seed_{model_seed}"
            metadata_path = run_dir / "metadata.json"
            history_path = run_dir / "history.csv"

            if not metadata_path.exists():
                errors.append(f"Missing metadata: {metadata_path}")
                continue
            if not history_path.exists():
                errors.append(f"Missing history: {history_path}")
                continue

            metadata = _load_json(metadata_path)
            history = _load_history(history_path)
            if not history:
                errors.append(f"Empty history: {history_path}")
                continue

            final = history[-1]
            expected_sampling_seed = (
                int(baseline_cfg["optimisation"]["sampling_seed_offset"])
                + model_seed * len(method_items)
                + method_index
            )

            validations = [
                (metadata["method_name"] == method_name, "method_name"),
                (metadata["sampling_method"] == sampling_method, "sampling_method"),
                (int(metadata["nominal_batch_size"]) == batch_size, "batch_size"),
                (int(metadata["model_seed"]) == model_seed, "model_seed"),
                (int(metadata["sampling_seed"]) == expected_sampling_seed, "sampling_seed"),
                (
                    metadata["initial_state_checksum"]
                    == metadata["loaded_initial_state_checksum"],
                    "initial_state_checksum",
                ),
                (
                    int(metadata["actual_examples_processed"]) == target_examples,
                    "metadata_actual_examples_processed",
                ),
                (
                    int(final["cumulative_examples_processed"]) == target_examples,
                    "history_actual_examples_processed",
                ),
                (
                    int(metadata["final_step_count"]) == int(final["step"]),
                    "final_step_count",
                ),
            ]
            for ok, label in validations:
                if not ok:
                    errors.append(f"{metadata_path}: invalid {label}")

            checksum_path = (
                run_root
                / "initial_states"
                / f"seed_{model_seed}"
                / "initial_state_checksum.json"
            )
            if checksum_path.exists():
                checksum_metadata = _load_json(checksum_path)
                if (
                    checksum_metadata["initial_state_checksum"]
                    != metadata["initial_state_checksum"]
                ):
                    errors.append(f"{metadata_path}: seed checksum file mismatch")
            else:
                warnings.append(f"Missing initial-state checksum file: {checksum_path}")

            checksums_by_seed[model_seed].add(metadata["initial_state_checksum"])

            non_finite_fields: list[str] = []
            for row_index, row in enumerate(history):
                for field in HISTORY_NUMERIC_FIELDS:
                    value = row.get(field)
                    if not _is_finite_or_none(value):
                        non_finite_fields.append(f"history[{row_index}].{field}")
            for field, value in metadata["final_metrics"].items():
                if not _is_finite_or_none(value):
                    non_finite_fields.append(f"final_metrics.{field}")

            if non_finite_fields:
                errors.append(
                    f"{metadata_path}: non-finite fields: "
                    + ", ".join(non_finite_fields[:10])
                )

            runs.append(
                {
                    "method_name": method_name,
                    "sampling_method": sampling_method,
                    "nominal_batch_size": batch_size,
                    "model_seed": model_seed,
                    "metadata": metadata,
                    "history": history,
                    "final": final,
                    "metadata_path": metadata_path,
                    "history_path": history_path,
                }
            )

    for model_seed, checksums in checksums_by_seed.items():
        if len(checksums) > 1:
            errors.append(
                f"Model seed {model_seed} has multiple initial-state checksums"
            )

    if len(runs) != expected_run_count:
        errors.append(
            f"Loaded {len(runs)} completed runs, expected {expected_run_count}"
        )

    report = {
        "expected_run_count": expected_run_count,
        "loaded_run_count": len(runs),
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    return runs, report


def _final_row(run: dict[str, Any]) -> dict[str, Any]:
    final = run["final"]
    metadata = run["metadata"]
    final_metrics = metadata["final_metrics"]
    training_mse = float(final["training_mse"])
    test_noisy_mse = float(final_metrics["test_prediction_mse"])
    epoch_count = metadata["epoch_count"]
    if run["sampling_method"] in {
        "single_with_replacement",
        "minibatch_with_replacement",
    }:
        epoch_count = None
    return {
        "method_name": run["method_name"],
        "sampling_method": run["sampling_method"],
        "nominal_batch_size": run["nominal_batch_size"],
        "model_seed": run["model_seed"],
        "training_mse": training_mse,
        "validation_mse": float(final["validation_mse"]),
        "test_noisy_mse": test_noisy_mse,
        "test_function_mse": float(final_metrics["test_function_mse"]),
        "generalisation_gap": test_noisy_mse - training_mse,
        "update_gradient_norm": float(final["update_gradient_norm"]),
        "full_gradient_norm": float(final["full_gradient_norm"]),
        "parameter_norm": float(final["parameter_norm"]),
        "optimiser_steps": int(final["step"]),
        "examples_processed": int(final["cumulative_examples_processed"]),
        "data_equivalent_passes": float(final["data_equivalent_passes"]),
        "wall_clock_time": float(
            metadata["elapsed_time"].get(
                "run_elapsed_seconds",
                metadata["elapsed_time"]["total_elapsed_seconds"],
            )
        ),
        "epoch_count": epoch_count,
    }


def _build_final_tables(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_values = [_final_row(run) for run in runs]
    methods = sorted({row["method_name"] for row in seed_values})
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in seed_values if row["method_name"] == method]
        for metric in FINAL_METRICS:
            numeric_values = [
                float(row[metric]) for row in method_rows if row[metric] is not None
            ]
            if not numeric_values:
                continue
            stats = _summary(numeric_values)
            seed_columns = {
                f"seed_{row['model_seed']}": row[metric] for row in method_rows
            }
            summary_rows.append(
                {
                    "method_name": method,
                    "sampling_method": method_rows[0]["sampling_method"],
                    "nominal_batch_size": method_rows[0]["nominal_batch_size"],
                    "metric": metric,
                    **stats,
                    **seed_columns,
                }
            )
    return seed_values, summary_rows


def _paired_comparison_tables(
    seed_values: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_method_seed = {
        (row["method_name"], row["model_seed"]): row for row in seed_values
    }
    difference_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for comparison_name, methods in COMPARISONS.items():
        for left, right in itertools.combinations(methods, 2):
            for metric in FINAL_METRICS:
                values: list[float] = []
                for seed in sorted(
                    {
                        row["model_seed"]
                        for row in seed_values
                        if row["method_name"] in {left, right}
                    }
                ):
                    left_value = by_method_seed[(left, seed)][metric]
                    right_value = by_method_seed[(right, seed)][metric]
                    if left_value is None or right_value is None:
                        continue
                    difference = float(right_value) - float(left_value)
                    values.append(difference)
                    difference_rows.append(
                        {
                            "comparison": comparison_name,
                            "left_method": left,
                            "right_method": right,
                            "metric": metric,
                            "model_seed": seed,
                            "right_minus_left": difference,
                            "left_value": left_value,
                            "right_value": right_value,
                        }
                    )
                if values:
                    summary_rows.append(
                        {
                            "comparison": comparison_name,
                            "left_method": left,
                            "right_method": right,
                            "metric": metric,
                            **_summary(values),
                        }
                    )
    return difference_rows, summary_rows


def _risk_identity_tables(
    seed_values: list[dict[str, Any]],
    *,
    sigma_squared: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for row in seed_values:
        risk_difference = row["test_noisy_mse"] - row["test_function_mse"]
        rows.append(
            {
                "method_name": row["method_name"],
                "model_seed": row["model_seed"],
                "noisy_test_mse": row["test_noisy_mse"],
                "function_mse": row["test_function_mse"],
                "risk_difference": risk_difference,
                "risk_difference_minus_sigma_squared": (
                    risk_difference - sigma_squared
                ),
                "sigma_squared": sigma_squared,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for method in sorted({row["method_name"] for row in rows}):
        values = [
            float(row["risk_difference_minus_sigma_squared"])
            for row in rows
            if row["method_name"] == method
        ]
        summary_rows.append(
            {
                "scope": "method",
                "method_name": method,
                **_summary(values),
            }
        )
    all_values = [
        float(row["risk_difference_minus_sigma_squared"]) for row in rows
    ]
    summary_rows.append(
        {
            "scope": "all_methods_and_seeds",
            "method_name": "all",
            **_summary(all_values),
        }
    )
    return rows, summary_rows


def _plot_history_metric(
    *,
    runs: list[dict[str, Any]],
    methods: list[str],
    x_key: str,
    y_key: str,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(5 * len(methods), 4),
        sharey=True,
    )
    if len(methods) == 1:
        axes = [axes]
    for axis, method in zip(axes, methods):
        method_runs = sorted(
            [run for run in runs if run["method_name"] == method],
            key=lambda item: item["model_seed"],
        )
        for run in method_runs:
            history = run["history"]
            plot_rows = [
                row
                for row in history
                if row[x_key] is not None and row[y_key] is not None
            ]
            x_values = [row[x_key] for row in plot_rows]
            y_values = [row[y_key] for row in plot_rows]
            axis.plot(
                x_values,
                y_values,
                linewidth=1.2,
                alpha=0.8,
                label=f"seed {run['model_seed']}",
            )
        axis.set_title(method)
        axis.set_xlabel(x_key)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel(y_key)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_gradient_norms(
    *,
    runs: list[dict[str, Any]],
    methods: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(5 * len(methods), 4),
        sharey=True,
    )
    if len(methods) == 1:
        axes = [axes]
    for axis, method in zip(axes, methods):
        method_runs = [run for run in runs if run["method_name"] == method]
        mean_by_key: dict[str, list[float]] = {}
        first_history = method_runs[0]["history"]
        for key in ["update_gradient_norm", "full_gradient_norm"]:
            curves = []
            for run in method_runs:
                curves.append(
                    [
                        row[key]
                        for row in run["history"]
                        if row[key] is not None
                    ]
                )
            min_length = min(len(curve) for curve in curves)
            x_values = [
                row["cumulative_examples_processed"]
                for row in first_history
                if row[key] is not None
            ][:min_length]
            mean_by_key[key] = [
                float(np.mean([curve[idx] for curve in curves]))
                for idx in range(min_length)
            ]
            axis.plot(
                x_values,
                mean_by_key[key],
                linewidth=1.8,
                label=key,
            )
        axis.set_title(method)
        axis.set_xlabel("cumulative_examples_processed")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("gradient norm")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Gradient Norms: Update vs Full Gradient")
    fig.tight_layout(rect=(0, 0.0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_final_function_mse_paired(
    *,
    seed_values: list[dict[str, Any]],
    methods: list[str],
    output_path: Path,
) -> None:
    seeds = sorted({row["model_seed"] for row in seed_values})
    by_method_seed = {
        (row["method_name"], row["model_seed"]): row for row in seed_values
    }
    fig, axis = plt.subplots(figsize=(max(7, 1.3 * len(methods)), 4.5))
    x_positions = np.arange(len(methods))
    for seed in seeds:
        y_values = [
            by_method_seed[(method, seed)]["test_function_mse"]
            for method in methods
        ]
        axis.plot(
            x_positions,
            y_values,
            marker="o",
            linewidth=1.2,
            alpha=0.85,
            label=f"seed {seed}",
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(methods, rotation=30, ha="right")
    axis.set_ylabel("test_function_mse")
    axis.set_title("Final Function MSE Paired by Model Seed")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_test_error_decomposition(
    *,
    seed_values: list[dict[str, Any]],
    sigma_squared: float,
    output_path: Path,
) -> None:
    methods = sorted({row["method_name"] for row in seed_values})
    noisy_means = []
    function_means = []
    for method in methods:
        method_rows = [row for row in seed_values if row["method_name"] == method]
        noisy_means.append(_mean([row["test_noisy_mse"] for row in method_rows]))
        function_means.append(_mean([row["test_function_mse"] for row in method_rows]))

    x_positions = np.arange(len(methods))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x_positions - width / 2, noisy_means, width, label="noisy test MSE")
    axis.bar(x_positions + width / 2, function_means, width, label="function MSE")
    axis.axhline(sigma_squared, color="black", linestyle="--", label="noise variance")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(methods, rotation=30, ha="right")
    axis.set_ylabel("MSE")
    axis.set_title("Noisy Test MSE, Function MSE, and Known Noise Variance")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _make_figures(
    *,
    runs: list[dict[str, Any]],
    seed_values: list[dict[str, Any]],
    figure_dir: Path,
    sigma_squared: float,
) -> list[str]:
    figure_paths: list[str] = []
    for comparison_name, methods in COMPARISONS.items():
        prefix = figure_dir / comparison_name
        figure_specs = [
            (
                "training_mse_vs_examples.png",
                "cumulative_examples_processed",
                "training_mse",
                "Training MSE vs Examples Processed",
            ),
            (
                "training_mse_vs_steps.png",
                "step",
                "training_mse",
                "Training MSE vs Optimiser Steps",
            ),
            (
                "function_mse_vs_examples.png",
                "cumulative_examples_processed",
                "validation_function_mse",
                "Validation Function MSE vs Examples Processed",
            ),
            (
                "function_mse_vs_wall_clock.png",
                "total_elapsed_seconds",
                "validation_function_mse",
                "Validation Function MSE vs Wall-Clock Time",
            ),
        ]
        for filename, x_key, y_key, title in figure_specs:
            output_path = prefix / filename
            _plot_history_metric(
                runs=runs,
                methods=methods,
                x_key=x_key,
                y_key=y_key,
                title=f"{comparison_name}: {title}",
                output_path=output_path,
            )
            figure_paths.append(str(output_path))

        output_path = prefix / "gradient_norms_vs_examples.png"
        _plot_gradient_norms(runs=runs, methods=methods, output_path=output_path)
        figure_paths.append(str(output_path))

        output_path = prefix / "final_function_mse_paired_by_seed.png"
        _plot_final_function_mse_paired(
            seed_values=seed_values,
            methods=methods,
            output_path=output_path,
        )
        figure_paths.append(str(output_path))

    test_decomposition_path = figure_dir / "test_error_decomposition.png"
    _plot_test_error_decomposition(
        seed_values=seed_values,
        sigma_squared=sigma_squared,
        output_path=test_decomposition_path,
    )
    figure_paths.append(str(test_decomposition_path))

    rr_methods = [
        "random_reshuffling_b32",
        "random_reshuffling_b256",
    ]
    epoch_path = figure_dir / "supplementary_random_reshuffling_epoch_plots.png"
    _plot_history_metric(
        runs=runs,
        methods=rr_methods,
        x_key="epoch",
        y_key="training_mse",
        title="Supplementary Random-Reshuffling Epoch Plot",
        output_path=epoch_path,
    )
    figure_paths.append(str(epoch_path))
    return figure_paths


def analyse_baseline(
    *,
    baseline_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
) -> dict[str, Any]:
    raw_root = Path(baseline_cfg["paths"]["results_raw_dir"]) / "week1_gradient_methods"
    run_root = raw_root / "baseline_comparison_runs"
    analysis_dir = raw_root / "baseline_analysis"
    figure_dir = (
        Path(baseline_cfg["paths"]["results_figures_dir"])
        / "week1_gradient_methods"
        / "baseline_analysis"
    )
    sigma_squared = float(baseline_cfg["dataset"]["noise_std"]) ** 2

    runs, validation_report = _load_runs(
        baseline_cfg=baseline_cfg,
        experiment_cfg=experiment_cfg,
        run_root=run_root,
    )
    _write_json(validation_report, analysis_dir / "validation_report.json")
    if not validation_report["ok"]:
        raise RuntimeError(
            "Baseline analysis validation failed. See validation_report.json."
        )

    seed_values, method_summary = _build_final_tables(runs)
    paired_differences, paired_summary = _paired_comparison_tables(seed_values)
    risk_rows, risk_summary = _risk_identity_tables(
        seed_values,
        sigma_squared=sigma_squared,
    )

    _write_csv(seed_values, analysis_dir / "method_seed_values.csv")
    _write_csv(method_summary, analysis_dir / "method_summary.csv")
    _write_csv(paired_differences, analysis_dir / "paired_differences.csv")
    _write_csv(paired_summary, analysis_dir / "paired_difference_summary.csv")
    _write_csv(risk_rows, analysis_dir / "risk_identity_by_run.csv")
    _write_csv(risk_summary, analysis_dir / "risk_identity_summary.csv")

    figure_paths = _make_figures(
        runs=runs,
        seed_values=seed_values,
        figure_dir=figure_dir,
        sigma_squared=sigma_squared,
    )

    manifest = {
        "analysis": "baseline_comparison",
        "run_count": len(runs),
        "comparisons": COMPARISONS,
        "metrics": FINAL_METRICS,
        "sigma_squared": sigma_squared,
        "interpretation_boundaries": {
            "exact_or_directly_calculated": [
                "empirical training loss",
                "validation and test loss",
                "function-estimation error",
                "noisy prediction error",
                "update-gradient norm",
                "full-gradient norm",
                "parameter norm",
                "optimiser steps",
                "examples processed",
                "data-equivalent passes",
                "elapsed time",
                "epoch and step-within-epoch where defined",
            ],
            "empirical_proxies": [
                "lowest training loss found across runs",
                "difference from the lowest observed reference loss",
                "achieved approximation quality of the network",
            ],
            "limitations": [
                "the neural-network empirical objective is non-convex",
                "a small full-gradient norm does not establish global optimality",
                "network parameter error is not a meaningful function-recovery metric",
                "random reshuffling is not conditionally unbiased like independent sampling",
                "an epoch is not defined for with-replacement methods",
                "wall-clock comparisons depend on framework and hardware overhead",
                "one common learning rate may not be each method's best practical tuning",
                "worst-case generalisation bounds may be looser than observed errors",
            ],
        },
        "artifacts": {
            "validation_report": analysis_dir / "validation_report.json",
            "method_seed_values": analysis_dir / "method_seed_values.csv",
            "method_summary": analysis_dir / "method_summary.csv",
            "paired_differences": analysis_dir / "paired_differences.csv",
            "paired_difference_summary": (
                analysis_dir / "paired_difference_summary.csv"
            ),
            "risk_identity_by_run": analysis_dir / "risk_identity_by_run.csv",
            "risk_identity_summary": analysis_dir / "risk_identity_summary.csv",
            "figures": figure_paths,
        },
        "claim_boundary": (
            "These results compare one target function, one architecture, one common "
            "learning rate and five paired seeds; they are not universal optimiser claims."
        ),
        "risk_identity_note": (
            "Finite-sample noisy_test_mse - function_mse differs from sigma_squared "
            "because realised test noise variance and the empirical cross term are "
            "not exactly their population expectations."
        ),
    }
    _write_json(manifest, analysis_dir / "analysis_manifest.json")
    return manifest


def main() -> None:
    baseline_cfg = _load_json(Path("configs/baseline.json"))
    experiment_cfg = _load_json(Path("configs/experiments/week1_gradient_methods.json"))
    manifest = analyse_baseline(
        baseline_cfg=baseline_cfg,
        experiment_cfg=experiment_cfg,
    )
    print(f"Analysed {manifest['run_count']} baseline runs.")
    print(
        "Analysis manifest: "
        "results/raw/week1_gradient_methods/baseline_analysis/analysis_manifest.json"
    )


if __name__ == "__main__":
    main()
