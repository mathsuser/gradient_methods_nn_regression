from __future__ import annotations

import argparse
import csv
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


DIMENSIONS = [6, 20, 100]
STRESS_DIMENSIONS = [20, 100]
BASELINE_PARAMETER_COUNT = 129
N_RELEVANT_FEATURES = 6
FINAL_METRICS = [
    "training_mse",
    "validation_mse",
    "test_noisy_mse",
    "test_function_mse",
    "training_elapsed_seconds",
    "run_elapsed_seconds",
    "parameter_count",
]
HISTORY_NUMERIC_FIELDS = [
    "dimension",
    "n_relevant_features",
    "parameter_count",
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
                if field in parsed:
                    parsed[field] = _parse_optional_float(parsed[field])
            history.append(parsed)
    return history


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(data), handle, indent=2)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _method_items(experiment_cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return list(experiment_cfg["experiment"]["methods"].items())


def _expected_stress_path(
    *,
    root: Path,
    dimension: int,
    method_name: str,
    model_seed: int,
    filename: str,
) -> Path:
    return root / f"dimension_{dimension}" / method_name / f"seed_{model_seed}" / filename


def preflight(
    *,
    baseline_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
) -> dict[str, Any]:
    model_seeds = [int(seed) for seed in baseline_cfg["optimisation"]["model_seeds"]]
    methods = [
        str(cfg.get("method_name", key))
        for key, cfg in _method_items(experiment_cfg)
    ]
    baseline_root = (
        Path(baseline_cfg["paths"]["results_raw_dir"])
        / "week1_gradient_methods"
        / "baseline_comparison_runs"
    )
    stress_root = Path("results/raw/week1_dimension_stress")
    missing_baseline: list[str] = []
    missing_stress: list[str] = []

    for method in methods:
        for seed in model_seeds:
            for filename in ["metadata.json", "history.csv"]:
                path = baseline_root / method / f"seed_{seed}" / filename
                if not path.exists():
                    missing_baseline.append(str(path))
            for dimension in STRESS_DIMENSIONS:
                for filename in ["metadata.json", "history.csv"]:
                    path = _expected_stress_path(
                        root=stress_root,
                        dimension=dimension,
                        method_name=method,
                        model_seed=seed,
                        filename=filename,
                    )
                    if not path.exists():
                        missing_stress.append(str(path))

    return {
        "baseline_reusable": not missing_baseline,
        "baseline_dimension": 6,
        "missing_baseline_files": missing_baseline,
        "stress_outputs_complete": not missing_stress,
        "missing_stress_file_count": len(missing_stress),
        "expected_new_run_count": len(STRESS_DIMENSIONS) * len(methods)
        * len(model_seeds),
        "dimensions": DIMENSIONS,
        "model_seeds": model_seeds,
        "methods": methods,
        "parameter_counts": {
            "6": BASELINE_PARAMETER_COUNT,
            "20": 353,
            "100": 1633,
        },
        "note": (
            "Full analysis requires complete d=20 and d=100 stress outputs. "
            "Preflight is allowed before those runs exist."
        ),
    }


def _load_one_run(
    *,
    dimension: int,
    n_relevant_features: int,
    parameter_count: int,
    metadata_path: Path,
    history_path: Path,
) -> dict[str, Any]:
    metadata = _load_json(metadata_path)
    history = _load_history(history_path)
    for row in history:
        row["dimension"] = dimension
        row["n_relevant_features"] = n_relevant_features
        row["parameter_count"] = parameter_count
    final = history[-1]
    return {
        "dimension": dimension,
        "n_relevant_features": n_relevant_features,
        "parameter_count": parameter_count,
        "method_name": metadata["method_name"],
        "sampling_method": metadata["sampling_method"],
        "nominal_batch_size": int(metadata["nominal_batch_size"]),
        "model_seed": int(metadata["model_seed"]),
        "metadata": metadata,
        "history": history,
        "final": final,
        "metadata_path": metadata_path,
        "history_path": history_path,
    }


def _load_all_runs(
    *,
    baseline_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = preflight(baseline_cfg=baseline_cfg, experiment_cfg=experiment_cfg)
    errors: list[str] = []
    if not report["baseline_reusable"]:
        errors.extend(report["missing_baseline_files"])
    if not report["stress_outputs_complete"]:
        errors.append(
            f"Missing {report['missing_stress_file_count']} stress output files"
        )
    if errors:
        return [], {**report, "ok": False, "errors": errors}

    model_seeds = [int(seed) for seed in baseline_cfg["optimisation"]["model_seeds"]]
    method_items = _method_items(experiment_cfg)
    baseline_root = (
        Path(baseline_cfg["paths"]["results_raw_dir"])
        / "week1_gradient_methods"
        / "baseline_comparison_runs"
    )
    stress_root = Path("results/raw/week1_dimension_stress")
    runs: list[dict[str, Any]] = []

    for method_key, method_cfg in method_items:
        method_name = str(method_cfg.get("method_name", method_key))
        for seed in model_seeds:
            runs.append(
                _load_one_run(
                    dimension=6,
                    n_relevant_features=N_RELEVANT_FEATURES,
                    parameter_count=BASELINE_PARAMETER_COUNT,
                    metadata_path=baseline_root / method_name / f"seed_{seed}"
                    / "metadata.json",
                    history_path=baseline_root / method_name / f"seed_{seed}"
                    / "history.csv",
                )
            )
            for dimension in STRESS_DIMENSIONS:
                metadata_path = _expected_stress_path(
                    root=stress_root,
                    dimension=dimension,
                    method_name=method_name,
                    model_seed=seed,
                    filename="metadata.json",
                )
                metadata = _load_json(metadata_path)
                runs.append(
                    _load_one_run(
                        dimension=dimension,
                        n_relevant_features=int(metadata["n_relevant_features"]),
                        parameter_count=int(metadata["parameter_count"]),
                        metadata_path=metadata_path,
                        history_path=_expected_stress_path(
                            root=stress_root,
                            dimension=dimension,
                            method_name=method_name,
                            model_seed=seed,
                            filename="history.csv",
                        ),
                    )
                )

    return runs, {**report, "ok": True, "errors": []}


def _final_row(run: dict[str, Any]) -> dict[str, Any]:
    final_metrics = run["metadata"]["final_metrics"]
    final = run["final"]
    return {
        "dimension": run["dimension"],
        "n_relevant_features": run["n_relevant_features"],
        "parameter_count": run["parameter_count"],
        "method_name": run["method_name"],
        "sampling_method": run["sampling_method"],
        "nominal_batch_size": run["nominal_batch_size"],
        "model_seed": run["model_seed"],
        "training_mse": float(final["training_mse"]),
        "validation_mse": float(final["validation_mse"]),
        "validation_function_mse": float(final["validation_function_mse"]),
        "test_noisy_mse": float(final_metrics["test_prediction_mse"]),
        "test_function_mse": float(final_metrics["test_function_mse"]),
        "training_elapsed_seconds": float(final["training_elapsed_seconds"]),
        "run_elapsed_seconds": float(
            run["metadata"]["elapsed_time"].get(
                "run_elapsed_seconds",
                run["metadata"]["elapsed_time"]["total_elapsed_seconds"],
            )
        ),
        "optimiser_steps": int(final["step"]),
        "examples_processed": int(final["cumulative_examples_processed"]),
        "data_equivalent_passes": float(final["data_equivalent_passes"]),
    }


def _build_tables(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows = [_final_row(run) for run in runs]
    summary_rows: list[dict[str, Any]] = []
    variability_rows: list[dict[str, Any]] = []
    methods = sorted({row["method_name"] for row in final_rows})

    for dimension in DIMENSIONS:
        for method in methods:
            rows = [
                row for row in final_rows
                if row["dimension"] == dimension and row["method_name"] == method
            ]
            for metric in FINAL_METRICS:
                values = [float(row[metric]) for row in rows]
                summary_rows.append(
                    {
                        "dimension": dimension,
                        "method_name": method,
                        "metric": metric,
                        **_summary(values),
                    }
                )
            variability_rows.append(
                {
                    "dimension": dimension,
                    "method_name": method,
                    "metric": "test_function_mse",
                    "seed_std": _summary(
                        [float(row["test_function_mse"]) for row in rows]
                    )["std"],
                    "seed_min": min(float(row["test_function_mse"]) for row in rows),
                    "seed_max": max(float(row["test_function_mse"]) for row in rows),
                }
            )

    return final_rows, summary_rows, variability_rows


def _paired_changes(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["dimension"], row["method_name"], row["model_seed"]): row
        for row in final_rows
    }
    rows: list[dict[str, Any]] = []
    methods = sorted({row["method_name"] for row in final_rows})
    seeds = sorted({row["model_seed"] for row in final_rows})
    metrics = [
        "test_function_mse",
        "test_noisy_mse",
        "training_mse",
        "validation_function_mse",
        "run_elapsed_seconds",
    ]
    for dimension in STRESS_DIMENSIONS:
        for method in methods:
            for seed in seeds:
                baseline = by_key[(6, method, seed)]
                stress = by_key[(dimension, method, seed)]
                for metric in metrics:
                    rows.append(
                        {
                            "dimension": dimension,
                            "baseline_dimension": 6,
                            "method_name": method,
                            "model_seed": seed,
                            "metric": metric,
                            "stress_value": stress[metric],
                            "baseline_value": baseline[metric],
                            "stress_minus_baseline": (
                                float(stress[metric]) - float(baseline[metric])
                            ),
                        }
                    )
    return rows


def _parameter_count_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        values = {
            int(row["parameter_count"])
            for row in final_rows
            if row["dimension"] == dimension
        }
        rows.append(
            {
                "dimension": dimension,
                "n_relevant_features": N_RELEVANT_FEATURES,
                "parameter_count": values.pop(),
                "design_note": (
                    "Input dimension changes ambient features and first-layer "
                    "parameter count; hidden width remains fixed at 16."
                ),
            }
        )
    return rows


def _mean_metric(
    final_rows: list[dict[str, Any]],
    *,
    dimension: int,
    method: str,
    metric: str,
) -> float:
    values = [
        float(row[metric])
        for row in final_rows
        if row["dimension"] == dimension and row["method_name"] == method
    ]
    return float(statistics.mean(values))


def _plot_final_metric(
    *,
    final_rows: list[dict[str, Any]],
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    methods = sorted({row["method_name"] for row in final_rows})
    x_positions = np.arange(len(methods))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11, 5))
    for index, dimension in enumerate(DIMENSIONS):
        values = [
            _mean_metric(
                final_rows,
                dimension=dimension,
                method=method,
                metric=metric,
            )
            for method in methods
        ]
        axis.bar(
            x_positions + (index - 1) * width,
            values,
            width,
            label=f"d={dimension}",
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(methods, rotation=30, ha="right")
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_history_metric(
    *,
    runs: list[dict[str, Any]],
    y_key: str,
    title: str,
    output_path: Path,
) -> None:
    methods = sorted({run["method_name"] for run in runs})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes_flat = axes.flatten()
    for axis, method in zip(axes_flat, methods):
        for dimension in DIMENSIONS:
            dim_runs = [
                run for run in runs
                if run["method_name"] == method and run["dimension"] == dimension
            ]
            histories = [run["history"] for run in dim_runs]
            x_values = [
                row["cumulative_examples_processed"]
                for row in histories[0]
                if row[y_key] is not None
            ]
            curves = [
                [row[y_key] for row in history if row[y_key] is not None]
                for history in histories
            ]
            min_len = min(len(curve) for curve in curves)
            y_values = [
                float(np.mean([curve[idx] for curve in curves]))
                for idx in range(min_len)
            ]
            axis.plot(x_values[:min_len], y_values, label=f"d={dimension}")
        axis.set_title(method)
        axis.set_xlabel("examples processed")
        axis.grid(True, alpha=0.25)
    axes_flat[0].set_ylabel(y_key)
    axes_flat[-1].legend()
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_parameter_counts(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        [str(row["dimension"]) for row in rows],
        [row["parameter_count"] for row in rows],
    )
    axis.set_xlabel("dimension")
    axis.set_ylabel("parameter_count")
    axis.set_title("Parameter Count by Dimension")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def analyse_dimension_stress(
    *,
    baseline_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
) -> dict[str, Any]:
    raw_root = Path("results/raw/week1_dimension_stress")
    analysis_dir = raw_root / "analysis"
    figure_dir = Path("results/figures/week1_dimension_stress")

    runs, validation_report = _load_all_runs(
        baseline_cfg=baseline_cfg,
        experiment_cfg=experiment_cfg,
    )
    _write_json(validation_report, analysis_dir / "validation_report.json")
    if not validation_report["ok"]:
        raise RuntimeError(
            "Dimension stress analysis validation failed. "
            "See validation_report.json."
        )

    final_rows, summary_rows, variability_rows = _build_tables(runs)
    paired_rows = _paired_changes(final_rows)
    parameter_rows = _parameter_count_rows(final_rows)

    _write_csv(final_rows, analysis_dir / "dimension_method_seed_values.csv")
    _write_csv(summary_rows, analysis_dir / "dimension_method_summary.csv")
    _write_csv(variability_rows, analysis_dir / "seed_variability.csv")
    _write_csv(paired_rows, analysis_dir / "paired_changes_vs_d6.csv")
    _write_csv(parameter_rows, analysis_dir / "parameter_counts.csv")

    figures = [
        figure_dir / "final_function_mse_by_dimension_method.png",
        figure_dir / "final_noisy_test_mse_by_dimension_method.png",
        figure_dir / "training_mse_vs_examples_by_dimension.png",
        figure_dir / "validation_function_mse_vs_examples_by_dimension.png",
        figure_dir / "training_elapsed_time_by_dimension_method.png",
        figure_dir / "seed_variability_function_mse.png",
        figure_dir / "parameter_count_by_dimension.png",
    ]
    _plot_final_metric(
        final_rows=final_rows,
        metric="test_function_mse",
        title="Final Function MSE by Dimension and Method",
        output_path=figures[0],
    )
    _plot_final_metric(
        final_rows=final_rows,
        metric="test_noisy_mse",
        title="Final Noisy Test MSE by Dimension and Method",
        output_path=figures[1],
    )
    _plot_history_metric(
        runs=runs,
        y_key="training_mse",
        title="Training MSE vs Examples Processed by Dimension",
        output_path=figures[2],
    )
    _plot_history_metric(
        runs=runs,
        y_key="validation_function_mse",
        title="Validation Function MSE vs Examples Processed by Dimension",
        output_path=figures[3],
    )
    _plot_final_metric(
        final_rows=final_rows,
        metric="training_elapsed_seconds",
        title="Training Elapsed Time by Dimension and Method",
        output_path=figures[4],
    )
    _plot_final_metric(
        final_rows=[
            {
                "dimension": row["dimension"],
                "method_name": row["method_name"],
                "model_seed": 0,
                "test_function_mse_seed_std": row["seed_std"],
            }
            for row in variability_rows
        ],
        metric="test_function_mse_seed_std",
        title="Variability Across Paired Seeds",
        output_path=figures[5],
    )
    _plot_parameter_counts(parameter_rows, figures[6])

    manifest = {
        "analysis": "week1_dimension_stress",
        "baseline_reuse": {
            "dimension": 6,
            "source": "results/raw/week1_gradient_methods/baseline_comparison_runs",
            "parameter_count": BASELINE_PARAMETER_COUNT,
        },
        "stress_dimensions": STRESS_DIMENSIONS,
        "run_count_including_reused_baseline": len(runs),
        "new_stress_run_count": len(runs) - 30,
        "dimension_note": (
            "Increasing dimension also increases the number of first-layer "
            "trainable parameters because hidden width remains fixed."
        ),
        "claim_boundary": (
            "This is one target function, one architecture family, one common "
            "learning rate and five paired seeds; it is not a universal method "
            "ranking."
        ),
        "artifacts": {
            "validation_report": analysis_dir / "validation_report.json",
            "dimension_method_seed_values": (
                analysis_dir / "dimension_method_seed_values.csv"
            ),
            "dimension_method_summary": (
                analysis_dir / "dimension_method_summary.csv"
            ),
            "seed_variability": analysis_dir / "seed_variability.csv",
            "paired_changes_vs_d6": analysis_dir / "paired_changes_vs_d6.csv",
            "parameter_counts": analysis_dir / "parameter_counts.csv",
            "figures": figures,
        },
    }
    _write_json(manifest, analysis_dir / "analysis_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check baseline reuse and expected stress outputs without analysing.",
    )
    args = parser.parse_args()

    baseline_cfg = _load_json(Path("configs/baseline.json"))
    experiment_cfg = _load_json(Path("configs/experiments/week1_gradient_methods.json"))

    if args.preflight:
        report = preflight(
            baseline_cfg=baseline_cfg,
            experiment_cfg=experiment_cfg,
        )
        print(json.dumps(_to_jsonable(report), indent=2))
        return

    manifest = analyse_dimension_stress(
        baseline_cfg=baseline_cfg,
        experiment_cfg=experiment_cfg,
    )
    print(
        "Analysed dimension stress runs: "
        f"{manifest['run_count_including_reused_baseline']} total including d=6"
    )
    print("Manifest: results/raw/week1_dimension_stress/analysis/analysis_manifest.json")


if __name__ == "__main__":
    main()
