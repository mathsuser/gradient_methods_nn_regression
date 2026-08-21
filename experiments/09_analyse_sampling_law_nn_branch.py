from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Optional, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gradient_methods_nn_regression_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


BRANCH_CONFIG_PATH = Path("configs/experiments/sampling_law_nn_branch.json")
BASELINE_CONFIG_PATH = Path("configs/baseline.json")

METHODS = ["wr_1", "rr_1"]
METRIC_MAP = {
    "function_mse": "validation_function_mse",
    "training_mse": "training_mse",
    "full_gradient_norm": "full_gradient_norm",
}
TERMINAL_METRICS = [
    "training_mse",
    "evaluation_function_mse",
    "full_gradient_norm",
    "evaluation_noisy_mse",
    "training_elapsed_seconds",
    "total_elapsed_seconds",
    "run_elapsed_seconds",
]
HISTORY_NUMERIC_FIELDS = [
    "trajectory_id",
    "sampling_seed",
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


class IntegrityError(RuntimeError):
    def __init__(self, failures: list[str]) -> None:
        super().__init__("\n".join(failures))
        self.failures = failures


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


def parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed


def load_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        history: list[dict[str, Any]] = []
        for row in reader:
            parsed = dict(row)
            for field in HISTORY_NUMERIC_FIELDS:
                if field in row:
                    parsed[field] = parse_optional_float(row.get(field))
            history.append(parsed)
    return history


def branch_raw_dir(branch_cfg: dict[str, Any]) -> Path:
    return Path(branch_cfg["experiment"]["paths"]["raw_dir"])


def branch_figures_dir(branch_cfg: dict[str, Any]) -> Path:
    return Path(branch_cfg["experiment"]["paths"]["figures_dir"])


def expected_checkpoint_examples(branch_cfg: dict[str, Any]) -> list[int]:
    training_cfg = branch_cfg["experiment"]["training"]
    target = int(training_cfg["target_examples_processed"])
    every = int(training_cfg["evaluation_every_examples"])
    checkpoints = [0, *range(every, target + 1, every)]
    if checkpoints[-1] != target:
        checkpoints.append(target)
    return checkpoints


def sampling_seed_for(
    *,
    branch_cfg: dict[str, Any],
    method_name: str,
    trajectory_id: int,
) -> int:
    trajectory_ids = [
        int(value)
        for value in branch_cfg["experiment"]["sampling_seeds"]["trajectory_ids"]
    ]
    seed_index = trajectory_ids.index(int(trajectory_id))
    return int(branch_cfg["experiment"]["sampling_seeds"]["by_method"][method_name][seed_index])


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _history_path_for(run_root: Path, method: str, trajectory_id: int) -> Path:
    return run_root / method / f"trajectory_{trajectory_id}" / "history.csv"


def _metadata_path_for(run_root: Path, method: str, trajectory_id: int) -> Path:
    return run_root / method / f"trajectory_{trajectory_id}" / "metadata.json"


def load_and_validate_runs(
    *,
    baseline_cfg: dict[str, Any],
    branch_cfg: dict[str, Any],
    run_root: Optional[Path] = None,
    expected_trajectory_ids: Optional[list[int]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del baseline_cfg
    run_root = branch_raw_dir(branch_cfg) if run_root is None else run_root
    expected_trajectory_ids = (
        [int(value) for value in branch_cfg["experiment"]["sampling_seeds"]["trajectory_ids"]]
        if expected_trajectory_ids is None
        else [int(value) for value in expected_trajectory_ids]
    )

    training_cfg = branch_cfg["experiment"]["training"]
    target_examples = int(training_cfg["target_examples_processed"])
    learning_rate = float(training_cfg["learning_rate"])
    model_seed = int(training_cfg["model_seed"])
    n_train = int(branch_cfg["experiment"].get("n_train", 5000))
    expected_passes = target_examples / n_train
    checkpoints = expected_checkpoint_examples(branch_cfg)
    expected_run_count = len(METHODS) * len(expected_trajectory_ids)

    failures: list[str] = []
    runs: list[dict[str, Any]] = []
    initial_checksums: set[str] = set()
    data_provenance: list[dict[str, Any]] = []
    expected_run_dirs = {
        run_root / method / f"trajectory_{trajectory_id}"
        for method in METHODS
        for trajectory_id in expected_trajectory_ids
    }
    discovered_run_dirs = {
        path
        for path in run_root.glob("*/trajectory_*")
        if path.is_dir()
    }
    unexpected_run_dirs = sorted(discovered_run_dirs - expected_run_dirs)
    if unexpected_run_dirs:
        failures.extend(f"unexpected run directory: {path}" for path in unexpected_run_dirs)

    if set(branch_cfg["experiment"]["methods"]) != set(METHODS):
        failures.append("branch config must contain exactly wr_1 and rr_1")

    for method in METHODS:
        method_cfg = branch_cfg["experiment"]["methods"].get(method, {})
        if int(method_cfg.get("batch_size", -1)) != 1:
            failures.append(f"{method}: batch_size is not 1")
        for trajectory_id in expected_trajectory_ids:
            metadata_path = _metadata_path_for(run_root, method, trajectory_id)
            history_path = _history_path_for(run_root, method, trajectory_id)
            if not metadata_path.exists():
                failures.append(f"missing metadata: {metadata_path}")
                continue
            if not history_path.exists():
                failures.append(f"missing history: {history_path}")
                continue

            metadata = load_json(metadata_path)
            history = load_history(history_path)
            if not history:
                failures.append(f"empty history: {history_path}")
                continue

            final = history[-1]
            observed_checkpoints = [int(row["checkpoint_examples"]) for row in history]
            expected_seed = sampling_seed_for(
                branch_cfg=branch_cfg,
                method_name=method,
                trajectory_id=trajectory_id,
            )
            metadata_data_paths = metadata.get("data_paths")
            if isinstance(metadata_data_paths, dict):
                data_provenance.append(metadata_data_paths)

            checks = [
                (metadata.get("method_name") == method, "metadata method_name"),
                (int(metadata.get("trajectory_id", -1)) == trajectory_id, "trajectory_id"),
                (int(metadata.get("sampling_seed", -1)) == expected_seed, "sampling_seed"),
                (int(metadata.get("model_seed", -1)) == model_seed, "model_seed"),
                (float(metadata.get("learning_rate", math.nan)) == learning_rate, "learning_rate"),
                (int(metadata.get("batch_size", -1)) == 1, "batch_size"),
                (
                    metadata.get("initial_state_checksum")
                    == metadata.get("loaded_initial_state_checksum"),
                    "loaded initial-state checksum",
                ),
                (int(final["step"]) == target_examples, "final step"),
                (
                    int(final["cumulative_examples_processed"]) == target_examples,
                    "final examples processed",
                ),
                (
                    math.isclose(
                        float(final["data_equivalent_passes"]),
                        expected_passes,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    "final data-equivalent passes",
                ),
                (observed_checkpoints == checkpoints, "checkpoint schedule"),
                (len(history) == len(checkpoints), "checkpoint count"),
            ]
            for ok, label in checks:
                if not ok:
                    failures.append(f"{metadata_path}: invalid {label}")

            checksum = metadata.get("initial_state_checksum")
            if checksum:
                initial_checksums.add(str(checksum))
            else:
                failures.append(f"{metadata_path}: missing initial_state_checksum")

            for row_index, row in enumerate(history):
                for field in METRIC_MAP.values():
                    if not _finite(row.get(field)):
                        failures.append(f"{history_path}: non-finite {field} at row {row_index}")
                if "validation_mse" in row and row["validation_mse"] is not None:
                    if not _finite(row["validation_mse"]):
                        failures.append(
                            f"{history_path}: non-finite validation_mse at row {row_index}"
                        )

            runs.append(
                {
                    "method": method,
                    "trajectory_id": trajectory_id,
                    "sampling_seed": expected_seed,
                    "metadata": metadata,
                    "history": history,
                    "metadata_path": metadata_path,
                    "history_path": history_path,
                }
            )

    method_counts = {
        method: sum(1 for run in runs if run["method"] == method)
        for method in METHODS
    }
    if len(runs) != expected_run_count:
        failures.append(f"expected {expected_run_count} total runs, found {len(runs)}")
    for method in METHODS:
        expected_method_count = len(expected_trajectory_ids)
        if method_counts[method] != expected_method_count:
            failures.append(
                f"expected {expected_method_count} {method} runs, "
                f"found {method_counts[method]}"
            )
    if len(initial_checksums) != 1:
        failures.append(
            f"expected one shared initial_state_checksum, found {len(initial_checksums)}"
        )
    if data_provenance and any(item != data_provenance[0] for item in data_provenance):
        failures.append("data provenance differs across runs")

    if failures:
        raise IntegrityError(failures)

    audit = {
        "run_count": len(runs),
        "method_counts": method_counts,
        "checkpoint_count": sum(len(run["history"]) for run in runs),
        "expected_checkpoint_count": expected_run_count * len(checkpoints),
        "checkpoint_examples": checkpoints,
        "initial_state_checksum": next(iter(initial_checksums)),
        "data_provenance": data_provenance[0] if data_provenance else {},
    }
    return runs, audit


def descriptive_stats(values: list[float]) -> dict[str, Union[float, int]]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def checkpoint_summary_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoint_values = sorted(
        {
            int(row["checkpoint_examples"])
            for run in runs
            for row in run["history"]
        }
    )
    for method in METHODS:
        method_runs = [run for run in runs if run["method"] == method]
        for checkpoint in checkpoint_values:
            for metric, history_field in METRIC_MAP.items():
                values = [
                    float(row[history_field])
                    for run in method_runs
                    for row in run["history"]
                    if int(row["checkpoint_examples"]) == checkpoint
                ]
                rows.append(
                    {
                        "method": method,
                        "checkpoint_examples": checkpoint,
                        "metric": metric,
                        **descriptive_stats(values),
                    }
                )
    return rows


def terminal_runs_rows(runs: list[dict[str, Any]], *, final_checkpoint: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: (item["method"], item["trajectory_id"])):
        matches = [
            row
            for row in run["history"]
            if int(row["checkpoint_examples"]) == int(final_checkpoint)
        ]
        if len(matches) != 1:
            raise IntegrityError(
                [
                    f"{run['history_path']}: expected one terminal checkpoint "
                    f"at {final_checkpoint}, found {len(matches)}"
                ]
            )
        final = matches[0]
        elapsed = run["metadata"].get("elapsed_time", {})
        rows.append(
            {
                "method": run["method"],
                "trajectory_id": int(run["trajectory_id"]),
                "sampling_seed": int(run["sampling_seed"]),
                "training_mse": float(final["training_mse"]),
                "evaluation_function_mse": float(final["validation_function_mse"]),
                "full_gradient_norm": float(final["full_gradient_norm"]),
                "evaluation_noisy_mse": final.get("validation_mse"),
                "training_elapsed_seconds": final.get("training_elapsed_seconds"),
                "total_elapsed_seconds": final.get("total_elapsed_seconds"),
                "run_elapsed_seconds": elapsed.get("run_elapsed_seconds"),
            }
        )
    return rows


def terminal_summary_rows(terminal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in terminal_rows if row["method"] == method]
        for metric in TERMINAL_METRICS:
            values = [
                float(row[metric])
                for row in method_rows
                if row.get(metric) is not None
            ]
            if values:
                rows.append({"method": method, "metric": metric, **descriptive_stats(values)})
    return rows


def terminal_comparison_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["method"], row["metric"]): row for row in summary_rows}
    rows: list[dict[str, Any]] = []
    for metric in TERMINAL_METRICS:
        wr = by_key.get(("wr_1", metric))
        rr = by_key.get(("rr_1", metric))
        if wr is None or rr is None:
            continue
        rows.append(
            {
                "metric": metric,
                "wr_mean": wr["mean"],
                "rr_mean": rr["mean"],
                "rr_minus_wr_mean": float(rr["mean"]) - float(wr["mean"]),
                "wr_median": wr["median"],
                "rr_median": rr["median"],
                "rr_minus_wr_median": float(rr["median"]) - float(wr["median"]),
                "wr_std": wr["std"],
                "rr_std": rr["std"],
                "rr_minus_wr_std": float(rr["std"]) - float(wr["std"]),
            }
        )
    return rows


def checkpoint_difference_rows(
    checkpoint_rows: list[dict[str, Any]],
    *,
    metric: str = "function_mse",
    exclude_checkpoint_zero: bool = True,
) -> list[dict[str, Any]]:
    by_key = {
        (row["method"], row["metric"], int(row["checkpoint_examples"])): row
        for row in checkpoint_rows
    }
    checkpoints = sorted(
        {
            int(row["checkpoint_examples"])
            for row in checkpoint_rows
            if row["metric"] == metric
        }
    )
    if exclude_checkpoint_zero:
        checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint != 0]

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        wr = by_key[("wr_1", metric, checkpoint)]
        rr = by_key[("rr_1", metric, checkpoint)]
        rows.append(
            {
                "checkpoint_examples": checkpoint,
                "wr_mean": wr["mean"],
                "rr_mean": rr["mean"],
                "rr_minus_wr_mean": float(rr["mean"]) - float(wr["mean"]),
                "wr_median": wr["median"],
                "rr_median": rr["median"],
                "rr_minus_wr_median": float(rr["median"]) - float(wr["median"]),
                "wr_std": wr["std"],
                "rr_std": rr["std"],
                "rr_minus_wr_std": float(rr["std"]) - float(wr["std"]),
            }
        )
    return rows


def function_mse_trajectory_series(
    runs: list[dict[str, Any]],
    *,
    exclude_checkpoint_zero: bool = False,
) -> dict[str, dict[str, Any]]:
    series: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        method_runs = [run for run in runs if run["method"] == method]
        checkpoints = [
            int(row["checkpoint_examples"]) for row in method_runs[0]["history"]
        ]
        values = np.asarray(
            [
                [float(row["validation_function_mse"]) for row in run["history"]]
                for run in method_runs
            ],
            dtype=float,
        )
        if exclude_checkpoint_zero:
            keep = [index for index, checkpoint in enumerate(checkpoints) if checkpoint != 0]
            checkpoints = [checkpoints[index] for index in keep]
            values = values[:, keep]
        series[method] = {
            "checkpoints": checkpoints,
            "median": np.median(values, axis=0),
            "q25": np.quantile(values, 0.25, axis=0),
            "q75": np.quantile(values, 0.75, axis=0),
        }
    return series


def plot_function_mse_zoomed_trajectory(
    runs: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    series = function_mse_trajectory_series(
        runs,
        exclude_checkpoint_zero=True,
    )
    for method, color in [("wr_1", "tab:blue"), ("rr_1", "tab:green")]:
        method_series = series[method]
        checkpoints = method_series["checkpoints"]
        median = method_series["median"]
        q25 = method_series["q25"]
        q75 = method_series["q75"]
        axis.plot(checkpoints, median, label=method, color=color)
        axis.fill_between(checkpoints, q25, q75, color=color, alpha=0.18, linewidth=0)
    axis.set_xlabel("examples processed")
    axis.set_ylabel("evaluation function MSE")
    axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_function_mse_median_difference(
    difference_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    checkpoints = [int(row["checkpoint_examples"]) for row in difference_rows]
    differences = [float(row["rr_minus_wr_median"]) for row in difference_rows]
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    axis.plot(checkpoints, differences, color="tab:purple", marker="o")
    axis.set_xlabel("examples processed")
    axis.set_ylabel("RR median - WR median function MSE")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_terminal_function_mse_points(
    terminal_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5.8, 4.2))
    method_positions = {"wr_1": 1.0, "rr_1": 2.0}
    colors = {"wr_1": "tab:blue", "rr_1": "tab:green"}
    box_values: list[list[float]] = []
    labels: list[str] = []

    for method in METHODS:
        values = [
            float(row["evaluation_function_mse"])
            for row in terminal_rows
            if row["method"] == method
        ]
        box_values.append(values)
        labels.append(method)
        offsets = np.linspace(-0.12, 0.12, len(values)) if values else []
        x_values = [method_positions[method] + float(offset) for offset in offsets]
        axis.scatter(
            x_values,
            values,
            color=colors[method],
            alpha=0.75,
            s=24,
            edgecolors="none",
            zorder=3,
        )

    boxplot_kwargs = {
        "positions": [method_positions[method] for method in METHODS],
        "widths": 0.42,
        "showfliers": False,
        "medianprops": {"color": "black", "linewidth": 1.4},
        "boxprops": {"color": "0.35"},
        "whiskerprops": {"color": "0.35"},
        "capprops": {"color": "0.35"},
    }
    try:
        axis.boxplot(box_values, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        axis.boxplot(box_values, labels=labels, **boxplot_kwargs)
    axis.set_xlabel("method")
    axis.set_ylabel("terminal evaluation function MSE")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def analyse_sampling_law_branch(
    *,
    baseline_cfg: dict[str, Any],
    branch_cfg: dict[str, Any],
    run_root: Optional[Path] = None,
    analysis_dir: Optional[Path] = None,
    figure_dir: Optional[Path] = None,
) -> dict[str, Any]:
    run_root = branch_raw_dir(branch_cfg) if run_root is None else run_root
    analysis_dir = run_root / "analysis" if analysis_dir is None else analysis_dir
    figure_dir = branch_figures_dir(branch_cfg) if figure_dir is None else figure_dir

    runs, audit = load_and_validate_runs(
        baseline_cfg=baseline_cfg,
        branch_cfg=branch_cfg,
        run_root=run_root,
    )
    checkpoints = expected_checkpoint_examples(branch_cfg)
    terminal_rows = terminal_runs_rows(runs, final_checkpoint=checkpoints[-1])
    checkpoint_rows = checkpoint_summary_rows(runs)
    terminal_summary = terminal_summary_rows(terminal_rows)
    comparison_rows = terminal_comparison_rows(terminal_summary)
    checkpoint_difference = checkpoint_difference_rows(checkpoint_rows)

    checkpoint_summary_path = analysis_dir / "checkpoint_summary.csv"
    checkpoint_difference_path = analysis_dir / "checkpoint_wr_rr_difference.csv"
    terminal_runs_path = analysis_dir / "terminal_runs.csv"
    terminal_summary_path = analysis_dir / "terminal_summary.csv"
    terminal_comparison_path = analysis_dir / "terminal_wr_rr_comparison.csv"
    zoomed_trajectory_figure_path = figure_dir / "function_mse_vs_examples_zoomed.png"
    median_difference_figure_path = figure_dir / "function_mse_median_difference.png"
    terminal_points_figure_path = figure_dir / "terminal_function_mse_points.png"
    manifest_path = analysis_dir / "analysis_manifest.json"

    write_csv(checkpoint_rows, checkpoint_summary_path)
    write_csv(checkpoint_difference, checkpoint_difference_path)
    write_csv(terminal_rows, terminal_runs_path)
    write_csv(terminal_summary, terminal_summary_path)
    write_csv(comparison_rows, terminal_comparison_path)
    plot_function_mse_zoomed_trajectory(
        runs,
        zoomed_trajectory_figure_path,
    )
    plot_function_mse_median_difference(
        checkpoint_difference,
        median_difference_figure_path,
    )
    plot_terminal_function_mse_points(terminal_rows, terminal_points_figure_path)

    manifest = {
        "ok": True,
        "integrity": audit,
        "outputs": {
            "checkpoint_summary_csv": checkpoint_summary_path,
            "checkpoint_wr_rr_difference_csv": checkpoint_difference_path,
            "terminal_runs_csv": terminal_runs_path,
            "terminal_summary_csv": terminal_summary_path,
            "terminal_wr_rr_comparison_csv": terminal_comparison_path,
            "function_mse_vs_examples_zoomed_png": zoomed_trajectory_figure_path,
            "function_mse_median_difference_png": median_difference_figure_path,
            "terminal_function_mse_points_png": terminal_points_figure_path,
        },
    }
    write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-config", type=Path, default=BRANCH_CONFIG_PATH)
    parser.add_argument("--baseline-config", type=Path, default=BASELINE_CONFIG_PATH)
    args = parser.parse_args()

    try:
        manifest = analyse_sampling_law_branch(
            baseline_cfg=load_json(args.baseline_config),
            branch_cfg=load_json(args.branch_config),
        )
    except IntegrityError as exc:
        print("Integrity gate failed:")
        for failure in exc.failures:
            print(f"- {failure}")
        raise SystemExit(1) from exc

    print(json.dumps(to_jsonable(manifest), indent=2))


if __name__ == "__main__":
    main()
