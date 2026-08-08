from __future__ import annotations

import copy
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from gradient_methods_nn_regression.model import TinyRegressionModel
from gradient_methods_nn_regression.training import train_model


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_split(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(path) as data:
        x_np = data["x"].astype(np.float32, copy=False)
        y_np = data["y"].astype(np.float32, copy=False).reshape(-1, 1)
        f_true_np = data["f_true"].astype(np.float32, copy=False).reshape(-1, 1)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    f_true = torch.from_numpy(f_true_np)
    return x, y, f_true


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return float(value)
        return str(value)
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def _history_is_finite(history: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    numeric_fields = [
        "training_mse",
        "validation_mse",
        "validation_function_mse",
        "full_gradient_norm",
        "parameter_norm",
        "training_elapsed_seconds",
        "total_elapsed_seconds",
    ]

    for record in history:
        for field in numeric_fields:
            value = float(record[field])
            if not math.isfinite(value):
                reasons.append(f"{field} became non-finite")
                return False, reasons

        if record["update_gradient_norm"] is not None:
            update_value = float(record["update_gradient_norm"])
            if not math.isfinite(update_value):
                reasons.append("update_gradient_norm became non-finite")
                return False, reasons

        if record["batch_loss"] is not None:
            batch_value = float(record["batch_loss"])
            if not math.isfinite(batch_value):
                reasons.append("batch_loss became non-finite")
                return False, reasons

    return True, reasons


def _is_explosive_without_recovery(history: list[dict[str, Any]]) -> bool:
    training_curve = [float(row["training_mse"]) for row in history]
    if not training_curve:
        return False

    initial = training_curve[0]
    max_value = max(training_curve)
    min_value = min(training_curve)
    final_value = training_curve[-1]

    explosive_threshold = max(1.0e6, 100.0 * max(initial, 1.0e-12))
    if max_value < explosive_threshold:
        return False

    # If the run spikes but closes near its best achieved value, treat it as recovered.
    return final_value > 5.0 * max(min_value, 1.0e-12)


def _material_reduction(history: list[dict[str, Any]], *, ratio_threshold: float) -> bool:
    initial = float(history[0]["training_mse"])
    final = float(history[-1]["training_mse"])
    return final <= ratio_threshold * initial


def _summarise_run(
    *,
    history: list[dict[str, Any]],
    method_name: str,
    sampling_method: str,
    learning_rate: float,
    material_ratio_threshold: float,
) -> dict[str, Any]:
    finite_ok, finite_reasons = _history_is_finite(history)

    reasons: list[str] = []
    if not finite_ok:
        reasons.extend(finite_reasons)

    if finite_ok and _is_explosive_without_recovery(history):
        reasons.append("trajectory became explosive and did not recover")

    stable = len(reasons) == 0
    useful = stable and _material_reduction(
        history,
        ratio_threshold=material_ratio_threshold,
    )
    if stable and not useful:
        reasons.append(
            f"training_mse reduction below material threshold ({material_ratio_threshold:.2f}x initial)"
        )

    update_gradients = [
        float(row["update_gradient_norm"])
        for row in history
        if row["update_gradient_norm"] is not None
    ]
    max_update_gradient_norm = (
        max(update_gradients) if update_gradients else None
    )

    summary = {
        "method": method_name,
        "sampling_method": sampling_method,
        "learning_rate": learning_rate,
        "stable": stable,
        "useful": useful,
        "reasons": reasons,
        "initial_training_mse": float(history[0]["training_mse"]),
        "final_training_mse": float(history[-1]["training_mse"]),
        "final_validation_mse": float(history[-1]["validation_mse"]),
        "final_validation_function_mse": float(history[-1]["validation_function_mse"]),
        "max_update_gradient_norm": max_update_gradient_norm,
        "max_full_gradient_norm": max(float(row["full_gradient_norm"]) for row in history),
        "max_parameter_norm": max(float(row["parameter_norm"]) for row in history),
        "training_elapsed_seconds": float(history[-1]["training_elapsed_seconds"]),
        "total_elapsed_seconds": float(history[-1]["total_elapsed_seconds"]),
        "actual_steps": int(history[-1]["step"]),
        "actual_examples_processed": int(history[-1]["cumulative_examples_processed"]),
    }
    return summary


def _commit_hash() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        head_path = Path(".git") / "HEAD"
        try:
            head = head_path.read_text(encoding="utf-8").strip()
            if head.startswith("ref:"):
                ref_path = Path(".git") / head.split(" ", maxsplit=1)[1]
                return ref_path.read_text(encoding="utf-8").strip()
            return head
        except Exception:
            return "unknown"


def _plot_metric(
    *,
    histories: dict[tuple[str, float], list[dict[str, Any]]],
    method_order: list[str],
    learning_rates: list[float],
    x_key: str,
    y_key: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    axes_flat = axes.flatten()

    for idx, method_name in enumerate(method_order):
        ax = axes_flat[idx]
        for learning_rate in learning_rates:
            history = histories[(method_name, learning_rate)]
            x_values = [float(row[x_key]) for row in history]
            y_values = [float(row[y_key]) for row in history]
            ax.plot(x_values, y_values, marker="o", linewidth=1.5, markersize=3, label=f"lr={learning_rate:g}")

        ax.set_title(method_name)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(title)
    fig.legend(handles, labels, loc="lower center", ncol=len(learning_rates), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_random_reshuffling_epoch_diagnostic(
    *,
    histories: dict[tuple[str, float], list[dict[str, Any]]],
    method_order: list[str],
    learning_rates: list[float],
    output_path: Path,
) -> None:
    rr_methods = [
        method for method in method_order if method.startswith("random_reshuffling")
    ]
    if not rr_methods:
        return

    fig, axes = plt.subplots(1, len(rr_methods), figsize=(7 * len(rr_methods), 4), sharey=True)
    if len(rr_methods) == 1:
        axes = [axes]

    for axis, method_name in zip(axes, rr_methods):
        for learning_rate in learning_rates:
            history = histories[(method_name, learning_rate)]
            epoch_rows = [row for row in history if row["epoch"] is not None]
            if not epoch_rows:
                continue
            x_values = [int(row["epoch"]) for row in epoch_rows]
            y_values = [float(row["training_mse"]) for row in epoch_rows]
            axis.plot(x_values, y_values, marker="o", linewidth=1.5, markersize=3, label=f"lr={learning_rate:g}")

        axis.set_title(method_name)
        axis.set_xlabel("epoch (supplementary axis)")
        axis.set_ylabel("training_mse")
        axis.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Supplementary Diagnostic: Training MSE vs Epoch (Random Reshuffling)")
    fig.legend(handles, labels, loc="lower center", ncol=len(learning_rates), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _select_common_learning_rate(
    *,
    learning_rates: list[float],
    summaries_by_rate: dict[float, list[dict[str, Any]]],
) -> tuple[float | None, dict[float, dict[str, Any]]]:
    decisions: dict[float, dict[str, Any]] = {}
    accepted: list[float] = []

    for learning_rate in learning_rates:
        method_summaries = summaries_by_rate[learning_rate]
        unstable = [item for item in method_summaries if not item["stable"]]
        not_useful = [item for item in method_summaries if item["stable"] and not item["useful"]]

        if unstable:
            decisions[learning_rate] = {
                "accepted": False,
                "reason": "not stable for all methods",
                "failing_methods": [
                    {
                        "method": item["method"],
                        "reasons": item["reasons"],
                    }
                    for item in unstable
                ],
            }
            continue

        if not_useful:
            decisions[learning_rate] = {
                "accepted": False,
                "reason": "stable but lacks material reduction for at least one method",
                "failing_methods": [
                    {
                        "method": item["method"],
                        "reasons": item["reasons"],
                    }
                    for item in not_useful
                ],
            }
            continue

        decisions[learning_rate] = {
            "accepted": True,
            "reason": "stable and materially useful for all methods",
            "failing_methods": [],
        }
        accepted.append(learning_rate)

    selected = max(accepted) if accepted else None
    return selected, decisions


def run_pilot_for_grid(
    *,
    baseline_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    learning_rates: list[float],
) -> dict[str, Any]:
    generated_data_dir = Path(baseline_cfg["paths"]["generated_data_dir"])
    raw_dir = Path(baseline_cfg["paths"]["results_raw_dir"]) / "week1_gradient_methods"
    figure_dir = Path(baseline_cfg["paths"]["results_figures_dir"]) / "week1_gradient_methods"
    history_dir = raw_dir / "learning_rate_pilot_histories"

    raw_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, _ = _load_split(generated_data_dir / "baseline_train.npz")
    x_val, y_val, f_val = _load_split(generated_data_dir / "baseline_validation.npz")

    pilot_model_seed = int(experiment_cfg["experiment"]["training"]["pilot_model_seed"])
    sampling_seed_offset = int(baseline_cfg["optimisation"]["sampling_seed_offset"])
    momentum = float(baseline_cfg["optimisation"]["momentum"])
    weight_decay = float(baseline_cfg["optimisation"]["weight_decay"])

    # Use a short 10-pass budget to screen learning rates before the full baseline.
    n_train = int(x_train.shape[0])
    pilot_passes = 10
    pilot_budget_examples = pilot_passes * n_train
    evaluation_every_examples = 5000

    methods_cfg = experiment_cfg["experiment"]["methods"]
    method_items = list(methods_cfg.items())
    method_order = [name for name, _ in method_items]

    torch.manual_seed(pilot_model_seed)
    reference_model = TinyRegressionModel()
    reference_state = copy.deepcopy(reference_model.state_dict())

    sampling_seeds: dict[str, int] = {}
    histories: dict[tuple[str, float], list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    summaries_by_rate: dict[float, list[dict[str, Any]]] = {
        learning_rate: [] for learning_rate in learning_rates
    }

    for method_index, (method_key, method_cfg) in enumerate(method_items):
        method_name = str(method_cfg.get("method_name", method_key))
        sampling_method = str(method_cfg["sampling_method"])
        batch_size = int(method_cfg["batch_size"])
        # Give each method a deterministic sampling stream for reproducible pilot runs.
        sampling_seed = sampling_seed_offset + method_index
        sampling_seeds[method_name] = sampling_seed

        for learning_rate in learning_rates:
            model = TinyRegressionModel()
            model.load_state_dict(reference_state)

            optimiser = torch.optim.SGD(
                model.parameters(),
                lr=float(learning_rate),
                momentum=momentum,
                weight_decay=weight_decay,
            )

            history = train_model(
                model=model,
                optimiser=optimiser,
                loss_function=torch.nn.functional.mse_loss,
                training_data=(x_train, y_train),
                evaluation_data=(x_val, y_val, f_val),
                method=method_name,
                sampling_method=sampling_method,
                batch_size=batch_size,
                target_examples_processed=pilot_budget_examples,
                sampling_seed=sampling_seed,
                evaluation_every_examples=evaluation_every_examples,
            )

            histories[(method_name, learning_rate)] = history

            history_path = history_dir / f"{method_name}__lr_{learning_rate:g}.json"
            with history_path.open("w", encoding="utf-8") as handle:
                json.dump(_to_jsonable(history), handle, indent=2)

            run_summary = _summarise_run(
                history=history,
                method_name=method_name,
                sampling_method=sampling_method,
                learning_rate=float(learning_rate),
                material_ratio_threshold=0.90,
            )
            summaries.append(run_summary)
            summaries_by_rate[learning_rate].append(run_summary)

    _plot_metric(
        histories=histories,
        method_order=method_order,
        learning_rates=learning_rates,
        x_key="cumulative_examples_processed",
        y_key="training_mse",
        title="Pilot: Full Training MSE vs Examples Processed",
        y_label="training_mse",
        output_path=figure_dir / "pilot_training_mse_vs_examples.png",
    )
    _plot_metric(
        histories=histories,
        method_order=method_order,
        learning_rates=learning_rates,
        x_key="step",
        y_key="training_mse",
        title="Pilot: Full Training MSE vs Optimiser Steps",
        y_label="training_mse",
        output_path=figure_dir / "pilot_training_mse_vs_steps.png",
    )
    _plot_metric(
        histories=histories,
        method_order=method_order,
        learning_rates=learning_rates,
        x_key="cumulative_examples_processed",
        y_key="validation_function_mse",
        title="Pilot: Validation Function MSE vs Examples Processed",
        y_label="validation_function_mse",
        output_path=figure_dir / "pilot_validation_function_mse_vs_examples.png",
    )
    _plot_random_reshuffling_epoch_diagnostic(
        histories=histories,
        method_order=method_order,
        learning_rates=learning_rates,
        output_path=figure_dir / "pilot_random_reshuffling_epoch_diagnostic.png",
    )

    selected_rate, decisions = _select_common_learning_rate(
        learning_rates=learning_rates,
        summaries_by_rate=summaries_by_rate,
    )

    result = {
        "selected_common_learning_rate": selected_rate,
        "pilot_budget": {
            "data_equivalent_passes": pilot_passes,
            "target_examples_processed": pilot_budget_examples,
            "evaluation_every_examples": evaluation_every_examples,
        },
        "model_seed": pilot_model_seed,
        "sampling_seeds": sampling_seeds,
        "learning_rates_evaluated": learning_rates,
        "rejected_rates": {
            str(rate): detail
            for rate, detail in decisions.items()
            if not detail["accepted"]
        },
        "rate_decisions": {str(rate): detail for rate, detail in decisions.items()},
        "method_by_rate_summary": summaries,
        "commit_hash": _commit_hash(),
        "artifacts": {
            "history_dir": str(history_dir),
            "figures": [
                str(figure_dir / "pilot_training_mse_vs_examples.png"),
                str(figure_dir / "pilot_training_mse_vs_steps.png"),
                str(figure_dir / "pilot_validation_function_mse_vs_examples.png"),
                str(figure_dir / "pilot_random_reshuffling_epoch_diagnostic.png"),
            ],
        },
    }
    return result


def main() -> None:
    baseline_cfg = _load_json(Path("configs/baseline.json"))
    experiment_cfg = _load_json(Path("configs/experiments/week1_gradient_methods.json"))
    base_grid = [float(v) for v in baseline_cfg["optimisation"]["learning_rate_pilot_grid"]]

    result = run_pilot_for_grid(
        baseline_cfg=baseline_cfg,
        experiment_cfg=experiment_cfg,
        learning_rates=base_grid,
    )

    if result["selected_common_learning_rate"] is None:
        extended_rate = base_grid[0] / 3.0
        extended_grid = [extended_rate] + base_grid
        extended_result = run_pilot_for_grid(
            baseline_cfg=baseline_cfg,
            experiment_cfg=experiment_cfg,
            learning_rates=extended_grid,
        )
        extended_result["grid_extension"] = {
            "applied": True,
            "reason": "No candidate in initial grid satisfied common stability + usefulness criteria.",
            "original_grid": base_grid,
            "extended_grid": extended_grid,
        }
        result = extended_result
    else:
        result["grid_extension"] = {
            "applied": False,
            "reason": "At least one common learning rate met stability and usefulness criteria.",
            "original_grid": base_grid,
        }

    output_path = Path("results/raw/week1_gradient_methods/learning_rate_selection.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(result), handle, indent=2)

    selected = result["selected_common_learning_rate"]
    if selected is None:
        print("No common learning rate satisfied criteria; review required.")
    else:
        print(f"Selected common learning rate: {selected}")
    print(f"Selection summary: {output_path}")


if __name__ == "__main__":
    main()
