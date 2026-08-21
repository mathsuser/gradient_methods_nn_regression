from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASELINE_CONFIG_PATH = Path("configs/baseline.json")
V3_CONFIG_PATH = Path("configs/experiments/week1_gradient_methods_v3.json")
METHOD_ORDER = [
    "full_batch_gd",
    "wr_b1",
    "wr_b32",
    "wr_b256",
    "rr_b1",
    "rr_b32",
    "rr_b256",
]
RR_METHODS = ["rr_b1", "rr_b32", "rr_b256"]
REQUIRED_HISTORY_FIELDS = [
    "training_mse",
    "validation_mse",
    "validation_function_mse",
    "full_gradient_norm",
    "parameter_norm",
    "training_elapsed_seconds",
    "total_elapsed_seconds",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2)


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
    if value.__class__.__name__ in {"dtype", "device"}:
        return str(value).replace("torch.", "")
    return str(value)


def method_items(v3_cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    methods = v3_cfg["experiment"]["methods"]
    return [(method_name, methods[method_name]) for method_name in METHOD_ORDER]


def raw_dir(v3_cfg: dict[str, Any]) -> Path:
    return Path(v3_cfg["experiment"]["paths"]["raw_dir"])


def figures_dir(v3_cfg: dict[str, Any]) -> Path:
    return Path(v3_cfg["experiment"]["paths"]["figures_dir"])


def history_dir(v3_cfg: dict[str, Any]) -> Path:
    return Path(v3_cfg["experiment"]["paths"]["learning_rate_pilot_histories_dir"])


def preflight_dir(v3_cfg: dict[str, Any]) -> Path:
    return Path(v3_cfg["experiment"]["paths"]["preflight_dir"])


def selection_path(v3_cfg: dict[str, Any]) -> Path:
    return Path(v3_cfg["experiment"]["paths"]["learning_rate_selection"])


def pilot_cfg(v3_cfg: dict[str, Any]) -> dict[str, Any]:
    return v3_cfg["experiment"]["pilot"]


def pilot_learning_rates(v3_cfg: dict[str, Any]) -> list[float]:
    return [float(value) for value in pilot_cfg(v3_cfg)["learning_rates"]]


def pilot_seed_for(v3_cfg: dict[str, Any], method_name: str) -> int:
    return int(pilot_cfg(v3_cfg)["sampling_seeds"][method_name])


def future_baseline_seed_for(
    v3_cfg: dict[str, Any],
    *,
    method_name: str,
    model_seed: int,
) -> int:
    seed_map = v3_cfg["experiment"]["future_baseline"][
        "sampling_seeds_by_method_and_model_seed"
    ]
    return int(seed_map[method_name][str(int(model_seed))])


def expected_checkpoint_examples(
    *,
    target_examples_processed: int,
    evaluation_every_examples: int,
) -> list[int]:
    checkpoints = [0, *range(evaluation_every_examples, target_examples_processed + 1, evaluation_every_examples)]
    if checkpoints[-1] != target_examples_processed:
        checkpoints.append(target_examples_processed)
    return checkpoints


def validate_v3_config(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
) -> None:
    methods = v3_cfg["experiment"]["methods"]
    if list(methods) != METHOD_ORDER:
        raise ValueError("V3 methods must match the locked seven-method order")

    expected_mapping = {
        "full_batch_gd": ("full_batch", 5000),
        "wr_b1": ("single_with_replacement", 1),
        "wr_b32": ("minibatch_with_replacement", 32),
        "wr_b256": ("minibatch_with_replacement", 256),
        "rr_b1": ("random_reshuffling", 1),
        "rr_b32": ("random_reshuffling", 32),
        "rr_b256": ("random_reshuffling", 256),
    }
    for method_name, (sampling_method, batch_size) in expected_mapping.items():
        method_cfg = methods[method_name]
        if method_cfg["method_name"] != method_name:
            raise ValueError(f"{method_name} method_name must equal its config key")
        if method_cfg["sampling_method"] != sampling_method:
            raise ValueError(f"{method_name} has incorrect sampling_method")
        if int(method_cfg["batch_size"]) != batch_size:
            raise ValueError(f"{method_name} has incorrect batch_size")

    n_train = int(baseline_cfg["dataset"]["n_train"])
    training_cfg = v3_cfg["experiment"]["training"]
    if int(training_cfg["data_equivalent_passes"]) != 100:
        raise ValueError("V3 baseline horizon must be 100 data-equivalent passes")
    if int(training_cfg["target_examples_processed"]) != 100 * n_train:
        raise ValueError("V3 baseline target_examples_processed must equal 100 * n_train")
    if int(training_cfg["evaluation_every_examples"]) != n_train:
        raise ValueError("V3 baseline checkpoint cadence must equal n_train")

    p_cfg = pilot_cfg(v3_cfg)
    if int(p_cfg["model_seed"]) != 0:
        raise ValueError("V3 pilot model_seed must be 0")
    if int(p_cfg["data_equivalent_passes"]) != 10:
        raise ValueError("V3 pilot horizon must be 10 data-equivalent passes")
    if int(p_cfg["target_examples_processed"]) != 10 * n_train:
        raise ValueError("V3 pilot target_examples_processed must equal 10 * n_train")
    if int(p_cfg["evaluation_every_examples"]) != n_train:
        raise ValueError("V3 pilot checkpoint cadence must equal n_train")
    if pilot_learning_rates(v3_cfg) != [0.001, 0.003, 0.01, 0.03, 0.1]:
        raise ValueError("V3 pilot grid changed")
    if float(p_cfg["material_reduction_ratio"]) != 0.9:
        raise ValueError("V3 material_reduction_ratio must be 0.9")

    pilot_seeds = p_cfg["sampling_seeds"]
    if set(pilot_seeds) != set(METHOD_ORDER):
        raise ValueError("V3 pilot seed map must contain exactly the seven methods")
    if len(set(int(seed) for seed in pilot_seeds.values())) != len(METHOD_ORDER):
        raise ValueError("V3 pilot seeds must be unique")

    model_seeds = [int(seed) for seed in baseline_cfg["optimisation"]["model_seeds"]]
    future_cfg = v3_cfg["experiment"]["future_baseline"]
    if [int(seed) for seed in future_cfg["model_seeds"]] != model_seeds:
        raise ValueError("V3 future-baseline model seeds must match baseline.json")
    for method_name in METHOD_ORDER:
        method_seed_map = future_cfg["sampling_seeds_by_method_and_model_seed"][method_name]
        if set(method_seed_map) != {str(seed) for seed in model_seeds}:
            raise ValueError(f"{method_name} future seed map does not cover all model seeds")

    paths = v3_cfg["experiment"]["paths"]
    forbidden_prefixes = [
        "results/raw/week1_gradient_methods",
        "results/figures/week1_gradient_methods",
        "results/raw/sampling_law_branch",
        "results/figures/sampling_law_branch",
    ]
    for path_value in paths.values():
        path_text = str(path_value)
        if path_text.startswith("results/raw/week1_gradient_methods/"):
            raise ValueError("V3 raw output path collides with V2 raw namespace")
        if path_text.startswith("results/figures/week1_gradient_methods/"):
            raise ValueError("V3 figure output path collides with V2 figure namespace")
        for prefix in forbidden_prefixes[2:]:
            if path_text.startswith(prefix):
                raise ValueError("V3 output path collides with sampling-law branch namespace")


def build_pilot_run_specs(
    *,
    v3_cfg: dict[str, Any],
    learning_rates: list[float] | None = None,
    target_examples_processed: int | None = None,
    evaluation_every_examples: int | None = None,
) -> list[dict[str, Any]]:
    p_cfg = pilot_cfg(v3_cfg)
    selected_learning_rates = pilot_learning_rates(v3_cfg) if learning_rates is None else learning_rates
    target_examples = (
        int(p_cfg["target_examples_processed"])
        if target_examples_processed is None
        else int(target_examples_processed)
    )
    evaluation_every = (
        int(p_cfg["evaluation_every_examples"])
        if evaluation_every_examples is None
        else int(evaluation_every_examples)
    )
    specs: list[dict[str, Any]] = []
    for method_name, method_cfg in method_items(v3_cfg):
        for learning_rate in selected_learning_rates:
            specs.append(
                {
                    "method_name": method_name,
                    "sampling_method": str(method_cfg["sampling_method"]),
                    "batch_size": int(method_cfg["batch_size"]),
                    "learning_rate": float(learning_rate),
                    "model_seed": int(p_cfg["model_seed"]),
                    "sampling_seed": pilot_seed_for(v3_cfg, method_name),
                    "target_examples_processed": target_examples,
                    "evaluation_every_examples": evaluation_every,
                }
            )
    return specs


def build_preflight_run_specs(v3_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    p_cfg = pilot_cfg(v3_cfg)
    return build_pilot_run_specs(
        v3_cfg=v3_cfg,
        learning_rates=[float(p_cfg["preflight_learning_rate"])],
        target_examples_processed=int(p_cfg["preflight_target_examples_processed"]),
        evaluation_every_examples=int(p_cfg["preflight_evaluation_every_examples"]),
    )


def preflight_plan(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
) -> dict[str, Any]:
    validate_v3_config(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg)
    p_cfg = pilot_cfg(v3_cfg)
    n_train = int(baseline_cfg["dataset"]["n_train"])
    return {
        "experiment_name": v3_cfg["experiment"]["name"],
        "method_definitions": {
            method_name: {
                "sampling_method": method_cfg["sampling_method"],
                "batch_size": int(method_cfg["batch_size"]),
            }
            for method_name, method_cfg in method_items(v3_cfg)
        },
        "learning_rate_grid": pilot_learning_rates(v3_cfg),
        "expected_full_pilot_run_count": len(METHOD_ORDER) * len(pilot_learning_rates(v3_cfg)),
        "pilot_model_seed": int(p_cfg["model_seed"]),
        "pilot_sampling_seeds": dict(p_cfg["sampling_seeds"]),
        "pilot_budget": {
            "target_examples_processed": int(p_cfg["target_examples_processed"]),
            "data_equivalent_passes": int(p_cfg["target_examples_processed"]) / n_train,
            "evaluation_every_examples": int(p_cfg["evaluation_every_examples"]),
            "checkpoint_examples": expected_checkpoint_examples(
                target_examples_processed=int(p_cfg["target_examples_processed"]),
                evaluation_every_examples=int(p_cfg["evaluation_every_examples"]),
            ),
        },
        "preflight_budget": {
            "learning_rate": float(p_cfg["preflight_learning_rate"]),
            "target_examples_processed": int(p_cfg["preflight_target_examples_processed"]),
            "data_equivalent_passes": int(p_cfg["preflight_target_examples_processed"]) / n_train,
            "checkpoint_examples": expected_checkpoint_examples(
                target_examples_processed=int(p_cfg["preflight_target_examples_processed"]),
                evaluation_every_examples=int(p_cfg["preflight_evaluation_every_examples"]),
            ),
        },
        "output_paths": dict(v3_cfg["experiment"]["paths"]),
    }


def history_is_finite(history: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for record in history:
        for field in REQUIRED_HISTORY_FIELDS:
            value = float(record[field])
            if not math.isfinite(value):
                return False, [f"{field} became non-finite"]
        for optional_field in ["update_gradient_norm", "batch_loss"]:
            value = record.get(optional_field)
            if value is not None and not math.isfinite(float(value)):
                return False, [f"{optional_field} became non-finite"]
    return True, reasons


def is_explosive_without_recovery(history: list[dict[str, Any]]) -> bool:
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
    return final_value > 5.0 * max(min_value, 1.0e-12)


def material_reduction(
    history: list[dict[str, Any]],
    *,
    ratio_threshold: float,
) -> bool:
    initial = float(history[0]["training_mse"])
    final = float(history[-1]["training_mse"])
    return final <= ratio_threshold * initial


def summarise_history(
    *,
    history: list[dict[str, Any]],
    spec: dict[str, Any],
    material_ratio_threshold: float,
) -> dict[str, Any]:
    finite_ok, finite_reasons = history_is_finite(history)
    reasons: list[str] = []
    if not finite_ok:
        reasons.extend(finite_reasons)
    if finite_ok and is_explosive_without_recovery(history):
        reasons.append("trajectory became explosive and did not recover")
    stable = len(reasons) == 0
    useful = stable and material_reduction(
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
        if row.get("update_gradient_norm") is not None
    ]
    return {
        "method": spec["method_name"],
        "sampling_method": spec["sampling_method"],
        "batch_size": int(spec["batch_size"]),
        "learning_rate": float(spec["learning_rate"]),
        "sampling_seed": int(spec["sampling_seed"]),
        "stable": stable,
        "useful": useful,
        "accepted": stable and useful,
        "reasons": reasons,
        "initial_training_mse": float(history[0]["training_mse"]),
        "final_training_mse": float(history[-1]["training_mse"]),
        "final_validation_mse": float(history[-1]["validation_mse"]),
        "final_validation_function_mse": float(history[-1]["validation_function_mse"]),
        "max_update_gradient_norm": max(update_gradients) if update_gradients else None,
        "max_full_gradient_norm": max(float(row["full_gradient_norm"]) for row in history),
        "max_parameter_norm": max(float(row["parameter_norm"]) for row in history),
        "actual_steps": int(history[-1]["step"]),
        "actual_examples_processed": int(history[-1]["cumulative_examples_processed"]),
        "data_equivalent_passes": float(history[-1]["data_equivalent_passes"]),
    }


def selection_decisions(
    *,
    learning_rates: list[float],
    summaries_by_rate: dict[float, list[dict[str, Any]]],
) -> dict[float, dict[str, Any]]:
    decisions: dict[float, dict[str, Any]] = {}
    for learning_rate in learning_rates:
        summaries = summaries_by_rate[learning_rate]
        rejected = [summary for summary in summaries if not summary["accepted"]]
        decisions[learning_rate] = {
            "accepted": len(rejected) == 0,
            "method_summaries": summaries,
            "rejection_reasons": {
                summary["method"]: summary["reasons"]
                for summary in rejected
            },
        }
    return decisions


def select_common_learning_rate(
    *,
    learning_rates: list[float],
    summaries_by_rate: dict[float, list[dict[str, Any]]],
) -> tuple[float | None, dict[float, dict[str, Any]]]:
    decisions = selection_decisions(
        learning_rates=learning_rates,
        summaries_by_rate=summaries_by_rate,
    )
    accepted = [
        learning_rate
        for learning_rate in learning_rates
        if decisions[learning_rate]["accepted"]
    ]
    return (max(accepted) if accepted else None), decisions


def extension_learning_rate(learning_rates: list[float], *, factor: float) -> float:
    if factor <= 1:
        raise ValueError("grid extension factor must be greater than 1")
    return min(learning_rates) / factor


def commit_hash() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def worktree_status() -> str:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        return "clean" if proc.stdout.strip() == "" else "dirty"
    except Exception:
        return "unknown"


def state_checksum(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def format_progress_start(run_index: int, total_runs: int, spec: dict[str, Any]) -> str:
    return (
        f"[{run_index}/{total_runs}] START {spec['method_name']} "
        f"lr={spec['learning_rate']:g} seed={spec['sampling_seed']}"
    )


def format_progress_done(
    *,
    run_index: int,
    total_runs: int,
    spec: dict[str, Any],
    run_seconds: float,
    total_elapsed_seconds: float,
    eta_seconds: float,
) -> str:
    return (
        f"[{run_index}/{total_runs}] DONE {spec['method_name']} "
        f"lr={spec['learning_rate']:g}\n"
        f"run={run_seconds:.1f}s\n"
        f"total_elapsed={total_elapsed_seconds / 60:.1f}min\n"
        f"ETA={eta_seconds / 60:.1f}min"
    )


def torch_dtype_from_name(dtype_name: str, *, torch_module: Any) -> Any:
    dtype_by_name = {
        "float32": torch_module.float32,
        "float64": torch_module.float64,
    }
    try:
        return dtype_by_name[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported configured dtype: {dtype_name}") from exc


def load_split(
    path: Path,
    *,
    expected_dtype: np.dtype,
    torch_dtype: Any,
    device: Any,
    torch_module: Any,
) -> tuple[Any, Any, Any]:
    with np.load(path) as data:
        x_np = data["x"]
        y_np = data["y"].reshape(-1, 1)
        f_true_np = data["f_true"].reshape(-1, 1)
    for name, array in [("x", x_np), ("y", y_np), ("f_true", f_true_np)]:
        if array.dtype != expected_dtype:
            raise ValueError(f"{path}: {name} dtype is {array.dtype}, expected {expected_dtype}")
    x = torch_module.as_tensor(x_np, dtype=torch_dtype, device=device)
    y = torch_module.as_tensor(y_np, dtype=torch_dtype, device=device)
    f_true = torch_module.as_tensor(f_true_np, dtype=torch_dtype, device=device)
    return x, y, f_true


def import_training_stack() -> tuple[Any, Any, Any]:
    import torch

    from gradient_methods_nn_regression.model import TinyRegressionModel
    from gradient_methods_nn_regression.training import train_model

    return torch, TinyRegressionModel, train_model


def run_specs(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    specs: list[dict[str, Any]],
    output_root: Path,
    mode: str,
    write_histories: bool,
) -> dict[str, Any]:
    validate_v3_config(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg)
    torch, TinyRegressionModel, train_model = import_training_stack()

    ds_cfg = baseline_cfg["dataset"]
    opt_cfg = baseline_cfg["optimisation"]
    p_cfg = pilot_cfg(v3_cfg)
    configured_dtype = np.dtype(ds_cfg["dtype"])
    torch_dtype = torch_dtype_from_name(str(configured_dtype), torch_module=torch)
    device = torch.device("cpu")
    generated_data_dir = Path(baseline_cfg["paths"]["generated_data_dir"])

    x_train, y_train, _ = load_split(
        generated_data_dir / p_cfg["data_splits"]["training"],
        expected_dtype=configured_dtype,
        torch_dtype=torch_dtype,
        device=device,
        torch_module=torch,
    )
    x_val, y_val, f_val = load_split(
        generated_data_dir / p_cfg["data_splits"]["evaluation"],
        expected_dtype=configured_dtype,
        torch_dtype=torch_dtype,
        device=device,
        torch_module=torch,
    )

    torch.manual_seed(int(p_cfg["model_seed"]))
    reference_model = TinyRegressionModel().to(device=device, dtype=torch_dtype)
    reference_state = copy.deepcopy(reference_model.state_dict())
    initial_checksum = state_checksum(reference_state)
    momentum = float(opt_cfg["momentum"])
    weight_decay = float(opt_cfg["weight_decay"])
    material_ratio = float(p_cfg["material_reduction_ratio"])
    output_root.mkdir(parents=True, exist_ok=True)

    summaries_by_rate: dict[float, list[dict[str, Any]]] = {}
    history_paths: dict[str, str] = {}
    run_records: list[dict[str, Any]] = []
    total_runs = len(specs)
    completed_seconds: list[float] = []
    run_start_all = time.perf_counter()

    for run_index, spec in enumerate(specs, start=1):
        print(format_progress_start(run_index, total_runs, spec), flush=True)
        run_start = time.perf_counter()
        model = TinyRegressionModel().to(device=device, dtype=torch_dtype)
        model.load_state_dict(reference_state)
        loaded_checksum = state_checksum(model.state_dict())
        if loaded_checksum != initial_checksum:
            raise RuntimeError("Initial-state checksum mismatch")
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=float(spec["learning_rate"]),
            momentum=momentum,
            weight_decay=weight_decay,
        )
        history = train_model(
            model=model,
            optimiser=optimiser,
            loss_function=torch.nn.functional.mse_loss,
            training_data=(x_train, y_train),
            evaluation_data=(x_val, y_val, f_val),
            method=spec["method_name"],
            sampling_method=spec["sampling_method"],
            batch_size=int(spec["batch_size"]),
            target_examples_processed=int(spec["target_examples_processed"]),
            sampling_seed=int(spec["sampling_seed"]),
            evaluation_every_examples=int(spec["evaluation_every_examples"]),
        )
        summary = summarise_history(
            history=history,
            spec=spec,
            material_ratio_threshold=material_ratio,
        )
        learning_rate = float(spec["learning_rate"])
        summaries_by_rate.setdefault(learning_rate, []).append(summary)

        history_path = (
            output_root
            / f"{spec['method_name']}__lr_{learning_rate:g}__history.json"
        )
        if write_histories:
            write_json(
                {
                    "spec": spec,
                    "initial_state_checksum": initial_checksum,
                    "loaded_initial_state_checksum": loaded_checksum,
                    "history": history,
                    "summary": summary,
                },
                history_path,
            )
            history_paths[f"{spec['method_name']}__{learning_rate:g}"] = str(history_path)

        run_seconds = time.perf_counter() - run_start
        completed_seconds.append(run_seconds)
        elapsed = time.perf_counter() - run_start_all
        remaining = total_runs - run_index
        eta = remaining * (sum(completed_seconds) / len(completed_seconds))
        print(
            format_progress_done(
                run_index=run_index,
                total_runs=total_runs,
                spec=spec,
                run_seconds=run_seconds,
                total_elapsed_seconds=elapsed,
                eta_seconds=eta,
            ),
            flush=True,
        )
        run_records.append(
            {
                "spec": spec,
                "summary": summary,
                "history_json": history_path if write_histories else None,
            }
        )

    return {
        "mode": mode,
        "run_count": total_runs,
        "initial_state_checksum": initial_checksum,
        "torch_dtype": torch_dtype,
        "device": device,
        "summaries_by_rate": summaries_by_rate,
        "run_records": run_records,
        "history_paths": history_paths,
        "elapsed_seconds": time.perf_counter() - run_start_all,
    }


def plot_metric_grid(
    *,
    histories_by_key: dict[tuple[str, float], list[dict[str, Any]]],
    learning_rates: list[float],
    x_key: str,
    y_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n_methods = len(METHOD_ORDER)
    n_cols = 4
    n_rows = math.ceil(n_methods / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8.5), sharey=False)
    axes_flat = np.asarray(axes).reshape(-1)
    for index, method_name in enumerate(METHOD_ORDER):
        axis = axes_flat[index]
        for learning_rate in learning_rates:
            history = histories_by_key[(method_name, learning_rate)]
            axis.plot(
                [float(row[x_key]) for row in history],
                [float(row[y_key]) for row in history],
                marker="o",
                linewidth=1.4,
                markersize=3,
                label=f"lr={learning_rate:g}",
            )
        axis.set_title(method_name)
        axis.set_xlabel(x_key)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
    for index in range(n_methods, len(axes_flat)):
        axes_flat[index].set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(title)
    fig.legend(handles, labels, loc="lower center", ncol=len(learning_rates), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rr_epoch_diagnostic(
    *,
    histories_by_key: dict[tuple[str, float], list[dict[str, Any]]],
    learning_rates: list[float],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(RR_METHODS), figsize=(18, 4.5), sharey=True)
    for axis, method_name in zip(axes, RR_METHODS):
        for learning_rate in learning_rates:
            history = histories_by_key[(method_name, learning_rate)]
            x_values = [
                float(row["epoch"])
                for row in history
                if row["epoch"] is not None
            ]
            y_values = [
                float(row["training_mse"])
                for row in history
                if row["epoch"] is not None
            ]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.4,
                markersize=3,
                label=f"lr={learning_rate:g}",
            )
        axis.set_title(method_name)
        axis.set_xlabel("completed RR epochs")
        axis.set_ylabel("training MSE")
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(learning_rates), frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_histories_for_figures(history_paths: dict[str, str]) -> dict[tuple[str, float], list[dict[str, Any]]]:
    histories: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for key, path_text in history_paths.items():
        method_name, lr_text = key.rsplit("__", maxsplit=1)
        histories[(method_name, float(lr_text))] = load_json(Path(path_text))["history"]
    return histories


def run_preflight(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
) -> dict[str, Any]:
    specs = build_preflight_run_specs(v3_cfg)
    result = run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        specs=specs,
        output_root=preflight_dir(v3_cfg),
        mode="preflight",
        write_histories=True,
    )
    expected_checkpoints = expected_checkpoint_examples(
        target_examples_processed=int(pilot_cfg(v3_cfg)["preflight_target_examples_processed"]),
        evaluation_every_examples=int(pilot_cfg(v3_cfg)["preflight_evaluation_every_examples"]),
    )
    failures: list[str] = []
    for record in result["run_records"]:
        spec = record["spec"]
        summary = record["summary"]
        if summary["actual_examples_processed"] != spec["target_examples_processed"]:
            failures.append(f"{spec['method_name']}: incorrect final examples")
        if not math.isclose(summary["data_equivalent_passes"], 1.0, abs_tol=1e-12):
            failures.append(f"{spec['method_name']}: incorrect preflight DEP")
        if not summary["stable"]:
            failures.append(f"{spec['method_name']}: non-finite or unstable metrics")
    manifest = {
        "ok": len(failures) == 0,
        "failures": failures,
        "mode": "preflight",
        "expected_checkpoints": expected_checkpoints,
        "method_count": len(METHOD_ORDER),
        "run_count": result["run_count"],
        "initial_state_checksum": result["initial_state_checksum"],
        "dtype": result["torch_dtype"],
        "device": result["device"],
        "summaries_by_rate": result["summaries_by_rate"],
        "history_paths": result["history_paths"],
    }
    write_json(manifest, preflight_dir(v3_cfg) / "preflight_manifest.json")
    if failures:
        raise RuntimeError("V3 preflight failed: " + "; ".join(failures))
    return manifest


def run_full_pilot(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
) -> dict[str, Any]:
    learning_rates = pilot_learning_rates(v3_cfg)
    specs = build_pilot_run_specs(v3_cfg=v3_cfg, learning_rates=learning_rates)
    result = run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        specs=specs,
        output_root=history_dir(v3_cfg),
        mode="full",
        write_histories=True,
    )
    summaries_by_rate = result["summaries_by_rate"]
    selected, decisions = select_common_learning_rate(
        learning_rates=learning_rates,
        summaries_by_rate=summaries_by_rate,
    )
    extended_learning_rates: list[float] = []
    if selected is None:
        extra_rate = extension_learning_rate(
            learning_rates,
            factor=float(pilot_cfg(v3_cfg)["grid_extension_factor"]),
        )
        extended_learning_rates.append(extra_rate)
        extra_specs = build_pilot_run_specs(v3_cfg=v3_cfg, learning_rates=[extra_rate])
        extra_result = run_specs(
            baseline_cfg=baseline_cfg,
            v3_cfg=v3_cfg,
            specs=extra_specs,
            output_root=history_dir(v3_cfg),
            mode="full_extension",
            write_histories=True,
        )
        result["history_paths"].update(extra_result["history_paths"])
        result["run_records"].extend(extra_result["run_records"])
        summaries_by_rate[extra_rate] = extra_result["summaries_by_rate"][extra_rate]
        learning_rates = [extra_rate, *learning_rates]
        selected, decisions = select_common_learning_rate(
            learning_rates=learning_rates,
            summaries_by_rate=summaries_by_rate,
        )

    histories = load_histories_for_figures(result["history_paths"])
    figs_dir = figures_dir(v3_cfg)
    plot_metric_grid(
        histories_by_key=histories,
        learning_rates=learning_rates,
        x_key="cumulative_examples_processed",
        y_key="training_mse",
        y_label="training MSE",
        title="V3 pilot training MSE vs examples processed",
        output_path=figs_dir / "pilot_training_mse_vs_examples.png",
    )
    plot_metric_grid(
        histories_by_key=histories,
        learning_rates=learning_rates,
        x_key="step",
        y_key="training_mse",
        y_label="training MSE",
        title="V3 pilot training MSE vs optimiser steps",
        output_path=figs_dir / "pilot_training_mse_vs_steps.png",
    )
    plot_metric_grid(
        histories_by_key=histories,
        learning_rates=learning_rates,
        x_key="cumulative_examples_processed",
        y_key="validation_function_mse",
        y_label="validation function MSE",
        title="V3 pilot validation function MSE vs examples processed",
        output_path=figs_dir / "pilot_validation_function_mse_vs_examples.png",
    )
    plot_rr_epoch_diagnostic(
        histories_by_key=histories,
        learning_rates=learning_rates,
        output_path=figs_dir / "pilot_rr_epoch_diagnostic.png",
    )

    manifest = {
        "v3_config_path": V3_CONFIG_PATH,
        "experiment_name": v3_cfg["experiment"]["name"],
        "source_git_commit_hash": commit_hash(),
        "worktree_status": worktree_status(),
        "human_approved": False,
        "pilot_model_seed": int(pilot_cfg(v3_cfg)["model_seed"]),
        "explicit_pilot_sampling_seeds": dict(pilot_cfg(v3_cfg)["sampling_seeds"]),
        "learning_rates_evaluated": learning_rates,
        "original_learning_rate_grid": pilot_learning_rates(v3_cfg),
        "extended_learning_rates": extended_learning_rates,
        "pilot_budget": {
            "target_examples_processed": int(pilot_cfg(v3_cfg)["target_examples_processed"]),
            "data_equivalent_passes": int(pilot_cfg(v3_cfg)["data_equivalent_passes"]),
            "evaluation_every_examples": int(pilot_cfg(v3_cfg)["evaluation_every_examples"]),
        },
        "runtime": {
            "dtype": result["torch_dtype"],
            "device": result["device"],
            "python_version": platform.python_version(),
        },
        "method_definitions": {
            name: cfg for name, cfg in method_items(v3_cfg)
        },
        "per_rate_decisions": decisions,
        "selected_common_learning_rate": selected,
        "selection_note": "Calculated by script; requires human review before downstream baseline use.",
        "history_paths": result["history_paths"],
        "figures": {
            "training_mse_vs_examples": figs_dir / "pilot_training_mse_vs_examples.png",
            "training_mse_vs_steps": figs_dir / "pilot_training_mse_vs_steps.png",
            "validation_function_mse_vs_examples": figs_dir / "pilot_validation_function_mse_vs_examples.png",
            "rr_epoch_diagnostic": figs_dir / "pilot_rr_epoch_diagnostic.png",
        },
    }
    write_json(manifest, selection_path(v3_cfg))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    selected_modes = sum([args.plan_only, args.preflight, args.full])
    if selected_modes != 1:
        raise SystemExit("Choose exactly one of --plan-only, --preflight, or --full.")

    baseline_cfg = load_json(BASELINE_CONFIG_PATH)
    v3_cfg = load_json(V3_CONFIG_PATH)
    if args.plan_only:
        print(json.dumps(to_jsonable(preflight_plan(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg)), indent=2))
        return
    if args.preflight:
        print(json.dumps(to_jsonable(run_preflight(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg)), indent=2))
        return
    if args.full:
        print(json.dumps(to_jsonable(run_full_pilot(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg)), indent=2))
        return


if __name__ == "__main__":
    main()
