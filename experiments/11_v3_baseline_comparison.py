from __future__ import annotations

import argparse
import copy
import csv
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
EXPECTED_APPROVED_LEARNING_RATE = 0.03
METHOD_ORDER = [
    "full_batch_gd",
    "wr_b1",
    "wr_b32",
    "wr_b256",
    "rr_b1",
    "rr_b32",
    "rr_b256",
]
EXPECTED_METHOD_MAPPING = {
    "full_batch_gd": ("full_batch", 5000),
    "wr_b1": ("single_with_replacement", 1),
    "wr_b32": ("minibatch_with_replacement", 32),
    "wr_b256": ("minibatch_with_replacement", 256),
    "rr_b1": ("random_reshuffling", 1),
    "rr_b32": ("random_reshuffling", 32),
    "rr_b256": ("random_reshuffling", 256),
}
HISTORY_FIELDNAMES = [
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
FINITE_HISTORY_FIELDS = [
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


def write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field) for field in HISTORY_FIELDNAMES})


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def method_items(v3_cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    methods = v3_cfg["experiment"]["methods"]
    return [(method_name, methods[method_name]) for method_name in METHOD_ORDER]


def raw_dir(v3_cfg: dict[str, Any]) -> Path:
    return Path(v3_cfg["experiment"]["paths"]["raw_dir"])


def baseline_output_root(v3_cfg: dict[str, Any]) -> Path:
    return raw_dir(v3_cfg) / "baseline_comparison_runs"


def preflight_output_root(v3_cfg: dict[str, Any]) -> Path:
    return raw_dir(v3_cfg) / "baseline_preflight"


def lr_decision_artifact_path(v3_cfg: dict[str, Any]) -> Path:
    return Path(v3_cfg["experiment"]["paths"]["learning_rate_selection"])


def data_paths(baseline_cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(baseline_cfg["paths"]["generated_data_dir"])
    return {
        "train": root / "baseline_train.npz",
        "validation": root / "baseline_validation.npz",
        "test": root / "baseline_test.npz",
    }


def expected_checkpoint_examples(
    *,
    target_examples_processed: int,
    evaluation_every_examples: int,
) -> list[int]:
    checkpoints = [
        0,
        *range(
            evaluation_every_examples,
            target_examples_processed + 1,
            evaluation_every_examples,
        ),
    ]
    if checkpoints[-1] != target_examples_processed:
        checkpoints.append(target_examples_processed)
    return checkpoints


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


def validate_v3_config(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
) -> None:
    methods = v3_cfg["experiment"]["methods"]
    if list(methods) != METHOD_ORDER:
        raise ValueError("V3 methods must match the locked seven-method order")
    for method_name, (sampling_method, batch_size) in EXPECTED_METHOD_MAPPING.items():
        method_cfg = methods[method_name]
        if method_cfg["method_name"] != method_name:
            raise ValueError(f"{method_name} method_name must equal its config key")
        if method_cfg["sampling_method"] != sampling_method:
            raise ValueError(f"{method_name} has incorrect sampling_method")
        if int(method_cfg["batch_size"]) != batch_size:
            raise ValueError(f"{method_name} has incorrect batch_size")

    ds_cfg = baseline_cfg["dataset"]
    model_cfg = baseline_cfg["model"]
    opt_cfg = baseline_cfg["optimisation"]
    training_cfg = v3_cfg["experiment"]["training"]
    n_train = int(ds_cfg["n_train"])
    if int(ds_cfg["n_validation"]) != 2000 or int(ds_cfg["n_test"]) != 20000:
        raise ValueError("Baseline validation/test sizes changed")
    if str(ds_cfg["dtype"]) != "float64":
        raise ValueError("Definitive baseline dtype must be float64")
    if [int(model_cfg["input_dim"]), int(model_cfg["hidden_dim"]), int(model_cfg["output_dim"])] != [6, 16, 1]:
        raise ValueError("Definitive baseline model must be 6 -> 16 -> 1")
    if str(model_cfg["activation"]) != "tanh":
        raise ValueError("Definitive baseline activation must be tanh")
    if str(opt_cfg["optimizer"]) != "SGD":
        raise ValueError("Definitive baseline optimiser must be SGD")
    if float(opt_cfg["momentum"]) != 0.0 or float(opt_cfg["weight_decay"]) != 0.0:
        raise ValueError("Definitive baseline SGD momentum and weight_decay must be zero")

    if int(training_cfg["data_equivalent_passes"]) != 100:
        raise ValueError("V3 baseline horizon must be 100 data-equivalent passes")
    if int(training_cfg["target_examples_processed"]) != 500000:
        raise ValueError("V3 baseline target must be 500000 examples")
    if int(training_cfg["target_examples_processed"]) != 100 * n_train:
        raise ValueError("V3 baseline target_examples_processed must equal 100 * n_train")
    if int(training_cfg["evaluation_every_examples"]) != 5000:
        raise ValueError("V3 checkpoint cadence must be 5000 examples")

    expected_model_seeds = [0, 1, 2, 3, 4]
    future_cfg = v3_cfg["experiment"]["future_baseline"]
    if [int(seed) for seed in future_cfg["model_seeds"]] != expected_model_seeds:
        raise ValueError("V3 future baseline model seeds must be [0, 1, 2, 3, 4]")
    if [int(seed) for seed in opt_cfg["model_seeds"]] != expected_model_seeds:
        raise ValueError("baseline.json model seeds must be [0, 1, 2, 3, 4]")
    for method_name in METHOD_ORDER:
        method_seed_map = future_cfg["sampling_seeds_by_method_and_model_seed"][method_name]
        if set(method_seed_map) != {str(seed) for seed in expected_model_seeds}:
            raise ValueError(f"{method_name} future seed map does not cover all model seeds")


def validate_approved_lr_decision(path: Path) -> dict[str, Any]:
    decision = load_json(path)
    if decision.get("human_approved") is not True:
        raise ValueError(f"{path} is not human approved")
    selected = decision.get("selected_common_learning_rate")
    if selected is None or not math.isfinite(float(selected)):
        raise ValueError(f"{path} does not contain a finite selected_common_learning_rate")
    selected_lr = float(selected)
    if selected_lr != EXPECTED_APPROVED_LEARNING_RATE:
        raise ValueError(
            f"Approved learning rate must be {EXPECTED_APPROVED_LEARNING_RATE:g}; got {selected_lr:g}"
        )
    return {
        "path": path,
        "sha256": file_sha256(path),
        "selected_common_learning_rate": selected_lr,
        "human_approved": True,
        "source_git_commit_hash": decision.get("source_git_commit_hash"),
    }


def build_baseline_run_specs(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    learning_rate: float,
    target_examples_processed: int | None = None,
    evaluation_every_examples: int | None = None,
    model_seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    validate_v3_config(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg)
    training_cfg = v3_cfg["experiment"]["training"]
    selected_model_seeds = (
        [int(seed) for seed in v3_cfg["experiment"]["future_baseline"]["model_seeds"]]
        if model_seeds is None
        else [int(seed) for seed in model_seeds]
    )
    target_examples = (
        int(training_cfg["target_examples_processed"])
        if target_examples_processed is None
        else int(target_examples_processed)
    )
    evaluation_every = (
        int(training_cfg["evaluation_every_examples"])
        if evaluation_every_examples is None
        else int(evaluation_every_examples)
    )
    specs: list[dict[str, Any]] = []
    for model_seed in selected_model_seeds:
        for method_name, method_cfg in method_items(v3_cfg):
            specs.append(
                {
                    "method_name": method_name,
                    "sampling_method": str(method_cfg["sampling_method"]),
                    "batch_size": int(method_cfg["batch_size"]),
                    "model_seed": int(model_seed),
                    "sampling_seed": future_baseline_seed_for(
                        v3_cfg,
                        method_name=method_name,
                        model_seed=model_seed,
                    ),
                    "learning_rate": float(learning_rate),
                    "target_examples_processed": target_examples,
                    "evaluation_every_examples": evaluation_every,
                }
            )
    return specs


def build_preflight_run_specs(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    learning_rate: float,
) -> list[dict[str, Any]]:
    n_train = int(baseline_cfg["dataset"]["n_train"])
    return build_baseline_run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        learning_rate=learning_rate,
        target_examples_processed=n_train,
        evaluation_every_examples=n_train,
        model_seeds=[0],
    )


def sampling_seed_table(v3_cfg: dict[str, Any]) -> dict[str, dict[str, int]]:
    model_seeds = [int(seed) for seed in v3_cfg["experiment"]["future_baseline"]["model_seeds"]]
    return {
        method_name: {
            str(model_seed): future_baseline_seed_for(
                v3_cfg,
                method_name=method_name,
                model_seed=model_seed,
            )
            for model_seed in model_seeds
        }
        for method_name in METHOD_ORDER
    }


def plan(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    lr_decision: dict[str, Any],
) -> dict[str, Any]:
    specs = build_baseline_run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        learning_rate=float(lr_decision["selected_common_learning_rate"]),
    )
    training_cfg = v3_cfg["experiment"]["training"]
    n_train = int(baseline_cfg["dataset"]["n_train"])
    return {
        "experiment_name": v3_cfg["experiment"]["name"],
        "mode": "plan-only",
        "human_approved_learning_rate": lr_decision["human_approved"],
        "learning_rate": lr_decision["selected_common_learning_rate"],
        "learning_rate_decision_artifact": lr_decision["path"],
        "learning_rate_decision_artifact_sha256": lr_decision["sha256"],
        "learning_rate_decision_source_git_commit_hash": lr_decision[
            "source_git_commit_hash"
        ],
        "method_definitions": {
            method_name: {
                "sampling_method": method_cfg["sampling_method"],
                "batch_size": int(method_cfg["batch_size"]),
            }
            for method_name, method_cfg in method_items(v3_cfg)
        },
        "model_seeds": [int(seed) for seed in v3_cfg["experiment"]["future_baseline"]["model_seeds"]],
        "expected_run_count": len(METHOD_ORDER) * 5,
        "planned_run_count": len(specs),
        "run_specs": specs,
        "explicit_sampling_seed_table": sampling_seed_table(v3_cfg),
        "target_examples_processed": int(training_cfg["target_examples_processed"]),
        "data_equivalent_passes": int(training_cfg["target_examples_processed"]) / n_train,
        "evaluation_every_examples": int(training_cfg["evaluation_every_examples"]),
        "checkpoint_examples": expected_checkpoint_examples(
            target_examples_processed=int(training_cfg["target_examples_processed"]),
            evaluation_every_examples=int(training_cfg["evaluation_every_examples"]),
        ),
        "output_namespace": baseline_output_root(v3_cfg),
        "preflight_namespace": preflight_output_root(v3_cfg),
    }


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


def import_training_stack() -> tuple[Any, Any, Any, Any, Any]:
    import torch

    from gradient_methods_nn_regression.metrics import (
        function_estimation_mse,
        noisy_prediction_mse,
        parameter_norm,
    )
    from gradient_methods_nn_regression.model import TinyRegressionModel
    from gradient_methods_nn_regression.training import train_model

    return torch, TinyRegressionModel, train_model, {
        "function_estimation_mse": function_estimation_mse,
        "noisy_prediction_mse": noisy_prediction_mse,
        "parameter_norm": parameter_norm,
    }


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


def prepare_reference_states(
    *,
    model_seeds: list[int],
    TinyRegressionModel: Any,
    torch_module: Any,
    torch_dtype: Any,
    device: Any,
) -> dict[int, dict[str, Any]]:
    references: dict[int, dict[str, Any]] = {}
    for model_seed in model_seeds:
        torch_module.manual_seed(int(model_seed))
        reference_model = TinyRegressionModel().to(device=device, dtype=torch_dtype)
        reference_state = copy.deepcopy(reference_model.state_dict())
        references[int(model_seed)] = {
            "state_dict": reference_state,
            "checksum": state_checksum(reference_state),
        }
    return references


def final_split_metrics(
    *,
    model: Any,
    x: Any,
    y: Any,
    f_true: Any,
    metrics: dict[str, Any],
    torch_module: Any,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    with torch_module.no_grad():
        predictions = model(x)
        prediction_mse = metrics["noisy_prediction_mse"](predictions, y).item()
        function_mse = metrics["function_estimation_mse"](predictions, f_true).item()
    model.train(was_training)
    return {
        "prediction_mse": float(prediction_mse),
        "function_mse": float(function_mse),
    }


def history_is_finite(history: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    for row in history:
        for field in FINITE_HISTORY_FIELDS:
            if not math.isfinite(float(row[field])):
                return False, [f"{field} became non-finite"]
        for field in ["update_gradient_norm", "batch_loss"]:
            value = row.get(field)
            if value is not None and not math.isfinite(float(value)):
                return False, [f"{field} became non-finite"]
    return True, []


def ensure_output_dir_available(run_dir: Path) -> None:
    history_path = run_dir / "history.csv"
    metadata_path = run_dir / "metadata.json"
    if not run_dir.exists():
        return
    if history_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"{run_dir} already contains baseline output; refusing to overwrite"
        )


def ensure_experiment_output_available(output_root: Path) -> None:
    manifest_path = output_root / "baseline_comparison_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} already exists; refusing to overwrite experiment output"
        )


def load_data(
    *,
    baseline_cfg: dict[str, Any],
    torch_module: Any,
    torch_dtype: Any,
    device: Any,
) -> dict[str, tuple[Any, Any, Any]]:
    configured_dtype = np.dtype(baseline_cfg["dataset"]["dtype"])
    paths = data_paths(baseline_cfg)
    return {
        split: load_split(
            path,
            expected_dtype=configured_dtype,
            torch_dtype=torch_dtype,
            device=device,
            torch_module=torch_module,
        )
        for split, path in paths.items()
    }


def split_file_provenance(baseline_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    for split, path in data_paths(baseline_cfg).items():
        provenance[split] = {
            "path": path,
            "sha256": file_sha256(path) if path.exists() else None,
        }
    return provenance


def run_experiment_specs(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    lr_decision: dict[str, Any],
    specs: list[dict[str, Any]],
    output_root: Path,
    mode: str,
) -> dict[str, Any]:
    torch, TinyRegressionModel, train_model, metrics = import_training_stack()
    ensure_experiment_output_available(output_root)
    ds_cfg = baseline_cfg["dataset"]
    opt_cfg = baseline_cfg["optimisation"]
    configured_dtype = np.dtype(ds_cfg["dtype"])
    torch_dtype = torch_dtype_from_name(str(configured_dtype), torch_module=torch)
    device = torch.device("cpu")
    data = load_data(
        baseline_cfg=baseline_cfg,
        torch_module=torch,
        torch_dtype=torch_dtype,
        device=device,
    )
    x_train, y_train, f_train = data["train"]
    x_val, y_val, f_val = data["validation"]
    x_test, y_test, f_test = data["test"]
    if int(x_train.shape[0]) != int(ds_cfg["n_train"]):
        raise ValueError("Training split row count does not match baseline config")

    model_seeds = sorted({int(spec["model_seed"]) for spec in specs})
    references = prepare_reference_states(
        model_seeds=model_seeds,
        TinyRegressionModel=TinyRegressionModel,
        torch_module=torch,
        torch_dtype=torch_dtype,
        device=device,
    )
    for model_seed, reference in references.items():
        write_json(
            {
                "model_seed": model_seed,
                "initial_state_checksum": reference["checksum"],
                "torch_dtype": torch_dtype,
                "device": device,
                "source_git_commit_hash": commit_hash(),
            },
            output_root / "initial_states" / f"model_seed_{model_seed}" / "metadata.json",
        )

    output_root.mkdir(parents=True, exist_ok=True)
    total_runs = len(specs)
    completed_runs: list[dict[str, Any]] = []
    completed_seconds: list[float] = []
    start_all = time.perf_counter()
    source_commit = commit_hash()
    data_provenance = split_file_provenance(baseline_cfg)

    for run_index, spec in enumerate(specs, start=1):
        method_name = str(spec["method_name"])
        model_seed = int(spec["model_seed"])
        run_dir = output_root / method_name / f"model_seed_{model_seed}"
        ensure_output_dir_available(run_dir)
        print(
            f"[{run_index:02d}/{total_runs}] START {method_name} "
            f"model_seed={model_seed} sampling_seed={spec['sampling_seed']}",
            flush=True,
        )
        run_start = time.perf_counter()
        model = TinyRegressionModel().to(device=device, dtype=torch_dtype)
        reference = references[model_seed]
        model.load_state_dict(reference["state_dict"])
        loaded_checksum = state_checksum(model.state_dict())
        if loaded_checksum != reference["checksum"]:
            raise RuntimeError(
                f"Initial-state checksum mismatch for {method_name}, model_seed={model_seed}"
            )
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=float(lr_decision["selected_common_learning_rate"]),
            momentum=float(opt_cfg["momentum"]),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
        history = train_model(
            model=model,
            optimiser=optimiser,
            loss_function=torch.nn.functional.mse_loss,
            training_data=(x_train, y_train),
            evaluation_data=(x_val, y_val, f_val),
            method=method_name,
            sampling_method=str(spec["sampling_method"]),
            batch_size=int(spec["batch_size"]),
            target_examples_processed=int(spec["target_examples_processed"]),
            sampling_seed=int(spec["sampling_seed"]),
            evaluation_every_examples=int(spec["evaluation_every_examples"]),
        )
        finite, finite_reasons = history_is_finite(history)
        if not finite:
            raise RuntimeError(f"{method_name}, model_seed={model_seed}: {finite_reasons}")

        final_history = history[-1]
        train_metrics = final_split_metrics(
            model=model,
            x=x_train,
            y=y_train,
            f_true=f_train,
            metrics=metrics,
            torch_module=torch,
        )
        validation_metrics = final_split_metrics(
            model=model,
            x=x_val,
            y=y_val,
            f_true=f_val,
            metrics=metrics,
            torch_module=torch,
        )
        test_metrics = final_split_metrics(
            model=model,
            x=x_test,
            y=y_test,
            f_true=f_test,
            metrics=metrics,
            torch_module=torch,
        )
        final_metrics = {
            "training_prediction_mse": train_metrics["prediction_mse"],
            "training_function_mse": train_metrics["function_mse"],
            "validation_prediction_mse": validation_metrics["prediction_mse"],
            "validation_function_mse": validation_metrics["function_mse"],
            "test_prediction_mse": test_metrics["prediction_mse"],
            "test_function_mse": test_metrics["function_mse"],
            "parameter_norm": float(metrics["parameter_norm"](list(model.parameters())).item()),
        }
        history_path = run_dir / "history.csv"
        metadata_path = run_dir / "metadata.json"
        write_history_csv(history, history_path)

        run_elapsed = time.perf_counter() - run_start
        achieved_examples = int(final_history["cumulative_examples_processed"])
        achieved_dep = float(final_history["data_equivalent_passes"])
        metadata = {
            "experiment_name": v3_cfg["experiment"]["name"],
            "mode": mode,
            "method_name": method_name,
            "sampling_method": spec["sampling_method"],
            "nominal_batch_size": int(spec["batch_size"]),
            "model_seed": model_seed,
            "sampling_seed": int(spec["sampling_seed"]),
            "sampling_seed_source": "explicit future_baseline method x model-seed table",
            "learning_rate": lr_decision["selected_common_learning_rate"],
            "learning_rate_decision_artifact": lr_decision["path"],
            "learning_rate_decision_artifact_sha256": lr_decision["sha256"],
            "learning_rate_decision_source_git_commit_hash": lr_decision[
                "source_git_commit_hash"
            ],
            "initial_state_checksum": reference["checksum"],
            "loaded_initial_state_checksum": loaded_checksum,
            "target_examples_processed": int(spec["target_examples_processed"]),
            "achieved_examples_processed": achieved_examples,
            "achieved_data_equivalent_passes": achieved_dep,
            "evaluation_every_examples": int(spec["evaluation_every_examples"]),
            "dtype": configured_dtype,
            "device": device,
            "optimiser": {
                "name": "SGD",
                "momentum": float(opt_cfg["momentum"]),
                "weight_decay": float(opt_cfg["weight_decay"]),
            },
            "data_provenance": {
                "paths": data_paths(baseline_cfg),
                "configured_sample_sizes": {
                    "train": int(ds_cfg["n_train"]),
                    "validation": int(ds_cfg["n_validation"]),
                    "test": int(ds_cfg["n_test"]),
                },
                "dtype": configured_dtype,
                "file_hashes": data_provenance,
            },
            "runtime": {
                "run_elapsed_seconds": run_elapsed,
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "source_git_commit_hash": source_commit,
                "worktree_status": worktree_status(),
            },
            "final_step_count": int(final_history["step"]),
            "final_epoch": final_history["epoch"],
            "final_split_metrics": final_metrics,
            "artifacts": {
                "history_csv": history_path,
                "metadata_json": metadata_path,
            },
            "test_usage_note": "Test split is evaluated once at the end of the locked run.",
        }
        write_json(metadata, metadata_path)
        completed_record = {
            "method_name": method_name,
            "model_seed": model_seed,
            "sampling_seed": int(spec["sampling_seed"]),
            "metadata_json": metadata_path,
            "history_csv": history_path,
            "actual_examples_processed": achieved_examples,
            "data_equivalent_passes": achieved_dep,
            "test_function_mse": final_metrics["test_function_mse"],
        }
        completed_runs.append(completed_record)
        completed_seconds.append(run_elapsed)
        elapsed = time.perf_counter() - start_all
        remaining = total_runs - run_index
        eta = remaining * (sum(completed_seconds) / len(completed_seconds))
        print(
            f"[{run_index:02d}/{total_runs}] DONE {method_name} "
            f"run={run_elapsed:.1f}s total_elapsed={elapsed / 60:.1f}min "
            f"ETA={eta / 60:.1f}min",
            flush=True,
        )

    manifest = {
        "experiment_name": v3_cfg["experiment"]["name"],
        "mode": mode,
        "expected_run_count": total_runs,
        "actual_completed_run_count": len(completed_runs),
        "method_definitions": {
            method_name: method_cfg for method_name, method_cfg in method_items(v3_cfg)
        },
        "model_seeds": model_seeds,
        "explicit_sampling_seed_table": sampling_seed_table(v3_cfg),
        "learning_rate": lr_decision["selected_common_learning_rate"],
        "human_approval_confirmed": lr_decision["human_approved"],
        "learning_rate_decision_artifact": lr_decision["path"],
        "learning_rate_decision_artifact_sha256": lr_decision["sha256"],
        "learning_rate_decision_source_git_commit_hash": lr_decision[
            "source_git_commit_hash"
        ],
        "target_examples_processed": specs[0]["target_examples_processed"] if specs else None,
        "data_equivalent_passes": (
            specs[0]["target_examples_processed"] / int(ds_cfg["n_train"])
            if specs
            else None
        ),
        "evaluation_every_examples": specs[0]["evaluation_every_examples"] if specs else None,
        "checkpoint_schedule": expected_checkpoint_examples(
            target_examples_processed=int(specs[0]["target_examples_processed"]),
            evaluation_every_examples=int(specs[0]["evaluation_every_examples"]),
        )
        if specs
        else [],
        "common_config": {
            "baseline_config_path": BASELINE_CONFIG_PATH,
            "v3_config_path": V3_CONFIG_PATH,
            "dtype": configured_dtype,
            "device": device,
            "data_provenance": data_provenance,
        },
        "source_git_commit_hash": source_commit,
        "worktree_status": worktree_status(),
        "run_metadata_paths": [record["metadata_json"] for record in completed_runs],
        "runs": completed_runs,
        "total_elapsed_seconds": time.perf_counter() - start_all,
    }
    write_json(manifest, output_root / "baseline_comparison_manifest.json")
    return manifest


def run_preflight(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    lr_decision: dict[str, Any],
) -> dict[str, Any]:
    specs = build_preflight_run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        learning_rate=float(lr_decision["selected_common_learning_rate"]),
    )
    output_root = preflight_output_root(v3_cfg)
    manifest = run_experiment_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        lr_decision=lr_decision,
        specs=specs,
        output_root=output_root,
        mode="preflight",
    )
    failures: list[str] = []
    for run in manifest["runs"]:
        if int(run["actual_examples_processed"]) != int(baseline_cfg["dataset"]["n_train"]):
            failures.append(f"{run['method_name']}: incorrect final examples")
        if not math.isclose(float(run["data_equivalent_passes"]), 1.0, abs_tol=1e-12):
            failures.append(f"{run['method_name']}: incorrect DEP")
    metadata_paths = [Path(path) for path in manifest["run_metadata_paths"]]
    hashes = {
        load_json(path)["learning_rate_decision_artifact_sha256"]
        for path in metadata_paths
    }
    if hashes != {lr_decision["sha256"]}:
        failures.append("run metadata did not record the approved LR artifact hash")
    initial_checksums = {
        load_json(path)["initial_state_checksum"]
        for path in metadata_paths
    }
    if len(initial_checksums) != 1:
        failures.append("preflight methods did not share one model-seed initial state")
    if failures:
        raise RuntimeError("V3 baseline preflight failed: " + "; ".join(failures))
    manifest["preflight_ok"] = True
    manifest["preflight_failures"] = []
    write_json(manifest, output_root / "baseline_comparison_manifest.json")
    return manifest


def run_full(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    lr_decision: dict[str, Any],
) -> dict[str, Any]:
    specs = build_baseline_run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        learning_rate=float(lr_decision["selected_common_learning_rate"]),
    )
    return run_experiment_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        lr_decision=lr_decision,
        specs=specs,
        output_root=baseline_output_root(v3_cfg),
        mode="full",
    )


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
    validate_v3_config(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg)
    lr_decision = validate_approved_lr_decision(lr_decision_artifact_path(v3_cfg))

    if args.plan_only:
        print(json.dumps(to_jsonable(plan(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg, lr_decision=lr_decision)), indent=2))
        return
    if args.preflight:
        print(json.dumps(to_jsonable(run_preflight(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg, lr_decision=lr_decision)), indent=2))
        return
    if args.full:
        print(json.dumps(to_jsonable(run_full(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg, lr_decision=lr_decision)), indent=2))
        return


if __name__ == "__main__":
    main()
