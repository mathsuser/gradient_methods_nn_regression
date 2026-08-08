from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradient_methods_nn_regression.metrics import (
    function_estimation_mse,
    noisy_prediction_mse,
    parameter_norm,
)
from gradient_methods_nn_regression.model import TinyRegressionModel
from gradient_methods_nn_regression.training import train_model


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_by_name = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return dtype_by_name[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported configured dtype for training: {dtype_name}") from exc


def _load_split(
    path: Path,
    *,
    expected_dtype: np.dtype,
    torch_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(path) as data:
        x_np = data["x"]
        y_np = data["y"].reshape(-1, 1)
        f_true_np = data["f_true"].reshape(-1, 1)

    for name, array in [
        ("x", x_np),
        ("y", y_np),
        ("f_true", f_true_np),
    ]:
        if array.dtype != expected_dtype:
            raise ValueError(
                f"{path.name}:{name} has dtype {array.dtype}, expected {expected_dtype}"
            )

    x = torch.as_tensor(x_np, dtype=torch_dtype, device=device)
    y = torch.as_tensor(y_np, dtype=torch_dtype, device=device)
    f_true = torch.as_tensor(f_true_np, dtype=torch_dtype, device=device)
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
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


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


def _state_checksum(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_history_csv(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field) for field in HISTORY_FIELDNAMES})


def _write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(data), handle, indent=2)


def _final_split_metrics(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    f_true: torch.Tensor,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        prediction_mse = noisy_prediction_mse(predictions, y).item()
        function_mse = function_estimation_mse(predictions, f_true).item()
    model.train(was_training)
    return {
        "prediction_mse": float(prediction_mse),
        "function_mse": float(function_mse),
    }


def _load_selected_learning_rate(path: Path) -> float:
    selection = _load_json(path)
    selected = selection.get("selected_common_learning_rate")
    if selected is None:
        raise ValueError(
            f"No selected_common_learning_rate is locked in {path}; run or review the pilot first."
        )
    return float(selected)


def run_baseline_comparison(
    *,
    baseline_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    learning_rate: float,
    learning_rate_selection_path: Path,
) -> dict[str, Any]:
    ds_cfg = baseline_cfg["dataset"]
    opt_cfg = baseline_cfg["optimisation"]
    training_cfg = experiment_cfg["experiment"]["training"]

    configured_dtype = np.dtype(ds_cfg["dtype"])
    torch_dtype = _torch_dtype(str(configured_dtype))
    device = torch.device("cpu")

    generated_data_dir = Path(baseline_cfg["paths"]["generated_data_dir"])
    raw_root = Path(baseline_cfg["paths"]["results_raw_dir"]) / "week1_gradient_methods"
    output_root = raw_root / "baseline_comparison_runs"
    output_root.mkdir(parents=True, exist_ok=True)

    x_train, y_train, f_train = _load_split(
        generated_data_dir / "baseline_train.npz",
        expected_dtype=configured_dtype,
        torch_dtype=torch_dtype,
        device=device,
    )
    x_val, y_val, f_val = _load_split(
        generated_data_dir / "baseline_validation.npz",
        expected_dtype=configured_dtype,
        torch_dtype=torch_dtype,
        device=device,
    )
    x_test, y_test, f_test = _load_split(
        generated_data_dir / "baseline_test.npz",
        expected_dtype=configured_dtype,
        torch_dtype=torch_dtype,
        device=device,
    )

    expected_n_train = int(ds_cfg["n_train"])
    if int(x_train.shape[0]) != expected_n_train:
        raise ValueError(
            f"Training split has {x_train.shape[0]} rows, expected {expected_n_train}"
        )

    target_examples_processed = int(training_cfg["target_examples_processed"])
    evaluation_every_examples = int(training_cfg["evaluation_every_examples"])
    data_equivalent_passes = float(training_cfg["data_equivalent_passes"])
    if target_examples_processed != int(data_equivalent_passes * expected_n_train):
        raise ValueError(
            "Configured target_examples_processed does not match "
            "data_equivalent_passes * n_train"
        )

    methods_cfg = experiment_cfg["experiment"]["methods"]
    method_items = list(methods_cfg.items())
    model_seeds = [int(seed) for seed in opt_cfg["model_seeds"]]
    sampling_seed_offset = int(opt_cfg["sampling_seed_offset"])
    momentum = float(opt_cfg["momentum"])
    weight_decay = float(opt_cfg["weight_decay"])
    commit = _commit_hash()

    run_records: list[dict[str, Any]] = []
    comparison_start = time.perf_counter()

    for model_seed in model_seeds:
        torch.manual_seed(model_seed)
        reference_model = TinyRegressionModel().to(device=device, dtype=torch_dtype)
        reference_state = copy.deepcopy(reference_model.state_dict())
        initial_checksum = _state_checksum(reference_state)

        seed_dir = output_root / "initial_states" / f"seed_{model_seed}"
        _write_json(
            {
                "model_seed": model_seed,
                "initial_state_checksum": initial_checksum,
                "torch_dtype": torch_dtype,
                "device": device,
                "commit_hash": commit,
            },
            seed_dir / "initial_state_checksum.json",
        )

        for method_index, (method_key, method_cfg) in enumerate(method_items):
            method_name = str(method_cfg.get("method_name", method_key))
            sampling_method = str(method_cfg["sampling_method"])
            batch_size = int(method_cfg["batch_size"])
            sampling_seed = sampling_seed_offset + model_seed * len(method_items) + method_index

            model = TinyRegressionModel().to(device=device, dtype=torch_dtype)
            model.load_state_dict(reference_state)
            loaded_checksum = _state_checksum(model.state_dict())
            if loaded_checksum != initial_checksum:
                raise RuntimeError(
                    f"Initial-state checksum mismatch for model_seed={model_seed}, "
                    f"method={method_name}"
                )

            optimiser = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )

            run_start = time.perf_counter()
            history = train_model(
                model=model,
                optimiser=optimiser,
                loss_function=torch.nn.functional.mse_loss,
                training_data=(x_train, y_train),
                evaluation_data=(x_val, y_val, f_val),
                method=method_name,
                sampling_method=sampling_method,
                batch_size=batch_size,
                target_examples_processed=target_examples_processed,
                sampling_seed=sampling_seed,
                evaluation_every_examples=evaluation_every_examples,
            )
            run_elapsed_seconds = time.perf_counter() - run_start

            final_history = history[-1]
            train_metrics = _final_split_metrics(
                model=model,
                x=x_train,
                y=y_train,
                f_true=f_train,
            )
            validation_metrics = _final_split_metrics(
                model=model,
                x=x_val,
                y=y_val,
                f_true=f_val,
            )
            test_metrics = _final_split_metrics(
                model=model,
                x=x_test,
                y=y_test,
                f_true=f_test,
            )

            final_metrics = {
                "training_prediction_mse": train_metrics["prediction_mse"],
                "training_function_mse": train_metrics["function_mse"],
                "validation_prediction_mse": validation_metrics["prediction_mse"],
                "validation_function_mse": validation_metrics["function_mse"],
                "test_prediction_mse": test_metrics["prediction_mse"],
                "test_function_mse": test_metrics["function_mse"],
                "parameter_norm": float(parameter_norm(list(model.parameters())).item()),
            }

            run_dir = output_root / method_name / f"seed_{model_seed}"
            history_path = run_dir / "history.csv"
            metadata_path = run_dir / "metadata.json"
            _write_history_csv(history, history_path)

            metadata = {
                "method_name": method_name,
                "sampling_method": sampling_method,
                "nominal_batch_size": batch_size,
                "learning_rate": learning_rate,
                "learning_rate_selection_artifact": learning_rate_selection_path,
                "model_seed": model_seed,
                "sampling_seed": sampling_seed,
                "sampling_seed_derivation": (
                    "sampling_seed_offset + model_seed * number_of_methods + method_index"
                ),
                "initial_state_checksum": initial_checksum,
                "loaded_initial_state_checksum": loaded_checksum,
                "data_seeds": ds_cfg["seeds"],
                "baseline_config_snapshot": baseline_cfg,
                "experiment_config_snapshot": experiment_cfg,
                "target_examples_processed": target_examples_processed,
                "evaluation_every_examples": evaluation_every_examples,
                "early_stopping": False,
                "final_step_count": int(final_history["step"]),
                "actual_examples_processed": int(
                    final_history["cumulative_examples_processed"]
                ),
                "data_equivalent_passes": float(final_history["data_equivalent_passes"]),
                "epoch_count": final_history["epoch"],
                "elapsed_time": {
                    "training_elapsed_seconds": float(
                        final_history["training_elapsed_seconds"]
                    ),
                    "total_elapsed_seconds": float(final_history["total_elapsed_seconds"]),
                    "run_elapsed_seconds": float(run_elapsed_seconds),
                },
                "final_metrics": final_metrics,
                "runtime": {
                    "device": device,
                    "torch_version": torch.__version__,
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "commit_hash": commit,
                },
                "artifacts": {
                    "history_csv": history_path,
                    "metadata_json": metadata_path,
                },
                "test_usage_note": (
                    "Test split is evaluated only after the locked learning-rate "
                    "decision and after this training run completes."
                ),
            }
            _write_json(metadata, metadata_path)

            run_records.append(
                {
                    "method_name": method_name,
                    "model_seed": model_seed,
                    "sampling_seed": sampling_seed,
                    "history_csv": history_path,
                    "metadata_json": metadata_path,
                    "final_step_count": metadata["final_step_count"],
                    "actual_examples_processed": metadata["actual_examples_processed"],
                    "data_equivalent_passes": metadata["data_equivalent_passes"],
                    "epoch_count": metadata["epoch_count"],
                    "test_function_mse": final_metrics["test_function_mse"],
                    "test_prediction_mse": final_metrics["test_prediction_mse"],
                }
            )

    manifest = {
        "experiment_name": experiment_cfg["experiment"]["name"],
        "run_count": len(run_records),
        "expected_run_count": len(method_items) * len(model_seeds),
        "model_seeds": model_seeds,
        "method_order": [name for name, _ in method_items],
        "learning_rate": learning_rate,
        "learning_rate_selection_artifact": learning_rate_selection_path,
        "target_examples_processed": target_examples_processed,
        "evaluation_every_examples": evaluation_every_examples,
        "output_root": output_root,
        "elapsed_seconds": time.perf_counter() - comparison_start,
        "runtime": {
            "device": device,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "commit_hash": commit,
        },
        "runs": run_records,
    }
    _write_json(manifest, raw_root / "baseline_comparison_manifest.json")
    return manifest


def main() -> None:
    baseline_cfg = _load_json(Path("configs/baseline.json"))
    experiment_cfg = _load_json(Path("configs/experiments/week1_gradient_methods.json"))
    learning_rate_selection_path = Path(
        "results/raw/week1_gradient_methods/learning_rate_selection.json"
    )
    learning_rate = _load_selected_learning_rate(learning_rate_selection_path)

    manifest = run_baseline_comparison(
        baseline_cfg=baseline_cfg,
        experiment_cfg=experiment_cfg,
        learning_rate=learning_rate,
        learning_rate_selection_path=learning_rate_selection_path,
    )

    print(
        "Completed baseline comparison: "
        f"{manifest['run_count']} runs at learning_rate={learning_rate:g}"
    )
    print("Manifest: results/raw/week1_gradient_methods/baseline_comparison_manifest.json")


if __name__ == "__main__":
    main()
