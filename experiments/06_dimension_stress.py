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

from gradient_methods_nn_regression.data import generate_synthetic_regression_data


STRESS_DIMENSIONS = [20, 100]
HISTORY_FIELDNAMES = [
    "dimension",
    "n_relevant_features",
    "parameter_count",
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


def _import_torch() -> Any:
    import torch

    return torch


def _make_model(*, torch_module: Any, input_dim: int, hidden_dim: int) -> Any:
    return torch_module.nn.Sequential(
        torch_module.nn.Linear(input_dim, hidden_dim),
        torch_module.nn.Tanh(),
        torch_module.nn.Linear(hidden_dim, 1),
    )


def _count_trainable_parameters(model: Any) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _torch_dtype(dtype_name: str, *, torch_module: Any) -> Any:
    dtype_by_name = {
        "float32": torch_module.float32,
        "float64": torch_module.float64,
    }
    try:
        return dtype_by_name[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported configured dtype for training: {dtype_name}") from exc


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return float(value)
        return str(value)
    if value.__class__.__name__ == "dtype" and str(value).startswith("torch."):
        return str(value).replace("torch.", "")
    if value.__class__.__name__ == "device":
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return str(value)


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(data), handle, indent=2)


def _write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field) for field in HISTORY_FIELDNAMES})


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
        return "unknown"


def _state_checksum(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_selected_learning_rate(path: Path) -> float:
    selection = _load_json(path)
    selected = selection.get("selected_common_learning_rate")
    if selected is None:
        raise ValueError(f"No selected_common_learning_rate is locked in {path}")
    return float(selected)


def _make_split(
    *,
    n_samples: int,
    n_features: int,
    noise_std: float,
    seed: int,
    numpy_dtype: np.dtype,
    torch_dtype: Any,
    device: Any,
    torch_module: Any,
) -> tuple[Any, Any, Any]:
    x_np, y_np, f_true_np, _ = generate_synthetic_regression_data(
        n_samples=n_samples,
        n_features=n_features,
        noise_std=noise_std,
        seed=seed,
        dtype=numpy_dtype,
    )
    x = torch_module.as_tensor(x_np, dtype=torch_dtype, device=device)
    y = torch_module.as_tensor(
        y_np.reshape(-1, 1),
        dtype=torch_dtype,
        device=device,
    )
    f_true = torch_module.as_tensor(
        f_true_np.reshape(-1, 1),
        dtype=torch_dtype,
        device=device,
    )
    return x, y, f_true


def _make_splits_for_dimension(
    *,
    dimension: int,
    baseline_cfg: dict[str, Any],
    numpy_dtype: np.dtype,
    torch_dtype: Any,
    device: Any,
    torch_module: Any,
) -> dict[str, tuple[Any, Any, Any]]:
    ds_cfg = baseline_cfg["dataset"]
    return {
        "train": _make_split(
            n_samples=int(ds_cfg["n_train"]),
            n_features=dimension,
            noise_std=float(ds_cfg["noise_std"]),
            seed=int(ds_cfg["seeds"]["train"]),
            numpy_dtype=numpy_dtype,
            torch_dtype=torch_dtype,
            device=device,
            torch_module=torch_module,
        ),
        "validation": _make_split(
            n_samples=int(ds_cfg["n_validation"]),
            n_features=dimension,
            noise_std=float(ds_cfg["noise_std"]),
            seed=int(ds_cfg["seeds"]["validation"]),
            numpy_dtype=numpy_dtype,
            torch_dtype=torch_dtype,
            device=device,
            torch_module=torch_module,
        ),
        "test": _make_split(
            n_samples=int(ds_cfg["n_test"]),
            n_features=dimension,
            noise_std=float(ds_cfg["noise_std"]),
            seed=int(ds_cfg["seeds"]["test"]),
            numpy_dtype=numpy_dtype,
            torch_dtype=torch_dtype,
            device=device,
            torch_module=torch_module,
        ),
    }


def _final_split_metrics(
    *,
    model: Any,
    x: Any,
    y: Any,
    f_true: Any,
    torch_module: Any,
    noisy_prediction_mse: Any,
    function_estimation_mse: Any,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    with torch_module.no_grad():
        predictions = model(x)
        prediction_mse = noisy_prediction_mse(predictions, y).item()
        function_mse = function_estimation_mse(predictions, f_true).item()
    model.train(was_training)
    return {
        "prediction_mse": float(prediction_mse),
        "function_mse": float(function_mse),
    }


def _parameter_counts(
    *,
    dimensions: list[int],
    baseline_cfg: dict[str, Any],
) -> dict[int, int]:
    hidden_dim = int(baseline_cfg["model"]["hidden_dim"])
    torch_module = _import_torch()
    return {
        dimension: _count_trainable_parameters(
            _make_model(
                torch_module=torch_module,
                input_dim=dimension,
                hidden_dim=hidden_dim,
            )
        )
        for dimension in dimensions
    }


def _expected_parameter_counts(
    *,
    dimensions: list[int],
    baseline_cfg: dict[str, Any],
) -> dict[int, int]:
    hidden_dim = int(baseline_cfg["model"]["hidden_dim"])
    output_dim = int(baseline_cfg["model"]["output_dim"])
    return {
        dimension: (dimension + 1) * hidden_dim + (hidden_dim + 1) * output_dim
        for dimension in dimensions
    }


def preflight(
    *,
    baseline_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    model_seeds = [int(seed) for seed in baseline_cfg["optimisation"]["model_seeds"]]
    method_items = list(experiment_cfg["experiment"]["methods"].items())
    parameter_counts = _expected_parameter_counts(
        dimensions=[6, *STRESS_DIMENSIONS],
        baseline_cfg=baseline_cfg,
    )
    expected_run_count = len(STRESS_DIMENSIONS) * len(method_items) * len(model_seeds)
    report = {
        "ok": True,
        "dimensions_to_run": STRESS_DIMENSIONS,
        "reused_baseline_dimension": 6,
        "model_seeds": model_seeds,
        "method_count": len(method_items),
        "expected_new_run_count": expected_run_count,
        "learning_rate": learning_rate,
        "parameter_counts": parameter_counts,
        "n_relevant_features": int(baseline_cfg["dataset"]["n_relevant_features"]),
        "training_budget": experiment_cfg["experiment"]["training"],
        "output_root": "results/raw/week1_dimension_stress",
    }
    return report


def run_dimension_stress(
    *,
    baseline_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    learning_rate: float,
    learning_rate_selection_path: Path,
) -> dict[str, Any]:
    ds_cfg = baseline_cfg["dataset"]
    opt_cfg = baseline_cfg["optimisation"]
    training_cfg = experiment_cfg["experiment"]["training"]
    model_cfg = baseline_cfg["model"]

    n_relevant_features = int(ds_cfg["n_relevant_features"])
    if n_relevant_features != 6:
        raise ValueError("Step 16 expects n_relevant_features == 6")

    numpy_dtype = np.dtype(ds_cfg["dtype"])
    torch_module = _import_torch()
    from gradient_methods_nn_regression.metrics import (
        function_estimation_mse,
        noisy_prediction_mse,
        parameter_norm,
    )
    from gradient_methods_nn_regression.training import train_model

    torch_dtype = _torch_dtype(str(numpy_dtype), torch_module=torch_module)
    device = torch_module.device("cpu")
    output_root = Path(baseline_cfg["paths"]["results_raw_dir"]).parent
    output_root = output_root / "raw" / "week1_dimension_stress"
    output_root.mkdir(parents=True, exist_ok=True)

    target_examples_processed = int(training_cfg["target_examples_processed"])
    evaluation_every_examples = int(training_cfg["evaluation_every_examples"])
    model_seeds = [int(seed) for seed in opt_cfg["model_seeds"]]
    method_items = list(experiment_cfg["experiment"]["methods"].items())
    sampling_seed_offset = int(opt_cfg["sampling_seed_offset"])
    hidden_dim = int(model_cfg["hidden_dim"])
    momentum = float(opt_cfg["momentum"])
    weight_decay = float(opt_cfg["weight_decay"])
    commit = _commit_hash()

    run_records: list[dict[str, Any]] = []
    stress_start = time.perf_counter()

    for dimension in STRESS_DIMENSIONS:
        splits = _make_splits_for_dimension(
            dimension=dimension,
            baseline_cfg=baseline_cfg,
            numpy_dtype=numpy_dtype,
            torch_dtype=torch_dtype,
            device=device,
            torch_module=torch_module,
        )

        for model_seed in model_seeds:
            torch_module.manual_seed(model_seed)
            reference_model = _make_model(
                torch_module=torch_module,
                input_dim=dimension,
                hidden_dim=hidden_dim,
            ).to(device=device, dtype=torch_dtype)
            parameter_count = _count_trainable_parameters(reference_model)
            reference_state = copy.deepcopy(reference_model.state_dict())
            initial_checksum = _state_checksum(reference_state)

            checksum_dir = (
                output_root / f"dimension_{dimension}" / "initial_states"
                / f"seed_{model_seed}"
            )
            _write_json(
                {
                    "dimension": dimension,
                    "n_relevant_features": n_relevant_features,
                    "parameter_count": parameter_count,
                    "model_seed": model_seed,
                    "initial_state_checksum": initial_checksum,
                    "torch_dtype": torch_dtype,
                    "device": device,
                    "commit_hash": commit,
                },
                checksum_dir / "initial_state_checksum.json",
            )

            for method_index, (method_key, method_cfg) in enumerate(method_items):
                method_name = str(method_cfg.get("method_name", method_key))
                sampling_method = str(method_cfg["sampling_method"])
                batch_size = int(method_cfg["batch_size"])
                sampling_seed = (
                    sampling_seed_offset
                    + model_seed * len(method_items)
                    + method_index
                )

                model = _make_model(
                    torch_module=torch_module,
                    input_dim=dimension,
                    hidden_dim=hidden_dim,
                ).to(device=device, dtype=torch_dtype)
                model.load_state_dict(reference_state)
                loaded_checksum = _state_checksum(model.state_dict())
                if loaded_checksum != initial_checksum:
                    raise RuntimeError(
                        f"Initial-state checksum mismatch for d={dimension}, "
                        f"seed={model_seed}, method={method_name}"
                    )

                optimiser = torch_module.optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )

                run_start = time.perf_counter()
                history = train_model(
                    model=model,
                    optimiser=optimiser,
                    loss_function=torch_module.nn.functional.mse_loss,
                    training_data=(splits["train"][0], splits["train"][1]),
                    evaluation_data=(
                        splits["validation"][0],
                        splits["validation"][1],
                        splits["validation"][2],
                    ),
                    method=method_name,
                    sampling_method=sampling_method,
                    batch_size=batch_size,
                    target_examples_processed=target_examples_processed,
                    sampling_seed=sampling_seed,
                    evaluation_every_examples=evaluation_every_examples,
                )
                run_elapsed_seconds = time.perf_counter() - run_start

                enriched_history = [
                    {
                        "dimension": dimension,
                        "n_relevant_features": n_relevant_features,
                        "parameter_count": parameter_count,
                        **row,
                    }
                    for row in history
                ]

                train_metrics = _final_split_metrics(
                    model=model,
                    x=splits["train"][0],
                    y=splits["train"][1],
                    f_true=splits["train"][2],
                    torch_module=torch_module,
                    noisy_prediction_mse=noisy_prediction_mse,
                    function_estimation_mse=function_estimation_mse,
                )
                validation_metrics = _final_split_metrics(
                    model=model,
                    x=splits["validation"][0],
                    y=splits["validation"][1],
                    f_true=splits["validation"][2],
                    torch_module=torch_module,
                    noisy_prediction_mse=noisy_prediction_mse,
                    function_estimation_mse=function_estimation_mse,
                )
                test_metrics = _final_split_metrics(
                    model=model,
                    x=splits["test"][0],
                    y=splits["test"][1],
                    f_true=splits["test"][2],
                    torch_module=torch_module,
                    noisy_prediction_mse=noisy_prediction_mse,
                    function_estimation_mse=function_estimation_mse,
                )

                final_history = history[-1]
                final_metrics = {
                    "training_prediction_mse": train_metrics["prediction_mse"],
                    "training_function_mse": train_metrics["function_mse"],
                    "validation_prediction_mse": validation_metrics["prediction_mse"],
                    "validation_function_mse": validation_metrics["function_mse"],
                    "test_prediction_mse": test_metrics["prediction_mse"],
                    "test_function_mse": test_metrics["function_mse"],
                    "parameter_norm": float(
                        parameter_norm(list(model.parameters())).item()
                    ),
                }

                run_dir = (
                    output_root / f"dimension_{dimension}" / method_name
                    / f"seed_{model_seed}"
                )
                history_path = run_dir / "history.csv"
                metadata_path = run_dir / "metadata.json"
                _write_history_csv(enriched_history, history_path)

                metadata = {
                    "dimension": dimension,
                    "n_relevant_features": n_relevant_features,
                    "parameter_count": parameter_count,
                    "method_name": method_name,
                    "sampling_method": sampling_method,
                    "nominal_batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "learning_rate_selection_artifact": learning_rate_selection_path,
                    "model_seed": model_seed,
                    "sampling_seed": sampling_seed,
                    "sampling_seed_derivation": (
                        "sampling_seed_offset + model_seed * number_of_methods "
                        "+ method_index"
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
                    "data_equivalent_passes": float(
                        final_history["data_equivalent_passes"]
                    ),
                    "epoch_count": final_history["epoch"],
                    "elapsed_time": {
                        "training_elapsed_seconds": float(
                            final_history["training_elapsed_seconds"]
                        ),
                        "total_elapsed_seconds": float(
                            final_history["total_elapsed_seconds"]
                        ),
                        "run_elapsed_seconds": float(run_elapsed_seconds),
                    },
                    "final_metrics": final_metrics,
                    "runtime": {
                        "device": device,
                    "torch_version": torch_module.__version__,
                        "python_version": platform.python_version(),
                        "platform": platform.platform(),
                        "commit_hash": commit,
                    },
                    "artifacts": {
                        "history_csv": history_path,
                        "metadata_json": metadata_path,
                    },
                    "design_note": (
                        "Features after the first six are independent irrelevant "
                        "standard Gaussian variables. Increasing dimension also "
                        "increases first-layer parameter count."
                    ),
                }
                _write_json(metadata, metadata_path)

                run_records.append(
                    {
                        "dimension": dimension,
                        "n_relevant_features": n_relevant_features,
                        "parameter_count": parameter_count,
                        "method_name": method_name,
                        "model_seed": model_seed,
                        "sampling_seed": sampling_seed,
                        "history_csv": history_path,
                        "metadata_json": metadata_path,
                        "final_step_count": metadata["final_step_count"],
                        "actual_examples_processed": metadata[
                            "actual_examples_processed"
                        ],
                        "data_equivalent_passes": metadata[
                            "data_equivalent_passes"
                        ],
                        "epoch_count": metadata["epoch_count"],
                        "test_function_mse": final_metrics["test_function_mse"],
                        "test_prediction_mse": final_metrics["test_prediction_mse"],
                    }
                )

    manifest = {
        "experiment_name": "week1_dimension_stress",
        "stress_dimensions": STRESS_DIMENSIONS,
        "reused_baseline_dimension": 6,
        "run_count": len(run_records),
        "expected_run_count": len(STRESS_DIMENSIONS) * len(method_items)
        * len(model_seeds),
        "model_seeds": model_seeds,
        "method_order": [name for name, _ in method_items],
        "learning_rate": learning_rate,
        "learning_rate_selection_artifact": learning_rate_selection_path,
        "target_examples_processed": target_examples_processed,
        "evaluation_every_examples": evaluation_every_examples,
        "output_root": output_root,
        "elapsed_seconds": time.perf_counter() - stress_start,
        "runtime": {
            "device": device,
            "torch_version": torch_module.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "commit_hash": commit,
        },
        "runs": run_records,
    }
    _write_json(manifest, output_root / "dimension_stress_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate planned Step 16 run shape without training.",
    )
    args = parser.parse_args()

    baseline_cfg = _load_json(Path("configs/baseline.json"))
    experiment_cfg = _load_json(Path("configs/experiments/week1_gradient_methods.json"))
    learning_rate_selection_path = Path(
        "results/raw/week1_gradient_methods/learning_rate_selection.json"
    )
    learning_rate = _load_selected_learning_rate(learning_rate_selection_path)

    if args.preflight:
        report = preflight(
            baseline_cfg=baseline_cfg,
            experiment_cfg=experiment_cfg,
            learning_rate=learning_rate,
        )
        print(json.dumps(_to_jsonable(report), indent=2))
        return

    manifest = run_dimension_stress(
        baseline_cfg=baseline_cfg,
        experiment_cfg=experiment_cfg,
        learning_rate=learning_rate,
        learning_rate_selection_path=learning_rate_selection_path,
    )
    print(
        "Completed dimension stress: "
        f"{manifest['run_count']} new runs at learning_rate={learning_rate:g}"
    )
    print("Manifest: results/raw/week1_dimension_stress/dimension_stress_manifest.json")


if __name__ == "__main__":
    main()
