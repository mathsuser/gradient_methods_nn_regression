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


BASELINE_CONFIG_PATH = Path("configs/baseline.json")
V3_CONFIG_PATH = Path("configs/experiments/week1_gradient_methods_v3.json")
LR_DECISION_PATH = Path("results/raw/week1_gradient_methods_v3/learning_rate_selection.json")
ACCEPTED_BASELINE_MANIFEST_PATH = Path(
    "results/raw/week1_gradient_methods_v3/baseline_comparison_runs/baseline_comparison_manifest.json"
)
OUTPUT_ROOT = Path("results/raw/week1_dimension_stress_v3")
FIGURE_ROOT = Path("results/figures/week1_dimension_stress_v3")
STRESS_DIMENSIONS = [20, 100]
REUSED_BASELINE_DIMENSION = 6
N_RELEVANT_FEATURES = 6
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
MODEL_SEEDS = [0, 1, 2, 3, 4]
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


def arrays_sha256(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(contiguous.tobytes())
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


def expected_parameter_count(*, dimension: int, hidden_dim: int = 16, output_dim: int = 1) -> int:
    return (dimension + 1) * hidden_dim + (hidden_dim + 1) * output_dim


def parameter_counts() -> dict[int, int]:
    return {
        dimension: expected_parameter_count(dimension=dimension)
        for dimension in [REUSED_BASELINE_DIMENSION, *STRESS_DIMENSIONS]
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


def sampling_seed_table(v3_cfg: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        method_name: {
            str(model_seed): future_baseline_seed_for(
                v3_cfg,
                method_name=method_name,
                model_seed=model_seed,
            )
            for model_seed in MODEL_SEEDS
        }
        for method_name in METHOD_ORDER
    }


def validate_configs(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
) -> None:
    ds_cfg = baseline_cfg["dataset"]
    model_cfg = baseline_cfg["model"]
    opt_cfg = baseline_cfg["optimisation"]
    training_cfg = v3_cfg["experiment"]["training"]
    if int(ds_cfg["n_relevant_features"]) != N_RELEVANT_FEATURES:
        raise ValueError("n_relevant_features must remain 6")
    if str(ds_cfg["dtype"]) != "float64":
        raise ValueError("dtype must remain float64")
    if int(ds_cfg["n_train"]) != 5000 or int(ds_cfg["n_validation"]) != 2000 or int(ds_cfg["n_test"]) != 20000:
        raise ValueError("baseline sample sizes changed")
    if float(ds_cfg["noise_std"]) != 0.3:
        raise ValueError("noise_std must remain 0.3")
    if [int(ds_cfg["seeds"][split]) for split in ["train", "validation", "test"]] != [101, 102, 103]:
        raise ValueError("baseline data split seeds changed")
    if int(model_cfg["hidden_dim"]) != 16 or str(model_cfg["activation"]) != "tanh":
        raise ValueError("hidden width and activation must remain 16/tanh")
    if str(opt_cfg["optimizer"]) != "SGD":
        raise ValueError("optimiser must remain SGD")
    if float(opt_cfg["momentum"]) != 0.0 or float(opt_cfg["weight_decay"]) != 0.0:
        raise ValueError("momentum and weight_decay must remain zero")
    if [int(seed) for seed in opt_cfg["model_seeds"]] != MODEL_SEEDS:
        raise ValueError("model seeds must be [0, 1, 2, 3, 4]")
    if [int(seed) for seed in v3_cfg["experiment"]["future_baseline"]["model_seeds"]] != MODEL_SEEDS:
        raise ValueError("V3 future baseline seeds must be [0, 1, 2, 3, 4]")
    if list(v3_cfg["experiment"]["methods"]) != METHOD_ORDER:
        raise ValueError("V3 methods must match the seven-method order")
    for method_name, (sampling_method, batch_size) in EXPECTED_METHOD_MAPPING.items():
        method_cfg = v3_cfg["experiment"]["methods"][method_name]
        if method_cfg["method_name"] != method_name:
            raise ValueError(f"{method_name} method_name changed")
        if method_cfg["sampling_method"] != sampling_method:
            raise ValueError(f"{method_name} sampling_method changed")
        if int(method_cfg["batch_size"]) != batch_size:
            raise ValueError(f"{method_name} batch_size changed")
    if int(training_cfg["target_examples_processed"]) != 500000:
        raise ValueError("target_examples_processed must be 500000")
    if int(training_cfg["data_equivalent_passes"]) != 100:
        raise ValueError("data_equivalent_passes must be 100")
    if int(training_cfg["evaluation_every_examples"]) != 5000:
        raise ValueError("evaluation_every_examples must be 5000")


def validate_approved_lr(
    *,
    lr_decision_path: Path,
    accepted_baseline_manifest: dict[str, Any],
) -> dict[str, Any]:
    decision = load_json(lr_decision_path)
    if decision.get("human_approved") is not True:
        raise ValueError("Approved LR artifact is not human approved")
    learning_rate = float(decision.get("selected_common_learning_rate", float("nan")))
    if learning_rate != EXPECTED_APPROVED_LEARNING_RATE:
        raise ValueError("Approved LR must be 0.03")
    sha = file_sha256(lr_decision_path)
    if sha != accepted_baseline_manifest.get("learning_rate_decision_artifact_sha256"):
        raise ValueError("Approved LR artifact SHA does not match accepted d=6 baseline manifest")
    return {
        "path": lr_decision_path,
        "sha256": sha,
        "selected_common_learning_rate": learning_rate,
        "source_git_commit_hash": decision.get("source_git_commit_hash"),
    }


def build_run_specs(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    learning_rate: float,
    dimensions: list[int] | None = None,
    model_seeds: list[int] | None = None,
    target_examples_processed: int | None = None,
    evaluation_every_examples: int | None = None,
) -> list[dict[str, Any]]:
    validate_configs(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg)
    training_cfg = v3_cfg["experiment"]["training"]
    selected_dimensions = STRESS_DIMENSIONS if dimensions is None else [int(d) for d in dimensions]
    selected_model_seeds = MODEL_SEEDS if model_seeds is None else [int(seed) for seed in model_seeds]
    target_examples = int(training_cfg["target_examples_processed"]) if target_examples_processed is None else int(target_examples_processed)
    evaluation_every = int(training_cfg["evaluation_every_examples"]) if evaluation_every_examples is None else int(evaluation_every_examples)
    specs: list[dict[str, Any]] = []
    for dimension in selected_dimensions:
        for model_seed in selected_model_seeds:
            for method_name, method_cfg in method_items(v3_cfg):
                specs.append(
                    {
                        "dimension": int(dimension),
                        "n_relevant_features": N_RELEVANT_FEATURES,
                        "parameter_count": expected_parameter_count(dimension=int(dimension)),
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
    return build_run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        learning_rate=learning_rate,
        dimensions=STRESS_DIMENSIONS,
        model_seeds=[0],
        target_examples_processed=int(baseline_cfg["dataset"]["n_train"]),
        evaluation_every_examples=int(baseline_cfg["dataset"]["n_train"]),
    )


def plan(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    lr_decision: dict[str, Any],
) -> dict[str, Any]:
    specs = build_run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        learning_rate=float(lr_decision["selected_common_learning_rate"]),
    )
    return {
        "mode": "plan-only",
        "stress_dimensions": STRESS_DIMENSIONS,
        "reused_baseline_dimension": REUSED_BASELINE_DIMENSION,
        "n_relevant_features": N_RELEVANT_FEATURES,
        "methods": METHOD_ORDER,
        "model_seeds": MODEL_SEEDS,
        "expected_new_run_count": 70,
        "planned_run_count": len(specs),
        "run_specs": specs,
        "approved_learning_rate": lr_decision["selected_common_learning_rate"],
        "approved_lr_artifact_path": lr_decision["path"],
        "approved_lr_artifact_sha256": lr_decision["sha256"],
        "explicit_sampling_seed_table": sampling_seed_table(v3_cfg),
        "parameter_counts": parameter_counts(),
        "target_examples_processed": int(v3_cfg["experiment"]["training"]["target_examples_processed"]),
        "data_equivalent_passes": int(v3_cfg["experiment"]["training"]["data_equivalent_passes"]),
        "evaluation_every_examples": int(v3_cfg["experiment"]["training"]["evaluation_every_examples"]),
        "checkpoint_schedule": expected_checkpoint_examples(
            target_examples_processed=int(v3_cfg["experiment"]["training"]["target_examples_processed"]),
            evaluation_every_examples=int(v3_cfg["experiment"]["training"]["evaluation_every_examples"]),
        ),
        "output_namespace": OUTPUT_ROOT,
        "figures_namespace": FIGURE_ROOT,
    }


def import_training_stack() -> tuple[Any, Any, dict[str, Any]]:
    import torch

    from gradient_methods_nn_regression.metrics import (
        function_estimation_mse,
        noisy_prediction_mse,
        parameter_norm,
    )
    from gradient_methods_nn_regression.training import train_model

    return torch, train_model, {
        "function_estimation_mse": function_estimation_mse,
        "noisy_prediction_mse": noisy_prediction_mse,
        "parameter_norm": parameter_norm,
    }


def torch_dtype_from_name(dtype_name: str, *, torch_module: Any) -> Any:
    dtype_by_name = {"float32": torch_module.float32, "float64": torch_module.float64}
    try:
        return dtype_by_name[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported configured dtype: {dtype_name}") from exc


def make_model(*, torch_module: Any, input_dim: int, hidden_dim: int) -> Any:
    return torch_module.nn.Sequential(
        torch_module.nn.Linear(input_dim, hidden_dim),
        torch_module.nn.Tanh(),
        torch_module.nn.Linear(hidden_dim, 1),
    )


def count_trainable_parameters(model: Any) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


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


def generate_split_np(
    *,
    n_samples: int,
    dimension: int,
    noise_std: float,
    seed: int,
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    x, y, f_true, noise = generate_synthetic_regression_data(
        n_samples=n_samples,
        n_features=dimension,
        noise_std=noise_std,
        seed=seed,
        dtype=dtype,
    )
    return {"x": x, "y": y.reshape(-1, 1), "f_true": f_true.reshape(-1, 1), "noise": noise.reshape(-1, 1)}


def generate_dimension_data_np(
    *,
    dimension: int,
    baseline_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    ds_cfg = baseline_cfg["dataset"]
    dtype = np.dtype(ds_cfg["dtype"])
    split_cfg = {
        "train": ("n_train", "train"),
        "validation": ("n_validation", "validation"),
        "test": ("n_test", "test"),
    }
    data: dict[str, dict[str, Any]] = {}
    for split, (n_key, seed_key) in split_cfg.items():
        arrays = generate_split_np(
            n_samples=int(ds_cfg[n_key]),
            dimension=dimension,
            noise_std=float(ds_cfg["noise_std"]),
            seed=int(ds_cfg["seeds"][seed_key]),
            dtype=dtype,
        )
        data[split] = {
            "arrays": arrays,
            "sha256": arrays_sha256([arrays["x"], arrays["y"], arrays["f_true"], arrays["noise"]]),
            "n_samples": int(ds_cfg[n_key]),
            "seed": int(ds_cfg["seeds"][seed_key]),
            "dtype": str(dtype),
        }
    return data


def tensors_for_dimension_data(
    *,
    data_np: dict[str, dict[str, Any]],
    torch_module: Any,
    torch_dtype: Any,
    device: Any,
) -> dict[str, tuple[Any, Any, Any]]:
    tensors: dict[str, tuple[Any, Any, Any]] = {}
    for split, split_data in data_np.items():
        arrays = split_data["arrays"]
        tensors[split] = (
            torch_module.as_tensor(arrays["x"], dtype=torch_dtype, device=device),
            torch_module.as_tensor(arrays["y"], dtype=torch_dtype, device=device),
            torch_module.as_tensor(arrays["f_true"], dtype=torch_dtype, device=device),
        )
    return tensors


def data_provenance_for_dimension(
    *,
    dimension: int,
    baseline_cfg: dict[str, Any],
    data_np: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ds_cfg = baseline_cfg["dataset"]
    return {
        "dimension": dimension,
        "n_relevant_features": N_RELEVANT_FEATURES,
        "target_function": "true_regression_function uses the first six coordinates",
        "noise_std": float(ds_cfg["noise_std"]),
        "dtype": str(np.dtype(ds_cfg["dtype"])),
        "split_seeds": dict(ds_cfg["seeds"]),
        "split_hashes": {
            split: {
                "sha256": split_data["sha256"],
                "n_samples": split_data["n_samples"],
                "seed": split_data["seed"],
            }
            for split, split_data in data_np.items()
        },
    }


def final_split_metrics(
    *,
    model: Any,
    x: Any,
    y: Any,
    f_true: Any,
    torch_module: Any,
    metrics: dict[str, Any],
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    with torch_module.no_grad():
        predictions = model(x)
        prediction_mse = metrics["noisy_prediction_mse"](predictions, y).item()
        function_mse = metrics["function_estimation_mse"](predictions, f_true).item()
    model.train(was_training)
    return {"prediction_mse": float(prediction_mse), "function_mse": float(function_mse)}


def history_is_finite(history: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    for row in history:
        for field in FINITE_HISTORY_FIELDS:
            value = row.get(field)
            if value is None or not math.isfinite(float(value)):
                return False, [f"{field} became non-finite"]
        for field in ["update_gradient_norm", "batch_loss"]:
            value = row.get(field)
            if value is not None and not math.isfinite(float(value)):
                return False, [f"{field} became non-finite"]
    return True, []


def ensure_output_available(root: Path) -> None:
    manifest_path = root / "dimension_stress_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists; refusing to overwrite")


def ensure_run_dir_available(run_dir: Path) -> None:
    if (run_dir / "history.csv").exists() or (run_dir / "metadata.json").exists():
        raise FileExistsError(f"{run_dir} already contains stress output; refusing to overwrite")


def run_specs(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    accepted_baseline_manifest: dict[str, Any],
    lr_decision: dict[str, Any],
    specs: list[dict[str, Any]],
    output_root: Path,
    mode: str,
) -> dict[str, Any]:
    torch_module, train_model, metrics = import_training_stack()
    ensure_output_available(output_root)
    ds_cfg = baseline_cfg["dataset"]
    model_cfg = baseline_cfg["model"]
    opt_cfg = baseline_cfg["optimisation"]
    numpy_dtype = np.dtype(ds_cfg["dtype"])
    torch_dtype = torch_dtype_from_name(str(numpy_dtype), torch_module=torch_module)
    device = torch_module.device("cpu")
    hidden_dim = int(model_cfg["hidden_dim"])
    dimensions = sorted({int(spec["dimension"]) for spec in specs})
    source_commit = commit_hash()
    output_root.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()
    completed: list[dict[str, Any]] = []
    completed_seconds: list[float] = []
    data_provenance: dict[int, dict[str, Any]] = {}
    data_tensors: dict[int, dict[str, tuple[Any, Any, Any]]] = {}

    for dimension in dimensions:
        data_np = generate_dimension_data_np(dimension=dimension, baseline_cfg=baseline_cfg)
        data_provenance[dimension] = data_provenance_for_dimension(
            dimension=dimension,
            baseline_cfg=baseline_cfg,
            data_np=data_np,
        )
        data_tensors[dimension] = tensors_for_dimension_data(
            data_np=data_np,
            torch_module=torch_module,
            torch_dtype=torch_dtype,
            device=device,
        )

    references: dict[tuple[int, int], dict[str, Any]] = {}
    for dimension in dimensions:
        for model_seed in sorted({int(spec["model_seed"]) for spec in specs if int(spec["dimension"]) == dimension}):
            torch_module.manual_seed(model_seed)
            reference_model = make_model(
                torch_module=torch_module,
                input_dim=dimension,
                hidden_dim=hidden_dim,
            ).to(device=device, dtype=torch_dtype)
            constructed_count = count_trainable_parameters(reference_model)
            expected_count = expected_parameter_count(dimension=dimension, hidden_dim=hidden_dim)
            if constructed_count != expected_count:
                raise ValueError(f"Parameter count mismatch for d={dimension}: {constructed_count} != {expected_count}")
            reference_state = copy.deepcopy(reference_model.state_dict())
            checksum = state_checksum(reference_state)
            references[(dimension, model_seed)] = {
                "state_dict": reference_state,
                "checksum": checksum,
                "parameter_count": constructed_count,
            }
            write_json(
                {
                    "dimension": dimension,
                    "n_relevant_features": N_RELEVANT_FEATURES,
                    "model_seed": model_seed,
                    "parameter_count": constructed_count,
                    "initial_state_checksum": checksum,
                    "torch_dtype": torch_dtype,
                    "device": device,
                },
                output_root / f"dimension_{dimension}" / "initial_states" / f"model_seed_{model_seed}" / "metadata.json",
            )

    total_runs = len(specs)
    for run_number, spec in enumerate(specs, start=1):
        dimension = int(spec["dimension"])
        model_seed = int(spec["model_seed"])
        method_name = str(spec["method_name"])
        run_dir = output_root / f"dimension_{dimension}" / method_name / f"model_seed_{model_seed}"
        ensure_run_dir_available(run_dir)
        print(
            f"[{run_number:02d}/{total_runs}] START d={dimension} {method_name} "
            f"model_seed={model_seed} sampling_seed={spec['sampling_seed']}",
            flush=True,
        )
        run_start = time.perf_counter()
        model = make_model(
            torch_module=torch_module,
            input_dim=dimension,
            hidden_dim=hidden_dim,
        ).to(device=device, dtype=torch_dtype)
        reference = references[(dimension, model_seed)]
        model.load_state_dict(reference["state_dict"])
        loaded_checksum = state_checksum(model.state_dict())
        if loaded_checksum != reference["checksum"]:
            raise RuntimeError(f"Initial-state checksum mismatch for d={dimension}, seed={model_seed}, method={method_name}")
        optimiser = torch_module.optim.SGD(
            model.parameters(),
            lr=float(lr_decision["selected_common_learning_rate"]),
            momentum=float(opt_cfg["momentum"]),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
        tensors = data_tensors[dimension]
        history = train_model(
            model=model,
            optimiser=optimiser,
            loss_function=torch_module.nn.functional.mse_loss,
            training_data=(tensors["train"][0], tensors["train"][1]),
            evaluation_data=(tensors["validation"][0], tensors["validation"][1], tensors["validation"][2]),
            method=method_name,
            sampling_method=str(spec["sampling_method"]),
            batch_size=int(spec["batch_size"]),
            target_examples_processed=int(spec["target_examples_processed"]),
            sampling_seed=int(spec["sampling_seed"]),
            evaluation_every_examples=int(spec["evaluation_every_examples"]),
        )
        finite, reasons = history_is_finite(history)
        if not finite:
            raise RuntimeError(f"d={dimension}, {method_name}, seed={model_seed}: {reasons}")
        enriched_history = [
            {
                "dimension": dimension,
                "n_relevant_features": N_RELEVANT_FEATURES,
                "parameter_count": int(reference["parameter_count"]),
                **row,
            }
            for row in history
        ]
        train_metrics = final_split_metrics(
            model=model,
            x=tensors["train"][0],
            y=tensors["train"][1],
            f_true=tensors["train"][2],
            torch_module=torch_module,
            metrics=metrics,
        )
        validation_metrics = final_split_metrics(
            model=model,
            x=tensors["validation"][0],
            y=tensors["validation"][1],
            f_true=tensors["validation"][2],
            torch_module=torch_module,
            metrics=metrics,
        )
        test_metrics = final_split_metrics(
            model=model,
            x=tensors["test"][0],
            y=tensors["test"][1],
            f_true=tensors["test"][2],
            torch_module=torch_module,
            metrics=metrics,
        )
        final = history[-1]
        run_elapsed = time.perf_counter() - run_start
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
        write_history_csv(enriched_history, history_path)
        metadata = {
            "dimension": dimension,
            "n_relevant_features": N_RELEVANT_FEATURES,
            "parameter_count": int(reference["parameter_count"]),
            "method_name": method_name,
            "sampling_method": spec["sampling_method"],
            "nominal_batch_size": int(spec["batch_size"]),
            "model_seed": model_seed,
            "sampling_seed": int(spec["sampling_seed"]),
            "sampling_seed_source": "explicit V3 baseline method x model-seed table",
            "learning_rate": lr_decision["selected_common_learning_rate"],
            "approved_lr_artifact_path": lr_decision["path"],
            "approved_lr_artifact_sha256": lr_decision["sha256"],
            "initial_state_checksum": reference["checksum"],
            "loaded_initial_state_checksum": loaded_checksum,
            "target_examples_processed": int(spec["target_examples_processed"]),
            "actual_examples_processed": int(final["cumulative_examples_processed"]),
            "data_equivalent_passes": float(final["data_equivalent_passes"]),
            "evaluation_every_examples": int(spec["evaluation_every_examples"]),
            "dtype": numpy_dtype,
            "device": device,
            "optimiser": {
                "name": "SGD",
                "momentum": float(opt_cfg["momentum"]),
                "weight_decay": float(opt_cfg["weight_decay"]),
            },
            "data_generation_provenance": data_provenance[dimension],
            "runtime": {
                "run_elapsed_seconds": run_elapsed,
                "torch_version": torch_module.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "source_git_commit_hash": source_commit,
                "worktree_status": worktree_status(),
            },
            "final_step_count": int(final["step"]),
            "final_epoch": final["epoch"],
            "final_split_metrics": final_metrics,
            "artifacts": {
                "history_csv": history_path,
                "metadata_json": metadata_path,
            },
        }
        write_json(metadata, metadata_path)
        record = {
            "dimension": dimension,
            "method_name": method_name,
            "model_seed": model_seed,
            "sampling_seed": int(spec["sampling_seed"]),
            "parameter_count": int(reference["parameter_count"]),
            "metadata_json": metadata_path,
            "history_csv": history_path,
            "actual_examples_processed": int(final["cumulative_examples_processed"]),
            "data_equivalent_passes": float(final["data_equivalent_passes"]),
            "test_function_mse": final_metrics["test_function_mse"],
        }
        completed.append(record)
        completed_seconds.append(run_elapsed)
        elapsed = time.perf_counter() - start_all
        remaining = total_runs - run_number
        eta = remaining * (sum(completed_seconds) / len(completed_seconds))
        print(
            f"[{run_number:02d}/{total_runs}] DONE d={dimension} {method_name} "
            f"run={run_elapsed:.1f}s total_elapsed={elapsed / 60:.1f}min ETA={eta / 60:.1f}min",
            flush=True,
        )

    manifest = {
        "experiment_name": "week1_dimension_stress_v3",
        "mode": mode,
        "expected_new_run_count": len(specs),
        "actual_completed_count": len(completed),
        "dimensions_run": dimensions,
        "reused_baseline_dimension": REUSED_BASELINE_DIMENSION,
        "accepted_d6_baseline_manifest_path": ACCEPTED_BASELINE_MANIFEST_PATH,
        "accepted_d6_baseline_source_commit": accepted_baseline_manifest.get("source_git_commit_hash"),
        "accepted_lr_artifact_sha256": lr_decision["sha256"],
        "methods": METHOD_ORDER,
        "model_seeds": sorted({int(spec["model_seed"]) for spec in specs}),
        "explicit_sampling_seed_table": sampling_seed_table(v3_cfg),
        "parameter_counts": parameter_counts(),
        "data_generation_provenance": data_provenance,
        "target_examples_processed": specs[0]["target_examples_processed"] if specs else None,
        "data_equivalent_passes": (specs[0]["target_examples_processed"] / int(ds_cfg["n_train"])) if specs else None,
        "evaluation_every_examples": specs[0]["evaluation_every_examples"] if specs else None,
        "checkpoint_schedule": expected_checkpoint_examples(
            target_examples_processed=int(specs[0]["target_examples_processed"]),
            evaluation_every_examples=int(specs[0]["evaluation_every_examples"]),
        )
        if specs
        else [],
        "run_metadata_paths": [record["metadata_json"] for record in completed],
        "source_git_commit_hash": source_commit,
        "worktree_status": worktree_status(),
        "total_elapsed_seconds": time.perf_counter() - start_all,
        "runs": completed,
    }
    write_json(manifest, output_root / "dimension_stress_manifest.json")
    return manifest


def run_preflight(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    accepted_baseline_manifest: dict[str, Any],
    lr_decision: dict[str, Any],
) -> dict[str, Any]:
    specs = build_preflight_run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        learning_rate=float(lr_decision["selected_common_learning_rate"]),
    )
    output_root = OUTPUT_ROOT / "preflight"
    manifest = run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        accepted_baseline_manifest=accepted_baseline_manifest,
        lr_decision=lr_decision,
        specs=specs,
        output_root=output_root,
        mode="preflight",
    )
    failures: list[str] = []
    if manifest["actual_completed_count"] != 14:
        failures.append("preflight did not complete 14 runs")
    if manifest["dimensions_run"] != STRESS_DIMENSIONS:
        failures.append("preflight dimensions changed")
    checksums: dict[tuple[int, int], set[str]] = {}
    for path in manifest["run_metadata_paths"]:
        metadata = load_json(Path(path))
        key = (int(metadata["dimension"]), int(metadata["model_seed"]))
        checksums.setdefault(key, set()).add(metadata["initial_state_checksum"])
        if int(metadata["actual_examples_processed"]) != 5000:
            failures.append(f"{path}: final examples != 5000")
        if float(metadata["data_equivalent_passes"]) != 1.0:
            failures.append(f"{path}: DEP != 1")
        if metadata["approved_lr_artifact_sha256"] != lr_decision["sha256"]:
            failures.append(f"{path}: missing LR artifact SHA")
    for key, values in checksums.items():
        if len(values) != 1:
            failures.append(f"Initial-state checksum mismatch within {key}")
    if failures:
        manifest["preflight_ok"] = False
        manifest["preflight_failures"] = failures
        write_json(manifest, output_root / "dimension_stress_manifest.json")
        raise RuntimeError("V3 dimension stress preflight failed: " + "; ".join(failures))
    manifest["preflight_ok"] = True
    manifest["preflight_failures"] = []
    write_json(manifest, output_root / "dimension_stress_manifest.json")
    return manifest


def run_full(
    *,
    baseline_cfg: dict[str, Any],
    v3_cfg: dict[str, Any],
    accepted_baseline_manifest: dict[str, Any],
    lr_decision: dict[str, Any],
) -> dict[str, Any]:
    specs = build_run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        learning_rate=float(lr_decision["selected_common_learning_rate"]),
    )
    return run_specs(
        baseline_cfg=baseline_cfg,
        v3_cfg=v3_cfg,
        accepted_baseline_manifest=accepted_baseline_manifest,
        lr_decision=lr_decision,
        specs=specs,
        output_root=OUTPUT_ROOT,
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
    accepted_baseline_manifest = load_json(ACCEPTED_BASELINE_MANIFEST_PATH)
    validate_configs(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg)
    lr_decision = validate_approved_lr(
        lr_decision_path=LR_DECISION_PATH,
        accepted_baseline_manifest=accepted_baseline_manifest,
    )
    if args.plan_only:
        print(json.dumps(to_jsonable(plan(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg, lr_decision=lr_decision)), indent=2))
        return
    if args.preflight:
        print(json.dumps(to_jsonable(run_preflight(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg, accepted_baseline_manifest=accepted_baseline_manifest, lr_decision=lr_decision)), indent=2))
        return
    if args.full:
        print(json.dumps(to_jsonable(run_full(baseline_cfg=baseline_cfg, v3_cfg=v3_cfg, accepted_baseline_manifest=accepted_baseline_manifest, lr_decision=lr_decision)), indent=2))
        return


if __name__ == "__main__":
    main()
