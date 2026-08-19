from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np


BRANCH_CONFIG_PATH = Path("configs/experiments/sampling_law_nn_branch.json")
BASELINE_CONFIG_PATH = Path("configs/baseline.json")
V2_RAW_PREFIX = Path("results/raw/week1_gradient_methods")
V2_FIGURE_PREFIX = Path("results/figures/week1_gradient_methods")

HISTORY_FIELDNAMES = [
    "trajectory_id",
    "method",
    "sampling_method",
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_jsonable(value: Any) -> Any:
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
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if value.__class__.__name__ in {"dtype", "device"}:
        return str(value).replace("torch.", "")
    return str(value)


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


def method_items(branch_cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return list(branch_cfg["experiment"]["methods"].items())


def branch_raw_dir(branch_cfg: dict[str, Any]) -> Path:
    return Path(branch_cfg["experiment"]["paths"]["raw_dir"])


def branch_figures_dir(branch_cfg: dict[str, Any]) -> Path:
    return Path(branch_cfg["experiment"]["paths"]["figures_dir"])


def validate_branch_config(branch_cfg: dict[str, Any]) -> None:
    methods = branch_cfg["experiment"]["methods"]
    if set(methods) != {"wr_1", "rr_1"}:
        raise ValueError("branch config must contain exactly wr_1 and rr_1")

    expected = {
        "wr_1": "single_with_replacement",
        "rr_1": "random_reshuffling",
    }
    for method_name, sampling_method in expected.items():
        method_cfg = methods[method_name]
        if method_cfg["sampling_method"] != sampling_method:
            raise ValueError(f"{method_name} has incorrect sampling_method")
        if int(method_cfg["batch_size"]) != 1:
            raise ValueError(f"{method_name} must use batch_size=1")

    trajectory_ids = branch_cfg["experiment"]["sampling_seeds"]["trajectory_ids"]
    seed_map = branch_cfg["experiment"]["sampling_seeds"]["by_method"]
    for method_name in methods:
        if len(seed_map[method_name]) != len(trajectory_ids):
            raise ValueError(f"{method_name} seed count does not match trajectory_ids")

    raw_dir = branch_raw_dir(branch_cfg)
    figures_dir = branch_figures_dir(branch_cfg)
    if _path_has_prefix(raw_dir, V2_RAW_PREFIX):
        raise ValueError("branch raw_dir collides with V2 raw output namespace")
    if _path_has_prefix(figures_dir, V2_FIGURE_PREFIX):
        raise ValueError("branch figures_dir collides with V2 figure namespace")


def _path_has_prefix(path: Path, prefix: Path) -> bool:
    path_parts = path.parts
    prefix_parts = prefix.parts
    return path_parts[: len(prefix_parts)] == prefix_parts


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
    try:
        seed_index = trajectory_ids.index(int(trajectory_id))
    except ValueError as exc:
        raise ValueError(f"Unknown trajectory_id: {trajectory_id}") from exc
    return int(branch_cfg["experiment"]["sampling_seeds"]["by_method"][method_name][seed_index])


def build_run_specs(
    *,
    branch_cfg: dict[str, Any],
    trajectory_ids: list[int],
) -> list[dict[str, Any]]:
    training_cfg = branch_cfg["experiment"]["training"]
    specs: list[dict[str, Any]] = []
    for trajectory_id in trajectory_ids:
        for method_key, method_cfg in method_items(branch_cfg):
            method_name = str(method_cfg.get("method_name", method_key))
            specs.append(
                {
                    "trajectory_id": int(trajectory_id),
                    "method_name": method_name,
                    "sampling_method": str(method_cfg["sampling_method"]),
                    "batch_size": int(method_cfg["batch_size"]),
                    "sampling_seed": sampling_seed_for(
                        branch_cfg=branch_cfg,
                        method_name=method_key,
                        trajectory_id=int(trajectory_id),
                    ),
                    "target_examples_processed": int(
                        training_cfg["target_examples_processed"]
                    ),
                    "evaluation_every_examples": int(
                        training_cfg["evaluation_every_examples"]
                    ),
                    "learning_rate": float(training_cfg["learning_rate"]),
                    "model_seed": int(training_cfg["model_seed"]),
                }
            )
    return specs


def expected_checkpoint_examples(branch_cfg: dict[str, Any]) -> list[int]:
    training_cfg = branch_cfg["experiment"]["training"]
    target = int(training_cfg["target_examples_processed"])
    every = int(training_cfg["evaluation_every_examples"])
    checkpoints = [0, *range(every, target + 1, every)]
    if checkpoints[-1] != target:
        checkpoints.append(target)
    return checkpoints


def _torch_dtype(dtype_name: str, *, torch_module: Any) -> Any:
    dtype_by_name = {
        "float32": torch_module.float32,
        "float64": torch_module.float64,
    }
    try:
        return dtype_by_name[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported configured dtype for training: {dtype_name}") from exc


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
            raise ValueError(
                f"{path.name}:{name} has dtype {array.dtype}, expected {expected_dtype}"
            )

    return (
        torch_module.as_tensor(x_np, dtype=torch_dtype, device=device),
        torch_module.as_tensor(y_np, dtype=torch_dtype, device=device),
        torch_module.as_tensor(f_true_np, dtype=torch_dtype, device=device),
    )


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


def _is_finite_history(history: list[dict[str, Any]]) -> bool:
    numeric_fields = [
        "training_mse",
        "validation_mse",
        "validation_function_mse",
        "full_gradient_norm",
        "parameter_norm",
        "training_elapsed_seconds",
        "total_elapsed_seconds",
    ]
    for row in history:
        for field in numeric_fields:
            if not math.isfinite(float(row[field])):
                return False
        for optional_field in ["batch_loss", "update_gradient_norm"]:
            value = row[optional_field]
            if value is not None and not math.isfinite(float(value)):
                return False
    return True


def run_branch(
    *,
    baseline_cfg: dict[str, Any],
    branch_cfg: dict[str, Any],
    trajectory_ids: list[int],
    write_outputs: bool = True,
) -> dict[str, Any]:
    validate_branch_config(branch_cfg)

    import torch

    from gradient_methods_nn_regression.model import TinyRegressionModel
    from gradient_methods_nn_regression.training import train_model

    ds_cfg = baseline_cfg["dataset"]
    configured_dtype = np.dtype(ds_cfg["dtype"])
    torch_dtype = _torch_dtype(str(configured_dtype), torch_module=torch)
    device = torch.device("cpu")
    generated_data_dir = Path(baseline_cfg["paths"]["generated_data_dir"])

    x_train, y_train, _ = load_split(
        generated_data_dir / "baseline_train.npz",
        expected_dtype=configured_dtype,
        torch_dtype=torch_dtype,
        device=device,
        torch_module=torch,
    )
    x_test, y_test, f_test = load_split(
        generated_data_dir / "baseline_test.npz",
        expected_dtype=configured_dtype,
        torch_dtype=torch_dtype,
        device=device,
        torch_module=torch,
    )

    training_cfg = branch_cfg["experiment"]["training"]
    target_examples = int(training_cfg["target_examples_processed"])
    evaluation_every_examples = int(training_cfg["evaluation_every_examples"])
    expected_checkpoints = expected_checkpoint_examples(branch_cfg)

    torch.manual_seed(int(training_cfg["model_seed"]))
    reference_model = TinyRegressionModel().to(device=device, dtype=torch_dtype)
    reference_state = copy.deepcopy(reference_model.state_dict())
    initial_checksum = state_checksum(reference_state)

    run_records: list[dict[str, Any]] = []
    branch_start = time.perf_counter()
    output_root = branch_raw_dir(branch_cfg)
    if write_outputs:
        output_root.mkdir(parents=True, exist_ok=True)

    specs = build_run_specs(branch_cfg=branch_cfg, trajectory_ids=trajectory_ids)
    total_runs = len(specs)
    completed_run_seconds: list[float] = []

    for run_index, spec in enumerate(specs, start=1):
        print(_format_progress_start(run_index, total_runs, spec), flush=True)
        progress_run_start = time.perf_counter()
        model = TinyRegressionModel().to(device=device, dtype=torch_dtype)
        model.load_state_dict(reference_state)
        loaded_checksum = state_checksum(model.state_dict())
        if loaded_checksum != initial_checksum:
            raise RuntimeError("initial-state checksum mismatch")

        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=float(spec["learning_rate"]),
            momentum=float(baseline_cfg["optimisation"]["momentum"]),
            weight_decay=float(baseline_cfg["optimisation"]["weight_decay"]),
        )

        run_start = time.perf_counter()
        history = train_model(
            model=model,
            optimiser=optimiser,
            loss_function=torch.nn.functional.mse_loss,
            training_data=(x_train, y_train),
            evaluation_data=(x_test, y_test, f_test),
            method=spec["method_name"],
            sampling_method=spec["sampling_method"],
            batch_size=spec["batch_size"],
            target_examples_processed=target_examples,
            sampling_seed=spec["sampling_seed"],
            evaluation_every_examples=evaluation_every_examples,
        )
        run_elapsed_seconds = time.perf_counter() - run_start
        enriched_history = [
            {
                "trajectory_id": spec["trajectory_id"],
                "sampling_seed": spec["sampling_seed"],
                **row,
            }
            for row in history
        ]

        checkpoint_examples = [
            int(row["checkpoint_examples"]) for row in enriched_history
        ]
        if checkpoint_examples != expected_checkpoints:
            raise RuntimeError(
                f"Unexpected checkpoint schedule for {spec['method_name']}: "
                f"{checkpoint_examples}"
            )
        if not _is_finite_history(enriched_history):
            raise RuntimeError(f"Non-finite metric in {spec['method_name']}")

        run_dir = (
            output_root
            / spec["method_name"]
            / f"trajectory_{spec['trajectory_id']}"
        )
        history_path = run_dir / "history.csv"
        metadata_path = run_dir / "metadata.json"
        metadata = {
            **spec,
            "initial_state_checksum": initial_checksum,
            "loaded_initial_state_checksum": loaded_checksum,
            "data_paths": {
                "training": generated_data_dir / "baseline_train.npz",
                "evaluation": generated_data_dir / "baseline_test.npz",
            },
            "dataset_config_snapshot": ds_cfg,
            "branch_config_snapshot": branch_cfg,
            "baseline_optimisation_snapshot": baseline_cfg["optimisation"],
            "checkpoint_examples": checkpoint_examples,
            "elapsed_time": {
                "training_elapsed_seconds": float(
                    enriched_history[-1]["training_elapsed_seconds"]
                ),
                "total_elapsed_seconds": float(
                    enriched_history[-1]["total_elapsed_seconds"]
                ),
                "run_elapsed_seconds": float(run_elapsed_seconds),
            },
            "final_metrics": {
                "training_mse": float(enriched_history[-1]["training_mse"]),
                "evaluation_noisy_mse": float(enriched_history[-1]["validation_mse"]),
                "evaluation_function_mse": float(
                    enriched_history[-1]["validation_function_mse"]
                ),
                "full_gradient_norm": float(enriched_history[-1]["full_gradient_norm"]),
            },
            "runtime": {
                "device": device,
                "torch_dtype": torch_dtype,
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "artifacts": {
                "history_csv": history_path,
                "metadata_json": metadata_path,
            },
        }

        if write_outputs:
            write_history_csv(enriched_history, history_path)
            write_json(metadata, metadata_path)

        progress_run_seconds = time.perf_counter() - progress_run_start
        completed_run_seconds.append(progress_run_seconds)
        remaining_runs = total_runs - run_index
        average_completed_run_seconds = sum(completed_run_seconds) / len(
            completed_run_seconds
        )
        print(
            _format_progress_done(
                run_index=run_index,
                total_runs=total_runs,
                spec=spec,
                run_seconds=progress_run_seconds,
                total_elapsed_seconds=time.perf_counter() - branch_start,
                eta_seconds=average_completed_run_seconds * remaining_runs,
            ),
            flush=True,
        )

        run_records.append(
            {
                "trajectory_id": spec["trajectory_id"],
                "method_name": spec["method_name"],
                "sampling_method": spec["sampling_method"],
                "sampling_seed": spec["sampling_seed"],
                "history_csv": history_path,
                "metadata_json": metadata_path,
                "checkpoint_count": len(enriched_history),
                "final_step": int(enriched_history[-1]["step"]),
                "final_examples_processed": int(
                    enriched_history[-1]["cumulative_examples_processed"]
                ),
                "final_data_equivalent_passes": float(
                    enriched_history[-1]["data_equivalent_passes"]
                ),
                "training_elapsed_seconds": float(
                    enriched_history[-1]["training_elapsed_seconds"]
                ),
                "run_elapsed_seconds": float(run_elapsed_seconds),
                "evaluation_function_mse": float(
                    enriched_history[-1]["validation_function_mse"]
                ),
                "full_gradient_norm": float(enriched_history[-1]["full_gradient_norm"]),
                "finite_metrics": True,
            }
        )

    elapsed_seconds = time.perf_counter() - branch_start
    projected_30_per_method = _project_runtime_30_per_method(run_records)
    manifest = {
        "experiment_name": branch_cfg["experiment"]["name"],
        "mode": "preflight" if len(trajectory_ids) < 30 else "full",
        "trajectory_ids": trajectory_ids,
        "run_count": len(run_records),
        "methods": [name for name, _ in method_items(branch_cfg)],
        "target_examples_processed": target_examples,
        "evaluation_every_examples": evaluation_every_examples,
        "expected_checkpoint_examples": expected_checkpoints,
        "output_root": output_root,
        "elapsed_seconds": elapsed_seconds,
        "projected_runtime_30_per_method_seconds": projected_30_per_method,
        "runs": run_records,
    }
    if write_outputs:
        manifest_name = (
            "preflight_manifest.json"
            if manifest["mode"] == "preflight"
            else "branch_manifest.json"
        )
        write_json(manifest, output_root / manifest_name)
    return manifest


def _format_progress_start(
    run_index: int,
    total_runs: int,
    spec: dict[str, Any],
) -> str:
    return (
        f"[{run_index}/{total_runs}] START {spec['method_name']} "
        f"trajectory={spec['trajectory_id']} seed={spec['sampling_seed']}"
    )


def _format_progress_done(
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
        f"trajectory={spec['trajectory_id']}\n"
        f"run={run_seconds:.1f}s\n"
        f"total_elapsed={total_elapsed_seconds / 60:.1f}min\n"
        f"ETA={eta_seconds / 60:.1f}min"
    )


def _project_runtime_30_per_method(run_records: list[dict[str, Any]]) -> dict[str, float]:
    projections: dict[str, float] = {}
    for method_name in sorted({row["method_name"] for row in run_records}):
        values = [
            float(row["run_elapsed_seconds"])
            for row in run_records
            if row["method_name"] == method_name
        ]
        if values:
            projections[method_name] = float(sum(values) / len(values) * 30)
    return projections


def preflight_plan(
    *,
    baseline_cfg: dict[str, Any],
    branch_cfg: dict[str, Any],
) -> dict[str, Any]:
    validate_branch_config(branch_cfg)
    specs = build_run_specs(
        branch_cfg=branch_cfg,
        trajectory_ids=[
            int(value)
            for value in branch_cfg["experiment"]["sampling_seeds"]["smoke_trajectory_ids"]
        ],
    )
    return {
        "ok": True,
        "baseline_data": {
            "training": Path(baseline_cfg["paths"]["generated_data_dir"])
            / "baseline_train.npz",
            "evaluation": Path(baseline_cfg["paths"]["generated_data_dir"])
            / "baseline_test.npz",
            "n_train": int(baseline_cfg["dataset"]["n_train"]),
            "n_test": int(baseline_cfg["dataset"]["n_test"]),
            "dtype": baseline_cfg["dataset"]["dtype"],
        },
        "model": baseline_cfg["model"],
        "optimisation": {
            "learning_rate": branch_cfg["experiment"]["training"]["learning_rate"],
            "model_seed": branch_cfg["experiment"]["training"]["model_seed"],
            "momentum": baseline_cfg["optimisation"]["momentum"],
            "weight_decay": baseline_cfg["optimisation"]["weight_decay"],
        },
        "checkpoint_examples": expected_checkpoint_examples(branch_cfg),
        "smoke_run_specs": specs,
        "output_root": branch_raw_dir(branch_cfg),
        "figures_root": branch_figures_dir(branch_cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and print the branch run plan without training.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run the configured tiny preflight subset under the full branch budget.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all configured sampling-law branch trajectories.",
    )
    args = parser.parse_args()

    baseline_cfg = load_json(BASELINE_CONFIG_PATH)
    branch_cfg = load_json(BRANCH_CONFIG_PATH)

    selected_modes = sum([args.plan_only, args.preflight, args.full])
    if selected_modes != 1:
        raise SystemExit("Choose exactly one of --plan-only, --preflight, or --full.")

    if args.plan_only:
        print(
            json.dumps(
                to_jsonable(
                    preflight_plan(baseline_cfg=baseline_cfg, branch_cfg=branch_cfg)
                ),
                indent=2,
            )
        )
        return

    if args.preflight:
        trajectory_ids = [
            int(value)
            for value in branch_cfg["experiment"]["sampling_seeds"]["smoke_trajectory_ids"]
        ]
        manifest = run_branch(
            baseline_cfg=baseline_cfg,
            branch_cfg=branch_cfg,
            trajectory_ids=trajectory_ids,
            write_outputs=True,
        )
        print(json.dumps(to_jsonable(manifest), indent=2))
        return

    if args.full:
        trajectory_ids = [
            int(value)
            for value in branch_cfg["experiment"]["sampling_seeds"]["trajectory_ids"]
        ]
        manifest = run_branch(
            baseline_cfg=baseline_cfg,
            branch_cfg=branch_cfg,
            trajectory_ids=trajectory_ids,
            write_outputs=True,
        )
        print(json.dumps(to_jsonable(manifest), indent=2))
        return


if __name__ == "__main__":
    main()
