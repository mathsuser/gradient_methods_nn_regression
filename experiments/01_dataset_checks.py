from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gradient_methods_nn_regression.data import generate_synthetic_regression_data


def _dataset_summary(name: str, x: np.ndarray, y: np.ndarray, f_true: np.ndarray, noise: np.ndarray) -> dict:
    residual = (y.astype(np.float64) - f_true.astype(np.float64)) - noise.astype(np.float64)
    return {
        "name": name,
        "shape": {
            "x": list(x.shape),
            "y": list(y.shape),
            "f_true": list(f_true.shape),
            "noise": list(noise.shape),
        },
        "dtype": {
            "x": str(x.dtype),
            "y": str(y.dtype),
            "f_true": str(f_true.dtype),
            "noise": str(noise.dtype),
        },
        "feature_mean": x.mean(axis=0).tolist(),
        "feature_std": x.std(axis=0, ddof=0).tolist(),
        "target": {
            "y_mean": float(y.mean()),
            "y_std": float(y.std(ddof=0)),
            "f_true_mean": float(f_true.mean()),
            "f_true_std": float(f_true.std(ddof=0)),
        },
        "noise": {
            "mean": float(noise.mean()),
            "std": float(noise.std(ddof=0)),
        },
        "decomposition_check": {
            "max_abs_error": float(np.max(np.abs(residual))),
            "mean_abs_error": float(np.mean(np.abs(residual))),
        },
    }


def _save_split_diagnostics(
    split: str,
    x: np.ndarray,
    y: np.ndarray,
    f_true: np.ndarray,
    noise: np.ndarray,
    fig_dir: Path,
) -> None:
    # Figure 1: target/noiseless/noise distributions + Y vs f*(X) scatter.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(y, bins=60, alpha=0.85)
    axes[0, 0].set_title(f"{split}: noisy target Y")
    axes[0, 0].set_xlabel("Y")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(f_true, bins=60, alpha=0.85)
    axes[0, 1].set_title(f"{split}: noiseless target f*(X)")
    axes[0, 1].set_xlabel("f*(X)")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].hist(noise, bins=60, alpha=0.85)
    axes[1, 0].set_title(f"{split}: realized noise")
    axes[1, 0].set_xlabel("noise")
    axes[1, 0].set_ylabel("count")

    idx = np.arange(y.shape[0])
    if y.shape[0] > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(y.shape[0], size=5000, replace=False)
    axes[1, 1].scatter(f_true[idx], y[idx], s=8, alpha=0.3)
    axes[1, 1].set_title(f"{split}: Y vs f*(X)")
    axes[1, 1].set_xlabel("f*(X)")
    axes[1, 1].set_ylabel("Y")

    fig.tight_layout()
    fig.savefig(fig_dir / f"baseline_{split}_distributions_scatter.png", dpi=150)
    plt.close(fig)

    # Figure 2: feature means/stds.
    feature_idx = np.arange(x.shape[1])
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(feature_idx, means)
    axes[0].set_title(f"{split}: feature means")
    axes[0].set_xlabel("feature index")
    axes[0].set_ylabel("mean")

    axes[1].bar(feature_idx, stds)
    axes[1].set_title(f"{split}: feature std")
    axes[1].set_xlabel("feature index")
    axes[1].set_ylabel("std")

    fig.tight_layout()
    fig.savefig(fig_dir / f"baseline_{split}_feature_summary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg_path = Path("configs/baseline.json")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ds_cfg = cfg["dataset"]
    out_dir = Path(cfg["paths"]["generated_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(cfg["paths"]["results_figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.dtype(ds_cfg["dtype"])
    noise_std = float(ds_cfg["noise_std"])
    n_features = int(ds_cfg["n_features"])

    specs = [
        ("train", int(ds_cfg["n_train"]), int(ds_cfg["seeds"]["train"]), "baseline_train.npz"),
        ("validation", int(ds_cfg["n_validation"]), int(ds_cfg["seeds"]["validation"]), "baseline_validation.npz"),
        ("test", int(ds_cfg["n_test"]), int(ds_cfg["seeds"]["test"]), "baseline_test.npz"),
    ]

    manifest = {
        "config": {
            "dataset": ds_cfg,
            "paths": cfg["paths"],
        },
        "datasets": {},
    }

    for split, n_samples, seed, filename in specs:
        x, y, f_true, noise = generate_synthetic_regression_data(
            n_samples=n_samples,
            n_features=n_features,
            noise_std=noise_std,
            seed=seed,
            dtype=dtype,
        )

        # Sanity check the decomposition relation with dtype-aware tolerance.
        eps = np.finfo(dtype).eps
        scale = max(1.0, float(np.max(np.abs(y))))
        atol = 16.0 * eps * scale
        np.testing.assert_allclose(
            (y.astype(np.float64) - f_true.astype(np.float64)) - noise.astype(np.float64),
            0.0,
            rtol=0.0,
            atol=atol,
        )

        np.savez_compressed(out_dir / filename, x=x, y=y, f_true=f_true, noise=noise)
        manifest["datasets"][split] = _dataset_summary(split, x, y, f_true, noise)
        _save_split_diagnostics(split, x, y, f_true, noise, fig_dir)

    manifest_path = out_dir / "baseline_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote datasets to: {out_dir}")
    for _, _, _, filename in specs:
        print(f"- {filename}")
    print(f"- {manifest_path.name}")
    print(f"Wrote figures to: {fig_dir}")
    for split, _, _, _ in specs:
        print(f"- baseline_{split}_distributions_scatter.png")
        print(f"- baseline_{split}_feature_summary.png")

    print("Decomposition checks (|Y - f_true - noise|):")
    for split in ["train", "validation", "test"]:
        dec = manifest["datasets"][split]["decomposition_check"]
        print(
            f"- {split}: max_abs_error={dec['max_abs_error']:.3e}, "
            f"mean_abs_error={dec['mean_abs_error']:.3e}"
        )


if __name__ == "__main__":
    main()
