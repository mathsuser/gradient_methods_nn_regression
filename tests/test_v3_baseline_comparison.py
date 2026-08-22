from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    module_path = Path("experiments/11_v3_baseline_comparison.py")
    spec = importlib.util.spec_from_file_location("v3_baseline_comparison", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3b = _load_module()


def _baseline_cfg() -> dict:
    return v3b.load_json(v3b.BASELINE_CONFIG_PATH)


def _v3_cfg() -> dict:
    return v3b.load_json(v3b.V3_CONFIG_PATH)


def _approved_artifact(tmp_path: Path, *, approved: bool = True, lr: float = 0.03) -> Path:
    path = tmp_path / "learning_rate_selection.json"
    path.write_text(
        json.dumps(
            {
                "human_approved": approved,
                "selected_common_learning_rate": lr,
                "source_git_commit_hash": "abc123",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_baseline_plan_contains_exactly_35_runs() -> None:
    baseline_cfg = _baseline_cfg()
    cfg = _v3_cfg()
    lr_decision = v3b.validate_approved_lr_decision(v3b.lr_decision_artifact_path(cfg))
    plan = v3b.plan(baseline_cfg=baseline_cfg, v3_cfg=cfg, lr_decision=lr_decision)

    assert plan["expected_run_count"] == 35
    assert plan["planned_run_count"] == 35
    assert len(plan["run_specs"]) == 35


def test_baseline_plan_is_exactly_seven_methods_by_five_model_seeds() -> None:
    specs = v3b.build_baseline_run_specs(
        baseline_cfg=_baseline_cfg(),
        v3_cfg=_v3_cfg(),
        learning_rate=0.03,
    )

    assert {spec["method_name"] for spec in specs} == set(v3b.METHOD_ORDER)
    assert {spec["model_seed"] for spec in specs} == {0, 1, 2, 3, 4}
    assert len({(spec["method_name"], spec["model_seed"]) for spec in specs}) == 35


def test_approved_artifact_is_required(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        v3b.validate_approved_lr_decision(tmp_path / "missing.json")


def test_human_approved_false_fails_closed(tmp_path: Path) -> None:
    path = _approved_artifact(tmp_path, approved=False)

    with pytest.raises(ValueError, match="not human approved"):
        v3b.validate_approved_lr_decision(path)


def test_learning_rate_is_loaded_from_artifact(tmp_path: Path) -> None:
    path = _approved_artifact(tmp_path, approved=True, lr=0.03)

    decision = v3b.validate_approved_lr_decision(path)

    assert decision["selected_common_learning_rate"] == pytest.approx(0.03)


def test_non_locked_learning_rate_fails(tmp_path: Path) -> None:
    path = _approved_artifact(tmp_path, approved=True, lr=0.01)

    with pytest.raises(ValueError, match="Approved learning rate"):
        v3b.validate_approved_lr_decision(path)


def test_sha256_uses_exact_artifact_bytes(tmp_path: Path) -> None:
    path = _approved_artifact(tmp_path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    assert v3b.file_sha256(path) == expected
    assert v3b.validate_approved_lr_decision(path)["sha256"] == expected


def test_manifest_and_run_metadata_share_artifact_hash(tmp_path: Path) -> None:
    artifact = _approved_artifact(tmp_path)
    decision = v3b.validate_approved_lr_decision(artifact)
    metadata = {
        "learning_rate_decision_artifact_sha256": decision["sha256"],
    }
    manifest = {
        "learning_rate_decision_artifact_sha256": decision["sha256"],
        "runs": [metadata],
    }

    assert manifest["learning_rate_decision_artifact_sha256"] == metadata[
        "learning_rate_decision_artifact_sha256"
    ]


def test_exact_explicit_baseline_seed_mapping() -> None:
    cfg = _v3_cfg()

    assert v3b.sampling_seed_table(cfg) == {
        "full_batch_gd": {"0": 50000, "1": 50001, "2": 50002, "3": 50003, "4": 50004},
        "wr_b1": {"0": 51000, "1": 51001, "2": 51002, "3": 51003, "4": 51004},
        "wr_b32": {"0": 52000, "1": 52001, "2": 52002, "3": 52003, "4": 52004},
        "wr_b256": {"0": 53000, "1": 53001, "2": 53002, "3": 53003, "4": 53004},
        "rr_b1": {"0": 61000, "1": 61001, "2": 61002, "3": 61003, "4": 61004},
        "rr_b32": {"0": 62000, "1": 62001, "2": 62002, "3": 62003, "4": 62004},
        "rr_b256": {"0": 63000, "1": 63001, "2": 63002, "3": 63003, "4": 63004},
    }


def test_method_reordering_does_not_change_seed_lookup() -> None:
    cfg = _v3_cfg()
    reordered = copy.deepcopy(cfg)
    methods = reordered["experiment"]["methods"]
    reordered["experiment"]["methods"] = {
        key: methods[key]
        for key in reversed(list(methods))
    }

    assert v3b.future_baseline_seed_for(
        reordered,
        method_name="wr_b256",
        model_seed=3,
    ) == 53003
    assert v3b.future_baseline_seed_for(
        reordered,
        method_name="rr_b1",
        model_seed=0,
    ) == 61000


def test_one_model_seed_shares_reference_initialisation_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from gradient_methods_nn_regression.model import TinyRegressionModel

    references = v3b.prepare_reference_states(
        model_seeds=[0, 1],
        TinyRegressionModel=TinyRegressionModel,
        torch_module=torch,
        torch_dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert references[0]["checksum"] != references[1]["checksum"]

    loaded_checksums = []
    for _ in v3b.METHOD_ORDER:
        model = TinyRegressionModel().to(device=torch.device("cpu"), dtype=torch.float64)
        model.load_state_dict(references[0]["state_dict"])
        loaded_checksums.append(v3b.state_checksum(model.state_dict()))

    assert set(loaded_checksums) == {references[0]["checksum"]}


def test_baseline_target_and_dep_are_locked() -> None:
    cfg = _v3_cfg()
    training = cfg["experiment"]["training"]

    assert int(training["target_examples_processed"]) == 500000
    assert int(training["data_equivalent_passes"]) == 100


def test_checkpoint_cadence_is_locked() -> None:
    training = _v3_cfg()["experiment"]["training"]

    assert int(training["evaluation_every_examples"]) == 5000


def test_full_plan_never_uses_pilot_sampling_seeds() -> None:
    cfg = _v3_cfg()
    pilot_seeds = set(int(seed) for seed in cfg["experiment"]["pilot"]["sampling_seeds"].values())
    specs = v3b.build_baseline_run_specs(
        baseline_cfg=_baseline_cfg(),
        v3_cfg=cfg,
        learning_rate=0.03,
    )

    assert {int(spec["sampling_seed"]) for spec in specs}.isdisjoint(pilot_seeds)


def test_outputs_are_separate_from_other_experiments() -> None:
    cfg = _v3_cfg()

    assert v3b.baseline_output_root(cfg) == Path(
        "results/raw/week1_gradient_methods_v3/baseline_comparison_runs"
    )
    assert v3b.preflight_output_root(cfg) == Path(
        "results/raw/week1_gradient_methods_v3/baseline_preflight"
    )
    assert not str(v3b.baseline_output_root(cfg)).startswith(
        "results/raw/week1_gradient_methods/"
    )
    assert "sampling_law_branch" not in str(v3b.baseline_output_root(cfg))
    assert "dimension_stress" not in str(v3b.baseline_output_root(cfg))
    assert "learning_rate_pilot_histories" not in str(v3b.baseline_output_root(cfg))


def test_no_test_depends_on_checked_out_generated_data() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_path = "/".join(["data", "generated", "baseline_train.npz"])

    assert forbidden_path not in source


def test_v3_baseline_script_does_not_use_method_index_seed_formula() -> None:
    source = Path("experiments/11_v3_baseline_comparison.py").read_text(
        encoding="utf-8"
    )

    assert "method_index" not in source
    assert "sampling_seed_offset" not in source
