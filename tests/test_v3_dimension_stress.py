from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    module_path = Path("experiments/13_v3_dimension_stress.py")
    spec = importlib.util.spec_from_file_location("v3_dimension_stress", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3s = _load_module()


def _baseline_cfg() -> dict:
    return v3s.load_json(v3s.BASELINE_CONFIG_PATH)


def _v3_cfg() -> dict:
    return v3s.load_json(v3s.V3_CONFIG_PATH)


def _accepted_manifest(tmp_path: Path, *, lr_sha: str) -> dict:
    return {
        "learning_rate_decision_artifact_sha256": lr_sha,
        "source_git_commit_hash": "abc123",
    }


def _lr_artifact(tmp_path: Path, *, approved: bool = True, lr: float = 0.03) -> Path:
    path = tmp_path / "learning_rate_selection.json"
    path.write_text(
        json.dumps(
            {
                "human_approved": approved,
                "selected_common_learning_rate": lr,
                "source_git_commit_hash": "lr-source",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_stress_dimensions_and_reused_baseline_dimension_are_locked() -> None:
    assert v3s.STRESS_DIMENSIONS == [20, 100]
    assert v3s.REUSED_BASELINE_DIMENSION == 6


def test_n_relevant_features_remains_six() -> None:
    assert v3s.N_RELEVANT_FEATURES == 6
    assert int(_baseline_cfg()["dataset"]["n_relevant_features"]) == 6


def test_exactly_seven_methods_and_five_model_seeds() -> None:
    cfg = _v3_cfg()

    assert list(cfg["experiment"]["methods"]) == v3s.METHOD_ORDER
    assert len(v3s.METHOD_ORDER) == 7
    assert v3s.MODEL_SEEDS == [0, 1, 2, 3, 4]


def test_full_plan_has_exactly_70_new_runs(tmp_path: Path) -> None:
    artifact = _lr_artifact(tmp_path)
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    decision = v3s.validate_approved_lr(
        lr_decision_path=artifact,
        accepted_baseline_manifest=_accepted_manifest(tmp_path, lr_sha=sha),
    )

    plan = v3s.plan(baseline_cfg=_baseline_cfg(), v3_cfg=_v3_cfg(), lr_decision=decision)

    assert plan["expected_new_run_count"] == 70
    assert plan["planned_run_count"] == 70
    assert len(plan["run_specs"]) == 70


def test_preflight_plan_has_exactly_14_runs() -> None:
    specs = v3s.build_preflight_run_specs(
        baseline_cfg=_baseline_cfg(),
        v3_cfg=_v3_cfg(),
        learning_rate=0.03,
    )

    assert len(specs) == 14
    assert {spec["dimension"] for spec in specs} == {20, 100}
    assert {spec["model_seed"] for spec in specs} == {0}
    assert {spec["target_examples_processed"] for spec in specs} == {5000}


def test_approved_learning_rate_is_required(tmp_path: Path) -> None:
    artifact = _lr_artifact(tmp_path, approved=False)
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="not human approved"):
        v3s.validate_approved_lr(
            lr_decision_path=artifact,
            accepted_baseline_manifest=_accepted_manifest(tmp_path, lr_sha=sha),
        )


def test_lr_artifact_sha_must_match_accepted_d6_manifest(tmp_path: Path) -> None:
    artifact = _lr_artifact(tmp_path)

    with pytest.raises(ValueError, match="SHA"):
        v3s.validate_approved_lr(
            lr_decision_path=artifact,
            accepted_baseline_manifest=_accepted_manifest(tmp_path, lr_sha="bad"),
        )


def test_explicit_sampling_seed_mapping_is_correct() -> None:
    assert v3s.sampling_seed_table(_v3_cfg()) == {
        "full_batch_gd": {"0": 50000, "1": 50001, "2": 50002, "3": 50003, "4": 50004},
        "wr_b1": {"0": 51000, "1": 51001, "2": 51002, "3": 51003, "4": 51004},
        "wr_b32": {"0": 52000, "1": 52001, "2": 52002, "3": 52003, "4": 52004},
        "wr_b256": {"0": 53000, "1": 53001, "2": 53002, "3": 53003, "4": 53004},
        "rr_b1": {"0": 61000, "1": 61001, "2": 61002, "3": 61003, "4": 61004},
        "rr_b32": {"0": 62000, "1": 62001, "2": 62002, "3": 62003, "4": 62004},
        "rr_b256": {"0": 63000, "1": 63001, "2": 63002, "3": 63003, "4": 63004},
    }


def test_method_reordering_cannot_change_seed_lookup() -> None:
    cfg = _v3_cfg()
    reordered = copy.deepcopy(cfg)
    methods = reordered["experiment"]["methods"]
    reordered["experiment"]["methods"] = {
        key: methods[key]
        for key in reversed(list(methods))
    }

    assert v3s.future_baseline_seed_for(reordered, method_name="wr_b256", model_seed=3) == 53003
    assert v3s.future_baseline_seed_for(reordered, method_name="rr_b1", model_seed=0) == 61000


def test_same_method_and_model_seed_uses_same_sampling_seed_across_stress_dimensions() -> None:
    specs = v3s.build_run_specs(
        baseline_cfg=_baseline_cfg(),
        v3_cfg=_v3_cfg(),
        learning_rate=0.03,
    )
    seeds = {
        spec["dimension"]: spec["sampling_seed"]
        for spec in specs
        if spec["method_name"] == "rr_b32" and spec["model_seed"] == 4
    }

    assert seeds == {20: 62004, 100: 62004}


def test_expected_parameter_counts_are_verified() -> None:
    assert v3s.parameter_counts() == {6: 129, 20: 353, 100: 1633}
    assert v3s.expected_parameter_count(dimension=20) == 353
    assert v3s.expected_parameter_count(dimension=100) == 1633


def test_within_dimension_model_seed_initial_state_is_shared_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    model_seed = 0
    dimension = 20
    checksums = []
    torch.manual_seed(model_seed)
    reference = v3s.make_model(
        torch_module=torch,
        input_dim=dimension,
        hidden_dim=16,
    ).to(device=torch.device("cpu"), dtype=torch.float64)
    state = copy.deepcopy(reference.state_dict())
    expected = v3s.state_checksum(state)
    for _ in v3s.METHOD_ORDER:
        model = v3s.make_model(
            torch_module=torch,
            input_dim=dimension,
            hidden_dim=16,
        ).to(device=torch.device("cpu"), dtype=torch.float64)
        model.load_state_dict(state)
        checksums.append(v3s.state_checksum(model.state_dict()))

    assert set(checksums) == {expected}


def test_training_budget_is_locked() -> None:
    training = _v3_cfg()["experiment"]["training"]

    assert int(training["target_examples_processed"]) == 500000
    assert int(training["data_equivalent_passes"]) == 100
    assert int(training["evaluation_every_examples"]) == 5000


def test_output_namespace_is_separate_from_old_dimension_stress_outputs() -> None:
    assert v3s.OUTPUT_ROOT == Path("results/raw/week1_dimension_stress_v3")
    assert v3s.FIGURE_ROOT == Path("results/figures/week1_dimension_stress_v3")
    assert str(v3s.OUTPUT_ROOT) != "results/raw/week1_dimension_stress"


def test_no_test_requires_local_generated_baseline_npz() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = "/".join(["data", "generated", "baseline_train.npz"])

    assert forbidden not in source


def test_script_does_not_use_method_position_seed_formula() -> None:
    source = Path("experiments/13_v3_dimension_stress.py").read_text(encoding="utf-8")

    assert "method_index" not in source
    assert "len(methods)" not in source
    assert "sampling_seed_offset" not in source
