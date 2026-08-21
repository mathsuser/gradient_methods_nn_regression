from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


def _load_v3_module():
    module_path = Path("experiments/10_v3_learning_rate_pilot.py")
    spec = importlib.util.spec_from_file_location("v3_learning_rate_pilot", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = _load_v3_module()


def _baseline_cfg() -> dict:
    return v3.load_json(v3.BASELINE_CONFIG_PATH)


def _v3_cfg() -> dict:
    return v3.load_json(v3.V3_CONFIG_PATH)


def test_v3_config_contains_exactly_seven_methods() -> None:
    cfg = _v3_cfg()

    assert list(cfg["experiment"]["methods"]) == v3.METHOD_ORDER
    assert len(cfg["experiment"]["methods"]) == 7


def test_v3_conceptual_matrix_and_sampler_mapping_are_locked() -> None:
    methods = _v3_cfg()["experiment"]["methods"]

    assert {
        name: (method["sampling_method"], int(method["batch_size"]))
        for name, method in methods.items()
    } == {
        "full_batch_gd": ("full_batch", 5000),
        "wr_b1": ("single_with_replacement", 1),
        "wr_b32": ("minibatch_with_replacement", 32),
        "wr_b256": ("minibatch_with_replacement", 256),
        "rr_b1": ("random_reshuffling", 1),
        "rr_b32": ("random_reshuffling", 32),
        "rr_b256": ("random_reshuffling", 256),
    }


def test_v3_baseline_and_pilot_horizons_are_locked() -> None:
    cfg = _v3_cfg()
    training = cfg["experiment"]["training"]
    pilot = cfg["experiment"]["pilot"]

    assert int(training["data_equivalent_passes"]) == 100
    assert int(training["target_examples_processed"]) == 500000
    assert int(training["evaluation_every_examples"]) == 5000
    assert int(pilot["data_equivalent_passes"]) == 10
    assert int(pilot["target_examples_processed"]) == 50000
    assert int(pilot["evaluation_every_examples"]) == 5000


def test_v3_pilot_grid_and_material_threshold_are_locked() -> None:
    pilot = _v3_cfg()["experiment"]["pilot"]

    assert pilot["learning_rates"] == [0.001, 0.003, 0.01, 0.03, 0.1]
    assert float(pilot["material_reduction_ratio"]) == pytest.approx(0.9)


def test_v3_explicit_pilot_seed_map_is_stable() -> None:
    seeds = _v3_cfg()["experiment"]["pilot"]["sampling_seeds"]

    assert seeds == {
        "full_batch_gd": 30000,
        "wr_b1": 31000,
        "wr_b32": 32000,
        "wr_b256": 33000,
        "rr_b1": 41000,
        "rr_b32": 42000,
        "rr_b256": 43000,
    }


def test_v3_explicit_future_baseline_seed_map_is_stable() -> None:
    cfg = _v3_cfg()

    assert {
        method: [
            v3.future_baseline_seed_for(
                cfg,
                method_name=method,
                model_seed=model_seed,
            )
            for model_seed in range(5)
        ]
        for method in v3.METHOD_ORDER
    } == {
        "full_batch_gd": [50000, 50001, 50002, 50003, 50004],
        "wr_b1": [51000, 51001, 51002, 51003, 51004],
        "wr_b32": [52000, 52001, 52002, 52003, 52004],
        "wr_b256": [53000, 53001, 53002, 53003, 53004],
        "rr_b1": [61000, 61001, 61002, 61003, 61004],
        "rr_b32": [62000, 62001, 62002, 62003, 62004],
        "rr_b256": [63000, 63001, 63002, 63003, 63004],
    }


def test_seed_lookup_is_order_independent() -> None:
    cfg = _v3_cfg()
    reordered = copy.deepcopy(cfg)
    methods = reordered["experiment"]["methods"]
    reordered["experiment"]["methods"] = {
        key: methods[key]
        for key in reversed(list(methods))
    }

    assert v3.pilot_seed_for(reordered, "rr_b1") == 41000
    assert v3.future_baseline_seed_for(
        reordered,
        method_name="wr_b256",
        model_seed=3,
    ) == 53003


def test_pilot_run_specs_do_not_derive_seeds_from_method_position() -> None:
    cfg = _v3_cfg()
    specs = v3.build_pilot_run_specs(v3_cfg=cfg)

    assert len(specs) == 35
    assert {spec["model_seed"] for spec in specs} == {0}
    for method in v3.METHOD_ORDER:
        method_specs = [spec for spec in specs if spec["method_name"] == method]
        assert {spec["sampling_seed"] for spec in method_specs} == {
            v3.pilot_seed_for(cfg, method)
        }


def test_v3_script_does_not_use_method_index_seed_formula() -> None:
    source = Path("experiments/10_v3_learning_rate_pilot.py").read_text(
        encoding="utf-8"
    )

    assert "method_index" not in source
    assert "number_of_methods" not in source
    assert "sampling_seed_offset" not in source


def test_output_paths_are_separate_from_v2_and_branch() -> None:
    paths = _v3_cfg()["experiment"]["paths"]

    assert paths["raw_dir"] == "results/raw/week1_gradient_methods_v3"
    assert paths["figures_dir"] == "results/figures/week1_gradient_methods_v3"
    assert "sampling_law_branch" not in paths["raw_dir"]
    assert "sampling_law_branch" not in paths["figures_dir"]


def _summary(method: str, lr: float, *, accepted: bool) -> dict:
    reasons = [] if accepted else ["failed"]
    return {
        "method": method,
        "learning_rate": lr,
        "accepted": accepted,
        "stable": accepted,
        "useful": accepted,
        "reasons": reasons,
    }


def test_selection_helper_chooses_largest_rate_accepted_by_all_methods() -> None:
    rates = [0.001, 0.003, 0.01]
    summaries = {
        rate: [_summary(method, rate, accepted=rate != 0.01) for method in v3.METHOD_ORDER]
        for rate in rates
    }

    selected, decisions = v3.select_common_learning_rate(
        learning_rates=rates,
        summaries_by_rate=summaries,
    )

    assert selected == pytest.approx(0.003)
    assert decisions[0.01]["accepted"] is False


def test_failing_seventh_method_rejects_candidate_rate() -> None:
    rate = 0.03
    summaries = {
        rate: [
            _summary(method, rate, accepted=(method != "rr_b256"))
            for method in v3.METHOD_ORDER
        ]
    }

    selected, decisions = v3.select_common_learning_rate(
        learning_rates=[rate],
        summaries_by_rate=summaries,
    )

    assert selected is None
    assert decisions[rate]["accepted"] is False
    assert "rr_b256" in decisions[rate]["rejection_reasons"]


def test_grid_extension_logic_uses_factor_three_below_smallest_rate() -> None:
    assert v3.extension_learning_rate([0.001, 0.003], factor=3) == pytest.approx(
        0.001 / 3
    )


def test_preflight_plan_contains_seven_runs_at_one_dep() -> None:
    cfg = _v3_cfg()
    specs = v3.build_preflight_run_specs(cfg)
    plan = v3.preflight_plan(baseline_cfg=_baseline_cfg(), v3_cfg=cfg)

    assert len(specs) == 7
    assert {spec["learning_rate"] for spec in specs} == {0.03}
    assert {spec["target_examples_processed"] for spec in specs} == {5000}
    assert plan["preflight_budget"]["data_equivalent_passes"] == pytest.approx(1.0)
    assert plan["preflight_budget"]["checkpoint_examples"] == [0, 5000]


def test_validate_v3_config_accepts_current_files() -> None:
    v3.validate_v3_config(baseline_cfg=_baseline_cfg(), v3_cfg=_v3_cfg())
