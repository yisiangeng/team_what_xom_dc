"""dwc26.config: load_config and ConfigError."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from dwc26.config import ConfigError, load_config
from our_work.proxy_model_sijie.tests.synth import synthetic_config, write_config

REPO = Path(__file__).resolve().parents[1]


def test_repo_config_loads():
    cfg = load_config(REPO / "configs" / "data.yaml")
    assert cfg["rates"]["definition"] == "auto"
    assert list(cfg["file01"]["columns"].values()) == ["fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]
    assert cfg["split"] == {"train": [1, 70], "val": [71, 85], "test": [86, 100]}


def _write(tmp_path, cfg):
    return write_config(cfg, tmp_path / "c.yaml")


def test_synthetic_config_loads(tmp_path):
    cfg = synthetic_config(tmp_path)
    assert load_config(_write(tmp_path, cfg)) == cfg


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.pop("seed"),
        lambda c: c["paths"].pop("db"),
        lambda c: c["file02"]["sheets"].pop("gas_cum"),
        lambda c: c["file02"]["sheets"]["oil_rate"].pop("unit"),
        lambda c: c["checks"].pop("c6_rel_tol"),
        lambda c: c["rates"].update(definition="R3"),
        lambda c: c["x_bounds"].update(source="prior"),
        lambda c: c["file02"].update(date_round="s"),
        lambda c: c["checks"].update(c4_rel_tol=1e-2),       # outside 1e-9 .. 1e-4
        lambda c: c["checks"].update(c6_rel_tol=0.5),        # outside 1e-6 .. 0.05
        lambda c: c["checks"].update(c6_floor_frac=0.5),     # outside 0 .. 1e-2
        lambda c: c["checks"].update(c8_abs_tol=-1.0),       # outside 0 .. 1
        lambda c: c["checks"].pop("c10_rel_tol"),            # A1: required like the others
        lambda c: c["checks"].update(c10_rel_tol=2.0e-3),    # outside 0 .. 1e-3
        lambda c: c["checks"].update(c10_rel_tol=-1.0e-9),
        lambda c: c["checks"].update(c10_rel_tol="1e-5"),    # a string, not a number
        lambda c: c["file01"].update(label_pattern="(unclosed"),
        lambda c: c.update(seed="zero"),
    ],
)
def test_invalid_config(tmp_path, mutate):
    cfg = copy.deepcopy(synthetic_config(tmp_path))
    mutate(cfg)
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, cfg))


@pytest.mark.parametrize("value", [0.0, 1.0e-3])
def test_c10_rel_tol_range_ends_accepted(tmp_path, value):
    cfg = copy.deepcopy(synthetic_config(tmp_path))
    cfg["checks"]["c10_rel_tol"] = value
    assert load_config(_write(tmp_path, cfg))["checks"]["c10_rel_tol"] == value


def test_repo_config_c10_rel_tol():
    cfg = load_config(REPO / "configs" / "data.yaml")
    assert cfg["checks"]["c10_rel_tol"] == 1.0e-5


def test_not_a_mapping(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump([1, 2]), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


# --- M35: model configs (spec task2 section 3) -------------------------------------

from dwc26.config import load_model_config  # noqa: E402
from our_work.proxy_model_sijie.tests.synth_models import tiny_cfg  # noqa: E402

MODEL_CFGS = ["rnn", "lstm", "dmd", "b0", "b1"]


@pytest.mark.parametrize("model", MODEL_CFGS)
def test_m35_deliverable_configs_load(model):
    cfg = load_model_config(REPO / "configs" / "models" / f"{model}.yaml")
    assert cfg["model"] == model
    if model in ("rnn", "lstm", "dmd"):
        assert cfg["fluid_weight_sets"]["d012"] == [0.25, 0.25, 0.5]
        assert abs(sum(cfg["fluid_weight_sets"]["equal"]) - 1.0) <= 1e-12
    if model in ("rnn", "lstm"):
        assert cfg["tuning"]["n_trials"] == 30 and cfg["training"]["seeds"] == [0, 1, 2, 3, 4]
        assert list(cfg["tuning"]["space"]) == (["num_layers", "hidden_size", "lr", "batch_size", "fluid_weights"]
                                                + (["cell"] if model == "rnn" else []))
    if model == "dmd":
        assert list(cfg["tuning"]["space"]) == ["rank", "kernel", "fluid_weights", "gp_target"]
        assert cfg["tuning"]["space"]["rank"]["values"] == [2, 3, 4, 5, 6, 8]
        assert cfg["dmd"]["basis"] == "real" and cfg["dmd"]["energy_report"] == 0.99


@pytest.mark.parametrize("model", MODEL_CFGS)
def test_m35_tiny_configs_load(tmp_path, model):
    load_model_config(tiny_cfg(tmp_path, model))


@pytest.mark.parametrize(
    "model, overrides",
    [
        ("lstm", {"tuning__n_trials": 31}),
        ("lstm", {"tuning__n_trials": 29}),                           # not divisible by 3 options
        ("lstm", {"tuning__space__hidden_size": {"type": "choice", "values": [16, 128]}}),
        ("lstm", {"tuning__space__num_layers": {"type": "choice", "values": [1, 3]}}),
        ("lstm", {"fluid_weight_sets__equal": [0.3, 0.3, 0.3]}),      # sums to 0.9
        ("lstm", {"tuning__space__cell": {"type": "choice", "values": ["gru"]}}),
        ("rnn", {"tuning__space__cell": {"type": "choice", "values": ["lstm"]}}),
        ("lstm", {"tuning__space__lr": {"type": "loguniform", "low": 1e-6, "high": 1e-2}}),
        ("lstm", {"training__seeds": [0]}),
        ("lstm", {"training__tune_seeds": 3}),
        ("lstm", {"training__predict_batch": 100}),
        ("lstm", {"network__input_options": ["A", "D"]}),
        ("lstm", {"fluid_weight_sets__d012": [0.5, 0.5, 0.0]}),        # water weight 0
        ("lstm", {"seed": -1}),
        ("lstm", {"data__t_cut": 2}),
        ("dmd", {"tuning__space__energy": {"type": "choice", "values": [0.99]}}),   # A2: energy is not a dimension
        ("dmd", {"tuning__space__rank": {"type": "choice", "values": [0]}}),
        ("dmd", {"tuning__space__rank": {"type": "choice", "values": [2.5]}}),
        ("dmd", {"tuning__space__rank": {"type": "choice", "values": [3, 3]}}),
        ("dmd", {"tuning__space__rank": {"type": "choice", "values": [13]}}),
        ("dmd", {"dmd__basis": "hankel"}),
        ("dmd", {"dmd__real_kappa_sep": 0.1}),
        ("dmd", {"dmd__real_kappa_max": 0.1}),
        ("dmd", {"dmd__energy_report": 0.8}),
        ("dmd", {"tuning__n_trials": 1}),                              # grid of 2 > n_trials
        ("dmd", {"gp__noise_init": 2.0}),
        ("dmd", {"dmd__unstable": "drop"}),
        ("dmd", {"tuning__sampler": "lhs"}),
        ("b1", {"knn__n_neighbors": 5}),
        ("b0", {"report__plateau_frac": 0.4}),
    ],
)
def test_m35_invalid_model_config(tmp_path, model, overrides):
    with pytest.raises(ConfigError):
        load_model_config(tiny_cfg(tmp_path, model, **overrides))


def test_m35_missing_and_unknown_keys(tmp_path):
    import yaml as _yaml

    for mutate in (lambda c: c.pop("objective"), lambda c: c.update(extra=1), lambda c: c["gp"].pop("alpha"),
                   lambda c: c.update(model="svr")):
        p = tiny_cfg(tmp_path, "dmd")
        c = _yaml.safe_load(p.read_text(encoding="utf-8"))
        mutate(c)
        p.write_text(_yaml.safe_dump(c), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_model_config(p)


def test_m35_dmd_deliverable_grid_needs_24(tmp_path):
    import yaml as _yaml

    c = _yaml.safe_load((REPO / "configs" / "models" / "dmd.yaml").read_text(encoding="utf-8"))
    c["tuning"]["n_trials"] = 23
    p = tmp_path / "dmd.yaml"
    p.write_text(_yaml.safe_dump(c), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_model_config(p)


def test_m35_dmd_missing_a2_key(tmp_path):
    import yaml as _yaml

    p = tiny_cfg(tmp_path, "dmd")
    c = _yaml.safe_load(p.read_text(encoding="utf-8"))
    c["dmd"].pop("basis")
    p.write_text(_yaml.safe_dump(c), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_model_config(p)


# --- E29: load_eval_config (spec task3 sections 1.2 and 3.1) -------------------------------------

def _eval_yaml(tmp_path, mutate=None):
    import copy

    import yaml as _yaml

    base = _yaml.safe_load((REPO / "configs" / "evaluate.yaml").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)
    if mutate:
        mutate(cfg)
    path = tmp_path / "evaluate.yaml"
    path.write_text(_yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_e29_deliverable_eval_config_loads():
    from dwc26.config import load_eval_config

    cfg = load_eval_config(REPO / "configs" / "evaluate.yaml")
    assert cfg["candidates"]["runs"] == {m: "latest" for m in ("b0", "b1", "rnn", "lstm", "dmd")}
    assert cfg["mismatch"]["window"] == ["1999-01-01", "2008-01-01"]
    assert cfg["continuation"]["end_date"] == "2028-01-01" and cfg["backtest"]["cutoff_date"] == "2005-01-01"
    assert cfg["selection"]["secondary_only"] == ["dmd"] and cfg["freeze"]["repo_dir"] is None


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(unknown_key=1),
    lambda c: c["report"].update(extra=1),
    lambda c: c["metrics"].update(objective_fluids=["oil"]),
    lambda c: c["selection"].update(band=0.3),
    lambda c: c["plausibility"].update(p3_fc_quarters=[16, 9]),
    lambda c: c["continuation"].update(end_date="2028-02-01"),
    lambda c: c.update(strategy={"rnn": "S3", "lstm": "S3", "dmd": "S4", "b1": "S3"}),
    lambda c: c["backtest"].update(v6_horizons=[0, 12]),
    lambda c: c["plausibility"].update(p2_mode="warn"),
    lambda c: c["candidates"]["runs"].pop("b0"),
    lambda c: c["backtest"].update(models=["rnn", "lstm"]),
    lambda c: c["strategy"].update(dmd="S5"),
    lambda c: c["freeze"].update(min_override_chars=5),
    lambda c: c["freeze"].update(decisions_path="/abs/decisions.md"),
    lambda c: c["mismatch"].update(window=["2008-01-01", "1999-01-01"]),
    lambda c: c["mismatch"].update(alpha=0.6),
    lambda c: c["report"].update(dpi=400),
    lambda c: c["selection"].update(secondary_only=["b0"]),
    lambda c: c["backtest"].update(cutoff_date="2005-13-01"),
    lambda c: c.pop("cost"),
])
def test_e29_eval_config_errors(tmp_path, mutate):
    from dwc26.config import ConfigError, load_eval_config

    with pytest.raises(ConfigError):
        load_eval_config(_eval_yaml(tmp_path, mutate))
