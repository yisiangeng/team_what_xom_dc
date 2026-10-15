"""M11, M12: dwc26.models.tuning (Latin-hypercube plan, DMD grid, resolved settings)."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np
import pytest

from dwc26.config import ConfigError, load_model_config
from dwc26.models.tuning import lhs_unit, make_plan, resolve_settings

REPO = Path(__file__).resolve().parents[1]


def _cfg(model):
    return load_model_config(REPO / "configs" / "models" / f"{model}.yaml")


def test_lhs_unit():
    u = lhs_unit(10, 3, np.random.default_rng(0))
    assert u.shape == (10, 3) and (u >= 0).all() and (u < 1).all()
    for d in range(3):
        assert sorted(np.floor(u[:, d] * 10).astype(int).tolist()) == list(range(10))


@pytest.mark.parametrize("model", ["lstm", "rnn"])
def test_m11_lhs_plan(model):
    cfg = _cfg(model)
    plan = make_plan(cfg)
    assert len(plan) == 30
    assert [t["trial"] for t in plan] == list(range(30))
    assert [t["variant"] for t in plan] == ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    assert make_plan(copy.deepcopy(cfg)) == plan
    other = copy.deepcopy(cfg)
    other["seed"] = 1
    assert make_plan(other) != plan
    space = cfg["tuning"]["space"]
    for v in "ABC":
        trials = [t["settings"] for t in plan if t["variant"] == v]
        assert all(list(s) == list(space) for s in trials)          # draw order = config order
        lr = np.array([s["lr"] for s in trials])
        lo, hi = space["lr"]["low"], space["lr"]["high"]
        assert (lr >= lo).all() and (lr <= hi).all()
        bins = np.floor((np.log(lr) - np.log(lo)) / (np.log(hi) - np.log(lo)) * 10).astype(int)
        assert sorted(bins.tolist()) == list(range(10))
        for dim, spec in space.items():
            if spec["type"] == "choice":
                got = [s[dim] for s in trials]
                assert set(got) == set(spec["values"]), (v, dim)    # every value at least once
                assert all(g in spec["values"] for g in got)


@pytest.mark.parametrize("n_trials", [31, 29])
def test_m11_bad_n_trials(n_trials):
    cfg = _cfg("lstm")
    cfg["tuning"]["n_trials"] = n_trials
    with pytest.raises(ConfigError):
        make_plan(cfg)


def test_m11_strata_follow_config():
    cfg = _cfg("lstm")
    cfg["network"]["input_options"] = ["C", "A"]
    cfg["tuning"]["n_trials"] = 4
    plan = make_plan(cfg)
    assert [t["variant"] for t in plan] == ["C", "C", "A", "A"]


def test_m12_dmd_grid():
    cfg = _cfg("dmd")
    plan = make_plan(cfg)
    assert len(plan) == 24
    pts = [tuple(t["settings"].values()) for t in plan]
    assert len(set(pts)) == 24
    space = cfg["tuning"]["space"]
    assert pts == list(itertools.product(*(space[d]["values"] for d in space)))
    assert plan[0]["settings"] == {"rank": 2, "kernel": "matern25", "fluid_weights": "d012", "gp_target": "coef"}
    assert plan[1]["settings"]["gp_target"] == "proj"                # last dimension fastest
    assert plan[-1]["settings"] == {"rank": 8, "kernel": "matern25", "fluid_weights": "equal", "gp_target": "proj"}
    assert sorted({t["settings"]["rank"] for t in plan}) == [2, 3, 4, 5, 6, 8]
    for t in plan:                                                    # A2: rank, no energy
        s = resolve_settings(cfg, "main", t["settings"])
        assert "rank" in s and "energy" not in s
    assert all(t["variant"] == "main" for t in plan)
    cfg["tuning"]["n_trials"] = 23
    with pytest.raises(ConfigError):
        make_plan(cfg)


def test_benchmarks_have_no_plan():
    assert make_plan(_cfg("b0")) == [] and make_plan(_cfg("b1")) == []


def test_resolve_settings_neural():
    cfg = _cfg("rnn")
    t = make_plan(cfg)[12]
    s = resolve_settings(cfg, t["variant"], t["settings"])
    assert s["input_option"] == "B" and s["cell"] == t["settings"]["cell"]
    assert s["fluid_weights_name"] == t["settings"]["fluid_weights"]
    assert s["fluid_weights"] == cfg["fluid_weight_sets"][t["settings"]["fluid_weights"]]
    for k in ("seeds", "max_epochs", "patience", "min_delta", "grad_clip_norm", "adam", "dtype", "x_affine",
              "predict_batch", "objective_fluids", "num_layers", "hidden_size", "lr", "batch_size"):
        assert k in s, k
    lst = _cfg("lstm")
    s = resolve_settings(lst, "A", make_plan(lst)[0]["settings"])
    assert s["cell"] == "lstm"
    with pytest.raises(ConfigError):
        resolve_settings(lst, "A", {"num_layers": 1})                # missing tuned values


def test_resolve_settings_dmd_and_benchmarks():
    cfg = _cfg("dmd")
    s = resolve_settings(cfg, "main", make_plan(cfg)[0]["settings"])
    assert s["rank"] == 2 and s["fluid_weights"] == [0.25, 0.25, 0.5] and s["fluid_weights_name"] == "d012"
    assert s["seed"] == 0 and s["dmd"] == cfg["dmd"] and s["gp"] == cfg["gp"]
    assert resolve_settings(_cfg("b0"), "main", {}) == {}
    assert resolve_settings(_cfg("b1"), "main", {}) == {"knn": {"n_neighbors": 3, "weights": "distance",
                                                                "algorithm": "brute"}}
