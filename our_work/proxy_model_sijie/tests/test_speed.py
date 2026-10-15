"""M38 (fast) and M39 (slow): predict speed of each proxy at the largest settings (spec section 4.3)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from dwc26.config import load_model_config
from dwc26.models.dmd import DMDProxy
from our_work.proxy_model_sijie.tests.synth_models import fit_model, fit_settings, make_curves, unit_X

REPO = Path(__file__).resolve().parents[1]
SPEED = load_model_config(REPO / "configs" / "models" / "lstm.yaml")["speed"]
PER_CALL_LIMIT = SPEED["limit_s"] / SPEED["mcmc_calls"]          # 0.12 s


@pytest.fixture(scope="module")
def largest(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("speed")
    X, Xv = unit_X(70, 61), unit_X(15, 62)
    T, k0 = 41, 4
    Y, Yv = make_curves(X, T, k0), make_curves(Xv, T, k0)
    models = {}
    for name, extra in (("lstm", {}), ("rnn", {"cell": "gru"})):
        m = fit_model(name, tmp, X, Y, (Xv, Yv), variant="A", seeds=[0, 1, 2, 3, 4], num_layers=2,
                      hidden_size=64, **extra)
        models[name] = m
    s = fit_settings("dmd", tmp, rank=8)                    # A2: basis real, the largest rank in the search
    s["dmd"]["basis"] = "real"
    s["gp"]["n_restarts_optimizer"] = 0
    models["dmd"] = DMDProxy().fit(X, Y, s)
    return models


def test_largest_settings_built(largest):
    assert largest["lstm"].settings["num_layers"] == 2 and largest["lstm"].settings["hidden_size"] == 64
    assert len(largest["lstm"].nets) == 5 and largest["rnn"].settings["cell"] == "gru"
    assert largest["dmd"].settings["rank"] == 8 and largest["dmd"].diagnostics["basis"] == "real"


@pytest.mark.parametrize("name", ["lstm", "rnn", "dmd"])
def test_m38_per_call(largest, name, capsys):
    m = largest[name]
    X = np.random.default_rng(0).random((SPEED["call_rows"], 4))
    for _ in range(3):
        m.predict(X)
    t0 = time.perf_counter()
    for _ in range(50):
        m.predict(X)
    per_call = (time.perf_counter() - t0) / 50
    with capsys.disabled():
        extra = f", r_eff {m.diagnostics['r_eff']}" if name == "dmd" else ""
        print(f"\nM38 {name}: {per_call * 1e3:.2f} ms per call of {SPEED['call_rows']} rows{extra}")
    assert per_call < PER_CALL_LIMIT


@pytest.mark.slow
@pytest.mark.parametrize("name", ["lstm", "rnn", "dmd"])
def test_m39_mcmc_and_batch(largest, name, capsys):
    m = largest[name]
    X = np.random.default_rng(0).random((SPEED["call_rows"], 4))
    t0 = time.perf_counter()
    for _ in range(SPEED["mcmc_calls"]):
        m.predict(X)
    t_mcmc = time.perf_counter() - t0
    n_sets = SPEED["mcmc_calls"] * SPEED["call_rows"]
    Xb = np.random.default_rng(1).random((n_sets, 4))
    t0 = time.perf_counter()
    P = m.predict(Xb)
    t_batch = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\nM39 {name}: {SPEED['mcmc_calls']} calls x {SPEED['call_rows']} rows {t_mcmc:.1f} s; "
              f"one call of {n_sets} rows {t_batch:.1f} s")
    assert P.shape == (n_sets, 41, 3)
    assert t_mcmc < SPEED["limit_s"] and t_batch < SPEED["limit_s"]
