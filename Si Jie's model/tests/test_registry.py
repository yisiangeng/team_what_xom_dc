"""M01: dwc26.models.registry."""

from __future__ import annotations

import pytest

from dwc26.config import ConfigError
from dwc26.data.check import ContractError
from dwc26.models import registry
from dwc26.models.base import Proxy
from dwc26.models.registry import BENCHMARKS, MODEL_NAMES, PROXIES, get_class, load_model, parse_proxy_id
from synth_models import ALL_MODELS, curves_data, fit_model


def test_m01_names():
    assert set(PROXIES) == {"rnn", "lstm", "dmd"}
    assert set(BENCHMARKS) == {"b0", "b1"}
    assert MODEL_NAMES == ("rnn", "lstm", "dmd", "b0", "b1")


@pytest.mark.parametrize("name", ALL_MODELS)
def test_m01_classes(name, tmp_path):
    cls = get_class(name)
    assert cls.name == name
    assert cls.extrapolates_time is (name in PROXIES)
    X, Y, Xv, Yv = curves_data()
    m = fit_model(name, tmp_path, X, Y, (Xv, Yv))
    assert isinstance(m, Proxy)
    for attr in ("name", "variant", "extrapolates_time", "T_fit", "k0", "last_n_clipped"):
        assert hasattr(m, attr), attr
    for meth in ("fit", "predict", "predict_std", "rollout", "forecast", "save", "load"):
        assert callable(getattr(m, meth)), meth


def test_m01_unknown_and_wrong_flag(monkeypatch):
    with pytest.raises(ConfigError):
        get_class("svr")
    monkeypatch.setitem(registry.PROXIES, "b0x", "dwc26.models.baselines:MeanCurveBenchmark")
    with pytest.raises(ContractError):
        get_class("b0x")                        # a benchmark registered as a proxy


def test_m01_benchmark_refused(tmp_path):
    X, Y, Xv, Yv = curves_data()
    m = fit_model("b0", tmp_path, X, Y, (Xv, Yv))
    m.set_context(run_id="run-1")
    m.save(tmp_path / "models" / "run-1" / "main")
    with pytest.raises(ContractError):
        load_model("run-1/main", tmp_path / "models", proxies_only=True)
    assert load_model("run-1/main", tmp_path / "models", proxies_only=False).name == "b0"


@pytest.mark.parametrize("bad", ["run", "run/", "/A", "a/b/c", " run/A"])
def test_parse_proxy_id(bad):
    assert parse_proxy_id("train-20261013T120000000000Z/B") == ("train-20261013T120000000000Z", "B")
    with pytest.raises(ConfigError):
        parse_proxy_id(bad)
