"""M02 to M05 and M08: the Proxy contract for every registered model (PRD section 6)."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from dwc26.data.check import ContractError
from dwc26.data.db import connect
from dwc26.models.base import META_KEYS
from dwc26.models.dataset import load_dataset
from dwc26.models.registry import get_class, load_model
from our_work.proxy_model_sijie.tests.synth_models import (
    ALL_MODELS,
    NEURAL_MODELS,
    PROXY_MODELS,
    curves_data,
    fit_model,
    synthetic_task1_db,
)

T = 13
ATOL = {"dmd": 1e-12, "rnn": 1e-6, "lstm": 1e-6}


@pytest.fixture(scope="module")
def data():
    return curves_data(n_fit=12, n_val=4, T=T, k0=2)


@pytest.fixture(scope="module")
def fitted(data, tmp_path_factory):
    X, Y, Xv, Yv = data
    tmp = tmp_path_factory.mktemp("contract")
    return {name: fit_model(name, tmp, X, Y, (Xv, Yv)) for name in ALL_MODELS}


@pytest.mark.parametrize("name", ALL_MODELS)
def test_m02_contract(name, data, fitted, tmp_path):
    X, Y, Xv, Yv = data
    m = fitted[name]
    P = m.predict(Xv)
    assert P.shape == (4, T, 3) and P.dtype == np.float64 and np.isfinite(P).all() and (P >= 0).all()
    S = m.predict_std(Xv)
    if name in PROXY_MODELS:
        assert S.shape == (4, T, 3) and S.dtype == np.float64 and np.isfinite(S).all() and (S >= 0).all()
    else:
        assert S is None
    again = fit_model(name, tmp_path, X, Y, (Xv, Yv))
    assert np.array_equal(again.predict(Xv), P)
    if S is not None:
        assert np.array_equal(again.predict_std(Xv), S)


@pytest.mark.parametrize("name", NEURAL_MODELS)
def test_m03_seeds_matter(name, data, fitted, tmp_path):
    X, Y, Xv, Yv = data
    other = fit_model(name, tmp_path, X, Y, (Xv, Yv), seeds=[2, 3])
    assert not np.array_equal(other.predict(Xv), fitted[name].predict(Xv))


@pytest.mark.parametrize("name", PROXY_MODELS)
def test_m04_continuation(name, data, fitted):
    X, Y, Xv, Yv = data
    m = fitted[name]
    F = m.forecast(Xv, 80)
    assert F.shape == (4, 80, 3) and F.dtype == np.float64 and np.isfinite(F).all() and (F >= 0).all()
    R = m.rollout(Xv, T + 80)
    assert R.shape == (4, T + 80, 3)
    np.testing.assert_allclose(R[:, :T], m.predict(Xv), rtol=0, atol=ATOL[name])
    assert np.array_equal(F, R[:, T:])
    np.testing.assert_allclose(m.forecast(Xv, 10), F[:, :10], rtol=0, atol=ATOL[name])
    assert m.extrapolates_time is True


@pytest.mark.parametrize("name", ["b0", "b1"])
def test_m05_benchmarks_do_not_continue(name, data, fitted):
    X, Y, Xv, Yv = data
    m = fitted[name]
    with pytest.raises(NotImplementedError):
        m.forecast(Xv, 1)
    with pytest.raises(NotImplementedError):
        m.rollout(Xv, T + 1)
    assert np.array_equal(m.rollout(Xv, T), m.predict(Xv))
    assert m.extrapolates_time is False


ARTEFACT = {"rnn": "weights.pt", "lstm": "weights.pt", "dmd": "gps.pkl", "b0": "b0.npz", "b1": "b1.npz"}


@pytest.mark.parametrize("name", ALL_MODELS)
def test_m08_save_load(name, tmp_path):
    info = synthetic_task1_db(tmp_path)
    con = connect(info["db"])
    d = load_dataset(con)
    m = fit_model(name, tmp_path, d.X_fit, d.Y_fit, (d.X_val, d.Y_val))
    run_id = "train-20000101T000000000000Z"
    proxy_id = f"{run_id}/{m.variant}"
    m.set_context(run_id=run_id, proxy_id=proxy_id, t_cut=None, settings_from=None, x_names=list(d.x_names),
                  fluids=list(d.fluids), y_factor=d.y_factor, x_lower=d.x_lower, x_upper=d.x_upper,
                  prepare_run_id=d.prepare_run_id, arrays_sha256=d.arrays_sha256)
    models = tmp_path / "models"
    path = models / run_id / m.variant
    m.save(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    assert set(META_KEYS) <= set(meta)
    assert meta["model"] == name and meta["proxy_id"] == proxy_id and meta["k0"] == 2 and meta["T_fit"] == 13
    assert set(meta["packages"]) == {"numpy", "scipy", "torch", "scikit-learn"}
    assert ARTEFACT[name] in meta["files"]
    loaded = load_model(proxy_id, models, con=con, proxies_only=False)
    Xq = d.X_val
    assert np.array_equal(loaded.predict(Xq), m.predict(Xq))
    if name in PROXY_MODELS:
        assert np.array_equal(loaded.predict_std(Xq), m.predict_std(Xq))
        assert np.array_equal(loaded.rollout(Xq, 13 + 80), m.rollout(Xq, 13 + 80))
        assert load_model(proxy_id, models, con=con).name == name
    else:
        assert loaded.predict_std(Xq) is None
        with pytest.raises(ContractError, match="benchmark"):
            load_model(proxy_id, models, con=con)
    # a changed byte in an artefact
    f = path / ARTEFACT[name]
    raw = bytearray(f.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    f.write_bytes(bytes(raw))
    with pytest.raises(ContractError, match="sha256"):
        get_class(name).load(path)
    raw[len(raw) // 2] ^= 0xFF
    f.write_bytes(bytes(raw))
    get_class(name).load(path)
    # the DB scalers differ from meta.json (V4)
    con.execute("UPDATE scalers SET factor = factor * (1 + 1e-12) WHERE name = 'y_water'")
    con.commit()
    with pytest.raises(ContractError, match="V4"):
        load_model(proxy_id, models, con=con, proxies_only=False)
    con.close()


def test_m08_format_version(tmp_path, data):
    X, Y, Xv, Yv = data
    m = fit_model("b0", tmp_path, X, Y, (Xv, Yv))
    m.save(tmp_path / "r" / "main")
    meta_path = tmp_path / "r" / "main" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format_version"] = 2
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ContractError, match="format_version"):
        get_class("b0").load(tmp_path / "r" / "main")
    assert isinstance(sqlite3.sqlite_version, str)
