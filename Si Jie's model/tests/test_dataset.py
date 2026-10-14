"""M28, M29: dwc26.models.dataset.load_dataset on the synthetic Task 1 database."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dwc26.data.check import ContractError
from dwc26.data.db import connect
from dwc26.models.dataset import Dataset, load_dataset
from synth_models import K0_DB, T_DB, TEST_IDS, TRAIN_IDS, VAL_IDS, synthetic_task1_db


@pytest.fixture
def sdb(tmp_path):
    info = synthetic_task1_db(tmp_path)
    con = connect(info["db"])
    yield info, con
    con.close()


def test_m28_load_dataset(sdb):
    info, con = sdb
    d = load_dataset(con)
    assert isinstance(d, Dataset)
    np.testing.assert_array_equal(d.case_fit, np.array(list(TRAIN_IDS)))
    np.testing.assert_array_equal(d.case_val, np.array(list(VAL_IDS)))
    assert d.case_fit.dtype == np.int64 and d.case_val.dtype == np.int64
    assert d.X_fit.shape == (18, 4) and d.X_val.shape == (6, 4)
    assert d.Y_fit.shape == (18, T_DB, 3) and d.Y_val.shape == (6, T_DB, 3)
    assert d.dates.shape == (T_DB,) and d.dates.dtype == np.dtype("datetime64[D]")
    assert d.t_days.shape == (T_DB,) and d.T_full == T_DB and d.t_cut is None
    assert d.y_factor.shape == (3,) and d.x_lower.shape == (4,) and d.x_upper.shape == (4,)
    assert d.fluids == ("oil", "gas", "water") and len(d.x_names) == 4
    assert d.prepare_run_id == info["run_id"]
    assert d.arrays_sha256 == info["arrays_sha256"]
    np.testing.assert_allclose(d.Y_fit * d.y_factor, info["Y_phys"][:18], rtol=1e-12)
    np.testing.assert_allclose(d.X_fit, info["X"][:18], atol=1e-12)
    # no test case or test array anywhere in the Dataset (V3)
    test_ids = set(TEST_IDS)
    assert not (set(d.case_fit.tolist()) | set(d.case_val.tolist())) & test_ids
    for f in dataclasses.fields(d):
        v = getattr(d, f.name)
        if isinstance(v, np.ndarray) and v.ndim >= 1 and f.name not in ("dates", "t_days", "y_factor",
                                                                         "x_lower", "x_upper"):
            assert v.shape[0] in (18, 6), f.name
    Xtest = info["X"][24:]
    for X in (d.X_fit, d.X_val):
        for x in Xtest:
            assert not np.isclose(X, x, atol=1e-12, rtol=0).all(axis=1).any()


def test_m28_hash_mismatch(sdb):
    _, con = sdb
    con.execute("UPDATE production_grid SET oil_rate = oil_rate * 1.01 WHERE case_id = 3 AND date_idx = 5")
    con.commit()
    with pytest.raises(ContractError, match="arrays_sha256"):
        load_dataset(con)


def test_m28_failed_prepare_run(sdb):
    info, con = sdb
    con.execute("UPDATE runs SET status = 'failed' WHERE run_id = ?", (info["run_id"],))
    con.commit()
    with pytest.raises(ContractError):
        load_dataset(con)


def test_m29_t_cut(sdb):
    _, con = sdb
    full = load_dataset(con)
    d = load_dataset(con, t_cut=7)
    assert d.Y_fit.shape == (18, 7, 3) and d.Y_val.shape == (6, 7, 3)
    np.testing.assert_array_equal(d.dates, full.dates[:7])
    np.testing.assert_array_equal(d.t_days, full.t_days[:7])
    np.testing.assert_array_equal(d.Y_fit, full.Y_fit[:, :7])
    assert d.t_cut == 7 and d.T_full == T_DB
    assert load_dataset(con, t_cut=K0_DB + 3).Y_fit.shape[1] == K0_DB + 3
    assert load_dataset(con, t_cut=T_DB).Y_fit.shape[1] == T_DB
    for bad in (4, 14):
        with pytest.raises(ContractError):
            load_dataset(con, t_cut=bad)
